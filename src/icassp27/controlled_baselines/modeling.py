from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


class FactorizedLinear(nn.Module):
    """A complete projection W ~= U @ V, used by the Llama-3 SVD-LLM port."""

    def __init__(self, in_features: int, out_features: int, rank: int, *, bias: bool = False,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.v_proj = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.u_proj = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.u_proj(self.v_proj(inputs))


class SharedBasisLinear(nn.Module):
    """One layer coefficient composed with a basis shared across adjacent layers."""

    def __init__(self, basis: nn.Linear, out_features: int, *, bias: bool = False,
                 device=None, dtype=None):
        super().__init__()
        if basis.bias is not None:
            raise ValueError("shared basis must not have bias")
        self.basis = basis
        self.in_features = int(basis.in_features)
        self.rank = int(basis.out_features)
        self.out_features = int(out_features)
        self.coefficient = nn.Linear(self.rank, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.coefficient(self.basis(inputs))


class LoRAResidualProjection(nn.Module):
    """Standard LoRA residual outside the complete compressed projection."""

    def __init__(self, base_projection: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not hasattr(base_projection, "in_features") or not hasattr(base_projection, "out_features"):
            raise TypeError(f"projection lacks in/out features: {type(base_projection).__name__}")
        self.base_projection = base_projection
        self.in_features = int(base_projection.in_features)
        self.out_features = int(base_projection.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        reference = next(base_projection.parameters())
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False,
                                device=reference.device, dtype=reference.dtype)
        self.lora_b = nn.Linear(self.rank, self.out_features, bias=False,
                                device=reference.device, dtype=reference.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base_projection.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling
        return self.base_projection(inputs) + residual


def get_submodule(root: nn.Module, name: str) -> nn.Module:
    return root.get_submodule(name)


def set_submodule(root: nn.Module, name: str, value: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, value)


def decoder_layers(model: nn.Module):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a Hugging Face LlamaForCausalLM-compatible model")
    return layers


def projection_names(model: nn.Module, suffixes: Iterable[str]) -> list[str]:
    allowed = set(suffixes)
    result = []
    for layer_index, layer in enumerate(decoder_layers(model)):
        for local_name, module in layer.named_modules():
            suffix = local_name.rsplit(".", 1)[-1]
            if suffix in allowed and isinstance(module, nn.Linear):
                result.append(f"model.layers.{layer_index}.{local_name}")
    return result


def freeze_and_inject_lora(model: nn.Module, *, rank: int, alpha: float, dropout: float,
                           targets: Iterable[str]) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_list = list(targets)
    names = []
    for layer_index, layer in enumerate(decoder_layers(model)):
        for suffix in target_list:
            local_name = f"self_attn.{suffix}"
            module = layer.get_submodule(local_name)
            if not isinstance(module, (nn.Linear, FactorizedLinear, SharedBasisLinear)):
                raise TypeError(f"logical projection {layer_index}.{local_name} has unsupported type "
                                f"{type(module).__name__}")
            full_name = f"model.layers.{layer_index}.{local_name}"
            set_submodule(model, full_name, LoRAResidualProjection(module, rank, alpha, dropout))
            names.append(full_name)
    expected = len(decoder_layers(model)) * len(target_list)
    if len(names) != expected:
        raise RuntimeError(f"expected {expected} LoRA targets, injected {len(names)}: {names[:8]}")
    return names


def trainable_parameter_report(model: nn.Module) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [name for name, _ in trainable if ".lora_" not in name]
    if invalid:
        raise RuntimeError(f"non-LoRA parameters are trainable: {invalid[:20]}")
    return {
        "trainable_parameters": int(sum(parameter.numel() for _, parameter in trainable)),
        "adapter_parameters": int(sum(parameter.numel() for _, parameter in trainable)),
        "trainable_names": [name for name, _ in trainable],
        "only_lora_trainable": True,
    }


def unique_parameter_count(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        pointer = parameter.untyped_storage().data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += parameter.numel()
    return int(total)


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if ".lora_a." in name or ".lora_b." in name}


def load_adapter_state(model: nn.Module, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_adapter = [name for name in missing if ".lora_a." in name or ".lora_b." in name]
    if missing_adapter or unexpected:
        raise RuntimeError(f"adapter mismatch missing={missing_adapter[:8]} unexpected={unexpected[:8]}")


def _apply_structure(model: nn.Module, structure: dict[str, Any]) -> None:
    if structure["method"] == "svd_llm":
        for entry in structure["projections"]:
            old = get_submodule(model, entry["name"])
            replacement = FactorizedLinear(entry["in_features"], entry["out_features"], entry["rank"],
                                           bias=bool(entry.get("bias", False)),
                                           device=old.weight.device, dtype=old.weight.dtype)
            set_submodule(model, entry["name"], replacement)
    elif structure["method"] == "basis_sharing":
        for group in structure["groups"]:
            first = get_submodule(model, group["module_names"][0])
            basis = nn.Linear(group["in_features"], group["rank"], bias=False,
                              device=first.weight.device, dtype=first.weight.dtype)
            for name in group["module_names"]:
                old = get_submodule(model, name)
                replacement = SharedBasisLinear(basis, group["out_features"],
                                                bias=bool(group.get("bias", False)),
                                                device=old.weight.device, dtype=old.weight.dtype)
                set_submodule(model, name, replacement)
    else:
        raise ValueError(f"unknown compression method {structure['method']!r}")


def load_compressed_checkpoint(checkpoint: str | Path, *, dtype: torch.dtype | None = None):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    root = Path(checkpoint).resolve()
    required = ["config.json", "compressed_state.pt", "compression_structure.json",
                "compression_report.json", "parameter_report.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete raw checkpoint {root}: missing {missing}")
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)
    structure = json.loads((root / "compression_structure.json").read_text())
    _apply_structure(model, structure)
    try:
        state = torch.load(root / "compressed_state.pt", map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state = torch.load(root / "compressed_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    if dtype is not None:
        model.to(dtype=dtype)
    # This is a Llama tokenizer. Transformers 4.57.x can mis-detect locally
    # re-saved >100k-vocabulary tokenizers as Mistral and emit a false-positive
    # regex warning. Explicit False preserves the upstream Llama tokenization.
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, fix_mistral_regex=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, structure


def save_compressed_checkpoint(model: nn.Module, tokenizer, structure: dict[str, Any], output: str | Path,
                               reports: dict[str, Any]) -> None:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(root)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(root)
    tokenizer.save_pretrained(root)
    temporary = root / f"compressed_state.pt.tmp.{os.getpid()}"
    torch.save(model.state_dict(), temporary)
    temporary.replace(root / "compressed_state.pt")
    (root / "compression_structure.json").write_text(json.dumps(structure, indent=2, sort_keys=True) + "\n")
    for name, value in reports.items():
        (root / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (root / "model_loader.py").write_text(
        "from icassp27.controlled_baselines.modeling import load_compressed_checkpoint\n"
        "def load(path, dtype=None):\n"
        "    return load_compressed_checkpoint(path, dtype=dtype)\n"
    )
