from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(model_cfg: dict[str, Any]):
    source = model_cfg.get("model_path", model_cfg["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=model_cfg["tokenizer_revision"], use_fast=True,
        local_files_only=bool(model_cfg.get("model_path")),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_reference(model_cfg: dict[str, Any], training: bool = False):
    source = model_cfg.get("model_path", model_cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=model_cfg["revision"],
        torch_dtype=torch_dtype(model_cfg["dtype"]),
        low_cpu_mem_usage=True,
        local_files_only=bool(model_cfg.get("model_path")),
    )
    model.to("cuda")
    model.train(training)
    return model


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise TypeError("Expected a Hugging Face Llama-like model with model.layers")
    return layers


class PrivateResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=dtype, device=device)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SharedFFNWithAdapter(nn.Module):
    def __init__(self, core: nn.Module, hidden_size: int, rank: int):
        super().__init__()
        self.core = core
        parameter = next(core.parameters())
        self.adapter = PrivateResidualAdapter(hidden_size, rank, parameter.dtype, parameter.device)
        self.capture_update = False
        self.last_update: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.core(x) + self.adapter(x)
        if self.capture_update:
            self.last_update = update
        return update


@dataclass
class SharedBuild:
    model: nn.Module
    group_of_layer: list[int]
    representatives: list[int]


def build_shared_student(reference: nn.Module, group_manifest: dict[str, Any], adapter_rank: int) -> SharedBuild:
    layers = decoder_layers(reference)
    groups = [list(map(int, group)) for group in group_manifest["groups"]]
    representatives = list(map(int, group_manifest["representatives"]))
    if len(groups) != len(representatives):
        raise ValueError("Each group needs exactly one representative")
    cores = [layers[rep].mlp for rep in representatives]
    group_of_layer = [-1] * len(layers)
    for group_id, group in enumerate(groups):
        for layer_id in group:
            if group_of_layer[layer_id] != -1:
                raise ValueError(f"Layer {layer_id} appears in multiple groups")
            group_of_layer[layer_id] = group_id
    if any(group < 0 for group in group_of_layer):
        raise ValueError("Group manifest does not cover all layers")
    hidden_size = int(reference.config.hidden_size)
    for layer_id, layer in enumerate(layers):
        layer.mlp = SharedFFNWithAdapter(cores[group_of_layer[layer_id]], hidden_size, adapter_rank)
    return SharedBuild(reference, group_of_layer, representatives)


def freeze_for_recovery(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in decoder_layers(model):
        if not isinstance(layer.mlp, SharedFFNWithAdapter):
            raise TypeError("Student must be shared before recovery freezing")
        for parameter in layer.mlp.core.parameters():
            parameter.requires_grad_(True)
        for parameter in layer.mlp.adapter.parameters():
            parameter.requires_grad_(True)


def unique_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    seen: set[int] = set()
    result = []
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def save_compact_student(path: str | Path, build: SharedBuild, extra: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = decoder_layers(build.model)
    state: dict[str, torch.Tensor] = {}
    for group_id, rep in enumerate(build.representatives):
        core = layers[rep].mlp.core
        for name, tensor in core.state_dict().items():
            state[f"shared_ffn.{group_id}.{name}"] = tensor.detach().cpu()
    for layer_id, layer in enumerate(layers):
        for name, tensor in layer.mlp.adapter.state_dict().items():
            state[f"adapter.{layer_id}.{name}"] = tensor.detach().cpu()
    payload = {"state_dict": state, "group_of_layer": build.group_of_layer,
               "representatives": build.representatives, "extra": extra}
    torch.save(payload, path)
    return {"bytes": path.stat().st_size, "tensor_count": len(state)}


def load_compact_student(path: str | Path, build: SharedBuild) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    layers = decoder_layers(build.model)
    for group_id, rep in enumerate(build.representatives):
        core = layers[rep].mlp.core
        prefix = f"shared_ffn.{group_id}."
        core.load_state_dict({key[len(prefix):]: value for key, value in payload["state_dict"].items() if key.startswith(prefix)})
    for layer_id, layer in enumerate(layers):
        prefix = f"adapter.{layer_id}."
        layer.mlp.adapter.load_state_dict({key[len(prefix):]: value for key, value in payload["state_dict"].items() if key.startswith(prefix)})
    return payload.get("extra", {})
