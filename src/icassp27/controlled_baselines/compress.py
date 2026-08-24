from __future__ import annotations

import gc
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import raw_checkpoint_dir, sha256_file
from .distributed import barrier, initialize, is_main, seed_everything
from .modeling import (FactorizedLinear, SharedBasisLinear, decoder_layers, get_submodule,
                       save_compressed_checkpoint, set_submodule, unique_parameter_count)


def _target_specs(model: nn.Module, cfg: dict[str, Any], method: str) -> list[dict[str, Any]]:
    target_suffixes = set(cfg["compression"]["target_modules"])
    modules: dict[tuple[int, str], tuple[str, nn.Linear]] = {}
    for layer_index, layer in enumerate(decoder_layers(model)):
        for local_name, module in layer.named_modules():
            suffix = local_name.rsplit(".", 1)[-1]
            if suffix in target_suffixes and isinstance(module, nn.Linear):
                full_name = f"model.layers.{layer_index}.{local_name}"
                modules[(layer_index, suffix)] = (full_name, module)
    expected = len(decoder_layers(model)) * len(target_suffixes)
    if len(modules) != expected:
        raise RuntimeError(f"Llama projection discovery found {len(modules)}, expected {expected}")

    specs = []
    if method == "svd_llm":
        for (layer_index, suffix), (name, module) in sorted(modules.items()):
            specs.append({"kind": "svd", "layer_indices": [layer_index], "suffix": suffix,
                          "module_names": [name], "in_features": module.in_features,
                          "out_features": module.out_features, "bias": module.bias is not None})
    elif method == "basis_sharing":
        group_size = int(cfg["compression"]["basis_group_size"])
        shared = set(cfg["compression"]["basis_shared_modules"])
        private = set(cfg["compression"]["basis_private_modules"])
        if shared | private != target_suffixes or shared & private:
            raise ValueError("Basis Sharing shared/private target partition is invalid")
        layer_count = len(decoder_layers(model))
        for suffix in sorted(target_suffixes):
            size = group_size if suffix in shared else 1
            for start in range(0, layer_count, size):
                layer_indices = list(range(start, min(start + size, layer_count)))
                names = [modules[(index, suffix)][0] for index in layer_indices]
                shapes = {(modules[(index, suffix)][1].in_features,
                           modules[(index, suffix)][1].out_features) for index in layer_indices}
                if len(shapes) != 1:
                    raise RuntimeError(f"Basis Sharing group has heterogeneous shapes: {names}")
                in_features, out_features = shapes.pop()
                specs.append({"kind": "basis", "layer_indices": layer_indices, "suffix": suffix,
                              "module_names": names, "in_features": in_features,
                              "out_features": out_features, "bias": False})
    else:
        raise ValueError(method)
    return specs


def _assign_ranks(specs: list[dict[str, Any]], dense_parameters: int, target_reduction: float) -> None:
    original_target = sum(len(spec["module_names"]) * spec["in_features"] * spec["out_features"]
                          for spec in specs)
    untouched = dense_parameters - original_target
    target_total = round(dense_parameters * (1.0 - target_reduction))
    target_compressed = target_total - untouched
    if target_compressed <= 0:
        raise ValueError("target reduction cannot be reached by configured projections")

    def rank_for(spec, retention):
        group = len(spec["module_names"])
        n, m = spec["in_features"], spec["out_features"]
        rank = math.floor(group * n * m * retention / (n + group * m))
        return max(1, min(rank, n, group * m))

    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        count = sum(rank_for(spec, middle) *
                    (spec["in_features"] + len(spec["module_names"]) * spec["out_features"])
                    for spec in specs)
        if count <= target_compressed:
            low = middle
        else:
            high = middle
    for spec in specs:
        spec["rank"] = rank_for(spec, low)
    # Spend any remaining budget greedily on the least expensive next rank.
    used = sum(spec["rank"] * (spec["in_features"] + len(spec["module_names"]) * spec["out_features"])
               for spec in specs)
    while True:
        choices = [(spec["in_features"] + len(spec["module_names"]) * spec["out_features"], index)
                   for index, spec in enumerate(specs)
                   if spec["rank"] < min(spec["in_features"],
                                          len(spec["module_names"]) * spec["out_features"])]
        if not choices:
            break
        cost, index = min(choices)
        if used + cost > target_compressed:
            break
        specs[index]["rank"] += 1
        used += cost


def _source_name(spec: dict[str, Any], module_name: str) -> str:
    layer_prefix = module_name.rsplit(".", 1)[0]
    suffix = spec["suffix"]
    if suffix in {"k_proj", "v_proj"}:
        return f"{layer_prefix}.q_proj"
    if suffix == "up_proj":
        return f"{layer_prefix}.gate_proj"
    return module_name


def _load_calibration(path: str | Path, sequences: int, length: int) -> list[torch.Tensor]:
    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, list) or not values or not all(torch.is_tensor(value) for value in values):
        raise TypeError("calibration_blocks must be a non-empty list of token tensors")
    result = []
    for value in values[:sequences]:
        flat = value.flatten().long()
        if flat.numel() < length:
            raise ValueError(f"calibration block has {flat.numel()} tokens, expected at least {length}")
        result.append(flat[:length])
    if len(result) != sequences:
        raise ValueError(f"calibration has {len(result)} sequences, expected {sequences}")
    return result


def _collect_covariances(model: nn.Module, specs: list[dict[str, Any]], blocks: list[torch.Tensor],
                         device: torch.device, token_limit: int) -> dict[str, torch.Tensor]:
    required = sorted({_source_name(spec, name) for spec in specs for name in spec["module_names"]})
    covariances: dict[str, torch.Tensor] = {}
    handles = []

    for name in required:
        module = get_submodule(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"calibration source is not linear: {name}")
        covariances[name] = torch.zeros((module.in_features, module.in_features),
                                        dtype=torch.float32, device=device)

        def hook(_module, inputs, _output, key=name):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            if values.shape[0] > token_limit:
                positions = torch.linspace(0, values.shape[0] - 1, token_limit,
                                           device=values.device).round().long()
                values = values.index_select(0, positions)
            covariances[key].addmm_(values.T, values)

        handles.append(module.register_forward_hook(hook))
    model.eval()
    model.config.use_cache = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for block in blocks:
            ids = block.unsqueeze(0).to(device)
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for handle in handles:
        handle.remove()
    return covariances


def _stable_cholesky(covariance: torch.Tensor, epsilon: float, *, transpose: bool) -> torch.Tensor:
    covariance = covariance.float()
    scale = covariance.diagonal().mean().clamp_min(1.0)
    identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    jitter = float(epsilon) * scale
    for _ in range(8):
        factor, info = torch.linalg.cholesky_ex(covariance + jitter * identity)
        if int(info.max()) == 0:
            return factor.T if transpose else factor
        jitter *= 10
    raise RuntimeError(f"calibration covariance is not positive definite after jitter={float(jitter)}")


@torch.no_grad()
def _factor_svd(weight: torch.Tensor, covariance: torch.Tensor, rank: int, epsilon: float):
    scaling = _stable_cholesky(covariance, epsilon, transpose=False)
    inverse = torch.linalg.inv(scaling)
    transformed = weight.float() @ scaling
    u, singular, vh = torch.linalg.svd(transformed, full_matrices=False)
    root = singular[:rank].sqrt()
    left = u[:, :rank] * root.unsqueeze(0)
    right = (root.unsqueeze(1) * vh[:rank]) @ inverse
    return left.to(weight.dtype).cpu(), right.to(weight.dtype).cpu()


@torch.no_grad()
def _factor_basis(weights: list[torch.Tensor], covariance: torch.Tensor, rank: int, epsilon: float):
    # This follows Basis Sharing's activation-whitened joint SVD: concatenate
    # W_i^T, factor S @ [W_1^T ... W_g^T], then share the left basis.
    scaling = _stable_cholesky(covariance, epsilon, transpose=True)
    inverse = torch.linalg.inv(scaling)
    joined = torch.cat([weight.float().T for weight in weights], dim=1)
    u, singular, vh = torch.linalg.svd(scaling @ joined, full_matrices=False)
    basis = inverse @ (u[:, :rank] * singular[:rank].unsqueeze(0))
    coefficients = vh[:rank]
    width = weights[0].shape[0]
    pieces = [coefficients[:, index * width:(index + 1) * width].T
              for index in range(len(weights))]
    dtype = weights[0].dtype
    return basis.T.to(dtype).cpu(), [piece.to(dtype).cpu() for piece in pieces]


@torch.no_grad()
def _apply_factors(model: nn.Module, method: str, specs: list[dict[str, Any]], factors: list[dict[str, Any]]) -> None:
    by_index = {int(value["spec_index"]): value for value in factors}
    if len(by_index) != len(specs):
        raise RuntimeError(f"factor shards contain {len(by_index)} of {len(specs)} specifications")
    for index, spec in enumerate(specs):
        value = by_index[index]
        first = get_submodule(model, spec["module_names"][0])
        if method == "svd_llm":
            replacement = FactorizedLinear(spec["in_features"], spec["out_features"], spec["rank"],
                                           bias=spec["bias"], device="cpu", dtype=first.weight.dtype)
            replacement.u_proj.weight.copy_(value["u"])
            replacement.v_proj.weight.copy_(value["v"])
            if spec["bias"]:
                replacement.u_proj.bias.copy_(first.bias)
            set_submodule(model, spec["module_names"][0], replacement)
        else:
            basis = nn.Linear(spec["in_features"], spec["rank"], bias=False,
                              device="cpu", dtype=first.weight.dtype)
            basis.weight.copy_(value["basis"])
            for name, coefficient in zip(spec["module_names"], value["coefficients"]):
                old = get_submodule(model, name)
                replacement = SharedBasisLinear(basis, spec["out_features"], bias=spec["bias"],
                                                device="cpu", dtype=old.weight.dtype)
                replacement.coefficient.weight.copy_(coefficient)
                set_submodule(model, name, replacement)


def compress(cfg: dict[str, Any], method: str, backbone: str, target_reduction: float) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world, device = initialize()
    if world != 1:
        raise RuntimeError(f"compression must run with exactly 1 H200 worker, got world_size={world}")
    seed_everything(int(cfg["compression"]["calibration_seed"]))
    model_cfg = cfg["backbones"][backbone]
    output = raw_checkpoint_dir(cfg, method, backbone, target_reduction)
    shard_root = output.parent / f".{output.name}.factor_shards"
    if is_main():
        shard_root.mkdir(parents=True, exist_ok=True)
    barrier()

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_path"], torch_dtype=torch.bfloat16,
                                                local_files_only=True, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dense_parameters = unique_parameter_count(model)
    specs = _target_specs(model, cfg, method)
    _assign_ranks(specs, dense_parameters, float(target_reduction))
    owned = [(index, spec) for index, spec in enumerate(specs) if index % world == rank]
    blocks = _load_calibration(model_cfg["calibration_blocks"],
                               int(cfg["compression"]["calibration_sequences"]),
                               int(cfg["compression"]["calibration_length"]))
    model.to(device)
    covariances = _collect_covariances(model, [spec for _, spec in owned], blocks, device,
                                       int(cfg["compression"]["covariance_tokens_per_sequence"]))
    epsilon = float(cfg["compression"]["svd_epsilon"])
    factors = []
    for spec_index, spec in owned:
        sources = [_source_name(spec, name) for name in spec["module_names"]]
        covariance = sum((covariances[name] for name in sources),
                         torch.zeros_like(covariances[sources[0]]))
        weights = [get_submodule(model, name).weight for name in spec["module_names"]]
        if method == "svd_llm":
            left, right = _factor_svd(weights[0], covariance, int(spec["rank"]), epsilon)
            factors.append({"spec_index": spec_index, "u": left, "v": right})
        else:
            basis, coefficients = _factor_basis(weights, covariance, int(spec["rank"]), epsilon)
            factors.append({"spec_index": spec_index, "basis": basis, "coefficients": coefficients})
    torch.save(factors, shard_root / f"rank_{rank}.pt")
    del covariances, factors
    model.cpu()
    torch.cuda.empty_cache()
    barrier()

    if is_main():
        merged = []
        for worker in range(world):
            merged.extend(torch.load(shard_root / f"rank_{worker}.pt", map_location="cpu", weights_only=True))
        _apply_factors(model, method, specs, merged)
        compressed_parameters = unique_parameter_count(model)
        actual_reduction = 1.0 - compressed_parameters / dense_parameters
        structure_key = "projections" if method == "svd_llm" else "groups"
        structure_entries = []
        for spec in specs:
            entry = {key: value for key, value in spec.items() if key not in {"kind"}}
            if method == "svd_llm":
                entry["name"] = entry.pop("module_names")[0]
            structure_entries.append(entry)
        structure = {"schema_version": 1, "method": method, "backbone": backbone,
                     "target_reduction": float(target_reduction), structure_key: structure_entries}
        compression_report = {
            "method": method, "backbone": backbone, "target_reduction": float(target_reduction),
            "actual_reduction": actual_reduction, "dense_parameters": dense_parameters,
            "compressed_parameters": compressed_parameters, "standalone_bytes": 0,
            "already_recovered": False, "base_model": model_cfg["model_id"],
            "base_revision": model_cfg["revision"], "tokenizer_revision": model_cfg["tokenizer_revision"],
            "dtype": "bfloat16", "calibration_file": model_cfg["calibration_blocks"],
            "calibration_sha256": sha256_file(model_cfg["calibration_blocks"]),
            "calibration_sequences": int(cfg["compression"]["calibration_sequences"]),
            "calibration_length": int(cfg["compression"]["calibration_length"]),
            "covariance_tokens_per_sequence": int(cfg["compression"]["covariance_tokens_per_sequence"]),
            "decomposition": cfg["compression"]["decomposition"],
            "upstream_repository": cfg["methods"][method]["repository"],
            "upstream_commit": cfg["methods"][method]["commit"],
            "llama3_port_note": cfg["methods"][method]["port_note"],
            "elapsed_seconds": time.time() - started,
        }
        parameter_report = {
            "dense_parameters": dense_parameters, "compressed_parameters": compressed_parameters,
            "unique_parameters": compressed_parameters, "actual_reduction": actual_reduction,
            "rank_histogram": dict(sorted(Counter(str(spec["rank"]) for spec in specs).items())),
            "projection_groups": len(specs), "shared_parameter_aliases_preserved": method == "basis_sharing",
        }
        save_compressed_checkpoint(model, tokenizer, structure, output, {
            "compression_report.json": compression_report,
            "parameter_report.json": parameter_report,
        })
        standalone = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        compression_report["standalone_bytes"] = standalone
        (output / "compression_report.json").write_text(json.dumps(compression_report, indent=2, sort_keys=True) + "\n")
        compression_report["standalone_bytes"] = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        (output / "compression_report.json").write_text(json.dumps(compression_report, indent=2, sort_keys=True) + "\n")
        (output / ".complete").write_text("PASS\n")
        for worker in range(world):
            (shard_root / f"rank_{worker}.pt").unlink()
        shard_root.rmdir()
    barrier()
    gc.collect()
    return output
