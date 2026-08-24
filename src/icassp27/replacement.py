from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import model_config, run_dir
from .modeling import decoder_layers, load_reference
from .utils import atomic_json, base_manifest, checksum_equal, module_checksum


def _blocks(cfg: dict[str, Any], backbone: str, valid_tokens: int) -> list[torch.Tensor]:
    path = run_dir(cfg, backbone, "data", "calibration_blocks.pt")
    blocks = torch.load(path, map_location="cpu", weights_only=False)
    selected, count = [], 0
    for block in blocks:
        if count >= valid_tokens:
            break
        take = min(len(block) - 1, valid_tokens - count)
        selected.append(block[: take + 1])
        count += take
    if count != valid_tokens:
        raise ValueError(f"Requested {valid_tokens} tokens but only selected {count}")
    return selected


def _weight_distance(layers) -> np.ndarray:
    count = len(layers)
    norms = np.zeros(count, dtype=np.float64)
    for i, layer in enumerate(layers):
        norms[i] = sum(float(p.detach().double().square().sum().item()) for p in layer.mlp.parameters())
    distances = np.zeros((count, count), dtype=np.float64)
    with torch.no_grad():
        for i in range(count):
            params_i = dict(layers[i].mlp.named_parameters())
            for j in range(i + 1, count):
                params_j = dict(layers[j].mlp.named_parameters())
                squared = sum(float((params_i[k].detach().double() - params_j[k].detach().double()).square().sum().item()) for k in params_i)
                value = math.sqrt(squared / max(math.sqrt(norms[i] * norms[j]), 1e-30))
                distances[i, j] = distances[j, i] = value
    return distances


def cache_baseline(cfg: dict[str, Any], backbone: str, valid_tokens: int) -> None:
    mcfg = model_config(cfg, backbone)
    output = run_dir(cfg, backbone, "replacement", f"tokens_{valid_tokens}")
    logits_dir = output / "baseline_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    model = load_reference(mcfg)
    layers = decoder_layers(model)
    input_sums = [torch.zeros(model.config.hidden_size, device="cuda", dtype=torch.float64) for _ in layers]
    input_counts = [0 for _ in layers]
    hooks = []
    for layer_id, layer in enumerate(layers):
        def hook(_module, args, _output, layer_id=layer_id):
            x = args[0].detach().double()
            input_sums[layer_id].add_(x.sum(dim=(0, 1)))
            input_counts[layer_id] += x.shape[0] * x.shape[1]
        hooks.append(layer.mlp.register_forward_hook(hook))
    blocks = _blocks(cfg, backbone, valid_tokens)
    baseline_nll_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch_id, block in enumerate(blocks):
            input_ids = block[:-1].unsqueeze(0).cuda()
            labels = block[1:].cuda()
            logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
            baseline_nll_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            token_count += labels.numel()
            np.save(logits_dir / f"batch_{batch_id:05d}.npy", logits.half().cpu().numpy())
    for hook in hooks:
        hook.remove()
    means = torch.stack([value / count for value, count in zip(input_sums, input_counts)]).float()
    means = F.normalize(means, dim=1)
    input_cosine = (means @ means.T).cpu().numpy()
    np.save(output / "ffn_input_cosine.npy", input_cosine)
    np.save(output / "normalized_weight_distance.npy", _weight_distance(layers))
    manifest = base_manifest(cfg, "replacement_baseline", backbone)
    manifest.update({
        "model_id": mcfg["model_id"], "model_revision": mcfg["revision"],
        "tokenizer_revision": mcfg["tokenizer_revision"], "valid_tokens": token_count,
        "block_length": cfg["data"]["calibration"]["block_length"],
        "num_sequences": len(blocks), "baseline_nll": baseline_nll_sum / token_count,
        "logit_storage_dtype": "float16", "kl_accumulation_dtype": "float32",
        "axis_convention": "directed_matrix[source_layer, target_layer]",
    })
    atomic_json(output / "baseline_manifest.json", manifest)


def _cost_from_logits(base_np: np.ndarray, alt: torch.Tensor, labels: torch.Tensor, chunk: int = 64):
    delta_nll_sum = 0.0
    kl_sum = 0.0
    base = torch.from_numpy(base_np).to(device=alt.device, dtype=torch.float32)
    alt = alt.float()
    for start in range(0, alt.shape[0], chunk):
        stop = min(start + chunk, alt.shape[0])
        b = base[start:stop]
        a = alt[start:stop]
        y = labels[start:stop]
        base_logp = F.log_softmax(b, dim=-1)
        alt_logp = F.log_softmax(a, dim=-1)
        delta_nll_sum += float((F.nll_loss(alt_logp, y, reduction="sum") - F.nll_loss(base_logp, y, reduction="sum")).item())
        kl_sum += float(torch.sum(base_logp.exp() * (base_logp - alt_logp)).item())
    return delta_nll_sum, kl_sum


def replacement_row(cfg: dict[str, Any], backbone: str, valid_tokens: int, target: int) -> None:
    mcfg = model_config(cfg, backbone)
    output = run_dir(cfg, backbone, "replacement", f"tokens_{valid_tokens}")
    row_dir = output / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    model = load_reference(mcfg)
    layers = decoder_layers(model)
    if not 0 <= target < len(layers):
        raise IndexError(target)
    blocks = _blocks(cfg, backbone, valid_tokens)
    original = layers[target].mlp
    before = module_checksum(original)
    delta = np.full(len(layers), np.nan, dtype=np.float64)
    kl = np.full(len(layers), np.nan, dtype=np.float64)
    restore_checks = []
    for source in range(len(layers)):
        if source == target:
            delta[source] = 0.0
            kl[source] = 0.0
            continue
        layers[target].mlp = layers[source].mlp
        delta_sum = kl_sum = 0.0
        count = 0
        with torch.inference_mode():
            for batch_id, block in enumerate(blocks):
                input_ids = block[:-1].unsqueeze(0).cuda()
                labels = block[1:].cuda()
                logits = model(input_ids=input_ids, use_cache=False).logits[0]
                base = np.load(output / "baseline_logits" / f"batch_{batch_id:05d}.npy", mmap_mode="r")
                d, k = _cost_from_logits(base, logits, labels)
                delta_sum += d
                kl_sum += k
                count += labels.numel()
        layers[target].mlp = original
        after = module_checksum(layers[target].mlp)
        restored = checksum_equal(before, after)
        restore_checks.append({"source": source, "target": target, "restored": restored, "checksum": after})
        if not restored:
            raise RuntimeError(f"Target FFN {target} failed restoration after source {source}")
        delta[source] = delta_sum / count
        kl[source] = kl_sum / count
    cost = delta + float(cfg["replacement"]["lambda_kl"]) * kl
    np.savez(row_dir / f"target_{target:03d}.npz", delta_nll=delta, kl=kl, cost=cost)
    manifest = base_manifest(cfg, "replacement_row", backbone)
    manifest.update({"target": target, "valid_tokens": valid_tokens,
                     "source_target_axis": "arrays indexed by source; fixed target",
                     "before_checksum": before, "restoration_checks": restore_checks})
    atomic_json(row_dir / f"target_{target:03d}.manifest.json", manifest)


def consolidate_replacement(cfg: dict[str, Any], backbone: str, valid_tokens: int) -> None:
    mcfg = model_config(cfg, backbone)
    layers = int(mcfg["layers"])
    output = run_dir(cfg, backbone, "replacement", f"tokens_{valid_tokens}")
    delta = np.full((layers, layers), np.nan, dtype=np.float64)
    kl = np.full_like(delta, np.nan)
    cost = np.full_like(delta, np.nan)
    for target in range(layers):
        row_path = output / "rows" / f"target_{target:03d}.npz"
        if not row_path.exists():
            raise FileNotFoundError(row_path)
        row = np.load(row_path)
        delta[:, target] = row["delta_nll"]
        kl[:, target] = row["kl"]
        cost[:, target] = row["cost"]
    np.fill_diagonal(delta, 0.0)
    np.fill_diagonal(kl, 0.0)
    np.fill_diagonal(cost, 0.0)
    bidirectional = np.maximum(cost, cost.T)
    cosine = np.load(output / "ffn_input_cosine.npy")
    ids = np.arange(layers)
    regimes = np.minimum(ids * int(cfg["replacement"]["depth_regimes"]) // layers,
                         int(cfg["replacement"]["depth_regimes"]) - 1)
    valid = (regimes[:, None] == regimes[None, :]) & (cosine >= float(cfg["replacement"]["input_cosine_min"]))
    np.fill_diagonal(valid, True)
    np.save(output / "directed_delta_nll.npy", delta)
    np.save(output / "directed_kl.npy", kl)
    np.save(output / "directed_cost.npy", cost)
    np.save(output / "bidirectional_max_cost.npy", bidirectional)
    np.save(output / "valid_pair_mask.npy", valid)
    manifest = base_manifest(cfg, "replacement_consolidated", backbone)
    manifest.update({"valid_tokens": valid_tokens, "layers": layers,
                     "lambda_kl": cfg["replacement"]["lambda_kl"],
                     "axis_convention": "[source_layer, target_layer]",
                     "depth_regimes": regimes.tolist(),
                     "input_cosine_min": cfg["replacement"]["input_cosine_min"],
                     "complete": bool(np.isfinite(cost).all())})
    atomic_json(output / "replacement_manifest.json", manifest)
