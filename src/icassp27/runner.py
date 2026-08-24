from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import model_config, run_dir
from .evaluation import evaluate_split, heldout_nll
from .modeling import (build_shared_student, decoder_layers, load_compact_student,
                       load_reference, load_tokenizer)
from .teacher import load_teacher
from .utils import atomic_json, base_manifest, read_json


def load_role(cfg, backbone: str, role: str, valid_tokens: int | None = None,
              k: int | None = None, policy: str = "full", checkpoint: str | None = None):
    mcfg = model_config(cfg, backbone)
    tokenizer = load_tokenizer(mcfg)
    if role == "reference":
        return load_reference(mcfg), tokenizer, None
    if role == "teacher":
        return load_teacher(cfg, backbone), tokenizer, None
    if role in {"step0", "student"}:
        if valid_tokens is None or k is None:
            raise ValueError("Shared roles require valid_tokens and k")
        manifest = read_json(run_dir(cfg, backbone, "groups", f"tokens_{valid_tokens}", f"k_{k}", f"{policy}.json"))
        build = build_shared_student(load_reference(mcfg), manifest, int(cfg["student"]["adapter_rank"]))
        extra = None
        if role == "student":
            if not checkpoint:
                raise ValueError("student role requires --checkpoint")
            extra = load_compact_student(checkpoint, build)
        build.model.eval()
        return build.model, tokenizer, {"build": build, "checkpoint_extra": extra, "group_manifest": manifest}
    raise ValueError(role)


def evaluate_role(cfg, backbone: str, role: str, output: str, valid_tokens: int | None = None,
                  k: int | None = None, policy: str = "full", checkpoint: str | None = None) -> None:
    model, tokenizer, context = load_role(cfg, backbone, role, valid_tokens, k, policy, checkpoint)
    output_path = Path(output)
    metrics = evaluate_split(cfg, backbone, role, model, tokenizer, run_dir(cfg, backbone, "data"),
                             output_path, "final")
    blocks = torch.load(run_dir(cfg, backbone, "data", "heldout_nll_blocks.pt"), map_location="cpu", weights_only=False)
    nll = heldout_nll(model, blocks, int(cfg["data"]["heldout_nll"]["valid_tokens"]))
    manifest = read_json(output_path / "metrics.json")
    manifest["heldout_nll"] = nll
    manifest["model_revision"] = model_config(cfg, backbone)["revision"]
    manifest["tokenizer_revision"] = model_config(cfg, backbone)["tokenizer_revision"]
    manifest["k"] = k
    manifest["policy"] = policy if role in {"step0", "student"} else None
    manifest["checkpoint"] = checkpoint
    if checkpoint:
        recovery_manifest = Path(checkpoint).parent / "recovery_manifest.json"
        if recovery_manifest.exists():
            recovery = read_json(recovery_manifest)
            manifest["recovery"] = {key: recovery.get(key) for key in [
                "variant", "seed", "projection_rank", "lambda_align", "supervised_layers",
                "updates", "group_manifest", "groups_frozen_before_recovery",
                "checkpoint_rule", "best_update", "final_evaluation_used_for_selection",
            ]}
    if context and context.get("group_manifest"):
        manifest["structural_diagnostics"] = context["group_manifest"]["diagnostics"]
        layers = decoder_layers(model)
        ffn_numel = _unique_numel([layers[0].mlp.core])
        adapter_numel = _unique_numel([layer.mlp.adapter for layer in layers])
        shared_total = _unique_numel([model])
        dense_total = shared_total - int(k) * ffn_numel - adapter_numel + len(layers) * ffn_numel
        manifest["parameter_accounting"] = {
            "logical_layers": len(layers), "distinct_ffn_sets": int(k),
            "one_ffn_parameters": ffn_numel, "adapter_parameters": adapter_numel,
            "dense_total_parameters": dense_total, "shared_total_parameters": shared_total,
            "distinct_ffn_reduction": 1.0 - int(k) / len(layers),
            "total_parameter_reduction_including_adapters": 1.0 - shared_total / dense_total,
            "compact_checkpoint_bytes": Path(checkpoint).stat().st_size if checkpoint else None,
        }
    else:
        manifest["parameter_accounting"] = {
            "logical_layers": len(decoder_layers(model)), "distinct_ffn_sets": len(decoder_layers(model)),
            "dense_total_parameters": _unique_numel([model]), "shared_total_parameters": _unique_numel([model]),
            "distinct_ffn_reduction": 0.0, "total_parameter_reduction_including_adapters": 0.0,
        }
    atomic_json(output_path / "metrics.json", manifest)


def _unique_numel(modules) -> int:
    seen, total = set(), 0
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
    return total


@torch.inference_mode()
def efficiency_audit(cfg, backbone: str, role: str, output: str, valid_tokens: int | None = None,
                     k: int | None = None, policy: str = "full", checkpoint: str | None = None):
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model, tokenizer, context = load_role(cfg, backbone, role, valid_tokens, k, policy, checkpoint)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    ecfg = cfg["efficiency"]
    prompt_len = int(ecfg["prompt_length"])
    generate_len = int(ecfg["generation_length"])
    batch_size = int(ecfg["batch_size"])
    vocab = int(model.config.vocab_size)
    generator = torch.Generator(device="cuda").manual_seed(int(cfg["project"]["seed"]))
    input_ids = torch.randint(0, vocab, (batch_size, prompt_len), generator=generator, device="cuda")
    warmup = int(ecfg["warmup_repetitions"])
    repetitions = int(ecfg["timed_repetitions"])
    for _ in range(warmup):
        model(input_ids=input_ids, use_cache=False)
    prefill_times = []
    for _ in range(repetitions):
        torch.cuda.synchronize(); begin = time.perf_counter()
        model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize(); prefill_times.append(time.perf_counter() - begin)
    for _ in range(warmup):
        model.generate(input_ids, max_new_tokens=generate_len, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    decode_times = []
    for _ in range(repetitions):
        torch.cuda.synchronize(); begin = time.perf_counter()
        model.generate(input_ids, max_new_tokens=generate_len, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize(); decode_times.append(time.perf_counter() - begin)
    layers = decoder_layers(model)
    if context and context.get("build"):
        ffn_numel = _unique_numel([layer.mlp.core for layer in layers])
        adapter_numel = _unique_numel([layer.mlp.adapter for layer in layers])
        distinct_ffn = k
    else:
        ffn_numel = _unique_numel([layer.mlp for layer in layers])
        adapter_numel = 0
        distinct_ffn = len(layers)
    total_numel = _unique_numel([model])
    checkpoint_bytes = Path(checkpoint).stat().st_size if checkpoint else None
    def stats(values):
        return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "median": statistics.median(values)}
    result = base_manifest(cfg, "efficiency", backbone)
    result.update({
        "role": role, "k": k, "logical_layers": len(layers), "distinct_ffn_sets": distinct_ffn,
        "ffn_parameter_count": ffn_numel, "adapter_parameter_count": adapter_numel,
        "total_parameter_count_unique": total_numel, "compact_checkpoint_bytes": checkpoint_bytes,
        "peak_accelerator_bytes": torch.cuda.max_memory_allocated(), "model_load_seconds": load_seconds,
        "protocol": ecfg, "precision": model_config(cfg, backbone)["dtype"],
        "prefill_seconds": stats(prefill_times),
        "prefill_tokens_per_second": stats([batch_size * prompt_len / x for x in prefill_times]),
        "decode_seconds": stats(decode_times),
        "decode_tokens_per_second": stats([batch_size * generate_len / x for x in decode_times]),
        "flop_reduction_claimed": False,
    })
    atomic_json(output, result)
