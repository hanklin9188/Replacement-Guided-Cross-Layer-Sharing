from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .grouping import build_groups
from .modeling import (build_shared_student, freeze_for_recovery,
                       load_compact_student, save_compact_student,
                       unique_trainable_parameters)
from .report import replacement_plots
from .utils import atomic_json


def run_smoke(cfg):
    smoke = copy.deepcopy(cfg)
    root = Path(cfg["project"]["output_root"]) / "smoke_fixture"
    smoke["project"]["output_root"] = str(root)
    backbone = "tiny"
    smoke["models"] = {backbone: {
        "model_id": "synthetic", "revision": "fixture", "tokenizer_revision": "fixture",
        "layers": 8, "budgets": [6, 4, 2], "main_k": 4, "dtype": "float32",
    }}
    smoke["replacement"]["depth_regimes"] = 2
    smoke["replacement"]["input_cosine_min"] = 0.5
    tokens, k = 128, 4
    source = root / backbone / "replacement" / f"tokens_{tokens}"
    source.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2027)
    directed = rng.uniform(0.0, 0.15, size=(8, 8))
    np.fill_diagonal(directed, 0.0)
    cosine = np.full((8, 8), 0.95)
    cosine[:4, 4:] = cosine[4:, :4] = 0.2
    weight = np.abs(np.subtract.outer(np.arange(8), np.arange(8))) / 8
    regimes = np.repeat([0, 1], 4)
    valid = regimes[:, None] == regimes[None, :]
    np.save(source / "directed_cost.npy", directed)
    np.save(source / "bidirectional_max_cost.npy", np.maximum(directed, directed.T))
    np.save(source / "ffn_input_cosine.npy", cosine)
    np.save(source / "normalized_weight_distance.npy", weight)
    np.save(source / "valid_pair_mask.npy", valid)
    build_groups(smoke, backbone, tokens, k)
    replacement_plots(smoke, backbone, tokens, k)
    group = root / backbone / "groups" / f"tokens_{tokens}" / f"k_{k}" / "full.json"
    figure = root / backbone / "analysis" / f"tokens_{tokens}" / f"k_{k}" / "replacement_diagnostics.pdf"
    if not group.exists() or not figure.exists() or figure.stat().st_size == 0:
        raise RuntimeError("Smoke fixture failed to create grouping/figure artifacts")

    tiny_config = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    model = LlamaForCausalLM(tiny_config).cuda()
    tiny_groups = {"groups": [[0, 1], [2, 3]], "representatives": [0, 2]}
    build = build_shared_student(model, tiny_groups, adapter_rank=4)
    freeze_for_recovery(build.model)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 12), device="cuda")
    loss = build.model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    gradients = [parameter.grad for parameter in unique_trainable_parameters(build.model)]
    if not gradients or not all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Tiny shared Llama did not produce finite gradients for every trainable parameter")
    checkpoint = root / "tiny_shared_student.pt"
    checkpoint_info = save_compact_student(checkpoint, build, {"smoke": True})
    extra = load_compact_student(checkpoint, build)
    if extra.get("smoke") is not True:
        raise RuntimeError("Compact checkpoint metadata round-trip failed")
    atomic_json(root / "SMOKE_SUCCESS.json", {
        "passed": True, "group_manifest": str(group), "figure": str(figure),
        "tiny_llama_loss": float(loss.detach()), "trainable_tensor_count": len(gradients),
        "compact_checkpoint": checkpoint_info,
    })
