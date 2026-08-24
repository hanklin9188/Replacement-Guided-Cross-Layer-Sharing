from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .config import model_config, run_dir
from .data import RecoveryDataset, causal_collator, load_task_rows
from .evaluation import evaluate_rows, heldout_nll
from .modeling import (
    SharedFFNWithAdapter,
    build_shared_student,
    decoder_layers,
    freeze_for_recovery,
    load_reference,
    load_tokenizer,
    load_compact_student,
    save_compact_student,
    unique_trainable_parameters,
)
from .teacher import load_teacher
from .utils import atomic_json, base_manifest, read_json, select_final, set_seed


def _manifest_path(cfg, backbone: str, valid_tokens: int, k: int, policy: str) -> Path:
    return run_dir(cfg, backbone, "groups", f"tokens_{valid_tokens}", f"k_{k}", f"{policy}.json")


def _capture_hooks(model, layer_ids: list[int], storage: dict[int, torch.Tensor], detach: bool):
    hooks = []
    for layer_id in layer_ids:
        def hook(_module, _args, output, layer_id=layer_id):
            storage[layer_id] = output.detach() if detach else output
        hooks.append(decoder_layers(model)[layer_id].mlp.register_forward_hook(hook))
    return hooks


def estimate_geometry(cfg: dict[str, Any], backbone: str, valid_tokens: int, k: int, policy: str = "full") -> None:
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    group_manifest = read_json(_manifest_path(cfg, backbone, valid_tokens, k, policy))
    teacher = load_teacher(cfg, backbone)
    tokenizer = load_tokenizer(model_config(cfg, backbone))
    rows = load_task_rows(run_dir(cfg, backbone, "data"), "recovery", cfg["data"]["tasks"])
    limit = int(cfg["recovery"]["geometry_examples"])
    rows = rows[:limit]
    dataset = RecoveryDataset(rows, tokenizer, int(cfg["recovery"]["max_length"]))
    loader = DataLoader(dataset, batch_size=int(cfg["recovery"]["batch_size"]), shuffle=False,
                        collate_fn=causal_collator(tokenizer), num_workers=2)
    layer_storage: dict[int, torch.Tensor] = {}
    layer_ids = list(range(len(decoder_layers(teacher))))
    hooks = _capture_hooks(teacher, layer_ids, layer_storage, detach=True)
    samples: dict[int, list[torch.Tensor]] = defaultdict(list)
    per_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    group_of = {}
    for group_id, group in enumerate(group_manifest["groups"]):
        for layer_id in group:
            group_of[int(layer_id)] = group_id
    with torch.inference_mode():
        for batch in loader:
            attention_mask = batch["attention_mask"].cuda()
            teacher(input_ids=batch["input_ids"].cuda(), attention_mask=attention_mask, use_cache=False)
            for layer_id in layer_ids:
                vector = select_final(layer_storage[layer_id], attention_mask).float().cpu()
                samples[group_of[layer_id]].append(vector)
                per_layer[layer_id].append(vector)
            layer_storage.clear()
    for hook in hooks:
        hook.remove()
    max_rank = max(int(value) for value in cfg["recovery"]["projection_ranks"])
    geometry = {"groups": {}, "layer_sigma": {}, "sample_ids": [row["id"] for row in rows]}
    for group_id in range(len(group_manifest["groups"])):
        values = torch.cat(samples[group_id], dim=0).float()
        center = values.mean(dim=0)
        centered = values - center
        q = min(max_rank, centered.shape[0] - 1, centered.shape[1])
        if q < 1:
            raise RuntimeError(f"Not enough teacher samples for PCA group {group_id}")
        _, _, basis = torch.pca_lowrank(centered.cuda(), q=q, center=False, niter=4)
        basis = basis.cpu()
        geometry["groups"][group_id] = {"center": center, "basis": basis, "available_rank": q}
        for layer_id in group_manifest["groups"][group_id]:
            layer_values = torch.cat(per_layer[int(layer_id)], dim=0).float()
            z = (layer_values - center) @ basis
            sigma = torch.sqrt(z.var(dim=0, unbiased=False) + float(cfg["recovery"]["epsilon"]))
            geometry["layer_sigma"][int(layer_id)] = sigma
    output = run_dir(cfg, backbone, "geometry", f"tokens_{valid_tokens}", f"k_{k}", policy)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(geometry, output / "geometry.pt")
    manifest = base_manifest(cfg, "teacher_geometry", backbone)
    manifest.update({"policy": policy, "k": k, "valid_tokens": valid_tokens,
                     "max_projection_rank": max_rank, "epsilon": cfg["recovery"]["epsilon"],
                     "sample_ids": geometry["sample_ids"], "teacher_geometry_frozen": True})
    atomic_json(output / "geometry_manifest.json", manifest)


def _selected_layers(group_manifest: dict[str, Any], mode: str) -> list[int]:
    if mode == "final":
        return [int(group_manifest["layers"]) - 1]
    if mode == "all_shared":
        return sorted(int(layer) for group in group_manifest["groups"] if len(group) > 1 for layer in group)
    raise ValueError(mode)


def _alignment_loss(variant: str, selected: list[int], teacher_updates, student_updates,
                    teacher_hidden, student_hidden, attention_mask, group_of, geometry, rank: int):
    losses = []
    for layer_id in selected:
        if variant == "hidden_mse":
            teacher_vec = select_final(teacher_hidden[layer_id + 1], attention_mask).float()
            student_vec = select_final(student_hidden[layer_id + 1], attention_mask).float()
            losses.append(F.mse_loss(student_vec, teacher_vec))
            continue
        teacher_vec = select_final(teacher_updates[layer_id], attention_mask).float()
        student_vec = select_final(student_updates[layer_id], attention_mask).float()
        if variant == "raw_update_mse":
            losses.append(F.mse_loss(student_vec, teacher_vec))
            continue
        group_id = group_of[layer_id]
        info = geometry["groups"][group_id]
        basis = info["basis"][:, :rank].to(student_vec.device)
        center = info["center"].to(student_vec.device)
        z_teacher = (teacher_vec - center) @ basis
        z_student = (student_vec - center) @ basis
        error = z_student - z_teacher
        if variant == "teacher_scaled_alignment":
            sigma = geometry["layer_sigma"][layer_id][:rank].to(error.device)
            error = error / sigma
        losses.append(error.square().mean())
    return torch.stack(losses).mean()


def train_student(cfg: dict[str, Any], backbone: str, valid_tokens: int, k: int,
                  policy: str, variant: str, seed: int, rank: int, lambda_align: float,
                  supervised: str, selection_only: bool = False) -> None:
    allowed = {"ce", "hidden_mse", "raw_update_mse", "projected_unscaled", "teacher_scaled_alignment"}
    if variant not in allowed:
        raise ValueError(f"Unknown recovery variant {variant}; choose {sorted(allowed)}")
    mcfg = model_config(cfg, backbone)
    if rank <= 0 or lambda_align < 0:
        selected_path = run_dir(cfg, backbone, "hyperparameters", f"tokens_{valid_tokens}",
                                f"k_{int(mcfg['main_k'])}", "full",
                                "selected.json")
        selected_hparams = read_json(selected_path)
        if rank <= 0:
            rank = int(selected_hparams["projection_rank"])
        if lambda_align < 0:
            lambda_align = float(selected_hparams["lambda_align"])
    set_seed(seed)
    group_manifest = read_json(_manifest_path(cfg, backbone, valid_tokens, k, policy))
    tokenizer = load_tokenizer(mcfg)
    build = build_shared_student(load_reference(mcfg, training=True), group_manifest,
                                 int(cfg["student"]["adapter_rank"]))
    freeze_for_recovery(build.model)
    teacher = None if variant == "ce" else load_teacher(cfg, backbone)
    selected = _selected_layers(group_manifest, supervised)
    group_of = {int(layer): group_id for group_id, group in enumerate(group_manifest["groups"]) for layer in group}
    geometry = None
    if variant in {"projected_unscaled", "teacher_scaled_alignment"}:
        geometry_path = run_dir(cfg, backbone, "geometry", f"tokens_{valid_tokens}", f"k_{k}", policy, "geometry.pt")
        geometry = torch.load(geometry_path, map_location="cpu", weights_only=False)
        if any(int(geometry["groups"][group_of[layer]]["available_rank"]) < rank for layer in selected):
            raise ValueError(f"Projection rank {rank} exceeds available teacher geometry")

    data_dir = run_dir(cfg, backbone, "data")
    train_rows = load_task_rows(data_dir, "recovery", cfg["data"]["tasks"])
    validation_rows = load_task_rows(data_dir, "validation", cfg["data"]["tasks"])
    per_task_limit = int(cfg["recovery"]["selection_examples_per_task"])
    selection_rows = []
    for task in cfg["data"]["tasks"]:
        selection_rows.extend([row for row in validation_rows if row["task"] == task][:per_task_limit])
    dataset = RecoveryDataset(train_rows, tokenizer, int(cfg["recovery"]["max_length"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(cfg["recovery"]["batch_size"]), shuffle=True,
                        generator=generator, collate_fn=causal_collator(tokenizer), num_workers=2)
    parameters = unique_trainable_parameters(build.model)
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg["recovery"]["learning_rate"]),
                                  weight_decay=float(cfg["recovery"]["weight_decay"]))
    updates_target = int(cfg["recovery"]["updates"])
    accumulation = int(cfg["recovery"]["gradient_accumulation"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(updates_target * float(cfg["recovery"]["warmup_ratio"])), updates_target
    )
    output = run_dir(cfg, backbone, "recovery", f"tokens_{valid_tokens}", f"k_{k}", policy,
                     variant, f"rank_{rank}", f"lambda_{lambda_align:g}", supervised, f"seed_{seed}")
    output.mkdir(parents=True, exist_ok=True)
    teacher_updates: dict[int, torch.Tensor] = {}
    student_updates: dict[int, torch.Tensor] = {}
    teacher_hooks = _capture_hooks(teacher, selected, teacher_updates, detach=True) if teacher is not None else []
    student_hooks = _capture_hooks(build.model, selected, student_updates, detach=False) if teacher is not None else []
    history = []
    best_key = (-float("inf"), float("inf"))
    best_update = -1
    update = 0
    micro_step = 0
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    while update < updates_target:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        inputs = {key: value.cuda() for key, value in batch.items() if torch.is_tensor(value)}
        attention_mask = inputs["attention_mask"]
        teacher_hidden = None
        if teacher is not None:
            with torch.inference_mode():
                teacher_out = teacher(input_ids=inputs["input_ids"], attention_mask=attention_mask,
                                      use_cache=False, output_hidden_states=variant == "hidden_mse")
                teacher_hidden = teacher_out.hidden_states
        student_out = build.model(**inputs, use_cache=False, output_hidden_states=variant == "hidden_mse")
        ce = student_out.loss
        align = torch.zeros((), device=ce.device)
        if teacher is not None:
            align = _alignment_loss(variant, selected, teacher_updates, student_updates,
                                    teacher_hidden, student_out.hidden_states, attention_mask,
                                    group_of, geometry, rank)
        total = ce + (0.0 if variant == "ce" else lambda_align * align)
        (total / accumulation).backward()
        micro_step += 1
        teacher_updates.clear()
        student_updates.clear()
        if micro_step % accumulation:
            continue
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        history.append({"update": update, "ce": float(ce.detach()), "alignment": float(align.detach()),
                        "total": float(total.detach()), "lr": scheduler.get_last_lr()[0]})
        interval = int(cfg["recovery"]["validation_interval"])
        if update % interval == 0 or update == updates_target:
            metrics = evaluate_rows(build.model, tokenizer, selection_rows, int(cfg["recovery"]["max_length"]))
            key = (metrics["macro_accuracy"], -float(total.detach()))
            history[-1]["selection_metrics"] = metrics
            if key > best_key:
                best_key = key
                best_update = update
                save_compact_student(output / "best_student.pt", build, {"update": update, "metrics": metrics})
            build.model.train()
    for hook in teacher_hooks + student_hooks:
        hook.remove()
    final_validation = evaluate_rows(build.model, tokenizer, validation_rows, int(cfg["recovery"]["max_length"]),
                                     output / "last_validation_predictions.jsonl")
    compact = save_compact_student(output / "last_student.pt", build, {"update": update, "metrics": final_validation})
    final_metrics = None
    final_nll = None
    if not selection_only:
        load_compact_student(output / "best_student.pt", build)
        final_rows = load_task_rows(data_dir, "final", cfg["data"]["tasks"])
        final_output = output / "final"
        final_output.mkdir(parents=True, exist_ok=True)
        final_metrics = evaluate_rows(build.model, tokenizer, final_rows, int(cfg["recovery"]["max_length"]),
                                      final_output / "predictions.jsonl")
        heldout_blocks = torch.load(data_dir / "heldout_nll_blocks.pt", map_location="cpu", weights_only=False)
        final_nll = heldout_nll(build.model, heldout_blocks, int(cfg["data"]["heldout_nll"]["valid_tokens"]))
        layers = decoder_layers(build.model)
        core_numel = sum(parameter.numel() for parameter in layers[0].mlp.core.parameters())
        adapter_numel = sum(parameter.numel() for layer in layers for parameter in layer.mlp.adapter.parameters())
        shared_total = sum(parameter.numel() for parameter in build.model.parameters())
        dense_total = shared_total - k * core_numel - adapter_numel + len(layers) * core_numel
        evaluation_manifest = base_manifest(cfg, "evaluation", backbone)
        evaluation_manifest.update({
            "role": "student", "split": "final", "metrics": final_metrics, "heldout_nll": final_nll,
            "sample_ids": [row["id"] for row in final_rows], "model_revision": mcfg["revision"],
            "tokenizer_revision": mcfg["tokenizer_revision"], "k": k, "policy": policy,
            "checkpoint": str(output / "best_student.pt"),
            "recovery": {"variant": variant, "seed": seed, "projection_rank": rank,
                         "lambda_align": lambda_align, "supervised_layers": selected,
                         "updates": updates_target, "groups_frozen_before_recovery": True,
                         "final_evaluation_used_for_selection": False},
            "structural_diagnostics": group_manifest["diagnostics"],
            "parameter_accounting": {
                "logical_layers": len(layers), "distinct_ffn_sets": k, "one_ffn_parameters": core_numel,
                "adapter_parameters": adapter_numel, "dense_total_parameters": dense_total,
                "shared_total_parameters": shared_total, "distinct_ffn_reduction": 1 - k / len(layers),
                "total_parameter_reduction_including_adapters": 1 - shared_total / dense_total,
                "compact_checkpoint_bytes": (output / "best_student.pt").stat().st_size,
            },
        })
        atomic_json(final_output / "metrics.json", evaluation_manifest)
    manifest = base_manifest(cfg, "student_recovery", backbone)
    manifest.update({
        "role": "M_S", "source_role": "M_P", "teacher_role": "M_T", "policy": policy,
        "group_manifest": str(_manifest_path(cfg, backbone, valid_tokens, k, policy)),
        "groups_frozen_before_recovery": True, "variant": variant, "seed": seed,
        "k": k, "adapter_rank": cfg["student"]["adapter_rank"], "projection_rank": rank,
        "lambda_align": lambda_align, "epsilon": cfg["recovery"]["epsilon"],
        "supervised_layers": selected, "updates": updates_target,
        "train_sample_ids": [row["id"] for row in train_rows],
        "validation_sample_ids": [row["id"] for row in validation_rows],
        "checkpoint_rule": cfg["recovery"]["checkpoint_rule"], "best_update": best_update,
        "history": history, "last_validation": final_validation, "final_metrics": final_metrics,
        "final_heldout_nll": final_nll, "selection_only": selection_only, "compact_checkpoint": compact,
        "final_evaluation_used_for_selection": False,
    })
    atomic_json(output / "recovery_manifest.json", manifest)


def select_hyperparameters(cfg: dict[str, Any], backbone: str, valid_tokens: int, k: int,
                           policy: str = "full") -> None:
    root = run_dir(cfg, backbone, "recovery", f"tokens_{valid_tokens}", f"k_{k}", policy,
                   "teacher_scaled_alignment")
    candidates = []
    for manifest_path in root.glob("rank_*/lambda_*/final/seed_*/recovery_manifest.json"):
        item = read_json(manifest_path)
        if int(item["seed"]) != int(cfg["project"]["seed"]):
            continue
        metrics = item.get("last_validation", {})
        candidates.append({
            "projection_rank": int(item["projection_rank"]),
            "lambda_align": float(item["lambda_align"]),
            "macro_accuracy": float(metrics["macro_accuracy"]),
            "weighted_accuracy": float(metrics["weighted_accuracy"]),
            "manifest": str(manifest_path),
        })
    expected = len(cfg["recovery"]["projection_ranks"]) * len(cfg["recovery"]["lambda_grid"])
    if len(candidates) != expected:
        raise RuntimeError(f"Hyperparameter pilot is incomplete: found {len(candidates)}, expected {expected}")
    selected = max(candidates, key=lambda row: (row["macro_accuracy"], row["weighted_accuracy"],
                                                -row["projection_rank"], -row["lambda_align"]))
    output = run_dir(cfg, backbone, "hyperparameters", f"tokens_{valid_tokens}", f"k_{k}", policy)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "selected.json", {
        **selected, "selection_split": "validation", "final_evaluation_used": False,
        "checkpoint_rule": cfg["recovery"]["checkpoint_rule"], "candidates": candidates,
    })
