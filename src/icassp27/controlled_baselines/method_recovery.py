from __future__ import annotations

import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from .config import raw_checkpoint_dir, rate_label, sha256_file
from .data import deterministic_indices, encode_multiple_choice, load_recovery_rows, sample_id_hash
from .distributed import barrier, initialize, is_main, seed_everything
from .modeling import (FactorizedLinear, load_compressed_checkpoint, save_compressed_checkpoint,
                       unique_parameter_count)
from .objectives import controlled_loss
from .train_eval import evaluate_seven_tasks, validation_decision_ce


METHODS = ("basis_sharing", "svd_llm")
BACKBONES = ("llama32_3b", "llama31_8b")
OBJECTIVES = ("ce", "ce_kd")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def load_method_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    cfg = yaml.safe_load(source.read_text())
    cfg["_config_path"] = str(source)
    cfg["_config_sha256"] = sha256_file(source)
    matrix = cfg["matrix"]
    if tuple(matrix["methods"]) != METHODS or tuple(matrix["backbones"]) != BACKBONES:
        raise ValueError("method-recovery matrix must contain the locked methods and backbones")
    if tuple(matrix["objectives"]) != OBJECTIVES or list(map(int, matrix["seeds"])) != [42, 43, 44]:
        raise ValueError("objectives/seeds must be CE, CE+KD and 42/43/44")
    if list(map(float, matrix["target_reductions"])) != [0.15, 0.20, 0.25]:
        raise ValueError("target reductions must be 15/20/25 percent")
    if int(cfg["slurm"]["gpus_per_job"]) != 4 or cfg["slurm"]["gpu_type"] != "H200":
        raise ValueError("every experimental job must request exactly 4 H200 GPUs")
    recovery = cfg["recovery"]
    if recovery["loss_scope"] != "decision" or recovery["selection_metric"] != "decision_ce":
        raise ValueError("the locked objective and checkpoint selector are decision CE")
    if list(recovery["svd_stage_order"]) != ["u", "v"]:
        raise ValueError("SVD-LLM stage order must be U then V")
    for backbone, value in cfg["backbones"].items():
        expected = int(value["per_gpu_batch"]) * int(value["gradient_accumulation"]) * 4
        if expected != int(value["effective_global_batch"]):
            raise ValueError(f"{backbone} effective_global_batch mismatch: {expected}")
    return cfg


def output_dir(cfg: dict[str, Any], spec: dict[str, Any]) -> Path:
    stage = spec["stage"]
    if stage == "recover":
        if spec.get("pilot", False):
            lr = f"{float(spec['learning_rate']):.0e}".replace("-", "m")
            return (Path(cfg["paths"]["pilot_root"]) / spec["method"] / spec["backbone"] /
                    spec["objective"] / f"lr_{lr}" / f"seed_{int(spec['seed'])}" /
                    rate_label(float(spec["target_reduction"])))
        return (Path(cfg["paths"]["result_root"]) / spec["method"] / spec["backbone"] /
                rate_label(float(spec["target_reduction"])) / f"seed_{int(spec['seed'])}" /
                spec["objective"])
    if stage == "pure":
        return (Path(cfg["paths"]["control_root"]) / "pure" / spec["method"] /
                spec["backbone"] / rate_label(float(spec["target_reduction"])))
    if stage == "dense":
        return Path(cfg["paths"]["control_root"]) / "dense" / spec["backbone"]
    if stage == "smoke":
        return (Path(cfg["paths"]["output_root"]) / "smoke" / spec["method"] /
                spec["backbone"] / spec["objective"])
    raise ValueError(stage)


def _set_basis_full_trainable(model: torch.nn.Module) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _factor_parameter_names(model: torch.nn.Module, factor: str | None = None) -> list[str]:
    result = []
    for module_name, module in model.named_modules():
        if not isinstance(module, FactorizedLinear):
            continue
        factors = (factor,) if factor is not None else ("u", "v")
        for selected in factors:
            projection = module.u_proj if selected == "u" else module.v_proj
            prefix = f"{module_name}.{selected}_proj"
            for local_name, _parameter in projection.named_parameters(recurse=False):
                result.append(f"{prefix}.{local_name}")
    if not result:
        raise RuntimeError(f"no FactorizedLinear {factor or 'U/V'} parameters found")
    return sorted(result)


def _set_svd_factor_trainable(model: torch.nn.Module, factor: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = set(_factor_parameter_names(model, factor))
    observed = []
    for name, parameter in model.named_parameters():
        if name in selected:
            parameter.requires_grad_(True)
            observed.append(name)
    if set(observed) != selected:
        missing = sorted(selected - set(observed))
        raise RuntimeError(f"failed to activate exact SVD factor parameters: {missing[:8]}")
    return observed


def _snapshot(model: torch.nn.Module, names: Iterable[str]) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in names}


@torch.no_grad()
def _restore(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"snapshot contains unknown parameters: {missing[:8]}")
    for name, value in state.items():
        parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))


def _lr_at_step(base_lr: float, step: int, total_steps: int, warmup_steps: int,
                minimum_ratio: float) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return base_lr * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def _trainable_report(model: torch.nn.Module, expected_names: Iterable[str], scope: str) -> dict[str, Any]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = sorted(expected_names)
    if sorted(names) != expected:
        raise RuntimeError(f"trainable scope mismatch for {scope}: expected={len(expected)} observed={len(names)}")
    parameters = dict(model.named_parameters())
    return {"scope": scope, "trainable_parameters": sum(parameters[name].numel() for name in names),
            "trainable_parameter_tensors": len(names), "trainable_names": names}


def _teacher(cfg: dict[str, Any], backbone: str, objective: str, device: torch.device,
             student) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    if objective == "ce":
        return None, {"teacher_loaded": False, "teacher_forward_executed": False,
                      "teacher_frozen": False, "teacher_path": None, "teacher_sha256": None}
    from transformers import AutoModelForCausalLM
    root = Path(cfg["backbones"][backbone]["teacher_path"])
    teacher = AutoModelForCausalLM.from_pretrained(
        root, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).to(device)
    teacher.eval()
    teacher.config.use_cache = False
    teacher.requires_grad_(False)
    if int(teacher.config.vocab_size) != int(student.config.vocab_size):
        raise RuntimeError("teacher/student vocabulary sizes differ")
    weights = root / "model.safetensors"
    return teacher, {"teacher_loaded": True, "teacher_forward_executed": True,
                     "teacher_frozen": True, "teacher_path": str(root),
                     "teacher_sha256": sha256_file(weights)}


def _ddp(model: torch.nn.Module, device: torch.device) -> DistributedDataParallel:
    return DistributedDataParallel(model, device_ids=[device.index], output_device=device.index,
                                   broadcast_buffers=False, find_unused_parameters=False)


def _recovery_rows(cfg: dict[str, Any], backbone_cfg: dict[str, Any]):
    cache = Path(cfg["paths"]["output_root"]) / "data_cache/recovery_rows.pt"
    if cache.is_file():
        value = torch.load(cache, map_location="cpu", weights_only=False)
        if (value["train_source_sha256"] == sha256_file(backbone_cfg["recovery_train"]) and
                value["validation_source_sha256"] == sha256_file(backbone_cfg["recovery_validation"])):
            return value["train_rows"], value["validation_rows"]
    return (load_recovery_rows(backbone_cfg["recovery_train"]),
            load_recovery_rows(backbone_cfg["recovery_validation"]))


def recover(cfg: dict[str, Any], spec: dict[str, Any]) -> Path:
    method, backbone = spec["method"], spec["backbone"]
    objective, seed = spec["objective"], int(spec["seed"])
    reduction = float(spec["target_reduction"])
    if method not in METHODS or backbone not in BACKBONES or objective not in OBJECTIVES:
        raise ValueError(spec)
    rank, world, device = initialize()
    if world != 4:
        raise RuntimeError(f"method-specific recovery requires exactly 4 H200 workers, got {world}")
    output = output_dir(cfg, spec)
    if (output / ".complete").is_file():
        barrier()
        return output
    if is_main():
        output.mkdir(parents=True, exist_ok=True)
    barrier()
    seed_everything(seed)

    backbone_cfg = cfg["backbones"][backbone]
    recovery_cfg = cfg["recovery"]
    pilot = bool(spec.get("pilot", False))
    smoke = spec["stage"] == "smoke"
    raw = raw_checkpoint_dir(cfg, method, backbone, reduction)
    model, tokenizer, structure = load_compressed_checkpoint(raw, dtype=torch.bfloat16)
    model.config.use_cache = False
    if bool(recovery_cfg["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(device)

    train_rows, validation_rows = _recovery_rows(cfg, backbone_cfg)
    validation_maximum = int(spec.get(
        "validation_maximum",
        cfg["lr_tuning"]["validation_maximum"] if pilot else recovery_cfg["validation_maximum"],
    ))
    validation_rows = validation_rows[:validation_maximum]
    teacher, teacher_report = _teacher(cfg, backbone, objective, device, model)

    total_steps = int(spec.get(
        "max_steps", cfg["lr_tuning"]["pilot_steps"] if pilot else backbone_cfg["max_steps"]
    ))
    if total_steps < (2 if method == "svd_llm" else 1):
        raise ValueError("total steps are too small for the requested recovery")
    base_lr = float(spec["learning_rate"])
    accumulation = int(backbone_cfg["gradient_accumulation"])
    per_gpu_batch = int(backbone_cfg["per_gpu_batch"])
    effective_batch = world * accumulation * per_gpu_batch
    if effective_batch != int(backbone_cfg["effective_global_batch"]):
        raise RuntimeError("effective global batch invariant violated")
    warmup_steps = min(int(backbone_cfg["warmup_steps"]), max(1, total_steps // 10)) if pilot or smoke \
        else int(backbone_cfg["warmup_steps"])
    validation_interval = int(spec.get("validation_interval", min(250, total_steps)
                                       if pilot or smoke else recovery_cfg["validation_interval"]))
    total_examples = total_steps * accumulation * world * per_gpu_batch
    order = deterministic_indices(len(train_rows), 0, total_examples, seed)

    if method == "basis_sharing":
        all_update_names = _set_basis_full_trainable(model)
        stages = [("full", 1, total_steps)]
    else:
        all_update_names = _factor_parameter_names(model)
        split = total_steps // 2
        stages = [("u", 1, split), ("v", split + 1, total_steps)]

    history: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    best_value = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.time()

    def validate(step: int, stage_name: str, *, eligible: bool) -> None:
        nonlocal best_value, best_step, best_state
        metrics = validation_decision_ce(
            model, tokenizer, validation_rows, max_length=int(recovery_cfg["max_length"]),
            batch_size=int(backbone_cfg["eval_batch"]), rank=rank, world=world, device=device,
        )
        row = {"step": step, "stage": stage_name, "validation_decision_ce": metrics["decision_ce"],
               "validation_accuracy": metrics["accuracy"], "validation_examples": metrics["examples"],
               "checkpoint_eligible": eligible}
        history.append(row)
        if eligible and metrics["decision_ce"] < best_value - float(recovery_cfg["minimum_improvement"]):
            best_value = metrics["decision_ce"]
            best_step = step
            best_state = _snapshot(model, all_update_names)

    for stage_index, (stage_name, first_step, last_step) in enumerate(stages):
        if method == "basis_sharing":
            active_names = _set_basis_full_trainable(model)
            scope = "all_existing_compressed_model_parameters"
        else:
            active_names = _set_svd_factor_trainable(model, stage_name)
            scope = f"factor_{stage_name}_only"
        report = _trainable_report(model, active_names, scope)
        report.update({"stage": stage_name, "first_step": first_step, "last_step": last_step,
                       "steps": last_step - first_step + 1})
        stage_reports.append(report)
        training_model = _ddp(model, device)
        parameters = [parameter for parameter in training_model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=base_lr,
                                      weight_decay=float(recovery_cfg["weight_decay"]))
        optimizer.zero_grad(set_to_none=True)

        if method == "svd_llm" and stage_name == "v":
            validate(first_step - 1, "v_initial_after_u", eligible=True)
            training_model.train()

        for global_step in range(first_step, last_step + 1):
            learning_rate = _lr_at_step(base_lr, global_step, total_steps, warmup_steps,
                                        float(recovery_cfg["minimum_lr_ratio"]))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            step_total = step_ce = step_kd = 0.0
            for micro in range(accumulation):
                global_micro = (global_step - 1) * accumulation + micro
                global_start = global_micro * world * per_gpu_batch + rank * per_gpu_batch
                indices = order[global_start:global_start + per_gpu_batch]
                rows = [train_rows[index] for index in indices]
                encoded = encode_multiple_choice(rows, tokenizer, int(recovery_cfg["max_length"]), device)
                sync = nullcontext() if micro == accumulation - 1 else training_model.no_sync()
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, ce, kd, _ = controlled_loss(
                        training_model, teacher, rows, encoded, objective=objective,
                        temperature=float(recovery_cfg["temperature"]),
                        lambda_ce=float(recovery_cfg["lambda_ce"]),
                        lambda_kd=float(recovery_cfg["lambda_kd"]),
                        eos_token_id=tokenizer.eos_token_id,
                        exclude_eos=bool(recovery_cfg["loss_exclude_eos"]),
                    )
                    (loss / accumulation).backward()
                step_total += float(loss.detach()) / accumulation
                step_ce += float(ce.detach()) / accumulation
                step_kd += float(kd.detach()) / accumulation
            torch.nn.utils.clip_grad_norm_(parameters, float(recovery_cfg["gradient_clip_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if is_main() and (global_step % 100 == 0 or global_step == last_step):
                _json(output / "progress.json", {
                    "status": "training", "method": method, "backbone": backbone,
                    "objective": objective, "seed": seed, "stage": stage_name,
                    "step": global_step, "total_steps": total_steps,
                    "fraction": global_step / total_steps,
                    "training_total_rank0": step_total,
                    "training_decision_ce_rank0": step_ce,
                    "training_kd_rank0": step_kd,
                    "learning_rate": learning_rate,
                    "elapsed_seconds": time.time() - started,
                })

            if global_step % validation_interval == 0 or global_step == last_step:
                eligible = method == "basis_sharing" or stage_name == "v"
                validate(global_step, stage_name, eligible=eligible)
                history[-1].update({"training_total_rank0": step_total,
                                    "training_decision_ce_rank0": step_ce,
                                    "training_kd_rank0": step_kd,
                                    "learning_rate": learning_rate})
                training_model.train()

        del optimizer, training_model
        torch.cuda.empty_cache()
        barrier()
        if method == "svd_llm" and stage_name == "u" and is_main():
            _torch_save(output / "stage1_u/factor_state.pt", _snapshot(model, all_update_names))
            _json(output / "stage1_u/manifest.json", {**report, "raw_checkpoint": str(raw),
                                                       "next_stage": "v"})
        barrier()

    if best_state is None or best_step == 0:
        raise RuntimeError("validation never selected an eligible checkpoint")
    _restore(model, best_state)
    model.to(dtype=torch.float16)
    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()
    barrier()

    metrics = None
    if not pilot and not smoke:
        metrics = evaluate_seven_tasks(model, tokenizer, cfg, backbone, output, rank, world, device)

    if is_main():
        compression = json.loads((raw / "compression_report.json").read_text())
        recovered_parameters = unique_parameter_count(model)
        recovered_reduction = 1.0 - recovered_parameters / int(compression["dense_parameters"])
        if recovered_parameters != int(compression["compressed_parameters"]):
            raise RuntimeError("method-specific recovery unexpectedly changed the deployment parameter count")
        run_config = {
            "schema_version": 1, "config_path": cfg["_config_path"],
            "config_sha256": cfg["_config_sha256"], "spec": spec,
            "raw_checkpoint": str(raw), "raw_compression_report_sha256": sha256_file(raw / "compression_report.json"),
            "recovery_train": backbone_cfg["recovery_train"],
            "recovery_train_sha256": sha256_file(backbone_cfg["recovery_train"]),
            "recovery_train_ids_sha256": sample_id_hash(train_rows),
            "recovery_validation": backbone_cfg["recovery_validation"],
            "recovery_validation_sha256": sha256_file(backbone_cfg["recovery_validation"]),
            "recovery_validation_ids_sha256": sample_id_hash(validation_rows),
            "sample_order_seed": seed, "world_size": world, "gpu_type": "H200",
        }
        training_report = {
            "method": method, "backbone": backbone, "target_reduction": reduction,
            "objective": objective, "seed": seed, "recovery_protocol":
                "basis_full_parameter" if method == "basis_sharing" else "svd_sequential_u_then_v",
            "loss_scope": "multiple_choice_candidate_decision_ce",
            "kd_scope": "all_non_padding_shifted_tokens_full_vocabulary" if objective == "ce_kd" else None,
            "lambda_ce": float(recovery_cfg["lambda_ce"]),
            "lambda_kd": float(recovery_cfg["lambda_kd"]) if objective == "ce_kd" else 0.0,
            "temperature": float(recovery_cfg["temperature"]) if objective == "ce_kd" else None,
            "total_steps": total_steps, "per_gpu_batch": per_gpu_batch,
            "gradient_accumulation": accumulation, "world_size": world,
            "effective_global_batch": effective_batch, "learning_rate": base_lr,
            "warmup_steps": warmup_steps, "best_step": best_step,
            "best_validation_decision_ce": best_value, "selection_metric": "minimum validation decision CE",
            "validation_test_isolation": True, "stages": stage_reports,
            "updated_parameter_names": all_update_names,
            "updated_parameters": sum(dict(model.named_parameters())[name].numel() for name in all_update_names),
            "dense_weights_recreated": False, "generic_lora_used": False,
            "elapsed_seconds": time.time() - started, **teacher_report,
        }
        _json(output / "run_config.json", run_config)
        _json(output / "training_report.json", training_report)
        _json(output / "validation_history.json", history)
        summary = {
            "method": method, "backbone": backbone, "target_reduction": reduction,
            "actual_raw_reduction": float(compression["actual_reduction"]),
            "objective": objective, "seed": seed, "learning_rate": base_lr,
            "recovery_protocol": training_report["recovery_protocol"],
            "total_steps": total_steps, "best_step": best_step,
            "best_validation_decision_ce": best_value,
            "recovered_total_parameters": recovered_parameters,
            "recovered_reduction": recovered_reduction,
            "pilot": pilot, "smoke": smoke,
        }
        if metrics is not None:
            names = {"social_i_qa": "social_iqa"}
            summary.update({names.get(task, task): float(metrics[task])
                            for task in cfg["evaluation"]["tasks"]})
            summary["macro"] = float(metrics["macro"])

        if not pilot and not smoke:
            best_dir = output / "best_model"
            parameter_report = {
                "dense_parameters": int(compression["dense_parameters"]),
                "compressed_parameters": recovered_parameters,
                "unique_parameters": recovered_parameters,
                "actual_reduction": recovered_reduction,
                "generic_lora_parameters": 0,
                "dense_weights_recreated": False,
                "shared_parameter_aliases_preserved": method == "basis_sharing",
            }
            save_compressed_checkpoint(model.cpu(), tokenizer, structure, best_dir, {
                "compression_report.json": {**compression, "already_recovered": True,
                                             "recovery_protocol": training_report["recovery_protocol"]},
                "parameter_report.json": parameter_report,
            })
            standalone_bytes = sum(path.stat().st_size for path in best_dir.rglob("*") if path.is_file())
            summary["standalone_bytes"] = standalone_bytes
            _json(output / "parameter_report.json", {**parameter_report,
                                                       "standalone_bytes": standalone_bytes})
            if method == "svd_llm":
                _torch_save(output / "stage2_v/factor_state.pt", best_state)
                _json(output / "stage2_v/manifest.json", {"best_step": best_step,
                                                           "best_validation_decision_ce": best_value,
                                                           "raw_checkpoint": str(raw)})
        _json(output / "summary.json", summary)
        _json(output / "progress.json", {"status": "complete", "step": total_steps,
                                          "total_steps": total_steps, "fraction": 1.0,
                                          "elapsed_seconds": time.time() - started})
        (output / ".complete").write_text("PASS\n")
    barrier()
    return output


def evaluate_control(cfg: dict[str, Any], spec: dict[str, Any]) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rank, world, device = initialize()
    if world != 4:
        raise RuntimeError(f"control evaluation requires exactly 4 H200 workers, got {world}")
    output = output_dir(cfg, spec)
    if (output / ".complete").is_file():
        barrier()
        return output
    if is_main():
        output.mkdir(parents=True, exist_ok=True)
    barrier()
    backbone = spec["backbone"]
    if spec["stage"] == "pure":
        raw = raw_checkpoint_dir(cfg, spec["method"], backbone, float(spec["target_reduction"]))
        model, tokenizer, _ = load_compressed_checkpoint(raw, dtype=torch.float16)
        compression = json.loads((raw / "compression_report.json").read_text())
        source = str(raw)
    else:
        backbone_cfg = cfg["backbones"][backbone]
        model = AutoModelForCausalLM.from_pretrained(backbone_cfg["model_path"], torch_dtype=torch.float16,
                                                     local_files_only=True, attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained(backbone_cfg["model_path"], local_files_only=True,
                                                  fix_mistral_regex=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dense = unique_parameter_count(model)
        compression = {"actual_reduction": 0.0, "dense_parameters": dense,
                       "compressed_parameters": dense, "standalone_bytes": 0}
        source = backbone_cfg["model_path"]
    model.to(device)
    metrics = evaluate_seven_tasks(model, tokenizer, cfg, backbone, output, rank, world, device)
    if is_main():
        names = {"social_i_qa": "social_iqa"}
        summary = {"stage": spec["stage"], "method": spec.get("method", "dense"),
                   "backbone": backbone, "target_reduction": spec.get("target_reduction", 0.0),
                   "actual_raw_reduction": float(compression["actual_reduction"]),
                   "source": source, "world_size": world,
                   **{names.get(task, task): float(metrics[task]) for task in cfg["evaluation"]["tasks"]},
                   "macro": float(metrics["macro"])}
        _json(output / "summary.json", summary)
        (output / ".complete").write_text("PASS\n")
    barrier()
    return output
