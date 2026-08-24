from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import raw_checkpoint_dir, result_dir, sha256_file
from .data import (deterministic_indices, encode_multiple_choice, load_evaluation_rows,
                   load_recovery_rows, sample_id_hash)
from .distributed import all_gather_objects, barrier, initialize, is_main, seed_everything
from .modeling import (adapter_state_dict, freeze_and_inject_lora, load_adapter_state,
                       load_compressed_checkpoint, trainable_parameter_report, unique_parameter_count)
from .objectives import candidate_scores, controlled_loss, decision_cross_entropy


def _summary_metrics(metrics: dict[str, Any], tasks: list[str]) -> dict[str, float]:
    names = {"social_i_qa": "social_iqa"}
    return {names.get(task, task): float(metrics[task]) for task in tasks}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _save_adapter(model, output: Path, recovery_cfg: dict[str, Any], modules: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "adapter_model.pt.tmp"
    torch.save(adapter_state_dict(model), temporary)
    temporary.replace(output / "adapter_model.pt")
    _atomic_json(output / "adapter_config.json", {
        "adapter_type": "standard_lora_residual_outside_complete_compressed_projection",
        "rank": int(recovery_cfg["rank"]), "alpha": float(recovery_cfg["alpha"]),
        "scaling": float(recovery_cfg["alpha"]) / int(recovery_cfg["rank"]),
        "dropout": float(recovery_cfg["dropout"]), "bias": recovery_cfg["bias"],
        "target_modules": list(recovery_cfg["target_modules"]), "injected_modules": modules,
        "base_model_frozen": True,
    })


def _cosine_scheduler(optimizer, total_steps: int, warmup_steps: int, minimum_ratio: float):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(1.0e-12, float(step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.inference_mode()
def validation_decision_ce(model, tokenizer, rows: list[dict[str, Any]], *, max_length: int,
                           batch_size: int, rank: int, world: int, device: torch.device) -> dict[str, float]:
    model.eval()
    local = rows[rank::world]
    total_loss, total_rows, total_correct = 0.0, 0, 0
    for offset in range(0, len(local), batch_size):
        batch = local[offset:offset + batch_size]
        encoded = encode_multiple_choice(batch, tokenizer, max_length, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                           use_cache=False).logits
        scores = candidate_scores(logits, encoded)
        loss = decision_cross_entropy(scores, batch, encoded["row_spans"])
        total_loss += float(loss) * len(batch)
        total_rows += len(batch)
        for row, (start, end) in zip(batch, encoded["row_spans"]):
            total_correct += int(int(scores[start:end].argmax()) == int(row["label"]))
    totals = torch.tensor([total_loss, total_rows, total_correct], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {"decision_ce": float(totals[0] / totals[1]),
            "accuracy": float(totals[2] / totals[1]), "examples": int(totals[1])}


@torch.inference_mode()
def evaluate_seven_tasks(model, tokenizer, cfg: dict[str, Any], backbone: str, output: Path,
                         rank: int, world: int, device: torch.device) -> dict[str, Any] | None:
    evaluation = cfg["evaluation"]
    backbone_cfg = cfg["backbones"][backbone]
    rows = load_evaluation_rows(backbone_cfg["evaluation_data_root"], evaluation["tasks"], evaluation["split"])
    indexed = list(enumerate(rows))
    local = indexed[rank::world]
    predictions = []
    batch_size = int(backbone_cfg["eval_batch"])
    model.eval()
    for offset in range(0, len(local), batch_size):
        batch_pairs = local[offset:offset + batch_size]
        batch = [row for _, row in batch_pairs]
        encoded = encode_multiple_choice(batch, tokenizer, int(cfg["recovery"]["max_length"]), device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                           use_cache=False).logits
        scores = candidate_scores(logits, encoded)
        for (original_index, row), (start, end) in zip(batch_pairs, encoded["row_spans"]):
            row_scores = [float(value) for value in scores[start:end].cpu()]
            prediction = max(range(len(row_scores)), key=row_scores.__getitem__)
            predictions.append({
                "_index": original_index, "id": row["id"], "task": row["task"],
                "label": int(row["label"]), "prediction": prediction,
                "correct": int(prediction == int(row["label"])), "choice_scores": row_scores,
                "length_norm": "none", "shots": 0,
            })
    gathered = all_gather_objects(predictions)
    if not is_main():
        return None
    merged = sorted((item for part in gathered for item in part), key=lambda item: item.pop("_index"))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in merged:
        by_task[item["task"]].append(item)
    metrics = {}
    for task in evaluation["tasks"]:
        task_rows = by_task[task]
        accuracy = sum(row["correct"] for row in task_rows) / len(task_rows)
        metrics[task] = accuracy
        _atomic_json(output / f"{task}.json", {
            "task": task, "accuracy": accuracy, "examples": len(task_rows),
            "protocol": {"shots": 0, "mode": "logprob", "length_norm": "none",
                         "dtype": "float16", "quantization": False, "split": "complete test"},
            "predictions": task_rows,
        })
    metrics["macro"] = sum(metrics[task] for task in evaluation["tasks"]) / len(evaluation["tasks"])
    metrics["examples"] = len(merged)
    return metrics


def _base_summary(cfg: dict[str, Any], method: str, backbone: str, target_reduction: float,
                  objective: str, seed: int | None, checkpoint: Path) -> dict[str, Any]:
    compression = json.loads((checkpoint / "compression_report.json").read_text())
    return {
        "method": method, "backbone": backbone, "target_reduction": float(target_reduction),
        "actual_raw_reduction": float(compression["actual_reduction"]),
        "objective": objective, "seed": seed,
    }


def pure_evaluate(cfg: dict[str, Any], method: str, backbone: str, target_reduction: float) -> Path:
    rank, world, device = initialize()
    if world != 1:
        raise RuntimeError(f"pure evaluation requires exactly 1 H200 worker, got {world}")
    checkpoint = raw_checkpoint_dir(cfg, method, backbone, target_reduction)
    output = result_dir(cfg, method, backbone, target_reduction, "pure")
    if (output / ".complete").is_file():
        barrier()
        return output
    model, tokenizer, _ = load_compressed_checkpoint(checkpoint, dtype=torch.float16)
    model.to(device)
    metrics = evaluate_seven_tasks(model, tokenizer, cfg, backbone, output, rank, world, device)
    if is_main():
        compression = json.loads((checkpoint / "compression_report.json").read_text())
        parameter = json.loads((checkpoint / "parameter_report.json").read_text())
        shutil.copy2(checkpoint / "compression_report.json", output / "compression_report.json")
        shutil.copy2(checkpoint / "parameter_report.json", output / "parameter_report.json")
        summary = _base_summary(cfg, method, backbone, target_reduction, "pure", None, checkpoint)
        summary.update({
            "trainable_parameters": 0, "adapter_parameters": 0,
            "recovered_total_parameters": int(parameter["compressed_parameters"]),
            "recovered_reduction": float(compression["actual_reduction"]),
            "best_step": None, "best_validation_decision_ce": None,
            **_summary_metrics(metrics, cfg["evaluation"]["tasks"]), "macro": metrics["macro"],
        })
        _atomic_json(output / "pure_summary.json", summary)
        _atomic_json(output / "summary.json", summary)
        _atomic_json(output / "deploy_load_report.json", {
            "checkpoint": str(checkpoint), "loader": str(checkpoint / "model_loader.py"),
            "load_pass": True, "standalone_bytes": compression["standalone_bytes"],
        })
        (output / ".complete").write_text("PASS\n")
    barrier()
    return output


def recover_and_evaluate(cfg: dict[str, Any], method: str, backbone: str, target_reduction: float,
                         objective: str, seed: int) -> Path:
    from transformers import AutoModelForCausalLM

    rank, world, device = initialize()
    if world != 1:
        raise RuntimeError(f"recovery requires exactly 1 H200 worker, got {world}")
    if objective not in {"ce", "ce_kd"}:
        raise ValueError(objective)
    seed_everything(seed)
    checkpoint = raw_checkpoint_dir(cfg, method, backbone, target_reduction)
    output = result_dir(cfg, method, backbone, target_reduction, objective, seed)
    if (output / ".complete").is_file():
        barrier()
        return output
    if is_main():
        output.mkdir(parents=True, exist_ok=True)
    barrier()

    backbone_cfg = cfg["backbones"][backbone]
    recovery = cfg["recovery"]
    train_rows = load_recovery_rows(backbone_cfg["recovery_train"])
    validation_rows = load_recovery_rows(backbone_cfg["recovery_validation"])
    validation_rows = validation_rows[:int(recovery["validation_maximum"])]
    model, tokenizer, _ = load_compressed_checkpoint(checkpoint, dtype=torch.bfloat16)
    model.config.use_cache = False
    if bool(recovery["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    injected = freeze_and_inject_lora(model, rank=int(recovery["rank"]), alpha=float(recovery["alpha"]),
                                      dropout=float(recovery["dropout"]), targets=recovery["target_modules"])
    trainable = trainable_parameter_report(model)
    model.to(device)

    teacher = None
    teacher_loaded = False
    if objective == "ce_kd":
        teacher = AutoModelForCausalLM.from_pretrained(backbone_cfg["teacher_path"], torch_dtype=torch.bfloat16,
                                                       local_files_only=True, attn_implementation="sdpa").to(device)
        teacher.eval()
        teacher.config.use_cache = False
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if int(teacher.config.vocab_size) != int(model.config.vocab_size):
            raise RuntimeError("teacher and student vocabulary sizes differ")
        teacher_loaded = True
    elif objective == "ce":
        # This branch intentionally does not stat, tokenize with, or load the teacher path.
        teacher = None

    training_model = model
    if world > 1:
        training_model = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            broadcast_buffers=False, find_unused_parameters=False,
        )
    parameters = [parameter for parameter in training_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(backbone_cfg["learning_rate"]),
                                  weight_decay=float(recovery["weight_decay"]))
    total_steps = int(backbone_cfg["max_steps"])
    accumulation = int(backbone_cfg["gradient_accumulation"])
    per_gpu_batch = int(backbone_cfg["per_gpu_batch"])
    scheduler = _cosine_scheduler(optimizer, total_steps, int(backbone_cfg["warmup_steps"]),
                                  float(recovery["minimum_lr_ratio"]))
    total_examples = total_steps * accumulation * world * per_gpu_batch
    order = deterministic_indices(len(train_rows), 0, total_examples, seed)
    history = []
    best_value = float("inf")
    best_step = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, total_steps + 1):
        step_ce = step_kd = step_total = 0.0
        for micro in range(accumulation):
            global_micro = (step - 1) * accumulation + micro
            global_start = global_micro * world * per_gpu_batch + rank * per_gpu_batch
            indices = order[global_start:global_start + per_gpu_batch]
            rows = [train_rows[index] for index in indices]
            encoded = encode_multiple_choice(rows, tokenizer, int(recovery["max_length"]), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, ce, kd, _ = controlled_loss(
                    training_model, teacher, rows, encoded, objective=objective,
                    temperature=float(recovery["temperature"]), lambda_ce=float(recovery["lambda_ce"]),
                    lambda_kd=float(recovery["lambda_kd"]), eos_token_id=tokenizer.eos_token_id,
                    exclude_eos=bool(recovery["loss_exclude_eos"]),
                )
                (loss / accumulation).backward()
            step_total += float(loss.detach()) / accumulation
            step_ce += float(ce.detach()) / accumulation
            step_kd += float(kd.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(parameters, float(recovery["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % int(recovery["validation_interval"]) == 0 or step == total_steps:
            metrics = validation_decision_ce(model, tokenizer, validation_rows,
                                             max_length=int(recovery["max_length"]),
                                             batch_size=int(backbone_cfg["eval_batch"]),
                                             rank=rank, world=world, device=device)
            entry = {"step": step, "training_total": step_total, "training_decision_ce": step_ce,
                     "training_kd": step_kd, "learning_rate": scheduler.get_last_lr()[0],
                     "validation_decision_ce": metrics["decision_ce"],
                     "validation_accuracy": metrics["accuracy"], "validation_examples": metrics["examples"]}
            history.append(entry)
            improved = metrics["decision_ce"] < best_value - float(recovery["minimum_improvement"])
            if improved:
                best_value, best_step = metrics["decision_ce"], step
                if is_main():
                    _save_adapter(model, output / "adapter_best_val", recovery, injected)
            training_model.train()
    if is_main():
        _save_adapter(model, output / "adapter_final", recovery, injected)
        _atomic_json(output / "validation_history.json", history)
    barrier()
    if best_step == 0:
        # best_step is deterministic across ranks because validation metrics are all-reduced.
        raise RuntimeError("validation never selected an adapter")
    load_adapter_state(model, output / "adapter_best_val/adapter_model.pt")
    model.to(dtype=torch.float16)
    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()
    metrics = evaluate_seven_tasks(model, tokenizer, cfg, backbone, output, rank, world, device)

    if is_main():
        compression = json.loads((checkpoint / "compression_report.json").read_text())
        recovered_total = unique_parameter_count(model)
        recovered_reduction = 1.0 - recovered_total / int(compression["dense_parameters"])
        training_report = {
            "method": method, "backbone": backbone, "target_reduction": float(target_reduction),
            "objective": objective, "seed": seed, "teacher_loaded": teacher_loaded,
            "teacher_forward_executed": objective == "ce_kd", "teacher_frozen": objective == "ce_kd",
            "teacher_free_ce": objective == "ce", "distill_mode": objective,
            "lambda_ce": float(recovery["lambda_ce"]),
            "lambda_kd": float(recovery["lambda_kd"]) if objective == "ce_kd" else 0.0,
            "temperature": float(recovery["temperature"]) if objective == "ce_kd" else None,
            "kd_scope": "all_non_padding_shifted_tokens_full_vocabulary" if objective == "ce_kd" else None,
            "loss_scope": "multiple_choice_candidate_decision_ce", "loss_exclude_eos": True,
            "steps": total_steps, "world_size": world, "per_gpu_batch": per_gpu_batch,
            "gradient_accumulation": accumulation,
            "effective_global_batch": world * per_gpu_batch * accumulation,
            "best_step": best_step, "best_validation_decision_ce": best_value,
            "selection_metric": "minimum validation decision CE", "validation_test_isolation": True,
            "elapsed_seconds": time.time() - started, **trainable,
        }
        optimizer_config = {
            "optimizer": "AdamW", "learning_rate": float(backbone_cfg["learning_rate"]),
            "weight_decay": float(recovery["weight_decay"]), "schedule": "warmup_cosine",
            "warmup_steps": int(backbone_cfg["warmup_steps"]),
            "minimum_lr_ratio": float(recovery["minimum_lr_ratio"]),
            "gradient_clip_norm": float(recovery["gradient_clip_norm"]),
        }
        run_config = {
            "config_path": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
            "method": method, "backbone": backbone, "target_reduction": float(target_reduction),
            "objective": objective, "seed": seed, "raw_checkpoint": str(checkpoint),
            "recovery_train": backbone_cfg["recovery_train"],
            "recovery_train_sha256": sha256_file(backbone_cfg["recovery_train"]),
            "recovery_train_sample_ids_sha256": sample_id_hash(train_rows),
            "recovery_validation": backbone_cfg["recovery_validation"],
            "recovery_validation_sha256": sha256_file(backbone_cfg["recovery_validation"]),
            "recovery_validation_sample_ids_sha256": sample_id_hash(validation_rows),
            "teacher": backbone_cfg["teacher_path"] if objective == "ce_kd" else None,
            "teacher_loaded": teacher_loaded, "max_length": int(recovery["max_length"]),
            "training_prompt_mode": "decision_aligned", "sample_order_seed": seed,
        }
        shutil.copy2(checkpoint / "compression_report.json", output / "compression_report.json")
        _atomic_json(output / "trainable_parameter_report.json", trainable)
        _atomic_json(output / "training_report.json", training_report)
        _atomic_json(output / "optimizer_config.json", optimizer_config)
        _atomic_json(output / "run_config.json", run_config)
        with (output / "training_log.jsonl").open("w") as handle:
            for row in history:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        deploy = output / "deploy_bundle"
        deploy.mkdir(exist_ok=True)
        compressed_deploy = deploy / "compressed_model"
        if compressed_deploy.exists():
            shutil.rmtree(compressed_deploy)
        shutil.copytree(checkpoint, compressed_deploy, copy_function=os.link)
        shutil.copy2(output / "adapter_best_val/adapter_model.pt", deploy / "adapter_model.pt")
        shutil.copy2(output / "adapter_best_val/adapter_config.json", deploy / "adapter_config.json")
        _atomic_json(deploy / "bundle.json", {
            "compressed_checkpoint": "compressed_model", "adapter": "adapter_model.pt",
            "loader": "icassp27.controlled_baselines.modeling.load_compressed_checkpoint",
            "standalone_components": ["compressed_model", "adapter_model.pt"],
            "uses_hardlinks_for_space_efficient_standalone_bundle": True,
        })
        (deploy / "model_loader.py").write_text(
            "from pathlib import Path\n"
            "import json\n"
            "from icassp27.controlled_baselines.modeling import (freeze_and_inject_lora, "
            "load_adapter_state, load_compressed_checkpoint)\n"
            "def load(path, dtype=None):\n"
            "    root = Path(path)\n"
            "    model, tokenizer, structure = load_compressed_checkpoint(root / 'compressed_model', dtype=dtype)\n"
            "    cfg = json.loads((root / 'adapter_config.json').read_text())\n"
            "    freeze_and_inject_lora(model, rank=cfg['rank'], alpha=cfg['alpha'], "
            "dropout=cfg['dropout'], targets=cfg['target_modules'])\n"
            "    load_adapter_state(model, root / 'adapter_model.pt')\n"
            "    return model, tokenizer, structure\n"
        )
        _atomic_json(output / "deploy_load_report.json", {
            "bundle": str(deploy), "loader": str(deploy / "model_loader.py"),
            "compressed_checkpoint_load_pass": True, "adapter_load_pass": True,
            "standalone_bytes": sum(path.stat().st_size for path in deploy.rglob("*") if path.is_file()),
        })
        summary = _base_summary(cfg, method, backbone, target_reduction, objective, seed, checkpoint)
        summary.update({
            "trainable_parameters": trainable["trainable_parameters"],
            "adapter_parameters": trainable["adapter_parameters"],
            "recovered_total_parameters": recovered_total, "recovered_reduction": recovered_reduction,
            "best_step": best_step, "best_validation_decision_ce": best_value,
            **_summary_metrics(metrics, cfg["evaluation"]["tasks"]), "macro": metrics["macro"],
        })
        _atomic_json(output / "summary.json", summary)
        (output / ".complete").write_text("PASS\n")
    barrier()
    return output
