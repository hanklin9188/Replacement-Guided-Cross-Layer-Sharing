#!/usr/bin/env python3
"""Deterministic RTN quantization and seven-task evaluation for compressed baselines."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch

from icassp27.controlled_baselines.data import encode_multiple_choice, load_evaluation_rows
from icassp27.controlled_baselines.distributed import (
    all_gather_objects, barrier, close, initialize, is_main, seed_everything,
)
from icassp27.controlled_baselines.method_recovery import load_method_config
from icassp27.controlled_baselines.modeling import load_compressed_checkpoint
from icassp27.controlled_baselines.objectives import candidate_scores


TASK_DISPLAY = {
    "arc_challenge": "ARC_C", "arc_easy": "ARC_E", "hellaswag": "HellaSwag",
    "openbookqa": "OpenBookQA", "piqa": "PIQA", "social_i_qa": "SocialIQA",
    "winogrande": "WinoGrande",
}
FIELDS = [
    "method", "model_id", "regime", "seed", "precision", "task", "example_id",
    "source_index", "gold_answer", "prediction", "correct", "candidate_scores",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{name: row[name] for name in FIELDS} for row in rows])
    temporary.replace(path)


def quantization_target(name: str, parameter: torch.Tensor) -> bool:
    """All decoder matrix weights; embeddings/head/norms and non-matrices stay BF16."""
    return parameter.ndim == 2 and name.startswith("model.layers.") and name.endswith(".weight")


def alias_plan(model: torch.nn.Module) -> tuple[dict[int, str], dict[str, str]]:
    pointer_to_canonical: dict[int, str] = {}
    aliases: dict[str, str] = {}
    for name, value in model.state_dict().items():
        pointer = value.untyped_storage().data_ptr()
        if pointer in pointer_to_canonical:
            aliases[name] = pointer_to_canonical[pointer]
        else:
            pointer_to_canonical[pointer] = name
    return pointer_to_canonical, aliases


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.int8 or values.shape[-1] % 2:
        raise ValueError("INT4 packing requires signed int8 values and an even last dimension")
    unsigned = values.to(torch.int16).bitwise_and(0x0F)
    packed = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
    return packed.to(torch.uint8)


def unpack_int4(values: torch.Tensor) -> torch.Tensor:
    low = (values.to(torch.int16) & 0x0F)
    high = ((values.to(torch.int16) >> 4) & 0x0F)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    result = torch.empty((*values.shape[:-1], values.shape[-1] * 2), device=values.device, dtype=torch.int8)
    result[..., 0::2] = low.to(torch.int8)
    result[..., 1::2] = high.to(torch.int8)
    return result


def quantize_parameter(weight: torch.Tensor, precision: str, *, capture_packed: bool = True) -> tuple[torch.Tensor, dict | None]:
    source = weight.detach().float()
    if precision == "W8A16":
        maxima = source.abs().amax(dim=1, keepdim=True)
        scale = (maxima / 127.0).to(torch.float16)
        scale = torch.where(maxima.eq(0), torch.ones_like(scale), scale)
        if bool(scale.eq(0).any()):
            raise RuntimeError("FP16 W8 scale underflowed to zero")
        qweight = torch.round(source / scale.float()).clamp(-127, 127).to(torch.int8)
        dequantized = qweight.float() * scale.float()
        packed = {
            "encoding": "int8_symmetric_per_output_channel",
            "qweight": qweight.cpu(), "scale": scale.cpu(), "shape": list(weight.shape),
        } if capture_packed else None
    elif precision == "W4A16":
        group_size = 128
        rows, columns = source.shape
        padded_columns = math.ceil(columns / group_size) * group_size
        if padded_columns != columns:
            source = torch.nn.functional.pad(source, (0, padded_columns - columns))
        grouped = source.reshape(rows, padded_columns // group_size, group_size)
        maxima = grouped.abs().amax(dim=-1, keepdim=True)
        scale = (maxima / 7.0).to(torch.float16)
        scale = torch.where(maxima.eq(0), torch.ones_like(scale), scale)
        if bool(scale.eq(0).any()):
            raise RuntimeError("FP16 W4 scale underflowed to zero")
        qweight = torch.round(grouped / scale.float()).clamp(-7, 7).to(torch.int8)
        packed_weight = pack_int4(qweight.reshape(rows, padded_columns))
        unpacked = unpack_int4(packed_weight).reshape_as(qweight)
        if not torch.equal(unpacked, qweight):
            raise RuntimeError("INT4 pack/unpack round trip failed")
        dequantized = (unpacked.float() * scale.float()).reshape(rows, padded_columns)[:, :columns]
        packed = {
            "encoding": "int4_symmetric_grouped",
            "qweight_packed": packed_weight.cpu(), "scale": scale.squeeze(-1).cpu(),
            "shape": list(weight.shape), "group_size": group_size,
            "padded_input_features": padded_columns,
        } if capture_packed else None
    else:
        raise ValueError(precision)
    return dequantized.to(torch.bfloat16), packed


def prepare_artifact_and_model(model: torch.nn.Module, precision: str, artifact: Path,
                               method: str, checkpoint: Path, device: torch.device, rank: int) -> dict:
    pointer_to_canonical, aliases = alias_plan(model)
    parameters = list(model.named_parameters())
    target_parameters = [(name, parameter) for name, parameter in parameters if quantization_target(name, parameter)]
    target_pointers = {parameter.untyped_storage().data_ptr() for _, parameter in target_parameters}
    target_count = sum(parameter.numel() for _, parameter in target_parameters)
    total_count = sum(parameter.numel() for _, parameter in parameters)

    if precision == "BF16":
        if rank == 0 and not artifact.is_file():
            atomic_torch_save(artifact, model.state_dict())
        model.to(device)
        return {"target_tensor_count": 0, "target_parameter_count": 0,
                "total_unique_parameter_count": total_count, "aliases": len(aliases)}

    capture_artifact = rank == 0 and not artifact.is_file()
    preserved = {}
    if capture_artifact:
        for name, value in model.state_dict().items():
            pointer = value.untyped_storage().data_ptr()
            if pointer_to_canonical[pointer] != name or pointer in target_pointers:
                continue
            preserved[name] = {"encoding": "bf16", "data": value.detach().to(torch.bfloat16).cpu().clone(),
                               "shape": list(value.shape)}
    model.to(device)
    packed_tensors = {} if capture_artifact else None
    quantized_names = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not quantization_target(name, parameter):
                continue
            canonical = pointer_to_canonical.get(parameter.untyped_storage().data_ptr(), name)
            # Moving a module changes storage pointers; canonical parameter names remain stable.
            canonical = name if name in pointer_to_canonical.values() else canonical
            dequantized, packed = quantize_parameter(parameter, precision, capture_packed=capture_artifact)
            parameter.copy_(dequantized)
            quantized_names.append(name)
            if capture_artifact:
                packed_tensors[name] = packed
    if capture_artifact:
        payload = {
            "format_version": 1,
            "artifact_type": "weight_only_quantized_compressed_model",
            "method": method,
            "source_checkpoint": str(checkpoint.resolve()),
            "precision": precision,
            "activation_precision": "BF16",
            "scale_dtype": "FP16",
            "target_policy": "all decoder 2-D weight matrices; embeddings, LM head, norms, non-matrices BF16",
            "canonical_tensors": {**preserved, **packed_tensors},
            "aliases": aliases,
        }
        atomic_torch_save(artifact, payload)
    del preserved, packed_tensors
    gc.collect()
    return {"target_tensor_count": len(quantized_names), "target_parameter_count": target_count,
            "total_unique_parameter_count": total_count, "aliases": len(aliases)}


def selected_rows(rows: list[dict], task_index: int, total: int | None) -> list[dict]:
    if total is None:
        return rows
    base, remainder = divmod(total, len(TASK_DISPLAY))
    count = base + int(task_index < remainder)
    return rows[:count]


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    cfg = load_method_config(args.config)
    rank, world, device = initialize()
    seed_everything(args.seed)
    complete = args.output / ".complete"
    if complete.is_file() and not args.overwrite:
        if is_main():
            print(json.dumps({"status": "SKIP_COMPLETE", "output": str(args.output)}))
        barrier(); close(); return

    report = json.loads((args.checkpoint / "compression_report.json").read_text())
    expected = {"Basis Sharing": "basis_sharing", "SVD-LLM": "svd_llm"}[args.method]
    if report["method"] != expected:
        raise ValueError(f"checkpoint method mismatch: {report['method']} != {expected}")
    model, tokenizer, _ = load_compressed_checkpoint(args.checkpoint, dtype=torch.bfloat16)
    quantization_report = prepare_artifact_and_model(
        model, args.precision, args.artifact, args.method, args.checkpoint, device, rank,
    )
    barrier()
    model.eval()

    backbone = dict(cfg["backbones"]["llama31_8b"])
    if args.evaluation_data_root is not None:
        backbone["evaluation_data_root"] = str(args.evaluation_data_root)
    evaluation = cfg["evaluation"]
    local_predictions: list[dict] = []
    source_files = []
    global_order = 0
    for task_index, internal_task in enumerate(evaluation["tasks"]):
        source = Path(backbone["evaluation_data_root"]) / f"{internal_task}.{evaluation['split']}.jsonl"
        source_files.append({"task": internal_task, "path": str(source), "sha256": sha256(source)})
        rows = load_evaluation_rows(backbone["evaluation_data_root"], [internal_task], evaluation["split"])
        rows = selected_rows(rows, task_index, args.max_total_examples)
        indexed = [(global_order + index, row) for index, row in enumerate(rows)]
        global_order += len(rows)
        local = indexed[rank::world]
        for offset in range(0, len(local), int(backbone["eval_batch"])):
            pairs = local[offset:offset + int(backbone["eval_batch"])]
            batch = [row for _, row in pairs]
            encoded = encode_multiple_choice(batch, tokenizer, int(cfg["recovery"]["max_length"]), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                               use_cache=False).logits
            scores = candidate_scores(logits, encoded)
            for (order, row), (start, end) in zip(pairs, encoded["row_spans"]):
                row_scores = [float(value) for value in scores[start:end].cpu()]
                prediction = max(range(len(row_scores)), key=row_scores.__getitem__)
                local_predictions.append({
                    "_order": order, "method": args.method, "model_id": args.model_id,
                    "regime": "CE+KD", "seed": args.seed, "precision": args.precision,
                    "task": TASK_DISPLAY[internal_task], "example_id": row["id"],
                    "source_index": int(row["source_index"]), "gold_answer": int(row["label"]),
                    "prediction": int(prediction), "correct": int(prediction == int(row["label"])),
                    "candidate_scores": json.dumps(row_scores, separators=(",", ":")),
                })
    gathered = all_gather_objects(local_predictions)
    if is_main():
        merged = sorted((row for part in gathered for row in part), key=lambda row: row["_order"])
        if len({(row["task"], row["example_id"]) for row in merged}) != len(merged):
            raise RuntimeError("duplicate task/example_id")
        by_task = defaultdict(list)
        for row in merged:
            by_task[row["task"]].append(row)
        if set(by_task) != set(TASK_DISPLAY.values()):
            raise RuntimeError(f"incomplete task coverage: {sorted(by_task)}")
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / "predictions.csv", merged)
        score_rows = []
        for row in merged:
            item = {name: row[name] for name in FIELDS}
            item["candidate_scores"] = json.loads(item["candidate_scores"])
            score_rows.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        atomic_text(args.output / "predictions_with_scores.jsonl", "\n".join(score_rows) + "\n")
        task_accuracy = {task: sum(x["correct"] for x in rows) / len(rows) for task, rows in sorted(by_task.items())}
        metadata = {
            "status": "PASS", "training_invoked": False, "evaluation_only": True,
            "method": args.method, "model_id": args.model_id, "regime": "CE+KD", "seed": args.seed,
            "source_checkpoint": str(args.checkpoint.resolve()), "precision": args.precision,
            "artifact": str(args.artifact.resolve()), "artifact_bytes": args.artifact.stat().st_size,
            "artifact_sha256": sha256(args.artifact), "quantization": quantization_report,
            "protocol": {"mode": "logprob", "length_norm": "none", "shots": 0,
                         "compute_dtype": "bfloat16", "split": evaluation["split"],
                         "max_length": int(cfg["recovery"]["max_length"]),
                         "weight_quantization": args.precision,
                         "inference_kernel": "BF16 after deterministic dequantization" if args.precision != "BF16" else "BF16"},
            "world_size": world, "rows": len(merged), "smoke": args.max_total_examples is not None,
            "task_counts": {task: len(rows) for task, rows in sorted(by_task.items())},
            "task_accuracy": task_accuracy,
            "macro_accuracy": sum(task_accuracy.values()) / len(task_accuracy),
            "source_files": source_files,
        }
        atomic_text(args.output / "validation.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        atomic_text(complete, "PASS\n")
        print(json.dumps({"status": "PASS", "output": str(args.output), "rows": len(merged),
                          "macro_accuracy": metadata["macro_accuracy"]}, sort_keys=True))
    barrier(); close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=["Basis Sharing", "SVD-LLM"], required=True)
    parser.add_argument("--model-id", choices=["8b_15", "8b_25"], required=True)
    parser.add_argument("--precision", choices=["BF16", "W8A16", "W4A16"], required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-data-root", type=Path)
    parser.add_argument("--max-total-examples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    evaluate(args)


if __name__ == "__main__":
    main()
