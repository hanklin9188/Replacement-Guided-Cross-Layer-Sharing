#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Sequence

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import information_energy_sufficiency_exit_eval_final_llama as ies
import newthesis_pipeline_final_llama as pipe


def _parse_csv_list(value: str) -> List[str]:
    out: List[str] = []
    for chunk in str(value or "").replace(",", " ").split():
        item = chunk.strip()
        if item:
            out.append(item)
    return out


def _parse_float_list(value: str) -> List[float]:
    vals: List[float] = []
    seen = set()
    for item in _parse_csv_list(value):
        try:
            fval = float(item)
        except Exception:
            continue
        key = round(fval, 6)
        if key not in seen:
            vals.append(fval)
            seen.add(key)
    return vals


def _parse_int_list(value: str) -> List[int]:
    vals: List[int] = []
    seen = set()
    for item in _parse_csv_list(value):
        try:
            ivalue = int(item)
        except Exception:
            continue
        if ivalue not in seen:
            vals.append(ivalue)
            seen.add(ivalue)
    return vals


def _safe_name(prefix: str, threshold: float) -> str:
    return f"{prefix}_{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _new_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "early_count": 0,
        "exit_layer_sum": 0,
        "compute_ratio_sum": 0.0,
        "exit_layer_hist": Counter(),
    }


def _update(stats: Dict[str, Any], *, pred: str, answer: str, layer: int, final_layer: int) -> None:
    stats["total"] += 1
    stats["correct"] += int(str(pred).strip() == str(answer).strip())
    stats["early_count"] += int(int(layer) < int(final_layer))
    stats["exit_layer_sum"] += int(layer)
    stats["compute_ratio_sum"] += float(layer) / float(max(1, final_layer))
    stats["exit_layer_hist"][str(layer)] += 1


def _finalize(stats: Dict[str, Any]) -> Dict[str, Any]:
    total = max(1, int(stats["total"]))
    return {
        "samples": int(stats["total"]),
        "correct": int(stats["correct"]),
        "accuracy": float(stats["correct"] / total),
        "early_exit_rate": float(stats["early_count"] / total),
        "avg_exit_layer": float(stats["exit_layer_sum"] / total),
        "avg_compute_ratio": float(stats["compute_ratio_sum"] / total),
        "estimated_layer_savings": float(1.0 - stats["compute_ratio_sum"] / total),
        "exit_layer_hist": dict(sorted(stats["exit_layer_hist"].items(), key=lambda kv: int(kv[0]))),
    }


def _choose_confidence_exit(
    *,
    layer_metrics: Dict[int, Dict[str, Any]],
    exit_layers: Sequence[int],
    final_layer: int,
    threshold: float,
) -> tuple[int, Dict[str, Any]]:
    for layer in exit_layers:
        metrics = layer_metrics.get(int(layer))
        if metrics and float(metrics["confidence"]) >= float(threshold):
            return int(layer), dict(metrics)
    return int(final_layer), dict(layer_metrics[int(final_layer)])


@torch.no_grad()
def evaluate_dataset(args: argparse.Namespace, *, model, tokenizer) -> Dict[str, Any]:
    dataset = str(args.dataset)
    candidates = pipe._candidates_for_dataset(dataset)
    data_file = os.path.join(str(args.test_data_root), dataset, "test.json")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(data_file)
    with open(data_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{data_file} must contain a JSON list.")
    records = payload[: int(args.max_samples)] if int(args.max_samples) > 0 else payload
    thresholds = _parse_float_list(str(args.thresholds))
    if not thresholds:
        raise ValueError("--thresholds produced no valid numeric values.")
    exit_layers_requested = _parse_int_list(str(args.exit_layers))
    device = next(model.parameters()).device

    method_names = ["full"] + [_safe_name("conf", tau) for tau in thresholds]
    method_thresholds = {"full": None}
    method_thresholds.update({_safe_name("conf", tau): float(tau) for tau in thresholds})
    stats = {name: _new_stats() for name in method_names}
    records_path = ""
    records_handle = None
    if bool(args.save_records):
        records_path = os.path.join(str(args.output_dir), f"{dataset}_confidence_sweep_records.jsonl")
        records_handle = open(records_path, "w", encoding="utf-8")

    t0 = time.perf_counter()
    iterator = pipe.iter_progress(records, total=len(records), desc=f"[ConfSweep] {dataset}")
    for sample_id, item in enumerate(iterator):
        prompt = pipe._build_prompt(item, dataset)
        answer = ies._normalize_answer(item.get("answer", item.get("label", "")), candidates)
        scores_by_layer, _, _, final_layer = ies.score_candidate_layers(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidates=candidates,
            exit_layers=exit_layers_requested,
            device=device,
            length_norm=str(args.length_norm),
        )
        exit_layers = [x for x in exit_layers_requested if 0 < int(x) < int(final_layer)]
        layer_metrics: Dict[int, Dict[str, Any]] = {}
        for layer_id in sorted(set(exit_layers + [final_layer])):
            if int(layer_id) not in scores_by_layer:
                continue
            metrics = ies._answer_metrics(candidates, scores_by_layer[int(layer_id)], float(args.temperature))
            metrics["layer"] = int(layer_id)
            layer_metrics[int(layer_id)] = metrics
        if final_layer not in layer_metrics:
            raise RuntimeError(f"final layer {final_layer} metrics missing.")

        decisions: Dict[str, Dict[str, Any]] = {}
        full_metrics = dict(layer_metrics[int(final_layer)])
        _update(stats["full"], pred=str(full_metrics["pred"]), answer=answer, layer=final_layer, final_layer=final_layer)
        decisions["full"] = {
            "pred": str(full_metrics["pred"]),
            "exit_layer": int(final_layer),
            "correct": bool(str(full_metrics["pred"]) == str(answer)),
            "confidence": float(full_metrics["confidence"]),
        }
        for tau in thresholds:
            name = _safe_name("conf", tau)
            layer, chosen = _choose_confidence_exit(
                layer_metrics=layer_metrics,
                exit_layers=exit_layers,
                final_layer=final_layer,
                threshold=float(tau),
            )
            pred = str(chosen["pred"])
            _update(stats[name], pred=pred, answer=answer, layer=layer, final_layer=final_layer)
            decisions[name] = {
                "pred": pred,
                "exit_layer": int(layer),
                "correct": bool(pred == answer),
                "confidence": float(chosen["confidence"]),
                "threshold": float(tau),
            }

        if records_handle is not None:
            row = {
                "sample_id": int(sample_id),
                "answer": answer,
                "decisions": decisions,
                "layer_confidence": {
                    str(layer): {
                        "pred": str(metrics["pred"]),
                        "confidence": float(metrics["confidence"]),
                        "answer_energy": float(metrics.get("answer_energy", -math.log(max(1e-12, float(metrics["confidence"]))))),
                        "entropy": float(metrics["entropy"]),
                        "energy_margin": float(metrics["energy_margin"]),
                        "scores": {str(k): float(v) for k, v in metrics.get("scores", {}).items()},
                        "probs": {str(k): float(v) for k, v in metrics.get("probs", {}).items()},
                    }
                    for layer, metrics in layer_metrics.items()
                },
            }
            records_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if records_handle is not None:
        records_handle.close()

    elapsed = time.perf_counter() - t0
    summary = {name: _finalize(stats[name]) for name in method_names}
    report = {
        "dataset": dataset,
        "candidates": list(candidates),
        "samples": int(len(records)),
        "elapsed_sec": float(elapsed),
        "deploy_bundle": str(args.deploy_bundle),
        "exit_layers_requested": exit_layers_requested,
        "thresholds": thresholds,
        "methods": summary,
        "method_thresholds": method_thresholds,
        "records_jsonl": records_path,
    }
    out_json = os.path.join(str(args.output_dir), f"{dataset}_confidence_sweep_report.json")
    pipe.save_json(out_json, report)
    print(f"[ConfSweep] saved {out_json}", flush=True)
    for name in method_names:
        metrics = summary[name]
        tau_text = "" if name == "full" else f" tau={method_thresholds[name]:.4f}"
        print(
            f"[ConfSweep] {dataset}|{name}{tau_text} acc={metrics['accuracy']:.4f} "
            f"early={metrics['early_exit_rate']:.4f} avg_layer={metrics['avg_exit_layer']:.2f} "
            f"savings={metrics['estimated_layer_savings']:.4f}",
            flush=True,
        )
    return report


def _write_summary_csv(output_dir: str, reports: Sequence[Dict[str, Any]]) -> str:
    path = os.path.join(output_dir, "confidence_threshold_sweep_summary.csv")
    rows: List[Dict[str, Any]] = []
    for report in reports:
        dataset = str(report["dataset"])
        method_thresholds = report.get("method_thresholds", {})
        for method, metrics in report["methods"].items():
            rows.append({
                "dataset": dataset,
                "method": method,
                "threshold": "" if method == "full" else method_thresholds.get(method, ""),
                "samples": metrics["samples"],
                "accuracy": metrics["accuracy"],
                "early_exit_rate": metrics["early_exit_rate"],
                "avg_exit_layer": metrics["avg_exit_layer"],
                "avg_compute_ratio": metrics["avg_compute_ratio"],
                "estimated_layer_savings": metrics["estimated_layer_savings"],
                "exit_layer_hist": json.dumps(metrics["exit_layer_hist"], sort_keys=True),
            })
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "threshold",
                "samples",
                "accuracy",
                "early_exit_rate",
                "avg_exit_layer",
                "avg_compute_ratio",
                "estimated_layer_savings",
                "exit_layer_hist",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_aggregate_csv(output_dir: str, reports: Sequence[Dict[str, Any]]) -> str:
    path = os.path.join(output_dir, "confidence_threshold_sweep_aggregate.csv")
    method_names: List[str] = []
    method_thresholds: Dict[str, Any] = {}
    for report in reports:
        for method in report["methods"].keys():
            if method not in method_names:
                method_names.append(method)
        method_thresholds.update(report.get("method_thresholds", {}))
    rows: List[Dict[str, Any]] = []
    for method in method_names:
        parts = []
        for report in reports:
            if method in report["methods"]:
                item = dict(report["methods"][method])
                item["dataset"] = report["dataset"]
                parts.append(item)
        if not parts:
            continue
        samples = [int(x["samples"]) for x in parts]
        total_samples = sum(samples)
        accs = [float(x["accuracy"]) for x in parts]
        rows.append({
            "method": method,
            "threshold": "" if method == "full" else method_thresholds.get(method, ""),
            "num_tasks": len(parts),
            "samples": total_samples,
            "macro_accuracy": sum(accs) / len(accs),
            "weighted_accuracy": sum(float(x["accuracy"]) * int(x["samples"]) for x in parts) / max(1, total_samples),
            "weighted_early_exit_rate": sum(float(x["early_exit_rate"]) * int(x["samples"]) for x in parts) / max(1, total_samples),
            "weighted_avg_exit_layer": sum(float(x["avg_exit_layer"]) * int(x["samples"]) for x in parts) / max(1, total_samples),
            "weighted_layer_savings": sum(float(x["estimated_layer_savings"]) * int(x["samples"]) for x in parts) / max(1, total_samples),
        })
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "threshold",
                "num_tasks",
                "samples",
                "macro_accuracy",
                "weighted_accuracy",
                "weighted_early_exit_rate",
                "weighted_avg_exit_layer",
                "weighted_layer_savings",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Confidence threshold sweep for intermediate-layer early exit.")
    parser.add_argument("--deploy_bundle", type=str, required=True)
    parser.add_argument("--test_data_root", type=str, default=os.path.join(pipe.WORKSPACE_ROOT, "data", "datasets"))
    parser.add_argument("--datasets", type=str, default="piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--exit_layers", type=str, default="12,16,20,24")
    parser.add_argument("--thresholds", type=str, default="0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.995")
    parser.add_argument("--length_norm", choices=["none", "avg"], default="none")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use_quant_bank_int4", type=pipe.str2bool, default=False)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", type=pipe.str2bool, default=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default="")
    parser.add_argument("--save_records", type=pipe.str2bool, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipe.set_seed(int(args.seed))
    os.makedirs(str(args.output_dir), exist_ok=True)
    device = pipe.resolve_device(str(args.device), -1)
    dtype = pipe.get_target_dtype(device)
    bundle = torch.load(str(args.deploy_bundle), map_location="cpu")
    if not isinstance(bundle, dict):
        raise ValueError("--deploy_bundle must point to a deploy_bundle.pt dict.")
    model, quant_report = pipe._build_shared_model_for_eval(
        base_model=str(bundle["base_model"]),
        atlas_payload=bundle.get("atlas", {}),
        shared_payload=bundle["shared_student"],
        quant_bank_int4=bundle.get("quant_bank_int4"),
        use_quant_bank_int4=bool(args.use_quant_bank_int4),
        device=device,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    shared_meta = bundle.get("shared_student", {}).get("meta", {}) if isinstance(bundle.get("shared_student"), dict) else {}
    tokenizer_path = str(args.tokenizer_name_or_path).strip() or str(shared_meta.get("teacher_model", "")) or str(bundle.get("base_model", ""))
    tokenizer = pipe.load_tokenizer(tokenizer_path, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets = [str(args.dataset).strip()] if str(args.dataset).strip() else _parse_csv_list(str(args.datasets))
    print(f"[ConfSweep] deploy_bundle={args.deploy_bundle}", flush=True)
    print(f"[ConfSweep] quant={quant_report}", flush=True)
    print(f"[ConfSweep] datasets={' '.join(datasets)}", flush=True)
    print(f"[ConfSweep] exit_layers={args.exit_layers} thresholds={args.thresholds}", flush=True)

    reports: List[Dict[str, Any]] = []
    for dataset in datasets:
        args.dataset = dataset
        reports.append(evaluate_dataset(args, model=model, tokenizer=tokenizer))
    summary_csv = _write_summary_csv(str(args.output_dir), reports)
    aggregate_csv = _write_aggregate_csv(str(args.output_dir), reports)
    all_report = {
        "deploy_bundle": str(args.deploy_bundle),
        "output_dir": str(args.output_dir),
        "summary_csv": summary_csv,
        "aggregate_csv": aggregate_csv,
        "datasets": [report["dataset"] for report in reports],
        "reports": reports,
    }
    pipe.save_json(os.path.join(str(args.output_dir), "confidence_threshold_sweep_all_report.json"), all_report)
    print(f"[ConfSweep] summary_csv={summary_csv}", flush=True)
    print(f"[ConfSweep] aggregate_csv={aggregate_csv}", flush=True)


if __name__ == "__main__":
    main()
