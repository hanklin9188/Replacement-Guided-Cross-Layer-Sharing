#!/usr/bin/env python3
"""Aggregate accuracy, paired bootstrap, and actual packed-byte results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {"basis_sharing": "Basis Sharing", "svd_llm": "SVD-LLM"}
PRECISIONS = ["BF16", "W8A16", "W4A16"]
DISPLAY_PRECISION = {"BF16": "BF16", "W8A16": "INT8", "W4A16": "INT4"}
TASKS = ["ARC_C", "ARC_E", "HellaSwag", "OpenBookQA", "PIQA", "SocialIQA", "WinoGrande"]
DENSE_8B_BYTES = 16_060_643_284
DENSE_8B_SHA256 = "93c3d08914a1b81856ac94fb07138bba7ecbd5d9946d9704456d995004fcfdd5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_difference(diff: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.asarray([(diff == -1).sum(), (diff == 0).sum(), (diff == 1).sum()])
    draws = rng.multinomial(len(diff), counts / len(diff), size=n_bootstrap)
    return (draws[:, 2] - draws[:, 0]) / len(diff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=44)
    parser.add_argument("--model-id", choices=["8b_15", "8b_25"], default="8b_15")
    args = parser.parse_args()
    expected_rows = 64 if args.mode == "smoke" else 19_149
    all_accuracy = []
    all_bootstrap = []
    byte_rows = []
    validation = {"status": "PASS", "mode": args.mode, "methods": {},
                  "n_bootstrap": args.n_bootstrap, "bootstrap_seed": args.bootstrap_seed}
    rng = np.random.default_rng(args.bootstrap_seed)

    for slug, method in METHODS.items():
        precision_data = {}
        metadata = {}
        for precision in PRECISIONS:
            run = args.root / args.mode / slug / precision.lower()
            prediction_path = run / "predictions.csv"
            validation_path = run / "validation.json"
            if not (prediction_path.is_file() and validation_path.is_file() and (run / ".complete").is_file()):
                raise FileNotFoundError(f"incomplete run: {run}")
            data = pd.read_csv(prediction_path, dtype={"example_id": "string"})
            if len(data) != expected_rows or set(data.method) != {method} or set(data.precision) != {precision}:
                raise ValueError(f"{prediction_path}: identity/row mismatch")
            if (set(data.task) != set(TASKS) or set(data.seed) != {44} or
                    set(data.regime) != {"CE+KD"} or set(data.model_id) != {args.model_id}):
                raise ValueError(f"{prediction_path}: protocol coverage mismatch")
            if data.duplicated(["task", "example_id"]).any() or not set(data.correct).issubset({0, 1}):
                raise ValueError(f"{prediction_path}: duplicate IDs or invalid correctness")
            precision_data[precision] = data
            metadata[precision] = json.loads(validation_path.read_text())
            for task in TASKS:
                cell = data[data.task.eq(task)]
                all_accuracy.append({"method": method, "precision": precision, "scope": task,
                                     "examples": len(cell), "accuracy": cell.correct.mean()})
            macro = np.mean([data[data.task.eq(task)].correct.mean() for task in TASKS])
            all_accuracy.append({"method": method, "precision": precision, "scope": "Macro",
                                 "examples": len(data), "accuracy": macro})

        reference = precision_data["BF16"]
        for precision in ["W8A16", "W4A16"]:
            comparison = precision_data[precision]
            task_samples = []
            task_points = []
            for task in TASKS:
                left = comparison[comparison.task.eq(task)][["example_id", "gold_answer", "correct"]].rename(
                    columns={"correct": "quantized"})
                right = reference[reference.task.eq(task)][["example_id", "gold_answer", "correct"]].rename(
                    columns={"gold_answer": "reference_gold", "correct": "bf16"})
                paired = left.merge(right, on="example_id", validate="one_to_one")
                if len(paired) != len(left) or not (paired.gold_answer == paired.reference_gold).all():
                    raise ValueError(f"unmatched example IDs/gold labels for {method}, {precision}, {task}")
                diff = paired.quantized.to_numpy(np.int8) - paired.bf16.to_numpy(np.int8)
                samples = bootstrap_difference(diff, args.n_bootstrap, rng)
                point = float(diff.mean())
                low, high = np.quantile(samples, [0.025, 0.975])
                all_bootstrap.append({"method": method, "comparison": f"{precision} - BF16",
                                      "scope": task, "examples": len(diff), "difference": point,
                                      "difference_pp": 100 * point, "ci95_low": float(low),
                                      "ci95_high": float(high), "ci95_low_pp": 100 * float(low),
                                      "ci95_high_pp": 100 * float(high), "n_bootstrap": args.n_bootstrap})
                task_samples.append(samples)
                task_points.append(point)
            macro_samples = np.mean(np.vstack(task_samples), axis=0)
            macro_point = float(np.mean(task_points))
            low, high = np.quantile(macro_samples, [0.025, 0.975])
            all_bootstrap.append({"method": method, "comparison": f"{precision} - BF16",
                                  "scope": "Macro", "examples": sum(len(precision_data[precision][precision_data[precision].task.eq(t)]) for t in TASKS),
                                  "difference": macro_point, "difference_pp": 100 * macro_point,
                                  "ci95_low": float(low), "ci95_high": float(high),
                                  "ci95_low_pp": 100 * float(low), "ci95_high_pp": 100 * float(high),
                                  "n_bootstrap": args.n_bootstrap})

        if args.mode == "formal":
            bf16_bytes = int(metadata["BF16"]["artifact_bytes"])
            for precision in PRECISIONS:
                artifact = Path(metadata[precision]["artifact"])
                artifact_bytes = artifact.stat().st_size
                if artifact_bytes != int(metadata[precision]["artifact_bytes"]):
                    raise ValueError(f"artifact size changed after evaluation: {artifact}")
                byte_rows.append({
                    "method": method, "model_id": args.model_id, "regime": "CE+KD", "seed": 44,
                    "precision": precision, "artifact_path": str(artifact),
                    "serialization_format": "weight-only torch.save; actual packed INT tensors and FP16 scales" if precision != "BF16" else "torch.save(state_dict), BF16 weight-only",
                    "artifact_bytes": artifact_bytes, "artifact_sha256": sha256(artifact),
                    "compressed_bf16_standalone_bytes": bf16_bytes,
                    "dense_8b_teacher_bytes": DENSE_8B_BYTES,
                    "dense_8b_teacher_sha256": DENSE_8B_SHA256,
                    "reduction_vs_compressed_bf16_pct": 100 * (1 - artifact_bytes / bf16_bytes),
                    "reduction_vs_dense_8b_teacher_pct": 100 * (1 - artifact_bytes / DENSE_8B_BYTES),
                })
        validation["methods"][method] = {
            "rows_per_precision": expected_rows,
            "prediction_sha256": {p: sha256(args.root / args.mode / slug / p.lower() / "predictions.csv") for p in PRECISIONS},
            "macro_accuracy": {p: metadata[p]["macro_accuracy"] for p in PRECISIONS},
        }

    output = args.root / "summary" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    accuracy = pd.DataFrame(all_accuracy)
    bootstrap = pd.DataFrame(all_bootstrap)
    accuracy.to_csv(output / "accuracy.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap_10000.csv", index=False)
    if byte_rows:
        pd.DataFrame(byte_rows).to_csv(output / "packed_byte_manifest.csv", index=False)
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    lines = [f"# Basis Sharing / SVD-LLM quantization ({args.model_id}, {args.mode})", "",
             "All accuracy uses BF16 compute. INT8/INT4 weights are evaluated after deterministic unpack/dequantization; no integer-kernel speed claim is made.", "",
             "## Accuracy", ""]
    for method in METHODS.values():
        lines += [f"### {method}", "", "| Task | BF16 | INT8 | INT4 |", "|---|---:|---:|---:|"]
        for scope in TASKS + ["Macro"]:
            values = []
            for precision in PRECISIONS:
                value = accuracy[(accuracy.method == method) & (accuracy.precision == precision) & (accuracy.scope == scope)].iloc[0].accuracy
                values.append(value)
            lines.append(f"| {scope} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
        lines.append("")
    lines += ["## Paired bootstrap differences", ""]
    for method in METHODS.values():
        lines += [f"### {method}", "", "| Comparison | Scope | Difference (pp) | 95% CI (pp) |",
                  "|---|---|---:|---:|"]
        for row in bootstrap[bootstrap.method.eq(method)].itertuples(index=False):
            comparison = row.comparison.replace("W8A16", "INT8").replace("W4A16", "INT4")
            lines.append(f"| {comparison} | {row.scope} | {row.difference_pp:+.3f} | "
                         f"[{row.ci95_low_pp:+.3f}, {row.ci95_high_pp:+.3f}] |")
        lines.append("")
    if byte_rows:
        lines += ["", "## Actual serialized bytes", "",
                  "| Method | Precision | GiB | Reduction vs method BF16 | Reduction vs dense teacher |",
                  "|---|---|---:|---:|---:|"]
        for row in byte_rows:
            lines.append(f"| {row['method']} | {DISPLAY_PRECISION[row['precision']]} | {row['artifact_bytes']/(1024**3):.3f} | "
                         f"{row['reduction_vs_compressed_bf16_pct']:.3f}% | {row['reduction_vs_dense_8b_teacher_pct']:.3f}% |")
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output), "accuracy_rows": len(accuracy),
                      "bootstrap_rows": len(bootstrap), "byte_rows": len(byte_rows)}, indent=2))


if __name__ == "__main__":
    main()
