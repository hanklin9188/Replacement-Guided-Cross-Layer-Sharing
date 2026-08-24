#!/usr/bin/env python3
"""Strict final validation for baseline RTN quantization exports."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd
import torch


METHODS = {"basis_sharing": "Basis Sharing", "svd_llm": "SVD-LLM"}
PRECISIONS = ["BF16", "W8A16", "W4A16"]
TASK_COUNTS = {"ARC_C": 1172, "ARC_E": 2376, "HellaSwag": 10042, "OpenBookQA": 500,
               "PIQA": 1838, "SocialIQA": 1954, "WinoGrande": 1267}
EXPECTED_FIELDS = ["method", "model_id", "regime", "seed", "precision", "task", "example_id",
                   "source_index", "gold_answer", "prediction", "correct", "candidate_scores"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-id", choices=["8b_15", "8b_25"], default="8b_15")
    args = parser.parse_args()
    root_path = args.root.resolve()
    canonical = None
    runs = []
    for slug, method in METHODS.items():
        for precision in PRECISIONS:
            root = root_path / "formal" / slug / precision.lower()
            data = pd.read_csv(root / "predictions.csv", dtype={"example_id": "string"})
            metadata = json.loads((root / "validation.json").read_text())
            if list(data.columns) != EXPECTED_FIELDS or len(data) != sum(TASK_COUNTS.values()):
                raise ValueError(f"{root}: schema or row count mismatch")
            if set(data.method) != {method} or set(data.precision) != {precision}:
                raise ValueError(f"{root}: method/precision mismatch")
            if set(data.model_id) != {args.model_id} or set(data.regime) != {"CE+KD"} or set(data.seed) != {44}:
                raise ValueError(f"{root}: model/regime/seed mismatch")
            if not set(data.correct).issubset({0, 1}) or data.duplicated(["task", "example_id"]).any():
                raise ValueError(f"{root}: invalid correct or duplicate IDs")
            if not ((data.gold_answer == data.prediction).astype(int) == data.correct).all():
                raise ValueError(f"{root}: correctness mismatch")
            for task, expected in TASK_COUNTS.items():
                cell = data[data.task.eq(task)]
                if len(cell) != expected or cell.example_id.nunique() != expected or cell.source_index.nunique() != expected:
                    raise ValueError(f"{root}: incomplete {task}")
            mapping = data[["task", "example_id", "source_index", "gold_answer"]].sort_values(
                ["task", "source_index"]).reset_index(drop=True)
            if canonical is None:
                canonical = mapping
            elif not mapping.equals(canonical):
                raise ValueError(f"{root}: example mapping differs")
            for row in data.itertuples(index=False):
                scores = json.loads(row.candidate_scores)
                if len(scores) < 2 or max(range(len(scores)), key=scores.__getitem__) != row.prediction:
                    raise ValueError(f"{root}: candidate score/prediction mismatch")
            if metadata["training_invoked"] or not metadata["evaluation_only"]:
                raise ValueError(f"{root}: training/evaluation invariant failed")
            protocol = metadata["protocol"]
            required_protocol = {"mode": "logprob", "length_norm": "none", "shots": 0,
                                 "compute_dtype": "bfloat16", "split": "final", "max_length": 384}
            if any(protocol.get(key) != value for key, value in required_protocol.items()):
                raise ValueError(f"{root}: protocol mismatch")
            runs.append({"method": method, "precision": precision, "rows": len(data),
                         "macro_accuracy": metadata["macro_accuracy"], "artifact": metadata["artifact"],
                         "artifact_bytes": metadata["artifact_bytes"], "artifact_sha256": metadata["artifact_sha256"]})

    artifact_audit = []
    expected_quantized = {"basis_sharing": 368, "svd_llm": 448}
    expected_aliases = {"basis_sharing": 80, "svd_llm": 0}
    run_artifacts = {(run["method"], run["precision"]): Path(run["artifact"]) for run in runs}
    for slug, method in METHODS.items():
        for precision in PRECISIONS:
            path = run_artifacts[(method, precision)]
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if precision == "BF16":
                if len(payload) != 515 or {str(x.dtype) for x in payload.values()} != {"torch.bfloat16"}:
                    raise ValueError(f"{path}: BF16 state invariant failed")
                encodings = {"bf16": 515}
            else:
                tensors = payload["canonical_tensors"]
                if len(tensors) + len(payload["aliases"]) != 515 or len(payload["aliases"]) != expected_aliases[slug]:
                    raise ValueError(f"{path}: canonical/alias coverage failed")
                encodings = {}
                for value in tensors.values():
                    encodings[value["encoding"]] = encodings.get(value["encoding"], 0) + 1
                expected_encoding = "int8_symmetric_per_output_channel" if precision == "W8A16" else "int4_symmetric_grouped"
                if encodings.get(expected_encoding) != expected_quantized[slug] or encodings.get("bf16") != 67:
                    raise ValueError(f"{path}: encoding counts failed")
                for value in tensors.values():
                    if "scale" in value and value["scale"].dtype != torch.float16:
                        raise ValueError(f"{path}: scale is not FP16")
                    if "qweight" in value and value["qweight"].dtype != torch.int8:
                        raise ValueError(f"{path}: W8 tensor is not INT8")
                    if "qweight_packed" in value and value["qweight_packed"].dtype != torch.uint8:
                        raise ValueError(f"{path}: packed W4 tensor is not uint8")
            artifact_audit.append({"method": method, "precision": precision, "path": str(path),
                                   "bytes": path.stat().st_size, "encodings": encodings})

    report = {"status": "PASS", "formal_runs": len(runs), "rows_per_run": sum(TASK_COUNTS.values()),
              "total_prediction_rows": sum(run["rows"] for run in runs), "runs": runs,
              "artifact_audit": artifact_audit,
              "claims": {"accuracy": True, "packed_checkpoint_bytes": True, "compression_ratio": True,
                         "integer_kernel_latency": False, "integer_kernel_tokens_per_second": False,
                         "integer_kernel_peak_vram": False}}
    destination = root_path / "summary/formal/final_validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "formal_runs": len(runs),
                      "total_prediction_rows": report["total_prediction_rows"],
                      "artifact_rows": len(artifact_audit), "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
