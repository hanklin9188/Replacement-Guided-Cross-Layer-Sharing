#!/usr/bin/env python3
"""Validate Basis Sharing and SVD-LLM payloads before paper integration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

METHODS = {
    "basis_sharing": "Basis Sharing",
    "svd_llm": "SVD-LLM",
}
MODEL_IDS = {"3b_15", "3b_20", "3b_25", "8b_15", "8b_20", "8b_25"}
SEEDS = {42, 43, 44}
TASK_ALIASES = {
    "ARC-Challenge": "ARC-Challenge", "ARC_C": "ARC-Challenge",
    "ARC-Easy": "ARC-Easy", "ARC_E": "ARC-Easy",
    "hellaswag": "hellaswag", "HellaSwag": "hellaswag",
    "openbookqa": "openbookqa", "OpenBookQA": "openbookqa",
    "piqa": "piqa", "PIQA": "piqa",
    "social_i_qa": "social_i_qa", "SocialIQA": "social_i_qa",
    "winogrande": "winogrande", "WinoGrande": "winogrande",
}
TASKS = set(TASK_ALIASES.values())
PREDICTION_COLUMNS = {
    "method", "model_id", "regime", "seed", "task", "example_id",
    "source_index", "gold_answer", "prediction", "correct",
}
BYTE_COLUMNS = {
    "method", "backbone", "nominal_target", "dense_serialized_bytes",
    "compressed_serialized_bytes",
}
QUANT_COLUMNS = {
    "method", "model_id", "backbone", "nominal_target", "precision",
    "macro_accuracy", "serialized_gib", "quantizer", "source_status",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def validate_predictions(path: Path, method: str) -> dict[str, object]:
    coverage: Counter[tuple[str, int, str]] = Counter()
    seen: set[tuple[str, int, str, int]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = PREDICTION_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if row["method"] != method:
                raise RuntimeError(f"{path}:{line_number}: expected method {method!r}")
            model_id, seed = row["model_id"], int(row["seed"])
            if model_id not in MODEL_IDS or seed not in SEEDS or row["task"] not in TASK_ALIASES:
                raise RuntimeError(f"{path}:{line_number}: unsupported model/seed/task")
            task = TASK_ALIASES[row["task"]]
            source_index = int(row["source_index"])
            key = (model_id, seed, task, source_index)
            if key in seen:
                raise RuntimeError(f"{path}:{line_number}: duplicate {key}")
            seen.add(key)
            parse_bool(row["correct"])
            coverage[(model_id, seed, task)] += 1
    expected = {(model, seed, task) for model in MODEL_IDS for seed in SEEDS for task in TASKS}
    if set(coverage) != expected:
        raise RuntimeError(f"{path}: prediction coverage mismatch")
    if any(count <= 0 for count in coverage.values()):
        raise RuntimeError(f"{path}: empty prediction group")
    return {"rows": len(seen), "groups": len(coverage), "group_sizes": sorted(set(coverage.values()))}


def validate_bytes(path: Path, method: str) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = BYTE_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    keys = set()
    for row in rows:
        if row["method"] != method:
            raise RuntimeError(f"{path}: wrong method label")
        key = (row["backbone"], int(row["nominal_target"]))
        keys.add(key)
        dense, compressed = int(row["dense_serialized_bytes"]), int(row["compressed_serialized_bytes"])
        if not 0 < compressed < dense:
            raise RuntimeError(f"{path}: invalid byte counts for {key}")
    expected = {(backbone, target) for backbone in ("Llama-3.2-3B", "Llama-3.1-8B") for target in (15, 20, 25)}
    if len(rows) != 6 or keys != expected:
        raise RuntimeError(f"{path}: byte-manifest coverage mismatch")
    return {"rows": len(rows)}


def validate_quantization(path: Path, method: str) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = QUANT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    keys = set()
    for row in rows:
        if row["method"] != method or row["source_status"] != "observed":
            raise RuntimeError(f"{path}: method/status mismatch")
        key = (row["model_id"], row["precision"])
        keys.add(key)
        accuracy, size = float(row["macro_accuracy"]), float(row["serialized_gib"])
        if not 0.0 <= accuracy <= 1.0 or size <= 0.0:
            raise RuntimeError(f"{path}: invalid quantization values for {key}")
    expected = {(model, precision) for model in ("8b_15", "8b_25") for precision in ("bf16", "w8a16", "w4a16")}
    if len(rows) != 6 or keys != expected:
        raise RuntimeError(f"{path}: quantization coverage mismatch")
    return {"rows": len(rows)}


def validate_manifest(path: Path, method: str) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "method", "upstream_repository", "upstream_revision", "model_revisions",
        "evaluator_revision", "length_norm", "seeds", "commands", "environment",
    }
    missing = required - set(manifest)
    if missing:
        raise RuntimeError(f"{path}: missing keys {sorted(missing)}")
    if manifest["method"] != method or manifest["length_norm"] != "none" or manifest["seeds"] != [42, 43, 44]:
        raise RuntimeError(f"{path}: method/evaluator contract mismatch")
    if len(str(manifest["upstream_revision"])) < 7:
        raise RuntimeError(f"{path}: upstream revision is not immutable enough")
    return {"upstream_revision": manifest["upstream_revision"], "evaluator_revision": manifest["evaluator_revision"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "external")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "external_import_manifest.json")
    args = parser.parse_args()
    report: dict[str, object] = {"status": "PASS", "methods": {}}
    for directory, method in METHODS.items():
        incoming = args.root / directory / "incoming"
        files = {
            "predictions": incoming / "predictions.csv",
            "bytes": incoming / "byte_manifest.csv",
            "quantization": incoming / "quantization.csv",
            "manifest": incoming / "run_manifest.json",
        }
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        report["methods"][method] = {
            "predictions": validate_predictions(files["predictions"], method),
            "bytes": validate_bytes(files["bytes"], method),
            "quantization": validate_quantization(files["quantization"], method),
            "manifest": validate_manifest(files["manifest"], method),
            "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in files.items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
