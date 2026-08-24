#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List


DEFAULT_DATASETS = ["piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]


def _parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").replace(",", " ").split() if x.strip()]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 7-task train-split calibration data in eval layout.")
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--datasets", type=str, default=" ".join(DEFAULT_DATASETS))
    parser.add_argument("--max_per_dataset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    datasets = _parse_list(str(args.datasets))
    _ensure_dir(str(args.output_root))
    manifest = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "seed": int(args.seed),
        "max_per_dataset": int(args.max_per_dataset),
        "datasets": {},
    }
    for dataset in datasets:
        source_path = os.path.join(str(args.source_root), dataset, "train.json")
        if not os.path.isfile(source_path):
            raise FileNotFoundError(source_path)
        rows = _load_json(source_path)
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        if int(args.max_per_dataset) > 0:
            idxs = idxs[: int(args.max_per_dataset)]
        sampled = [rows[i] for i in idxs]
        out_dir = os.path.join(str(args.output_root), dataset)
        _ensure_dir(out_dir)
        out_path = os.path.join(out_dir, "test.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(sampled, handle, ensure_ascii=False, indent=2)
        manifest["datasets"][dataset] = {
            "source_path": source_path,
            "output_path": out_path,
            "source_samples": int(len(rows)),
            "samples": int(len(sampled)),
        }
        print(f"[CalibData] {dataset}: {len(sampled)} / {len(rows)} -> {out_path}", flush=True)

    manifest_path = os.path.join(str(args.output_root), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[CalibData] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
