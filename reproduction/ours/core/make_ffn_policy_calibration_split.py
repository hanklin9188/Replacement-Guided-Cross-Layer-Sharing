#!/usr/bin/env python3
"""Build record-disjoint FFN policy-calibration subsets from training data.

The split is made at the rendered-text hash-group level, so duplicate examples
cannot leak from the FFN-input feature subset into the intervention subset.
The source training file is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    records = [dict(item) for item in payload if isinstance(item, dict)]
    if not records:
        raise ValueError(f"no object records found: {path}")
    return records


def render_record(row: Mapping[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    parts: List[str] = []
    if instruction:
        parts.append("Instruction:\n" + instruction)
    if input_text:
        parts.append("Input:\n" + input_text)
    if output:
        parts.append("Answer:\n" + output)
    return "\n\n".join(parts).strip()


def text_hash(row: Mapping[str, Any]) -> str:
    rendered = render_record(row)
    if not rendered:
        raise ValueError("encountered a record with empty rendered text")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def take_hash_groups(
    keys: Sequence[str],
    groups: Mapping[str, Sequence[Dict[str, Any]]],
    start: int,
    requested_records: int,
) -> tuple[List[Dict[str, Any]], List[str], int]:
    selected: List[Dict[str, Any]] = []
    selected_keys: List[str] = []
    cursor = int(start)
    while cursor < len(keys) and len(selected) < int(requested_records):
        key = str(keys[cursor])
        selected_keys.append(key)
        selected.extend(dict(item) for item in groups[key])
        cursor += 1
    if len(selected) < int(requested_records):
        raise RuntimeError(
            f"not enough disjoint records: requested {requested_records}, selected {len(selected)}"
        )
    return selected, selected_keys, cursor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-records", type=int, default=4096)
    parser.add_argument("--intervention-records", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.feature_records) <= 0 or int(args.intervention_records) <= 0:
        raise ValueError("requested record counts must be positive")

    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    feature_path = output_dir / "feature_records.json"
    intervention_path = output_dir / "intervention_records.json"
    manifest_path = output_dir / "split_manifest.json"

    records = load_records(source)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[text_hash(row)].append(row)

    group_keys = sorted(groups)
    random.Random(int(args.seed)).shuffle(group_keys)
    feature, feature_keys, cursor = take_hash_groups(
        group_keys, groups, 0, int(args.feature_records)
    )
    intervention, intervention_keys, _cursor = take_hash_groups(
        group_keys, groups, cursor, int(args.intervention_records)
    )

    overlap = sorted(set(feature_keys) & set(intervention_keys))
    if overlap:
        raise AssertionError(f"feature/intervention hash overlap: {len(overlap)}")

    write_json(feature_path, feature)
    write_json(intervention_path, intervention)
    manifest = {
        "schema_version": 1,
        "method": "seeded_shuffle_of_rendered_text_sha256_groups",
        "seed": int(args.seed),
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "source_record_count": len(records),
        "source_unique_rendered_text_count": len(groups),
        "feature": {
            "path": str(feature_path),
            "sha256": file_sha256(feature_path),
            "requested_record_count": int(args.feature_records),
            "actual_record_count": len(feature),
            "unique_rendered_text_count": len(feature_keys),
        },
        "intervention": {
            "path": str(intervention_path),
            "sha256": file_sha256(intervention_path),
            "requested_record_count": int(args.intervention_records),
            "actual_record_count": len(intervention),
            "unique_rendered_text_count": len(intervention_keys),
        },
        "feature_intervention_hash_overlap_count": len(overlap),
        "assertions": {
            "source_unchanged": True,
            "feature_intervention_rendered_text_disjoint": len(overlap) == 0,
        },
    }
    write_json(manifest_path, manifest)
    print(f"[FFN-Calib-Split] feature={feature_path} records={len(feature)}")
    print(f"[FFN-Calib-Split] intervention={intervention_path} records={len(intervention)}")
    print(f"[FFN-Calib-Split] manifest={manifest_path} overlap={len(overlap)}")


if __name__ == "__main__":
    main()
