#!/usr/bin/env python3
"""Select the safer shared-FFN initialization after simultaneous multi-merge validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


def load_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def decision_ce(report: Dict[str, Any], path: Path) -> float:
    val_last = report.get("val_last", {})
    if not isinstance(val_last, dict):
        raise ValueError(f"missing val_last in {path}")
    value = float(val_last.get("decision_ce", float("nan")))
    if not math.isfinite(value):
        raise ValueError(f"non-finite zero-shot decision_ce in {path}")
    if int(report.get("actual_final_step", -1)) != 0:
        raise ValueError(f"zero-shot report unexpectedly trained parameters: {path}")
    if str(report.get("training_prompt_mode")) != "decision_aligned":
        raise ValueError(f"zero-shot report is not decision-aligned: {path}")
    if str(report.get("loss_scope")) != "decision":
        raise ValueError(f"zero-shot report did not use decision scope: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-medoid-report", required=True)
    parser.add_argument("--atlas-medoid-report", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-strategy", required=True)
    parser.add_argument(
        "--policy-medoid-tie-tolerance",
        type=float,
        default=1e-8,
        help="Prefer the measured functional medoid when scores are tied within this tolerance.",
    )
    args = parser.parse_args()

    policy_path = Path(args.policy).expanduser().resolve()
    policy = load_object(policy_path)
    reports = {
        "policy_medoid": (
            Path(args.policy_medoid_report).expanduser().resolve(),
            load_object(Path(args.policy_medoid_report).expanduser().resolve()),
        ),
        "medoid": (
            Path(args.atlas_medoid_report).expanduser().resolve(),
            load_object(Path(args.atlas_medoid_report).expanduser().resolve()),
        ),
    }
    scores = {name: decision_ce(report, path) for name, (path, report) in reports.items()}

    policy_report = reports["policy_medoid"][1]
    if str(policy_report.get("proto_seed_strategy_resolved")) != "policy_medoid":
        raise ValueError("policy-medoid trial did not resolve to policy_medoid")
    expected_shared_medoids = {
        int(group["medoid_layer"])
        for group in policy.get("groups", [])
        if isinstance(group, dict)
    }
    actual_policy_seeds = {int(value) for value in policy_report.get("proto_seed_layers", {}).values()}
    if not expected_shared_medoids.issubset(actual_policy_seeds):
        raise ValueError(
            "policy-medoid trial ignored measured medoids: "
            f"expected={sorted(expected_shared_medoids)} actual={sorted(actual_policy_seeds)}"
        )

    tolerance = max(0.0, float(args.policy_medoid_tie_tolerance))
    if scores["policy_medoid"] <= scores["medoid"] + tolerance:
        selected = "policy_medoid"
        reason = "lower_or_tied_simultaneous_merge_decision_ce"
    else:
        selected = "medoid"
        reason = "atlas_medoid_is_safer_under_simultaneous_merge_decision_ce"

    output = {
        "status": "PASS",
        "selection_metric": "zero_shot_simultaneous_multi_merge_decision_ce",
        "selected_strategy": selected,
        "selection_reason": reason,
        "scores": scores,
        "policy_medoid_minus_atlas_medoid": scores["policy_medoid"] - scores["medoid"],
        "policy": str(policy_path),
        "reports": {name: str(path) for name, (path, _report) in reports.items()},
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    output_strategy = Path(args.output_strategy).expanduser().resolve()
    output_strategy.parent.mkdir(parents=True, exist_ok=True)
    output_strategy.write_text(selected + "\n", encoding="utf-8")
    print(
        "[ZeroShot-Seed-Gate] "
        f"policy_medoid={scores['policy_medoid']:.8f} "
        f"atlas_medoid={scores['medoid']:.8f} selected={selected}"
    )
    print(f"[ZeroShot-Seed-Gate] report={output_json}")


if __name__ == "__main__":
    main()
