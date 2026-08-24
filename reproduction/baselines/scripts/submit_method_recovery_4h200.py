#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from icassp27.controlled_baselines.config import stable_json_hash
from icassp27.controlled_baselines.method_recovery import load_method_config


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the complete method-specific 4xH200 rerun")
    parser.add_argument("--config", default=str(CODE_ROOT / "configs/method_recovery_4h200.example.yaml"))
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    cfg = load_method_config(args.config)
    preflight = subprocess.run(
        [sys.executable, str(SCRIPT_ROOT / "preflight_method_recovery.py"), cfg["_config_path"]],
        cwd=CODE_ROOT, text=True, capture_output=True,
    )
    if preflight.returncode != 0:
        sys.stderr.write(preflight.stdout)
        sys.stderr.write(preflight.stderr)
        raise SystemExit(preflight.returncode)
    print(json.dumps({"preflight": "PASS", "report":
                      str(Path(cfg["paths"]["output_root"]) / "PREFLIGHT.json")}), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_root = Path(cfg["paths"]["output_root"]) / "submissions" / f"{stamp}_p{os.getpid()}"
    spec_root = launch_root / "specs"
    spec_root.mkdir(parents=True, exist_ok=False)

    phases: dict[str, list[str]] = {"smoke": [], "pilots": [], "controls": [], "production": []}

    def write_spec(phase: str, spec: dict) -> None:
        directory = spec_root / phase
        directory.mkdir(parents=True, exist_ok=True)
        digest = stable_json_hash(spec)[:16]
        path = directory / f"{spec['stage']}_{digest}.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        phases[phase].append(str(path.resolve()))

    for method in cfg["matrix"]["methods"]:
        write_spec("smoke", {"stage": "smoke", "method": method, "backbone": "llama31_8b",
                             "target_reduction": 0.20, "objective": "ce_kd", "seed": 42,
                             "learning_rate": 3.0e-5, "max_steps": 2,
                             "validation_interval": 1, "validation_maximum": 8, "pilot": False})
    for method in cfg["matrix"]["methods"]:
        for backbone in cfg["matrix"]["backbones"]:
            for objective in cfg["matrix"]["objectives"]:
                for lr in cfg["lr_tuning"]["grid"]:
                    write_spec("pilots", {"stage": "recover", "pilot": True, "method": method,
                                          "backbone": backbone,
                                          "target_reduction": float(cfg["lr_tuning"]["target_reduction"]),
                                          "objective": objective, "seed": int(cfg["lr_tuning"]["seed"]),
                                          "learning_rate": float(lr)})
    for backbone in cfg["matrix"]["backbones"]:
        write_spec("controls", {"stage": "dense", "backbone": backbone})
    for method in cfg["matrix"]["methods"]:
        for backbone in cfg["matrix"]["backbones"]:
            for reduction in cfg["matrix"]["target_reductions"]:
                write_spec("controls", {"stage": "pure", "method": method, "backbone": backbone,
                                        "target_reduction": float(reduction)})

    if {key: len(value) for key, value in phases.items()} != {
        "smoke": 2, "pilots": 32, "controls": 14, "production": 0,
    }:
        raise RuntimeError("bootstrap matrix count mismatch")
    manifest = {"schema_version": 1, "config": cfg["_config_path"],
                "config_sha256": cfg["_config_sha256"], "launch_root": str(launch_root),
                "created_at_utc": stamp, "phases": phases,
                "expected": {"smoke": 2, "pilots": 32, "controls": 14, "production": 72}}
    manifest_path = launch_root / "WORKFLOW_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.submit:
        subprocess.run([sys.executable, str(SCRIPT_ROOT / "method_recovery_coordinator.py"),
                        "--manifest", str(manifest_path)], cwd=CODE_ROOT, check=True)
    summary = {"submitted": args.submit, "launch_root": str(launch_root),
               "manifest": str(manifest_path), "initial_jobs": 2 if args.submit else 0,
               "eventual_experimental_jobs": 120,
               "matrix": {"smoke": 2, "lr_pilots": 32, "pure": 12, "dense": 2,
                          "production_recovery": 72},
               "resources_per_experimental_job": "4xH200",
               "qos_dynamic_concurrency": 20,
               "final_summary": str(Path(cfg["paths"]["output_root"]) / "FINAL_SUMMARY.md")}
    (launch_root / "SUBMISSION.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
