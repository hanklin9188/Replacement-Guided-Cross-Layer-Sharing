#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT / "src"))

from icassp27.controlled_baselines.method_recovery import load_method_config


def count(root: Path) -> int:
    return sum(1 for _ in root.rglob(".complete")) if root.is_dir() else 0


def main() -> None:
    config = Path(sys.argv[1] if len(sys.argv) > 1 else CODE_ROOT / "configs/method_recovery_4h200.example.yaml")
    cfg = load_method_config(config)
    output = Path(cfg["paths"]["output_root"])
    submissions = sorted((output / "submissions").glob("*/WORKFLOW_STATE.json"))
    state = json.loads(submissions[-1].read_text()) if submissions else {}
    queue = subprocess.run(["squeue", "-h", "-u", "hank9188", "-o", "%i|%j|%t|%M|%R"],
                           text=True, capture_output=True, check=True).stdout.splitlines()
    progress = []
    for path in Path(cfg["paths"]["result_root"]).rglob("progress.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "training":
            progress.append({"path": str(path.parent), **row})
    result = {
        "workflow_status": state.get("status"), "phase": state.get("phase"),
        "pilots": {"complete": count(Path(cfg["paths"]["pilot_root"])), "expected": 32},
        "controls": {"complete": count(Path(cfg["paths"]["control_root"])), "expected": 14},
        "recovery": {"complete": count(Path(cfg["paths"]["result_root"])), "expected": 72},
        "active_jobs": len(queue), "queue": queue, "reported_training_progress": progress,
        "final_summary_exists": (output / "FINAL_SUMMARY.md").is_file(),
        "fairness_audit_exists": (output / "FAIRNESS_AUDIT.json").is_file(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
