#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from icassp27.controlled_baselines.config import stable_json_hash
from icassp27.controlled_baselines.method_recovery import load_method_config, output_dir


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _complete(cfg: dict[str, Any], spec_path: Path) -> bool:
    return (output_dir(cfg, _read(spec_path)) / ".complete").is_file()


def _active_job_ids() -> set[str]:
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", "hank9188"), "-o", "%A"],
        text=True, capture_output=True, check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _terminal(job_id: str) -> bool:
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", job_id, "--format=State", "--parsable2"],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        return False
    states = [line.strip().split("|", 1)[0].split("+", 1)[0]
              for line in result.stdout.splitlines() if line.strip()]
    terminal = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
                "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT"}
    return bool(states) and all(state in terminal for state in states)


def _job_name(spec: dict[str, Any]) -> str:
    stage = spec["stage"]
    if stage == "dense":
        return f"m4_dense_{spec['backbone'].split('_')[-1]}"
    method = "bas" if spec["method"] == "basis_sharing" else "svd"
    backbone = spec["backbone"].split("_")[-1]
    rate = round(100 * float(spec.get("target_reduction", 0)))
    if stage == "pure":
        return f"m4_{method}_{backbone}_{rate}_r0"
    objective = spec["objective"].replace("_", "")
    seed = int(spec["seed"])
    label = "smk" if stage == "smoke" else ("lr" if spec.get("pilot") else "rec")
    return f"m4_{method}_{backbone}_{rate}_{objective}_s{seed}_{label}"


def _latest_job(state: dict[str, Any], spec_path: str) -> dict[str, Any] | None:
    matches = [row for row in state["jobs"] if row["spec"] == spec_path]
    return matches[-1] if matches else None


def _eligible_specs(cfg: dict[str, Any], paths: list[str], state: dict[str, Any],
                    active: set[str], failed_now: set[str]) -> list[Path]:
    maximum = int(cfg["slurm"]["maximum_run_attempts"])
    eligible = []
    for value in paths:
        path = Path(value)
        if _complete(cfg, path):
            continue
        attempts = int(state["attempts"].get(value, 0))
        if attempts >= maximum:
            continue
        latest = _latest_job(state, value)
        if latest is None:
            eligible.append(path)
        elif value in failed_now:
            eligible.append(path)
        elif latest["job_id"] in active:
            continue
        elif _terminal(latest["job_id"]):
            eligible.append(path)
    # Longest-processing-time-first reduces the final tail while leaving every
    # experimental setting unchanged.  Within the same class, retain the
    # immutable manifest order through Python's stable sort.
    def priority(path: Path) -> tuple[int, int]:
        spec = _read(path)
        backbone_priority = 0 if spec.get("backbone") == "llama31_8b" else 1
        objective_priority = 0 if spec.get("objective") == "ce_kd" else 1
        return backbone_priority, objective_priority

    return sorted(eligible, key=priority)


def _select_lrs(cfg: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for value in manifest["phases"]["pilots"]:
        spec = _read(Path(value))
        summary_path = output_dir(cfg, spec) / "summary.json"
        summary = _read(summary_path)
        key = (spec["method"], spec["backbone"], spec["objective"])
        groups.setdefault(key, []).append({
            "learning_rate": float(spec["learning_rate"]),
            "best_validation_decision_ce": float(summary["best_validation_decision_ce"]),
            "summary": str(summary_path),
        })
    expected = len(cfg["matrix"]["methods"]) * len(cfg["matrix"]["backbones"]) * len(cfg["matrix"]["objectives"])
    if len(groups) != expected or any(len(rows) != len(cfg["lr_tuning"]["grid"]) for rows in groups.values()):
        raise RuntimeError("LR selection matrix is incomplete")
    selected = {}
    records = []
    for key, candidates in sorted(groups.items()):
        candidates.sort(key=lambda row: (row["best_validation_decision_ce"], row["learning_rate"]))
        best = candidates[0]
        compound = "/".join(key)
        selected[compound] = best["learning_rate"]
        records.append({"method": key[0], "backbone": key[1], "objective": key[2],
                        "selected_learning_rate": best["learning_rate"], "candidates": candidates})
    result = {"selection_metric": "minimum validation decision CE", "selected": selected,
              "records": records}
    path = Path(cfg["paths"]["output_root"]) / "SELECTED_LRS.json"
    _atomic_json(path, result)
    return result


def _write_production_specs(cfg: dict[str, Any], manifest_path: Path,
                            manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["phases"].get("production"):
        return manifest
    selected = _select_lrs(cfg, manifest)["selected"]
    spec_root = manifest_path.parent / "specs" / "production"
    spec_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for method in cfg["matrix"]["methods"]:
        for backbone in cfg["matrix"]["backbones"]:
            for reduction in cfg["matrix"]["target_reductions"]:
                for objective in cfg["matrix"]["objectives"]:
                    lr = selected[f"{method}/{backbone}/{objective}"]
                    for seed in cfg["matrix"]["seeds"]:
                        spec = {"stage": "recover", "method": method, "backbone": backbone,
                                "target_reduction": float(reduction), "objective": objective,
                                "seed": int(seed), "learning_rate": float(lr), "pilot": False}
                        digest = stable_json_hash(spec)[:16]
                        path = spec_root / f"recover_{digest}.json"
                        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
                        paths.append(str(path))
    if len(paths) != 72:
        raise RuntimeError(f"expected 72 production specs, got {len(paths)}")
    manifest["phases"]["production"] = paths
    _atomic_json(manifest_path, manifest)
    return manifest


def _submit(cfg: dict[str, Any], manifest_path: Path, spec_path: Path,
            state_path: Path, state: dict[str, Any]) -> bool:
    spec = _read(spec_path)
    command = [
        "sbatch", "--parsable", f"--job-name={_job_name(spec)}", "--export",
        f"ALL,METHOD_CONFIG={cfg['_config_path']},METHOD_SPEC={spec_path},METHOD_MANIFEST={manifest_path}",
        str(CODE_ROOT / "reproduction/baselines/slurm/method_recovery_4xh200.sbatch"),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if "QOSMaxSubmitJobPerUserLimit" in message or "AssocMaxSubmitJobLimit" in message:
            return False
        raise RuntimeError(message)
    job_id = result.stdout.strip().split(";", 1)[0]
    key = str(spec_path)
    state["attempts"][key] = int(state["attempts"].get(key, 0)) + 1
    state["jobs"].append({"job_id": job_id, "spec": key, "stage": spec["stage"],
                          "attempt": state["attempts"][key], "submitted_at": time.time()})
    _atomic_json(state_path, state)
    print(json.dumps({"submitted_job": job_id, "name": _job_name(spec), "spec": key}), flush=True)
    return True


def _failure_report(cfg: dict[str, Any], manifest: dict[str, Any], state: dict[str, Any],
                    active: set[str], failed_now: set[str]) -> list[dict[str, Any]]:
    failed = []
    maximum = int(cfg["slurm"]["maximum_run_attempts"])
    for phase in ("smoke", "pilots", "controls", "production"):
        for value in manifest["phases"].get(phase, []):
            attempts = int(state["attempts"].get(value, 0))
            latest = _latest_job(state, value)
            exhausted = (value in failed_now or
                         (latest is not None and latest["job_id"] not in active and
                          _terminal(latest["job_id"])))
            if not _complete(cfg, Path(value)) and attempts >= maximum and exhausted:
                failed.append({"phase": phase, "spec": value, "attempts": attempts,
                               "last_job_id": latest["job_id"] if latest else None})
    return failed


def coordinate(manifest_path: Path, completed_spec: str | None, exit_code: int | None) -> None:
    lock_path = manifest_path.parent / ".coordinator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _read(manifest_path)
        cfg = load_method_config(manifest["config"])
        state_path = manifest_path.parent / "WORKFLOW_STATE.json"
        state = _read(state_path) if state_path.is_file() else {
            "status": "RUNNING", "attempts": {}, "jobs": [], "events": [], "created_at": time.time()
        }
        failed_now: set[str] = set()
        if completed_spec is not None:
            canonical = str(Path(completed_spec).resolve())
            event = {"spec": canonical, "exit_code": int(exit_code or 0), "time": time.time(),
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
            state["events"].append(event)
            if int(exit_code or 0) != 0 or not _complete(cfg, Path(canonical)):
                failed_now.add(canonical)
            _atomic_json(state_path, state)

        if all(_complete(cfg, Path(value)) for value in manifest["phases"]["smoke"]):
            phase = "pilots"
            if all(_complete(cfg, Path(value)) for value in manifest["phases"]["pilots"]):
                manifest = _write_production_specs(cfg, manifest_path, manifest)
                phase = "work"
        else:
            phase = "smoke"

        if phase == "work":
            work = manifest["phases"]["controls"] + manifest["phases"]["production"]
            if all(_complete(cfg, Path(value)) for value in work):
                subprocess.run([sys.executable, str(SCRIPT_ROOT / "aggregate_method_recovery.py"),
                                cfg["_config_path"]], cwd=CODE_ROOT, check=True)
                state["status"] = "COMPLETE"
                state["completed_at"] = time.time()
                _atomic_json(state_path, state)
                return
            candidates = _eligible_specs(cfg, work, state, _active_job_ids(), failed_now)
        else:
            candidates = _eligible_specs(cfg, manifest["phases"][phase], state,
                                          _active_job_ids(), failed_now)

        active = _active_job_ids()
        failures = _failure_report(cfg, manifest, state, active, failed_now)
        if failures:
            state["status"] = "FAILED"
            state["terminal_failures"] = failures
            _atomic_json(state_path, state)
            _atomic_json(Path(cfg["paths"]["output_root"]) / "WORKFLOW_FAILED.json",
                         {"status": "FAILED", "failures": failures, "state": str(state_path)})
            return

        capacity = int(cfg["slurm"]["qos_max_jobs_per_user"])
        slots = max(0, capacity - len(active))
        for path in candidates[:slots]:
            if not _submit(cfg, manifest_path, path, state_path, state):
                break
        state["status"] = "RUNNING"
        state.pop("terminal_failures", None)
        state["phase"] = phase
        state["active_user_jobs_seen"] = len(active)
        state["updated_at"] = time.time()
        _atomic_json(state_path, state)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep the 4xH200 method-recovery DAG full under QOS=20")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--completed-spec")
    parser.add_argument("--exit-code", type=int)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    coordinate(Path(args.manifest).resolve(), args.completed_spec, args.exit_code)


if __name__ == "__main__":
    main()
