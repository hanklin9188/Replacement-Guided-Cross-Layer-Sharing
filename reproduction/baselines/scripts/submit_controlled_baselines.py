#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT / "src"))

from icassp27.controlled_baselines.config import load_config, raw_checkpoint_dir, stable_json_hash


def arguments():
    parser = argparse.ArgumentParser(description="Generate or submit the 1xH200 controlled baseline matrix")
    parser.add_argument("--config", default=str(CODE_ROOT / "configs/controlled_baselines.example.yaml"))
    parser.add_argument("--submit", action="store_true", help="actually call sbatch; default is a safe dry-run")
    parser.add_argument("--use-existing-raw", action="store_true", help="do not submit 12 compression jobs")
    parser.add_argument("--method", action="append", choices=["basis_sharing", "svd_llm"])
    parser.add_argument("--backbone", action="append", choices=["llama32_3b", "llama31_8b"])
    parser.add_argument("--rate", action="append", type=int, choices=[15, 20, 25])
    return parser.parse_args()


def submit(command: list[str], retries: int, delay: int) -> str:
    for attempt in range(1, retries + 1):
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip().split(";", 1)[0]
        message = (result.stderr or result.stdout).strip()
        if "QOSMaxSubmitJobPerUserLimit" not in message or attempt == retries:
            raise RuntimeError(message)
        time.sleep(delay)
    raise AssertionError


def main() -> None:
    args = arguments()
    cfg = load_config(args.config)
    if args.submit:
        command = [sys.executable, "-m", "icassp27.controlled_baselines.cli",
                   "--config", str(Path(args.config).resolve()), "preflight"]
        if args.use_existing_raw:
            command.append("--require-raw")
        subprocess.run(command, cwd=CODE_ROOT, env={**os.environ, "PYTHONPATH": str(CODE_ROOT / "src")}, check=True)
    methods = args.method or cfg["matrix"]["methods"]
    backbones = args.backbone or cfg["matrix"]["backbones"]
    reductions = [value / 100 for value in args.rate] if args.rate else cfg["matrix"]["target_reductions"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_root = Path(cfg["paths"]["output_root"]) / "submissions" / f"{timestamp}_p{os.getpid()}"
    spec_root = launch_root / "specs"
    spec_root.mkdir(parents=True, exist_ok=False)
    sbatch = CODE_ROOT / "reproduction/baselines/slurm/controlled_baseline_1xh200.sbatch"
    jobs = []
    compression_job: dict[tuple[str, str, float], str] = {}
    dry_index = 0

    def launch(spec: dict, dependency: str = "") -> str:
        nonlocal dry_index
        digest = stable_json_hash(spec)[:16]
        path = spec_root / f"{spec['stage']}_{digest}.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        name = f"cb_{spec['method'][:3]}_{spec['backbone'].split('_')[-1]}_{round(100*spec['target_reduction'])}_{spec['stage'][:3]}"
        if spec.get("objective"):
            name += f"_{spec['objective'].replace('_','')}_s{spec['seed']}"
        command = ["sbatch", "--parsable", f"--job-name={name}"]
        if dependency:
            command.append(f"--dependency=afterok:{dependency}")
        command.extend(["--export", f"ALL,CONTROLLED_CONFIG={Path(args.config).resolve()},CONTROLLED_SPEC={path}",
                        str(sbatch)])
        if args.submit:
            job_id = submit(command, int(cfg["slurm"]["submission_retries"]),
                            int(cfg["slurm"]["retry_seconds"]))
        else:
            dry_index += 1
            job_id = f"DRY{dry_index:04d}"
            print(" ".join(command))
        jobs.append({"job_id": job_id, "dependency": dependency, "spec": str(path), **spec})
        return job_id

    for method in methods:
        for backbone in backbones:
            for reduction in reductions:
                key = (method, backbone, float(reduction))
                if args.use_existing_raw:
                    root = raw_checkpoint_dir(cfg, *key)
                    if not (root / ".complete").is_file():
                        raise FileNotFoundError(f"--use-existing-raw requested but checkpoint is incomplete: {root}")
                    compression_job[key] = ""
                else:
                    compression_job[key] = launch({"stage": "compress", "method": method,
                                                   "backbone": backbone, "target_reduction": float(reduction)})
    for method in methods:
        for backbone in backbones:
            for reduction in reductions:
                parent = compression_job[(method, backbone, float(reduction))]
                launch({"stage": "pure", "method": method, "backbone": backbone,
                        "target_reduction": float(reduction)}, parent)
                for objective in cfg["matrix"]["objectives"]:
                    for seed in cfg["matrix"]["seeds"]:
                        launch({"stage": "recover", "method": method, "backbone": backbone,
                                "target_reduction": float(reduction), "objective": objective,
                                "seed": int(seed)}, parent)
    fields = sorted({key for row in jobs for key in row})
    with (launch_root / "JOB_REGISTRY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(jobs)
    summary = {"submitted": args.submit, "jobs": len(jobs),
               "compression_jobs": sum(row["stage"] == "compress" for row in jobs),
               "pure_jobs": sum(row["stage"] == "pure" for row in jobs),
               "recovery_jobs": sum(row["stage"] == "recover" for row in jobs),
               "all_jobs_request": "1xH200", "launch_root": str(launch_root)}
    (launch_root / "SUBMISSION.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
