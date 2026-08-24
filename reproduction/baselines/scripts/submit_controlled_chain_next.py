#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def append_event(path: Path, event: dict) -> None:
    event = {"time_utc": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit the next job in one controlled-baseline lane")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--current-index", required=True, type=int)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["entries"]
    next_index = args.current_index + 1
    events = Path(manifest["events"])
    append_event(events, {
        "event": "task_finished", "index": args.current_index,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    })
    if next_index >= len(entries):
        append_event(events, {"event": "lane_complete", "index": args.current_index})
        return

    entry = entries[next_index]
    command = [
        "sbatch", "--parsable", f"--job-name={entry['job_name']}",
        "--export",
        ",".join([
            "ALL", f"CONTROLLED_CONFIG={manifest['config']}",
            f"CONTROLLED_SPEC={entry['spec']}",
            f"CONTROLLED_CHAIN_MANIFEST={manifest_path}",
            f"CONTROLLED_CHAIN_INDEX={next_index}",
        ]),
        manifest["sbatch"],
    ]
    for attempt in range(1, 241):
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split(";", 1)[0]
            append_event(events, {
                "event": "job_submitted", "index": next_index,
                "job_id": job_id, "spec": entry["spec"],
            })
            return
        message = (result.stderr or result.stdout).strip()
        if "QOSMaxSubmitJobPerUserLimit" not in message:
            append_event(events, {
                "event": "submission_failed", "index": next_index,
                "attempt": attempt, "error": message,
            })
            raise RuntimeError(message)
        time.sleep(15)
    raise RuntimeError("timed out waiting for a free QOS submission slot")


if __name__ == "__main__":
    main()
