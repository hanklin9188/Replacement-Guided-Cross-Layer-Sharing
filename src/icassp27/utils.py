from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_revision(project_dir: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_dir, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_bucket(value: str, seed: int, buckets: int = 10_000) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slurm_metadata() -> dict[str, Any]:
    names = [
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
    ]
    return {name.lower(): os.environ.get(name) for name in names}


def base_manifest(cfg: dict[str, Any], stage: str, backbone: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "backbone": backbone,
        "created_at_utc": utc_now(),
        "code_revision": code_revision(cfg["_project_dir"]),
        "config_path": cfg["_config_path"],
        "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(),
        "slurm": slurm_metadata(),
    }


def module_checksum(module: torch.nn.Module) -> dict[str, float | int]:
    # A deterministic restoration checksum without copying full FFNs to host.
    count = 0
    total = torch.zeros((), device=next(module.parameters()).device, dtype=torch.float64)
    squared = torch.zeros_like(total)
    with torch.no_grad():
        for parameter in module.parameters():
            values = parameter.detach().double()
            count += values.numel()
            total += values.sum()
            squared += values.square().sum()
    return {"numel": count, "sum": total.item(), "squared_sum": squared.item()}


def checksum_equal(a: dict[str, float | int], b: dict[str, float | int]) -> bool:
    return a["numel"] == b["numel"] and np.allclose(
        [a["sum"], a["squared_sum"]], [b["sum"], b["squared_sum"]],
        rtol=1e-12, atol=1e-8,
    )


def final_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=1).sub(1).clamp_min(0)


def select_final(tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    index = final_valid_indices(attention_mask)
    batch = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[batch, index]
