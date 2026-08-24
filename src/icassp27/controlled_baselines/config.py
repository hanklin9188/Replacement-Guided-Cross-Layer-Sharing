from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from . import BACKBONES, METHODS, OBJECTIVES


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    cfg = yaml.safe_load(source.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("controlled baseline config must be a mapping")
    cfg["_config_path"] = str(source)
    cfg["_config_sha256"] = sha256_file(source)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    matrix = cfg["matrix"]
    if tuple(matrix["methods"]) != METHODS:
        raise ValueError(f"methods must be exactly {METHODS}")
    if tuple(matrix["backbones"]) != BACKBONES:
        raise ValueError(f"backbones must be exactly {BACKBONES}")
    if tuple(matrix["objectives"]) != OBJECTIVES:
        raise ValueError(f"objectives must be exactly {OBJECTIVES}")
    if list(map(int, matrix["seeds"])) != [42, 43, 44]:
        raise ValueError("controlled recovery seeds must be exactly 42, 43, 44")
    if [float(value) for value in matrix["target_reductions"]] != [0.15, 0.20, 0.25]:
        raise ValueError("target reductions must be exactly 15%, 20%, 25%")
    recovery = cfg["recovery"]
    required = {
        "rank": 128, "alpha": 128, "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"], "max_length": 384,
        "validation_interval": 500, "validation_maximum": 2048,
        "selection_metric": "decision_ce", "temperature": 2.0,
    }
    for key, expected in required.items():
        if recovery.get(key) != expected:
            raise ValueError(f"recovery.{key} must be {expected!r}, got {recovery.get(key)!r}")
    if int(cfg["slurm"]["gpus_per_job"]) != 1 or cfg["slurm"]["gpu_type"] != "H200":
        raise ValueError("every controlled job must request exactly 1 H200 GPU")


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def rate_label(reduction: float) -> str:
    return f"{round(100 * float(reduction)):02d}pct"


def raw_checkpoint_dir(cfg: dict[str, Any], method: str, backbone: str, reduction: float) -> Path:
    return Path(cfg["paths"]["raw_checkpoint_root"]) / method / backbone / rate_label(reduction)


def result_dir(cfg: dict[str, Any], method: str, backbone: str, reduction: float,
               objective: str, seed: int | None = None) -> Path:
    root = Path(cfg["paths"]["result_root"]) / method / backbone / rate_label(reduction) / objective
    return root if seed is None else root / f"seed_{int(seed)}"
