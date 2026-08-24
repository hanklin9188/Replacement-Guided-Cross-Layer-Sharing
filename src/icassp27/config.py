from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_project_dir"] = str(path.parent.parent)
    for key in ("output_root", "cache_root"):
        value = Path(cfg["project"][key])
        if not value.is_absolute():
            value = path.parent.parent / value
        cfg["project"][key] = str(value.resolve())
    return cfg


def model_config(cfg: dict[str, Any], backbone: str) -> dict[str, Any]:
    if backbone not in cfg["models"]:
        raise KeyError(f"Unknown backbone {backbone!r}; choose {list(cfg['models'])}")
    out = copy.deepcopy(cfg["models"][backbone])
    out["name"] = backbone
    return out


def run_dir(cfg: dict[str, Any], backbone: str, *parts: str) -> Path:
    path = Path(cfg["project"]["output_root"]) / backbone
    return path.joinpath(*parts)
