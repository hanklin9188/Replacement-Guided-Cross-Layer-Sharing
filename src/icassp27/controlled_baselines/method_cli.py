from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .method_recovery import evaluate_control, load_method_config, recover


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4xH200 method-specific baseline recovery runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    cfg = load_method_config(args.config)
    spec = json.loads(Path(args.spec).read_text())
    if spec["stage"] in {"recover", "smoke"}:
        output = recover(cfg, spec)
    elif spec["stage"] in {"pure", "dense"}:
        output = evaluate_control(cfg, spec)
    else:
        raise ValueError(spec["stage"])
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
