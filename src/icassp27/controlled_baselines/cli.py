from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .compress import compress
from .config import load_config, raw_checkpoint_dir, sha256_file
from .data import load_recovery_rows, sample_id_hash
from .report import aggregate
from .train_eval import pure_evaluate, recover_and_evaluate


def _preflight(cfg: dict[str, Any], require_raw: bool) -> dict[str, Any]:
    checks = []
    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    for method, value in cfg["methods"].items():
        source = Path(value["source_path"])
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                text=True, capture_output=True)
        observed = result.stdout.strip() if result.returncode == 0 else None
        check(f"{method}.upstream_commit", observed == value["commit"],
              {"source": str(source), "expected": value["commit"], "observed": observed})

    for backbone, value in cfg["backbones"].items():
        model = Path(value["model_path"])
        check(f"{backbone}.base_config", (model / "config.json").is_file(), str(model / "config.json"))
        check(f"{backbone}.base_tokenizer", (model / "tokenizer.json").is_file(), str(model / "tokenizer.json"))
        weights = list(model.glob("*.safetensors"))
        check(f"{backbone}.base_weights", bool(weights), [str(path) for path in weights])
        teacher = Path(value["teacher_path"])
        teacher_required = [teacher / "config.json", teacher / "tokenizer.json", teacher / "model.safetensors"]
        check(f"{backbone}.teacher", all(path.is_file() for path in teacher_required),
              [str(path) for path in teacher_required])
        if (model / "config.json").is_file() and (teacher / "config.json").is_file():
            base_config = json.loads((model / "config.json").read_text())
            teacher_config = json.loads((teacher / "config.json").read_text())
            fields = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                      "num_key_value_heads", "vocab_size"]
            differences = {field: [base_config.get(field), teacher_config.get(field)] for field in fields
                           if base_config.get(field) != teacher_config.get(field)}
            check(f"{backbone}.teacher_architecture", not differences, differences)
        calibration = Path(value["calibration_blocks"])
        check(f"{backbone}.calibration", calibration.is_file(), str(calibration))
        evaluation_root = Path(value["evaluation_data_root"])
        final_files = [evaluation_root / f"{task}.final.jsonl" for task in cfg["evaluation"]["tasks"]]
        check(f"{backbone}.seven_task_data", all(path.is_file() for path in final_files),
              [str(path) for path in final_files])
        train_path, val_path = Path(value["recovery_train"]), Path(value["recovery_validation"])
        check(f"{backbone}.recovery_train", train_path.is_file(), str(train_path))
        check(f"{backbone}.recovery_validation", val_path.is_file(), str(val_path))
        if train_path.is_file() and val_path.is_file():
            train_rows, val_rows = load_recovery_rows(train_path), load_recovery_rows(val_path)
            overlap = {row["id"] for row in train_rows} & {row["id"] for row in val_rows}
            check(f"{backbone}.split_isolation", not overlap, {"overlap": len(overlap)})
            checks.append({"name": f"{backbone}.data_evidence", "status": "PASS", "detail": {
                "train_records": len(train_rows), "validation_records": len(val_rows),
                "train_sha256": sha256_file(train_path), "validation_sha256": sha256_file(val_path),
                "train_ids_sha256": sample_id_hash(train_rows), "validation_ids_sha256": sample_id_hash(val_rows),
            }})
    if require_raw:
        for method in cfg["matrix"]["methods"]:
            for backbone in cfg["matrix"]["backbones"]:
                for reduction in cfg["matrix"]["target_reductions"]:
                    root = raw_checkpoint_dir(cfg, method, backbone, reduction)
                    required = ["config.json", "compressed_state.pt", "compression_structure.json",
                                "compression_report.json", "parameter_report.json", "model_loader.py", ".complete"]
                    missing = [name for name in required if not (root / name).is_file()]
                    check(f"raw.{method}.{backbone}.{reduction}", not missing,
                          {"path": str(root), "missing": missing})
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {"status": status, "config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
            "checks": checks}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled Basis Sharing and SVD-LLM experiment runner")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[3] /
                                                  "configs/controlled_baselines.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--require-raw", action="store_true")
    run = sub.add_parser("run-spec")
    run.add_argument("--spec", required=True)
    sub.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    cfg = load_config(args.config)
    if args.command == "preflight":
        report = _preflight(cfg, args.require_raw)
        output = Path(cfg["paths"]["output_root"]) / "PREFLIGHT.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(2)
    elif args.command == "run-spec":
        spec = json.loads(Path(args.spec).read_text())
        stage = spec["stage"]
        if stage == "compress":
            output = compress(cfg, spec["method"], spec["backbone"], float(spec["target_reduction"]))
        elif stage == "pure":
            output = pure_evaluate(cfg, spec["method"], spec["backbone"], float(spec["target_reduction"]))
        elif stage == "recover":
            output = recover_and_evaluate(cfg, spec["method"], spec["backbone"],
                                          float(spec["target_reduction"]), spec["objective"], int(spec["seed"]))
        else:
            raise ValueError(stage)
        if not torch_distributed_worker():
            print(json.dumps({"status": "PASS", "output": str(output)}))
    elif args.command == "aggregate":
        print(aggregate(cfg))


def torch_distributed_worker() -> bool:
    import os
    return int(os.environ.get("RANK", "0")) != 0


if __name__ == "__main__":
    main()
