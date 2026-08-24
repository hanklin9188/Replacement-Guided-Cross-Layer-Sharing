#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT / "src"))

from icassp27.controlled_baselines.config import raw_checkpoint_dir, sha256_file
from icassp27.controlled_baselines.data import load_recovery_rows, sample_id_hash
from icassp27.controlled_baselines.method_recovery import load_method_config


def main() -> None:
    config = Path(sys.argv[1] if len(sys.argv) > 1 else CODE_ROOT / "configs/method_recovery_4h200.example.yaml")
    cfg = load_method_config(config)
    checks: list[dict] = []

    def check(name: str, condition: bool, detail) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    canonical_train_hashes = set()
    canonical_val_hashes = set()
    cache_rows = None
    cache_sources = None
    for backbone, value in cfg["backbones"].items():
        base, teacher = Path(value["model_path"]), Path(value["teacher_path"])
        check(f"{backbone}.base", (base / "config.json").is_file(), str(base))
        check(f"{backbone}.teacher", (teacher / "model.safetensors").is_file(), str(teacher))
        train, validation = Path(value["recovery_train"]), Path(value["recovery_validation"])
        train_rows, val_rows = load_recovery_rows(train), load_recovery_rows(validation)
        if cache_rows is None:
            cache_rows = (train_rows, val_rows)
            cache_sources = (sha256_file(train), sha256_file(validation))
        overlap = {row["id"] for row in train_rows} & {row["id"] for row in val_rows}
        train_hash, val_hash = sample_id_hash(train_rows), sample_id_hash(val_rows)
        canonical_train_hashes.add(train_hash); canonical_val_hashes.add(val_hash)
        check(f"{backbone}.recovery_split", not overlap and len(train_rows) == 161899 and len(val_rows) == 8521,
              {"train": len(train_rows), "validation": len(val_rows), "overlap": len(overlap),
               "train_ids_sha256": train_hash, "validation_ids_sha256": val_hash})
        final = [Path(value["evaluation_data_root"]) / f"{task}.final.jsonl"
                 for task in cfg["evaluation"]["tasks"]]
        check(f"{backbone}.evaluation", all(path.is_file() for path in final), [str(path) for path in final])
        try:
            # The merged teachers were saved by a newer Transformers release and
            # name their wrapper ``TokenizersBackend``.  Token IDs are what KD
            # needs to share, so compare the underlying tokenizer.json maps
            # directly instead of depending on that version-specific wrapper.
            from tokenizers import Tokenizer
            base_tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
            teacher_tokenizer = Tokenizer.from_file(str(teacher / "tokenizer.json"))
            same_vocab = base_tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
            check(f"{backbone}.tokenizer_vocab", same_vocab,
                  {"base_size": base_tokenizer.get_vocab_size(with_added_tokens=True),
                   "teacher_size": teacher_tokenizer.get_vocab_size(with_added_tokens=True),
                   "comparison": "token_to_id_map"})
        except Exception as error:
            check(f"{backbone}.tokenizer_vocab", False, repr(error))

    check("canonical_train_ids_across_backbones", len(canonical_train_hashes) == 1,
          sorted(canonical_train_hashes))
    check("canonical_validation_ids_across_backbones", len(canonical_val_hashes) == 1,
          sorted(canonical_val_hashes))

    cache = Path(cfg["paths"]["output_root"]) / "data_cache/recovery_rows.pt"
    if cache_rows is not None and cache_sources is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(f".pt.tmp.{os.getpid()}")
        torch.save({"train_rows": cache_rows[0], "validation_rows": cache_rows[1],
                    "train_source_sha256": cache_sources[0],
                    "validation_source_sha256": cache_sources[1]}, temporary)
        temporary.replace(cache)
        check("recovery_cache", cache.is_file(), {"path": str(cache), "bytes": cache.stat().st_size})

    for method in cfg["matrix"]["methods"]:
        for backbone in cfg["matrix"]["backbones"]:
            expected_revision = cfg["backbones"][backbone]["revision"]
            expected_model = cfg["backbones"][backbone]["model_id"]
            for reduction in cfg["matrix"]["target_reductions"]:
                root = raw_checkpoint_dir(cfg, method, backbone, float(reduction))
                required = [".complete", "compressed_state.pt", "compression_structure.json",
                            "compression_report.json", "parameter_report.json", "model_loader.py"]
                missing = [name for name in required if not (root / name).is_file()]
                if missing:
                    check(f"raw.{method}.{backbone}.{reduction}", False,
                          {"path": str(root), "missing": missing})
                    continue
                report = json.loads((root / "compression_report.json").read_text())
                structure = json.loads((root / "compression_structure.json").read_text())
                actual = float(report["actual_reduction"])
                correct_source = report["base_model"] == expected_model and report["base_revision"] == expected_revision
                correct_structure = structure["method"] == method
                no_fad = "shared_student" not in json.dumps({"report": report, "structure": structure})
                check(f"raw.{method}.{backbone}.{reduction}",
                      correct_source and correct_structure and no_fad and abs(actual - float(reduction)) < 2e-5,
                      {"path": str(root), "actual_reduction": actual, "base_model": report["base_model"],
                       "base_revision": report["base_revision"], "structure_method": structure["method"],
                       "compression_report_sha256": sha256_file(root / "compression_report.json")})

    usage = shutil.disk_usage(Path(cfg["paths"]["output_root"]).parent)
    check("storage_free", usage.free > 2 * 1024**4, {"free_bytes": usage.free})
    check("four_h200", int(cfg["slurm"]["gpus_per_job"]) == 4, cfg["slurm"])
    report = {"status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
              "config": str(config.resolve()), "config_sha256": cfg["_config_sha256"], "checks": checks}
    output = Path(cfg["paths"]["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "PREFLIGHT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
