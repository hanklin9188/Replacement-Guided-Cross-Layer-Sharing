from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .config import rate_label, result_dir


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) != 3:
        raise RuntimeError(f"expected exactly three seeds, found {len(values)}")
    return statistics.mean(values), statistics.stdev(values)


def aggregate(cfg: dict[str, Any]) -> Path:
    report_root = Path(cfg["paths"]["output_root"]) / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    long_rows = []
    missing = []
    for method in cfg["matrix"]["methods"]:
        for backbone in cfg["matrix"]["backbones"]:
            for reduction in cfg["matrix"]["target_reductions"]:
                pure_path = result_dir(cfg, method, backbone, reduction, "pure") / "summary.json"
                if not pure_path.is_file():
                    missing.append(str(pure_path))
                else:
                    long_rows.append(json.loads(pure_path.read_text()))
                for objective in cfg["matrix"]["objectives"]:
                    for seed in cfg["matrix"]["seeds"]:
                        path = result_dir(cfg, method, backbone, reduction, objective, seed) / "summary.json"
                        if not path.is_file():
                            missing.append(str(path))
                        else:
                            long_rows.append(json.loads(path.read_text()))
    if missing:
        (report_root / "MISSING_RESULTS.json").write_text(json.dumps(missing, indent=2) + "\n")
        raise FileNotFoundError(f"cannot aggregate: {len(missing)} summaries are missing")

    fieldnames = sorted({key for row in long_rows for key in row})
    with (report_root / "comparison_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(long_rows)

    tasks = [(task, "social_iqa" if task == "social_i_qa" else task)
             for task in cfg["evaluation"]["tasks"]]
    lines = ["# Controlled Basis Sharing / SVD-LLM results", ""]
    summary_rows = []
    for backbone in cfg["matrix"]["backbones"]:
        lines.extend([f"## {backbone}", ""])
        for reduction in cfg["matrix"]["target_reductions"]:
            lines.extend([f"### {rate_label(reduction)}", "",
                          "| Method | Actual reduction | Pure | CE mean ± SD | CE+KD mean ± SD | LoRA params |",
                          "|---|---:|---:|---:|---:|---:|"])
            for method in cfg["matrix"]["methods"]:
                selected = [row for row in long_rows if row["method"] == method and
                            row["backbone"] == backbone and row["target_reduction"] == reduction]
                pure = next(row for row in selected if row["objective"] == "pure")
                ce = [row for row in selected if row["objective"] == "ce"]
                kd = [row for row in selected if row["objective"] == "ce_kd"]
                ce_mean, ce_std = _mean_std([row["macro"] for row in ce])
                kd_mean, kd_std = _mean_std([row["macro"] for row in kd])
                adapter = {row["adapter_parameters"] for row in ce + kd}
                if len(adapter) != 1:
                    raise RuntimeError(f"adapter parameter mismatch for {method}/{backbone}/{reduction}")
                lines.append(f"| {method} | {pure['actual_raw_reduction']:.4f} | {pure['macro']:.4f} | "
                             f"{ce_mean:.4f} ± {ce_std:.4f} | {kd_mean:.4f} ± {kd_std:.4f} | "
                             f"{next(iter(adapter)):,} |")
                summary_rows.append({"method": method, "backbone": backbone,
                                     "target_reduction": reduction,
                                     "actual_reduction": pure["actual_raw_reduction"],
                                     "pure_macro": pure["macro"], "ce_mean": ce_mean, "ce_std": ce_std,
                                     "ce_kd_mean": kd_mean, "ce_kd_std": kd_std,
                                     "lora_parameters": next(iter(adapter))})
            lines.append("")
    lines.extend(["## Per-task seed results", ""])
    for row in long_rows:
        values = " | ".join(f"{row[key]:.4f}" for _, key in tasks)
        lines.append(f"- {row['method']} / {row['backbone']} / {rate_label(row['target_reduction'])} / "
                     f"{row['objective']} / seed={row['seed']}: {values}; macro={row['macro']:.4f}")
    (report_root / "CONTROLLED_BASELINE_RESULTS.md").write_text("\n".join(lines) + "\n")
    (report_root / "comparison_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    return report_root
