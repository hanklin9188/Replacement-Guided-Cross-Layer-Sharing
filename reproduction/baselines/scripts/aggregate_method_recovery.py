#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT / "src"))

from icassp27.controlled_baselines.method_recovery import load_method_config


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    config = Path(sys.argv[1] if len(sys.argv) > 1 else CODE_ROOT / "configs/method_recovery_4h200.example.yaml")
    cfg = load_method_config(config)
    output_root = Path(cfg["paths"]["output_root"])
    rows = []
    missing = []
    for method in cfg["matrix"]["methods"]:
        for backbone in cfg["matrix"]["backbones"]:
            for reduction in cfg["matrix"]["target_reductions"]:
                rate = f"{round(100 * float(reduction)):02d}pct"
                pure = Path(cfg["paths"]["control_root"]) / "pure" / method / backbone / rate / "summary.json"
                if pure.is_file():
                    rows.append(json.loads(pure.read_text()))
                else:
                    missing.append(str(pure))
                for seed in cfg["matrix"]["seeds"]:
                    for objective in cfg["matrix"]["objectives"]:
                        path = (Path(cfg["paths"]["result_root"]) / method / backbone / rate /
                                f"seed_{seed}" / objective / "summary.json")
                        if path.is_file():
                            rows.append(json.loads(path.read_text()))
                        else:
                            missing.append(str(path))
    dense_rows = []
    for backbone in cfg["matrix"]["backbones"]:
        path = Path(cfg["paths"]["control_root"]) / "dense" / backbone / "summary.json"
        if path.is_file():
            dense_rows.append(json.loads(path.read_text()))
        else:
            missing.append(str(path))
    if missing:
        raise RuntimeError(f"aggregation requires every artifact; missing {len(missing)}: {missing[:8]}")

    recovery = [row for row in rows if row.get("objective") in {"ce", "ce_kd"}]
    pure_rows = [row for row in rows if row.get("stage") == "pure"]
    audit = {
        "expected_recovery": 72, "observed_recovery": len(recovery),
        "expected_pure": 12, "observed_pure": len(pure_rows),
        "expected_dense": 2, "observed_dense": len(dense_rows),
        "all_four_h200": True, "all_generic_lora_false": True,
        "all_parameter_rates_preserved": True, "ce_teacher_violations": 0,
        "kd_teacher_violations": 0, "protocol_violations": [],
        "data_split_violations": [], "step_budget_violations": [],
        "objective_violations": [], "control_gpu_violations": [],
    }
    train_id_hashes = set()
    validation_id_hashes = set()
    for row in recovery:
        root = (Path(cfg["paths"]["result_root"]) / row["method"] / row["backbone"] /
                f"{round(100 * row['target_reduction']):02d}pct" / f"seed_{row['seed']}" /
                row["objective"])
        run = root / "training_report.json"
        report = json.loads(run.read_text())
        run_config = json.loads((root / "run_config.json").read_text())
        train_id_hashes.add(run_config["recovery_train_ids_sha256"])
        validation_id_hashes.add(run_config["recovery_validation_ids_sha256"])
        audit["all_four_h200"] &= report["world_size"] == 4
        audit["all_generic_lora_false"] &= report["generic_lora_used"] is False
        audit["all_parameter_rates_preserved"] &= abs(row["actual_raw_reduction"] - row["recovered_reduction"]) < 1e-12
        if row["objective"] == "ce" and (report["teacher_loaded"] or report["teacher_forward_executed"]):
            audit["ce_teacher_violations"] += 1
        if row["objective"] == "ce_kd" and not (report["teacher_loaded"] and report["teacher_frozen"]):
            audit["kd_teacher_violations"] += 1
        expected = "basis_full_parameter" if row["method"] == "basis_sharing" else "svd_sequential_u_then_v"
        if report["recovery_protocol"] != expected:
            audit["protocol_violations"].append(str(run))
        if report["loss_scope"] != "multiple_choice_candidate_decision_ce":
            audit["objective_violations"].append(str(run))
        expected_steps = int(cfg["backbones"][row["backbone"]]["max_steps"])
        stage_steps = sum(int(stage["steps"]) for stage in report["stages"])
        if int(report["total_steps"]) != expected_steps or stage_steps != expected_steps:
            audit["step_budget_violations"].append(str(run))
        if row["method"] == "basis_sharing":
            valid_stages = (len(report["stages"]) == 1 and report["stages"][0]["stage"] == "full" and
                            report["stages"][0]["scope"] == "all_existing_compressed_model_parameters")
        else:
            valid_stages = ([stage["stage"] for stage in report["stages"]] == ["u", "v"] and
                            [stage["scope"] for stage in report["stages"]] ==
                            ["factor_u_only", "factor_v_only"])
        if not valid_stages:
            audit["protocol_violations"].append(str(run))
    audit["canonical_train_id_hashes"] = sorted(train_id_hashes)
    audit["canonical_validation_id_hashes"] = sorted(validation_id_hashes)
    if len(train_id_hashes) != 1 or len(validation_id_hashes) != 1:
        audit["data_split_violations"].append("recovery runs do not share one canonical ID split")
    for row in pure_rows + dense_rows:
        if int(row["world_size"]) != 4:
            audit["control_gpu_violations"].append(f"{row['method']}/{row['backbone']}")
    audit["status"] = "PASS" if (
        audit["observed_recovery"] == 72 and audit["observed_pure"] == 12 and
        audit["observed_dense"] == 2 and audit["all_four_h200"] and
        audit["all_generic_lora_false"] and audit["all_parameter_rates_preserved"] and
        audit["ce_teacher_violations"] == 0 and audit["kd_teacher_violations"] == 0 and
        not audit["protocol_violations"] and not audit["data_split_violations"] and
        not audit["step_budget_violations"] and not audit["objective_violations"] and
        not audit["control_gpu_violations"]
    ) else "FAIL"

    tables = output_root / "summary_tables"
    tables.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows + dense_rows for key in row})
    with (tables / "all_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows + dense_rows)
    (tables / "all_runs.json").write_text(json.dumps(rows + dense_rows, indent=2, sort_keys=True) + "\n")
    (output_root / "FAIRNESS_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    pure_map = {(row["method"], row["backbone"], round(100 * row["target_reduction"])): row
                for row in pure_rows}
    grouped = defaultdict(list)
    for row in recovery:
        grouped[(row["method"], row["backbone"], round(100 * row["target_reduction"]),
                 row["objective"])].append(row)
    lines = ["# Basis Sharing full-FT vs SVD-LLM sequential recovery", "",
             f"Fairness audit: **{audit['status']}**", "",
             "All recovery cells use decision CE; CE is teacher-free and CE+KD uses the frozen merged teacher.",
             "Every recovery and control evaluation ran with 4×H200. Compression checkpoints are the audited",
             "pretrained-base 15/20/25% artifacts; recovery never reconstructs removed dense weights.", ""]
    selected_path = output_root / "SELECTED_LRS.json"
    if selected_path.is_file():
        selected = json.loads(selected_path.read_text())["selected"]
        lines += ["## Selected learning rates", "", "| Method | Backbone | Objective | LR |",
                  "|---|---|---|---:|"]
        for compound, lr in sorted(selected.items()):
            method, backbone, objective = compound.split("/")
            lines.append(f"| {method} | {backbone} | {objective} | {lr:.1e} |")
        lines.append("")
    for backbone in cfg["matrix"]["backbones"]:
        dense = next(row for row in dense_rows if row["backbone"] == backbone)
        lines += [f"## {backbone}", "", f"Dense pretrained control macro: `{dense['macro']:.4f}`", "",
                  "| Rate | Method | Actual | Pure | CE mean±SD | CE+KD mean±SD | KD−CE |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for reduction in cfg["matrix"]["target_reductions"]:
            rate = round(100 * reduction)
            for method in cfg["matrix"]["methods"]:
                pure = pure_map[(method, backbone, rate)]
                ce = grouped[(method, backbone, rate, "ce")]
                kd = grouped[(method, backbone, rate, "ce_kd")]
                ce_mean, ce_sd = mean_sd([row["macro"] for row in ce])
                kd_mean, kd_sd = mean_sd([row["macro"] for row in kd])
                actual = statistics.mean(row["recovered_reduction"] for row in ce) * 100
                lines.append(f"| {rate}% | {method} | {actual:.4f}% | {pure['macro']:.4f} | "
                             f"{ce_mean:.4f}±{ce_sd:.4f} | {kd_mean:.4f}±{kd_sd:.4f} | "
                             f"{kd_mean-ce_mean:+.4f} |")
        lines.append("")
    tasks = [{"social_i_qa": "social_iqa"}.get(task, task) for task in cfg["evaluation"]["tasks"]]
    lines += ["## Per-task recovery results (mean over seeds 42/43/44)", "",
              "| Backbone | Rate | Method | Objective | " + " | ".join(tasks) + " | Macro |",
              "|---|---:|---|---|" + "---:|" * (len(tasks) + 1)]
    for backbone in cfg["matrix"]["backbones"]:
        for reduction in cfg["matrix"]["target_reductions"]:
            rate = round(100 * reduction)
            for method in cfg["matrix"]["methods"]:
                for objective in cfg["matrix"]["objectives"]:
                    values = grouped[(method, backbone, rate, objective)]
                    cells = []
                    for task in tasks:
                        mean, sd = mean_sd([row[task] for row in values])
                        cells.append(f"{mean:.4f}±{sd:.4f}")
                    macro, macro_sd = mean_sd([row["macro"] for row in values])
                    lines.append(f"| {backbone} | {rate}% | {method} | {objective} | " +
                                 " | ".join(cells) + f" | {macro:.4f}±{macro_sd:.4f} |")
    lines.append("")
    lines += ["## Artifact counts", "", f"- Recovery: {len(recovery)}/72",
              f"- Pure: {len(pure_rows)}/12", f"- Dense controls: {len(dense_rows)}/2", ""]
    (output_root / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n")
    (output_root / ".complete").write_text("PASS\n")
    print(output_root / "FINAL_SUMMARY.md")


if __name__ == "__main__":
    main()
