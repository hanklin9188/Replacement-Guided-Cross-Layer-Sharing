#!/usr/bin/env python3
"""Standard-library integrity checks for the public research artifact."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"PIQA", "SocialIQA", "WinoGrande", "ARC_C", "ARC_E", "HellaSwag", "OpenBookQA"}
BACKBONES = {"Llama-3.2-3B", "Llama-3.1-8B"}
TARGETS = {15, 20, 25}
REGIMES = {"Pure", "CE", "CE+KD"}
TEXT_SUFFIXES = {
    ".bib", ".cff", ".csv", ".html", ".json", ".md", ".py", ".sh",
    ".sbatch", ".tex", ".toml", ".yaml", ".yml",
}
PRIVATE_MARKERS = (
    "/work/hank9188",
    "/home/hank9188",
    "ghp_",
    "github_pat_",
    "BEGIN OPENSSH PRIVATE KEY",
)


class Audit:
    def __init__(self) -> None:
        self.checks = 0
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(message)


def rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float | str, right: float | str, tolerance: float = 1e-10) -> bool:
    a, b = float(left), float(right)
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def verify_required_files(audit: Audit) -> None:
    required = [
        "README.md",
        "README_zh-TW.md",
        "REPRODUCIBILITY.md",
        "LIMITATIONS.md",
        "CITATION.cff",
        "docs/METHOD.md",
        "docs/RESULTS.md",
        "docs/AUDIT.md",
        "docs/FILE_GUIDE.md",
        "paper/main.tex",
        "paper/main.pdf",
        "paper/references.bib",
        "paper/Figure/replace.pdf",
        "paper/Figure/diag_structural_validation.pdf",
        "assets/figures/framework.png",
        "scripts/verify_repository.py",
        "scripts/import_external_baselines.py",
        "scripts/generate_quantization_figure.py",
        "data/ours/main_summary.csv",
        "data/ours/per_task.csv",
        "data/ours/directed_costs.csv",
        "data/ours/pair_analysis.csv",
        "data/ours/group_analysis.csv",
        "data/ours/joint_analysis.csv",
        "data/ours/structural_ablation.csv",
        "data/ours/quantization.csv",
        "data/processed/paper_main_table.csv",
        "data/processed/paired_bootstrap_results.csv",
        "data/processed/paired_bootstrap_task_coverage.csv",
        "data/processed/serialized_byte_reduction_combined.csv",
    ]
    for relative in required:
        path = ROOT / relative
        audit.check(path.is_file() and path.stat().st_size > 0, f"missing or empty: {relative}")


def verify_ours_statistics(audit: Audit) -> None:
    long_rows = rows("data/ours/per_task.csv")
    audit.check(len(long_rows) == 294, "per_task.csv must contain 294 rows")
    runs: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in long_rows:
        key = (row["backbone"], int(row["target"]), row["regime"], row["seed"])
        runs[key].append(float(row["accuracy"]))
        audit.check(row["backbone"] in BACKBONES, f"unknown backbone in per_task: {row['backbone']}")
        audit.check(int(row["target"]) in TARGETS, f"unknown target in per_task: {row['target']}")
        audit.check(row["regime"] in REGIMES, f"unknown regime in per_task: {row['regime']}")
        audit.check(row["task"] in TASKS, f"unknown task in per_task: {row['task']}")
    audit.check(len(runs) == 42, "per_task.csv must define 42 runs")
    audit.check(all(len(values) == 7 for values in runs.values()), "every run must have seven tasks")

    run_macros = {key: statistics.mean(values) for key, values in runs.items()}
    summary = rows("data/ours/main_summary.csv")
    audit.check(len(summary) == 18, "main_summary.csv must contain 18 rows")
    for row in summary:
        backbone, target, regime = row["backbone"], int(row["nominal_target"]), row["regime"]
        values = [value for (b, t, r, _), value in run_macros.items() if (b, t, r) == (backbone, target, regime)]
        expected_count = 1 if regime == "Pure" else 3
        audit.check(len(values) == expected_count, f"wrong seed count: {backbone}/{target}/{regime}")
        audit.check(close(row["mean_macro"], statistics.mean(values)), f"mean mismatch: {backbone}/{target}/{regime}")
        audit.check(close(row["worst_seed_macro"], min(values)), f"minimum mismatch: {backbone}/{target}/{regime}")
        audit.check(close(row["best_seed_macro"], max(values)), f"maximum mismatch: {backbone}/{target}/{regime}")
        if regime == "Pure":
            audit.check(row["sd_macro"] == "NA", f"Pure SD must be NA: {backbone}/{target}")
        else:
            audit.check(close(row["sd_macro"], statistics.stdev(values)), f"SD mismatch: {backbone}/{target}/{regime}")


def verify_structural_data(audit: Audit) -> None:
    directed = rows("data/ours/directed_costs.csv")
    pairs = rows("data/ours/pair_analysis.csv")
    groups = rows("data/ours/group_analysis.csv")
    joint = rows("data/ours/joint_analysis.csv")
    ablations = rows("data/ours/structural_ablation.csv")
    audit.check(len(directed) == 1748, "directed_costs.csv must contain 1,748 rows")
    audit.check(len(pairs) == 874, "pair_analysis.csv must contain 874 rows")
    audit.check(len(groups) == 11, "group_analysis.csv must contain 11 non-singleton groups")
    audit.check(len(joint) == 6, "joint_analysis.csv must contain six operating points")
    audit.check(len(ablations) == 5, "structural_ablation.csv must contain five rows")
    for row in pairs:
        cij, cji = float(row["C_i_to_j"]), float(row["C_j_to_i"])
        audit.check(close(row["mutual_cost"], max(cij, cji)), "pair mutual_cost is not max(direction)")
        audit.check(close(row["asymmetry"], abs(cij - cji)), "pair asymmetry is not absolute difference")
    for row in groups:
        audit.check(float(row["delta"]) <= float(row["Delta"]) + 1e-12, "group violates delta <= Delta")
        audit.check(close(row["envelope_gap"], float(row["Delta"]) - float(row["delta"])), "group envelope gap mismatch")
    audit.check(all(float(row["C_joint"]) > 0 for row in joint), "joint distortion must be observed and positive")


def verify_paper_and_external_tables(audit: Audit) -> None:
    paper = rows("data/processed/paper_main_table.csv")
    audit.check(len(paper) == 18, "paper_main_table.csv must contain 18 rows")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in paper:
        grouped[(row["backbone"], int(row["nominal_target"]))].append(row)
    audit.check(len(grouped) == 6, "paper table must contain six operating points")
    for key, members in grouped.items():
        audit.check({row["method"] for row in members} == {"Ours", "Basis Sharing", "SVD-LLM"}, f"method coverage mismatch: {key}")
        ours = next(row for row in members if row["method"] == "Ours")
        audit.check(float(ours["cekd_mean"]) == max(float(row["cekd_mean"]) for row in members), f"Ours is not best CE+KD: {key}")

    bootstrap = rows("data/processed/paired_bootstrap_results.csv")
    coverage = rows("data/processed/paired_bootstrap_task_coverage.csv")
    serialized = rows("data/processed/serialized_byte_reduction_combined.csv")
    audit.check(len(bootstrap) == 288, "paired bootstrap table must contain 288 rows")
    audit.check(len(coverage) == 252, "paired coverage table must contain 252 rows")
    audit.check(len(serialized) == 18, "serialized-byte table must contain 18 rows")
    macro = [row for row in bootstrap if row["task"] == "macro"]
    audit.check(len(macro) == 36, "paired bootstrap must contain 36 macro comparisons")
    audit.check(all(float(row["ci95_low"]) > 0 for row in macro), "every macro interval must be above zero")
    audit.check(all(
        row["canonical_id_match"] == row["source_index_match"] == row["normalized_gold_match"] == "True"
        and row["ours_n"] == row["other_n"] == row["matched_n"]
        for row in coverage
    ), "paired-example coverage validation failed")

    quant = rows("data/ours/quantization.csv")
    expected_quant = {(model, precision) for model in ("8b_15", "8b_25") for precision in ("bf16", "w8a16", "w4a16")}
    audit.check(len(quant) == 6, "Ours quantization table must contain six rows")
    audit.check({(row["model_id"], row["precision"]) for row in quant} == expected_quant, "Ours quantization coverage mismatch")


def verify_citations_and_links(audit: Audit) -> None:
    tex = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    bib = (ROOT / "paper/references.bib").read_text(encoding="utf-8")
    cited = {key.strip() for block in re.findall(r"\\cite\{([^}]+)\}", tex) for key in block.split(",")}
    defined = set(re.findall(r"@[A-Za-z]+\{\s*([^,\s]+)", bib))
    audit.check(cited <= defined, f"undefined bibliography keys: {sorted(cited - defined)}")

    markdown_link = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for raw in markdown_link.findall(text):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^(?:https?://|mailto:)", target):
                continue
            resolved = (path.parent / target).resolve()
            audit.check(resolved.exists(), f"broken local link: {path.relative_to(ROOT)} -> {raw}")


def verify_public_hygiene(audit: Audit) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in PRIVATE_MARKERS:
            audit.check(marker not in text, f"private marker {marker!r} in {path.relative_to(ROOT)}")
    symlinks = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_symlink()]
    audit.check(not symlinks, f"public release must not contain symlinks: {symlinks}")


def main() -> None:
    audit = Audit()
    verify_required_files(audit)
    if not audit.failures:
        verify_ours_statistics(audit)
        verify_structural_data(audit)
        verify_paper_and_external_tables(audit)
        verify_citations_and_links(audit)
        verify_public_hygiene(audit)
    report = {
        "status": "PASS" if not audit.failures else "FAIL",
        "checks": audit.checks,
        "failures": audit.failures,
    }
    print(json.dumps(report, indent=2))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
