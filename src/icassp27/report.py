from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .config import run_dir
from .utils import atomic_json, read_json


def replacement_plots(cfg: dict[str, Any], backbone: str, valid_tokens: int, k: int, policy: str = "full") -> None:
    source = run_dir(cfg, backbone, "replacement", f"tokens_{valid_tokens}")
    group_path = run_dir(cfg, backbone, "groups", f"tokens_{valid_tokens}", f"k_{k}", f"{policy}.json")
    groups = read_json(group_path)
    directed = np.load(source / "directed_cost.npy")
    pair = np.maximum(directed, directed.T)
    n = directed.shape[0]
    upper = np.triu_indices(n, 1)
    x = directed[upper]
    y = directed.T[upper]
    asymmetry = np.abs(x - y)
    output = run_dir(cfg, backbone, "analysis", f"tokens_{valid_tokens}", f"k_{k}")
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.1), constrained_layout=True)
    axes[0].scatter(x, y, s=9, alpha=0.65)
    limits = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    axes[0].plot(limits, limits, "k--", lw=1)
    axes[0].set(xlabel=r"$C_{i\to j}$", ylabel=r"$C_{j\to i}$", title="Directed asymmetry")
    axes[1].hist(asymmetry, bins=24, color="#4C78A8")
    axes[1].set(xlabel=r"$|C_{i\to j}-C_{j\to i}|$", ylabel="Pairs", title="Asymmetry distribution")
    sns.heatmap(directed, ax=axes[2], cmap="mako", cbar=False)
    axes[2].set(xlabel="Target layer j", ylabel="Source layer i", title="Directed cost")
    order = [layer for group in groups["groups"] for layer in group]
    reordered = pair[np.ix_(order, order)]
    sns.heatmap(reordered, ax=axes[3], cmap="rocket", cbar=True)
    offset = 0
    for group, rep in zip(groups["groups"], groups["representatives"]):
        axes[3].add_patch(plt.Rectangle((offset, offset), len(group), len(group), fill=False, ec="cyan", lw=1.2))
        axes[3].text(offset + len(group) / 2, offset + len(group) / 2, str(rep + 1), color="white",
                     ha="center", va="center", fontsize=7, fontweight="bold")
        offset += len(group)
    axes[3].set(xlabel="Target (group order)", ylabel="Source (group order)", title="Max cost; rep labels")
    summary = {
        "backbone": backbone, "valid_tokens": valid_tokens, "k": k,
        "asymmetry_median": float(np.median(asymmetry)),
        "asymmetry_q75": float(np.percentile(asymmetry, 75)),
        "asymmetry_q95": float(np.percentile(asymmetry, 95)),
        "directed_spearman": float(spearmanr(x, y).statistic),
        **groups["diagnostics"],
    }
    fig.suptitle(f"{backbone}: replacement diagnostics, K={k}", fontsize=11)
    fig.savefig(output / "replacement_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(output / "replacement_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    atomic_json(output / "replacement_diagnostics.json", summary)
    pd.DataFrame(groups["diagnostics"]["per_group"]).to_csv(output / "group_diagnostics.csv", index=False)


def aggregate_report(cfg: dict[str, Any]) -> None:
    root = Path(cfg["project"]["output_root"])
    report = root / "report"
    (report / "figures").mkdir(parents=True, exist_ok=True)
    (report / "tables").mkdir(parents=True, exist_ok=True)
    rows = []
    audit = []
    for path in root.glob("**/metrics.json"):
        try:
            item = read_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if item.get("stage") != "evaluation":
            continue
        metric = item.get("metrics", {})
        row = {"path": str(path), "backbone": item.get("backbone"), "role": item.get("role"),
               "policy": item.get("policy"), "k": item.get("k"),
               "variant": item.get("recovery", {}).get("variant"),
               "seed": item.get("recovery", {}).get("seed"),
               "projection_rank": item.get("recovery", {}).get("projection_rank"),
               "lambda_align": item.get("recovery", {}).get("lambda_align"),
               "macro_accuracy": metric.get("macro_accuracy"), "weighted_accuracy": metric.get("weighted_accuracy"),
               "nll": item.get("heldout_nll", {}).get("nll"),
               "perplexity": item.get("heldout_nll", {}).get("perplexity"),
               "total_parameter_reduction": item.get("parameter_accounting", {}).get("total_parameter_reduction_including_adapters"),
               "checkpoint_gib": (item.get("parameter_accounting", {}).get("compact_checkpoint_bytes") or np.nan) / (1024 ** 3)}
        rows.append(row)
        required = ["model_revision", "tokenizer_revision", "sample_ids", "heldout_nll"]
        missing = [field for field in required if item.get(field) in (None, [], {})]
        audit.append({"path": str(path), "passes": not missing, "missing": ";".join(missing)})
    frame = pd.DataFrame(rows)
    frame.to_csv(report / "metrics_long.csv", index=False)
    pd.DataFrame(audit).to_csv(report / "audit_gate.csv", index=False)
    if not frame.empty:
        frame.to_csv(report / "tables" / "all_evaluations.csv", index=False)
        step0 = frame[frame["role"] == "step0"]
        final = frame[frame["role"] == "student"]
        if not final.empty:
            summary = final.groupby(["backbone", "k", "policy", "variant"], dropna=False).agg(
                final_macro_mean=("macro_accuracy", "mean"),
                final_macro_std=("macro_accuracy", "std"),
                final_nll_mean=("nll", "mean"),
                seed_count=("seed", "nunique"),
            ).reset_index()
            if not step0.empty:
                structural = step0.groupby(["backbone", "k", "policy"], dropna=False).agg(
                    step0_macro=("macro_accuracy", "mean"), step0_nll=("nll", "mean")
                ).reset_index()
                summary = summary.merge(structural, on=["backbone", "k", "policy"], how="left")
            summary.to_csv(report / "tables" / "matched_budget_summary.csv", index=False)
            summary.to_latex(report / "tables" / "matched_budget_summary.tex", index=False,
                             float_format=lambda value: f"{value:.4f}", na_rep="--")
        budget = frame[(frame["role"] == "student") & frame["k"].notna()].copy()
        if not budget.empty:
            fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
            for backbone, group in budget.groupby("backbone"):
                group = group.sort_values("k")
                axes[0].plot(100 * group["total_parameter_reduction"], group["macro_accuracy"], marker="o", label=backbone)
                axes[1].plot(100 * group["total_parameter_reduction"], group["perplexity"], marker="o", label=backbone)
            axes[0].set(xlabel="Total parameter reduction (%)", ylabel="Macro accuracy", title="Budget–accuracy")
            axes[1].set(xlabel="Total parameter reduction (%)", ylabel="Held-out perplexity", title="Budget–NLL")
            axes[0].legend(); axes[1].legend()
            fig.savefig(report / "figures" / "budget_curves.pdf", bbox_inches="tight")
            fig.savefig(report / "figures" / "budget_curves.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
    atomic_json(report / "report_manifest.json", {"evaluation_count": len(rows),
                                                   "audit_pass_count": sum(row["passes"] for row in audit),
                                                   "fabricated_values": False})
