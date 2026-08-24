#!/usr/bin/env python3
"""Single-panel 15%/25% quantization storage--accuracy comparison."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/quantization/storage_accuracy.csv"
OUTPUT = ROOT / "assets/figures/quantization"
PAPER_OUTPUT = ROOT / "paper/Figure/fig_quantization_15_25_storage_accuracy.png"

METHODS = ["Dense teacher", "FAD (Ours)", "Basis Sharing", "SVD-LLM"]
COMPRESSED_METHODS = ["FAD (Ours)", "Basis Sharing", "SVD-LLM"]
PRECISIONS = ["BF16", "INT8", "INT4"]
COLORS = {
    "Dense teacher": "#555555",
    "FAD (Ours)": "#0072B2",
    "Basis Sharing": "#D55E00",
    "SVD-LLM": "#009E73",
}
MARKERS = {"BF16": "o", "INT8": "s", "INT4": "^"}
LINESTYLES = {15: "-", 25: (0, (4.0, 2.2))}
DISPLAY_NAMES = {"FAD (Ours)": "Ours"}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def plot_trajectory(ax: plt.Axes, cell: pd.DataFrame, method: str,
                    linestyle, linewidth: float, alpha: float) -> None:
    cell = cell.set_index("precision").loc[PRECISIONS]
    ax.plot(cell.checkpoint_gib, cell.macro_accuracy, color=COLORS[method],
            linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=2)
    for precision, row in cell.iterrows():
        face = "white" if precision == "INT8" else COLORS[method]
        ax.scatter(row.checkpoint_gib, row.macro_accuracy, s=43,
                   marker=MARKERS[precision], facecolor=face,
                   edgecolor=COLORS[method], linewidth=1.15,
                   alpha=alpha, zorder=4)


def main() -> None:
    configure()
    df = pd.read_csv(DATA)
    fig, ax = plt.subplots(figsize=(5.20, 3.65))

    dense = df[df.method.eq("Dense teacher")]
    plot_trajectory(ax, dense, "Dense teacher", "-", 1.9, 0.95)
    for method in COMPRESSED_METHODS:
        for ratio in (15, 25):
            cell = df[(df.method == method) & (df.compression_pct == ratio)]
            plot_trajectory(ax, cell, method, LINESTYLES[ratio],
                            1.8 if ratio == 15 else 1.65,
                            0.92 if ratio == 15 else 1.0)

    # Direct labels identify the two structural-compression trajectories.
    label_offsets = {
        ("FAD (Ours)", 15): (5, 4), ("FAD (Ours)", 25): (5, -12),
        ("Basis Sharing", 15): (5, 5), ("Basis Sharing", 25): (5, -12),
        ("SVD-LLM", 15): (5, -13), ("SVD-LLM", 25): (5, 5),
    }
    for method in COMPRESSED_METHODS:
        for ratio in (15, 25):
            row = df[(df.method == method) & (df.compression_pct == ratio) &
                     (df.precision == "BF16")].iloc[0]
            dx, dy = label_offsets[(method, ratio)]
            ax.annotate(f"{ratio}%", (row.checkpoint_gib, row.macro_accuracy),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.7, color=COLORS[method], fontweight="bold")

    ax.set_xlim(3.75, 15.65)
    ax.set_ylim(0.32, 0.905)
    ax.set_xticks([4, 6, 8, 10, 12, 14])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Serialized model size (GiB)")
    ax.set_ylabel("Seven-task Macro accuracy")
    ax.grid(axis="both", color="#D8D8D8", linewidth=0.55, alpha=0.82, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    method_handles = [
        Line2D([0], [0], color=COLORS[m], marker="o", markersize=4.4,
               linewidth=1.65, label=DISPLAY_NAMES.get(m, m)) for m in METHODS
    ]
    precision_handles = [
        Line2D([0], [0], color="#333333", marker=MARKERS[p], linestyle="None",
               markerfacecolor=("white" if p == "INT8" else "#333333"),
               markersize=5.1, label=p) for p in PRECISIONS
    ]
    ratio_handles = [
        Line2D([0], [0], color="#333333", linestyle=LINESTYLES[r],
               linewidth=1.7, label=f"{r}% structural") for r in (15, 25)
    ]
    method_legend = ax.legend(handles=method_handles, loc="lower right",
                              bbox_to_anchor=(1.0, 0.29), frameon=False,
                              handlelength=1.8, labelspacing=0.38)
    ax.add_artist(method_legend)
    precision_legend = ax.legend(handles=precision_handles, loc="lower right",
                                 bbox_to_anchor=(1.0, 0.17), frameon=False,
                                 ncol=3, columnspacing=0.8, handletextpad=0.35)
    ax.add_artist(precision_legend)
    ax.legend(handles=ratio_handles, loc="lower right",
              bbox_to_anchor=(1.0, 0.035), frameon=False,
              ncol=2, columnspacing=0.9, handletextpad=0.45)

    fig.subplots_adjust(left=0.14, right=0.985, top=0.975, bottom=0.16)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight", pad_inches=0.035)
    svg_path = OUTPUT.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUTPUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.035, dpi=600)
    plt.close(fig)
    PAPER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT.with_suffix(".png"), PAPER_OUTPUT)
    print(f"Wrote {OUTPUT}.pdf/.svg/.png")
    print(f"Wrote {PAPER_OUTPUT}")


if __name__ == "__main__":
    main()
