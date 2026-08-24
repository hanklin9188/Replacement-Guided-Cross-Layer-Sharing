#!/usr/bin/env python3
"""Create the paper quantization figure only from complete observed inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD_FILES = {
    "Ours": ROOT / "data" / "ours" / "quantization.csv",
    "Basis Sharing": ROOT / "data" / "external" / "basis_sharing" / "incoming" / "quantization.csv",
    "SVD-LLM": ROOT / "data" / "external" / "svd_llm" / "incoming" / "quantization.csv",
}
COLORS = {"Ours": "#0072B2", "Basis Sharing": "#D55E00", "SVD-LLM": "#009E73"}
MARKERS = {"bf16": "o", "w8a16": "s", "w4a16": "^"}


def read_rows(path: Path, method: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Import the external baseline payload before generating the paper figure."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(model, precision) for model in ("8b_15", "8b_25") for precision in ("bf16", "w8a16", "w4a16")}
    observed = {(row["model_id"], row["precision"]) for row in rows if row["method"] == method and row["source_status"] == "observed"}
    if len(rows) != 6 or observed != expected:
        raise RuntimeError(f"{path}: expected six complete observed rows for {method}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper" / "Figure" / "fig_quantization_15_25_storage_accuracy.png",
    )
    args = parser.parse_args()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 160})
    all_rows = {method: read_rows(path, method) for method, path in METHOD_FILES.items()}
    figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), sharex=True, sharey=True)
    for axis, model_id, title in zip(axes, ("8b_15", "8b_25"), ("15% structural", "25% structural")):
        for method, method_rows in all_rows.items():
            selected = [row for row in method_rows if row["model_id"] == model_id]
            selected.sort(key=lambda row: {"bf16": 0, "w8a16": 1, "w4a16": 2}[row["precision"]])
            xs = [float(row["serialized_gib"]) for row in selected]
            ys = [100.0 * float(row["macro_accuracy"]) for row in selected]
            axis.plot(xs, ys, color=COLORS[method], linewidth=1.2, alpha=0.8, label=method)
            for row, x, y in zip(selected, xs, ys):
                axis.scatter(x, y, color=COLORS[method], marker=MARKERS[row["precision"]], s=28, edgecolor="white", linewidth=0.5, zorder=3)
        axis.set_title(title, fontweight="semibold")
        axis.set_xlabel("Standalone serialized size (GiB)")
    axes[0].set_ylabel("Seven-task macro accuracy (%)")
    axes[0].legend(frameon=False, loc="lower right")
    figure.tight_layout(pad=0.6, w_pad=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(ROOT / "assets" / "figures" / "quantization.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
