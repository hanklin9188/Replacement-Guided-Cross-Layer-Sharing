from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from .utils import atomic_json, read_json, read_jsonl


def group_labels(manifest: dict[str, Any]) -> np.ndarray:
    labels = np.full(int(manifest["layers"]), -1, dtype=int)
    for group_id, group in enumerate(manifest["groups"]):
        labels[np.asarray(group, dtype=int)] = group_id
    if np.any(labels < 0):
        raise ValueError("Incomplete group manifest")
    return labels


def calibration_robustness(replacement_dirs: list[str], group_manifests: list[str], output: str) -> None:
    if len(replacement_dirs) != len(group_manifests) or len(replacement_dirs) < 2:
        raise ValueError("Provide matching replacement/group paths for at least two calibration runs")
    matrices = [np.load(Path(path) / "directed_cost.npy") for path in replacement_dirs]
    groups = [read_json(path) for path in group_manifests]
    n = matrices[0].shape[0]
    upper = np.triu_indices(n, 1)
    pair_rank = np.eye(len(matrices))
    ari = np.eye(len(matrices))
    representative = np.eye(len(matrices))
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            pair_rank[i, j] = pair_rank[j, i] = spearmanr(
                np.maximum(matrices[i], matrices[i].T)[upper],
                np.maximum(matrices[j], matrices[j].T)[upper],
            ).statistic
            ari[i, j] = ari[j, i] = adjusted_rand_score(group_labels(groups[i]), group_labels(groups[j]))
            representative[i, j] = representative[j, i] = len(
                set(groups[i]["representatives"]) & set(groups[j]["representatives"])
            ) / len(groups[i]["representatives"])
    labels = [f"cal-{i + 1}" for i in range(len(matrices))]
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    for ax, matrix, title in zip(axes, [pair_rank, ari, representative],
                                 ["Pair-cost Spearman", "Co-assignment ARI", "Representative stability"]):
        sns.heatmap(matrix, vmin=0, vmax=1, annot=True, fmt=".2f", xticklabels=labels,
                    yticklabels=labels, cmap="viridis", ax=ax, cbar=False)
        ax.set_title(title)
    fig.savefig(output_path / "calibration_robustness.pdf", bbox_inches="tight")
    fig.savefig(output_path / "calibration_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    atomic_json(output_path / "calibration_robustness.json", {
        "replacement_dirs": replacement_dirs, "group_manifests": group_manifests,
        "pair_cost_spearman": pair_rank.tolist(), "adjusted_rand_index": ari.tolist(),
        "representative_stability": representative.tolist(),
    })


def paired_task_stratified_bootstrap(predictions_a: str, predictions_b: str, samples: int, seed: int):
    a = {row["id"]: row for row in read_jsonl(predictions_a)}
    b = {row["id"]: row for row in read_jsonl(predictions_b)}
    ids = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError("Paired bootstrap requires identical final-evaluation sample IDs")
    tasks: dict[str, list[str]] = {}
    for row_id in ids:
        tasks.setdefault(a[row_id]["task"], []).append(row_id)
    observed = np.mean([
        np.mean([b[row_id]["correct"] - a[row_id]["correct"] for row_id in task_ids])
        for task_ids in tasks.values()
    ])
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        task_deltas = []
        for task_ids in tasks.values():
            selected = rng.choice(task_ids, size=len(task_ids), replace=True)
            task_deltas.append(np.mean([b[row_id]["correct"] - a[row_id]["correct"] for row_id in selected]))
        draws[sample] = np.mean(task_deltas)
    return {"difference_b_minus_a": float(observed), "ci95_low": float(np.percentile(draws, 2.5)),
            "ci95_high": float(np.percentile(draws, 97.5)), "bootstrap_samples": samples,
            "seed": seed, "paired_examples": len(ids), "tasks": sorted(tasks)}
