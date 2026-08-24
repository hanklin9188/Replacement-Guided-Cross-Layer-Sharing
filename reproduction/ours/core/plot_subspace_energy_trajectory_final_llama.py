#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for plotting") from exc


TEACHER_BLUE = "#204bd8"
STEP0_RED = "#b80f2a"
BEST_BLUE = "#1f78b4"
TREND_BLACK = "#202020"
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "fad_energy",
    [
        (0.00, "#061fb2"),
        (0.13, "#1e63e6"),
        (0.28, "#76b6ff"),
        (0.43, "#d9edff"),
        (0.53, "#fff9d9"),
        (0.66, "#ffd27a"),
        (0.79, "#ff8440"),
        (0.91, "#e5312e"),
        (1.00, "#9a0018"),
    ],
)
REGIME_COLORS = {
    "llama_early": "#204bd8",
    "llama_mid": "#f0a33a",
    "llama_late": "#b80f2a",
}


def _step_from_path(path: str) -> Optional[int]:
    match = re.search(r"step_(\d+)_subspace_plot_data\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _safe_slug(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))
    return safe.strip("_") or "regime"


def _pretty_regime(name: str) -> str:
    text = str(name)
    return text.replace("llama_", "").replace("_", " ").title() if text.startswith("llama_") else text.replace("_", " ").title()


def _load_plot_data(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"plot data is not a dict: {path}")
    return payload


def _energy_for_regime(regime_payload: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_z = regime_payload.get("teacher_z")
    student_z = regime_payload.get("student_z")
    metric_diag = regime_payload.get("energy_metric_diag")
    if not torch.is_tensor(teacher_z) or not torch.is_tensor(student_z):
        raise ValueError("teacher_z/student_z missing")
    if not torch.is_tensor(metric_diag):
        raise ValueError("energy_metric_diag missing")
    teacher_z = teacher_z.float().cpu()
    student_z = student_z.float().cpu()
    metric_diag = metric_diag.float().cpu().view(-1).clamp(min=1e-12)
    n = min(int(teacher_z.size(0)), int(student_z.size(0)))
    d = min(int(teacher_z.size(1)), int(student_z.size(1)), int(metric_diag.numel()))
    error = student_z[:n, :d] - teacher_z[:n, :d]
    metric_diag = metric_diag[:d]
    energy = 0.5 * (error.pow(2) / metric_diag.view(1, -1)).sum(dim=1)
    whitened_error = error / torch.sqrt(metric_diag.view(1, -1))
    return error, whitened_error, energy


def _fit_error_projection(*clouds: torch.Tensor) -> torch.Tensor:
    usable = [cloud.float().cpu() for cloud in clouds if torch.is_tensor(cloud) and int(cloud.numel()) > 0]
    if not usable:
        return torch.zeros((1, 2), dtype=torch.float32)
    merged = torch.cat(usable, dim=0)
    centered = merged - merged.mean(dim=0, keepdim=True)
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        basis = torch.zeros((int(merged.size(1)), 2), dtype=torch.float32)
        basis[0, 0] = 1.0
        if int(merged.size(1)) > 1:
            basis[1, 1] = 1.0
        return basis
    comp = min(2, int(vh.size(0)))
    basis = vh[:comp, :].transpose(0, 1).contiguous()
    if comp < 2:
        basis = torch.cat([basis, torch.zeros((int(merged.size(1)), 2 - comp), dtype=basis.dtype)], dim=1)
    return basis.float()


def _project_error(whitened_error: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    d = min(int(whitened_error.size(1)), int(basis.size(0)))
    whitened = whitened_error[:, :d].float()
    basis = basis[:d, :2].float()
    pc1 = torch.matmul(whitened, basis[:, 0])
    pc2 = torch.matmul(whitened, basis[:, 1]) if int(basis.size(1)) >= 2 else torch.zeros_like(pc1)
    angle = torch.atan2(pc2, pc1)
    full_radius = torch.linalg.norm(whitened, dim=1)
    x = full_radius * torch.cos(angle)
    y = full_radius * torch.sin(angle)
    return torch.stack([x, y], dim=1)


def _draw_projected_energy_background(ax: Any, coords: torch.Tensor, reference_energy: torch.Tensor) -> Any:
    coords = coords.float().cpu()
    if int(coords.numel()) <= 0:
        coords = torch.zeros((1, 2), dtype=torch.float32)
    finite_mask = torch.isfinite(coords).all(dim=1)
    finite_coords = coords[finite_mask] if bool(finite_mask.any()) else torch.zeros((1, 2), dtype=torch.float32)
    if int(finite_coords.size(0)) >= 8:
        x_min = min(float(torch.quantile(finite_coords[:, 0], 0.02).item()), 0.0)
        x_max = max(float(torch.quantile(finite_coords[:, 0], 0.98).item()), 0.0)
        y_min = min(float(torch.quantile(finite_coords[:, 1], 0.02).item()), 0.0)
        y_max = max(float(torch.quantile(finite_coords[:, 1], 0.98).item()), 0.0)
    else:
        x_min = min(float(finite_coords[:, 0].min().item()), 0.0)
        x_max = max(float(finite_coords[:, 0].max().item()), 0.0)
        y_min = min(float(finite_coords[:, 1].min().item()), 0.0)
        y_max = max(float(finite_coords[:, 1].max().item()), 0.0)
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    x_pad = 0.08 * x_span
    y_pad = 0.08 * y_span
    x_values = torch.linspace(x_min - x_pad, x_max + x_pad, steps=150)
    y_values = torch.linspace(y_min - y_pad, y_max + y_pad, steps=150)
    yy, xx = torch.meshgrid(y_values, x_values, indexing="ij")
    energy = 0.5 * (xx.pow(2) + yy.pow(2))
    display_energy = torch.log10(1.0 + energy)
    ref = reference_energy.float().cpu().reshape(-1)
    ref = ref[torch.isfinite(ref)]
    if int(ref.numel()) > 0:
        high_energy = float(torch.quantile(ref, 0.92).item())
    else:
        high_energy = float(torch.quantile(energy.reshape(-1), 0.70).item())
    high = math.log10(1.0 + max(1e-8, high_energy))
    levels = torch.linspace(0.0, high, steps=15).tolist()
    filled = ax.contourf(
        xx.numpy(),
        yy.numpy(),
        display_energy.numpy(),
        levels=levels,
        cmap=ENERGY_CMAP,
        alpha=0.82,
        extend="max",
        zorder=0,
    )
    ax.contour(
        xx.numpy(),
        yy.numpy(),
        display_energy.numpy(),
        levels=levels[1:],
        colors="#9a3c35",
        linewidths=0.65,
        alpha=0.23,
        zorder=1,
    )
    ax.contour(
        xx.numpy(),
        yy.numpy(),
        display_energy.numpy(),
        levels=levels[1:4],
        colors="#1e63e6",
        linewidths=1.15,
        linestyles="--",
        alpha=0.78,
        zorder=2,
    )
    ax.set_xlim(float(x_values[0].item()), float(x_values[-1].item()))
    ax.set_ylim(float(y_values[0].item()), float(y_values[-1].item()))
    ax.set_autoscale_on(False)
    return filled


def _subsample(coords: torch.Tensor, energy: torch.Tensor, max_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(coords.size(0)) <= int(max_points):
        return coords, energy
    idx = torch.linspace(0, int(coords.size(0)) - 1, steps=int(max_points)).round().long()
    return coords[idx], energy[idx]


def _pca_elbow_dim(cumulative_ratio: Sequence[float]) -> int:
    values = [float(x) for x in cumulative_ratio if math.isfinite(float(x))]
    if not values:
        return 0
    if len(values) <= 2:
        return int(len(values))
    x0, y0 = 1.0, float(values[0])
    x1, y1 = float(len(values)), float(values[-1])
    dx = x1 - x0
    dy = y1 - y0
    denom = math.sqrt(dx * dx + dy * dy)
    if denom <= 1e-12:
        return 1
    best_idx = 0
    best_dist = -1.0
    for idx, value in enumerate(values):
        x = float(idx + 1)
        y = float(value)
        dist = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom
        if dist > best_dist:
            best_dist = dist
            best_idx = idx
    return int(best_idx + 1)


def _threshold_key(threshold: float) -> str:
    percent = float(threshold) * 100.0
    if abs(percent - round(percent)) <= 1e-6:
        return f"{int(round(percent))}%"
    return f"{percent:.1f}".rstrip("0").rstrip(".") + "%"


def _parse_variance_thresholds(raw: str) -> List[float]:
    thresholds: List[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if value > 1.0:
            value = value / 100.0
        if 0.0 < value <= 1.0 and math.isfinite(value):
            thresholds.append(float(value))
    return thresholds or [0.80, 0.90, 0.95, 0.99]


def _variance_threshold_dims(cumulative_ratio: Sequence[float], thresholds: Sequence[float]) -> Dict[str, Optional[int]]:
    values = [float(x) for x in cumulative_ratio if math.isfinite(float(x))]
    result: Dict[str, Optional[int]] = {}
    for threshold in thresholds:
        key = _threshold_key(float(threshold))
        dim: Optional[int] = None
        for idx, value in enumerate(values):
            if value >= float(threshold):
                dim = int(idx + 1)
                break
        result[key] = dim
    return result


def _pca_cumulative_variance(points: torch.Tensor, max_components: int = 64) -> Tuple[List[float], int]:
    if not torch.is_tensor(points) or points.dim() != 2 or int(points.size(0)) <= 1:
        return [], 0
    centered = points.float().cpu() - points.float().cpu().mean(dim=0, keepdim=True)
    try:
        _, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        return [], 0
    total = float((singular_values.pow(2)).sum().item())
    if not math.isfinite(total) or total <= 0.0:
        return [], 0
    available_components = int(singular_values.numel())
    if int(max_components) <= 0:
        max_count = available_components
    else:
        max_count = min(max(1, int(max_components)), available_components)
    explained = [(float(singular_values[idx].pow(2).item()) / total) for idx in range(max_count)]
    cumulative: List[float] = []
    running = 0.0
    for value in explained:
        running += float(value)
        cumulative.append(float(min(1.0, running)))
    return cumulative, _pca_elbow_dim(cumulative)


def _plot_regime_trajectory(
    *,
    regime_name: str,
    step0_payload: Dict[str, Any],
    best_payload: Dict[str, Any],
    output_dir: str,
    max_points: int,
) -> Dict[str, Any]:
    _, step0_white, step0_energy = _energy_for_regime(step0_payload)
    _, best_white, best_energy = _energy_for_regime(best_payload)
    basis = _fit_error_projection(step0_white, best_white)
    step0_coords = _project_error(step0_white, basis)
    best_coords = _project_error(best_white, basis)
    step0_plot, step0_energy_plot = _subsample(step0_coords, step0_energy, max_points)
    best_plot, best_energy_plot = _subsample(best_coords, best_energy, max_points)

    all_coords = torch.cat([step0_coords, best_coords, torch.zeros((1, 2), dtype=torch.float32)], dim=0)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    bg = _draw_projected_energy_background(ax, all_coords, torch.cat([step0_energy, best_energy], dim=0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.4%", pad=0.22)
    cbar = fig.colorbar(bg, cax=cax)
    cbar.set_label(r"$\log_{10}(1+E_{dev})$", fontsize=9)
    ticks = cbar.get_ticks()
    if len(ticks) >= 2:
        cbar.set_ticks([ticks[0], ticks[-1]])
        cbar.set_ticklabels(["Low", "High"])

    ax.scatter(
        step0_plot[:, 0].tolist(),
        step0_plot[:, 1].tolist(),
        c=torch.log10(1.0 + step0_energy_plot.float()).tolist(),
        cmap=ENERGY_CMAP,
        norm=bg.norm,
        s=18,
        alpha=0.58,
        edgecolors="#2F3136",
        linewidths=0.18,
        label="step 0 samples",
        zorder=3,
    )
    ax.scatter(
        best_plot[:, 0].tolist(),
        best_plot[:, 1].tolist(),
        c=torch.log10(1.0 + best_energy_plot.float()).tolist(),
        cmap=ENERGY_CMAP,
        norm=bg.norm,
        s=22,
        alpha=0.90,
        edgecolors="#FFFFFF",
        linewidths=0.38,
        label="best-val samples",
        zorder=4,
    )

    step0_mean_energy = float(step0_energy.mean().item())
    best_mean_energy = float(best_energy.mean().item())
    ax.text(
        0.02,
        0.98,
        f"E_dev: step 0 = {step0_mean_energy:.3f}  ->  best-val = {best_mean_energy:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#ffffff", "edgecolor": "#d8d8d8", "alpha": 0.90},
        zorder=10,
    )
    ax.set_title("")
    ax.set_xlabel("Dim1", labelpad=6)
    ax.set_ylabel("Dim2", labelpad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.13, linewidth=0.55, color="#ffffff")
    ax.legend(loc="upper right", fontsize=7.4, frameon=True, facecolor="#ffffff", edgecolor="#d0d4d8")
    fig.subplots_adjust(left=0.115, right=0.875, bottom=0.105, top=0.975)
    output_png = os.path.join(output_dir, f"energy_alignment_{_safe_slug(regime_name)}.png")
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return {
        "output_png": output_png,
        "step0_mean_energy": step0_mean_energy,
        "bestval_mean_energy": best_mean_energy,
        "energy_drop": step0_mean_energy - best_mean_energy,
    }


def _plot_regime_energy_distribution(
    *,
    regime_name: str,
    step0_payload: Dict[str, Any],
    best_payload: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    _, _, step0_energy = _energy_for_regime(step0_payload)
    _, _, best_energy = _energy_for_regime(best_payload)
    eps = 1e-8
    step0_log = torch.log10(step0_energy.float().clamp(min=eps))
    best_log = torch.log10(best_energy.float().clamp(min=eps))
    step0_mean = float(step0_energy.mean().item())
    best_mean = float(best_energy.mean().item())
    step0_median = float(step0_energy.median().item())
    best_median = float(best_energy.median().item())

    fig, ax = plt.subplots(figsize=(4.7, 4.2))
    parts = ax.violinplot(
        [step0_log.tolist(), best_log.tolist()],
        positions=[1, 2],
        widths=0.62,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    colors = [STEP0_RED, BEST_BLUE]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.45)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#111111")
        parts["cmedians"].set_linewidth(1.8)

    jitter0 = torch.linspace(-0.08, 0.08, steps=min(180, int(step0_log.numel())))
    jitter1 = torch.linspace(-0.08, 0.08, steps=min(180, int(best_log.numel())))
    ax.scatter(
        (1.0 + jitter0).tolist(),
        step0_log[: int(jitter0.numel())].tolist(),
        color=STEP0_RED,
        s=8,
        alpha=0.18,
        linewidths=0.0,
    )
    ax.scatter(
        (2.0 + jitter1).tolist(),
        best_log[: int(jitter1.numel())].tolist(),
        color=BEST_BLUE,
        s=8,
        alpha=0.18,
        linewidths=0.0,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Step 0", "Best-val"])
    ax.set_ylabel(r"$\log_{10}(E_{dev})$")
    ax.set_title(f"{regime_name}: full deviation-energy distribution", fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.22, linewidth=0.55)
    ax.text(
        0.5,
        0.98,
        f"mean E: {step0_mean:.3f} -> {best_mean:.3f}\nmedian E: {step0_median:.3f} -> {best_median:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "#d8d8d8", "alpha": 0.92},
    )
    fig.tight_layout()
    output_png = os.path.join(output_dir, f"energy_distribution_{_safe_slug(regime_name)}.png")
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return {
        "output_png": output_png,
        "step0_mean_energy": step0_mean,
        "bestval_mean_energy": best_mean,
        "step0_median_energy": step0_median,
        "bestval_median_energy": best_median,
    }


def _plot_energy_trend(
    *,
    step_payloads: Sequence[Tuple[int, Dict[str, Any]]],
    output_dir: str,
) -> Dict[str, Any]:
    regime_order = [str(x) for x in step_payloads[0][1].get("regime_order", [])] if step_payloads else []
    trend: Dict[str, Dict[str, List[float]]] = {
        name: {"steps": [], "mean_energy": []}
        for name in regime_order
    }
    for step, payload in step_payloads:
        regimes = dict(payload.get("regimes", {}))
        for regime_name in regime_order:
            regime_payload = regimes.get(regime_name)
            if not isinstance(regime_payload, dict):
                continue
            try:
                _, _, energy = _energy_for_regime(regime_payload)
            except Exception:
                continue
            trend[regime_name]["steps"].append(int(step))
            trend[regime_name]["mean_energy"].append(float(energy.mean().item()))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for regime_name in regime_order:
        steps = trend[regime_name]["steps"]
        values = trend[regime_name]["mean_energy"]
        if not steps:
            continue
        ax.plot(
            steps,
            values,
            marker="o",
            linewidth=1.9,
            markersize=4.0,
            color=REGIME_COLORS.get(regime_name, None),
            label=_pretty_regime(regime_name),
        )
    ax.set_title("Mean Deviation Energy During Training", fontsize=12, pad=8)
    ax.set_xlabel("Validation step")
    ax.set_ylabel(r"Mean $E_{dev}$")
    ax.grid(alpha=0.20, linewidth=0.55)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_png = os.path.join(output_dir, "energy_mean_trend_by_regime.png")
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return {"output_png": output_png, "trend": trend}


def _plot_mean_pair_l2_comparison(
    *,
    step0_payload: Dict[str, Any],
    best_payload: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    regime_order = [str(x) for x in step0_payload.get("regime_order", [])]
    step0_values_by_regime = dict(step0_payload.get("mean_pair_l2_by_regime", {}))
    best_values_by_regime = dict(best_payload.get("mean_pair_l2_by_regime", {}))
    names = [
        name for name in regime_order
        if name in step0_values_by_regime and name in best_values_by_regime
    ]
    step0_values = [float(step0_values_by_regime[name]) for name in names]
    best_values = [float(best_values_by_regime[name]) for name in names]
    step0_step = int(step0_payload.get("step", 0))
    best_step = int(best_payload.get("step", -1))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x_positions = torch.arange(len(names), dtype=torch.float32)
    width = 0.34
    step0_bars = ax.bar(
        (x_positions - width / 2.0).tolist(),
        step0_values,
        width=float(width),
        color="#6B7280",
        edgecolor="#4B5563",
        linewidth=0.45,
        alpha=0.92,
        label=f"step {step0_step}",
    )
    best_bars = ax.bar(
        (x_positions + width / 2.0).tolist(),
        best_values,
        width=float(width),
        color="#4C78A8",
        edgecolor="#2F4B7C",
        linewidth=0.45,
        alpha=0.94,
        label=f"best-val step {best_step}" if best_step >= 0 else "best-val",
    )
    top = max(step0_values + best_values) if (step0_values or best_values) else 0.0
    ax.set_ylim(0.0, top * 1.18 if top > 0.0 else 1.0)
    ax.set_ylabel("L2 gap", fontsize=10)
    ax.set_xlabel("Layer group", fontsize=10)
    ax.set_xticks(x_positions.tolist())
    ax.set_xticklabels([_pretty_regime(name) for name in names], fontsize=8.5)
    ax.grid(axis="y", color="#D8DEE9", alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4B5563")
    ax.spines["bottom"].set_color("#4B5563")
    ax.legend(loc="upper left", fontsize=8.2, frameon=True, facecolor="#ffffff", edgecolor="#D8DEE9")
    for bars in (step0_bars, best_bars):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                float(bar.get_x() + bar.get_width() / 2.0),
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#222222",
            )
    fig.tight_layout()
    output_png = os.path.join(output_dir, "mean_pair_l2_step0_vs_bestval.png")
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return {
        "output_png": output_png,
        "step0_step": step0_step,
        "bestval_step": best_step,
        "regimes": names,
        "step0_values": step0_values,
        "bestval_values": best_values,
    }


def _plot_deviation_error_pca_elbow(
    *,
    step0_payload: Dict[str, Any],
    best_payload: Dict[str, Any],
    output_dir: str,
    max_components: int = 64,
    variance_thresholds: Sequence[float],
    output_name: str,
) -> Dict[str, Any]:
    regime_order = [str(x) for x in step0_payload.get("regime_order", [])]
    step0_regimes = dict(step0_payload.get("regimes", {}))
    best_regimes = dict(best_payload.get("regimes", {}))
    names = [
        name for name in regime_order
        if isinstance(step0_regimes.get(name), dict) and isinstance(best_regimes.get(name), dict)
    ]
    if not names:
        return {"output_png": "", "reason": "no_regimes"}

    fig, axes = plt.subplots(1, len(names), figsize=(9.8, 3.45), sharey=True)
    if len(names) == 1:
        axes = [axes]

    summary: Dict[str, Any] = {}
    stage_styles = {
        "step0": {"label": "step 0", "color": "#6B7280", "linestyle": "--", "marker": "o"},
        "bestval": {"label": "best-val", "color": "#4C78A8", "linestyle": "-", "marker": "o"},
    }
    for ax, regime_name in zip(axes, names):
        _, step0_white, _ = _energy_for_regime(dict(step0_regimes[regime_name]))
        _, best_white, _ = _energy_for_regime(dict(best_regimes[regime_name]))
        step0_cumulative, step0_elbow = _pca_cumulative_variance(step0_white, max_components=max_components)
        best_cumulative, best_elbow = _pca_cumulative_variance(best_white, max_components=max_components)
        step0_threshold_dims = _variance_threshold_dims(step0_cumulative, variance_thresholds)
        best_threshold_dims = _variance_threshold_dims(best_cumulative, variance_thresholds)
        summary[regime_name] = {
            "step0_cumulative_explained_variance": step0_cumulative,
            "bestval_cumulative_explained_variance": best_cumulative,
            "step0_elbow_dim": int(step0_elbow),
            "bestval_elbow_dim": int(best_elbow),
            "step0_variance_threshold_dims": step0_threshold_dims,
            "bestval_variance_threshold_dims": best_threshold_dims,
        }
        plot_max_components = max(len(step0_cumulative), len(best_cumulative), 1)

        for key, cumulative, elbow_dim in [
            ("step0", step0_cumulative, step0_elbow),
            ("bestval", best_cumulative, best_elbow),
        ]:
            if not cumulative:
                continue
            style = stage_styles[key]
            xs = list(range(1, len(cumulative) + 1))
            ax.plot(
                xs,
                cumulative,
                color=str(style["color"]),
                linestyle=str(style["linestyle"]),
                marker=str(style["marker"]),
                linewidth=1.9,
                markersize=3.6,
                label=str(style["label"]),
            )
            if 1 <= int(elbow_dim) <= len(cumulative):
                ax.scatter(
                    [int(elbow_dim)],
                    [float(cumulative[int(elbow_dim) - 1])],
                    s=92,
                    facecolors="#111827",
                    edgecolors="#ffffff",
                    linewidths=1.15,
                    zorder=6,
                )

        for threshold in variance_thresholds:
            ax.axhline(
                float(threshold),
                color="#CBD5E1",
                linestyle=":",
                linewidth=0.65,
                alpha=0.8,
                zorder=0,
            )
        ax.set_title(_pretty_regime(regime_name), fontsize=9.6, pad=6)
        ax.set_xlabel("Dimensions", fontsize=9.2)
        ax.set_xlim(0.9, float(plot_max_components) + 0.35)
        ax.set_ylim(0.0, 1.02)
        ax.grid(color="#D8DEE9", alpha=0.56, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#4B5563")
        ax.spines["bottom"].set_color("#4B5563")
        ax.tick_params(axis="both", labelsize=8.4)
    axes[0].set_ylabel("Cumulative explained variance", fontsize=9.2)

    handles = [
        Line2D([0], [0], color="#6B7280", linestyle="--", marker="o", linewidth=1.9, markersize=4.0, label="step 0"),
        Line2D([0], [0], color="#4C78A8", linestyle="-", marker="o", linewidth=1.9, markersize=4.0, label="best-val"),
        Line2D([0], [0], color="#111827", marker="o", linestyle="", markersize=6.0, label="elbow"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        fontsize=8.5,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#D8DEE9",
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.17, top=0.76, wspace=0.24)
    output_png = os.path.join(output_dir, str(output_name or "deviation_error_pca_elbow_step0_vs_bestval.png"))
    fig.savefig(output_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return {
        "output_png": output_png,
        "max_components": int(max_components),
        "variance_thresholds": [float(x) for x in variance_thresholds],
        "energy_space": "whitened_deviation_error_D_inv_sqrt_zS_minus_zT",
        "regimes": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_points", type=int, default=900)
    parser.add_argument(
        "--pca_max_components",
        type=int,
        default=64,
        help="Maximum PCA components for deviation/error explained-variance plots. Use 0 for all available components.",
    )
    parser.add_argument(
        "--pca_variance_thresholds",
        default="0.80,0.90,0.95,0.99",
        help="Comma-separated cumulative explained-variance thresholds, e.g. 0.8,0.9,0.95 or 80,90,95.",
    )
    parser.add_argument(
        "--pca_output_name",
        default="deviation_error_pca_elbow_step0_vs_bestval.png",
        help="Output filename for the step0-vs-bestval PCA elbow plot.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(str(args.input_dir))
    output_dir = os.path.abspath(str(args.output_dir or args.input_dir))
    os.makedirs(output_dir, exist_ok=True)

    step_paths = []
    for path in glob.glob(os.path.join(input_dir, "step_*_subspace_plot_data.pt")):
        step = _step_from_path(path)
        if step is not None:
            step_paths.append((step, path))
    step_paths.sort(key=lambda item: item[0])
    if not step_paths:
        raise FileNotFoundError(f"no step_*_subspace_plot_data.pt files in {input_dir}")

    step_payloads = [(step, _load_plot_data(path)) for step, path in step_paths]
    step0_payload = step_payloads[0][1]
    best_path = os.path.join(input_dir, "bestval_subspace_plot_data.pt")
    best_payload = _load_plot_data(best_path) if os.path.isfile(best_path) else step_payloads[-1][1]

    regime_order = [str(x) for x in step0_payload.get("regime_order", [])]
    outputs: Dict[str, Any] = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "step0_step": int(step_payloads[0][0]),
        "bestval_step": int(best_payload.get("step", -1)),
        "regimes": {},
    }
    for regime_name in regime_order:
        step0_regime = dict(step0_payload.get("regimes", {})).get(regime_name)
        best_regime = dict(best_payload.get("regimes", {})).get(regime_name)
        if not isinstance(step0_regime, dict) or not isinstance(best_regime, dict):
            continue
        outputs["regimes"][regime_name] = _plot_regime_trajectory(
            regime_name=regime_name,
            step0_payload=step0_regime,
            best_payload=best_regime,
            output_dir=output_dir,
            max_points=max(1, int(args.max_points)),
        )
        outputs["regimes"][regime_name]["distribution"] = _plot_regime_energy_distribution(
            regime_name=regime_name,
            step0_payload=step0_regime,
            best_payload=best_regime,
            output_dir=output_dir,
        )

    outputs["trend"] = _plot_energy_trend(step_payloads=step_payloads, output_dir=output_dir)
    outputs["mean_pair_l2_comparison"] = _plot_mean_pair_l2_comparison(
        step0_payload=step0_payload,
        best_payload=best_payload,
        output_dir=output_dir,
    )
    outputs["deviation_error_pca_elbow"] = _plot_deviation_error_pca_elbow(
        step0_payload=step0_payload,
        best_payload=best_payload,
        output_dir=output_dir,
        max_components=int(args.pca_max_components),
        variance_thresholds=_parse_variance_thresholds(str(args.pca_variance_thresholds)),
        output_name=str(args.pca_output_name),
    )
    output_json = os.path.join(output_dir, "energy_alignment_summary.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(outputs, handle, ensure_ascii=False, indent=2)
    print(output_json)


if __name__ == "__main__":
    main()
