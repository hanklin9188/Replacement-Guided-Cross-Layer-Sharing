#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DATASET_CHOICES = {
    "piqa": 2,
    "winogrande": 2,
    "social_i_qa": 3,
    "hellaswag": 4,
    "ARC-Challenge": 4,
    "ARC-Easy": 4,
    "openbookqa": 4,
    "csqa": 5,
}

BASE_FEATURE_NAMES = [
    "layer_frac",
    "confidence",
    "margin",
    "entropy",
    "entropy_norm",
    "num_choices",
    "confidence_over_random",
    "confidence_growth",
    "margin_growth",
    "stable_count",
    "pred_changed",
]

INFO_ENERGY_FEATURE_NAMES = BASE_FEATURE_NAMES + [
    "answer_energy",
    "energy_gap",
    "score_range",
    "score_std",
    "top2_prob",
    "prob_margin",
    "entropy_slope",
    "entropy_drop",
    "energy_margin_growth",
    "answer_energy_growth",
    "prob_l2_change",
    "js_from_prev",
    "kl_from_prev",
    "top1_changed",
    "layer_remaining_frac",
]

INFO_ENERGY_LAYER_FEATURE_NAMES = INFO_ENERGY_FEATURE_NAMES + [
    "is_layer_16",
    "is_layer_20",
    "is_layer_24",
]

IE4_FEATURE_NAMES = [
    "margin",
    "entropy",
    "entropy_drop",
    "js_from_prev",
]

IE5_FEATURE_NAMES = [
    "layer_frac",
    *IE4_FEATURE_NAMES,
]

IE6_FEATURE_NAMES = [
    "energy_gap",
    "top_surprisal",
    "entropy_norm",
    "info_gain_ratio",
    "js_from_prev",
    "kl_from_prev",
]

IE6_DEPTH_FEATURE_NAMES = [
    "layer_frac",
    *IE6_FEATURE_NAMES,
]

IE6_FUTURE_FEATURE_NAMES = [
    "energy_gap",
    "top_surprisal",
    "entropy_norm",
    "info_gain_ratio",
    "js_from_prev",
    "prob_l2_change",
    "layer_remaining_frac",
]

USEFUL4_FEATURE_NAMES = [
    "entropy_norm",
    "top_surprisal",
    "js_from_prev",
    "energy_gap",
]

OPTSTOP5_FEATURE_NAMES = [
    "entropy_norm",
    "top_surprisal",
    "energy_gap",
    "js_from_prev",
    "layer_remaining_frac",
]

FEATURE_NAMES = BASE_FEATURE_NAMES


class ExitMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").replace(",", " ").split() if x.strip()]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_name(prefix: str, threshold: float) -> str:
    return f"{prefix}_{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _threshold_grid(value: str) -> List[float]:
    if ":" in str(value):
        start, end, step = [float(x) for x in str(value).split(":", 2)]
        vals: List[float] = []
        cur = start
        while cur <= end + 1e-12:
            vals.append(round(cur, 6))
            cur += step
        return vals
    return [float(x) for x in _parse_list(value)]


def _feature_names(feature_set: str) -> List[str]:
    name = str(feature_set or "base").strip().lower()
    if name == "ie4":
        return list(IE4_FEATURE_NAMES)
    if name == "ie5":
        return list(IE5_FEATURE_NAMES)
    if name == "ie6":
        return list(IE6_FEATURE_NAMES)
    if name == "ie6_depth":
        return list(IE6_DEPTH_FEATURE_NAMES)
    if name == "ie6_future":
        return list(IE6_FUTURE_FEATURE_NAMES)
    if name == "useful4":
        return list(USEFUL4_FEATURE_NAMES)
    if name == "optstop5":
        return list(OPTSTOP5_FEATURE_NAMES)
    if name == "info_energy":
        return list(INFO_ENERGY_FEATURE_NAMES)
    if name == "info_energy_layer":
        return list(INFO_ENERGY_LAYER_FEATURE_NAMES)
    return list(BASE_FEATURE_NAMES)


def _dataset_from_records_path(path: str) -> str:
    base = os.path.basename(path)
    suffix = "_confidence_sweep_records.jsonl"
    if base.endswith(suffix):
        return base[: -len(suffix)]
    return base.split("_confidence", 1)[0]


def _read_records(path: str, dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["dataset"] = dataset
            row["num_choices"] = int(DATASET_CHOICES.get(dataset, 4))
            rows.append(row)
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def _layer_metrics(record: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    items = [(int(layer), dict(metrics)) for layer, metrics in record.get("layer_confidence", {}).items()]
    items.sort(key=lambda x: x[0])
    if not items:
        raise ValueError("empty layer_confidence")
    return items


def _choose_by_threshold(record: Dict[str, Any], threshold: float, min_exit_layer: int) -> Tuple[int, str]:
    items = _layer_metrics(record)
    final_layer = int(items[-1][0])
    for layer, metrics in items[:-1]:
        if int(layer) < int(min_exit_layer):
            continue
        if float(metrics.get("confidence", 0.0)) >= float(threshold):
            return int(layer), str(metrics.get("pred", ""))
    return final_layer, str(items[-1][1].get("pred", ""))


def _find_oracle_threshold(
    records: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
    *,
    min_exit_layer: int,
    max_acc_drop: float,
) -> Dict[str, Any]:
    full_correct = 0
    final_layer_sum = 0
    for record in records:
        items = _layer_metrics(record)
        answer = str(record.get("answer", "")).strip()
        full_correct += int(str(items[-1][1].get("pred", "")) == answer)
        final_layer_sum += int(items[-1][0])
    total = max(1, len(records))
    full_acc = float(full_correct / total)
    rows = []
    for threshold in thresholds:
        correct = 0
        layer_sum = 0
        hist: Counter[str] = Counter()
        for record in records:
            answer = str(record.get("answer", "")).strip()
            layer, pred = _choose_by_threshold(record, threshold, min_exit_layer)
            correct += int(str(pred) == answer)
            layer_sum += int(layer)
            hist[str(layer)] += 1
        avg_layer = float(layer_sum / total)
        avg_final = float(final_layer_sum / total)
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(correct / total),
            "full_accuracy": full_acc,
            "avg_exit_layer": avg_layer,
            "estimated_layer_savings": float(1.0 - avg_layer / max(1e-9, avg_final)),
            "exit_layer_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        })
    feasible = [r for r in rows if float(r["accuracy"]) >= full_acc - float(max_acc_drop)]
    selected = max(feasible or rows, key=lambda r: (float(r["estimated_layer_savings"]), float(r["accuracy"])))
    return {"selected": selected, "sweep": rows}


def _normalise(values: Sequence[float]) -> List[float]:
    vals = [max(1e-12, float(x)) for x in values]
    total = max(1e-12, sum(vals))
    return [float(x / total) for x in vals]


def _probs_from_metrics(metrics: Dict[str, Any]) -> List[float]:
    raw = metrics.get("probs", {})
    if isinstance(raw, dict) and raw:
        return _normalise([float(raw[k]) for k in sorted(raw.keys())])
    conf = float(metrics.get("confidence", 0.0))
    return _normalise([conf, max(1e-12, 1.0 - conf)])


def _scores_from_metrics(metrics: Dict[str, Any]) -> List[float]:
    raw = metrics.get("scores", {})
    if isinstance(raw, dict) and raw:
        return [float(raw[k]) for k in sorted(raw.keys())]
    margin = float(metrics.get("margin", metrics.get("energy_margin", 0.0)))
    return [margin, 0.0]


def _logsumexp(values: Sequence[float]) -> float:
    vals = [float(x) for x in values]
    if not vals:
        return 0.0
    max_val = max(vals)
    return float(max_val + math.log(sum(math.exp(x - max_val) for x in vals)))


def _score_features(scores: Sequence[float]) -> Dict[str, float]:
    vals = [float(x) for x in scores]
    if not vals:
        return {"energy_gap": 0.0, "score_range": 0.0, "score_std": 0.0}
    order = sorted(vals, reverse=True)
    top = float(order[0])
    others = order[1:] or [0.0]
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
    return {
        "energy_gap": float(top - _logsumexp(others)),
        "score_range": float(max(vals) - min(vals)),
        "score_std": float(math.sqrt(max(0.0, var))),
    }


def _prob_features(probs: Sequence[float]) -> Dict[str, float]:
    vals = _normalise(probs)
    order = sorted(vals, reverse=True)
    top = float(order[0]) if order else 0.0
    second = float(order[1]) if len(order) > 1 else 0.0
    return {"top2_prob": second, "prob_margin": float(top - second)}


def _kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p_vals = _normalise(p)
    q_vals = _normalise(q)
    width = min(len(p_vals), len(q_vals))
    if width <= 0:
        return 0.0
    return float(sum(p_vals[i] * math.log(p_vals[i] / max(1e-12, q_vals[i])) for i in range(width)))


def _js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p_vals = _normalise(p)
    q_vals = _normalise(q)
    width = min(len(p_vals), len(q_vals))
    if width <= 0:
        return 0.0
    p_vals = p_vals[:width]
    q_vals = q_vals[:width]
    mid = [(a + b) * 0.5 for a, b in zip(p_vals, q_vals)]
    return float(0.5 * _kl_divergence(p_vals, mid) + 0.5 * _kl_divergence(q_vals, mid))


def _prob_l2(p: Sequence[float], q: Sequence[float]) -> float:
    p_vals = _normalise(p)
    q_vals = _normalise(q)
    width = min(len(p_vals), len(q_vals))
    if width <= 0:
        return 0.0
    return float(math.sqrt(sum((p_vals[i] - q_vals[i]) ** 2 for i in range(width))))


def _features_for_layer(
    *,
    layer: int,
    final_layer: int,
    metrics: Dict[str, Any],
    previous_metrics: Dict[str, Any] | None,
    stable_count: int,
    num_choices: int,
) -> Dict[str, float]:
    confidence = float(metrics.get("confidence", 0.0))
    margin = float(metrics.get("margin", metrics.get("energy_margin", 0.0)))
    entropy = float(metrics.get("entropy", 0.0))
    answer_energy = float(metrics.get("answer_energy", -math.log(max(1e-12, confidence))))
    choice_count = max(1, int(num_choices))
    max_entropy = math.log(float(choice_count)) if choice_count > 1 else 1.0
    probs = _probs_from_metrics(metrics)
    scores = _scores_from_metrics(metrics)
    score_stats = _score_features(scores)
    prob_stats = _prob_features(probs)
    if previous_metrics:
        prev_confidence = float(previous_metrics.get("confidence", 0.0))
        prev_margin = float(previous_metrics.get("margin", previous_metrics.get("energy_margin", 0.0)))
        prev_entropy = float(previous_metrics.get("entropy", entropy))
        prev_answer_energy = float(previous_metrics.get("answer_energy", -math.log(max(1e-12, prev_confidence))))
        pred_changed = float(str(previous_metrics.get("pred", "")) != str(metrics.get("pred", "")))
        prev_probs = _probs_from_metrics(previous_metrics)
        prob_l2_change = _prob_l2(probs, prev_probs)
        js_from_prev = _js_divergence(probs, prev_probs)
        kl_from_prev = _kl_divergence(probs, prev_probs)
    else:
        prev_confidence = confidence
        prev_margin = margin
        prev_entropy = entropy
        prev_answer_energy = answer_energy
        pred_changed = 0.0
        prob_l2_change = 0.0
        js_from_prev = 0.0
        kl_from_prev = 0.0
    entropy_drop = prev_entropy - entropy
    info_gain_ratio = entropy_drop / max(1e-12, abs(prev_entropy))
    return {
        "layer_frac": float(layer) / float(max(1, final_layer)),
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "entropy_norm": float(entropy / max(1e-12, max_entropy)),
        "num_choices": float(choice_count),
        "confidence_over_random": float(confidence * choice_count),
        "confidence_growth": confidence - prev_confidence,
        "margin_growth": margin - prev_margin,
        "stable_count": float(stable_count),
        "pred_changed": pred_changed,
        "answer_energy": answer_energy,
        "top_surprisal": answer_energy,
        "energy_gap": float(score_stats["energy_gap"]),
        "score_range": float(score_stats["score_range"]),
        "score_std": float(score_stats["score_std"]),
        "top2_prob": float(prob_stats["top2_prob"]),
        "prob_margin": float(prob_stats["prob_margin"]),
        "entropy_slope": entropy - prev_entropy,
        "entropy_drop": entropy_drop,
        "info_gain_ratio": info_gain_ratio,
        "energy_margin_growth": margin - prev_margin,
        "answer_energy_growth": answer_energy - prev_answer_energy,
        "prob_l2_change": prob_l2_change,
        "js_from_prev": js_from_prev,
        "kl_from_prev": kl_from_prev,
        "top1_changed": pred_changed,
        "layer_remaining_frac": float(max(0, final_layer - layer)) / float(max(1, final_layer)),
        "is_layer_16": float(int(layer) == 16),
        "is_layer_20": float(int(layer) == 20),
        "is_layer_24": float(int(layer) == 24),
    }


def _policy_rows(record: Dict[str, Any], oracle_exit_layer: int, min_exit_layer: int) -> List[Dict[str, Any]]:
    items = _layer_metrics(record)
    final_layer = int(items[-1][0])
    previous_metrics: Dict[str, Any] | None = None
    previous_pred = ""
    stable_count = 0
    rows: List[Dict[str, Any]] = []
    for layer, metrics in items:
        pred = str(metrics.get("pred", ""))
        stable_count = stable_count + 1 if pred == previous_pred else 1
        if int(layer) >= int(min_exit_layer) and int(layer) < final_layer:
            feats = _features_for_layer(
                layer=int(layer),
                final_layer=final_layer,
                metrics=metrics,
                previous_metrics=previous_metrics,
                stable_count=stable_count,
                num_choices=int(record.get("num_choices", 4)),
            )
            label = float(int(layer) >= int(oracle_exit_layer))
            rows.append({"features": feats, "label": label, "dataset": record.get("dataset", ""), "layer": int(layer)})
            if label >= 0.5:
                break
        previous_metrics = dict(metrics)
        previous_pred = pred
    return rows


def _answer_energy(metrics: Dict[str, Any], answer: str) -> float:
    probs = metrics.get("probs", {})
    if isinstance(probs, dict) and answer in probs:
        return float(-math.log(max(1e-12, float(probs[answer]))))
    return float(metrics.get("answer_energy", -math.log(max(1e-12, float(metrics.get("confidence", 0.0))))))


def _risk_gain_rows(record: Dict[str, Any], min_exit_layer: int) -> List[Dict[str, Any]]:
    items = _layer_metrics(record)
    final_layer = int(items[-1][0])
    answer = str(record.get("answer", "")).strip()
    future_best_energy: Dict[int, float] = {}
    best = float("inf")
    for layer, metrics in reversed(items):
        best = min(best, _answer_energy(metrics, answer))
        future_best_energy[int(layer)] = best

    previous_metrics: Dict[str, Any] | None = None
    previous_pred = ""
    stable_count = 0
    rows: List[Dict[str, Any]] = []
    for layer, metrics in items:
        pred = str(metrics.get("pred", ""))
        stable_count = stable_count + 1 if pred == previous_pred else 1
        if int(layer) >= int(min_exit_layer) and int(layer) < final_layer:
            feats = _features_for_layer(
                layer=int(layer),
                final_layer=final_layer,
                metrics=metrics,
                previous_metrics=previous_metrics,
                stable_count=stable_count,
                num_choices=int(record.get("num_choices", 4)),
            )
            current_energy = _answer_energy(metrics, answer)
            future_gain = max(0.0, current_energy - future_best_energy.get(int(layer), current_energy))
            rows.append({
                "features": feats,
                "label": float(str(pred).strip() != answer),
                "gain_label": float(future_gain),
                "dataset": record.get("dataset", ""),
                "layer": int(layer),
            })
        previous_metrics = dict(metrics)
        previous_pred = pred
    return rows


def _matrix(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
):
    x = torch.tensor([[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows], dtype=torch.float32)
    y = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32).view(-1, 1)
    if mean is None:
        mean = x.mean(dim=0)
    if std is None:
        std = x.std(dim=0).clamp_min(1e-6)
    return (x - mean) / std, y, mean, std


def _risk_gain_matrix(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
):
    x = torch.tensor([[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows], dtype=torch.float32)
    y_risk = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32).view(-1, 1)
    y_gain = torch.tensor([math.log1p(max(0.0, float(row.get("gain_label", 0.0)))) for row in rows], dtype=torch.float32).view(-1, 1)
    if mean is None:
        mean = x.mean(dim=0)
    if std is None:
        std = x.std(dim=0).clamp_min(1e-6)
    return (x - mean) / std, y_risk, y_gain, mean, std


def _prob(features: Dict[str, float], mean: Sequence[float], std: Sequence[float], weights: Sequence[float], bias: float) -> float:
    z = float(bias)
    for idx, name in enumerate(FEATURE_NAMES):
        z += float(weights[idx]) * ((float(features.get(name, 0.0)) - float(mean[idx])) / max(1e-12, float(std[idx])))
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


def _controller_probability(controller: Dict[str, Any], features: Dict[str, float]) -> float:
    names = [str(x) for x in controller["feature_names"]]
    mean = [float(x) for x in controller["mean"]]
    std = [max(1e-12, float(x)) for x in controller["std"]]
    vector = [(float(features.get(name, 0.0)) - mean[idx]) / std[idx] for idx, name in enumerate(names)]
    if str(controller.get("policy_kind", "")).lower() == "risk_gain":
        risk_logit = _head_logit(controller["risk_head"], vector)
        gain_log_value = _head_logit(controller["gain_head"], vector)
        risk = _sigmoid(risk_logit)
        gain = max(0.0, math.expm1(max(-20.0, min(20.0, gain_log_value))))
        remaining = float(features.get("layer_remaining_frac", 0.0))
        return float(
            float(controller.get("risk_limit", 0.01))
            - risk
            - float(controller.get("gain_weight", 1.0)) * gain
            + float(controller.get("cost_weight", 0.03)) * remaining
        )
    kind = str(controller.get("architecture", {}).get("kind", controller.get("model_type", "linear"))).lower()
    if kind == "mlp":
        hidden = vector
        layers = controller.get("layers", [])
        for layer_idx, layer in enumerate(layers):
            weight = [[float(v) for v in row] for row in layer["weight"]]
            bias = [float(v) for v in layer["bias"]]
            out: List[float] = []
            for row, b in zip(weight, bias):
                out.append(float(b) + sum(float(w) * float(x) for w, x in zip(row, hidden)))
            if layer_idx < len(layers) - 1:
                activation = str(layer.get("activation", "relu")).lower()
                if activation == "tanh":
                    hidden = [math.tanh(x) for x in out]
                else:
                    hidden = [max(0.0, x) for x in out]
            else:
                hidden = out
        if len(hidden) != 1:
            raise ValueError("MLP controller final layer must produce one logit")
        z = float(hidden[0])
    else:
        z = float(controller["bias"])
        weights = [float(x) for x in controller["weights"]]
        for idx in range(len(names)):
            z += weights[idx] * vector[idx]
    return _sigmoid(z)


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))


def _head_logit(head: Dict[str, Any], vector: Sequence[float]) -> float:
    kind = str(head.get("architecture", {}).get("kind", head.get("model_type", "linear"))).lower()
    if kind == "mlp":
        hidden = [float(x) for x in vector]
        layers = head.get("layers", [])
        for layer_idx, layer in enumerate(layers):
            weight = [[float(v) for v in row] for row in layer["weight"]]
            bias = [float(v) for v in layer["bias"]]
            out = [float(b) + sum(float(w) * float(x) for w, x in zip(row, hidden)) for row, b in zip(weight, bias)]
            if layer_idx < len(layers) - 1:
                hidden = [max(0.0, x) for x in out]
            else:
                hidden = out
        if len(hidden) != 1:
            raise ValueError("risk_gain MLP head final layer must produce one value")
        return float(hidden[0])
    weights = [float(x) for x in head["weights"]]
    z = float(head["bias"])
    for idx, value in enumerate(vector):
        z += weights[idx] * float(value)
    return float(z)


def _linear_payload(model: nn.Linear, mean: Sequence[float], std: Sequence[float]) -> Dict[str, Any]:
    return {
        "model_type": "linear",
        "architecture": {"kind": "linear"},
        "weights": [float(x) for x in model.weight.detach().cpu().view(-1).tolist()],
        "bias": float(model.bias.detach().cpu().view(-1)[0].item()),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }


def _linear_head_payload(model: nn.Linear) -> Dict[str, Any]:
    return {
        "model_type": "linear",
        "architecture": {"kind": "linear"},
        "weights": [float(x) for x in model.weight.detach().cpu().view(-1).tolist()],
        "bias": float(model.bias.detach().cpu().view(-1)[0].item()),
    }


def _mlp_payload(model: ExitMLP, mean: Sequence[float], std: Sequence[float], hidden_dim: int, dropout: float) -> Dict[str, Any]:
    first = model.net[0]
    last = model.net[3]
    assert isinstance(first, nn.Linear)
    assert isinstance(last, nn.Linear)
    return {
        "model_type": "mlp",
        "architecture": {
            "kind": "mlp",
            "hidden_dim": int(hidden_dim),
            "activation": "relu",
            "dropout": float(dropout),
        },
        "layers": [
            {
                "name": "hidden",
                "activation": "relu",
                "weight": [[float(v) for v in row] for row in first.weight.detach().cpu().tolist()],
                "bias": [float(v) for v in first.bias.detach().cpu().tolist()],
            },
            {
                "name": "output",
                "activation": "identity",
                "weight": [[float(v) for v in row] for row in last.weight.detach().cpu().tolist()],
                "bias": [float(v) for v in last.bias.detach().cpu().tolist()],
            },
        ],
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }


def _mlp_head_payload(model: ExitMLP, hidden_dim: int, dropout: float) -> Dict[str, Any]:
    payload = _mlp_payload(model, [], [], hidden_dim, dropout)
    payload.pop("mean", None)
    payload.pop("std", None)
    return payload


def _simulate(records: Sequence[Dict[str, Any]], thresholds_by_dataset: Dict[str, float], *,
              min_exit_layer: int, decision_threshold: float,
              controller: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    correct = 0
    oracle_correct = 0
    full_correct = 0
    layer_sum = 0
    hist: Counter[str] = Counter()
    dataset_totals: Counter[str] = Counter()
    dataset_correct: Counter[str] = Counter()
    dataset_full_correct: Counter[str] = Counter()
    for record in records:
        dataset = str(record.get("dataset", ""))
        oracle_layer, oracle_pred = _choose_by_threshold(record, thresholds_by_dataset[dataset], min_exit_layer)
        items = _layer_metrics(record)
        answer = str(record.get("answer", "")).strip()
        final_layer = int(items[-1][0])
        previous_metrics: Dict[str, Any] | None = None
        previous_pred = ""
        stable_count = 0
        chosen_layer = final_layer
        chosen_pred = str(items[-1][1].get("pred", ""))
        for layer, metrics in items:
            pred = str(metrics.get("pred", ""))
            stable_count = stable_count + 1 if pred == previous_pred else 1
            if int(layer) >= min_exit_layer and int(layer) < final_layer:
                feats = _features_for_layer(
                    layer=int(layer),
                    final_layer=final_layer,
                    metrics=metrics,
                    previous_metrics=previous_metrics,
                    stable_count=stable_count,
                    num_choices=int(record.get("num_choices", 4)),
                )
                if _controller_probability(controller, feats) >= float(decision_threshold):
                    chosen_layer = int(layer)
                    chosen_pred = pred
                    break
            previous_metrics = dict(metrics)
            previous_pred = pred
        total += 1
        correct += int(chosen_pred == answer)
        oracle_correct += int(str(oracle_pred) == answer)
        full_correct += int(str(items[-1][1].get("pred", "")) == answer)
        layer_sum += int(chosen_layer)
        hist[str(chosen_layer)] += 1
        dataset_totals[dataset] += 1
        dataset_correct[dataset] += int(chosen_pred == answer)
        dataset_full_correct[dataset] += int(str(items[-1][1].get("pred", "")) == answer)
    safe_total = max(1, total)
    per_dataset = {
        dataset: {
            "samples": int(dataset_totals[dataset]),
            "accuracy": float(dataset_correct[dataset] / max(1, dataset_totals[dataset])),
            "full_accuracy": float(dataset_full_correct[dataset] / max(1, dataset_totals[dataset])),
        }
        for dataset in sorted(dataset_totals)
    }
    macro_accuracy = sum(float(row["accuracy"]) for row in per_dataset.values()) / max(1, len(per_dataset))
    full_macro_accuracy = sum(float(row["full_accuracy"]) for row in per_dataset.values()) / max(1, len(per_dataset))
    return {
        "samples": int(total),
        "accuracy": float(correct / safe_total),
        "oracle_accuracy": float(oracle_correct / safe_total),
        "full_accuracy": float(full_correct / safe_total),
        "avg_exit_layer": float(layer_sum / safe_total),
        "estimated_layer_savings": float(1.0 - (layer_sum / safe_total) / 28.0),
        "exit_layer_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "macro_accuracy": float(macro_accuracy),
        "full_macro_accuracy": float(full_macro_accuracy),
        "per_dataset": per_dataset,
    }


def _simulate_risk_gain(records: Sequence[Dict[str, Any]], *, min_exit_layer: int, decision_threshold: float,
                        controller: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    correct = 0
    full_correct = 0
    layer_sum = 0
    hist: Counter[str] = Counter()
    dataset_totals: Counter[str] = Counter()
    dataset_correct: Counter[str] = Counter()
    dataset_full_correct: Counter[str] = Counter()
    for record in records:
        items = _layer_metrics(record)
        answer = str(record.get("answer", "")).strip()
        final_layer = int(items[-1][0])
        previous_metrics: Dict[str, Any] | None = None
        previous_pred = ""
        stable_count = 0
        chosen_layer = final_layer
        chosen_pred = str(items[-1][1].get("pred", ""))
        for layer, metrics in items:
            pred = str(metrics.get("pred", ""))
            stable_count = stable_count + 1 if pred == previous_pred else 1
            if int(layer) >= int(min_exit_layer) and int(layer) < final_layer:
                feats = _features_for_layer(
                    layer=int(layer),
                    final_layer=final_layer,
                    metrics=metrics,
                    previous_metrics=previous_metrics,
                    stable_count=stable_count,
                    num_choices=int(record.get("num_choices", 4)),
                )
                if _controller_probability(controller, feats) >= float(decision_threshold):
                    chosen_layer = int(layer)
                    chosen_pred = pred
                    break
            previous_metrics = dict(metrics)
            previous_pred = pred
        total += 1
        correct += int(chosen_pred == answer)
        full_correct += int(str(items[-1][1].get("pred", "")) == answer)
        layer_sum += int(chosen_layer)
        hist[str(chosen_layer)] += 1
        dataset = str(record.get("dataset", ""))
        dataset_totals[dataset] += 1
        dataset_correct[dataset] += int(chosen_pred == answer)
        dataset_full_correct[dataset] += int(str(items[-1][1].get("pred", "")) == answer)
    safe_total = max(1, total)
    accuracy = float(correct / safe_total)
    per_dataset = {
        dataset: {
            "samples": int(dataset_totals[dataset]),
            "accuracy": float(dataset_correct[dataset] / max(1, dataset_totals[dataset])),
            "full_accuracy": float(dataset_full_correct[dataset] / max(1, dataset_totals[dataset])),
        }
        for dataset in sorted(dataset_totals)
    }
    macro_accuracy = sum(float(row["accuracy"]) for row in per_dataset.values()) / max(1, len(per_dataset))
    full_macro_accuracy = sum(float(row["full_accuracy"]) for row in per_dataset.values()) / max(1, len(per_dataset))
    return {
        "samples": int(total),
        "accuracy": accuracy,
        "oracle_accuracy": accuracy,
        "full_accuracy": float(full_correct / safe_total),
        "avg_exit_layer": float(layer_sum / safe_total),
        "estimated_layer_savings": float(1.0 - (layer_sum / safe_total) / 28.0),
        "exit_layer_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "macro_accuracy": float(macro_accuracy),
        "full_macro_accuracy": float(full_macro_accuracy),
        "per_dataset": per_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an oracle-distilled exit policy from train-split calibration records.")
    parser.add_argument("--records_glob", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa")
    parser.add_argument("--thresholds", type=str, default="0.50:0.995:0.005")
    parser.add_argument("--min_exit_layer", type=int, default=16)
    parser.add_argument("--max_acc_drop", type=float, default=0.002)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--auto_decision_threshold", action="store_true")
    parser.add_argument("--decision_threshold_grid", type=str, default="0.30:0.90:0.025")
    parser.add_argument("--auto_max_acc_drop", type=float, default=0.002)
    parser.add_argument(
        "--auto_accuracy_metric",
        choices=["weighted", "macro"],
        default="weighted",
        help="Accuracy constraint used to select the controller threshold.",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.0,
        help=(
            "Per-dataset held-out fraction used only for controller-threshold selection. "
            "Zero preserves the legacy same-record calibration behavior."
        ),
    )
    parser.add_argument(
        "--auto_decision_objective",
        choices=["accuracy_first", "speed_constrained"],
        default="accuracy_first",
        help=(
            "How to select the controller decision threshold on calibration data. "
            "accuracy_first preserves the previous behavior. speed_constrained "
            "maximizes estimated layer savings subject to --auto_max_acc_drop."
        ),
    )
    parser.add_argument("--model_type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--policy_kind", choices=["oracle_distill", "risk_gain"], default="oracle_distill")
    parser.add_argument("--feature_set", choices=["base", "ie4", "ie5", "ie6", "ie6_depth", "ie6_future", "useful4", "optstop5", "info_energy", "info_energy_layer"], default="base")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gain_loss_weight", type=float, default=0.35)
    parser.add_argument("--risk_limit", type=float, default=0.01)
    parser.add_argument("--gain_weight", type=float, default=1.0)
    parser.add_argument("--cost_weight", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    _ensure_dir(str(args.output_dir))
    wanted = set(_parse_list(str(args.datasets)))
    paths = sorted(glob.glob(str(args.records_glob)))
    if not paths:
        raise FileNotFoundError(str(args.records_glob))
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for path in paths:
        dataset = _dataset_from_records_path(path)
        if dataset not in wanted:
            continue
        by_dataset[dataset] = _read_records(path, dataset)
    missing = sorted(wanted - set(by_dataset))
    if missing:
        raise FileNotFoundError(f"missing records for datasets: {missing}")

    validation_fraction = float(args.validation_fraction)
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    train_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    heldout_records: List[Dict[str, Any]] = []
    split_counts: Dict[str, Dict[str, int]] = {}
    for dataset_index, dataset in enumerate(sorted(by_dataset)):
        records = list(by_dataset[dataset])
        if validation_fraction <= 0.0:
            train_by_dataset[dataset] = records
            split_counts[dataset] = {"train": len(records), "validation": 0}
            continue
        if len(records) < 2:
            raise ValueError(f"dataset {dataset} needs at least two records for held-out calibration")
        indices = list(range(len(records)))
        random.Random(int(args.seed) + 1009 * (dataset_index + 1)).shuffle(indices)
        validation_count = min(len(records) - 1, max(1, int(round(len(records) * validation_fraction))))
        validation_indices = set(indices[:validation_count])
        train_records = [record for idx, record in enumerate(records) if idx not in validation_indices]
        validation_records = [record for idx, record in enumerate(records) if idx in validation_indices]
        train_by_dataset[dataset] = train_records
        heldout_records.extend(validation_records)
        split_counts[dataset] = {"train": len(train_records), "validation": len(validation_records)}
        print(
            f"[OracleDistill] split {dataset}: train={len(train_records)} "
            f"validation={len(validation_records)}",
            flush=True,
        )

    thresholds = _threshold_grid(str(args.thresholds))
    oracle: Dict[str, Any] = {}
    thresholds_by_dataset: Dict[str, float] = {}
    train_rows: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    for dataset, records in train_by_dataset.items():
        found = _find_oracle_threshold(records, thresholds, min_exit_layer=int(args.min_exit_layer), max_acc_drop=float(args.max_acc_drop))
        selected = found["selected"]
        tau = float(selected["threshold"])
        thresholds_by_dataset[dataset] = tau
        oracle[dataset] = selected
        all_records.extend(records)
        for record in records:
            if str(args.policy_kind) == "risk_gain":
                train_rows.extend(_risk_gain_rows(record, int(args.min_exit_layer)))
            else:
                oracle_layer, _ = _choose_by_threshold(record, tau, int(args.min_exit_layer))
                train_rows.extend(_policy_rows(record, oracle_layer, int(args.min_exit_layer)))
        print(
            f"[OracleDistill] {dataset} tau={tau:.3f} acc={float(selected['accuracy']):.4f} "
            f"full={float(selected['full_accuracy']):.4f} avg_layer={float(selected['avg_exit_layer']):.2f} "
            f"savings={float(selected['estimated_layer_savings']):.4f}",
            flush=True,
        )

    selected_feature_names = _feature_names(str(args.feature_set))
    if str(args.policy_kind) == "risk_gain":
        x_train, y_risk, y_gain, mean_t, std_t = _risk_gain_matrix(train_rows, selected_feature_names)
        if str(args.model_type).lower() == "mlp":
            risk_model: nn.Module = ExitMLP(int(x_train.shape[1]), int(args.hidden_dim), float(args.dropout))
            gain_model: nn.Module = ExitMLP(int(x_train.shape[1]), int(args.hidden_dim), float(args.dropout))
        else:
            risk_model = nn.Linear(x_train.shape[1], 1)
            gain_model = nn.Linear(x_train.shape[1], 1)
        pos = float(y_risk.sum().item())
        neg = float(y_risk.numel() - pos)
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
        opt = torch.optim.AdamW(
            list(risk_model.parameters()) + list(gain_model.parameters()),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        for _ in range(int(args.epochs)):
            risk_model.train()
            gain_model.train()
            opt.zero_grad(set_to_none=True)
            risk_loss = F.binary_cross_entropy_with_logits(risk_model(x_train), y_risk, pos_weight=pos_weight)
            gain_loss = F.smooth_l1_loss(gain_model(x_train), y_gain)
            loss = risk_loss + float(args.gain_loss_weight) * gain_loss
            loss.backward()
            opt.step()

        with torch.no_grad():
            risk_model.eval()
            gain_model.eval()
            risk_prob = torch.sigmoid(risk_model(x_train))
            cls_acc = float(((risk_prob >= 0.5).float() == y_risk).float().mean().item())
            pred_gain = torch.expm1(gain_model(x_train).clamp(-20.0, 20.0)).clamp_min(0.0)
            gain_mae = float((pred_gain - torch.expm1(y_gain)).abs().mean().item())
            mean = [float(x) for x in mean_t.detach().cpu().tolist()]
            std = [float(x) for x in std_t.detach().cpu().tolist()]
            if str(args.model_type).lower() == "mlp":
                assert isinstance(risk_model, ExitMLP)
                assert isinstance(gain_model, ExitMLP)
                risk_head = _mlp_head_payload(risk_model, int(args.hidden_dim), float(args.dropout))
                gain_head = _mlp_head_payload(gain_model, int(args.hidden_dim), float(args.dropout))
            else:
                assert isinstance(risk_model, nn.Linear)
                assert isinstance(gain_model, nn.Linear)
                risk_head = _linear_head_payload(risk_model)
                gain_head = _linear_head_payload(gain_model)
            learned_payload = {
                "model_type": str(args.model_type),
                "architecture": {"kind": "risk_gain", "head_kind": str(args.model_type)},
                "mean": mean,
                "std": std,
                "risk_head": risk_head,
                "gain_head": gain_head,
                "risk_limit": float(args.risk_limit),
                "gain_weight": float(args.gain_weight),
                "cost_weight": float(args.cost_weight),
                "gain_target": "max_future_answer_energy_drop_log1p",
                "gain_train_mae": gain_mae,
            }
    else:
        x_train, y_train, mean_t, std_t = _matrix(train_rows, selected_feature_names)
        if str(args.model_type).lower() == "mlp":
            model: nn.Module = ExitMLP(int(x_train.shape[1]), int(args.hidden_dim), float(args.dropout))
        else:
            model = nn.Linear(x_train.shape[1], 1)
        pos = float(y_train.sum().item())
        neg = float(y_train.numel() - pos)
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        for _ in range(int(args.epochs)):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x_train), y_train, pos_weight=pos_weight)
            loss.backward()
            opt.step()

        with torch.no_grad():
            model.eval()
            prob = torch.sigmoid(model(x_train))
            cls_acc = float(((prob >= float(args.decision_threshold)).float() == y_train).float().mean().item())
            mean = [float(x) for x in mean_t.detach().cpu().tolist()]
            std = [float(x) for x in std_t.detach().cpu().tolist()]
            if str(args.model_type).lower() == "mlp":
                assert isinstance(model, ExitMLP)
                learned_payload = _mlp_payload(model, mean, std, int(args.hidden_dim), float(args.dropout))
            else:
                assert isinstance(model, nn.Linear)
                learned_payload = _linear_payload(model, mean, std)

    decision_records = heldout_records or all_records
    sim_rows: List[Dict[str, Any]] = []
    base_controller = {
        "type": "oracle_distilled_exit_policy",
        "policy_kind": str(args.policy_kind),
        "trained_on": "7task_train_calibration",
        "feature_names": selected_feature_names,
        "feature_set": str(args.feature_set),
        "decision_threshold": float(args.decision_threshold),
        **learned_payload,
    }
    for threshold in _threshold_grid(str(args.decision_threshold_grid)):
        if str(args.policy_kind) == "risk_gain":
            sim_metrics = _simulate_risk_gain(
                decision_records,
                min_exit_layer=int(args.min_exit_layer),
                decision_threshold=threshold,
                controller=base_controller,
            )
        else:
            sim_metrics = _simulate(
                decision_records,
                thresholds_by_dataset,
                min_exit_layer=int(args.min_exit_layer),
                decision_threshold=threshold,
                controller=base_controller,
            )
        sim_rows.append({
            "decision_threshold": threshold,
            **sim_metrics,
        })
    sweep_csv = os.path.join(str(args.output_dir), "oracle_distilled_policy_decision_sweep.csv")
    with open(sweep_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sim_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sim_rows)

    if bool(args.auto_decision_threshold):
        accuracy_key = "macro_accuracy" if str(args.auto_accuracy_metric) == "macro" else "accuracy"
        full_accuracy_key = "full_macro_accuracy" if str(args.auto_accuracy_metric) == "macro" else "full_accuracy"
        feasible = [
            row for row in sim_rows
            if float(row[accuracy_key]) >= float(row[full_accuracy_key]) - float(args.auto_max_acc_drop)
        ]
        if str(args.auto_decision_objective) == "speed_constrained":
            selected_row = max(
                feasible or sim_rows,
                key=lambda r: (
                    float(r["estimated_layer_savings"]),
                    float(r[accuracy_key]),
                    -float(r["decision_threshold"]),
                ),
            )
        else:
            selected_row = max(
                feasible or sim_rows,
                key=lambda r: (
                    float(r[accuracy_key]),
                    float(r["estimated_layer_savings"]),
                    -float(r["decision_threshold"]),
                ),
            )
        selected_decision_threshold = float(selected_row["decision_threshold"])
    else:
        selected_decision_threshold = float(args.decision_threshold)

    if str(args.policy_kind) == "risk_gain":
        selected_sim = _simulate_risk_gain(
            decision_records,
            min_exit_layer=int(args.min_exit_layer),
            decision_threshold=selected_decision_threshold,
            controller=base_controller,
        )
    else:
        selected_sim = _simulate(
            decision_records,
            thresholds_by_dataset,
            min_exit_layer=int(args.min_exit_layer),
            decision_threshold=selected_decision_threshold,
            controller=base_controller,
        )
    controller = {
        "type": "oracle_distilled_exit_policy",
        "policy_kind": str(args.policy_kind),
        "trained_on": "7task_train_calibration",
        "feature_names": selected_feature_names,
        "feature_set": str(args.feature_set),
        **learned_payload,
        "decision_threshold": selected_decision_threshold,
        "decision_threshold_source": "auto_calibration" if bool(args.auto_decision_threshold) else "argument",
        "auto_decision_objective": str(args.auto_decision_objective),
        "auto_accuracy_metric": str(args.auto_accuracy_metric),
        "decision_threshold_grid": _threshold_grid(str(args.decision_threshold_grid)),
        "auto_max_acc_drop": float(args.auto_max_acc_drop),
        "validation_fraction": validation_fraction,
        "calibration_split_counts": split_counts,
        "threshold_selection_records": int(len(decision_records)),
        "min_exit_layer": int(args.min_exit_layer),
        "teacher_oracle_thresholds_by_dataset": thresholds_by_dataset,
        "teacher_oracle_metrics_by_dataset": oracle,
        "train_policy_examples": int(len(train_rows)),
        "train_classifier_accuracy": cls_acc,
        "calibration_simulation": selected_sim,
        "decision_sweep_csv": sweep_csv,
    }
    out_json = os.path.join(str(args.output_dir), "oracle_distilled_exit_policy.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(controller, handle, ensure_ascii=False, indent=2)
    print(
        f"[OracleDistill] saved {out_json} train_policy_examples={len(train_rows)} "
        f"calib_acc={selected_sim['accuracy']:.4f} oracle_acc={selected_sim['oracle_accuracy']:.4f} "
        f"avg_layer={selected_sim['avg_exit_layer']:.2f} savings={selected_sim['estimated_layer_savings']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
