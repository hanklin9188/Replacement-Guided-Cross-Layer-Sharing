#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import newthesis_pipeline_final_llama as pipe
import fad_weight_quantization as fad_quant


def _parse_csv_list(value: str) -> List[str]:
    out: List[str] = []
    for chunk in str(value or "").replace(",", " ").split():
        item = chunk.strip()
        if item:
            out.append(item)
    return out


def _parse_int_list(value: str) -> List[int]:
    vals: List[int] = []
    seen = set()
    for item in _parse_csv_list(value):
        try:
            ivalue = int(item)
        except Exception:
            continue
        if ivalue not in seen:
            vals.append(ivalue)
            seen.add(ivalue)
    return vals


def _parse_dataset_thresholds(value: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for chunk in str(value or "").replace(",", " ").split():
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid dataset threshold entry {item!r}; expected dataset=threshold.")
        name, raw_threshold = item.split("=", 1)
        dataset = name.strip()
        if not dataset:
            raise ValueError(f"Invalid dataset threshold entry {item!r}; dataset is empty.")
        mapping[dataset] = float(raw_threshold)
    return mapping


def _normalize_answer(answer_raw: Any, candidates: Sequence[str]) -> str:
    if isinstance(answer_raw, int):
        idx = int(answer_raw)
        return str(candidates[idx]) if 0 <= idx < len(candidates) else str(answer_raw)
    answer = str(answer_raw).strip()
    if answer.isdigit():
        idx = int(answer)
        if 0 <= idx < len(candidates):
            return str(candidates[idx])
    return answer


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _candidate_token_ids(tokenizer, candidates: Sequence[str]) -> List[List[int]]:
    out: List[List[int]] = []
    for cand in candidates:
        ids = tokenizer.encode(" " + str(cand), add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(str(cand), add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"candidate tokenization is empty: {cand!r}")
        out.append([int(x) for x in ids])
    return out


def _build_candidate_batch(
    *,
    tokenizer,
    prompt: str,
    candidates: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].to(device)
    base_len = int(prompt_ids.numel())
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    cand_ids_list = _candidate_token_ids(tokenizer, candidates)
    full_sequences: List[torch.Tensor] = []
    lengths: List[int] = []
    for ids in cand_ids_list:
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        seq = torch.cat([prompt_ids, ids_t], dim=0)
        full_sequences.append(seq)
        lengths.append(int(seq.numel()))

    max_len = max(lengths)
    input_ids = torch.full((len(candidates), max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(candidates), max_len), dtype=torch.long, device=device)
    for row_id, seq in enumerate(full_sequences):
        input_ids[row_id, : int(seq.numel())] = seq
        attention_mask[row_id, : int(seq.numel())] = 1

    batch_indices: List[int] = []
    token_indices: List[int] = []
    target_ids: List[int] = []
    candidate_position_ids: List[int] = []
    candidate_lengths = [max(1, len(ids)) for ids in cand_ids_list]
    for cand_idx, ids in enumerate(cand_ids_list):
        for j, target_id in enumerate(ids):
            batch_indices.append(int(cand_idx))
            token_indices.append(int(base_len + j - 1))
            target_ids.append(int(target_id))
            candidate_position_ids.append(int(cand_idx))

    return (
        input_ids,
        attention_mask,
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        torch.tensor(token_indices, dtype=torch.long, device=device),
        torch.tensor(target_ids, dtype=torch.long, device=device),
        candidate_lengths,
    )


def _scores_from_selected_hidden(
    *,
    model,
    selected_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    candidate_position_ids: Sequence[int],
    candidates: Sequence[str],
    candidate_lengths: Sequence[int],
    length_norm: str,
) -> Dict[str, float]:
    root = getattr(model, "model", model)
    hidden = root.norm(selected_hidden)
    logits = model.lm_head(hidden)
    token_logp = F.log_softmax(logits.float(), dim=-1).gather(1, target_ids.view(-1, 1)).squeeze(1)
    sums = torch.zeros((len(candidates),), dtype=torch.float64, device=selected_hidden.device)
    for pos_idx, cand_idx in enumerate(candidate_position_ids):
        sums[int(cand_idx)] += token_logp[int(pos_idx)].double()
    if str(length_norm).strip().lower() == "avg":
        for cand_idx, cand_len in enumerate(candidate_lengths):
            sums[int(cand_idx)] /= max(1, int(cand_len))
    return {str(candidates[i]): float(sums[i].detach().cpu().item()) for i in range(len(candidates))}


def _metrics_from_scores(candidates: Sequence[str], scores: Dict[str, float], temperature: float) -> Dict[str, Any]:
    vals = torch.tensor([float(scores[str(c)]) for c in candidates], dtype=torch.float64)
    probs = torch.softmax(vals / max(1e-6, float(temperature)), dim=-1)
    order = torch.argsort(vals, descending=True)
    best_idx = int(order[0].item())
    second_idx = int(order[1].item()) if len(candidates) > 1 else best_idx
    confidence = float(probs[best_idx].item())
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12)))).sum().item())
    pred = str(candidates[best_idx])
    return {
        "pred": pred,
        "confidence": confidence,
        "margin": float((vals[best_idx] - vals[second_idx]).item()),
        "entropy": entropy,
        "scores": {str(candidates[i]): float(vals[i].item()) for i in range(len(candidates))},
        "probs": {str(candidates[i]): float(probs[i].item()) for i in range(len(candidates))},
    }


def _load_exit_controller(path: str) -> Dict[str, Any]:
    if not str(path or "").strip():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"feature_names", "mean", "std", "decision_threshold"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"controller json is missing keys: {missing}")
    if str(payload.get("policy_kind", "")).lower() == "risk_gain":
        head_missing = sorted({"risk_head", "gain_head"} - set(payload.keys()))
        if head_missing:
            raise ValueError(f"risk_gain controller json is missing keys: {head_missing}")
        return payload
    kind = str(payload.get("architecture", {}).get("kind", payload.get("model_type", "linear"))).lower()
    if kind == "mlp":
        if "layers" not in payload:
            raise ValueError("MLP controller json is missing key: layers")
    else:
        linear_missing = sorted({"weights", "bias"} - set(payload.keys()))
        if linear_missing:
            raise ValueError(f"linear controller json is missing keys: {linear_missing}")
    return payload


def _apply_controller_threshold_override(controller: Dict[str, Any], override: float) -> Dict[str, Any]:
    if not controller or float(override) < 0.0:
        return controller
    out = dict(controller)
    out["decision_threshold_original"] = float(controller.get("decision_threshold", 0.0))
    out["decision_threshold"] = float(override)
    out["decision_threshold_source"] = "runtime_override"
    return out


def _controller_probability(controller: Dict[str, Any], features: Dict[str, float]) -> float:
    names = [str(x) for x in controller["feature_names"]]
    mean = [float(x) for x in controller["mean"]]
    std = [max(1e-12, float(x)) for x in controller["std"]]
    if not (len(names) == len(mean) == len(std)):
        raise ValueError("controller feature_names/mean/std length mismatch")
    vector = [(float(features.get(name, 0.0)) - mean[idx]) / std[idx] for idx, name in enumerate(names)]
    if str(controller.get("policy_kind", "")).lower() == "risk_gain":
        return _risk_gain_controller_score(controller, features, vector)["score"]
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
        logit = float(hidden[0])
    else:
        weights = [float(x) for x in controller["weights"]]
        if len(weights) != len(names):
            raise ValueError("controller feature_names/weights length mismatch")
        logit = float(controller["bias"])
        for idx, name in enumerate(names):
            logit += weights[idx] * vector[idx]
    if logit >= 0.0:
        z = math.exp(-logit)
        return float(1.0 / (1.0 + z))
    z = math.exp(logit)
    return float(z / (1.0 + z))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return float(1.0 / (1.0 + z))
    z = math.exp(value)
    return float(z / (1.0 + z))


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
    if len(weights) != len(vector):
        raise ValueError("risk_gain head weight/vector length mismatch")
    logit = float(head["bias"])
    for idx, value in enumerate(vector):
        logit += weights[idx] * float(value)
    return float(logit)


def _risk_gain_controller_score(controller: Dict[str, Any], features: Dict[str, float], vector: Sequence[float]) -> Dict[str, float]:
    risk_logit = _head_logit(controller["risk_head"], vector)
    gain_log_value = _head_logit(controller["gain_head"], vector)
    risk = _sigmoid(risk_logit)
    expected_gain = max(0.0, math.expm1(max(-20.0, min(20.0, gain_log_value))))
    remaining = float(features.get("layer_remaining_frac", 0.0))
    score = (
        float(controller.get("risk_limit", 0.01))
        - risk
        - float(controller.get("gain_weight", 1.0)) * expected_gain
        + float(controller.get("cost_weight", 0.03)) * remaining
    )
    return {
        "score": float(score),
        "risk": float(risk),
        "expected_future_gain": float(expected_gain),
        "remaining_compute_frac": float(remaining),
    }


def _controller_score_details(controller: Dict[str, Any], features: Dict[str, float]) -> Dict[str, float]:
    names = [str(x) for x in controller["feature_names"]]
    mean = [float(x) for x in controller["mean"]]
    std = [max(1e-12, float(x)) for x in controller["std"]]
    vector = [(float(features.get(name, 0.0)) - mean[idx]) / std[idx] for idx, name in enumerate(names)]
    if str(controller.get("policy_kind", "")).lower() == "risk_gain":
        return _risk_gain_controller_score(controller, features, vector)
    return {"exit_probability": _controller_probability(controller, features)}


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


def _controller_features(
    *,
    layer: int,
    final_layer: int,
    metrics: Dict[str, Any],
    previous_metrics: Dict[str, Any] | None,
    stable_count: int,
    num_choices: int,
) -> Dict[str, float]:
    confidence = float(metrics.get("confidence", 0.0))
    margin = float(metrics.get("margin", 0.0))
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
        prev_margin = float(previous_metrics.get("margin", 0.0))
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


@torch.inference_mode()
def runtime_confidence_predict(
    *,
    model,
    tokenizer,
    prompt: str,
    candidates: Sequence[str],
    exit_layers: Sequence[int],
    threshold: float,
    device: torch.device,
    length_norm: str,
    temperature: float,
    force_full: bool = False,
    controller: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    (
        input_ids,
        attention_mask,
        batch_indices,
        token_indices,
        target_ids,
        candidate_lengths,
    ) = _build_candidate_batch(tokenizer=tokenizer, prompt=prompt, candidates=candidates, device=device)
    candidate_position_ids: List[int] = []
    for cand_idx, cand_len in enumerate(candidate_lengths):
        candidate_position_ids.extend([int(cand_idx)] * int(cand_len))

    root = getattr(model, "model", model)
    layers = list(root.layers[: root.config.num_hidden_layers])
    final_layer = int(len(layers))
    exit_set = {int(x) for x in exit_layers if 0 < int(x) < final_layer}
    if bool(force_full):
        exit_set = set()

    hidden_states = root.embed_tokens(input_ids)
    position_ids = torch.arange(hidden_states.shape[1], device=device).unsqueeze(0)
    causal_mask = create_causal_mask(
        config=root.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = root.rotary_emb(hidden_states, position_ids=position_ids)

    chosen_layer = final_layer
    chosen_metrics: Dict[str, Any] = {}
    checked_layers: List[int] = []
    controller_trace: List[Dict[str, Any]] = []
    previous_metrics: Dict[str, Any] | None = None
    previous_pred = ""
    stable_count = 0
    active_controller = controller or {}
    for layer_idx, decoder_layer in enumerate(layers, start=1):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        should_check = int(layer_idx) in exit_set or int(layer_idx) == final_layer
        if not should_check:
            continue
        selected = hidden_states[batch_indices, token_indices, :]
        scores = _scores_from_selected_hidden(
            model=model,
            selected_hidden=selected,
            target_ids=target_ids,
            candidate_position_ids=candidate_position_ids,
            candidates=candidates,
            candidate_lengths=candidate_lengths,
            length_norm=length_norm,
        )
        metrics = _metrics_from_scores(candidates, scores, float(temperature))
        metrics["layer"] = int(layer_idx)
        checked_layers.append(int(layer_idx))
        chosen_layer = int(layer_idx)
        chosen_metrics = metrics
        pred = str(metrics["pred"])
        stable_count = stable_count + 1 if pred == previous_pred else 1
        should_exit = False
        if int(layer_idx) in exit_set:
            if active_controller:
                features = _controller_features(
                    layer=int(layer_idx),
                    final_layer=final_layer,
                    metrics=metrics,
                    previous_metrics=previous_metrics,
                    stable_count=stable_count,
                    num_choices=len(candidates),
                )
                exit_prob = _controller_probability(active_controller, features)
                decision_threshold = float(active_controller["decision_threshold"])
                should_exit = bool(exit_prob >= decision_threshold)
                trace_row = {
                    "layer": int(layer_idx),
                    "exit_probability": float(exit_prob),
                    "decision_threshold": decision_threshold,
                    "features": features,
                }
                if str(active_controller.get("policy_kind", "")).lower() == "risk_gain":
                    details = _controller_score_details(active_controller, features)
                    trace_row.update({
                        "exit_score": float(details["score"]),
                        "risk": float(details["risk"]),
                        "expected_future_gain": float(details["expected_future_gain"]),
                        "remaining_compute_frac": float(details["remaining_compute_frac"]),
                    })
                controller_trace.append(trace_row)
            else:
                should_exit = bool(float(metrics["confidence"]) >= float(threshold))
        previous_metrics = dict(metrics)
        previous_pred = pred
        if should_exit:
            break

    if not chosen_metrics:
        raise RuntimeError("runtime confidence predictor produced no layer metrics.")
    out = {
        "pred": str(chosen_metrics["pred"]),
        "exit_layer": int(chosen_layer),
        "final_layer": int(final_layer),
        "confidence": float(chosen_metrics["confidence"]),
        "margin": float(chosen_metrics["margin"]),
        "entropy": float(chosen_metrics["entropy"]),
        "checked_layers": checked_layers,
        "scores": chosen_metrics["scores"],
        "probs": chosen_metrics["probs"],
        "forward_tokens": int(attention_mask.sum().item()),
        "layer_token_compute": int(attention_mask.sum().item()) * int(chosen_layer),
    }
    if active_controller:
        out["controller_trace"] = controller_trace
    return out


def _build_compacted_candidate_batch(
    *,
    tokenizer,
    prompts: Sequence[str],
    candidates: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    cand_ids_list = _candidate_token_ids(tokenizer, candidates)

    full_sequences: List[torch.Tensor] = []
    lengths: List[int] = []
    groups: List[Dict[str, Any]] = []
    for sample_id, prompt in enumerate(prompts):
        enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        prompt_ids = enc["input_ids"][0].to(device)
        base_len = int(prompt_ids.numel())
        row_ids: List[int] = []
        batch_indices: List[int] = []
        token_indices: List[int] = []
        target_ids: List[int] = []
        candidate_position_ids: List[int] = []
        candidate_lengths = [max(1, len(ids)) for ids in cand_ids_list]

        for cand_idx, ids in enumerate(cand_ids_list):
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            seq = torch.cat([prompt_ids, ids_t], dim=0)
            row_id = len(full_sequences)
            full_sequences.append(seq)
            lengths.append(int(seq.numel()))
            row_ids.append(int(row_id))
            for j, target_id in enumerate(ids):
                batch_indices.append(int(row_id))
                token_indices.append(int(base_len + j - 1))
                target_ids.append(int(target_id))
                candidate_position_ids.append(int(cand_idx))

        groups.append({
            "sample_id": int(sample_id),
            "row_ids": row_ids,
            "batch_indices": torch.tensor(batch_indices, dtype=torch.long, device=device),
            "token_indices": torch.tensor(token_indices, dtype=torch.long, device=device),
            "target_ids": torch.tensor(target_ids, dtype=torch.long, device=device),
            "candidate_position_ids": candidate_position_ids,
            "candidate_lengths": candidate_lengths,
            "forward_tokens": int(sum(lengths[row_id] for row_id in row_ids)),
            "checked_layers": [],
            "controller_trace": [],
            "previous_metrics": None,
            "previous_pred": "",
            "stable_count": 0,
        })

    if not full_sequences:
        raise RuntimeError("compacted batch received no prompts.")
    max_len = max(lengths)
    input_ids = torch.full((len(full_sequences), max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(full_sequences), max_len), dtype=torch.long, device=device)
    for row_id, seq in enumerate(full_sequences):
        input_ids[int(row_id), : int(seq.numel())] = seq
        attention_mask[int(row_id), : int(seq.numel())] = 1
    return input_ids, attention_mask, groups


def _remap_compacted_groups(groups: Sequence[Dict[str, Any]], row_remap: Dict[int, int], device: torch.device) -> None:
    for group in groups:
        group["row_ids"] = [int(row_remap[int(row_id)]) for row_id in group["row_ids"]]
        old_batch_indices = group["batch_indices"].detach().cpu().tolist()
        group["batch_indices"] = torch.tensor(
            [int(row_remap[int(row_id)]) for row_id in old_batch_indices],
            dtype=torch.long,
            device=device,
        )


@torch.inference_mode()
def runtime_confidence_predict_compacted_batch(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    candidates: Sequence[str],
    exit_layers: Sequence[int],
    threshold: float,
    device: torch.device,
    length_norm: str,
    temperature: float,
    force_full: bool = False,
    controller: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    input_ids, attention_mask, active_groups = _build_compacted_candidate_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        candidates=candidates,
        device=device,
    )
    root = getattr(model, "model", model)
    layers = list(root.layers[: root.config.num_hidden_layers])
    final_layer = int(len(layers))
    exit_set = {int(x) for x in exit_layers if 0 < int(x) < final_layer}
    if bool(force_full):
        exit_set = set()

    hidden_states = root.embed_tokens(input_ids)
    active_attention_mask = attention_mask
    position_ids = torch.arange(hidden_states.shape[1], device=device).unsqueeze(0)

    def make_layer_inputs() -> Tuple[Any, Any]:
        causal = create_causal_mask(
            config=root.config,
            inputs_embeds=hidden_states,
            attention_mask=active_attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        rotary = root.rotary_emb(hidden_states, position_ids=position_ids)
        return causal, rotary

    causal_mask, position_embeddings = make_layer_inputs()
    active_controller = controller or {}
    results: List[Dict[str, Any] | None] = [None] * len(prompts)

    for layer_idx, decoder_layer in enumerate(layers, start=1):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        should_check = int(layer_idx) in exit_set or int(layer_idx) == final_layer
        if not should_check:
            continue

        keep_groups: List[Dict[str, Any]] = []
        for group in active_groups:
            selected = hidden_states[group["batch_indices"], group["token_indices"], :]
            scores = _scores_from_selected_hidden(
                model=model,
                selected_hidden=selected,
                target_ids=group["target_ids"],
                candidate_position_ids=group["candidate_position_ids"],
                candidates=candidates,
                candidate_lengths=group["candidate_lengths"],
                length_norm=length_norm,
            )
            metrics = _metrics_from_scores(candidates, scores, float(temperature))
            metrics["layer"] = int(layer_idx)
            group["checked_layers"].append(int(layer_idx))

            pred = str(metrics["pred"])
            group["stable_count"] = int(group["stable_count"]) + 1 if pred == str(group["previous_pred"]) else 1
            should_exit = bool(int(layer_idx) == final_layer)
            if int(layer_idx) in exit_set:
                if active_controller:
                    features = _controller_features(
                        layer=int(layer_idx),
                        final_layer=final_layer,
                        metrics=metrics,
                        previous_metrics=group["previous_metrics"],
                        stable_count=int(group["stable_count"]),
                        num_choices=len(candidates),
                    )
                    exit_prob = _controller_probability(active_controller, features)
                    decision_threshold = float(active_controller["decision_threshold"])
                    should_exit = bool(exit_prob >= decision_threshold)
                    trace_row = {
                        "layer": int(layer_idx),
                        "exit_probability": float(exit_prob),
                        "decision_threshold": decision_threshold,
                        "features": features,
                    }
                    if str(active_controller.get("policy_kind", "")).lower() == "risk_gain":
                        details = _controller_score_details(active_controller, features)
                        trace_row.update({
                            "exit_score": float(details["score"]),
                            "risk": float(details["risk"]),
                            "expected_future_gain": float(details["expected_future_gain"]),
                            "remaining_compute_frac": float(details["remaining_compute_frac"]),
                        })
                    group["controller_trace"].append(trace_row)
                else:
                    should_exit = bool(float(metrics["confidence"]) >= float(threshold))

            group["previous_metrics"] = dict(metrics)
            group["previous_pred"] = pred
            if should_exit:
                out = {
                    "pred": str(metrics["pred"]),
                    "exit_layer": int(layer_idx),
                    "final_layer": int(final_layer),
                    "confidence": float(metrics["confidence"]),
                    "margin": float(metrics["margin"]),
                    "entropy": float(metrics["entropy"]),
                    "checked_layers": list(group["checked_layers"]),
                    "scores": metrics["scores"],
                    "probs": metrics["probs"],
                    "forward_tokens": int(group["forward_tokens"]),
                    "layer_token_compute": int(group["forward_tokens"]) * int(layer_idx),
                    "active_compaction": True,
                }
                if active_controller:
                    out["controller_trace"] = list(group["controller_trace"])
                results[int(group["sample_id"])] = out
            else:
                keep_groups.append(group)

        active_groups = keep_groups
        if not active_groups:
            break

        keep_rows = sorted({int(row_id) for group in active_groups for row_id in group["row_ids"]})
        if len(keep_rows) < int(hidden_states.shape[0]):
            keep_idx = torch.tensor(keep_rows, dtype=torch.long, device=device)
            hidden_states = hidden_states.index_select(0, keep_idx).contiguous()
            active_attention_mask = active_attention_mask.index_select(0, keep_idx).contiguous()
            row_remap = {int(old): int(new) for new, old in enumerate(keep_rows)}
            _remap_compacted_groups(active_groups, row_remap, device)
            causal_mask, position_embeddings = make_layer_inputs()

    if any(result is None for result in results):
        missing = [idx for idx, result in enumerate(results) if result is None]
        raise RuntimeError(f"compacted runtime predictor produced no result for samples: {missing[:10]}")
    return [result for result in results if result is not None]


def _empty_stats() -> Dict[str, Any]:
    return {
        "samples": 0,
        "correct": 0,
        "elapsed_sec": 0.0,
        "forward_tokens": 0,
        "layer_token_compute": 0,
        "exit_layer_sum": 0,
        "early_count": 0,
        "final_layer": 0,
        "exit_layer_hist": Counter(),
    }


def _add_result(stats: Dict[str, Any], result: Dict[str, Any], answer: str, elapsed: float) -> None:
    final_layer = int(result["final_layer"])
    exit_layer = int(result["exit_layer"])
    stats["final_layer"] = max(int(stats.get("final_layer", 0)), int(final_layer))
    stats["samples"] += 1
    stats["correct"] += int(str(result["pred"]).strip() == str(answer).strip())
    stats["elapsed_sec"] += float(elapsed)
    stats["forward_tokens"] += int(result["forward_tokens"])
    stats["layer_token_compute"] += int(result["layer_token_compute"])
    stats["exit_layer_sum"] += int(exit_layer)
    stats["early_count"] += int(exit_layer < final_layer)
    stats["exit_layer_hist"][str(exit_layer)] += 1


def _finalize_stats(stats: Dict[str, Any], *, full_elapsed: float | None = None) -> Dict[str, Any]:
    samples = max(1, int(stats["samples"]))
    elapsed = max(1e-9, float(stats["elapsed_sec"]))
    avg_layer = float(stats["exit_layer_sum"]) / samples
    final_layer = max(1, int(stats.get("final_layer", 0)))
    out = {
        "samples": int(stats["samples"]),
        "correct": int(stats["correct"]),
        "accuracy": float(stats["correct"] / samples),
        "elapsed_sec": float(stats["elapsed_sec"]),
        "samples_per_s": float(stats["samples"] / elapsed),
        "forward_tokens": int(stats["forward_tokens"]),
        "tokens_per_s": float(stats["forward_tokens"] / elapsed),
        "layer_token_compute": int(stats["layer_token_compute"]),
        "avg_exit_layer": avg_layer,
        "early_exit_rate": float(stats["early_count"] / samples),
        "avg_compute_ratio": float(avg_layer / max(1, final_layer)),
        "estimated_layer_savings": float(1.0 - avg_layer / max(1, final_layer)),
        "exit_layer_hist": dict(sorted(stats["exit_layer_hist"].items(), key=lambda kv: int(kv[0]))),
    }
    if full_elapsed is not None and float(stats["elapsed_sec"]) > 0.0:
        out["wall_clock_speedup_vs_full"] = float(full_elapsed) / float(stats["elapsed_sec"])
    return out


def evaluate_dataset(args: argparse.Namespace, *, model, tokenizer) -> Dict[str, Any]:
    dataset = str(args.dataset)
    dataset_thresholds = getattr(args, "dataset_threshold_map", {}) or {}
    threshold = float(dataset_thresholds.get(dataset, float(args.threshold)))
    candidates = pipe._candidates_for_dataset(dataset)
    data_file = os.path.join(str(args.test_data_root), dataset, "test.json")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(data_file)
    with open(data_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{data_file} must contain a JSON list.")
    records = payload[: int(args.max_samples)] if int(args.max_samples) > 0 else payload
    exit_layers = _parse_int_list(str(args.exit_layers))
    controller = getattr(args, "exit_controller", {}) or {}
    device = next(model.parameters()).device
    active_compaction_batch_size = max(1, int(getattr(args, "active_compaction_batch_size", 1)))
    full_stats = _empty_stats()
    early_stats = _empty_stats()
    record_path = ""
    full_results: List[Dict[str, Any] | None] = [None] * len(records)
    early_results: List[Dict[str, Any]] = []
    answers: List[str] = []
    full_latencies_sec: List[float | None] = [None] * len(records)
    early_latencies_sec: List[float] = []

    # Warm up both paths before collecting latency.  This keeps CUDA context,
    # kernel-loading, and allocator setup out of the per-question comparison.
    # The benchmark used by the paired AE/teacher job sets batch size to one,
    # so every saved latency below corresponds to exactly one test question.
    warmup_samples = min(max(0, int(getattr(args, "warmup_samples", 0))), len(records))
    for warmup_id in range(warmup_samples):
        item = records[int(warmup_id)]
        prompt = pipe._build_prompt(item, dataset)
        if bool(args.run_full_baseline):
            runtime_confidence_predict(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidates=candidates,
                exit_layers=exit_layers,
                threshold=threshold,
                device=device,
                length_norm=str(args.length_norm),
                temperature=float(args.temperature),
                force_full=True,
                controller=None,
            )
        runtime_confidence_predict(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            candidates=candidates,
            exit_layers=exit_layers,
            threshold=threshold,
            device=device,
            length_norm=str(args.length_norm),
            temperature=float(args.temperature),
            force_full=False,
            controller=controller,
        )
    _sync_cuda()

    if bool(args.run_full_baseline):
        _sync_cuda()
        full_t0 = time.perf_counter()
        if active_compaction_batch_size > 1:
            batch_starts = list(range(0, len(records), active_compaction_batch_size))
            iterator = pipe.iter_progress(batch_starts, total=len(batch_starts), desc=f"[RuntimeExit][full-batch] {dataset}")
            for batch_start in iterator:
                batch_records = records[int(batch_start): int(batch_start) + active_compaction_batch_size]
                prompts = [pipe._build_prompt(item, dataset) for item in batch_records]
                batch_answers = [_normalize_answer(item.get("answer", item.get("label", "")), candidates) for item in batch_records]
                _sync_cuda()
                batch_t0 = time.perf_counter()
                batch_results = runtime_confidence_predict_compacted_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    candidates=candidates,
                    exit_layers=exit_layers,
                    threshold=threshold,
                    device=device,
                    length_norm=str(args.length_norm),
                    temperature=float(args.temperature),
                    force_full=True,
                    controller=None,
                )
                _sync_cuda()
                batch_elapsed = time.perf_counter() - batch_t0
                amortized_elapsed = batch_elapsed / max(1, len(batch_results))
                for offset, (answer, result) in enumerate(zip(batch_answers, batch_results)):
                    full_results[int(batch_start) + int(offset)] = result
                    full_latencies_sec[int(batch_start) + int(offset)] = float(amortized_elapsed)
                    _add_result(full_stats, result, answer, 0.0)
        else:
            iterator = pipe.iter_progress(records, total=len(records), desc=f"[RuntimeExit][full] {dataset}")
            for sample_id, item in enumerate(iterator):
                prompt = pipe._build_prompt(item, dataset)
                answer = _normalize_answer(item.get("answer", item.get("label", "")), candidates)
                _sync_cuda()
                sample_t0 = time.perf_counter()
                result = runtime_confidence_predict(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    candidates=candidates,
                    exit_layers=exit_layers,
                    threshold=threshold,
                    device=device,
                    length_norm=str(args.length_norm),
                    temperature=float(args.temperature),
                    force_full=True,
                    controller=None,
                )
                _sync_cuda()
                sample_elapsed = time.perf_counter() - sample_t0
                full_results[int(sample_id)] = result
                full_latencies_sec[int(sample_id)] = float(sample_elapsed)
                _add_result(full_stats, result, answer, 0.0)
        _sync_cuda()
        full_stats["elapsed_sec"] = time.perf_counter() - full_t0

    _sync_cuda()
    early_t0 = time.perf_counter()
    if active_compaction_batch_size > 1:
        batch_starts = list(range(0, len(records), active_compaction_batch_size))
        iterator = pipe.iter_progress(batch_starts, total=len(batch_starts), desc=f"[RuntimeExit][early-compact] {dataset}")
        for batch_start in iterator:
            batch_records = records[int(batch_start): int(batch_start) + active_compaction_batch_size]
            prompts = [pipe._build_prompt(item, dataset) for item in batch_records]
            batch_answers = [_normalize_answer(item.get("answer", item.get("label", "")), candidates) for item in batch_records]
            _sync_cuda()
            batch_t0 = time.perf_counter()
            batch_results = runtime_confidence_predict_compacted_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                candidates=candidates,
                exit_layers=exit_layers,
                threshold=threshold,
                device=device,
                length_norm=str(args.length_norm),
                temperature=float(args.temperature),
                force_full=False,
                controller=controller,
            )
            _sync_cuda()
            batch_elapsed = time.perf_counter() - batch_t0
            amortized_elapsed = batch_elapsed / max(1, len(batch_results))
            for answer, result in zip(batch_answers, batch_results):
                early_results.append(result)
                answers.append(answer)
                early_latencies_sec.append(float(amortized_elapsed))
                _add_result(early_stats, result, answer, 0.0)
    else:
        iterator = pipe.iter_progress(records, total=len(records), desc=f"[RuntimeExit][early] {dataset}")
        for sample_id, item in enumerate(iterator):
            prompt = pipe._build_prompt(item, dataset)
            answer = _normalize_answer(item.get("answer", item.get("label", "")), candidates)
            _sync_cuda()
            sample_t0 = time.perf_counter()
            result = runtime_confidence_predict(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidates=candidates,
                exit_layers=exit_layers,
                threshold=threshold,
                device=device,
                length_norm=str(args.length_norm),
                temperature=float(args.temperature),
                force_full=False,
                controller=controller,
            )
            _sync_cuda()
            sample_elapsed = time.perf_counter() - sample_t0
            early_results.append(result)
            answers.append(answer)
            early_latencies_sec.append(float(sample_elapsed))
            _add_result(early_stats, result, answer, 0.0)
    _sync_cuda()
    early_stats["elapsed_sec"] = time.perf_counter() - early_t0

    if bool(args.save_records):
        record_path = os.path.join(str(args.output_dir), f"{dataset}_runtime_conf_exit_records.jsonl")
        with open(record_path, "w", encoding="utf-8") as record_handle:
            for sample_id, (item, answer, early_result, early_latency_sec) in enumerate(
                zip(records, answers, early_results, early_latencies_sec)
            ):
                full_result = full_results[int(sample_id)] if bool(args.run_full_baseline) else None
                row = {
                    "sample_id": int(sample_id),
                    "instruction": str(item.get("instruction", "")),
                    "input": str(item.get("input", "")),
                    "answer": answer,
                    "full": full_result,
                    "early": early_result,
                    "full_correct": (
                        bool(str(full_result["pred"]) == answer) if full_result is not None else None
                    ),
                    "early_correct": bool(str(early_result["pred"]) == answer),
                    "full_latency_sec": full_latencies_sec[int(sample_id)],
                    "early_latency_sec": float(early_latency_sec),
                    "latency_measurement": (
                        "cuda_synchronized_per_question"
                        if active_compaction_batch_size == 1
                        else "cuda_synchronized_batch_amortized"
                    ),
                }
                record_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    full_report = _finalize_stats(full_stats) if bool(args.run_full_baseline) else {}
    early_report = _finalize_stats(
        early_stats,
        full_elapsed=float(full_report["elapsed_sec"]) if full_report else None,
    )
    if full_report:
        early_report["accuracy_delta_vs_full_runtime"] = float(early_report["accuracy"] - full_report["accuracy"])
    report = {
        "dataset": dataset,
        "candidates": list(candidates),
        "deploy_bundle": str(args.deploy_bundle),
        "model_name_or_path": str(getattr(args, "base_model_name_or_path", "") or getattr(args, "resolved_model_name_or_path", "")),
        "exit_layers": exit_layers,
        "threshold": threshold,
        "threshold_source": "controller" if controller else ("dataset_thresholds" if dataset in dataset_thresholds else "global_threshold"),
        "controller_json": str(args.controller_json) if controller else "",
        "active_compaction_batch_size": int(active_compaction_batch_size),
        "length_norm": str(args.length_norm),
        "temperature": float(args.temperature),
        "warmup_samples": int(warmup_samples),
        "full_runtime": full_report,
        "early_runtime": early_report,
        "records_jsonl": record_path,
    }
    out_path = os.path.join(str(args.output_dir), f"{dataset}_runtime_conf_exit_report.json")
    pipe.save_json(out_path, report)
    msg = (
        f"[RuntimeExit] {dataset} early acc={early_report['accuracy']:.4f} "
        f"avg_layer={early_report['avg_exit_layer']:.2f} savings={early_report['estimated_layer_savings']:.4f} "
        f"samples/s={early_report['samples_per_s']:.2f}"
    )
    if full_report:
        msg += (
            f" | full acc={full_report['accuracy']:.4f} samples/s={full_report['samples_per_s']:.2f} "
            f"speedup={early_report.get('wall_clock_speedup_vs_full', 0.0):.3f}x"
        )
    print(msg, flush=True)
    print(f"[RuntimeExit] saved {out_path}", flush=True)
    return report


def _write_summary(output_dir: str, reports: Sequence[Dict[str, Any]]) -> str:
    path = os.path.join(output_dir, "runtime_confidence_exit_summary.csv")
    rows: List[Dict[str, Any]] = []
    for report in reports:
        dataset = str(report["dataset"])
        for mode_key, metrics in (("full_runtime", report.get("full_runtime") or {}), ("early_runtime", report["early_runtime"])):
            if not metrics:
                continue
            rows.append({
                "dataset": dataset,
                "mode": mode_key,
                "threshold": report.get("threshold", ""),
                "threshold_source": report.get("threshold_source", ""),
                "active_compaction_batch_size": report.get("active_compaction_batch_size", 1),
                "samples": metrics["samples"],
                "accuracy": metrics["accuracy"],
                "elapsed_sec": metrics["elapsed_sec"],
                "samples_per_s": metrics["samples_per_s"],
                "tokens_per_s": metrics["tokens_per_s"],
                "avg_exit_layer": metrics["avg_exit_layer"],
                "early_exit_rate": metrics["early_exit_rate"],
                "estimated_layer_savings": metrics["estimated_layer_savings"],
                "wall_clock_speedup_vs_full": metrics.get("wall_clock_speedup_vs_full", ""),
                "accuracy_delta_vs_full_runtime": metrics.get("accuracy_delta_vs_full_runtime", ""),
                "exit_layer_hist": json.dumps(metrics["exit_layer_hist"], sort_keys=True),
            })
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="True runtime confidence early-exit eval for FAD shared-FFN Llama checkpoints.")
    parser.add_argument("--deploy_bundle", type=str, default="")
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="",
        help="Evaluate a plain teacher/base model instead of a FAD deploy bundle. Used for Teacher+AE ablations.",
    )
    parser.add_argument("--test_data_root", type=str, default=os.path.join(pipe.WORKSPACE_ROOT, "data", "datasets"))
    parser.add_argument("--datasets", type=str, default="piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--exit_layers", type=str, default="12,16,20,24")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--dataset_thresholds",
        type=str,
        default="",
        help="Optional per-dataset thresholds, e.g. 'piqa=0.94 social_i_qa=0.755 hellaswag=0.70'.",
    )
    parser.add_argument(
        "--controller_json",
        type=str,
        default="",
        help="Optional CSQA-trained exit controller JSON. If set, it replaces confidence threshold decisions.",
    )
    parser.add_argument(
        "--controller_decision_threshold",
        type=float,
        default=-1.0,
        help="Optional runtime override for controller decision_threshold. Negative means use JSON value.",
    )
    parser.add_argument("--length_norm", choices=["none", "avg"], default="none")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--active_compaction_batch_size",
        type=int,
        default=1,
        help="Batch multiple questions and compact out examples that already exit. 1 preserves the legacy per-question path.",
    )
    parser.add_argument("--run_full_baseline", type=pipe.str2bool, default=True)
    parser.add_argument("--use_quant_bank_int4", type=pipe.str2bool, default=False)
    parser.add_argument(
        "--weight_quantization",
        choices=["none", "fp8_e4m3", "fp4_e2m1"],
        default="none",
        help="Inference weight format. fp8_e4m3 is native Hopper W8A8; fp4_e2m1 is packed FP4 W4A16.",
    )
    parser.add_argument(
        "--quantize_lm_head",
        type=pipe.str2bool,
        default=False,
        help="Also quantize lm_head. False preserves tied BF16 embedding/output weights and is recommended.",
    )
    parser.add_argument(
        "--save_quantized_checkpoint",
        type=str,
        default="",
        help="Optional output path for the quantized model state_dict plus a reload manifest.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", type=pipe.str2bool, default=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default="")
    parser.add_argument("--save_records", type=pipe.str2bool, default=False)
    parser.add_argument(
        "--warmup_samples",
        type=int,
        default=0,
        help="Unmeasured per-dataset samples used to warm CUDA kernels before latency collection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dataset_threshold_map = _parse_dataset_thresholds(str(args.dataset_thresholds))
    args.exit_controller = _apply_controller_threshold_override(
        _load_exit_controller(str(args.controller_json)),
        float(args.controller_decision_threshold),
    )
    pipe.set_seed(int(args.seed))
    os.makedirs(str(args.output_dir), exist_ok=True)
    device = pipe.resolve_device(str(args.device), -1)
    dtype = pipe.get_target_dtype(device)
    deploy_bundle = str(args.deploy_bundle).strip()
    base_model_name_or_path = str(args.base_model_name_or_path).strip()
    bundle: Dict[str, Any] = {}
    shared_meta: Dict[str, Any] = {}
    if deploy_bundle:
        bundle = torch.load(deploy_bundle, map_location="cpu")
        if not isinstance(bundle, dict):
            raise ValueError("--deploy_bundle must point to a deploy_bundle.pt dict.")
        model, legacy_quant_report = pipe._build_shared_model_for_eval(
            base_model=str(bundle["base_model"]),
            atlas_payload=bundle.get("atlas", {}),
            shared_payload=bundle["shared_student"],
            quant_bank_int4=bundle.get("quant_bank_int4"),
            use_quant_bank_int4=bool(args.use_quant_bank_int4),
            device=device,
            dtype=dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        shared_meta = bundle.get("shared_student", {}).get("meta", {}) if isinstance(bundle.get("shared_student"), dict) else {}
        args.resolved_model_name_or_path = str(bundle.get("base_model", ""))
        model_source = "deploy_bundle"
    else:
        if not base_model_name_or_path:
            raise ValueError("Set either --deploy_bundle or --base_model_name_or_path.")
        model = pipe.AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=bool(args.trust_remote_code),
        ).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        legacy_quant_report = {
            "requested": bool(args.use_quant_bank_int4),
            "enabled": False,
            "available_count": 0,
            "applied_count": 0,
            "missing_count": 0,
            "missing_keys": [],
            "note": "plain teacher/base model path; quant_bank_int4 is not available",
        }
        args.resolved_model_name_or_path = base_model_name_or_path
        model_source = "base_model"
    if str(args.weight_quantization) != "none" and bool(args.use_quant_bank_int4):
        raise ValueError("Do not combine --weight_quantization with legacy --use_quant_bank_int4 fake-quantization.")
    quant_report = fad_quant.apply_weight_quantization(
        model,
        str(args.weight_quantization),
        quantize_lm_head=bool(args.quantize_lm_head),
    )
    quant_report["legacy_shared_bank_int4"] = legacy_quant_report
    quant_report["gpu_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    quant_report["cuda_capability"] = (
        list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None
    )
    quantized_checkpoint = ""
    if str(args.save_quantized_checkpoint).strip():
        if str(args.weight_quantization) == "none":
            raise ValueError("--save_quantized_checkpoint requires --weight_quantization != none")
        if not deploy_bundle:
            raise ValueError("Saving a quantized FAD checkpoint requires --deploy_bundle")
        quantized_checkpoint = fad_quant.save_quantized_checkpoint(
            str(args.save_quantized_checkpoint),
            model=model,
            quantization_report=quant_report,
            source_deploy_bundle=deploy_bundle,
        )
    tokenizer_path = (
        str(args.tokenizer_name_or_path).strip()
        or str(shared_meta.get("teacher_model", ""))
        or str(bundle.get("base_model", ""))
        or base_model_name_or_path
    )
    tokenizer = pipe.load_tokenizer(tokenizer_path, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets = [str(args.dataset).strip()] if str(args.dataset).strip() else _parse_csv_list(str(args.datasets))
    print(f"[RuntimeExit] model_source={model_source}", flush=True)
    print(f"[RuntimeExit] deploy_bundle={args.deploy_bundle}", flush=True)
    if base_model_name_or_path:
        print(f"[RuntimeExit] base_model_name_or_path={base_model_name_or_path}", flush=True)
    print(f"[RuntimeExit] quant={quant_report}", flush=True)
    if quantized_checkpoint:
        print(f"[RuntimeExit] quantized_checkpoint={quantized_checkpoint}", flush=True)
    print(
        f"[RuntimeExit] datasets={' '.join(datasets)} exit_layers={args.exit_layers} "
        f"threshold={float(args.threshold):.3f} run_full_baseline={bool(args.run_full_baseline)} "
        f"active_compaction_batch_size={max(1, int(args.active_compaction_batch_size))}",
        flush=True,
    )
    if args.dataset_threshold_map:
        print(f"[RuntimeExit] dataset_thresholds={args.dataset_threshold_map}", flush=True)
    if args.exit_controller:
        print(
            f"[RuntimeExit] controller_json={args.controller_json} "
            f"decision_threshold={float(args.exit_controller['decision_threshold']):.6f}",
            flush=True,
        )
    reports: List[Dict[str, Any]] = []
    for dataset in datasets:
        args.dataset = dataset
        reports.append(evaluate_dataset(args, model=model, tokenizer=tokenizer))
    summary_csv = _write_summary(str(args.output_dir), reports)
    all_report = {
        "deploy_bundle": str(args.deploy_bundle),
        "model_source": model_source,
        "model_name_or_path": str(args.resolved_model_name_or_path),
        "output_dir": str(args.output_dir),
        "summary_csv": summary_csv,
        "active_compaction_batch_size": max(1, int(args.active_compaction_batch_size)),
        "quantization": quant_report,
        "quantized_checkpoint": quantized_checkpoint,
        "reports": reports,
    }
    pipe.save_json(os.path.join(str(args.output_dir), "runtime_confidence_exit_all_report.json"), all_report)
    print(f"[RuntimeExit] summary_csv={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
