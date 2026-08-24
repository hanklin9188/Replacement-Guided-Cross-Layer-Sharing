from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import model_config, run_dir
from .utils import atomic_json, base_manifest


def _all_valid(group: list[int], valid: np.ndarray) -> bool:
    return all(bool(valid[i, j]) for i in group for j in group)


def agglomerative(
    pair_cost: np.ndarray,
    valid: np.ndarray,
    k: int,
    linkage: str = "complete",
) -> list[list[int]]:
    groups = [[i] for i in range(pair_cost.shape[0])]
    reducer: Callable = np.max if linkage == "complete" else np.min
    while len(groups) > k:
        candidates = []
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                merged = groups[a] + groups[b]
                if not _all_valid(merged, valid):
                    continue
                cross = pair_cost[np.ix_(groups[a], groups[b])]
                candidates.append((float(reducer(cross)), min(merged), max(merged), a, b))
        if not candidates:
            raise RuntimeError(f"No feasible {linkage}-link merge at {len(groups)} groups; requested K={k}")
        _, _, _, a, b = min(candidates)
        groups[a] = sorted(groups[a] + groups[b])
        del groups[b]
    return sorted(groups, key=lambda group: min(group))


def _allocate_chunks(regime_members: list[list[int]], k: int) -> list[int]:
    if k < len(regime_members):
        raise ValueError("K is smaller than the number of nonempty depth regimes")
    allocation = [1] * len(regime_members)
    while sum(allocation) < k:
        scores = [len(members) / allocation[i] if allocation[i] < len(members) else -1 for i, members in enumerate(regime_members)]
        allocation[int(np.argmax(scores))] += 1
    return allocation


def adjacent_groups(valid: np.ndarray, regimes: np.ndarray, k: int) -> list[list[int]]:
    # Find an exact-K contiguous partition under the *same* pairwise feasibility
    # mask as every other baseline. Balanced np.array_split boundaries can place
    # one low-cosine adjacent pair in a segment even when another boundary is
    # feasible, so solve the small ordered partition problem exactly.
    layers = len(regimes)
    target_size = layers / k
    dp: dict[tuple[int, int], tuple[float, list[list[int]]]] = {(0, 0): (0.0, [])}
    for start in range(layers):
        for used in range(k):
            state = dp.get((start, used))
            if state is None:
                continue
            base_cost, path = state
            for stop in range(start + 1, layers + 1):
                segment = list(range(start, stop))
                if regimes[stop - 1] != regimes[start]:
                    break
                if not _all_valid(segment, valid):
                    continue
                remaining_layers = layers - stop
                remaining_groups = k - used - 1
                if remaining_layers < remaining_groups:
                    continue
                score = base_cost + (len(segment) - target_size) ** 2
                key = (stop, used + 1)
                candidate = (score, path + [segment])
                incumbent = dp.get(key)
                if incumbent is None or (candidate[0], candidate[1]) < (incumbent[0], incumbent[1]):
                    dp[key] = candidate
    result = dp.get((layers, k))
    if result is None:
        raise RuntimeError("No exact-K contiguous grouping satisfies the common feasibility mask")
    return result[1]


def random_like(groups: list[list[int]], regimes: np.ndarray, valid: np.ndarray, seed: int, attempts: int = 10_000):
    rng = np.random.default_rng(seed)
    template: dict[int, list[int]] = {}
    for group in groups:
        regime = int(regimes[group[0]])
        template.setdefault(regime, []).append(len(group))
    for _ in range(attempts):
        candidate = []
        for regime, sizes in template.items():
            members = np.where(regimes == regime)[0]
            members = rng.permutation(members).tolist()
            offset = 0
            for size in sizes:
                candidate.append(sorted(members[offset : offset + size]))
                offset += size
        if all(_all_valid(group, valid) for group in candidate):
            return sorted(candidate, key=min)
    raise RuntimeError(f"Could not sample a feasibility-matched random assignment after {attempts} attempts")


def representatives(groups: list[list[int]], directed: np.ndarray, symmetric: bool = False) -> list[int]:
    result = []
    pair = np.maximum(directed, directed.T)
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
            continue
        scores = []
        for source in group:
            targets = [target for target in group if target != source]
            values = pair[source, targets] if symmetric else directed[source, targets]
            scores.append((float(np.max(values)), float(np.mean(values)), source))
        result.append(min(scores)[2])
    return result


def diagnostics(groups: list[list[int]], reps: list[int], directed: np.ndarray) -> dict[str, Any]:
    pair = np.maximum(directed, directed.T)
    all_within = []
    group_rows = []
    outward = []
    for group_id, (group, rep) in enumerate(zip(groups, reps)):
        within = [float(pair[i, j]) for p, i in enumerate(group) for j in group[p + 1 :]]
        rep_costs = [float(directed[rep, target]) for target in group if target != rep]
        all_within.extend(within)
        outward.extend(rep_costs)
        group_rows.append({
            "group_id": group_id, "members": group, "representative": rep,
            "size": len(group), "layer_span": max(group) - min(group),
            "c_max": max(within, default=0.0), "c_mean": float(np.mean(within)) if within else 0.0,
            "r_max": max(rep_costs, default=0.0),
        })
    return {
        "c_max": max(all_within, default=0.0),
        "c_mean": float(np.mean(all_within)) if all_within else 0.0,
        "c_p95": float(np.percentile(all_within, 95)) if all_within else 0.0,
        "r_max": max(outward, default=0.0),
        "singleton_count": sum(len(group) == 1 for group in groups),
        "group_sizes": [len(group) for group in groups],
        "layer_spans": [max(group) - min(group) for group in groups],
        "per_group": group_rows,
    }


def build_groups(cfg: dict[str, Any], backbone: str, valid_tokens: int, k: int) -> None:
    mcfg = model_config(cfg, backbone)
    source = run_dir(cfg, backbone, "replacement", f"tokens_{valid_tokens}")
    output = run_dir(cfg, backbone, "groups", f"tokens_{valid_tokens}", f"k_{k}")
    output.mkdir(parents=True, exist_ok=True)
    directed = np.load(source / "directed_cost.npy")
    pair_max = np.maximum(directed, directed.T)
    pair_mean = (directed + directed.T) / 2
    weight = np.load(source / "normalized_weight_distance.npy")
    cosine_distance = 1 - np.load(source / "ffn_input_cosine.npy")
    valid_full = np.load(source / "valid_pair_mask.npy").astype(bool)
    layers = int(mcfg["layers"])
    regimes = np.minimum(np.arange(layers) * int(cfg["replacement"]["depth_regimes"]) // layers,
                         int(cfg["replacement"]["depth_regimes"]) - 1)
    same_regime = regimes[:, None] == regimes[None, :]
    input_valid = np.load(source / "ffn_input_cosine.npy") >= float(cfg["replacement"]["input_cosine_min"])
    np.fill_diagonal(input_valid, True)

    specifications: dict[str, tuple[list[list[int]], bool]] = {}
    full = agglomerative(pair_max, valid_full, k, "complete")
    specifications["full"] = (full, False)
    specifications["directional_mean"] = (agglomerative(pair_mean, valid_full, k, "complete"), False)
    specifications["single_link"] = (agglomerative(pair_max, valid_full, k, "single"), False)
    specifications["symmetric_representative"] = (full, True)
    specifications["no_input_filter"] = (agglomerative(pair_max, same_regime, k, "complete"), False)
    specifications["no_depth_constraints"] = (agglomerative(pair_max, input_valid, k, "complete"), False)
    specifications["weight_distance"] = (agglomerative(weight, valid_full, k, "complete"), False)
    specifications["input_cosine"] = (agglomerative(cosine_distance, valid_full, k, "complete"), False)
    one_way = np.zeros_like(directed)
    for i in range(layers):
        for j in range(layers):
            one_way[i, j] = directed[min(i, j), max(i, j)]
    specifications["one_way_replacement"] = (agglomerative(one_way, valid_full, k, "complete"), False)
    specifications["adjacent"] = (adjacent_groups(valid_full, regimes, k), False)
    for random_id, seed in enumerate([1027, 2027, 3027, 4027, 5027]):
        specifications[f"random_{random_id}"] = (random_like(full, regimes, valid_full, seed), False)

    index = {}
    for policy, (groups, symmetric_rep) in specifications.items():
        reps = representatives(groups, directed, symmetric=symmetric_rep)
        diag = diagnostics(groups, reps, directed)
        manifest = base_manifest(cfg, "grouping", backbone)
        manifest.update({
            "policy": policy, "k": k, "layers": layers, "k_over_l": k / layers,
            "valid_tokens": valid_tokens, "groups": groups, "representatives": reps,
            "representative_rule": "symmetric_medoid" if symmetric_rep else "outward_directed_minimax",
            "feasibility_mask": "valid_pair_mask.npy", "diagnostics": diag,
            "grouping_uses_recovery_metrics": False,
        })
        filename = f"{policy}.json"
        atomic_json(output / filename, manifest)
        index[policy] = filename
    atomic_json(output / "group_manifest.json", {"policies": index, "backbone": backbone, "k": k, "valid_tokens": valid_tokens})
    atomic_json(output / "representative_manifest.json", {
        policy: {"groups": value[0], "representatives": representatives(value[0], directed, value[1])}
        for policy, value in specifications.items()
    })


def feasibility_audit(cfg: dict[str, Any], backbone: str, valid_tokens: int) -> None:
    mcfg = model_config(cfg, backbone)
    source = run_dir(cfg, backbone, "replacement", f"tokens_{valid_tokens}")
    directed = np.load(source / "directed_cost.npy")
    pair = np.maximum(directed, directed.T)
    cosine = np.load(source / "ffn_input_cosine.npy")
    layers = int(mcfg["layers"])
    regimes = np.minimum(np.arange(layers) * int(cfg["replacement"]["depth_regimes"]) // layers,
                         int(cfg["replacement"]["depth_regimes"]) - 1)
    same_regime = regimes[:, None] == regimes[None, :]
    rows = []
    for threshold in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.25, 0.0, -1.0]:
        valid = same_regime & (cosine >= threshold)
        np.fill_diagonal(valid, True)
        for k in mcfg["budgets"]:
            row = {"threshold": threshold, "k": int(k), "full_complete_link": False,
                   "adjacent": False, "error": None}
            try:
                agglomerative(pair, valid, int(k), "complete")
                row["full_complete_link"] = True
            except RuntimeError as error:
                row["error"] = str(error)
            try:
                adjacent_groups(valid, regimes, int(k))
                row["adjacent"] = True
            except RuntimeError as error:
                row["adjacent_error"] = str(error)
            rows.append(row)
    output = run_dir(cfg, backbone, "replacement", f"tokens_{valid_tokens}", "feasibility_audit.json")
    manifest = base_manifest(cfg, "feasibility_audit", backbone)
    manifest.update({"valid_tokens": valid_tokens, "depth_regimes": regimes.tolist(), "rows": rows,
                     "uses_recovery_or_final_metrics": False})
    atomic_json(output, manifest)
