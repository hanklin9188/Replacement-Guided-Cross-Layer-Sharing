#!/usr/bin/env python3
"""Retarget an FFN-input-similarity sharing policy to an exact prototype count."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _components(adjacency: List[List[bool]]) -> List[List[int]]:
    n = len(adjacency)
    seen = [False] * n
    out: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt, linked in enumerate(adjacency[cur]):
                if linked and not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        out.append(sorted(comp))
    return out


def _policy_for_threshold(policy: Dict[str, Any], threshold: float) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    upstream = policy["pairwise_similarity"]["upstream"]
    pairwise = policy.get("pairwise_similarity", {})
    layers = policy.get("layers") or []
    n = len(upstream)
    regimes = policy.get("regime_labels") or [str(x.get("regime", "")) for x in layers]
    reliability = [float(x.get("reliability", 1.0)) for x in layers] if layers else [1.0] * n

    adjacency = [[False] * n for _ in range(n)]
    for i in range(n):
        adjacency[i][i] = True
        for j in range(i + 1, n):
            if str(regimes[i]) != str(regimes[j]):
                continue
            if float(upstream[i][j]) >= float(threshold):
                adjacency[i][j] = True
                adjacency[j][i] = True

    raw_groups = _components(adjacency)
    groups: List[Dict[str, Any]] = []
    assigned: Dict[int, int] = {}
    for comp in raw_groups:
        if len(comp) <= 1:
            continue
        gid = len(groups)
        pairs = [(a, b) for idx, a in enumerate(comp) for b in comp[idx + 1 :]]
        denom = max(1, len(pairs))

        def mean_pair(name: str) -> float:
            matrix = pairwise.get(name) or upstream
            return sum(float(matrix[a][b]) for a, b in pairs) / float(denom)

        group = {
            "group_id": int(gid),
            "layers": [int(x) for x in comp],
            "regime": str(regimes[comp[0]]),
            "private_core": False,
            "mean_upstream_similarity": mean_pair("upstream"),
            "mean_attention_similarity": mean_pair("attention_delta"),
            "mean_horizon_h2_similarity": mean_pair("short_horizon_h2"),
            "mean_horizon_h3_similarity": mean_pair("short_horizon_h3"),
            "mean_horizon_similarity": 0.5 * mean_pair("short_horizon_h2") + 0.5 * mean_pair("short_horizon_h3"),
            "reliability_mean": sum(reliability[x] for x in comp) / float(len(comp)),
        }
        groups.append(group)
        for layer_id in comp:
            assigned[int(layer_id)] = int(gid)

    new_layers: List[Dict[str, Any]] = []
    for layer_id in range(n):
        old = dict(layers[layer_id]) if layer_id < len(layers) and isinstance(layers[layer_id], dict) else {}
        old.update(
            {
                "layer_id": int(layer_id),
                "regime": str(regimes[layer_id]),
                "group_id": int(assigned[layer_id]) if layer_id in assigned else -1,
                "private_core": bool(layer_id not in assigned),
                "reliability": float(reliability[layer_id]),
            }
        )
        new_layers.append(old)

    proto_count = n - sum(max(0, len(g["layers"]) - 1) for g in groups)
    return int(proto_count), groups, new_layers


def _candidate_thresholds(policy: Dict[str, Any]) -> List[float]:
    upstream = policy["pairwise_similarity"]["upstream"]
    layers = policy.get("layers") or []
    regimes = policy.get("regime_labels") or [str(x.get("regime", "")) for x in layers]
    vals = set()
    n = len(upstream)
    for i in range(n):
        for j in range(i + 1, n):
            if str(regimes[i]) == str(regimes[j]):
                vals.add(float(upstream[i][j]))
    vals.add(1.0 + 1e-8)
    vals.add(-1.0)
    return sorted(vals, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--target-proto-count", type=int, required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    path = Path(args.policy)
    policy = json.loads(path.read_text(encoding="utf-8"))
    target = int(args.target_proto_count)
    if target <= 0:
        raise ValueError("--target-proto-count must be positive")

    best = None
    best_abs = math.inf
    for threshold in _candidate_thresholds(policy):
        proto_count, groups, layers = _policy_for_threshold(policy, threshold)
        gap = abs(proto_count - target)
        if gap < best_abs:
            best = (threshold, proto_count, groups, layers)
            best_abs = gap
        if proto_count == target:
            best = (threshold, proto_count, groups, layers)
            break

    if best is None:
        raise RuntimeError("No candidate sharing policy could be constructed.")

    threshold, proto_count, groups, layers = best
    updated = dict(policy)
    updated["sharing_policy_mode"] = "upstream_only"
    updated["upstream_similarity_threshold"] = float(threshold)
    updated["target_proto_count"] = int(target)
    updated["actual_proto_count"] = int(proto_count)
    updated["sharing_group_count"] = int(len(groups))
    updated["groups"] = groups
    updated["layers"] = layers
    updated["retargeted_by"] = "retarget_sharing_policy_final_llama.py"

    output = Path(args.output) if args.output else path
    output.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shared_layers = sum(max(0, len(g["layers"]) - 1) for g in groups)
    print(
        f"[RetargetSharing] target_proto_count={target} actual_proto_count={proto_count} "
        f"threshold={threshold:.8f} shared_layers_equiv={shared_layers} groups={len(groups)} "
        f"output={output}",
        flush=True,
    )
    if proto_count != target:
        print(
            f"[RetargetSharing][Warn] exact target not attainable; used nearest proto_count={proto_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
