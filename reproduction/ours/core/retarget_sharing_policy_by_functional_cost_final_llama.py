#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def parse_layers(value: str) -> Optional[Set[int]]:
    value = value.strip()
    if not value:
        return None
    layers: Set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layers.add(int(item))
    return layers


def load_intervention_costs(path: Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    rows: Dict[Tuple[int, int], Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = int(row["source_layer"])
            target = int(row["target_layer"])
            rows[(source, target)] = {
                "merge_cost": float(row["merge_cost"]),
                "delta_nll": float(row["delta_nll"]),
                "kl": float(row["kl_baseline_to_intervention"]),
            }
    return rows


def candidate_layer_sets(policy: Dict[str, Any], explicit_layers: Optional[Set[int]]) -> List[List[int]]:
    if explicit_layers:
        return [sorted(explicit_layers)]
    groups = policy.get("groups", [])
    out: List[List[int]] = []
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            layers = group.get("layers", [])
            parsed = sorted(int(layer) for layer in layers)
            if len(parsed) >= 2:
                out.append(parsed)
    return out


def layer_regime_map(policy: Dict[str, Any]) -> Dict[int, str]:
    regimes: Dict[int, str] = {}
    layers = policy.get("layers", [])
    if isinstance(layers, list):
        for item in layers:
            if isinstance(item, dict) and "layer_id" in item:
                regimes[int(item["layer_id"])] = str(item.get("regime", "unknown"))
    return regimes


def edge_stats(
    a: int,
    b: int,
    costs: Dict[Tuple[int, int], Dict[str, float]],
) -> Optional[Dict[str, float]]:
    ab = costs.get((a, b))
    ba = costs.get((b, a))
    if ab is None or ba is None:
        return None
    return {
        "layer_a": float(a),
        "layer_b": float(b),
        "cost_a_to_b": float(ab["merge_cost"]),
        "cost_b_to_a": float(ba["merge_cost"]),
        "max_bidirectional_cost": float(max(ab["merge_cost"], ba["merge_cost"])),
        "mean_bidirectional_cost": float((ab["merge_cost"] + ba["merge_cost"]) / 2.0),
        "max_delta_nll": float(max(ab["delta_nll"], ba["delta_nll"])),
        "mean_delta_nll": float((ab["delta_nll"] + ba["delta_nll"]) / 2.0),
        "max_kl": float(max(ab["kl"], ba["kl"])),
        "mean_kl": float((ab["kl"] + ba["kl"]) / 2.0),
    }


def connected_components(nodes: Iterable[int], edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    remaining = set(nodes)
    adjacency: Dict[int, Set[int]] = {node: set() for node in remaining}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    components: List[List[int]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        seen: Set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for nxt in adjacency.get(node, set()):
                if nxt not in seen:
                    stack.append(nxt)
        remaining -= seen
        if len(seen) >= 2:
            components.append(sorted(seen))
    return components


def group_edge_summary(group_layers: List[int], costs: Dict[Tuple[int, int], Dict[str, float]]) -> Dict[str, Any]:
    stats: List[Dict[str, float]] = []
    for idx, a in enumerate(group_layers):
        for b in group_layers[idx + 1 :]:
            item = edge_stats(a, b, costs)
            if item is not None:
                stats.append(item)
    if not stats:
        return {}
    return {
        "mean_bidirectional_merge_cost": float(sum(x["mean_bidirectional_cost"] for x in stats) / len(stats)),
        "max_bidirectional_merge_cost": float(max(x["max_bidirectional_cost"] for x in stats)),
        "mean_delta_nll": float(sum(x["mean_delta_nll"] for x in stats) / len(stats)),
        "max_delta_nll": float(max(x["max_delta_nll"] for x in stats)),
        "mean_kl": float(sum(x["mean_kl"] for x in stats) / len(stats)),
        "max_kl": float(max(x["max_kl"] for x in stats)),
        "functional_edges": stats,
    }


def build_functional_groups(
    policy: Dict[str, Any],
    costs: Dict[Tuple[int, int], Dict[str, float]],
    explicit_layers: Optional[Set[int]],
    max_bidirectional_cost: float,
    max_layer_gap: int,
    require_same_regime: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    regimes = layer_regime_map(policy)
    groups: List[Dict[str, Any]] = []
    selected_edges: List[Dict[str, Any]] = []
    used_layers: Set[int] = set()

    for candidate_layers in candidate_layer_sets(policy, explicit_layers):
        nodes = [layer for layer in candidate_layers if layer not in used_layers]
        if len(nodes) < 2:
            continue
        edges: List[Tuple[int, int]] = []
        for idx, a in enumerate(nodes):
            for b in nodes[idx + 1 :]:
                if abs(a - b) > max_layer_gap:
                    continue
                if require_same_regime and regimes.get(a) != regimes.get(b):
                    continue
                item = edge_stats(a, b, costs)
                if item is None:
                    continue
                if item["max_bidirectional_cost"] <= max_bidirectional_cost:
                    edges.append((a, b))
                    selected_edges.append(item)
        for component in connected_components(nodes, edges):
            if any(layer in used_layers for layer in component):
                continue
            gid = len(groups)
            summary = group_edge_summary(component, costs)
            group = {
                "group_id": gid,
                "layers": component,
                "regime": regimes.get(component[0], "unknown"),
                "private_core": False,
                **summary,
            }
            groups.append(group)
            used_layers.update(component)
    return groups, selected_edges


class UnionFind:
    def __init__(self, nodes: Iterable[int]) -> None:
        self.parent = {int(node): int(node) for node in nodes}
        self.rank = {int(node): 0 for node in nodes}
        self.size = {int(node): 1 for node in nodes}

    def find(self, node: int) -> int:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def component_size(self, node: int) -> int:
        return self.size[self.find(node)]

    def union(self, a: int, b: int, max_component_size: int = 0) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if max_component_size > 0 and self.size[ra] + self.size[rb] > max_component_size:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def build_target_saved_groups(
    policy: Dict[str, Any],
    costs: Dict[Tuple[int, int], Dict[str, float]],
    explicit_layers: Optional[Set[int]],
    target_saved_mlps: int,
    max_layer_gap: int,
    max_group_size: int,
    require_same_regime: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    regimes = layer_regime_map(policy)
    candidate_sets = candidate_layer_sets(policy, explicit_layers)
    if not candidate_sets:
        raise RuntimeError("no candidate layer sets available for target-saved policy")

    selected_edges: List[Dict[str, Any]] = []
    selected_pairs: List[Tuple[int, int]] = []
    global_group_layers: Set[int] = set()

    for candidate_layers in candidate_sets:
        nodes = sorted(set(candidate_layers))
        if len(nodes) < 2:
            continue
        uf = UnionFind(nodes)
        edge_items: List[Tuple[float, float, int, int, Dict[str, float]]] = []
        for idx, a in enumerate(nodes):
            for b in nodes[idx + 1 :]:
                if abs(a - b) > max_layer_gap:
                    continue
                if require_same_regime and regimes.get(a) != regimes.get(b):
                    continue
                item = edge_stats(a, b, costs)
                if item is None:
                    continue
                edge_items.append(
                    (
                        float(item["max_bidirectional_cost"]),
                        float(item["mean_bidirectional_cost"]),
                        a,
                        b,
                        item,
                    )
                )
        for _max_cost, _mean_cost, a, b, item in sorted(edge_items):
            if len(selected_pairs) >= target_saved_mlps:
                break
            if a in global_group_layers and b in global_group_layers:
                pass
            if uf.union(a, b, max_component_size=max_group_size):
                selected_pairs.append((a, b))
                selected_edges.append(item)
                global_group_layers.add(a)
                global_group_layers.add(b)
        if len(selected_pairs) >= target_saved_mlps:
            break

    if len(selected_pairs) < target_saved_mlps:
        raise RuntimeError(
            f"target_saved_mlps={target_saved_mlps} is infeasible; selected only {len(selected_pairs)}"
        )

    all_nodes = sorted({layer for pair in selected_pairs for layer in pair})
    components = connected_components(all_nodes, selected_pairs)
    groups: List[Dict[str, Any]] = []
    for component in components:
        gid = len(groups)
        summary = group_edge_summary(component, costs)
        groups.append(
            {
                "group_id": gid,
                "layers": component,
                "regime": regimes.get(component[0], "unknown"),
                "private_core": False,
                **summary,
            }
        )
    return groups, selected_edges


def apply_groups(policy: Dict[str, Any], groups: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    layer_to_group: Dict[int, int] = {}
    for group in groups:
        for layer in group["layers"]:
            layer_to_group[int(layer)] = int(group["group_id"])

    layers = policy.get("layers", [])
    if isinstance(layers, list):
        for item in layers:
            if not isinstance(item, dict) or "layer_id" not in item:
                continue
            layer_id = int(item["layer_id"])
            if layer_id in layer_to_group:
                item["group_id"] = layer_to_group[layer_id]
                item["private_core"] = False
            else:
                item["group_id"] = -1
                item["private_core"] = True

    policy["source_view"] = "functional_ffn_intervention"
    policy["sharing_policy_mode"] = "functional_cost_retarget"
    policy["sharing_group_count"] = len(groups)
    policy["groups"] = groups
    policy["functional_retarget"] = metadata
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget an IETS sharing policy using measured FFN intervention costs.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--obs-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--eligible-layers", default="")
    parser.add_argument("--max-bidirectional-cost", type=float, default=0.32)
    parser.add_argument("--max-layer-gap", type=int, default=1)
    parser.add_argument("--target-saved-mlps", type=int, default=0)
    parser.add_argument("--target-compression-ratio", type=float, default=0.0)
    parser.add_argument("--saved-mlp-whole-model-ratio", type=float, default=0.0223333333)
    parser.add_argument("--max-group-size", type=int, default=0)
    parser.add_argument("--require-same-regime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    policy_path = Path(args.policy)
    obs_dir = Path(args.obs_dir)
    intervention_path = obs_dir / "intervention_pairs_sorted.csv"
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)
    if not intervention_path.exists():
        raise FileNotFoundError(intervention_path)

    policy = load_json(policy_path)
    costs = load_intervention_costs(intervention_path)
    explicit_layers = parse_layers(args.eligible_layers)
    target_saved_mlps = int(args.target_saved_mlps)
    if target_saved_mlps <= 0 and float(args.target_compression_ratio) > 0.0:
        target_saved_mlps = int(math.ceil(float(args.target_compression_ratio) / float(args.saved_mlp_whole_model_ratio)))
    if target_saved_mlps > 0:
        groups, selected_edges = build_target_saved_groups(
            policy=policy,
            costs=costs,
            explicit_layers=explicit_layers,
            target_saved_mlps=target_saved_mlps,
            max_layer_gap=int(args.max_layer_gap),
            max_group_size=int(args.max_group_size),
            require_same_regime=bool(args.require_same_regime),
        )
    else:
        groups, selected_edges = build_functional_groups(
            policy=policy,
            costs=costs,
            explicit_layers=explicit_layers,
            max_bidirectional_cost=float(args.max_bidirectional_cost),
            max_layer_gap=int(args.max_layer_gap),
            require_same_regime=bool(args.require_same_regime),
        )
    if not groups:
        raise RuntimeError(
            "functional retarget produced no sharing groups; relax max_bidirectional_cost or max_layer_gap"
        )

    total_shared_layers = sum(len(group["layers"]) for group in groups)
    saved_mlps = sum(max(0, len(group["layers"]) - 1) for group in groups)
    metadata = {
        "obs_dir": str(obs_dir),
        "intervention_csv": str(intervention_path),
        "eligible_layers": sorted(explicit_layers) if explicit_layers else "existing_policy_groups",
        "max_bidirectional_cost": float(args.max_bidirectional_cost),
        "max_layer_gap": int(args.max_layer_gap),
        "require_same_regime": bool(args.require_same_regime),
        "target_saved_mlps": int(target_saved_mlps),
        "target_compression_ratio": float(args.target_compression_ratio),
        "saved_mlp_whole_model_ratio": float(args.saved_mlp_whole_model_ratio),
        "estimated_whole_model_compression": float(saved_mlps * float(args.saved_mlp_whole_model_ratio)),
        "max_group_size": int(args.max_group_size),
        "selected_edge_count": len(selected_edges),
        "selected_edges": selected_edges,
        "shared_layer_count": int(total_shared_layers),
        "saved_mlp_count": int(saved_mlps),
        "note": "Groups are selected from measured directed FFN replacement costs; compression ratio is an outcome.",
    }
    updated = apply_groups(policy, groups, metadata)

    output_path = Path(args.output) if args.output else policy_path
    if output_path == policy_path and args.backup:
        backup_path = policy_path.with_name(policy_path.stem + ".before_functional_retarget.json")
        if not backup_path.exists():
            shutil.copy2(policy_path, backup_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(updated, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"[FunctionalPolicy] wrote {output_path}")
    print(f"[FunctionalPolicy] groups={ [group['layers'] for group in groups] }")
    print(f"[FunctionalPolicy] saved_mlp_count={saved_mlps}")
    if selected_edges:
        best = min(selected_edges, key=lambda item: item["max_bidirectional_cost"])
        worst = max(selected_edges, key=lambda item: item["max_bidirectional_cost"])
        print(
            "[FunctionalPolicy] selected_edge_max_bidirectional_cost_range="
            f"{best['max_bidirectional_cost']:.6f}..{worst['max_bidirectional_cost']:.6f}"
        )


if __name__ == "__main__":
    main()
