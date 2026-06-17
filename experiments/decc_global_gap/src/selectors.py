import itertools
import math
from typing import Dict, List, Optional, Tuple

from src.aggregation import apply_decc_aggregation
from src.latency_eval import evaluate_pipeline_latency


def select_decc_style(cost_table: List[List[Dict]]) -> List[int]:
    vector = []
    for row in cost_table:
        best_idx = min(range(len(row)), key=lambda j: row[j]["d"] + row[j]["t"] + row[j]["s"])
        vector.append(best_idx)
    return vector


def select_global_bruteforce(
    branches: List[Dict],
    cost_table: List[List[Dict]],
    max_combinations: Optional[int] = None,
) -> Tuple[List[int], float, List[Dict]]:
    ranges = [range(len(row)) for row in cost_table]
    combinations = math.prod(len(row) for row in cost_table) if cost_table else 0

    if max_combinations is not None and combinations > max_combinations:
        return _select_global_pareto_dp(branches, cost_table)

    best_vec = None
    best_latency = float("inf")
    best_events = []

    for raw_vec in itertools.product(*ranges):
        vec = apply_decc_aggregation(branches, list(raw_vec))
        latency, events = evaluate_pipeline_latency(cost_table, vec)
        if latency < best_latency:
            best_latency = latency
            best_vec = vec
            best_events = events

    return best_vec or [], best_latency, best_events


def _select_global_pareto_dp(branches: List[Dict], cost_table: List[List[Dict]]) -> Tuple[List[int], float, List[Dict]]:
    states = [(0.0, 0.0, 0.0, [])]
    future_sources = _future_dependency_sources(branches)

    for branch_idx, row in enumerate(cost_table):
        next_states = []
        prefix_branches = branches[:branch_idx + 1]

        for L_dev, L_tx, L_srv, vector in states:
            for cand_idx in range(len(row)):
                repaired = apply_decc_aggregation(prefix_branches, vector + [cand_idx])
                final_idx = repaired[branch_idx]
                cand = row[final_idx]

                new_L_dev = L_dev + cand["d"]
                new_L_tx_start = max(L_tx, new_L_dev)
                new_L_tx = new_L_tx_start + cand["t"]
                new_L_srv_start = max(L_srv, new_L_tx)
                new_L_srv = new_L_srv_start + cand["s"]
                next_states.append((new_L_dev, new_L_tx, new_L_srv, repaired))

        states = _pareto_prune(next_states, branches, future_sources[branch_idx])

    best = min(states, key=lambda s: s[2])
    best_vec = best[3]
    best_latency, best_events = evaluate_pipeline_latency(cost_table, best_vec)
    return best_vec, best_latency, best_events


def _future_dependency_sources(branches: List[Dict]) -> List[List[int]]:
    out = []
    for idx in range(len(branches)):
        sources = set()
        for later in branches[idx + 1:]:
            for dep in later.get("dependencies", []):
                if dep["from_branch"] <= idx:
                    sources.add(dep["from_branch"])
        out.append(sorted(sources))
    return out


def _pareto_prune(states, branches: List[Dict], source_branches: List[int]):
    buckets = {}
    for state in states:
        signature = _dependency_signature(state[3], branches, source_branches)
        bucket = buckets.setdefault(signature, [])
        if any(_dominates(existing, state) for existing in bucket):
            continue
        bucket[:] = [existing for existing in bucket if not _dominates(state, existing)]
        bucket.append(state)

    pruned = []
    for bucket in buckets.values():
        pruned.extend(bucket)
    return pruned


def _dependency_signature(vector: List[int], branches: List[Dict], source_branches: List[int]):
    signature = []
    for branch_id in source_branches:
        if branch_id >= len(vector):
            continue
        candidate_idx = vector[branch_id]
        cut_index = branches[branch_id]["candidates"][candidate_idx]["cut_index"]
        signature.append((branch_id, cut_index))
    return tuple(signature)


def _dominates(a, b) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]
