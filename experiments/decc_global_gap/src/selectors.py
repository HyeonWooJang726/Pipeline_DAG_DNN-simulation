import itertools
from typing import Dict, List, Tuple

from src.aggregation import apply_decc_aggregation
from src.latency_eval import evaluate_pipeline_latency


def select_decc_style(cost_table: List[List[Dict]]) -> List[int]:
    vector = []
    for row in cost_table:
        best_idx = min(range(len(row)), key=lambda j: row[j]["d"] + row[j]["t"] + row[j]["s"])
        vector.append(best_idx)
    return vector


def select_global_bruteforce(branches: List[Dict], cost_table: List[List[Dict]]) -> Tuple[List[int], float, List[Dict]]:
    ranges = [range(len(row)) for row in cost_table]

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
