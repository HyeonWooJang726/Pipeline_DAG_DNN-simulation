import itertools
import math
from typing import Callable, Dict, List, Optional

from src.latency_eval_v2 import evaluate_pipeline_latency_v2


def select_decc_style_v2(
    branches: List[Dict],
    cost_table: List[List[Dict]],
    aggregation_fn: Callable,
) -> Dict:
    raw_vector = [
        min(range(len(row)), key=lambda idx: row[idx]["d"] + row[idx]["t"] + row[idx]["s"])
        for row in cost_table
    ]
    aggregation = aggregation_fn(branches, raw_vector, cost_table)
    latency = evaluate_pipeline_latency_v2(aggregation["effective_costs"])
    return {
        "status": "evaluated",
        "raw_vector": raw_vector,
        "vector": aggregation["vector"],
        "latency_s": latency["final_latency_s"],
        "latency_eval": latency,
        "aggregation": aggregation,
        "combination_count": _combination_count(cost_table),
        "global_solver": "not_global",
    }


def select_global_exhaustive_v2(
    branches: List[Dict],
    cost_table: List[List[Dict]],
    aggregation_fn: Callable,
    max_combinations: Optional[int] = None,
) -> Dict:
    combinations = _combination_count(cost_table)
    if max_combinations is not None and combinations > max_combinations:
        return {
            "status": "skipped",
            "skip_reason": "skipped_too_many_combinations",
            "combination_count": combinations,
            "global_solver": "skipped_too_many_combinations",
        }

    best = None
    ranges = [range(len(row)) for row in cost_table]
    for raw_vector_tuple in itertools.product(*ranges):
        raw_vector = list(raw_vector_tuple)
        aggregation = aggregation_fn(branches, raw_vector, cost_table)
        latency = evaluate_pipeline_latency_v2(aggregation["effective_costs"])
        candidate = {
            "status": "evaluated",
            "raw_vector": raw_vector,
            "vector": aggregation["vector"],
            "latency_s": latency["final_latency_s"],
            "latency_eval": latency,
            "aggregation": aggregation,
            "combination_count": combinations,
            "global_solver": "bruteforce_exhaustive",
        }
        if best is None or candidate["latency_s"] < best["latency_s"]:
            best = candidate

    if best is None:
        return {
            "status": "skipped",
            "skip_reason": "no_candidate_combinations",
            "combination_count": combinations,
            "global_solver": "bruteforce_exhaustive",
        }
    return best


def _combination_count(cost_table: List[List[Dict]]) -> int:
    if not cost_table:
        return 0
    return math.prod(len(row) for row in cost_table)
