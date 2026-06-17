from typing import Dict, List

from src.profiler import profile_model_forward


def build_cost_table(
    model,
    branches: List[Dict],
    loader,
    bandwidth_mbps: float,
    mobile_latency_scale: float,
    server_latency_scale: float,
) -> List[List[Dict]]:
    """Build candidate cost table.

    Target final behavior:
    - profile mobile-side prefix segment for each branch/candidate -> d
    - compute activation bytes at cut tensor -> t = bytes / bandwidth
    - profile server-side suffix segment for each branch/candidate -> s

    Current scaffold uses deterministic proxy costs derived from full-model
    latency and candidate position. This must be replaced by segment profiling
    for final experiments.
    """
    base = profile_model_forward(model, loader)
    bandwidth_Bps = bandwidth_mbps * 1_000_000 / 8.0

    table = []
    num_branches = max(1, len(branches))
    branch_base = base / num_branches

    for b in branches:
        row = []
        num_candidates = max(1, len(b["candidates"]))
        for c in b["candidates"]:
            q = c["candidate_id"]
            frac = (q + 1) / num_candidates
            d = branch_base * frac * mobile_latency_scale
            s = branch_base * (1.0 - frac + 1.0 / num_candidates) * server_latency_scale

            # Proxy activation size: larger for early cuts, smaller for later cuts.
            # Replace with real cut tensor bytes.
            activation_bytes = (1.0 - frac + 0.1) * 1_000_000
            t = activation_bytes / bandwidth_Bps

            row.append({
                "branch_id": b["branch_id"],
                "candidate_id": q,
                "d": float(d),
                "t": float(t),
                "s": float(s),
                "cut_node": c.get("cut_node", ""),
            })
        table.append(row)
    return table
