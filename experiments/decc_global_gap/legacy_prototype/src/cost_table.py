from typing import Dict, List

from src.profiler import profile_fx_node_costs, profile_model_forward


def build_cost_table(
    model,
    branches: List[Dict],
    loader,
    bandwidth_mbps: float,
    mobile_latency_scale: float,
    server_latency_scale: float,
    profile_warmup: int = 1,
    profile_repeat: int = 3,
) -> List[List[Dict]]:
    """Build profiling-based candidate costs.

    For branch ``i`` and candidate ``q``:
    - ``d_i^q`` is the profiled prefix latency, scaled to emulate mobile.
    - ``t_i^q`` is profiled cut activation bytes divided by fixed bandwidth.
    - ``s_i^q`` is the profiled suffix latency, scaled to emulate server.
    """
    bandwidth_Bps = bandwidth_mbps * 1_000_000 / 8.0
    if bandwidth_Bps <= 0:
        raise ValueError("bandwidth_mbps must be positive")

    try:
        profile = profile_fx_node_costs(
            model,
            loader,
            warmup=profile_warmup,
            repeat=profile_repeat,
        )
        node_latency = profile["latency_by_node"]
        node_bytes = profile["activation_bytes_by_node"]
        input_bytes = int(profile.get("input_bytes", 0))
    except Exception:
        # Last-resort fallback for a model that cannot be FX-interpreted.  This
        # keeps the command runnable, and the README documents the limitation.
        base = profile_model_forward(model, loader, warmup=profile_warmup, repeat=profile_repeat)
        all_nodes = [node for branch in branches for node in branch["nodes"]]
        per_node = base / max(1, len(all_nodes))
        node_latency = {node: per_node for node in all_nodes}
        node_bytes = {}
        input_bytes = 0

    table = []
    for b in branches:
        row = []
        branch_nodes = b["nodes"]
        latencies = [float(node_latency.get(node, 0.0)) for node in branch_nodes]
        prefix_sums = _prefix_sums(latencies)

        for c in b["candidates"]:
            cut_index = int(c["cut_index"])
            cut_index = min(max(cut_index, 0), len(branch_nodes) - 1)

            prefix_latency = prefix_sums[cut_index + 1]
            suffix_latency = prefix_sums[-1] - prefix_latency
            d = prefix_latency * mobile_latency_scale
            s = suffix_latency * server_latency_scale

            activation_bytes = int(node_bytes.get(c["cut_node"], input_bytes))
            t = activation_bytes / bandwidth_Bps

            row.append({
                "branch_id": b["branch_id"],
                "candidate_id": c["candidate_id"],
                "d": float(d),
                "t": float(t),
                "s": float(s),
                "cut_node": c.get("cut_node", ""),
                "cut_index": cut_index,
                "activation_bytes": activation_bytes,
            })
        table.append(row)
    return table


def _prefix_sums(values: List[float]) -> List[float]:
    sums = [0.0]
    total = 0.0
    for value in values:
        total += value
        sums.append(total)
    return sums
