from typing import Dict, List


def build_cost_table_v2(
    branches: List[Dict],
    profile: Dict,
    bandwidth_mbps: float,
    mobile_latency_scale: float,
    server_latency_scale: float,
) -> List[List[Dict]]:
    bandwidth_Bps = bandwidth_mbps * 1_000_000.0 / 8.0
    if bandwidth_Bps <= 0:
        raise ValueError("bandwidth_mbps must be positive")

    node_latency = profile["latency_by_node"]
    node_bytes = profile["activation_bytes_by_node"]
    input_bytes = int(profile.get("input_bytes", 0))

    table = []
    for branch in branches:
        branch_nodes = branch["nodes"]
        raw_latencies = [float(node_latency.get(node, 0.0)) for node in branch_nodes]
        prefix_sums = _prefix_sums(raw_latencies)
        full_raw = prefix_sums[-1]
        full_branch_server_cost = full_raw * server_latency_scale
        full_branch_mobile_cost = full_raw * mobile_latency_scale
        server_cost_by_node = {
            node: raw_latencies[idx] * server_latency_scale
            for idx, node in enumerate(branch_nodes)
        }

        row = []
        for candidate in branch["candidates"]:
            cut_index = min(max(int(candidate["cut_index"]), 0), len(branch_nodes) - 1)
            cut_node = branch_nodes[cut_index]
            prefix_latency_raw = prefix_sums[cut_index + 1]
            suffix_latency_raw = full_raw - prefix_latency_raw
            activation_bytes = int(node_bytes.get(cut_node, input_bytes))

            row.append({
                "branch_id": branch["branch_id"],
                "candidate_id": candidate["candidate_id"],
                "cut_index": cut_index,
                "cut_node": cut_node,
                "d": prefix_latency_raw * mobile_latency_scale,
                "t": activation_bytes / bandwidth_Bps,
                "s": suffix_latency_raw * server_latency_scale,
                "prefix_latency_raw": prefix_latency_raw,
                "suffix_latency_raw": suffix_latency_raw,
                "activation_bytes": activation_bytes,
                "full_branch_server_cost": full_branch_server_cost,
                "full_branch_mobile_cost": full_branch_mobile_cost,
                "branch_nodes": list(branch_nodes),
                "node_latency_raw": dict(zip(branch_nodes, raw_latencies)),
                "server_cost_by_node": server_cost_by_node,
            })
        table.append(row)

    return table


def _prefix_sums(values: List[float]) -> List[float]:
    out = [0.0]
    total = 0.0
    for value in values:
        total += value
        out.append(total)
    return out
