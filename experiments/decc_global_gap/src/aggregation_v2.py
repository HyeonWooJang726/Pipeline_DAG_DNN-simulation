from typing import Dict, List, Optional


def apply_decc_aggregation_v2(
    branches: List[Dict],
    partition_vector: List[int],
    cost_table: List[List[Dict]],
    dependencies: Optional[List[Dict]] = None,
) -> Dict:
    """Build an effective DECC-style execution plan for one partition vector."""
    dependencies = dependencies if dependencies is not None else _collect_dependencies(branches)
    vector = _clamp_vector(branches, partition_vector)
    effective = []

    for branch, candidate_idx in zip(branches, vector):
        row = cost_table[branch["branch_id"]]
        selected = row[candidate_idx]
        effective.append({
            "branch_id": branch["branch_id"],
            "d": float(selected["d"]),
            "t": float(selected["t"]),
            "s": float(selected["s"]),
            "cut_index": int(selected["cut_index"]),
            "cut_node": selected["cut_node"],
            "full_branch_server_cost": float(selected["full_branch_server_cost"]),
            "full_branch_mobile_cost": float(selected["full_branch_mobile_cost"]),
            "cloud_only": False,
        })

    events = []
    full_cloud = [False for _ in branches]

    changed = True
    passes = 0
    while changed and passes < max(1, len(branches) * len(branches)):
        changed = False
        passes += 1
        for dep in _sorted_dependencies(dependencies):
            from_branch = int(dep["from_branch"])
            to_branch = int(dep["to_branch"])
            if from_branch >= to_branch or full_cloud[to_branch]:
                continue
            if _node_is_cloud(branches, effective, full_cloud, from_branch, int(dep["from_position"])):
                selected = cost_table[to_branch][vector[to_branch]]
                effective[to_branch].update({
                    "d": 0.0,
                    "t": 0.0,
                    "s": float(selected["full_branch_server_cost"]),
                    "cloud_only": True,
                })
                full_cloud[to_branch] = True
                changed = True
                events.append({
                    "event_type": "case1_full_cloud",
                    "from_branch": from_branch,
                    "from_node": dep["from_node"],
                    "to_branch": to_branch,
                    "to_node": dep["to_node"],
                    "reason": "dependency_source_already_cloud",
                })

    aggregated_nodes = set()
    for dep in _sorted_dependencies(dependencies):
        from_branch = int(dep["from_branch"])
        to_branch = int(dep["to_branch"])
        if from_branch >= to_branch:
            continue
        from_position = int(dep["from_position"])
        to_position = int(dep["to_position"])
        if not _node_is_cloud(branches, effective, full_cloud, from_branch, from_position):
            continue
        if not _node_is_cloud(branches, effective, full_cloud, to_branch, to_position):
            continue

        moved_nodes = _suffix_nodes(branches[to_branch], to_position)
        movable_nodes = [
            node for node in moved_nodes
            if (to_branch, node) not in aggregated_nodes
        ]
        moved_cost = _server_cost_for_nodes(cost_table[to_branch][vector[to_branch]], movable_nodes)
        if moved_cost <= 0.0 or not movable_nodes:
            continue

        for node in movable_nodes:
            aggregated_nodes.add((to_branch, node))
        effective[from_branch]["s"] += moved_cost
        effective[to_branch]["s"] = max(0.0, effective[to_branch]["s"] - moved_cost)
        events.append({
            "event_type": "case2_suffix_merge",
            "from_branch": from_branch,
            "from_node": dep["from_node"],
            "to_branch": to_branch,
            "to_node": dep["to_node"],
            "moved_nodes": movable_nodes,
            "moved_suffix_server_cost": moved_cost,
        })

    public_effective = [
        {
            "branch_id": cost["branch_id"],
            "d": cost["d"],
            "t": cost["t"],
            "s": cost["s"],
            "cut_index": cost["cut_index"],
            "cut_node": cost["cut_node"],
            "cloud_only": cost["cloud_only"],
        }
        for cost in effective
    ]

    return {
        "vector": vector,
        "effective_costs": public_effective,
        "aggregation_events": events,
        "full_cloud_branches": [idx for idx, value in enumerate(full_cloud) if value],
        "aggregated_nodes": [
            {"branch_id": branch_id, "node": node}
            for branch_id, node in sorted(aggregated_nodes)
        ],
    }


def _clamp_vector(branches: List[Dict], partition_vector: List[int]) -> List[int]:
    if len(partition_vector) != len(branches):
        raise ValueError("Partition vector length does not match branch count")
    out = []
    for branch, value in zip(branches, partition_vector):
        last = len(branch.get("candidates", [])) - 1
        if last < 0:
            raise ValueError(f"Branch {branch.get('branch_id')} has no candidates")
        out.append(min(max(int(value), 0), last))
    return out


def _node_is_cloud(branches, effective, full_cloud, branch_id: int, position: int) -> bool:
    if full_cloud[branch_id]:
        return True
    return position > int(effective[branch_id]["cut_index"])


def _suffix_nodes(branch: Dict, start_position: int) -> List[str]:
    start = min(max(start_position, 0), len(branch["nodes"]))
    return list(branch["nodes"][start:])


def _server_cost_for_nodes(selected_cost: Dict, node_names: List[str]) -> float:
    server_cost_by_node = selected_cost.get("server_cost_by_node", {})
    return sum(float(server_cost_by_node.get(node, 0.0)) for node in node_names)


def _collect_dependencies(branches: List[Dict]) -> List[Dict]:
    dependencies = []
    seen = set()
    for branch in branches:
        for dep in branch.get("dependencies", []):
            key = (
                dep["from_branch"],
                dep["from_position"],
                dep["to_branch"],
                dep["to_position"],
                dep["from_node"],
                dep["to_node"],
            )
            if key in seen:
                continue
            seen.add(key)
            dependencies.append(dep)
    return dependencies


def _sorted_dependencies(dependencies: List[Dict]) -> List[Dict]:
    return sorted(
        dependencies,
        key=lambda d: (d["to_branch"], d["to_position"], d["from_branch"], d["from_position"]),
    )
