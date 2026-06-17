from typing import Any, Dict, List, Tuple


def decc_branch_decompose_v2(block_module, max_candidates_per_branch: int = 3) -> Dict[str, Any]:
    """Trace a block FX graph and build deterministic DECC-style branches."""
    graph = trace_block_fx_graph(block_module)
    nodes = [n for n in graph["nodes"] if n["op"] not in ("placeholder", "output")]
    if not nodes:
        return {
            "branches": [],
            "dependencies": [],
            "branching_events": [],
            "graph_kind": "fx_block",
            "graph_nodes": graph["nodes"],
        }

    names = [n["name"] for n in nodes]
    order = {name: idx for idx, name in enumerate(names)}
    compute = set(names)
    by_name = {n["name"]: n for n in nodes}
    succs = {
        name: sorted([u for u in by_name[name]["users"] if u in compute], key=lambda u: order[u])
        for name in names
    }
    preds = {
        name: sorted([p for p in by_name[name]["predecessors"] if p in compute], key=lambda p: order[p])
        for name in names
    }

    raw_branches: List[List[str]] = [[]]
    node_to_branch: Dict[str, Tuple[int, int]] = {}
    events: List[Dict[str, Any]] = []

    def new_branch(reason: str, from_node: str = "") -> int:
        branch_id = len(raw_branches)
        raw_branches.append([])
        events.append({
            "event_type": "new_branch",
            "branch_id": branch_id,
            "reason": reason,
            "from_node": from_node,
        })
        return branch_id

    def append_or_move(node_name: str, branch_id: int) -> bool:
        if node_name not in node_to_branch:
            node_to_branch[node_name] = (branch_id, len(raw_branches[branch_id]))
            raw_branches[branch_id].append(node_name)
            events.append({
                "event_type": "assign_node",
                "branch_id": branch_id,
                "node": node_name,
            })
            return False

        old_branch, old_pos = node_to_branch[node_name]
        if old_branch == branch_id:
            return True

        suffix = raw_branches[old_branch][old_pos:]
        raw_branches[old_branch] = raw_branches[old_branch][:old_pos]
        start_pos = len(raw_branches[branch_id])
        raw_branches[branch_id].extend(suffix)
        for offset, moved in enumerate(suffix):
            node_to_branch[moved] = (branch_id, start_pos + offset)
        events.append({
            "event_type": "move_suffix",
            "from_branch": old_branch,
            "to_branch": branch_id,
            "join_node": node_name,
            "moved_nodes": list(suffix),
        })
        return True

    def walk(node_name: str, branch_id: int) -> None:
        current = node_name
        while True:
            already_seen = append_or_move(current, branch_id)
            if already_seen:
                return

            current_succs = succs[current]
            if not current_succs:
                return

            if len(current_succs) == 1:
                current = current_succs[0]
                continue

            events.append({
                "event_type": "dfs_fork",
                "branch_id": branch_id,
                "node": current,
                "successors": list(current_succs),
            })
            walk(current_succs[0], branch_id)
            for successor in current_succs[1:]:
                walk(successor, new_branch("dfs_backtrack", from_node=current))
            return

    starts = sorted([name for name in names if not preds[name]], key=lambda n: order[n])
    for idx, start in enumerate(starts):
        if start in node_to_branch:
            continue
        branch_id = 0 if idx == 0 and not raw_branches[0] else new_branch("additional_start")
        walk(start, branch_id)

    for name in names:
        if name not in node_to_branch:
            walk(name, new_branch("unassigned_fallback"))

    raw_branches = [branch for branch in raw_branches if branch]
    node_to_branch = {}
    for branch_id, branch_nodes in enumerate(raw_branches):
        for position, node_name in enumerate(branch_nodes):
            node_to_branch[node_name] = (branch_id, position)

    dependencies = _build_dependencies(raw_branches, preds, node_to_branch)
    branches = []
    for branch_id, branch_nodes in enumerate(raw_branches):
        branch_dependencies = [d for d in dependencies if d["to_branch"] == branch_id]
        branches.append({
            "branch_id": branch_id,
            "nodes": branch_nodes,
            "candidates": _make_candidates(branch_nodes, max_candidates_per_branch),
            "dependencies": branch_dependencies,
        })

    return {
        "branches": branches,
        "dependencies": dependencies,
        "branching_events": events,
        "graph_kind": "fx_block",
        "graph_nodes": graph["nodes"],
    }


def trace_block_fx_graph(block_module) -> Dict[str, Any]:
    from torch.fx import Node, map_arg, symbolic_trace

    traced = symbolic_trace(block_module)
    nodes = []
    for idx, node in enumerate(traced.graph.nodes):
        predecessors = []

        def collect(arg):
            if isinstance(arg, Node):
                predecessors.append(arg.name)
            return arg

        map_arg((node.args, node.kwargs), collect)
        nodes.append({
            "name": node.name,
            "op": str(node.op),
            "target": str(node.target),
            "index": idx,
            "predecessors": _unique_in_order(predecessors),
            "users": [user.name for user in node.users],
        })
    return {"kind": "fx_block", "nodes": nodes, "traced": traced}


def _build_dependencies(raw_branches, preds, node_to_branch):
    dependencies = []
    seen = set()
    for to_node, pred_names in preds.items():
        if to_node not in node_to_branch:
            continue
        to_branch, to_pos = node_to_branch[to_node]
        for from_node in pred_names:
            if from_node not in node_to_branch:
                continue
            from_branch, from_pos = node_to_branch[from_node]
            if from_branch == to_branch:
                continue
            key = (from_branch, from_pos, to_branch, to_pos, from_node, to_node)
            if key in seen:
                continue
            seen.add(key)
            dependencies.append({
                "from_branch": from_branch,
                "from_node": from_node,
                "from_position": from_pos,
                "to_branch": to_branch,
                "to_node": to_node,
                "to_position": to_pos,
                "edge_type": "fx_data",
            })

    return sorted(
        dependencies,
        key=lambda d: (d["to_branch"], d["to_position"], d["from_branch"], d["from_position"]),
    )


def _make_candidates(branch_nodes: List[str], max_candidates_per_branch: int) -> List[Dict[str, Any]]:
    count = min(max(1, max_candidates_per_branch), len(branch_nodes), 3)
    indices = _candidate_indices(len(branch_nodes), count)
    return [
        {
            "candidate_id": candidate_id,
            "cut_index": cut_index,
            "cut_node": branch_nodes[cut_index],
        }
        for candidate_id, cut_index in enumerate(indices)
    ]


def _candidate_indices(length: int, count: int) -> List[int]:
    if count <= 1:
        return [length - 1]
    raw = [0, (length - 1) // 2, length - 1] if count >= 3 else [0, length - 1]
    out = []
    for idx in raw:
        if idx not in out:
            out.append(idx)
    return out


def _unique_in_order(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
