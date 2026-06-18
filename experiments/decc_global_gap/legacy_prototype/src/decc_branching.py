from typing import Any, Dict, List


def decc_branch_decompose(graph: Dict[str, Any], max_candidates_per_branch: int = 5) -> List[Dict[str, Any]]:
    """Create DECC-style branches from a DAG using DFS order.

    A branch is the maximal single-successor/single-predecessor chain discovered
    by DFS.  When DFS reaches a fork or a join, the current branch is closed and
    successors become later branches once their inputs have been assigned.  This
    preserves execution order while keeping explicit cross-branch dependencies
    for DECC aggregation.
    """
    nodes = [n for n in graph.get("nodes", []) if n.get("op") not in ("placeholder", "output")]
    if not nodes:
        return []

    by_name = {n["name"]: n for n in nodes}
    order = {n["name"]: i for i, n in enumerate(nodes)}
    compute = set(by_name)
    succs = {
        name: sorted([u for u in n.get("users", []) if u in compute], key=lambda x: order[x])
        for name, n in by_name.items()
    }
    preds = {
        name: sorted([p for p in n.get("predecessors", []) if p in compute], key=lambda x: order[x])
        for name, n in by_name.items()
    }

    assigned = set()
    pending = set()
    raw_branches: List[List[str]] = []

    def ready(name: str) -> bool:
        return all(p in assigned for p in preds[name])

    def schedule_start(name: str) -> None:
        if name in assigned:
            return
        if ready(name):
            collect_branch(name)
        else:
            pending.add(name)

    def drain_pending() -> None:
        progressed = True
        while progressed:
            progressed = False
            for name in sorted(list(pending), key=lambda x: order[x]):
                if name not in assigned and ready(name):
                    pending.remove(name)
                    collect_branch(name)
                    progressed = True
                    break

    def collect_branch(start: str) -> None:
        if start in assigned:
            return

        branch_nodes = []
        current = start
        next_starts: List[str] = []

        while current not in assigned:
            assigned.add(current)
            branch_nodes.append(current)

            available_succs = [s for s in succs[current] if s not in assigned]
            if len(available_succs) == 1 and len(preds[available_succs[0]]) == 1:
                current = available_succs[0]
                continue

            next_starts = available_succs
            break

        if branch_nodes:
            raw_branches.append(branch_nodes)

        for nxt in next_starts:
            schedule_start(nxt)
        drain_pending()

    starts = sorted([name for name in compute if not preds[name]], key=lambda x: order[x])
    for start in starts:
        schedule_start(start)
        drain_pending()

    # Disconnected or fallback graphs should still be represented.
    for name in sorted(compute - assigned, key=lambda x: order[x]):
        collect_branch(name)
        drain_pending()

    node_to_branch = {}
    for bi, branch_nodes in enumerate(raw_branches):
        for pos, node_name in enumerate(branch_nodes):
            node_to_branch[node_name] = (bi, pos)

    branches = []
    for bi, branch_nodes in enumerate(raw_branches):
        candidate_count = min(max_candidates_per_branch, max(1, len(branch_nodes)))
        candidate_indices = _linspace_indices(len(branch_nodes), candidate_count)
        candidates = []
        for ci, idx in enumerate(candidate_indices):
            candidates.append({
                "candidate_id": ci,
                "cut_index": idx,
                "cut_node": branch_nodes[idx],
            })

        dependencies = []
        seen_deps = set()
        for to_pos, node_name in enumerate(branch_nodes):
            for pred in preds[node_name]:
                if pred not in node_to_branch:
                    continue
                from_branch, from_pos = node_to_branch[pred]
                if from_branch == bi:
                    continue
                key = (from_branch, from_pos, to_pos, pred, node_name)
                if key in seen_deps:
                    continue
                seen_deps.add(key)
                dependencies.append({
                    "from_branch": from_branch,
                    "from_node": pred,
                    "from_position": from_pos,
                    "to_branch": bi,
                    "to_node": node_name,
                    "to_position": to_pos,
                })

        branches.append({
            "branch_id": bi,
            "nodes": branch_nodes,
            "candidates": candidates,
            "dependencies": sorted(dependencies, key=lambda d: (d["from_branch"], d["from_position"], d["to_position"])),
        })
    return branches


def _linspace_indices(n: int, k: int):
    if k <= 1:
        return [max(0, n - 1)]
    return sorted(set(round(i * (n - 1) / (k - 1)) for i in range(k)))
