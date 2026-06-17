from typing import Any, Dict, List


def decc_branch_decompose(graph: Dict[str, Any], max_candidates_per_branch: int = 5) -> List[Dict[str, Any]]:
    """DECC-style DFS branch decomposition placeholder.

    Required final behavior:
    1. Treat the model graph as a DAG.
    2. Run DFS to obtain the first branch.
    3. Create a new branch at each DFS backtracking event.
    4. If a layer already exists in a previous branch, move that layer and its
       descendants to the later branch.
    5. Preserve branch execution order.

    Current implementation is a conservative scaffold:
    - for chain/fallback graphs, it creates coarse branches from ordered nodes;
    - Codex should replace this with a faithful DECC DFS decomposition.
    """
    nodes = [n for n in graph.get("nodes", []) if n.get("op") not in ("placeholder", "output")]
    if not nodes:
        return []

    # Coarse grouping to keep brute force manageable.
    group_size = max(1, len(nodes) // 8)
    raw_branches = [nodes[i:i + group_size] for i in range(0, len(nodes), group_size)]

    branches = []
    for bi, group in enumerate(raw_branches):
        candidate_count = min(max_candidates_per_branch, max(1, len(group)))
        candidate_indices = _linspace_indices(len(group), candidate_count)
        candidates = []
        for ci, idx in enumerate(candidate_indices):
            candidates.append({
                "candidate_id": ci,
                "cut_index": idx,
                "cut_node": group[idx]["name"],
            })
        branches.append({
            "branch_id": bi,
            "nodes": [n["name"] for n in group],
            "candidates": candidates,
            "dependencies": [],
        })
    return branches


def _linspace_indices(n: int, k: int):
    if k <= 1:
        return [max(0, n - 1)]
    return sorted(set(round(i * (n - 1) / (k - 1)) for i in range(k)))
