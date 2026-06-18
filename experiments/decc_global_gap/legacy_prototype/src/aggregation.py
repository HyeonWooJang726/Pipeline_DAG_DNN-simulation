from typing import Dict, List


def apply_decc_aggregation(branches: List[Dict], partition_vector: List[int]) -> List[int]:
    """Apply DECC-style dependency aggregation to a candidate vector.

    The vector format is intentionally small: each entry is a candidate index
    into that branch's candidate list.  A candidate's ``cut_index`` means branch
    nodes with position ``<= cut_index`` run on the device and later nodes run
    on the server.

    If an earlier dependency source already lives on the server, a later branch
    cannot keep a consumer of that value in its device-side prefix.  We repair
    the later branch to the nearest earlier cut candidate.  This is the vector
    form of DECC's aggregation rule: dependent cloud-side work is aggregated
    with the earlier branch instead of forcing an extra impossible transfer.
    """
    repaired = _clamp_vector(branches, partition_vector)

    changed = True
    max_passes = max(1, len(branches) * len(branches))
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1

        for branch in branches:
            branch_id = branch["branch_id"]
            for dep in branch.get("dependencies", []):
                from_branch = dep["from_branch"]
                if from_branch >= branch_id:
                    continue

                source_cut = _selected_cut_index(branches[from_branch], repaired[from_branch])
                source_on_server = dep["from_position"] > source_cut
                consumer_cut = _selected_cut_index(branch, repaired[branch_id])
                consumer_on_device = dep["to_position"] <= consumer_cut

                if source_on_server and consumer_on_device:
                    new_idx = _nearest_candidate_before(branch, dep["to_position"])
                    if new_idx != repaired[branch_id]:
                        repaired[branch_id] = new_idx
                        changed = True

    return repaired


def _clamp_vector(branches: List[Dict], partition_vector: List[int]) -> List[int]:
    if len(partition_vector) != len(branches):
        raise ValueError(
            f"Partition vector length {len(partition_vector)} does not match branch count {len(branches)}"
        )

    clamped = []
    for branch, cand_idx in zip(branches, partition_vector):
        last = len(branch.get("candidates", [])) - 1
        if last < 0:
            raise ValueError(f"Branch {branch.get('branch_id')} has no candidates")
        clamped.append(min(max(int(cand_idx), 0), last))
    return clamped


def _selected_cut_index(branch: Dict, candidate_idx: int) -> int:
    return int(branch["candidates"][candidate_idx]["cut_index"])


def _nearest_candidate_before(branch: Dict, node_position: int) -> int:
    best_idx = 0
    best_cut = -1
    for idx, cand in enumerate(branch["candidates"]):
        cut = int(cand["cut_index"])
        if cut < node_position and cut > best_cut:
            best_idx = idx
            best_cut = cut
    return best_idx
