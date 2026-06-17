from typing import Dict, List, Tuple


def evaluate_pipeline_latency(cost_table: List[List[Dict]], partition_vector: List[int]) -> Tuple[float, List[Dict]]:
    L_dev = 0.0
    L_tx = 0.0
    L_srv = 0.0
    events = []

    for i, cand_idx in enumerate(partition_vector):
        cand = cost_table[i][cand_idx]
        d = cand["d"]
        t = cand["t"]
        s = cand["s"]

        dev_start = L_dev
        L_dev = L_dev + d

        tx_start = max(L_tx, L_dev)
        L_tx = tx_start + t

        srv_start = max(L_srv, L_tx)
        L_srv = srv_start + s

        events.extend([
            {"branch": i, "stage": "device", "start": dev_start, "end": L_dev},
            {"branch": i, "stage": "tx", "start": tx_start, "end": L_tx},
            {"branch": i, "stage": "server", "start": srv_start, "end": L_srv},
        ])

    return L_srv, events
