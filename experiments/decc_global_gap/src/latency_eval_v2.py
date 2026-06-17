from typing import Dict, List


def evaluate_pipeline_latency_v2(effective_costs: List[Dict]) -> Dict:
    L_dev = 0.0
    L_tx = 0.0
    L_srv = 0.0
    gantt_events = []

    device_busy = 0.0
    tx_busy = 0.0
    server_busy = 0.0

    for index, cost in enumerate(effective_costs):
        branch_id = int(cost.get("branch_id", index))
        d = float(cost.get("d", 0.0))
        t = float(cost.get("t", 0.0))
        s = float(cost.get("s", 0.0))

        dev_start = L_dev
        L_dev = L_dev + d
        tx_start = max(L_tx, L_dev)
        L_tx = tx_start + t
        srv_start = max(L_srv, L_tx)
        L_srv = srv_start + s

        device_busy += d
        tx_busy += t
        server_busy += s

        gantt_events.extend([
            _event(branch_id, "device", dev_start, L_dev),
            _event(branch_id, "tx", tx_start, L_tx),
            _event(branch_id, "server", srv_start, L_srv),
        ])

    busy = {
        "device": device_busy,
        "tx": tx_busy,
        "server": server_busy,
    }
    bottleneck_stage = max(busy, key=busy.get) if busy else "none"

    return {
        "final_latency_s": L_srv,
        "gantt_events": gantt_events,
        "device_busy": device_busy,
        "tx_busy": tx_busy,
        "server_busy": server_busy,
        "bottleneck_stage": bottleneck_stage,
    }


def _event(branch_id: int, stage: str, start: float, end: float) -> Dict:
    return {
        "branch": branch_id,
        "stage": stage,
        "start": start,
        "end": end,
        "start_ms": start * 1000.0,
        "end_ms": end * 1000.0,
    }
