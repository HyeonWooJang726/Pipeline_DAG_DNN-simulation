import time
from typing import Iterable

import torch


@torch.no_grad()
def profile_model_forward(model, loader: Iterable, warmup: int = 2, repeat: int = 5) -> float:
    """Profile full model forward latency in seconds.

    This helper is intentionally simple. For candidate segment profiling, Codex
    should extend it to run prefix/suffix subgraphs.
    """
    xs = []
    for x, _ in loader:
        xs.append(x)
        if len(xs) >= max(warmup, repeat):
            break

    for x in xs[:warmup]:
        _ = model(x)

    t0 = time.perf_counter()
    count = 0
    for x in xs[:repeat]:
        _ = model(x)
        count += 1
    t1 = time.perf_counter()
    return (t1 - t0) / max(1, count)
