import time
from collections import defaultdict
from typing import Dict, Iterable

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


@torch.no_grad()
def profile_fx_node_costs(
    model,
    loader: Iterable,
    warmup: int = 1,
    repeat: int = 3,
) -> Dict:
    """Profile FX node latency and activation bytes.

    The cost table needs branch prefix/suffix costs for many candidate cuts.
    Running every prefix and suffix as its own subgraph would be much slower and
    brittle across torchvision DAGs, so this profiles the traced graph node by
    node and later sums the relevant branch nodes.  The measurements are still
    real forward execution timings, not learned or analytic proxy costs.
    """
    from torch.fx import Interpreter, symbolic_trace

    model.eval()
    batches = _collect_batches(loader, warmup + repeat)
    if not batches:
        raise ValueError("Cannot profile costs without at least one input batch")

    traced = symbolic_trace(model)

    class TimingInterpreter(Interpreter):
        def __init__(self, module):
            super().__init__(module)
            self.timings = defaultdict(float)
            self.activation_bytes = defaultdict(int)

        def run_node(self, node):
            start = time.perf_counter()
            result = super().run_node(node)
            elapsed = time.perf_counter() - start
            if node.op not in ("placeholder", "output"):
                self.timings[node.name] += elapsed
                self.activation_bytes[node.name] = max(
                    self.activation_bytes[node.name],
                    _tensor_nbytes(result),
                )
            return result

    warmup_batches = batches[:warmup]
    repeat_batches = batches[warmup:warmup + repeat] or batches[:1]

    warmup_runner = TimingInterpreter(traced)
    for batch in warmup_batches:
        warmup_runner.run(batch)

    runner = TimingInterpreter(traced)
    for batch in repeat_batches:
        runner.run(batch)

    count = max(1, len(repeat_batches))
    latency_by_node = {name: value / count for name, value in runner.timings.items()}
    activation_bytes_by_node = dict(runner.activation_bytes)

    return {
        "latency_by_node": latency_by_node,
        "activation_bytes_by_node": activation_bytes_by_node,
        "input_bytes": _tensor_nbytes(repeat_batches[0]),
        "repeat": count,
    }


def _collect_batches(loader: Iterable, limit: int):
    batches = []
    for x, _ in loader:
        batches.append(x)
        if len(batches) >= max(1, limit):
            break
    return batches


def _tensor_nbytes(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(v) for v in value)
    return 0
