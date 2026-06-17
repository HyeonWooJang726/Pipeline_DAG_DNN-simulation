import time
from collections import defaultdict
from typing import Dict, Iterable, List

import torch


@torch.no_grad()
def profile_block_fx_nodes(block_module, block_inputs: Iterable, max_samples: int = 1) -> Dict:
    """Measure FX node costs for a block using real collected block inputs."""
    from torch.fx import Interpreter, symbolic_trace

    inputs = list(block_inputs)[:max(1, max_samples)]
    if not inputs:
        raise ValueError("Cannot profile a block without collected block inputs")

    block_module.eval()
    traced = symbolic_trace(block_module)

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

    # One untimed run avoids measuring lazy module setup where it exists.
    warmup = TimingInterpreter(traced)
    _run_interpreter(warmup, inputs[0])

    runner = TimingInterpreter(traced)
    for block_input in inputs:
        _run_interpreter(runner, block_input)

    count = max(1, len(inputs))
    return {
        "latency_by_node": {name: value / count for name, value in runner.timings.items()},
        "activation_bytes_by_node": dict(runner.activation_bytes),
        "input_bytes": _tensor_nbytes(inputs[0]),
        "sample_count": count,
        "cost_model": "fx_node_measured_proxy",
    }


def _run_interpreter(interpreter, block_input):
    if isinstance(block_input, tuple):
        return interpreter.run(*block_input)
    return interpreter.run(block_input)


def _tensor_nbytes(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(v) for v in value)
    return 0
