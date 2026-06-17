from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class GraphNode:
    name: str
    op: str
    target: str
    users: List[str]


def extract_graph_summary(model, input_size: int) -> Dict[str, Any]:
    """Extract a lightweight FX graph summary.

    This is a starting point for Codex. For complex models, the extracted FX graph
    should be validated manually against forward behavior.
    """
    try:
        from torch.fx import symbolic_trace
        traced = symbolic_trace(model)
        nodes = []
        for n in traced.graph.nodes:
            nodes.append({
                "name": n.name,
                "op": str(n.op),
                "target": str(n.target),
                "users": [u.name for u in n.users],
            })
        return {"kind": "fx", "nodes": nodes}
    except Exception as e:
        # Fallback: sequential module list. This is weaker and should be marked as such.
        nodes = []
        for name, module in model.named_modules():
            if name:
                nodes.append({"name": name, "op": module.__class__.__name__, "target": name, "users": []})
        return {"kind": "module_fallback", "nodes": nodes, "error": str(e)}
