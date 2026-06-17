from typing import Any, Dict, List


def extract_graph_summary(model, input_size: int) -> Dict[str, Any]:
    """Extract a lightweight FX DAG summary.

    The experiment only needs the graph topology, not executable subgraphs.  FX
    node predecessors and users are enough for DFS branch decomposition,
    dependency repair, and matching profiled node costs back to branches.
    """
    try:
        from torch.fx import Node, map_arg, symbolic_trace

        traced = symbolic_trace(model)
        nodes = []
        for idx, n in enumerate(traced.graph.nodes):
            predecessors = []

            def collect_node(arg):
                if isinstance(arg, Node):
                    predecessors.append(arg.name)
                return arg

            map_arg((n.args, n.kwargs), collect_node)
            nodes.append({
                "name": n.name,
                "op": str(n.op),
                "target": str(n.target),
                "index": idx,
                "predecessors": _unique_in_order(predecessors),
                "users": [u.name for u in n.users],
            })
        return {"kind": "fx", "nodes": nodes}
    except Exception as e:
        # Fallback: sequential module list. This keeps the experiment runnable for
        # models that FX cannot trace, but the result is marked as weaker.
        nodes = []
        prev = None
        for idx, (name, module) in enumerate((p for p in model.named_modules() if p[0])):
            predecessors = [prev] if prev is not None else []
            if name:
                nodes.append({
                    "name": name,
                    "op": module.__class__.__name__,
                    "target": name,
                    "index": idx,
                    "predecessors": predecessors,
                    "users": [],
                })
                if prev is not None:
                    nodes[-2]["users"].append(name)
                prev = name
        return {"kind": "module_fallback", "nodes": nodes, "error": str(e)}


def _unique_in_order(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
