import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from src.model_loader import load_model
from src.data_loader import make_loader
from src.graph_extractor import extract_graph_summary
from src.decc_branching import decc_branch_decompose
from src.cost_table import build_cost_table
from src.selectors import select_decc_style, select_global_bruteforce
from src.aggregation import apply_decc_aggregation
from src.latency_eval import evaluate_pipeline_latency


def run_one_model(model_cfg, cfg):
    model_name = model_cfg["name"]
    input_size = int(model_cfg["input_size"])

    model = load_model(model_name)
    loader = make_loader(
        input_size=input_size,
        num_images=int(cfg["num_images"]),
        batch_size=int(cfg["batch_size"]),
    )

    graph = extract_graph_summary(model, input_size=input_size)
    branches = decc_branch_decompose(graph, max_candidates_per_branch=int(cfg["max_candidates_per_branch"]))

    cost_table = build_cost_table(
        model=model,
        branches=branches,
        loader=loader,
        bandwidth_mbps=float(cfg["bandwidth_mbps"]),
        mobile_latency_scale=float(cfg["mobile_latency_scale"]),
        server_latency_scale=float(cfg["server_latency_scale"]),
    )

    t0 = time.perf_counter()
    decc_raw = select_decc_style(cost_table)
    decc_vec = apply_decc_aggregation(branches, decc_raw)
    decc_latency, decc_events = evaluate_pipeline_latency(cost_table, decc_vec)
    decc_runtime = time.perf_counter() - t0

    t1 = time.perf_counter()
    global_vec, global_latency, global_events = select_global_bruteforce(branches, cost_table)
    global_runtime = time.perf_counter() - t1

    gap = 0.0
    if decc_latency > 0:
        gap = (decc_latency - global_latency) / decc_latency * 100.0

    return {
        "summary": {
            "model": model_name,
            "structure": model_cfg.get("structure", ""),
            "branches": len(branches),
            "candidates_total": sum(len(b["candidates"]) for b in branches),
            "decc_style_latency_ms": decc_latency * 1000.0,
            "global_latency_ms": global_latency * 1000.0,
            "optimality_gap_percent": gap,
            "decc_runtime_sec": decc_runtime,
            "global_runtime_sec": global_runtime,
        },
        "selected": {
            "model": model_name,
            "decc_style_vector": decc_vec,
            "global_vector": global_vec,
        },
        "gantt": {
            "model": model_name,
            "decc_style": decc_events,
            "global": global_events,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    selected = []
    gantt = []

    for model_cfg in cfg["models"]:
        print(f"[RUN] {model_cfg['name']}")
        out = run_one_model(model_cfg, cfg)
        summaries.append(out["summary"])
        selected.append(out["selected"])
        gantt.append(out["gantt"])

    pd.DataFrame(summaries).to_csv(results_dir / "summary.csv", index=False)
    with open(results_dir / "selected_partitions.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    with open(results_dir / "gantt_events.json", "w", encoding="utf-8") as f:
        json.dump(gantt, f, indent=2)

    print(pd.DataFrame(summaries))


if __name__ == "__main__":
    main()
