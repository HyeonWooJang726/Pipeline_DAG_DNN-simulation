import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml

from src.aggregation_v2 import apply_decc_aggregation_v2
from src.block_extractor import extract_model_blocks
from src.block_input_collector import collect_block_inputs, make_proof_loader
from src.block_profiler import profile_block_fx_nodes
from src.cost_table_v2 import build_cost_table_v2
from src.decc_branching_v2 import decc_branch_decompose_v2
from src.exhaustive_search import select_decc_style_v2, select_global_exhaustive_v2
from src.model_loader import load_model


SUMMARY_COLUMNS = [
    "model",
    "block_id",
    "block_type",
    "branches",
    "candidates_total",
    "combination_count",
    "global_solver",
    "decc_style_latency_ms",
    "global_latency_ms",
    "optimality_gap_percent",
    "relative_slowdown_vs_global_percent",
    "device_busy_ms",
    "tx_busy_ms",
    "server_busy_ms",
    "bottleneck_stage",
    "cost_model",
    "input_source",
    "graph_kind",
    "status",
    "skip_reason",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proof.yaml")
    args = parser.parse_args()

    experiment_dir = Path(__file__).resolve().parent
    config_path = _resolve_path(args.config, experiment_dir)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg.get("results_dir", "results_proof"))
    if not results_dir.is_absolute():
        results_dir = experiment_dir / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.get("seed", 0)))

    summaries = []
    block_results = []
    selected_partitions = []
    gantt_events = []
    aggregation_events = []
    cost_tables = []
    effective_costs = []

    for model_cfg in cfg["models"]:
        print(f"[RUN] {model_cfg['name']}")
        outputs = run_one_model(model_cfg, cfg, experiment_dir)
        summaries.extend(outputs["summaries"])
        block_results.extend(outputs["block_results"])
        selected_partitions.extend(outputs["selected_partitions"])
        gantt_events.extend(outputs["gantt_events"])
        aggregation_events.extend(outputs["aggregation_events"])
        cost_tables.extend(outputs["cost_tables"])
        effective_costs.extend(outputs["effective_costs"])

    summary_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    _write_json(results_dir / "block_results.json", block_results)
    _write_json(results_dir / "selected_partitions.json", selected_partitions)
    _write_json(results_dir / "gantt_events.json", gantt_events)
    _write_json(results_dir / "aggregation_events.json", aggregation_events)
    _write_json(results_dir / "cost_tables.json", cost_tables)
    _write_json(results_dir / "effective_costs.json", effective_costs)
    print(summary_df)


def run_one_model(model_cfg, cfg, experiment_dir: Path):
    model_name = model_cfg["name"]
    input_size = int(model_cfg["input_size"])
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    model = load_model(model_name)
    blocks = extract_model_blocks(model, model_name)
    cifar10_root = cfg.get("cifar10_root", "data")
    cifar10_root = str(_resolve_path(cifar10_root, experiment_dir))
    loader, input_source = make_proof_loader(
        input_size=input_size,
        num_images=int(cfg["num_images"]),
        batch_size=int(cfg["batch_size"]),
        seed=seed,
        cifar10_root=cifar10_root,
        require_real_cifar10=bool(cfg.get("require_real_cifar10", False)),
    )
    collected = collect_block_inputs(model, blocks, loader, input_source)

    outputs = {
        "summaries": [],
        "block_results": [],
        "selected_partitions": [],
        "gantt_events": [],
        "aggregation_events": [],
        "cost_tables": [],
        "effective_costs": [],
    }

    for block in blocks:
        result = run_one_block(block, collected.get(block["block_id"], {}), model_name, cfg)
        for key in outputs:
            outputs[key].append(result[key])

    return outputs


def run_one_block(block, collected_entry, model_name: str, cfg):
    block_id = block["block_id"]
    block_type = block["block_type"]
    input_source = collected_entry.get("input_source", "unknown")
    inputs = collected_entry.get("inputs", [])
    base_summary = {
        "model": model_name,
        "block_id": block_id,
        "block_type": block_type,
        "branches": 0,
        "candidates_total": 0,
        "combination_count": 0,
        "global_solver": "",
        "decc_style_latency_ms": "",
        "global_latency_ms": "",
        "optimality_gap_percent": "",
        "relative_slowdown_vs_global_percent": "",
        "device_busy_ms": "",
        "tx_busy_ms": "",
        "server_busy_ms": "",
        "bottleneck_stage": "",
        "cost_model": "",
        "input_source": input_source,
        "graph_kind": "",
        "status": "skipped",
        "skip_reason": "",
    }

    try:
        if not inputs:
            raise RuntimeError("no_collected_block_inputs")

        decomposition = decc_branch_decompose_v2(
            block["module"],
            max_candidates_per_branch=int(cfg["max_candidates_per_branch"]),
        )
        branches = decomposition["branches"]
        if not branches:
            raise RuntimeError("no_fx_compute_branches")

        profile = profile_block_fx_nodes(block["module"], inputs, max_samples=1)
        cost_table = build_cost_table_v2(
            branches=branches,
            profile=profile,
            bandwidth_mbps=float(cfg["bandwidth_mbps"]),
            mobile_latency_scale=float(cfg["mobile_latency_scale"]),
            server_latency_scale=float(cfg["server_latency_scale"]),
        )

        aggregation_fn = lambda b, v, c: apply_decc_aggregation_v2(
            b,
            v,
            c,
            dependencies=decomposition["dependencies"],
        )
        decc = select_decc_style_v2(branches, cost_table, aggregation_fn)
        global_result = select_global_exhaustive_v2(
            branches,
            cost_table,
            aggregation_fn,
            max_combinations=int(cfg["max_bruteforce_combinations_per_block"]),
        )

        candidates_total = sum(len(branch["candidates"]) for branch in branches)
        combination_count = global_result["combination_count"]
        summary = dict(base_summary)
        summary.update({
            "branches": len(branches),
            "candidates_total": candidates_total,
            "combination_count": combination_count,
            "global_solver": global_result["global_solver"],
            "decc_style_latency_ms": decc["latency_s"] * 1000.0,
            "cost_model": profile["cost_model"],
            "graph_kind": decomposition["graph_kind"],
        })

        if global_result["status"] == "skipped":
            summary.update({
                "status": "skipped",
                "skip_reason": global_result["skip_reason"],
            })
        else:
            decc_latency = decc["latency_s"]
            global_latency = global_result["latency_s"]
            gap = 0.0 if decc_latency <= 0.0 else (decc_latency - global_latency) / decc_latency * 100.0
            slowdown = 0.0 if global_latency <= 0.0 else (decc_latency - global_latency) / global_latency * 100.0
            global_eval = global_result["latency_eval"]
            summary.update({
                "global_latency_ms": global_latency * 1000.0,
                "optimality_gap_percent": gap,
                "relative_slowdown_vs_global_percent": slowdown,
                "device_busy_ms": global_eval["device_busy"] * 1000.0,
                "tx_busy_ms": global_eval["tx_busy"] * 1000.0,
                "server_busy_ms": global_eval["server_busy"] * 1000.0,
                "bottleneck_stage": global_eval["bottleneck_stage"],
                "status": "evaluated",
                "skip_reason": "",
            })

        return {
            "summaries": summary,
            "block_results": {
                "model": model_name,
                "block_id": block_id,
                "block_type": block_type,
                "status": summary["status"],
                "skip_reason": summary["skip_reason"],
                "input_count": len(inputs),
                "profile_sample_count": profile["sample_count"],
                "branches": branches,
                "dependencies": decomposition["dependencies"],
                "branching_events": decomposition["branching_events"],
                "graph_nodes": decomposition["graph_nodes"],
                "cost_model": profile["cost_model"],
            },
            "selected_partitions": {
                "model": model_name,
                "block_id": block_id,
                "decc_style_raw_vector": decc["raw_vector"],
                "decc_style_vector": decc["vector"],
                "global_raw_vector": global_result.get("raw_vector"),
                "global_vector": global_result.get("vector"),
                "decc_style_cut_nodes": _cut_nodes(branches, decc["vector"]),
                "global_cut_nodes": _cut_nodes(branches, global_result.get("vector") or []),
            },
            "gantt_events": {
                "model": model_name,
                "block_id": block_id,
                "decc_style": decc["latency_eval"]["gantt_events"],
                "global": global_result.get("latency_eval", {}).get("gantt_events", []),
            },
            "aggregation_events": {
                "model": model_name,
                "block_id": block_id,
                "decc_style": decc["aggregation"]["aggregation_events"],
                "global": global_result.get("aggregation", {}).get("aggregation_events", []),
            },
            "cost_tables": _cost_table_artifact(
                model_name=model_name,
                block=block,
                cost_table=cost_table,
                cost_model=profile["cost_model"],
            ),
            "effective_costs": _effective_costs_artifact(
                model_name=model_name,
                block_id=block_id,
                branches=branches,
                decc=decc,
                global_result=global_result,
            ),
        }

    except Exception as exc:
        summary = dict(base_summary)
        summary["skip_reason"] = str(exc)
        return {
            "summaries": summary,
            "block_results": {
                "model": model_name,
                "block_id": block_id,
                "block_type": block_type,
                "status": "skipped",
                "skip_reason": str(exc),
                "input_count": len(inputs),
            },
            "selected_partitions": {
                "model": model_name,
                "block_id": block_id,
                "decc_style_raw_vector": None,
                "decc_style_vector": None,
                "global_raw_vector": None,
                "global_vector": None,
                "decc_style_cut_nodes": [],
                "global_cut_nodes": [],
            },
            "gantt_events": {
                "model": model_name,
                "block_id": block_id,
                "decc_style": [],
                "global": [],
            },
            "aggregation_events": {
                "model": model_name,
                "block_id": block_id,
                "decc_style": [],
                "global": [],
            },
            "cost_tables": {
                "model": model_name,
                "block_id": block_id,
                "block_type": block_type,
                "status": "skipped",
                "skip_reason": str(exc),
                "branches": [],
            },
            "effective_costs": {
                "model": model_name,
                "block_id": block_id,
                "block_type": block_type,
                "status": "skipped",
                "skip_reason": str(exc),
                "decc_style": None,
                "global": None,
            },
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
    except Exception:
        pass


def _resolve_path(value, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() and path.exists():
        return path.resolve()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _write_json(path: Path, value) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def _cost_table_artifact(model_name: str, block, cost_table, cost_model: str):
    branches = []
    for branch_row in cost_table:
        if not branch_row:
            continue
        branch_id = int(branch_row[0]["branch_id"])
        branches.append({
            "branch_id": branch_id,
            "candidates": [
                {
                    "candidate_id": int(candidate["candidate_id"]),
                    "cut_index": int(candidate["cut_index"]),
                    "cut_node": candidate["cut_node"],
                    "d_ms": float(candidate["d"]) * 1000.0,
                    "t_ms": float(candidate["t"]) * 1000.0,
                    "s_ms": float(candidate["s"]) * 1000.0,
                    "total_local_ms": (
                        float(candidate["d"]) + float(candidate["t"]) + float(candidate["s"])
                    ) * 1000.0,
                    "prefix_latency_raw": float(candidate["prefix_latency_raw"]),
                    "suffix_latency_raw": float(candidate["suffix_latency_raw"]),
                    "activation_bytes": int(candidate["activation_bytes"]),
                    "full_branch_server_cost": float(candidate["full_branch_server_cost"]),
                    "full_branch_mobile_cost": float(candidate["full_branch_mobile_cost"]),
                }
                for candidate in branch_row
            ],
        })

    return {
        "model": model_name,
        "block_id": block["block_id"],
        "block_type": block["block_type"],
        "status": "evaluated",
        "cost_model": cost_model,
        "branches": branches,
    }


def _effective_costs_artifact(model_name: str, block_id: str, branches, decc, global_result):
    return {
        "model": model_name,
        "block_id": block_id,
        "status": "evaluated" if global_result.get("status") != "skipped" else "skipped",
        "decc_style": _method_effective_artifact(branches, decc),
        "global": _method_effective_artifact(branches, global_result),
    }


def _method_effective_artifact(branches, result):
    if not result or result.get("status") == "skipped":
        return {
            "status": result.get("status", "skipped") if result else "skipped",
            "skip_reason": result.get("skip_reason", "") if result else "",
            "raw_vector": result.get("raw_vector") if result else None,
            "repaired_vector": result.get("vector") if result else None,
            "cut_nodes": [],
            "effective_costs": [],
            "aggregation_events": [],
            "final_latency_ms": None,
            "device_busy_ms": None,
            "tx_busy_ms": None,
            "server_busy_ms": None,
            "bottleneck_stage": None,
        }

    latency_eval = result["latency_eval"]
    return {
        "status": result.get("status", "evaluated"),
        "raw_vector": result["raw_vector"],
        "repaired_vector": result["vector"],
        "cut_nodes": _cut_nodes(branches, result["vector"]),
        "effective_costs": [
            {
                "branch_id": int(cost["branch_id"]),
                "d_ms": float(cost["d"]) * 1000.0,
                "t_ms": float(cost["t"]) * 1000.0,
                "s_ms": float(cost["s"]) * 1000.0,
                "cut_index": int(cost["cut_index"]),
                "cut_node": cost["cut_node"],
                "cloud_only": bool(cost.get("cloud_only", False)),
            }
            for cost in result["aggregation"]["effective_costs"]
        ],
        "aggregation_events": result["aggregation"]["aggregation_events"],
        "final_latency_ms": float(latency_eval["final_latency_s"]) * 1000.0,
        "device_busy_ms": float(latency_eval["device_busy"]) * 1000.0,
        "tx_busy_ms": float(latency_eval["tx_busy"]) * 1000.0,
        "server_busy_ms": float(latency_eval["server_busy"]) * 1000.0,
        "bottleneck_stage": latency_eval["bottleneck_stage"],
    }


def _cut_nodes(branches, vector):
    if not vector or len(vector) != len(branches):
        return []
    return [
        branches[idx]["candidates"][candidate_idx]["cut_node"]
        for idx, candidate_idx in enumerate(vector)
    ]


if __name__ == "__main__":
    main()
