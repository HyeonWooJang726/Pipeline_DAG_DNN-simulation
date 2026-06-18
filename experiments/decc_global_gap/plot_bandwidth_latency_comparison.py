import sys
from pathlib import Path


BLOCK_ORDER = [
    "Mixed_5b",
    "Mixed_5c",
    "Mixed_5d",
    "Mixed_6a",
    "Mixed_6b",
    "Mixed_6c",
    "Mixed_6d",
    "Mixed_6e",
    "Mixed_7a",
    "Mixed_7b",
    "Mixed_7c",
]

REQUIRED_COLUMNS = [
    "model",
    "block_id",
    "status",
    "existing_latency_ms",
    "global_opt_latency_ms",
]

SUMMARY_COLUMNS = [
    "model",
    "input_source",
    "block_id",
    "bandwidth_mbps",
    "existing_latency_ms",
    "global_opt_latency_ms",
    "absolute_gap_ms",
    "gap_percent",
]

KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",
    "AppleGothic",
    "NanumGothic",
    "Nanum Gothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Noto Sans Korean",
    "UnDotum",
    "Baekmuk Gulim",
]


def main():
    experiment_dir = Path(__file__).resolve().parent
    inputs = {
        100.0: experiment_dir / "results_proof_100mbps" / "summary.csv",
        1000.0: experiment_dir / "results_proof_1000mbps" / "summary.csv",
    }
    output_dir = experiment_dir / "results_bandwidth_comparison" / "figures"
    output_csv = output_dir / "bandwidth_latency_summary_ko.csv"
    output_png = output_dir / "bandwidth_latency_comparison_ko.png"

    missing = [path for path in inputs.values() if not path.exists()]
    if missing:
        print("[ERROR] Missing required proof summary file(s):", file=sys.stderr)
        for path in missing:
            print(f"  expected: {path}", file=sys.stderr)
        print(
            "Run the proof experiments before plotting:\n"
            "  python run_proof_experiment.py --config configs/proof_100mbps.yaml\n"
            "  python run_proof_experiment.py --config configs/proof_1000mbps.yaml",
            file=sys.stderr,
        )
        return 1

    dependencies = import_dependencies()
    if dependencies is None:
        return 1
    pd, plt, font_manager = dependencies

    all_rows = []
    for bandwidth_mbps, summary_path in inputs.items():
        try:
            rows = load_inception_rows(summary_path, bandwidth_mbps, pd)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        all_rows.append(rows)

    summary = pd.concat(all_rows, ignore_index=True)
    if summary.empty:
        print(
            "[ERROR] No evaluated inception_v3 Mixed block rows were found in the proof summaries.",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    plot_latency_comparison(summary, output_png, plt, font_manager)

    print(f"[OK] Wrote {output_csv}")
    print(f"[OK] Wrote {output_png}")
    return 0


def import_dependencies():
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ModuleNotFoundError as exc:
        print(
            f"[ERROR] Missing Python dependency: {exc.name}. "
            "Install requirements with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return None
    return pd, plt, font_manager


def load_inception_rows(summary_path: Path, bandwidth_mbps: float, pd):
    df = pd.read_csv(summary_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{summary_path} is missing required column(s): {', '.join(missing_columns)}"
        )

    model_name = df["model"].astype(str).str.lower()
    if "input_source" not in df.columns:
        df["input_source"] = "unknown"
    else:
        df["input_source"] = (
            df["input_source"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown"})
        )

    filtered = df[
        model_name.isin(["inception_v3", "inceptionv3"])
        & (df["status"].astype(str).str.lower() == "evaluated")
        & df["block_id"].isin(BLOCK_ORDER)
    ].copy()

    if filtered.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    filtered["block_id"] = pd.Categorical(
        filtered["block_id"], categories=BLOCK_ORDER, ordered=True
    )
    filtered = filtered.sort_values("block_id")

    filtered["existing_latency_ms"] = pd.to_numeric(
        filtered["existing_latency_ms"], errors="coerce"
    )
    filtered["global_opt_latency_ms"] = pd.to_numeric(
        filtered["global_opt_latency_ms"], errors="coerce"
    )
    filtered = filtered.dropna(subset=["existing_latency_ms", "global_opt_latency_ms"])

    filtered["bandwidth_mbps"] = bandwidth_mbps
    filtered["absolute_gap_ms"] = (
        filtered["existing_latency_ms"] - filtered["global_opt_latency_ms"]
    )
    filtered["gap_percent"] = filtered.apply(
        lambda row: (
            (row["existing_latency_ms"] - row["global_opt_latency_ms"])
            / row["existing_latency_ms"]
            * 100.0
        )
        if row["existing_latency_ms"] != 0
        else float("nan"),
        axis=1,
    )

    missing_blocks = [
        block for block in BLOCK_ORDER if block not in set(filtered["block_id"].astype(str))
    ]
    if missing_blocks:
        print(
            f"[WARN] {summary_path} has no evaluated rows for: {', '.join(missing_blocks)}",
            file=sys.stderr,
        )

    return filtered[SUMMARY_COLUMNS]


def plot_latency_comparison(summary, output_png: Path, plt, font_manager):
    has_korean_font = configure_korean_font(plt, font_manager)
    labels = korean_labels() if has_korean_font else english_labels()
    if not has_korean_font:
        print(
            "[WARN] Korean font was not found. Falling back to English plot labels.",
            file=sys.stderr,
        )

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    colors = {
        "existing": "#4C78A8",
        "global": "#59A14F",
    }
    bar_width = 0.38
    x_positions = list(range(len(BLOCK_ORDER)))

    for axis, bandwidth_mbps in zip(axes, [100.0, 1000.0]):
        rows = (
            summary[summary["bandwidth_mbps"] == bandwidth_mbps]
            .set_index("block_id")
            .reindex(BLOCK_ORDER)
        )
        left_positions = [x - bar_width / 2 for x in x_positions]
        right_positions = [x + bar_width / 2 for x in x_positions]

        axis.bar(
            left_positions,
            rows["existing_latency_ms"],
            width=bar_width,
            label=labels["existing"],
            color=colors["existing"],
        )
        axis.bar(
            right_positions,
            rows["global_opt_latency_ms"],
            width=bar_width,
            label=labels["global"],
            color=colors["global"],
        )
        axis.set_title(f"{bandwidth_mbps:g} Mbps")
        axis.set_xlabel(labels["x"])
        axis.set_xticks(x_positions)
        axis.set_xticklabels(BLOCK_ORDER, rotation=45, ha="right")
        axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.legend()

    axes[0].set_ylabel(labels["y"])
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def configure_korean_font(plt, font_manager) -> bool:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in KOREAN_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


def korean_labels():
    return {
        "existing": "기존 방법",
        "global": "전수탐색 전역 최적 기준",
        "x": "InceptionV3 block",
        "y": "최종 파이프라인 처리시간 (ms)",
    }


def english_labels():
    return {
        "existing": "Existing branch-wise method",
        "global": "Global exhaustive method",
        "x": "InceptionV3 block",
        "y": "Final pipeline latency (ms)",
    }


if __name__ == "__main__":
    sys.exit(main())
