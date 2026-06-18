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
    summary_path = experiment_dir / "results_proof" / "summary.csv"
    output_path = (
        experiment_dir
        / "results_proof"
        / "figures"
        / "inceptionv3_block_latency_comparison_ko.png"
    )

    if not summary_path.exists():
        print(f"[ERROR] Missing required proof summary: {summary_path}", file=sys.stderr)
        print(
            "Run the proof experiment before plotting:\n"
            "  python run_proof_experiment.py --config configs/proof.yaml",
            file=sys.stderr,
        )
        return 1

    dependencies = import_dependencies()
    if dependencies is None:
        return 1
    pd, plt, font_manager = dependencies

    try:
        rows = load_inception_rows(summary_path, pd)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if rows.empty:
        print(
            "[ERROR] No evaluated inception_v3 Mixed block rows were found in results_proof/summary.csv.",
            file=sys.stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_latency_comparison(rows, output_path, plt, font_manager)
    print(f"[OK] Wrote {output_path}")
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


def load_inception_rows(summary_path: Path, pd):
    df = pd.read_csv(summary_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{summary_path} is missing required column(s): {', '.join(missing_columns)}"
        )

    rows = df[
        (df["model"].astype(str) == "inception_v3")
        & (df["status"].astype(str) == "evaluated")
        & df["block_id"].isin(BLOCK_ORDER)
    ].copy()

    if rows.empty:
        return rows

    rows["block_id"] = pd.Categorical(rows["block_id"], categories=BLOCK_ORDER, ordered=True)
    rows = rows.sort_values("block_id")
    rows["existing_latency_ms"] = pd.to_numeric(rows["existing_latency_ms"], errors="coerce")
    rows["global_opt_latency_ms"] = pd.to_numeric(rows["global_opt_latency_ms"], errors="coerce")
    rows = rows.dropna(subset=["existing_latency_ms", "global_opt_latency_ms"])

    missing_blocks = [block for block in BLOCK_ORDER if block not in set(rows["block_id"].astype(str))]
    if missing_blocks:
        print(
            f"[WARN] Missing evaluated rows for: {', '.join(missing_blocks)}",
            file=sys.stderr,
        )

    return rows


def plot_latency_comparison(rows, output_path: Path, plt, font_manager):
    has_korean_font = configure_korean_font(plt, font_manager)
    labels = korean_labels() if has_korean_font else english_labels()
    if not has_korean_font:
        print(
            "[WARN] Korean font was not found. Falling back to English plot labels.",
            file=sys.stderr,
        )

    rows = rows.set_index("block_id").reindex(BLOCK_ORDER).dropna(
        subset=["existing_latency_ms", "global_opt_latency_ms"]
    )
    x_positions = list(range(len(rows)))
    block_labels = [str(block) for block in rows.index]
    bar_width = 0.38
    left_positions = [x - bar_width / 2 for x in x_positions]
    right_positions = [x + bar_width / 2 for x in x_positions]

    fig, axis = plt.subplots(figsize=(18, 8))
    existing_bars = axis.bar(
        left_positions,
        rows["existing_latency_ms"],
        width=bar_width,
        label=labels["existing"],
        color="#4C78A8",
    )
    global_bars = axis.bar(
        right_positions,
        rows["global_opt_latency_ms"],
        width=bar_width,
        label=labels["global"],
        color="#59A14F",
    )

    max_latency = max(
        rows["existing_latency_ms"].max(),
        rows["global_opt_latency_ms"].max(),
    )
    y_offset = max(max_latency * 0.02, 0.25)
    annotate_bar_values(axis, existing_bars, y_offset)
    annotate_bar_values(axis, global_bars, y_offset)
    annotate_reductions(axis, rows, x_positions, y_offset)

    axis.set_title(labels["title"], pad=18)
    axis.set_xlabel(labels["x"])
    axis.set_ylabel(labels["y"])
    axis.set_xticks(x_positions)
    axis.set_xticklabels(block_labels, rotation=45, ha="right")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.legend()
    axis.set_ylim(0, max_latency + y_offset * 8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def annotate_bar_values(axis, bars, y_offset: float):
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.4,
            },
        )


def annotate_reductions(axis, rows, x_positions, y_offset: float):
    for x, (_, row) in zip(x_positions, rows.iterrows()):
        existing = float(row["existing_latency_ms"])
        global_opt = float(row["global_opt_latency_ms"])
        reduction = existing - global_opt
        axis.text(
            x,
            max(existing, global_opt) + y_offset * 4,
            f"-{reduction:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#A63D40",
            bbox={
                "facecolor": "white",
                "edgecolor": "#A63D40",
                "linewidth": 0.4,
                "alpha": 0.9,
                "pad": 1.6,
            },
        )


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
        "y": "최종 처리시간 (ms)",
        "title": "InceptionV3 block별 기존 방법과 전수탐색 전역 최적 기준 처리시간 비교",
    }


def english_labels():
    return {
        "existing": "Existing branch-wise method",
        "global": "Global exhaustive method",
        "x": "InceptionV3 block",
        "y": "Final latency (ms)",
        "title": "InceptionV3 block latency comparison",
    }


if __name__ == "__main__":
    sys.exit(main())
