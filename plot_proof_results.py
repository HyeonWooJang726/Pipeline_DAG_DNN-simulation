from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


MODEL_LABELS = {
    "vgg11": "VGG11",
    "resnet18": "ResNet18",
    "inception_v3": "InceptionV3",
}

INCEPTION_BLOCK_ORDER = [
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

SUMMARY_COLUMNS = [
    "model",
    "block_id",
    "block_type",
    "branches",
    "branch_lengths",
    "per_branch_candidate_counts",
    "combination_count",
    "existing_latency_ms",
    "global_opt_latency_ms",
    "device_busy_ms",
    "tx_busy_ms",
    "server_busy_ms",
    "bottleneck_stage",
    "cost_model",
    "status",
]

SLIDE_COLUMNS = [
    "model_label",
    "block_id",
    "branches",
    "per_branch_candidate_counts",
    "combination_count",
    "existing_latency_ms",
    "global_opt_latency_ms",
    "absolute_reduction_ms",
    "reduction_percent_vs_existing",
    "bottleneck_stage",
    "status",
]

NUMERIC_COLUMNS = [
    "branches",
    "combination_count",
    "existing_latency_ms",
    "global_opt_latency_ms",
    "device_busy_ms",
    "tx_busy_ms",
    "server_busy_ms",
]

METHOD_EXISTING = "기존 방법"
METHOD_GLOBAL = "전수탐색 전역 최적 기준"

MAIN_COMPARISON_OUTPUT = "inception_latency_comparison_ko.png"
AUX_REDUCTION_OUTPUT = "inception_latency_reduction_ms_ko.png"

GENERATED_OUTPUTS = {
    MAIN_COMPARISON_OUTPUT,
    AUX_REDUCTION_OUTPUT,
    "slide_summary_ko.csv",
}


def configure_korean_font() -> None:
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


def load_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    missing = [col for col in SUMMARY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required summary columns: {missing}")

    df = df[SUMMARY_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["absolute_reduction_ms"] = (
        df["existing_latency_ms"] - df["global_opt_latency_ms"]
    )
    df["reduction_percent_vs_existing"] = np.where(
        df["existing_latency_ms"] > 0,
        df["absolute_reduction_ms"] / df["existing_latency_ms"] * 100.0,
        0.0,
    )
    return df


def evaluated_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "evaluated"].copy()


def inception_rows(df: pd.DataFrame, positive_only: bool = False) -> pd.DataFrame:
    out = evaluated_rows(df)
    out = out[out["model"] == "inception_v3"].copy()
    if positive_only:
        out = out[out["absolute_reduction_ms"] > 0].copy()

    out["block_order"] = pd.Categorical(
        out["block_id"],
        categories=INCEPTION_BLOCK_ORDER,
        ordered=True,
    )
    return out.sort_values("block_order")


def add_bar_value_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_main_inception_latency_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = inception_rows(df, positive_only=True)
    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, 1.05 * max(1, len(plot_df))), 5.8))
    existing_bars = ax.bar(
        x - width / 2,
        plot_df["existing_latency_ms"],
        width=width,
        label=METHOD_EXISTING,
        color="#D95763",
    )
    global_bars = ax.bar(
        x + width / 2,
        plot_df["global_opt_latency_ms"],
        width=width,
        label=METHOD_GLOBAL,
        color="#3267D6",
    )
    add_bar_value_labels(ax, existing_bars)
    add_bar_value_labels(ax, global_bars)

    max_latency = float(
        plot_df[["existing_latency_ms", "global_opt_latency_ms"]].max().max()
    )
    top_padding = max(max_latency * 0.18, 1.0)
    reduction_offset = max(max_latency * 0.09, 0.5)
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        block_max = max(row.existing_latency_ms, row.global_opt_latency_ms)
        ax.text(
            idx,
            block_max + reduction_offset,
            f"-{row.absolute_reduction_ms:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
            fontweight="bold",
        )

    ax.set_ylim(0, max_latency + top_padding)
    ax.set_title("InceptionV3 block별 기존 방법과 전수탐색 전역 최적 기준 처리시간 비교")
    ax.set_xlabel("InceptionV3 block")
    ax.set_ylabel("최종 처리시간 (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["block_id"], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / MAIN_COMPARISON_OUTPUT, dpi=300)
    plt.close(fig)


def save_inception_latency_reduction_ms(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = inception_rows(df, positive_only=True)

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * max(1, len(plot_df))), 5))
    bars = ax.bar(plot_df["block_id"], plot_df["absolute_reduction_ms"], color="#3267D6")
    add_bar_value_labels(ax, bars)
    ax.set_title("InceptionV3 block별 감소량 (ms)")
    ax.set_xlabel("InceptionV3 block")
    ax.set_ylabel("감소량 (ms)")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_dir / AUX_REDUCTION_OUTPUT, dpi=300)
    plt.close(fig)


def save_slide_summary(df: pd.DataFrame, out_dir: Path) -> None:
    slide_df = df[SLIDE_COLUMNS].copy()
    slide_df = slide_df.sort_values(
        "absolute_reduction_ms",
        ascending=False,
        na_position="last",
    )
    slide_df.to_csv(out_dir / "slide_summary_ko.csv", index=False, encoding="utf-8-sig")


def remove_unlisted_png_outputs(out_dir: Path) -> None:
    for path in out_dir.glob("*.png"):
        if path.name not in GENERATED_OUTPUTS:
            path.unlink()


def main() -> None:
    configure_korean_font()

    repo_root = Path(__file__).resolve().parent
    summary_path = (
        repo_root
        / "experiments"
        / "decc_global_gap"
        / "results_proof"
        / "summary.csv"
    )
    out_dir = (
        repo_root
        / "experiments"
        / "decc_global_gap"
        / "results_proof"
        / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    remove_unlisted_png_outputs(out_dir)

    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find summary file: {summary_path}")

    df = load_summary(summary_path)
    save_main_inception_latency_comparison(df, out_dir)
    save_inception_latency_reduction_ms(df, out_dir)
    save_slide_summary(df, out_dir)

    print(f"[DONE] Figures and CSV saved to: {out_dir}")


if __name__ == "__main__":
    main()
