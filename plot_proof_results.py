from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


MODEL_LABELS = {
    "vgg11": "VGG11",
    "resnet18": "ResNet18",
    "inception_v3": "InceptionV3",
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

    numeric_cols = [
        "branches",
        "candidates_total",
        "combination_count",
        "decc_style_latency_ms",
        "global_latency_ms",
        "optimality_gap_percent",
        "relative_slowdown_vs_global_percent",
        "device_busy_ms",
        "tx_busy_ms",
        "server_busy_ms",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["status"] == "evaluated"].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["absolute_gap_ms"] = df["decc_style_latency_ms"] - df["global_latency_ms"]

    return df


def save_model_level_max_gap(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        df.groupby("model_label", sort=False, as_index=False)["optimality_gap_percent"]
        .max()
        .copy()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["model_label"], plot_df["optimality_gap_percent"], color="#2F6BFF")
    plt.title("DNN 모델별 최대 전역 최적 대비 손실률")
    plt.xlabel("DNN 모델")
    plt.ylabel("전역 최적 대비 손실률 (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "model_level_max_gap_ko.png", dpi=300)
    plt.close()


def save_inception_block_gap(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df[df["model"] == "inception_v3"].copy()

    if plot_df.empty:
        print("[WARN] No InceptionV3 rows found.")
        return

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df["block_id"], plot_df["optimality_gap_percent"], color="#2F6BFF")
    plt.title("InceptionV3 DAG block별 전역 최적 대비 손실률")
    plt.xlabel("InceptionV3 block")
    plt.ylabel("전역 최적 대비 손실률 (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "inception_block_gap_ko.png", dpi=300)
    plt.close()


def save_inception_latency_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df[
        (df["model"] == "inception_v3") & (df["optimality_gap_percent"] > 0)
    ].copy()

    if plot_df.empty:
        print("[WARN] No positive InceptionV3 optimality gap rows found.")
        return

    x = range(len(plot_df))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(
        [i - width / 2 for i in x],
        plot_df["decc_style_latency_ms"],
        width=width,
        label="기존 방법",
        color="#E85D75",
    )
    plt.bar(
        [i + width / 2 for i in x],
        plot_df["global_latency_ms"],
        width=width,
        label="전역 최적 기준",
        color="#2F6BFF",
    )

    plt.title("InceptionV3 block별 기존 방법과 전역 최적 기준의 지연 비교")
    plt.xlabel("InceptionV3 block")
    plt.ylabel("최종 지연 시간 (ms)")
    plt.xticks(list(x), plot_df["block_id"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "inception_latency_comparison_ko.png", dpi=300)
    plt.close()


def save_model_level_positive_block_count(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = (
        df.assign(positive_gap=df["optimality_gap_percent"] > 0)
        .groupby("model_label", sort=False, as_index=False)["positive_gap"]
        .sum()
        .rename(columns={"positive_gap": "positive_block_count"})
    )

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["model_label"], plot_df["positive_block_count"], color="#2F6BFF")
    plt.title("DNN 모델별 전역 최적과 차이가 발생한 block 수")
    plt.xlabel("DNN 모델")
    plt.ylabel("block 수")
    plt.tight_layout()
    plt.savefig(out_dir / "model_level_positive_block_count_ko.png", dpi=300)
    plt.close()


def save_slide_summary(df: pd.DataFrame, out_dir: Path) -> None:
    slide_df = pd.DataFrame(
        {
            "model_label": df["model_label"],
            "block_id": df["block_id"],
            "block_type": df["block_type"],
            "branches": df["branches"],
            "combination_count": df["combination_count"],
            "existing_latency_ms": df["decc_style_latency_ms"],
            "global_opt_latency_ms": df["global_latency_ms"],
            "absolute_gap_ms": df["absolute_gap_ms"],
            "optimality_gap_percent": df["optimality_gap_percent"],
            "bottleneck_stage": df["bottleneck_stage"],
        }
    )
    slide_df.to_csv(out_dir / "slide_summary_ko.csv", index=False, encoding="utf-8-sig")


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

    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find summary file: {summary_path}")

    df = load_summary(summary_path)

    save_model_level_max_gap(df, out_dir)
    save_inception_block_gap(df, out_dir)
    save_inception_latency_comparison(df, out_dir)
    save_model_level_positive_block_count(df, out_dir)
    save_slide_summary(df, out_dir)

    print(f"[DONE] Korean proof figures and CSV saved to: {out_dir}")


if __name__ == "__main__":
    main()
