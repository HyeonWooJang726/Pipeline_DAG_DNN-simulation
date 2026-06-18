from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_gap(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv)
    plt.figure()
    plt.bar(df["model"], df["optimality_gap_percent"])
    plt.ylabel("Optimality gap (%)")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


if __name__ == "__main__":
    results = Path("results")
    plot_gap(str(results / "summary.csv"), str(results / "optimality_gap.png"))
