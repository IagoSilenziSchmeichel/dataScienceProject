"""
Create a presentation plot for the Outperformance-LSTM Top-K backtest.

The plot uses the Outperformance-LSTM Top-K strategy returns and the same
Buy-and-Hold reference line as the normal Standard-LSTM Top-K backtest.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]

OUTPERFORMANCE_TOP_K_FILE = (
    EXPERIMENT_ROOT / "data" / "processed" / "lstm_outperformance_top_k_results.csv"
)
STANDARD_TOP_K_FILE = (
    EXPERIMENT_ROOT / "data" / "processed" / "lstm_standard_top_k_results.csv"
)
OUTPUT_FILE = (
    EXPERIMENT_ROOT / "plots" / "10_outperformance_top_k_return_corrected_bh.png"
)
OUTPUT_CSV = (
    EXPERIMENT_ROOT
    / "data"
    / "processed"
    / "lstm_outperformance_top_k_corrected_bh_results.csv"
)


def main():
    outperformance = pd.read_csv(OUTPERFORMANCE_TOP_K_FILE)
    standard = pd.read_csv(STANDARD_TOP_K_FILE)

    corrected_buy_and_hold = standard["buy_and_hold_return"].iloc[0]

    plot_data = outperformance.copy()
    plot_data["corrected_buy_and_hold_return"] = corrected_buy_and_hold
    plot_data["corrected_difference"] = (
        plot_data["strategy_return"] - corrected_buy_and_hold
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(OUTPUT_CSV, index=False)

    fig, axis = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("white")

    colors = []
    best_index = plot_data["strategy_return"].idxmax()

    for index, row in plot_data.iterrows():
        if index == best_index:
            colors.append("#7c3aed")
        elif row["strategy_return"] >= corrected_buy_and_hold:
            colors.append("#16a34a")
        else:
            colors.append("#dc2626")

    bars = axis.bar(
        [f"Top {int(top_k)}" for top_k in plot_data["top_k"]],
        plot_data["strategy_return"] * 100,
        color=colors,
        width=0.62,
    )

    axis.axhline(
        corrected_buy_and_hold * 100,
        color="#111827",
        linewidth=2.2,
        linestyle="--",
        label=f"Buy-and-Hold: {corrected_buy_and_hold * 100:.1f}%",
    )

    for index, bar in enumerate(bars):
        strategy_return = plot_data.iloc[index]["strategy_return"] * 100
        corrected_difference = plot_data.iloc[index]["corrected_difference"] * 100

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            strategy_return + 1.7,
            f"{strategy_return:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(strategy_return - 7, 4),
            f"{corrected_difference:+.1f} pp",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white" if colors[index] != "#16a34a" else "#111827",
        )

    best_bar = bars[best_index]
    best_bar.set_edgecolor("#f59e0b")
    best_bar.set_linewidth(4)
    axis.text(
        best_bar.get_x() + best_bar.get_width() / 2,
        plot_data.loc[best_index, "strategy_return"] * 100 + 6,
        "Best",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#92400e",
    )

    axis.set_title(
        "Outperformance-LSTM Top-K Backtest",
        fontsize=24,
        fontweight="bold",
        pad=26,
    )
    axis.text(
        0.5,
        1.02,
        "Strategy Return with corrected Buy-and-Hold reference from the normal Top-K backtest",
        transform=axis.transAxes,
        ha="center",
        fontsize=14,
        color="#374151",
    )

    axis.set_ylabel("Total Return (%)", fontsize=15)
    axis.tick_params(axis="x", labelsize=15)
    axis.tick_params(axis="y", labelsize=13)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper left", fontsize=13)
    axis.set_ylim(0, max(plot_data["strategy_return"].max() * 100 + 13, 90))

    fig.text(
        0.5,
        0.035,
        "Die gestrichelte Linie nutzt denselben Buy-and-Hold-Wert wie der normale Top-K-Backtest.",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )

    plt.tight_layout(rect=[0.04, 0.08, 1, 0.92])
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Corrected results saved to: {OUTPUT_CSV}")
    print(f"Presentation plot saved to: {OUTPUT_FILE}")
    print(
        plot_data[
            [
                "top_k",
                "strategy_return",
                "corrected_buy_and_hold_return",
                "corrected_difference",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
