"""
Create a fair Top-K comparison plot for the presentation.

Standard-LSTM and Outperformance-LSTM are evaluated on the same overlapping
test dates. Buy-and-Hold is calculated once from the same stock returns, so it
is identical for both models.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]

STANDARD_PREDICTIONS_FILE = (
    EXPERIMENT_ROOT / "data" / "processed" / "lstm_test_predictions.csv"
)
OUTPERFORMANCE_PREDICTIONS_FILE = (
    EXPERIMENT_ROOT / "data" / "processed" / "lstm_outperformance_predictions.csv"
)
TEST_FILE = PROJECT_ROOT / "experiments" / "exp_1_1" / "data" / "processed" / "test.csv"

OUTPUT_CSV = (
    EXPERIMENT_ROOT / "data" / "processed" / "lstm_common_period_top_k_results.csv"
)
OUTPUT_PLOT = (
    EXPERIMENT_ROOT / "plots" / "13_common_period_top_k_comparison.png"
)

TOP_K_VALUES = [1, 2, 3, 4, 5]


def load_shared_returns():
    """
    Calculate next-day stock returns from the same test.csv for both models.
    """
    test_data = pd.read_csv(TEST_FILE, parse_dates=["Date"])
    test_data = test_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    test_data["Next_Close"] = test_data.groupby("Ticker")["Close"].shift(-1)
    test_data["Future_Return"] = test_data["Next_Close"] / test_data["Close"] - 1
    test_data = test_data.dropna(subset=["Future_Return"]).copy()

    return test_data[["Date", "Ticker", "Future_Return"]]


def load_predictions(file_path, model_name):
    predictions = pd.read_csv(file_path, parse_dates=["Date"])

    required_columns = ["Date", "Ticker", "Probability"]
    missing_columns = [column for column in required_columns if column not in predictions.columns]

    if missing_columns:
        raise ValueError(f"{file_path} is missing columns: {missing_columns}")

    predictions = predictions[required_columns].copy()
    predictions["model_name"] = model_name

    return predictions.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def filter_common_period(standard, outperformance, returns):
    """
    Keep only dates that exist in both prediction files and in the return table.
    """
    standard_dates = set(standard["Date"])
    outperformance_dates = set(outperformance["Date"])
    return_dates = set(returns["Date"])
    common_dates = sorted(standard_dates & outperformance_dates & return_dates)

    if not common_dates:
        raise ValueError("No overlapping dates found.")

    standard = standard[standard["Date"].isin(common_dates)].copy()
    outperformance = outperformance[outperformance["Date"].isin(common_dates)].copy()
    returns = returns[returns["Date"].isin(common_dates)].copy()

    return standard, outperformance, returns


def calculate_top_k_results(predictions, returns, model_label):
    """
    Rank each day's probabilities and calculate Top-K returns.
    """
    data = predictions.merge(returns, on=["Date", "Ticker"], how="inner")

    rows = []
    daily_buy_hold = data.groupby("Date")["Future_Return"].mean()
    buy_and_hold_return = (1 + daily_buy_hold).prod() - 1

    for top_k in TOP_K_VALUES:
        ranked = data.copy()
        ranked["Rank"] = ranked.groupby("Date")["Probability"].rank(
            method="first",
            ascending=False,
        )
        selected = ranked[ranked["Rank"] <= top_k].copy()
        daily_strategy = selected.groupby("Date")["Future_Return"].mean()
        strategy_return = (1 + daily_strategy).prod() - 1

        rows.append(
            {
                "model": model_label,
                "top_k": top_k,
                "test_start": daily_buy_hold.index.min().date().isoformat(),
                "test_end": daily_buy_hold.index.max().date().isoformat(),
                "number_of_trading_days": len(daily_buy_hold),
                "strategy_return": strategy_return,
                "buy_and_hold_return": buy_and_hold_return,
                "difference": strategy_return - buy_and_hold_return,
            }
        )

    return rows


def create_plot(results):
    standard = results[results["model"] == "Standard-LSTM"].copy()
    outperformance = results[results["model"] == "Outperformance-LSTM"].copy()

    x_positions = np.arange(len(TOP_K_VALUES))
    width = 0.36

    standard_values = standard["difference"].to_numpy() * 100
    outperformance_values = outperformance["difference"].to_numpy() * 100
    all_values = list(standard_values) + list(outperformance_values)
    best_value = max(all_values)

    fig, axis = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    standard_bars = axis.bar(
        x_positions - width / 2,
        standard_values,
        width,
        label="Standard-LSTM Top-K",
        color="#2563eb",
    )
    outperformance_bars = axis.bar(
        x_positions + width / 2,
        outperformance_values,
        width,
        label="Outperformance-LSTM Top-K",
        color="#7c3aed",
    )

    for bars in [standard_bars, outperformance_bars]:
        for bar in bars:
            value = bar.get_height()

            if value < 0:
                bar.set_facecolor("#fecaca")
                bar.set_hatch("//")
                bar.set_edgecolor("#991b1b")
                bar.set_linewidth(2)

            if value == best_value:
                bar.set_edgecolor("#f59e0b")
                bar.set_linewidth(4)

            offset = 1.5 if value >= 0 else -2.0
            vertical_alignment = "bottom" if value >= 0 else "top"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:+.1f} pp",
                ha="center",
                va=vertical_alignment,
                fontsize=13,
                fontweight="bold",
            )

    axis.axhline(0, color="#111827", linewidth=1.6)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([f"Top {top_k}" for top_k in TOP_K_VALUES], fontsize=14, fontweight="bold")
    axis.set_ylabel("Difference vs. Buy-and-Hold (percentage points)", fontsize=14)
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(axis="y", alpha=0.2)
    legend_handles = [
        mpatches.Patch(color="#2563eb", label="Standard-LSTM Top-K"),
        mpatches.Patch(color="#7c3aed", label="Outperformance-LSTM Top-K"),
        mpatches.Patch(
            facecolor="#fecaca",
            edgecolor="#991b1b",
            hatch="//",
            label="Underperformed Buy-and-Hold",
        ),
    ]
    axis.legend(handles=legend_handles, loc="upper left", fontsize=13)

    buy_hold = results["buy_and_hold_return"].iloc[0] * 100
    test_start = results["test_start"].iloc[0]
    test_end = results["test_end"].iloc[0]

    fig.suptitle(
        "Fair Top-K Comparison on the Same Test Period",
        fontsize=23,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        f"Common period: {test_start} to {test_end} | Same Buy-and-Hold: {buy_hold:.1f}%",
        ha="center",
        fontsize=14,
        color="#374151",
    )
    fig.text(
        0.5,
        0.035,
        "Beide Top-K-Strategien werden auf denselben Handelstagen verglichen. "
        "Dadurch ist Buy-and-Hold fuer beide Modelle identisch.",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )

    axis.set_ylim(min(all_values) - 8, max(all_values) + 12)
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.9])

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    returns = load_shared_returns()
    standard = load_predictions(STANDARD_PREDICTIONS_FILE, "standard_lstm")
    outperformance = load_predictions(
        OUTPERFORMANCE_PREDICTIONS_FILE,
        "outperformance_lstm",
    )

    standard, outperformance, returns = filter_common_period(
        standard,
        outperformance,
        returns,
    )

    rows = []
    rows.extend(calculate_top_k_results(standard, returns, "Standard-LSTM"))
    rows.extend(calculate_top_k_results(outperformance, returns, "Outperformance-LSTM"))

    results = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    create_plot(results)

    print(f"Common-period Top-K results saved to: {OUTPUT_CSV}")
    print(f"Presentation plot saved to: {OUTPUT_PLOT}")
    print(
        results[
            [
                "model",
                "top_k",
                "test_start",
                "test_end",
                "strategy_return",
                "buy_and_hold_return",
                "difference",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
