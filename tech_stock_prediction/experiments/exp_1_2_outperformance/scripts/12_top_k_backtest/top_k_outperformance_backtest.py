"""
Top-K backtest for the outperformance Random Forest.

The best K is selected only on validation results. The test set is evaluated
afterwards and is not used for K selection.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import yaml
from formatting import format_percent, save_csv


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

TOP_K_VALUES = [1, 2, 3, 5, 10]
BENCHMARK_K = 10


def require_columns(data, file_path):
    required_columns = {"Date", "Ticker", "Probability", "Future_Return"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{file_path} is missing required columns: {missing_columns}. "
            "Run scripts/06_model_testing/evaluate_random_forest_outperformance.py first."
        )


def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdowns = cumulative_returns / running_max - 1
    return drawdowns.min()


def add_top_k_predictions(data, k):
    ranked_data = data.copy()
    ranked_data["Probability_Rank"] = (
        ranked_data
        .groupby("Date")["Probability"]
        .rank(method="first", ascending=False)
    )
    ranked_data["Top_K_Prediction"] = (ranked_data["Probability_Rank"] <= k).astype(int)
    return ranked_data


def backtest_top_k(data, k):
    backtest_data = add_top_k_predictions(data, k)
    backtest_data["K"] = k
    backtest_data["Strategy_Return"] = (
        backtest_data["Top_K_Prediction"] * backtest_data["Future_Return"]
    )
    backtest_data["Buy_Hold_Return"] = backtest_data["Future_Return"]

    daily_returns = (
        backtest_data
        .groupby("Date")[["Strategy_Return", "Buy_Hold_Return"]]
        .mean()
        .reset_index()
    )
    daily_returns["K"] = k
    daily_returns["Strategy_Cumulative"] = (
        1 + daily_returns["Strategy_Return"]
    ).cumprod()
    daily_returns["Buy_Hold_Cumulative"] = (
        1 + daily_returns["Buy_Hold_Return"]
    ).cumprod()

    strategy_total_return = daily_returns["Strategy_Cumulative"].iloc[-1] - 1
    buy_hold_total_return = daily_returns["Buy_Hold_Cumulative"].iloc[-1] - 1
    difference = strategy_total_return - buy_hold_total_return

    total_predictions = len(backtest_data)
    buy_signals = int(backtest_data["Top_K_Prediction"].sum())
    signal_rate = buy_signals / total_predictions if total_predictions else 0.0

    bought_trades = backtest_data[backtest_data["Top_K_Prediction"] == 1].copy()
    if len(bought_trades) > 0:
        average_trade_return = bought_trades["Future_Return"].mean()
        win_rate = (bought_trades["Future_Return"] > 0).mean()
    else:
        average_trade_return = 0.0
        win_rate = 0.0

    average_daily_return = daily_returns["Strategy_Return"].mean()
    volatility = daily_returns["Strategy_Return"].std()
    if pd.notna(volatility) and volatility > 0:
        sharpe_ratio = average_daily_return / volatility * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    result = {
        "K": k,
        "Total_Predictions": total_predictions,
        "Buy_Signals": buy_signals,
        "Signal_Rate": signal_rate,
        "Average_Trade_Return": average_trade_return,
        "Win_Rate": win_rate,
        "Strategy_Total_Return": strategy_total_return,
        "Buy_Hold_Total_Return": buy_hold_total_return,
        "Difference": difference,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": calculate_max_drawdown(daily_returns["Strategy_Cumulative"]),
        "Average_Daily_Return": average_daily_return,
        "Volatility": volatility,
        "Number_Of_Days": len(daily_returns),
    }

    return result, daily_returns


def evaluate_k_values(data):
    results = []
    daily_return_parts = []

    for k in TOP_K_VALUES:
        result, daily_returns = backtest_top_k(data, k)
        results.append(result)
        daily_return_parts.append(daily_returns)

    return pd.DataFrame(results), pd.concat(daily_return_parts, ignore_index=True)


def main():
    print("Outperformance Top-K Backtest")
    print("=============================")

    validation_predictions_file = (
        EXPERIMENT_ROOT / PARAMS["PREDICTIONS"]["VALIDATION_FILE"]
    )
    test_predictions_file = EXPERIMENT_ROOT / PARAMS["PREDICTIONS"]["TEST_FILE"]
    validation_results_file = (
        EXPERIMENT_ROOT / PARAMS["BACKTEST"]["TOP_K_VALIDATION_RESULTS_FILE"]
    )
    test_results_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["TOP_K_RESULTS_FILE"]
    daily_returns_file = (
        EXPERIMENT_ROOT / PARAMS["BACKTEST"]["TOP_K_DAILY_RETURNS_FILE"]
    )

    if not validation_predictions_file.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: {validation_predictions_file}"
        )
    if not test_predictions_file.exists():
        raise FileNotFoundError(f"Test predictions not found: {test_predictions_file}")

    validation_data = pd.read_csv(validation_predictions_file, parse_dates=["Date"])
    test_data = pd.read_csv(test_predictions_file, parse_dates=["Date"])
    require_columns(validation_data, validation_predictions_file)
    require_columns(test_data, test_predictions_file)

    validation_data = validation_data.dropna(subset=["Future_Return"]).copy()
    test_data = test_data.dropna(subset=["Future_Return"]).copy()
    validation_data = validation_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    test_data = test_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    validation_results, _ = evaluate_k_values(validation_data)
    test_results, test_daily_returns = evaluate_k_values(test_data)

    validation_results_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(validation_results, validation_results_file)
    save_csv(test_results, test_results_file)
    save_csv(test_daily_returns, daily_returns_file)

    selectable_validation_results = validation_results[
        validation_results["K"] != BENCHMARK_K
    ].copy()
    best_validation = (
        selectable_validation_results
        .sort_values(["Difference", "Strategy_Total_Return"], ascending=False)
        .iloc[0]
    )
    final_test = test_results[test_results["K"] == best_validation["K"]].iloc[0]
    benchmark_test = test_results[test_results["K"] == BENCHMARK_K].iloc[0]

    print("\nBest K on validation:")
    print(f"K:                {best_validation['K']:.0f}")
    print(f"Signal rate:      {format_percent(best_validation['Signal_Rate'])}")
    print(f"Strategy return:  {format_percent(best_validation['Strategy_Total_Return'])}")
    print(f"Buy-hold return:  {format_percent(best_validation['Buy_Hold_Total_Return'])}")
    print(f"Difference:       {format_percent(best_validation['Difference'])}")

    print("\nFinal test performance for selected K:")
    print(f"K:                {final_test['K']:.0f}")
    print(f"Signal rate:      {format_percent(final_test['Signal_Rate'])}")
    print(f"Strategy return:  {format_percent(final_test['Strategy_Total_Return'])}")
    print(f"Buy-hold return:  {format_percent(final_test['Buy_Hold_Total_Return'])}")
    print(f"Difference:       {format_percent(final_test['Difference'])}")

    print(f"\nK={BENCHMARK_K} benchmark on test:")
    print(f"Strategy return:  {format_percent(benchmark_test['Strategy_Total_Return'])}")
    print(f"Buy-hold return:  {format_percent(benchmark_test['Buy_Hold_Total_Return'])}")
    print(f"Difference:       {format_percent(benchmark_test['Difference'])}")

    if final_test["Difference"] > 0:
        print("\nResult: selected Top-K strategy beats Buy-and-Hold on test.")
    else:
        print("\nResult: selected Top-K strategy does not beat Buy-and-Hold on test.")

    print(f"\nValidation results saved to: {validation_results_file}")
    print(f"Test results saved to: {test_results_file}")
    print(f"Daily test returns saved to: {daily_returns_file}")


if __name__ == "__main__":
    main()
