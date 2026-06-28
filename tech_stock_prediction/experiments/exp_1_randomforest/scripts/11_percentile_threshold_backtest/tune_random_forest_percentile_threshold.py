"""
Tune Random Forest percentile thresholds on validation data.

The best mode and percentile are selected only on validation results. The test
set is evaluated afterwards and is not used for threshold selection.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import yaml
from formatting import format_percent, save_csv
from utils.backtest_metrics import calculate_backtest_metrics


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

VALIDATION_PREDICTIONS_FILE = "data/processed/validation_predictions.csv"
TEST_RESULTS_FILE = "data/processed/random_forest_percentile_threshold_results.csv"
VALIDATION_RESULTS_FILE = (
    "data/processed/random_forest_percentile_threshold_validation_results.csv"
)
DAILY_RETURNS_FILE = (
    "data/processed/random_forest_percentile_threshold_daily_returns.csv"
)

PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
MODES = ["global", "per_ticker"]


def require_columns(data, file_path):
    required_columns = {"Date", "Ticker", "Probability", "Future_Return"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{file_path} is missing required columns: {missing_columns}. "
            "Run scripts/06_model_testing/evaluate_random_forest.py first."
        )


def build_cutoffs(validation_data, mode, percentile):
    if mode == "global":
        cutoff = validation_data["Probability"].quantile(percentile / 100)
        return cutoff, None

    if mode == "per_ticker":
        cutoffs = (
            validation_data
            .groupby("Ticker")["Probability"]
            .quantile(percentile / 100)
        )
        return cutoffs.mean(), cutoffs

    raise ValueError(f"Unknown mode: {mode}")


def apply_cutoff(data, mode, cutoff, ticker_cutoffs):
    if mode == "global":
        return (data["Probability"] >= cutoff).astype(int)

    mapped_cutoffs = data["Ticker"].map(ticker_cutoffs)
    if mapped_cutoffs.isna().any():
        missing_tickers = sorted(data.loc[mapped_cutoffs.isna(), "Ticker"].unique())
        raise ValueError(f"Missing ticker cutoffs for: {missing_tickers}")

    return (data["Probability"] >= mapped_cutoffs).astype(int)


def backtest_threshold(data, mode, percentile, cutoff, ticker_cutoffs):
    threshold_data = data.copy()
    threshold_data["Mode"] = mode
    threshold_data["Percentile"] = percentile
    threshold_data["Threshold_Prediction"] = apply_cutoff(
        threshold_data,
        mode,
        cutoff,
        ticker_cutoffs,
    )
    threshold_data["Strategy_Return"] = (
        threshold_data["Threshold_Prediction"] * threshold_data["Future_Return"]
    )
    threshold_data["Buy_Hold_Return"] = threshold_data["Future_Return"]

    daily_returns = (
        threshold_data
        .groupby("Date")[["Strategy_Return", "Buy_Hold_Return"]]
        .mean()
        .reset_index()
    )
    daily_returns["Mode"] = mode
    daily_returns["Percentile"] = percentile
    daily_returns["Strategy_Cumulative"] = (
        1 + daily_returns["Strategy_Return"]
    ).cumprod()
    daily_returns["Buy_Hold_Cumulative"] = (
        1 + daily_returns["Buy_Hold_Return"]
    ).cumprod()

    strategy_metrics = calculate_backtest_metrics(daily_returns, "Strategy_Return")
    buy_hold_metrics = calculate_backtest_metrics(daily_returns, "Buy_Hold_Return")

    strategy_total_return = strategy_metrics["Total_Return"]
    buy_hold_total_return = buy_hold_metrics["Total_Return"]
    difference = strategy_total_return - buy_hold_total_return

    total_predictions = len(threshold_data)
    buy_signals = int(threshold_data["Threshold_Prediction"].sum())
    signal_rate = buy_signals / total_predictions if total_predictions else 0.0

    bought_trades = threshold_data[
        threshold_data["Threshold_Prediction"] == 1
    ].copy()
    if len(bought_trades) > 0:
        average_trade_return = bought_trades["Future_Return"].mean()
        win_rate = (bought_trades["Future_Return"] > 0).mean()
    else:
        average_trade_return = 0.0
        win_rate = 0.0

    average_daily_return = daily_returns["Strategy_Return"].mean()

    result = {
        "Mode": mode,
        "Percentile": percentile,
        "Cutoff": cutoff if mode == "global" else np.nan,
        "Average_Cutoff": cutoff,
        "Signal_Rate": signal_rate,
        "Total_Predictions": total_predictions,
        "Buy_Signals": buy_signals,
        "Average_Trade_Return": average_trade_return,
        "Win_Rate": win_rate,
        "Strategy_Total_Return": strategy_total_return,
        "Buy_Hold_Total_Return": buy_hold_total_return,
        "Difference": difference,
        "Strategy_CAGR": strategy_metrics["CAGR"],
        "Buy_Hold_CAGR": buy_hold_metrics["CAGR"],
        "Strategy_Sharpe": strategy_metrics["Sharpe"],
        "Buy_Hold_Sharpe": buy_hold_metrics["Sharpe"],
        "Strategy_Max_Drawdown": strategy_metrics["Max_Drawdown"],
        "Buy_Hold_Max_Drawdown": buy_hold_metrics["Max_Drawdown"],
        "Strategy_Volatility": strategy_metrics["Volatility"],
        "Buy_Hold_Volatility": buy_hold_metrics["Volatility"],
        "Sharpe_Ratio": strategy_metrics["Sharpe"],
        "Max_Drawdown": strategy_metrics["Max_Drawdown"],
        "Average_Daily_Return": average_daily_return,
        "Volatility": strategy_metrics["Volatility"],
        "Number_Of_Days": len(daily_returns),
    }

    return result, daily_returns


def evaluate_percentiles(data, validation_data):
    results = []
    daily_return_parts = []

    for mode in MODES:
        for percentile in PERCENTILES:
            cutoff, ticker_cutoffs = build_cutoffs(validation_data, mode, percentile)
            result, daily_returns = backtest_threshold(
                data,
                mode,
                percentile,
                cutoff,
                ticker_cutoffs,
            )
            results.append(result)
            daily_return_parts.append(daily_returns)

    return pd.DataFrame(results), pd.concat(daily_return_parts, ignore_index=True)


def main():
    print("Random Forest Percentile Threshold Backtest")
    print("===========================================")

    validation_predictions_file = EXPERIMENT_ROOT / VALIDATION_PREDICTIONS_FILE
    test_predictions_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["PREDICTIONS_FILE"]
    validation_results_file = EXPERIMENT_ROOT / VALIDATION_RESULTS_FILE
    test_results_file = EXPERIMENT_ROOT / TEST_RESULTS_FILE
    daily_returns_file = EXPERIMENT_ROOT / DAILY_RETURNS_FILE

    if not validation_predictions_file.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: {validation_predictions_file}\n"
            "Run scripts/06_model_testing/evaluate_random_forest.py first."
        )
    if not test_predictions_file.exists():
        raise FileNotFoundError(
            f"Test predictions not found: {test_predictions_file}\n"
            "Run scripts/06_model_testing/evaluate_random_forest.py first."
        )

    validation_data = pd.read_csv(validation_predictions_file, parse_dates=["Date"])
    test_data = pd.read_csv(test_predictions_file, parse_dates=["Date"])
    require_columns(validation_data, validation_predictions_file)
    require_columns(test_data, test_predictions_file)

    validation_data = validation_data.dropna(subset=["Future_Return"]).copy()
    test_data = test_data.dropna(subset=["Future_Return"]).copy()
    validation_data = validation_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    test_data = test_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    validation_results, _ = evaluate_percentiles(validation_data, validation_data)
    test_results, test_daily_returns = evaluate_percentiles(test_data, validation_data)

    validation_results_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(validation_results, validation_results_file)
    save_csv(test_results, test_results_file)
    save_csv(test_daily_returns, daily_returns_file)

    best_validation = (
        validation_results
        .sort_values(["Difference", "Strategy_Total_Return"], ascending=False)
        .iloc[0]
    )
    final_test = test_results[
        (test_results["Mode"] == best_validation["Mode"])
        & (test_results["Percentile"] == best_validation["Percentile"])
    ].iloc[0]

    print("\nBest validation combination:")
    print(f"Mode:             {best_validation['Mode']}")
    print(f"Percentile:       {best_validation['Percentile']:.0f}")
    print(f"Signal rate:      {format_percent(best_validation['Signal_Rate'])}")
    print(f"Strategy return:  {format_percent(best_validation['Strategy_Total_Return'])}")
    print(f"Buy-hold return:  {format_percent(best_validation['Buy_Hold_Total_Return'])}")
    print(f"Difference:       {format_percent(best_validation['Difference'])}")
    print(f"Strategy CAGR:    {format_percent(best_validation['Strategy_CAGR'])}")
    print(f"Buy-hold CAGR:    {format_percent(best_validation['Buy_Hold_CAGR'])}")
    print(f"Strategy Sharpe:  {best_validation['Strategy_Sharpe']:.5f}")
    print(f"Buy-hold Sharpe:  {best_validation['Buy_Hold_Sharpe']:.5f}")
    print(f"Strategy Max Drawdown: {format_percent(best_validation['Strategy_Max_Drawdown'])}")
    print(f"Buy-hold Max Drawdown: {format_percent(best_validation['Buy_Hold_Max_Drawdown'])}")
    print(f"Strategy Volatility: {format_percent(best_validation['Strategy_Volatility'])}")
    print(f"Buy-hold Volatility: {format_percent(best_validation['Buy_Hold_Volatility'])}")

    print("\nFinal test performance for selected combination:")
    print(f"Mode:             {final_test['Mode']}")
    print(f"Percentile:       {final_test['Percentile']:.0f}")
    print(f"Signal rate:      {format_percent(final_test['Signal_Rate'])}")
    print(f"Strategy return:  {format_percent(final_test['Strategy_Total_Return'])}")
    print(f"Buy-hold return:  {format_percent(final_test['Buy_Hold_Total_Return'])}")
    print(f"Difference:       {format_percent(final_test['Difference'])}")
    print(f"Strategy CAGR:    {format_percent(final_test['Strategy_CAGR'])}")
    print(f"Buy-hold CAGR:    {format_percent(final_test['Buy_Hold_CAGR'])}")
    print(f"Strategy Sharpe:  {final_test['Strategy_Sharpe']:.5f}")
    print(f"Buy-hold Sharpe:  {final_test['Buy_Hold_Sharpe']:.5f}")
    print(f"Strategy Max Drawdown: {format_percent(final_test['Strategy_Max_Drawdown'])}")
    print(f"Buy-hold Max Drawdown: {format_percent(final_test['Buy_Hold_Max_Drawdown'])}")
    print(f"Strategy Volatility: {format_percent(final_test['Strategy_Volatility'])}")
    print(f"Buy-hold Volatility: {format_percent(final_test['Buy_Hold_Volatility'])}")

    if final_test["Difference"] > 0:
        print("\nResult: selected strategy beats Buy-and-Hold on test.")
    else:
        print("\nResult: selected strategy does not beat Buy-and-Hold on test.")

    print(f"\nValidation results saved to: {validation_results_file}")
    print(f"Test results saved to: {test_results_file}")
    print(f"Daily test returns saved to: {daily_returns_file}")


if __name__ == "__main__":
    main()
