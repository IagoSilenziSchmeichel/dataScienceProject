"""
Validation-selected threshold backtest.

Thresholds are selected only on validation results. The selected threshold is
then evaluated once on the test set.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml
from formatting import format_percent, save_csv
from utils.backtest_metrics import calculate_backtest_metrics


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

THRESHOLDS = [0.40, 0.42, 0.45, 0.47, 0.49, 0.50, 0.51, 0.53, 0.55, 0.60]
VALIDATION_RESULTS_FILE = (
    "data/processed/validation_selected_threshold_validation_results.csv"
)
TEST_RESULT_FILE = "data/processed/validation_selected_threshold_test_result.csv"
DAILY_RETURNS_FILE = "data/processed/validation_selected_threshold_daily_returns.csv"


def require_columns(data, file_path):
    required_columns = {"Date", "Ticker", "Probability", "Future_Return"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{file_path} is missing required columns: {missing_columns}. "
            "Run scripts/06_model_testing/evaluate_random_forest.py first."
        )


def backtest_threshold(data, threshold):
    threshold_data = data.copy()
    threshold_data["Threshold"] = threshold
    threshold_data["Threshold_Prediction"] = (
        threshold_data["Probability"] >= threshold
    ).astype(int)
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
    daily_returns["Threshold"] = threshold
    daily_returns["Strategy_Cumulative"] = (
        1 + daily_returns["Strategy_Return"]
    ).cumprod()
    daily_returns["Buy_Hold_Cumulative"] = (
        1 + daily_returns["Buy_Hold_Return"]
    ).cumprod()

    strategy_metrics = calculate_backtest_metrics(daily_returns, "Strategy_Return")
    buy_hold_metrics = calculate_backtest_metrics(daily_returns, "Buy_Hold_Return")

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

    result = {
        "Threshold": threshold,
        "Strategy_Total_Return": strategy_metrics["Total_Return"],
        "Buy_Hold_Total_Return": buy_hold_metrics["Total_Return"],
        "Difference": (
            strategy_metrics["Total_Return"] - buy_hold_metrics["Total_Return"]
        ),
        "Strategy_CAGR": strategy_metrics["CAGR"],
        "Buy_Hold_CAGR": buy_hold_metrics["CAGR"],
        "Strategy_Sharpe": strategy_metrics["Sharpe"],
        "Buy_Hold_Sharpe": buy_hold_metrics["Sharpe"],
        "Strategy_Max_Drawdown": strategy_metrics["Max_Drawdown"],
        "Buy_Hold_Max_Drawdown": buy_hold_metrics["Max_Drawdown"],
        "Strategy_Volatility": strategy_metrics["Volatility"],
        "Buy_Hold_Volatility": buy_hold_metrics["Volatility"],
        "Signal_Rate": signal_rate,
        "Total_Predictions": total_predictions,
        "Buy_Signals": buy_signals,
        "Average_Trade_Return": average_trade_return,
        "Win_Rate": win_rate,
    }

    return result, daily_returns


def main():
    print("Validation-Selected Threshold Backtest")
    print("======================================")
    print(f"Thresholds tested: {THRESHOLDS}")

    validation_predictions_file = (
        EXPERIMENT_ROOT / "data" / "processed" / "validation_predictions.csv"
    )
    test_predictions_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["PREDICTIONS_FILE"]
    validation_results_file = EXPERIMENT_ROOT / VALIDATION_RESULTS_FILE
    test_result_file = EXPERIMENT_ROOT / TEST_RESULT_FILE
    daily_returns_file = EXPERIMENT_ROOT / DAILY_RETURNS_FILE

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

    validation_results = []
    for threshold in THRESHOLDS:
        result, _ = backtest_threshold(validation_data, threshold)
        validation_results.append(result)

    validation_results = pd.DataFrame(validation_results)
    validation_results = validation_results.sort_values(
        ["Difference", "Strategy_Sharpe"],
        ascending=False,
    )
    best_validation = validation_results.iloc[0]
    best_threshold = best_validation["Threshold"]

    test_result, test_daily_returns = backtest_threshold(test_data, best_threshold)
    test_result = pd.DataFrame([test_result])

    validation_results_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(validation_results, validation_results_file)
    save_csv(test_result, test_result_file)
    save_csv(test_daily_returns, daily_returns_file)

    final_test = test_result.iloc[0]

    print("\nBest threshold on validation:")
    print(f"Threshold:              {best_threshold:.2f}")
    print(f"Validation Strategy:    {format_percent(best_validation['Strategy_Total_Return'])}")
    print(f"Validation Buy-Hold:    {format_percent(best_validation['Buy_Hold_Total_Return'])}")
    print(f"Validation Difference:  {format_percent(best_validation['Difference'])}")
    print(f"Validation Sharpe:      {best_validation['Strategy_Sharpe']:.5f}")

    print("\nFinal test performance:")
    print(f"Threshold:              {final_test['Threshold']:.2f}")
    print(f"Test Strategy:          {format_percent(final_test['Strategy_Total_Return'])}")
    print(f"Test Buy-Hold:          {format_percent(final_test['Buy_Hold_Total_Return'])}")
    print(f"Test Difference:        {format_percent(final_test['Difference'])}")
    print(f"Test Sharpe:            {final_test['Strategy_Sharpe']:.5f}")

    if final_test["Difference"] > 0:
        print("\nResult: selected threshold beats Buy-and-Hold on test.")
    else:
        print("\nResult: selected threshold does not beat Buy-and-Hold on test.")

    print(f"\nValidation results saved to: {validation_results_file}")
    print(f"Final test result saved to: {test_result_file}")
    print(f"Daily test returns saved to: {daily_returns_file}")


if __name__ == "__main__":
    main()
