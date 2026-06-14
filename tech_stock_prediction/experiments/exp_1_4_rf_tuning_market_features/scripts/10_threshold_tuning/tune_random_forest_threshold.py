"""
Tune Random Forest probability thresholds.

This script loads the existing Random Forest test predictions and evaluates
several probability thresholds with the same long/cash idea as the simple
backtest.
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

THRESHOLDS = [0.45, 0.47, 0.49, 0.50, 0.51, 0.53, 0.55, 0.60, 0.65, 0.70]
RESULTS_FILE = "data/processed/random_forest_threshold_results.csv"
DAILY_RETURNS_FILE = "data/processed/random_forest_threshold_daily_returns.csv"


def main():
    print("Random Forest Threshold Tuning")
    print("==============================")

    predictions_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["PREDICTIONS_FILE"]
    results_file = EXPERIMENT_ROOT / RESULTS_FILE
    daily_returns_file = EXPERIMENT_ROOT / DAILY_RETURNS_FILE

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_file}\n"
            "Run scripts/06_model_testing/evaluate_random_forest.py first."
        )

    data = pd.read_csv(predictions_file, parse_dates=["Date"])

    required_columns = {"Date", "Ticker", "Probability", "Future_Return"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Threshold tuning needs these missing columns in test_predictions.csv: "
            f"{missing_columns}. Run evaluate_random_forest.py again after the "
            "Probability change."
        )

    data = data.dropna(subset=["Future_Return"]).copy()
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    all_results = []
    all_daily_returns = []

    for threshold in THRESHOLDS:
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
        all_daily_returns.append(daily_returns)

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

        all_results.append(
            {
                "Threshold": threshold,
                "Total_Predictions": total_predictions,
                "Buy_Signals": buy_signals,
                "Signal_Rate": signal_rate,
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
        )

    results = pd.DataFrame(all_results)
    daily_results = pd.concat(all_daily_returns, ignore_index=True)

    results_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(results, results_file)
    save_csv(daily_results, daily_returns_file)

    print("\nThreshold summary:")
    display_columns = [
        "Threshold",
        "Signal_Rate",
        "Strategy_Total_Return",
        "Buy_Hold_Total_Return",
        "Difference",
    ]
    print(results[display_columns].to_string(index=False))

    best_row = results.sort_values("Difference", ascending=False).iloc[0]
    print("\nBest threshold by Difference:")
    print(f"Threshold:        {best_row['Threshold']:.2f}")
    print(f"Signal rate:      {format_percent(best_row['Signal_Rate'])}")
    print(f"Strategy return:  {format_percent(best_row['Strategy_Total_Return'])}")
    print(f"Buy-hold return:  {format_percent(best_row['Buy_Hold_Total_Return'])}")
    print(f"Difference:       {format_percent(best_row['Difference'])}")
    print(f"Strategy CAGR:    {format_percent(best_row['Strategy_CAGR'])}")
    print(f"Buy-hold CAGR:    {format_percent(best_row['Buy_Hold_CAGR'])}")
    print(f"Strategy Sharpe:  {best_row['Strategy_Sharpe']:.5f}")
    print(f"Buy-hold Sharpe:  {best_row['Buy_Hold_Sharpe']:.5f}")
    print(f"Strategy Max Drawdown: {format_percent(best_row['Strategy_Max_Drawdown'])}")
    print(f"Buy-hold Max Drawdown: {format_percent(best_row['Buy_Hold_Max_Drawdown'])}")
    print(f"Strategy Volatility: {format_percent(best_row['Strategy_Volatility'])}")
    print(f"Buy-hold Volatility: {format_percent(best_row['Buy_Hold_Volatility'])}")

    print(f"\nThreshold results saved to: {results_file}")
    print(f"Daily threshold returns saved to: {daily_returns_file}")


if __name__ == "__main__":
    main()
