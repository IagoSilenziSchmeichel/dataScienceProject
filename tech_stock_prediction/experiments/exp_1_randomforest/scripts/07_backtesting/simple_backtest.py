"""
Simple Backtest.

Strategy:
- If Prediction = 1, hold the stock for the next trading day.
- If Prediction = 0, do not hold the stock.
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


def main():
    predictions_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["PREDICTIONS_FILE"]
    results_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["RESULTS_FILE"]

    data = pd.read_csv(predictions_file, parse_dates=["Date"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Return from today's close to the next available close for the same ticker.
    data["Next_Day_Return"] = data.groupby("Ticker")["Close"].shift(-1) / data["Close"] - 1

    data["Strategy_Return"] = data["Prediction"] * data["Next_Day_Return"]
    data["Buy_And_Hold_Return"] = data["Next_Day_Return"]
    data["Buy_Hold_Return"] = data["Next_Day_Return"]

    backtest_data = data.dropna(subset=["Next_Day_Return"]).copy()

    # Equal-weighted daily portfolio return across the selected tech stocks.
    daily_returns = backtest_data.groupby("Date")[["Strategy_Return", "Buy_Hold_Return"]].mean()

    strategy_metrics = calculate_backtest_metrics(daily_returns, "Strategy_Return")
    buy_hold_metrics = calculate_backtest_metrics(daily_returns, "Buy_Hold_Return")

    strategy_total_return = strategy_metrics["Total_Return"]
    buy_and_hold_total_return = buy_hold_metrics["Total_Return"]
    difference = strategy_total_return - buy_and_hold_total_return

    metric_values = {
        "Strategy_Total_Return": strategy_total_return,
        "Buy_Hold_Total_Return": buy_and_hold_total_return,
        "Difference": difference,
        "Strategy_CAGR": strategy_metrics["CAGR"],
        "Buy_Hold_CAGR": buy_hold_metrics["CAGR"],
        "Strategy_Sharpe": strategy_metrics["Sharpe"],
        "Buy_Hold_Sharpe": buy_hold_metrics["Sharpe"],
        "Strategy_Max_Drawdown": strategy_metrics["Max_Drawdown"],
        "Buy_Hold_Max_Drawdown": buy_hold_metrics["Max_Drawdown"],
        "Strategy_Volatility": strategy_metrics["Volatility"],
        "Buy_Hold_Volatility": buy_hold_metrics["Volatility"],
    }
    for column, value in metric_values.items():
        backtest_data[column] = value

    print("Simple Backtest")
    print("===============")
    print(f"Strategy Return:     {format_percent(strategy_total_return)}")
    print(f"Buy-and-Hold Return: {format_percent(buy_and_hold_total_return)}")
    print(f"Difference:          {format_percent(difference)}")
    print()
    print(f"Strategy CAGR:          {format_percent(strategy_metrics['CAGR'])}")
    print(f"Buy-and-Hold CAGR:      {format_percent(buy_hold_metrics['CAGR'])}")
    print(f"Strategy Sharpe:        {strategy_metrics['Sharpe']:.5f}")
    print(f"Buy-and-Hold Sharpe:    {buy_hold_metrics['Sharpe']:.5f}")
    print(f"Strategy Max Drawdown:  {format_percent(strategy_metrics['Max_Drawdown'])}")
    print(f"Buy-and-Hold Max Drawdown: {format_percent(buy_hold_metrics['Max_Drawdown'])}")
    print(f"Strategy Volatility:    {format_percent(strategy_metrics['Volatility'])}")
    print(f"Buy-and-Hold Volatility: {format_percent(buy_hold_metrics['Volatility'])}")

    save_csv(backtest_data, results_file)
    print(f"\nBacktest results saved to: {results_file}")


if __name__ == "__main__":
    main()
