"""
Simple Backtest.

Strategy:
- If Prediction = 1, hold the stock for the next trading day.
- If Prediction = 0, do not hold the stock.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml


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

    backtest_data = data.dropna(subset=["Next_Day_Return"]).copy()

    # Equal-weighted daily portfolio return across the selected tech stocks.
    daily_returns = backtest_data.groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]].mean()

    strategy_total_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
    buy_and_hold_total_return = (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1

    print("Simple Backtest")
    print("===============")
    print(f"Strategy Return:     {strategy_total_return:.2%}")
    print(f"Buy-and-Hold Return: {buy_and_hold_total_return:.2%}")
    print(f"Difference:          {strategy_total_return - buy_and_hold_total_return:.2%}")

    backtest_data.to_csv(results_file, index=False)
    print(f"\nBacktest results saved to: {results_file}")


if __name__ == "__main__":
    main()
