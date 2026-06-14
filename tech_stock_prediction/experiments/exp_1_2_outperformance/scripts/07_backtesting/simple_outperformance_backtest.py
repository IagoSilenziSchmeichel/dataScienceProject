"""
Simple backtest for outperformance predictions.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml
from formatting import format_percent, save_csv


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def main():
    print("Outperformance Simple Backtest")
    print("==============================")

    predictions_file = EXPERIMENT_ROOT / PARAMS["PREDICTIONS"]["TEST_FILE"]
    results_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["RESULTS_FILE"]

    data = pd.read_csv(predictions_file, parse_dates=["Date"])
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    data["Strategy_Return"] = data["Prediction"] * data["Future_Return"]
    data["Buy_Hold_Return"] = data["Future_Return"]

    daily_returns = data.groupby("Date")[["Strategy_Return", "Buy_Hold_Return"]].mean()

    strategy_total_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
    buy_hold_total_return = (1 + daily_returns["Buy_Hold_Return"]).prod() - 1

    print(f"Strategy Return:     {format_percent(strategy_total_return)}")
    print(f"Buy-and-Hold Return: {format_percent(buy_hold_total_return)}")
    print(f"Difference:          {format_percent(strategy_total_return - buy_hold_total_return)}")

    results_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(data, results_file)
    print(f"\nBacktest results saved to: {results_file}")


if __name__ == "__main__":
    main()
