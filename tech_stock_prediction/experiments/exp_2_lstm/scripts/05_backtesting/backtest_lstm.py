"""
Backtest LSTM predictions.

This is a simple backtest.

Strategy:
- If the LSTM predicts 1, we buy/hold the stock for the next period.
- If the LSTM predicts 0, we stay in cash for that stock.

Comparison:
- Buy-and-Hold means we always hold all stocks equally.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def load_test_data_with_next_close(test_file):
    """
    Load the original test data and make sure it contains Next_Close.

    If Next_Close is not available, it is calculated from Close by ticker.
    """
    test_data = pd.read_csv(test_file)

    test_data["Date"] = pd.to_datetime(test_data["Date"])

    test_data = test_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if "Next_Close" not in test_data.columns:
        test_data["Next_Close"] = (
            test_data
            .groupby("Ticker")["Close"]
            .shift(-1)
        )

    return test_data[["Date", "Ticker", "Next_Close"]]


def main():
    print("LSTM Backtest")
    print("=============")

    predictions_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["PREDICTIONS_FILE"]
    backtest_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["BACKTEST_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"]

    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    predictions = pd.read_csv(predictions_file)

    required_columns = [
        "Date",
        "Ticker",
        "Close",
        "Prediction",
        "Actual",
        "Probability",
    ]

    missing_columns = []

    for column in required_columns:
        if column not in predictions.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    predictions["Date"] = pd.to_datetime(predictions["Date"])

    predictions = predictions.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    test_next_close = load_test_data_with_next_close(test_file)

    predictions = predictions.merge(
        test_next_close,
        on=["Date", "Ticker"],
        how="left",
    )

    missing_next_close = predictions["Next_Close"].isna().sum()

    if missing_next_close > 0:
        print(f"Warning: Dropping {missing_next_close} rows without Next_Close.")
        predictions = predictions.dropna(subset=["Next_Close"]).copy()

    predictions["Future_Return"] = (
            predictions["Next_Close"] / predictions["Close"] - 1
    )

    # Strategy:
    # If Prediction = 1, take the future return.
    # If Prediction = 0, stay in cash and get return 0.
    predictions["Strategy_Return"] = np.where(
        predictions["Prediction"] == 1,
        predictions["Future_Return"],
        0.0,
        )

    # Buy-and-Hold:
    # Always take the future return.
    predictions["Buy_Hold_Return"] = predictions["Future_Return"]

    # Aggregate by date.
    # This creates one portfolio return per day.
    daily_returns = (
        predictions
        .groupby("Date")
        .agg(
            Strategy_Return=("Strategy_Return", "mean"),
            Buy_Hold_Return=("Buy_Hold_Return", "mean"),
            Number_Of_Signals=("Prediction", "sum"),
            Number_Of_Stocks=("Ticker", "count"),
        )
        .reset_index()
    )

    daily_returns["Strategy_Cumulative"] = (
            1 + daily_returns["Strategy_Return"]
    ).cumprod()

    daily_returns["Buy_Hold_Cumulative"] = (
            1 + daily_returns["Buy_Hold_Return"]
    ).cumprod()

    strategy_total_return = daily_returns["Strategy_Cumulative"].iloc[-1] - 1
    buy_hold_total_return = daily_returns["Buy_Hold_Cumulative"].iloc[-1] - 1
    difference = strategy_total_return - buy_hold_total_return

    total_predictions = len(predictions)
    total_buy_signals = int(predictions["Prediction"].sum())
    signal_rate = total_buy_signals / total_predictions

    bought_trades = predictions[predictions["Prediction"] == 1].copy()

    if len(bought_trades) > 0:
        average_trade_return = bought_trades["Future_Return"].mean()
        win_rate = (bought_trades["Future_Return"] > 0).mean()
    else:
        average_trade_return = 0.0
        win_rate = 0.0

    print(f"Total predictions:       {total_predictions}")
    print(f"Buy signals:             {total_buy_signals}")
    print(f"Signal rate:             {signal_rate:.2%}")
    print(f"Average trade return:    {average_trade_return:.5%}")
    print(f"Win rate of buy signals: {win_rate:.2%}")

    print("\nPortfolio comparison:")
    print(f"Strategy Return:         {strategy_total_return:.5%}")
    print(f"Buy-and-Hold Return:     {buy_hold_total_return:.5%}")
    print(f"Difference:              {difference:.5%}")

    backtest_file.parent.mkdir(parents=True, exist_ok=True)
    daily_returns.to_csv(backtest_file, index=False)

    print(f"\nBacktest results saved to: {backtest_file}")


if __name__ == "__main__":
    main()
