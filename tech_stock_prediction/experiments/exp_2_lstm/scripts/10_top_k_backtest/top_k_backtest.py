"""
Top-K backtest for LSTM probabilities.

Instead of buying every stock with Prediction = 1, this script buys only the
stocks with the highest predicted probabilities on each trading day.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml
from formatting import format_decimal, format_percent, save_csv


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

TOP_K_VALUES = [1, 2, 3, 4, 5]


def require_columns(data, required_columns, file_name):
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"{file_name} is missing required columns: {missing_columns}")


def load_test_next_close(test_file):
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    test_data = pd.read_csv(test_file, parse_dates=["Date"])
    require_columns(test_data, ["Date", "Ticker", "Close"], "test.csv")

    test_data = test_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    test_data["Next_Close"] = test_data.groupby("Ticker")["Close"].shift(-1)

    return test_data[["Date", "Ticker", "Next_Close"]]


def load_predictions(predictions_file):
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}. "
            "Run scripts/04_model_testing/evaluate_lstm.py first."
        )

    predictions = pd.read_csv(predictions_file, parse_dates=["Date"])
    require_columns(
        predictions,
        ["Date", "Ticker", "Close", "Probability"],
        "lstm_test_predictions.csv",
    )

    return predictions.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def prepare_backtest_data():
    predictions_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["PREDICTIONS_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"]

    predictions = load_predictions(predictions_file)
    test_next_close = load_test_next_close(test_file)

    predictions = predictions.merge(test_next_close, on=["Date", "Ticker"], how="left")

    missing_next_close = predictions["Next_Close"].isna().sum()

    if missing_next_close > 0:
        print(f"Warning: Dropping {missing_next_close} rows without Next_Close.")
        predictions = predictions.dropna(subset=["Next_Close"]).copy()

    predictions["Future_Return"] = predictions["Next_Close"] / predictions["Close"] - 1

    return predictions


def calculate_top_k_result(predictions, top_k):
    data = predictions.copy()

    data["Probability_Rank"] = (
        data
        .groupby("Date")["Probability"]
        .rank(method="first", ascending=False)
    )
    data["Selected"] = data["Probability_Rank"] <= top_k

    selected_trades = data[data["Selected"]].copy()

    daily_strategy_returns = (
        selected_trades
        .groupby("Date")["Future_Return"]
        .mean()
        .rename("Strategy_Return")
    )
    daily_buy_hold_returns = (
        data
        .groupby("Date")["Future_Return"]
        .mean()
        .rename("Buy_And_Hold_Return")
    )
    daily_positions = (
        selected_trades
        .groupby("Date")["Ticker"]
        .count()
        .rename("Number_Of_Positions")
    )

    daily_results = pd.concat(
        [daily_strategy_returns, daily_buy_hold_returns, daily_positions],
        axis=1,
    ).fillna(0)

    strategy_return = (1 + daily_results["Strategy_Return"]).prod() - 1
    buy_and_hold_return = (1 + daily_results["Buy_And_Hold_Return"]).prod() - 1

    return {
        "top_k": top_k,
        "strategy_return": strategy_return,
        "buy_and_hold_return": buy_and_hold_return,
        "difference": strategy_return - buy_and_hold_return,
        "average_number_of_positions": daily_results["Number_Of_Positions"].mean(),
        "number_of_trading_days": len(daily_results),
    }


def print_results(results):
    print("LSTM Top-K Backtest")
    print("===================")
    print(
        results.to_string(
            index=False,
            formatters={
                "strategy_return": format_percent,
                "buy_and_hold_return": format_percent,
                "difference": format_percent,
                "average_number_of_positions": format_decimal,
            },
        )
    )

    best_result = results.sort_values("strategy_return", ascending=False).iloc[0]

    print("\nBest Top-K by Strategy Return:")
    print(best_result.to_string())


def main():
    predictions = prepare_backtest_data()
    output_file = EXPERIMENT_ROOT / "data" / "processed" / "lstm_top_k_results.csv"

    rows = []

    for top_k in TOP_K_VALUES:
        rows.append(calculate_top_k_result(predictions, top_k))

    results = pd.DataFrame(rows)

    save_csv(results, output_file)
    print_results(results)

    print(f"\nTop-K results saved to: {output_file}")


if __name__ == "__main__":
    main()
