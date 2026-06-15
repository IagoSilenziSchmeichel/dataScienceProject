"""
Threshold backtest for LSTM predictions.

The normal LSTM evaluation turns probabilities into predictions with one fixed
threshold. This script tests several thresholds and checks how the trading
strategy changes when we only buy on stronger model signals.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import yaml
from formatting import format_decimal, format_percent, save_csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def require_columns(data, required_columns, file_name):
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(
            f"{file_name} is missing required columns: {missing_columns}"
        )


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

    if "Actual" not in predictions.columns and "Target" not in predictions.columns:
        raise ValueError(
            "lstm_test_predictions.csv needs either an 'Actual' or 'Target' column."
        )

    if "Actual" not in predictions.columns:
        predictions["Actual"] = predictions["Target"]

    return predictions.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def calculate_strategy_results(predictions, threshold):
    y_true = predictions["Actual"].astype(int)
    y_pred = (predictions["Probability"] >= threshold).astype(int)

    strategy_return = np.where(
        y_pred == 1,
        predictions["Future_Return"],
        0.0,
    )

    result_data = predictions[["Date", "Ticker"]].copy()
    result_data["Strategy_Return"] = strategy_return
    result_data["Buy_And_Hold_Return"] = predictions["Future_Return"]

    daily_returns = result_data.groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]].mean()
    strategy_total_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
    buy_and_hold_total_return = (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "Predicted_Up_Share": y_pred.mean(),
        "Buy_Signals": int(y_pred.sum()),
        "Total_Rows": len(y_pred),
        "Strategy_Return": strategy_total_return,
        "Buy_And_Hold_Return": buy_and_hold_total_return,
        "Difference": strategy_total_return - buy_and_hold_total_return,
    }


def print_results(results):
    print("LSTM Threshold Backtest")
    print("=======================")
    print(
        results[
            [
                "Threshold",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "Predicted_Up_Share",
                "Buy_Signals",
                "Strategy_Return",
                "Buy_And_Hold_Return",
                "Difference",
            ]
        ].to_string(
            index=False,
            formatters={
                "Threshold": format_decimal,
                "Accuracy": format_decimal,
                "Precision": format_decimal,
                "Recall": format_decimal,
                "F1_Score": format_decimal,
                "Predicted_Up_Share": format_decimal,
                "Strategy_Return": format_percent,
                "Buy_And_Hold_Return": format_percent,
                "Difference": format_percent,
            },
        )
    )

    best_f1 = results.sort_values("F1_Score", ascending=False).iloc[0]
    best_return = results.sort_values("Strategy_Return", ascending=False).iloc[0]

    print("\nBest threshold by F1-Score:")
    print(f"{format_decimal(best_f1['Threshold'])} with F1 = {format_decimal(best_f1['F1_Score'])}")

    print("\nBest threshold by Strategy Return:")
    print(
        f"{format_decimal(best_return['Threshold'])} with return = "
        f"{format_percent(best_return['Strategy_Return'])}"
    )


def main():
    predictions_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["PREDICTIONS_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"]
    output_file = EXPERIMENT_ROOT / "data" / "processed" / "lstm_threshold_results.csv"

    predictions = load_predictions(predictions_file)
    test_next_close = load_test_next_close(test_file)

    predictions = predictions.merge(test_next_close, on=["Date", "Ticker"], how="left")
    missing_next_close = predictions["Next_Close"].isna().sum()

    if missing_next_close > 0:
        print(f"Warning: Dropping {missing_next_close} rows without Next_Close.")
        predictions = predictions.dropna(subset=["Next_Close"]).copy()

    predictions["Future_Return"] = predictions["Next_Close"] / predictions["Close"] - 1

    rows = []
    for threshold in THRESHOLDS:
        rows.append(calculate_strategy_results(predictions, threshold))

    results = pd.DataFrame(rows)
    save_csv(results, output_file)

    print_results(results)
    print(f"\nThreshold results saved to: {output_file}")


if __name__ == "__main__":
    main()
