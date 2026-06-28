"""
Dummy Baseline ("Always Up").

This is a naive baseline that always predicts 1 (the stock rises tomorrow).
Because the market tends to go up more often than down, this trivial strategy
already reaches an accuracy equal to the share of rising days.

The Random Forest model is only useful if it clearly beats this baseline.
This script prints the baseline metrics next to the model metrics on the test
set, so we always see the honest comparison at the end of every pipeline run.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml
from formatting import format_decimal
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    predictions_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["PREDICTIONS_FILE"]
    target_column = PARAMS["MODELING"]["TARGET"]

    data = pd.read_csv(predictions_file, parse_dates=["Date"])

    y_true = data[target_column]
    model_prediction = data["Prediction"]
    always_yes_prediction = pd.Series(1, index=data.index)

    rising_share = y_true.mean()

    baseline_metrics = metrics(y_true, always_yes_prediction)
    model_metrics = metrics(y_true, model_prediction)

    print("Dummy Baseline (\"Always Up\" / immer 1)")
    print("======================================")
    print(f"Share of rising days in test set: {format_decimal(rising_share)}")
    print("(This is exactly the accuracy of a model that always predicts 1.)")

    print("\nComparison on the test set:")
    print(f"{'Metric':<12}{'Always-Yes':>14}{'Random Forest':>16}")
    for metric_name in baseline_metrics:
        baseline_value = format_decimal(baseline_metrics[metric_name])
        model_value = format_decimal(model_metrics[metric_name])
        print(f"{metric_name:<12}{baseline_value:>14}{model_value:>16}")

    accuracy_difference = model_metrics["Accuracy"] - baseline_metrics["Accuracy"]
    print(f"\nAccuracy difference (Model - Baseline): {format_decimal(accuracy_difference)}")

    if accuracy_difference > 0:
        print("=> The model beats the naive Always-Yes baseline.")
    else:
        print("=> The model does NOT beat the naive Always-Yes baseline.")


if __name__ == "__main__":
    main()
