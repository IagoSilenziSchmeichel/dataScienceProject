"""
Compare exp_1_3_market_features with exp_1_4_rf_tuning_market_features.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
from formatting import format_decimal, format_percent
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PROJECT_ROOT / "experiments" / "exp_1_3_market_features"
TUNED_ROOT = PROJECT_ROOT / "experiments" / "exp_1_4_rf_tuning_market_features"


def load_csv(path):
    if not path.exists():
        print(f"Missing optional file: {path}")
        return None
    data = pd.read_csv(path)
    if data.empty:
        print(f"Empty optional file: {path}")
        return None
    return data


def prediction_metrics(root, split):
    data = load_csv(root / "data" / "processed" / f"{split}_predictions.csv")
    if data is None or not {"Actual", "Prediction"}.issubset(data.columns):
        return {}

    y_true = data["Actual"]
    y_pred = data["Prediction"]
    return {
        f"{split.title()}_Accuracy": accuracy_score(y_true, y_pred),
        f"{split.title()}_Precision": precision_score(y_true, y_pred, zero_division=0),
        f"{split.title()}_Recall": recall_score(y_true, y_pred, zero_division=0),
        f"{split.title()}_F1": f1_score(y_true, y_pred, zero_division=0),
        f"{split.title()}_Always_Up_Accuracy": y_true.mean(),
    }


def first_row_metrics(path, prefix):
    data = load_csv(path)
    if data is None:
        return {}
    row = data.iloc[0]
    metrics = {}
    for column in [
        "Strategy_Total_Return",
        "Buy_Hold_Total_Return",
        "Difference",
        "Strategy_Sharpe",
        "Buy_Hold_Sharpe",
        "Strategy_Max_Drawdown",
        "Buy_Hold_Max_Drawdown",
    ]:
        if column in row.index:
            metrics[f"{prefix}_{column}"] = float(row[column])
    return metrics


def best_row_metrics(path, prefix):
    data = load_csv(path)
    if data is None or "Difference" not in data.columns:
        return {}
    if prefix == "Top_K" and "K" in data.columns:
        data = data[data["K"] != 10].copy()
        if data.empty:
            return {}
    row = data.sort_values("Difference", ascending=False).iloc[0]
    metrics = {}
    for column in [
        "Strategy_Total_Return",
        "Buy_Hold_Total_Return",
        "Difference",
        "Strategy_Sharpe",
        "Buy_Hold_Sharpe",
        "Strategy_Max_Drawdown",
        "Buy_Hold_Max_Drawdown",
    ]:
        if column in row.index:
            metrics[f"{prefix}_{column}"] = float(row[column])
    return metrics


def collect_metrics(root):
    metrics = {}
    metrics.update(prediction_metrics(root, "validation"))
    metrics.update(prediction_metrics(root, "test"))
    metrics.update(first_row_metrics(root / "data" / "processed" / "backtest_results.csv", "Simple_Backtest"))
    metrics.update(best_row_metrics(root / "data" / "processed" / "random_forest_threshold_results.csv", "Best_Threshold"))
    metrics.update(best_row_metrics(root / "data" / "processed" / "random_forest_top_k_results.csv", "Top_K"))
    return metrics


def print_comparison(baseline, tuned):
    print(f"{'Metric':<48}{'exp_1_3':>16}{'exp_1_4':>16}{'Delta':>16}")
    for metric in sorted(set(baseline) | set(tuned)):
        if metric not in baseline or metric not in tuned:
            continue
        baseline_value = baseline[metric]
        tuned_value = tuned[metric]
        delta = tuned_value - baseline_value
        is_percent = "Sharpe" not in metric
        if is_percent:
            print(
                f"{metric:<48}"
                f"{format_percent(baseline_value):>16}"
                f"{format_percent(tuned_value):>16}"
                f"{format_percent(delta):>16}"
            )
        else:
            print(
                f"{metric:<48}"
                f"{format_decimal(baseline_value):>16}"
                f"{format_decimal(tuned_value):>16}"
                f"{format_decimal(delta):>16}"
            )


def print_tuned_summary(tuned):
    test_accuracy = tuned.get("Test_Accuracy")
    always_up_accuracy = tuned.get("Test_Always_Up_Accuracy")
    best_backtest = max(
        [
            tuned.get("Simple_Backtest_Difference"),
            tuned.get("Best_Threshold_Difference"),
            tuned.get("Top_K_Difference"),
        ],
        key=lambda value: -999 if value is None else value,
    )

    print("\nTuned model summary:")
    if test_accuracy is not None and always_up_accuracy is not None:
        print(f"Test accuracy:       {format_percent(test_accuracy)}")
        print(f"Always-Up accuracy:  {format_percent(always_up_accuracy)}")
        if test_accuracy > always_up_accuracy:
            print("Accuracy result: tuned model beats Always-Up.")
        else:
            print("Accuracy result: tuned model does not beat Always-Up.")

    if best_backtest is not None:
        print(f"Best backtest difference vs Buy-and-Hold: {format_percent(best_backtest)}")
        if best_backtest > 0:
            print("Backtest result: at least one strategy beats Buy-and-Hold.")
        else:
            print("Backtest result: no strategy beats Buy-and-Hold.")


def main():
    print("Compare exp_1_3_market_features vs exp_1_4_rf_tuning_market_features")
    print("====================================================================")

    tuning_results = load_csv(
        TUNED_ROOT / "data" / "processed" / "random_forest_hyperparameter_tuning_results.csv"
    )
    if tuning_results is not None:
        best = tuning_results.sort_values(
            ["Validation_F1", "Validation_Accuracy"],
            ascending=False,
        ).iloc[0]
        print("\nBest tuned hyperparameters:")
        for column in [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "class_weight",
        ]:
            value = best[column]
            if pd.isna(value):
                value = None
            print(f"{column}: {value}")
        print(f"Validation Accuracy:  {format_decimal(best['Validation_Accuracy'])}")
        print(f"Validation Precision: {format_decimal(best['Validation_Precision'])}")
        print(f"Validation Recall:    {format_decimal(best['Validation_Recall'])}")
        print(f"Validation F1:        {format_decimal(best['Validation_F1'])}")

    baseline_metrics = collect_metrics(BASELINE_ROOT)
    tuned_metrics = collect_metrics(TUNED_ROOT)
    print()
    print_comparison(baseline_metrics, tuned_metrics)
    print_tuned_summary(tuned_metrics)


if __name__ == "__main__":
    main()
