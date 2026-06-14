"""
Evaluate the outperformance Random Forest model.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import joblib
import pandas as pd
import yaml
from formatting import format_decimal, save_csv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def load_feature_columns():
    feature_path = EXPERIMENT_ROOT / PARAMS["SOURCE"]["FEATURE_PATH"]
    return [line.strip() for line in open(feature_path) if line.strip()]


def evaluate_and_save_predictions(
    dataset_name,
    data,
    model,
    feature_columns,
    target_column,
    output_file,
):
    x_data = data[feature_columns]
    y_true = data[target_column]

    predictions = model.predict(x_data)
    class_1_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_data)[:, class_1_index]

    print(f"{dataset_name} results:")
    print(f"Accuracy:  {format_decimal(accuracy_score(y_true, predictions))}")
    print(f"Precision: {format_decimal(precision_score(y_true, predictions, zero_division=0))}")
    print(f"Recall:    {format_decimal(recall_score(y_true, predictions, zero_division=0))}")
    print(f"F1-Score:  {format_decimal(f1_score(y_true, predictions, zero_division=0))}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    columns = [
        "Date",
        "Ticker",
        "Close",
        "Future_Return",
        "Daily_Average_Future_Return",
    ]
    results = data[columns].copy()
    results["Actual"] = y_true
    results["Prediction"] = predictions
    results["Probability"] = probabilities

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(results, output_file)

    print(f"\nPredictions saved to: {output_file}")


def main():
    print("Evaluating outperformance Random Forest")
    print("=======================================")

    validation_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["VALIDATION_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["TEST_FILE"]
    model_file = EXPERIMENT_ROOT / PARAMS["MODELING"]["MODEL_FILE"]
    validation_predictions_file = (
        EXPERIMENT_ROOT / PARAMS["PREDICTIONS"]["VALIDATION_FILE"]
    )
    test_predictions_file = EXPERIMENT_ROOT / PARAMS["PREDICTIONS"]["TEST_FILE"]

    feature_columns = load_feature_columns()
    target_column = PARAMS["MODELING"]["TARGET"]

    validation_data = pd.read_csv(validation_file, parse_dates=["Date"])
    test_data = pd.read_csv(test_file, parse_dates=["Date"])
    model = joblib.load(model_file)

    evaluate_and_save_predictions(
        "Validation",
        validation_data,
        model,
        feature_columns,
        target_column,
        validation_predictions_file,
    )
    print()
    evaluate_and_save_predictions(
        "Test",
        test_data,
        model,
        feature_columns,
        target_column,
        test_predictions_file,
    )


if __name__ == "__main__":
    main()
