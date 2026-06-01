"""
Model Testing.

Loads the trained model, evaluates it on the test set and stores predictions.
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
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["FEATURE_PATH"]
    return [line.strip() for line in open(feature_path) if line.strip()]


def main():
    test_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["TEST_FILE"]
    model_file = EXPERIMENT_ROOT / PARAMS["MODELING"]["MODEL_FILE"]
    predictions_file = EXPERIMENT_ROOT / PARAMS["BACKTEST"]["PREDICTIONS_FILE"]

    feature_columns = load_feature_columns()
    target_column = PARAMS["MODELING"]["TARGET"]

    test_data = pd.read_csv(test_file, parse_dates=["Date"])
    model = joblib.load(model_file)

    x_test = test_data[feature_columns]
    y_test = test_data[target_column]

    test_predictions = model.predict(x_test)

    print("Test results:")
    print(f"Accuracy:  {format_decimal(accuracy_score(y_test, test_predictions))}")
    print(f"Precision: {format_decimal(precision_score(y_test, test_predictions, zero_division=0))}")
    print(f"Recall:    {format_decimal(recall_score(y_test, test_predictions, zero_division=0))}")
    print(f"F1-Score:  {format_decimal(f1_score(y_test, test_predictions, zero_division=0))}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    results = test_data.copy()
    results["Prediction"] = test_predictions
    save_csv(results, predictions_file)

    print(f"\nPredictions saved to: {predictions_file}")


if __name__ == "__main__":
    main()
