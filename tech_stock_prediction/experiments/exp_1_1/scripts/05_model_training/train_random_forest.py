"""
Model Training.

Trains a simple RandomForestClassifier on the training data and evaluates it on
the validation data.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def load_feature_columns():
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["FEATURE_PATH"]
    return [line.strip() for line in open(feature_path) if line.strip()]


def print_metrics(y_true, y_pred):
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")


def main():
    train_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["TRAIN_FILE"]
    validation_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["VALIDATION_FILE"]
    model_file = EXPERIMENT_ROOT / PARAMS["MODELING"]["MODEL_FILE"]

    feature_columns = load_feature_columns()
    target_column = PARAMS["MODELING"]["TARGET"]

    train_data = pd.read_csv(train_file, parse_dates=["Date"])
    validation_data = pd.read_csv(validation_file, parse_dates=["Date"])

    x_train = train_data[feature_columns]
    y_train = train_data[target_column]
    x_validation = validation_data[feature_columns]
    y_validation = validation_data[target_column]

    model = RandomForestClassifier(
        n_estimators=PARAMS["MODELING"]["N_ESTIMATORS"],
        max_depth=PARAMS["MODELING"]["MAX_DEPTH"],
        random_state=PARAMS["MODELING"]["RANDOM_STATE"],
        class_weight="balanced",
    )

    print("Training RandomForestClassifier...")
    model.fit(x_train, y_train)

    validation_predictions = model.predict(x_validation)

    print("\nValidation results:")
    print_metrics(y_validation, validation_predictions)

    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)

    print(f"\nModel saved to: {model_file}")


if __name__ == "__main__":
    main()
