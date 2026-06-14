"""
Tune Random Forest hyperparameters for the market-features experiment.

Only train data is used for fitting candidate models. Validation data is used
for hyperparameter selection. Test data is not touched here.
"""

from itertools import product
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import joblib
import pandas as pd
import yaml
from formatting import format_decimal, save_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))
TUNING_RESULTS_FILE = "data/processed/random_forest_hyperparameter_tuning_results.csv"
MAX_COMBINATIONS = 60


PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_leaf": [5, 10, 20, 50],
    "min_samples_split": [10, 25, 50],
    "max_features": ["sqrt", 0.3, 0.5],
    "class_weight": [None, "balanced"],
}


def load_feature_columns():
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["FEATURE_PATH"]
    return [line.strip() for line in open(feature_path) if line.strip()]


def sample_parameter_combinations():
    keys = list(PARAM_GRID.keys())
    all_combinations = [
        dict(zip(keys, values))
        for values in product(*(PARAM_GRID[key] for key in keys))
    ]
    random.seed(PARAMS["MODELING"]["RANDOM_STATE"])
    random.shuffle(all_combinations)
    return all_combinations[:MAX_COMBINATIONS]


def evaluate_model(model, x_validation, y_validation):
    predictions = model.predict(x_validation)
    return {
        "Validation_Accuracy": accuracy_score(y_validation, predictions),
        "Validation_Precision": precision_score(y_validation, predictions, zero_division=0),
        "Validation_Recall": recall_score(y_validation, predictions, zero_division=0),
        "Validation_F1": f1_score(y_validation, predictions, zero_division=0),
    }


def main():
    print("Random Forest Hyperparameter Tuning")
    print("===================================")

    train_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["TRAIN_FILE"]
    validation_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["VALIDATION_FILE"]
    model_file = EXPERIMENT_ROOT / PARAMS["MODELING"]["MODEL_FILE"]
    tuning_results_file = EXPERIMENT_ROOT / TUNING_RESULTS_FILE

    feature_columns = load_feature_columns()
    target_column = PARAMS["MODELING"]["TARGET"]

    train_data = pd.read_csv(train_file, parse_dates=["Date"])
    validation_data = pd.read_csv(validation_file, parse_dates=["Date"])

    x_train = train_data[feature_columns]
    y_train = train_data[target_column]
    x_validation = validation_data[feature_columns]
    y_validation = validation_data[target_column]

    parameter_combinations = sample_parameter_combinations()
    print(f"Testing {len(parameter_combinations)} parameter combinations.")

    results = []
    best_result = None
    best_params = None

    for index, params in enumerate(parameter_combinations, start=1):
        print(f"[{index}/{len(parameter_combinations)}] {params}", flush=True)
        model = RandomForestClassifier(
            random_state=PARAMS["MODELING"]["RANDOM_STATE"],
            n_jobs=-1,
            **params,
        )
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_validation, y_validation)

        row = {**params, **metrics}
        results.append(row)

        if best_result is None:
            best_result = row
            best_params = params
            continue

        current_key = (row["Validation_F1"], row["Validation_Accuracy"])
        best_key = (best_result["Validation_F1"], best_result["Validation_Accuracy"])
        if current_key > best_key:
            best_result = row
            best_params = params

    results_data = pd.DataFrame(results)
    results_data = results_data.sort_values(
        ["Validation_F1", "Validation_Accuracy"],
        ascending=False,
    )
    tuning_results_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(results_data, tuning_results_file)

    print("\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\nBest validation metrics:")
    print(f"Accuracy:  {format_decimal(best_result['Validation_Accuracy'])}")
    print(f"Precision: {format_decimal(best_result['Validation_Precision'])}")
    print(f"Recall:    {format_decimal(best_result['Validation_Recall'])}")
    print(f"F1-Score:  {format_decimal(best_result['Validation_F1'])}")

    best_model = RandomForestClassifier(
        random_state=PARAMS["MODELING"]["RANDOM_STATE"],
        n_jobs=-1,
        **best_params,
    )
    best_model.fit(x_train, y_train)

    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_file)

    print(f"\nTuning results saved to: {tuning_results_file}")
    print(f"Best tuned model saved to: {model_file}")


if __name__ == "__main__":
    main()
