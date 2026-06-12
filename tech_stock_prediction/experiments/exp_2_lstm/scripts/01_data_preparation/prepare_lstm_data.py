"""
Prepare LSTM data.

This script checks whether the required Random Forest processed data files exist.
The LSTM experiment reuses the processed train, validation and test files from
experiment exp_1_1 so that both models can be compared fairly.
"""

from pathlib import Path
import sys

import pandas as pd
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def load_feature_columns(feature_path):
    """
    Load feature names from features.txt.

    Empty lines and comment lines are ignored.
    """
    feature_columns = []

    with open(feature_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            feature_columns.append(line)

    return feature_columns


def check_file_exists(file_path, description):
    """
    Check whether a required file exists.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")

    print(f"{description} found: {file_path}")


def main():
    print("Preparing LSTM data")
    print("===================")

    train_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TRAIN_FILE"]
    validation_file = EXPERIMENT_ROOT / PARAMS["DATA"]["VALIDATION_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"]
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA"]["FEATURE_PATH"]

    check_file_exists(train_file, "Train file")
    check_file_exists(validation_file, "Validation file")
    check_file_exists(test_file, "Test file")
    check_file_exists(feature_path, "Feature list")

    feature_columns = load_feature_columns(feature_path)

    print("\nFeatures used for LSTM:")
    for feature in feature_columns:
        print(f"- {feature}")

    train_data = pd.read_csv(train_file)
    validation_data = pd.read_csv(validation_file)
    test_data = pd.read_csv(test_file)

    print("\nDataset sizes:")
    print(f"Train rows:      {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Test rows:       {len(test_data)}")

    print("\nLSTM data preparation check completed successfully.")


if __name__ == "__main__":
    main()