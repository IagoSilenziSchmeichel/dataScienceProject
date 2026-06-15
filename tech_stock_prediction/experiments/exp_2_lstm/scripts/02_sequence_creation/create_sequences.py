"""
Create LSTM sequences.

The LSTM does not learn from single rows like the Random Forest.
Instead, it learns from sequences of past days.

Example:
sequence_length = 30

Input:
last 30 days of features

Target:
Target value of the last day in the sequence
"""

from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


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


def load_dataset(file_path):
    """
    Load one processed CSV file and convert the Date column.
    """
    data = pd.read_csv(file_path)

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return data


def check_required_columns(data, feature_columns, target_column):
    """
    Check that all required columns exist in the data.
    """
    required_columns = feature_columns + [target_column, "Date", "Ticker", "Close"]

    missing_columns = []

    for column in required_columns:
        if column not in data.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")


def check_feature_values(data, feature_columns, dataset_name):
    """
    Check whether feature columns contain NaN or infinite values.

    LSTM models cannot train properly with NaN or infinity values.
    """
    missing_values = data[feature_columns].isna().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"{dataset_name} contains {missing_values} missing feature values."
        )

    values = data[feature_columns].values

    if not np.isfinite(values).all():
        raise ValueError(
            f"{dataset_name} contains infinite or too large feature values."
        )


def scale_features(train_data, validation_data, test_data, feature_columns, scaler_file):
    """
    Standardize LSTM input features without data leakage.

    LSTMs learn with gradient descent, so features on very different scales can
    make training unstable. Random Forests are less sensitive to this because
    they split by feature thresholds instead of learning weights with gradients.

    Important:
    The scaler is fitted only on the training data. Validation and test data
    are transformed with this already fitted scaler.
    """
    scaler = StandardScaler()

    train_scaled = train_data.copy()
    validation_scaled = validation_data.copy()
    test_scaled = test_data.copy()

    train_scaled[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    validation_scaled[feature_columns] = scaler.transform(validation_data[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test_data[feature_columns])

    scaler_file.parent.mkdir(parents=True, exist_ok=True)

    with open(scaler_file, "wb") as file:
        pickle.dump(scaler, file)

    return train_scaled, validation_scaled, test_scaled


def create_sequences_for_dataset(data, feature_columns, target_column, sequence_length):
    """
    Create LSTM sequences for one dataset.

    Important:
    Sequences are created separately per ticker.
    This avoids mixing AAPL data with MSFT data inside one sequence.
    """
    X_sequences = []
    y_values = []
    metadata_rows = []

    for ticker, ticker_data in data.groupby("Ticker"):
        ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)

        if len(ticker_data) < sequence_length:
            print(f"Skipping {ticker}: not enough rows.")
            continue

        for index in range(sequence_length - 1, len(ticker_data)):
            start_index = index - sequence_length + 1
            end_index = index + 1

            sequence = ticker_data.iloc[start_index:end_index][feature_columns].values

            target = ticker_data.iloc[index][target_column]

            X_sequences.append(sequence)
            y_values.append(target)

            metadata = {
                "Date": ticker_data.iloc[index]["Date"],
                "Ticker": ticker,
                "Close": ticker_data.iloc[index]["Close"],
                "Target": target,
            }

            if "Next_Close" in ticker_data.columns:
                metadata["Next_Close"] = ticker_data.iloc[index]["Next_Close"]

            metadata_rows.append(metadata)

    if len(X_sequences) == 0:
        raise ValueError(
            "No sequences were created. Check sequence_length and dataset size."
        )

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_values, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)

    return X, y, metadata


def main():
    print("Creating LSTM sequences")
    print("=======================")

    train_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TRAIN_FILE"]
    validation_file = EXPERIMENT_ROOT / PARAMS["DATA"]["VALIDATION_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"]
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA"]["FEATURE_PATH"]

    sequence_file = EXPERIMENT_ROOT / PARAMS["DATA"]["SEQUENCE_FILE"]
    scaler_file = EXPERIMENT_ROOT / PARAMS["DATA"]["SCALER_FILE"]
    test_metadata_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_METADATA_FILE"]

    target_column = PARAMS["MODEL"]["TARGET"]
    sequence_length = PARAMS["MODEL"]["SEQUENCE_LENGTH"]

    feature_columns = load_feature_columns(feature_path)

    train_data = load_dataset(train_file)
    validation_data = load_dataset(validation_file)
    test_data = load_dataset(test_file)

    check_required_columns(train_data, feature_columns, target_column)
    check_required_columns(validation_data, feature_columns, target_column)
    check_required_columns(test_data, feature_columns, target_column)

    check_feature_values(train_data, feature_columns, "Train data")
    check_feature_values(validation_data, feature_columns, "Validation data")
    check_feature_values(test_data, feature_columns, "Test data")

    train_data, validation_data, test_data = scale_features(
        train_data,
        validation_data,
        test_data,
        feature_columns,
        scaler_file,
    )

    print("\nFeature normalization:")
    print("StandardScaler fitted on train data only.")
    print("Validation and test data transformed with the train scaler.")
    print(f"Scaler saved to: {scaler_file}")

    X_train, y_train, _ = create_sequences_for_dataset(
        train_data,
        feature_columns,
        target_column,
        sequence_length,
    )

    X_validation, y_validation, _ = create_sequences_for_dataset(
        validation_data,
        feature_columns,
        target_column,
        sequence_length,
    )

    X_test, y_test, test_metadata = create_sequences_for_dataset(
        test_data,
        feature_columns,
        target_column,
        sequence_length,
    )

    sequence_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        sequence_file,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        feature_columns=np.array(feature_columns),
    )

    test_metadata_file.parent.mkdir(parents=True, exist_ok=True)
    test_metadata.to_csv(test_metadata_file, index=False)

    print("\nSequence file saved to:")
    print(sequence_file)

    print("\nTest metadata saved to:")
    print(test_metadata_file)

    print("\nShapes:")
    print(f"X_train:      {X_train.shape}")
    print(f"y_train:      {y_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"y_validation: {y_validation.shape}")
    print(f"X_test:       {X_test.shape}")
    print(f"y_test:       {y_test.shape}")

    print("\nSequence creation completed successfully.")


if __name__ == "__main__":
    main()
