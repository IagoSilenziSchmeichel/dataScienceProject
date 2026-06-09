"""
Train/Validation/Test Split.

Splits the data chronologically by date. This avoids data leakage because the
model never trains on data from the future.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml
from formatting import save_csv


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def load_feature_columns():
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["FEATURE_PATH"]
    return [line.strip() for line in open(feature_path) if line.strip()]


def calculate_train_stats(train_data, feature_columns):
    stats = {}

    for ticker, ticker_train_data in train_data.groupby("Ticker"):
        means = ticker_train_data[feature_columns].mean()
        stds = ticker_train_data[feature_columns].std()
        stds = stds.replace(0, 1).fillna(1)
        stats[ticker] = {"mean": means, "std": stds}

    global_means = train_data[feature_columns].mean()
    global_stds = train_data[feature_columns].std().replace(0, 1).fillna(1)
    stats["__GLOBAL__"] = {"mean": global_means, "std": global_stds}

    return stats


def normalize_with_train_stats(data, feature_columns, stats):
    normalized_data = data.copy()

    for ticker, ticker_data in normalized_data.groupby("Ticker"):
        ticker_stats = stats.get(ticker, stats["__GLOBAL__"])
        row_mask = normalized_data["Ticker"] == ticker
        normalized_data.loc[row_mask, feature_columns] = (
            ticker_data[feature_columns] - ticker_stats["mean"]
        ) / ticker_stats["std"]

    return normalized_data


def main():
    feature_data_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["FEATURE_DATA_FILE"]
    train_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["TRAIN_FILE"]
    validation_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["VALIDATION_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["TEST_FILE"]

    train_ratio = PARAMS["DATA_PREP"]["TRAIN_RATIO"]
    validation_ratio = PARAMS["DATA_PREP"]["VALIDATION_RATIO"]

    data = pd.read_csv(feature_data_file, parse_dates=["Date"])
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    unique_dates = sorted(data["Date"].unique())
    train_end = int(len(unique_dates) * train_ratio)
    validation_end = int(len(unique_dates) * (train_ratio + validation_ratio))

    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end:validation_end]
    test_dates = unique_dates[validation_end:]

    train_data = data[data["Date"].isin(train_dates)].copy()
    validation_data = data[data["Date"].isin(validation_dates)].copy()
    test_data = data[data["Date"].isin(test_dates)].copy()

    feature_columns = load_feature_columns()
    train_stats = calculate_train_stats(train_data, feature_columns)
    train_data = normalize_with_train_stats(train_data, feature_columns, train_stats)
    validation_data = normalize_with_train_stats(validation_data, feature_columns, train_stats)
    test_data = normalize_with_train_stats(test_data, feature_columns, train_stats)

    train_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(train_data, train_file)
    save_csv(validation_data, validation_file)
    save_csv(test_data, test_file)

    print("Chronological split saved.")
    print("Features normalized per ticker using training data only.")
    print(f"Train rows:      {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Test rows:       {len(test_data)}")

    print("\nDate ranges:")
    print(f"Train:      {train_data['Date'].min().date()} to {train_data['Date'].max().date()}")
    print(f"Validation: {validation_data['Date'].min().date()} to {validation_data['Date'].max().date()}")
    print(f"Test:       {test_data['Date'].min().date()} to {test_data['Date'].max().date()}")


if __name__ == "__main__":
    main()
