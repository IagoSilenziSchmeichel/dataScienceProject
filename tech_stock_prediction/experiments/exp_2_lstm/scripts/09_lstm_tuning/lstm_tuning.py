"""
Tune simple LSTM hyperparameters.

This script compares several LSTM configurations on the validation set.
We use the validation set because the test set should stay untouched for the
final honest evaluation.

The full grid has 72 combinations and can take a long time. Therefore the
default setting uses a smaller, representative selection. Set RUN_FULL_GRID to
True if the team wants to test every combination.
"""

from itertools import product
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from formatting import format_decimal, format_percent, save_csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

SEQUENCE_LENGTHS = [10, 20, 30, 60]
HIDDEN_SIZES = [32, 64, 128]
DROPOUTS = [0.1, 0.2, 0.3]
LEARNING_RATES = [0.001, 0.0005]

# Full grid = 4 * 3 * 3 * 2 = 72 trainings. That is useful, but slow.
RUN_FULL_GRID = False

# Small default selection so the experiment stays runnable for presentations.
SMALL_CONFIGS = [
    {"sequence_length": 10, "hidden_size": 32, "dropout": 0.1, "learning_rate": 0.001},
    {"sequence_length": 10, "hidden_size": 64, "dropout": 0.2, "learning_rate": 0.001},
    {"sequence_length": 20, "hidden_size": 64, "dropout": 0.2, "learning_rate": 0.001},
    {"sequence_length": 20, "hidden_size": 128, "dropout": 0.3, "learning_rate": 0.0005},
    {"sequence_length": 30, "hidden_size": 64, "dropout": 0.2, "learning_rate": 0.0005},
    {"sequence_length": 30, "hidden_size": 128, "dropout": 0.3, "learning_rate": 0.001},
    {"sequence_length": 60, "hidden_size": 64, "dropout": 0.2, "learning_rate": 0.0005},
    {"sequence_length": 60, "hidden_size": 128, "dropout": 0.3, "learning_rate": 0.0005},
]


class LSTMClassifier(nn.Module):
    """
    Simple LSTM model for binary classification.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_output, (hidden_state, cell_state) = self.lstm(x)
        last_hidden_state = hidden_state[-1]
        logits = self.output_layer(last_hidden_state)

        return logits.squeeze(1)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_feature_columns(feature_path):
    feature_columns = []

    with open(feature_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line and not line.startswith("#"):
                feature_columns.append(line)

    return feature_columns


def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return data


def check_columns(data, feature_columns, target_column, dataset_name):
    required_columns = feature_columns + [target_column, "Date", "Ticker", "Close"]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"{dataset_name} is missing columns: {missing_columns}")


def scale_features(train_data, validation_data, feature_columns):
    """
    Scale features for the tuning run without data leakage.

    The scaler is fitted only on the training data. The validation data is only
    transformed with that fitted scaler, because validation is used for model
    selection.
    """
    scaler = StandardScaler()

    train_scaled = train_data.copy()
    validation_scaled = validation_data.copy()

    train_scaled[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    validation_scaled[feature_columns] = scaler.transform(validation_data[feature_columns])

    return train_scaled, validation_scaled


def create_sequences(data, feature_columns, target_column, sequence_length):
    X_sequences = []
    y_values = []
    metadata_rows = []

    for ticker, ticker_data in data.groupby("Ticker"):
        ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)
        ticker_data["Next_Close"] = ticker_data["Close"].shift(-1)

        for index in range(sequence_length - 1, len(ticker_data)):
            start_index = index - sequence_length + 1
            end_index = index + 1

            sequence = ticker_data.iloc[start_index:end_index][feature_columns].values
            target = ticker_data.iloc[index][target_column]

            X_sequences.append(sequence)
            y_values.append(target)

            metadata_rows.append(
                {
                    "Date": ticker_data.iloc[index]["Date"],
                    "Ticker": ticker,
                    "Close": ticker_data.iloc[index]["Close"],
                    "Next_Close": ticker_data.iloc[index]["Next_Close"],
                }
            )

    if len(X_sequences) == 0:
        raise ValueError(f"No sequences created for sequence_length={sequence_length}")

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_values, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)

    return X, y, metadata


def create_loader(X, y, batch_size, shuffle):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_probabilities(model, loader, device):
    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()

            probabilities.extend(batch_probabilities)
            targets.extend(y_batch.numpy())

    return np.array(targets).astype(int), np.array(probabilities)


def train_one_model(X_train, y_train, X_validation, y_validation, config, device):
    batch_size = PARAMS["MODEL"]["BATCH_SIZE"]
    epochs = min(PARAMS["MODEL"]["EPOCHS"], 8)
    num_layers = PARAMS["MODEL"]["NUM_LAYERS"]

    train_loader = create_loader(X_train, y_train, batch_size, shuffle=True)
    validation_loader = create_loader(X_validation, y_validation, batch_size, shuffle=False)

    model = LSTMClassifier(
        input_size=X_train.shape[2],
        hidden_size=config["hidden_size"],
        num_layers=num_layers,
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    y_true, probabilities = get_probabilities(model, validation_loader, device)

    return y_true, probabilities


def calculate_backtest(metadata, predictions):
    result_data = metadata.copy()
    result_data["Prediction"] = predictions
    result_data = result_data.dropna(subset=["Next_Close"]).copy()

    result_data["Future_Return"] = result_data["Next_Close"] / result_data["Close"] - 1
    result_data["Strategy_Return"] = np.where(
        result_data["Prediction"] == 1,
        result_data["Future_Return"],
        0.0,
    )
    result_data["Buy_And_Hold_Return"] = result_data["Future_Return"]

    daily_returns = (
        result_data
        .groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]]
        .mean()
    )

    strategy_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
    buy_and_hold_return = (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1

    return strategy_return, buy_and_hold_return


def build_configurations():
    if not RUN_FULL_GRID:
        return SMALL_CONFIGS

    configurations = []

    for sequence_length, hidden_size, dropout, learning_rate in product(
        SEQUENCE_LENGTHS,
        HIDDEN_SIZES,
        DROPOUTS,
        LEARNING_RATES,
    ):
        configurations.append(
            {
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "learning_rate": learning_rate,
            }
        )

    return configurations


def print_results(results):
    print("\nLSTM tuning results")
    print("===================")
    print(
        results.to_string(
            index=False,
            formatters={
                "dropout": format_decimal,
                "learning_rate": format_decimal,
                "accuracy": format_decimal,
                "precision": format_decimal,
                "recall": format_decimal,
                "f1_score": format_decimal,
                "predicted_up_share": format_decimal,
                "strategy_return": format_percent,
                "buy_and_hold_return": format_percent,
                "difference": format_percent,
            },
        )
    )

    best_f1 = results.sort_values("f1_score", ascending=False).iloc[0]
    best_return = results.sort_values("strategy_return", ascending=False).iloc[0]

    print("\nBest configuration by F1-Score:")
    print(best_f1.to_string())

    print("\nBest configuration by Strategy Return:")
    print(best_return.to_string())


def main():
    print("LSTM Hyperparameter Tuning")
    print("==========================")

    set_random_seed(PARAMS["MODEL"]["RANDOM_STATE"])

    train_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TRAIN_FILE"]
    validation_file = EXPERIMENT_ROOT / PARAMS["DATA"]["VALIDATION_FILE"]
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA"]["FEATURE_PATH"]
    output_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["TUNING_RESULTS_FILE"]

    target_column = PARAMS["MODEL"]["TARGET"]
    feature_columns = load_feature_columns(feature_path)

    train_data = load_dataset(train_file)
    validation_data = load_dataset(validation_file)

    check_columns(train_data, feature_columns, target_column, "Train data")
    check_columns(validation_data, feature_columns, target_column, "Validation data")

    train_data, validation_data = scale_features(
        train_data,
        validation_data,
        feature_columns,
    )

    print("Feature scaling: StandardScaler fitted on train data only.")

    configurations = build_configurations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Configurations to test: {len(configurations)}")

    rows = []

    for number, config in enumerate(configurations, start=1):
        print("\n" + "-" * 80)
        print(f"Configuration {number}/{len(configurations)}: {config}")

        X_train, y_train, _ = create_sequences(
            train_data,
            feature_columns,
            target_column,
            config["sequence_length"],
        )
        X_validation, y_validation, validation_metadata = create_sequences(
            validation_data,
            feature_columns,
            target_column,
            config["sequence_length"],
        )

        y_true, probabilities = train_one_model(
            X_train,
            y_train,
            X_validation,
            y_validation,
            config,
            device,
        )

        predictions = (probabilities >= 0.5).astype(int)

        strategy_return, buy_and_hold_return = calculate_backtest(
            validation_metadata,
            predictions,
        )

        row = {
            "sequence_length": config["sequence_length"],
            "hidden_size": config["hidden_size"],
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1_score": f1_score(y_true, predictions, zero_division=0),
            "predicted_up_share": predictions.mean(),
            "strategy_return": strategy_return,
            "buy_and_hold_return": buy_and_hold_return,
            "difference": strategy_return - buy_and_hold_return,
        }

        rows.append(row)

        print(
            f"F1: {row['f1_score']:.5f} | "
            f"Strategy Return: {row['strategy_return']:.5%} | "
            f"Predicted-Up-Share: {row['predicted_up_share']:.5f}"
        )

    results = pd.DataFrame(rows)
    results = results.sort_values("f1_score", ascending=False).reset_index(drop=True)

    save_csv(results, output_file)
    print_results(results)

    print(f"\nTuning results saved to: {output_file}")


if __name__ == "__main__":
    main()
