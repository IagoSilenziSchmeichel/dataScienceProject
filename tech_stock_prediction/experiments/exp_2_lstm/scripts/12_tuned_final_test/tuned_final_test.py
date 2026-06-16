"""
Final test for tuned LSTM configurations.

The tuning script compares hyperparameters on the validation set. This script
takes the best validation configurations and evaluates them on the test set.

Important:
- Hyperparameters are selected from validation results.
- The final model is trained on train + validation data.
- The test set is used only once for the final result.
"""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

TOP_K_VALUES = PARAMS["MODEL"].get("TOP_K_VALUES", [1, 2, 3, 4, 5])


class LSTMClassifier(nn.Module):
    """
    Same simple LSTM structure as the main LSTM model.
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


def select_configurations(tuning_results):
    """
    Select the best config by validation F1 and by validation strategy return.
    Sometimes these are different, so we test both honestly on the test set.
    """
    best_f1 = tuning_results.sort_values("f1_score", ascending=False).iloc[0]
    best_return = tuning_results.sort_values("strategy_return", ascending=False).iloc[0]

    selected_rows = [
        ("best_validation_f1", best_f1),
        ("best_validation_strategy_return", best_return),
    ]

    configs = []
    seen = set()

    for name, row in selected_rows:
        key = (
            int(row["sequence_length"]),
            int(row["hidden_size"]),
            float(row["dropout"]),
            float(row["learning_rate"]),
        )

        if key in seen:
            continue

        seen.add(key)

        configs.append(
            {
                "selection_method": name,
                "sequence_length": key[0],
                "hidden_size": key[1],
                "dropout": key[2],
                "learning_rate": key[3],
                "validation_accuracy": row["accuracy"],
                "validation_precision": row["precision"],
                "validation_recall": row["recall"],
                "validation_f1_score": row["f1_score"],
                "validation_strategy_return": row["strategy_return"],
            }
        )

    return configs


def scale_features(final_train_data, test_data, feature_columns):
    """
    Fit scaler only on final train data.

    Here final train data means original train + validation. This is okay
    because validation has already been used for model selection and is before
    the test period chronologically.
    """
    scaler = StandardScaler()

    final_train_scaled = final_train_data.copy()
    test_scaled = test_data.copy()

    final_train_scaled[feature_columns] = scaler.fit_transform(final_train_data[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test_data[feature_columns])

    return final_train_scaled, test_scaled


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

            X_sequences.append(ticker_data.iloc[start_index:end_index][feature_columns].values)
            y_values.append(ticker_data.iloc[index][target_column])
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


def train_model(X_train, y_train, config, device):
    batch_size = PARAMS["MODEL"]["BATCH_SIZE"]
    epochs = PARAMS["MODEL"]["EPOCHS"]
    num_layers = PARAMS["MODEL"]["NUM_LAYERS"]

    train_loader = create_loader(X_train, y_train, batch_size, shuffle=True)

    model = LSTMClassifier(
        input_size=X_train.shape[2],
        hidden_size=config["hidden_size"],
        num_layers=num_layers,
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {average_loss:.5f}")

    return model


def predict(model, X_test, y_test, batch_size, device):
    test_loader = create_loader(X_test, y_test, batch_size, shuffle=False)

    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()

            probabilities.extend(batch_probabilities)
            targets.extend(y_batch.numpy())

    return np.array(targets).astype(int), np.array(probabilities)


def calculate_simple_backtest(predictions):
    data = predictions.dropna(subset=["Next_Close"]).copy()
    data["Future_Return"] = data["Next_Close"] / data["Close"] - 1
    data["Strategy_Return"] = np.where(data["Prediction"] == 1, data["Future_Return"], 0.0)
    data["Buy_And_Hold_Return"] = data["Future_Return"]

    daily_returns = data.groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]].mean()

    strategy_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
    buy_and_hold_return = (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1

    return (
        strategy_return,
        buy_and_hold_return,
        daily_returns.index.min().date().isoformat(),
        daily_returns.index.max().date().isoformat(),
        len(daily_returns),
    )


def calculate_top_k_results(predictions, selection_method):
    data = predictions.dropna(subset=["Next_Close"]).copy()
    data["Future_Return"] = data["Next_Close"] / data["Close"] - 1

    rows = []

    for top_k in TOP_K_VALUES:
        ranked = data.copy()
        ranked["Probability_Rank"] = (
            ranked
            .groupby("Date")["Probability"]
            .rank(method="first", ascending=False)
        )
        selected = ranked[ranked["Probability_Rank"] <= top_k].copy()

        daily_strategy_returns = selected.groupby("Date")["Future_Return"].mean()
        daily_buy_hold_returns = ranked.groupby("Date")["Future_Return"].mean()
        daily_positions = selected.groupby("Date")["Ticker"].count()

        daily_results = pd.concat(
            [
                daily_strategy_returns.rename("Strategy_Return"),
                daily_buy_hold_returns.rename("Buy_And_Hold_Return"),
                daily_positions.rename("Number_Of_Positions"),
            ],
            axis=1,
        ).fillna(0)

        strategy_return = (1 + daily_results["Strategy_Return"]).prod() - 1
        buy_and_hold_return = (1 + daily_results["Buy_And_Hold_Return"]).prod() - 1

        if strategy_return > 1.0:
            print(
                "Warning: Top-K result is very high. "
                "Check whether this result is dominated by a short period or single stock."
            )

        rows.append(
            {
                "selection_method": selection_method,
                "top_k": top_k,
                "test_start": daily_results.index.min().date().isoformat(),
                "test_end": daily_results.index.max().date().isoformat(),
                "strategy_return": strategy_return,
                "buy_and_hold_return": buy_and_hold_return,
                "difference": strategy_return - buy_and_hold_return,
                "average_number_of_positions": daily_results["Number_Of_Positions"].mean(),
                "number_of_trading_days": len(daily_results),
            }
        )

    return rows


def save_predictions(predictions, selection_method):
    safe_name = selection_method.replace(" ", "_")
    output_file = EXPERIMENT_ROOT / "data" / "processed" / f"lstm_tuned_test_predictions_{safe_name}.csv"
    save_csv(predictions, output_file)
    print(f"Predictions saved to: {output_file}")


def print_results(final_results, top_k_results):
    print("\nTuned LSTM final test results")
    print("=============================")
    print(
        final_results.to_string(
            index=False,
            formatters={
                "validation_accuracy": format_decimal,
                "validation_precision": format_decimal,
                "validation_recall": format_decimal,
                "validation_f1_score": format_decimal,
                "validation_strategy_return": format_percent,
                "test_accuracy": format_decimal,
                "test_precision": format_decimal,
                "test_recall": format_decimal,
                "test_f1_score": format_decimal,
                "test_predicted_up_share": format_decimal,
                "simple_strategy_return": format_percent,
                "buy_and_hold_return": format_percent,
                "simple_difference": format_percent,
            },
        )
    )

    print("\nTuned LSTM Top-K test results")
    print("=============================")
    print(
        top_k_results.to_string(
            index=False,
            formatters={
                "strategy_return": format_percent,
                "buy_and_hold_return": format_percent,
                "difference": format_percent,
                "average_number_of_positions": format_decimal,
            },
        )
    )


def main():
    print("Tuned LSTM Final Test")
    print("=====================")

    set_random_seed(PARAMS["MODEL"]["RANDOM_STATE"])

    train_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TRAIN_FILE"]
    validation_file = EXPERIMENT_ROOT / PARAMS["DATA"]["VALIDATION_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"]
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA"]["FEATURE_PATH"]
    tuning_file = EXPERIMENT_ROOT / "data" / "processed" / "lstm_tuning_results.csv"

    final_results_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["TUNED_FINAL_RESULTS_FILE"]
    top_k_results_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["TUNED_TOP_K_RESULTS_FILE"]

    if not tuning_file.exists():
        raise FileNotFoundError(
            f"Tuning results not found: {tuning_file}. "
            "Run scripts/09_lstm_tuning/lstm_tuning.py first."
        )

    target_column = PARAMS["MODEL"]["TARGET"]
    feature_columns = load_feature_columns(feature_path)

    train_data = load_dataset(train_file)
    validation_data = load_dataset(validation_file)
    test_data = load_dataset(test_file)

    check_columns(train_data, feature_columns, target_column, "Train data")
    check_columns(validation_data, feature_columns, target_column, "Validation data")
    check_columns(test_data, feature_columns, target_column, "Test data")

    # After hyperparameter selection, validation can be used for final training.
    final_train_data = (
        pd.concat([train_data, validation_data], ignore_index=True)
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    tuning_results = pd.read_csv(tuning_file)
    selected_configs = select_configurations(tuning_results)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = PARAMS["MODEL"]["BATCH_SIZE"]
    threshold = PARAMS["MODEL"].get("PREDICTION_THRESHOLD", 0.50)

    final_rows = []
    top_k_rows = []

    for config in selected_configs:
        print("\n" + "-" * 80)
        print(f"Selection method: {config['selection_method']}")
        print(
            "Config: "
            f"sequence_length={config['sequence_length']}, "
            f"hidden_size={config['hidden_size']}, "
            f"dropout={config['dropout']}, "
            f"learning_rate={config['learning_rate']}"
        )

        scaled_train, scaled_test = scale_features(
            final_train_data,
            test_data,
            feature_columns,
        )

        X_train, y_train, _ = create_sequences(
            scaled_train,
            feature_columns,
            target_column,
            config["sequence_length"],
        )
        X_test, y_test, test_metadata = create_sequences(
            scaled_test,
            feature_columns,
            target_column,
            config["sequence_length"],
        )

        print(f"Final train shape: {X_train.shape}")
        print(f"Test shape:        {X_test.shape}")

        model = train_model(X_train, y_train, config, device)
        y_true, probabilities = predict(model, X_test, y_test, batch_size, device)
        predictions = (probabilities >= threshold).astype(int)

        prediction_data = test_metadata.copy()
        prediction_data["Actual"] = y_true
        prediction_data["Probability"] = probabilities
        prediction_data["Prediction"] = predictions

        save_predictions(prediction_data, config["selection_method"])

        (
            strategy_return,
            buy_and_hold_return,
            test_start,
            test_end,
            number_of_trading_days,
        ) = calculate_simple_backtest(prediction_data)

        matrix = confusion_matrix(y_true, predictions)

        final_rows.append(
            {
                **config,
                "model_name": f"tuned_lstm_{config['selection_method']}",
                "num_layers": PARAMS["MODEL"]["NUM_LAYERS"],
                "threshold": threshold,
                "test_start": test_start,
                "test_end": test_end,
                "number_of_trading_days": number_of_trading_days,
                "total_predictions": len(predictions),
                "test_accuracy": accuracy_score(y_true, predictions),
                "test_precision": precision_score(y_true, predictions, zero_division=0),
                "test_recall": recall_score(y_true, predictions, zero_division=0),
                "test_f1_score": f1_score(y_true, predictions, zero_division=0),
                "test_predicted_up_share": predictions.mean(),
                "confusion_true_0_pred_0": matrix[0, 0],
                "confusion_true_0_pred_1": matrix[0, 1],
                "confusion_true_1_pred_0": matrix[1, 0],
                "confusion_true_1_pred_1": matrix[1, 1],
                "simple_strategy_return": strategy_return,
                "buy_and_hold_return": buy_and_hold_return,
                "simple_difference": strategy_return - buy_and_hold_return,
            }
        )

        top_k_rows.extend(calculate_top_k_results(prediction_data, config["selection_method"]))

    final_results = pd.DataFrame(final_rows)
    top_k_results = pd.DataFrame(top_k_rows)

    save_csv(final_results, final_results_file)
    save_csv(top_k_results, top_k_results_file)

    print_results(final_results, top_k_results)

    print(f"\nFinal tuned results saved to: {final_results_file}")
    print(f"Tuned Top-K results saved to: {top_k_results_file}")


if __name__ == "__main__":
    main()
