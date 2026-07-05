"""
Export the final Outperformance-LSTM for Alpaca Paper Trading.

This script does not search for a new model. It trains the selected final
variant once and saves the exact files that the Alpaca signal pipeline expects:

- Outperformance-LSTM
- Features: Technical + Relative Strength
- Target: stock beats QQQ on the next trading day
- Strategy after inference: Top-K, initially Top 1
"""

from datetime import timedelta
from pathlib import Path
import pickle
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import yaml
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml", encoding="utf-8"))

FEATURE_FILE = EXPERIMENT_ROOT / "conf" / "outperformance_alpaca_features.txt"
MODEL_FILE = EXPERIMENT_ROOT / "models" / "outperformance_lstm_model.pth"
SCALER_FILE = EXPERIMENT_ROOT / "models" / "outperformance_lstm_scaler.pkl"
SUMMARY_FILE = EXPERIMENT_ROOT / "data" / "processed" / "lstm_outperformance_alpaca_export_summary.csv"

TARGET_COLUMN = "Outperform_QQQ_Target"


class LSTMClassifier(nn.Module):
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


def load_feature_columns():
    return [line.strip() for line in open(FEATURE_FILE, encoding="utf-8") if line.strip()]


def load_split(file_path, split_name):
    data = pd.read_csv(file_path, parse_dates=["Date"])
    data["Split"] = split_name

    return data.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def download_qqq_data(start_date, end_date):
    qqq_ticker = PARAMS["MARKET"]["QQQ_TICKER"]
    print(f"Downloading benchmark data for {qqq_ticker}...")
    data = yf.download(
        qqq_ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError("No QQQ data downloaded. QQQ is required for the outperformance target.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[["Date", "Close"]].rename(columns={"Close": "QQQ_Close"})
    data = data.sort_values("Date").reset_index(drop=True)
    data["QQQ_Return"] = data["QQQ_Close"].pct_change()
    data["QQQ_Momentum_20"] = data["QQQ_Close"] / data["QQQ_Close"].shift(20) - 1
    data["Next_Day_QQQ_Return"] = data["QQQ_Close"].shift(-1) / data["QQQ_Close"] - 1

    return data


def add_outperformance_columns(data, qqq_data):
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    data["Stock_Daily_Return_Raw"] = data.groupby("Ticker")["Close"].pct_change()
    data["Stock_Momentum_20_Raw"] = data.groupby("Ticker")["Close"].transform(
        lambda values: values / values.shift(20) - 1
    )

    merged = data.merge(qqq_data, on="Date", how="left")
    merged["Relative_Return_QQQ"] = merged["Stock_Daily_Return_Raw"] - merged["QQQ_Return"]
    merged["Relative_Momentum_20_QQQ"] = merged["Stock_Momentum_20_Raw"] - merged["QQQ_Momentum_20"]

    # The next-day QQQ return is used only for the target, never as a feature.
    merged[TARGET_COLUMN] = (merged["Future_Return"] > merged["Next_Day_QQQ_Return"]).astype(int)

    return merged


def scale_data(train_data, validation_data, test_data, feature_columns):
    scaler = StandardScaler()
    train_scaled = train_data.copy()
    validation_scaled = validation_data.copy()
    test_scaled = test_data.copy()

    train_scaled[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    validation_scaled[feature_columns] = scaler.transform(validation_data[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test_data[feature_columns])

    return train_scaled, validation_scaled, test_scaled, scaler


def create_sequences(data, feature_columns, target_column, sequence_length):
    X_sequences = []
    y_values = []

    for ticker, ticker_data in data.groupby("Ticker"):
        ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)

        for index in range(sequence_length - 1, len(ticker_data)):
            start_index = index - sequence_length + 1
            end_index = index + 1
            X_sequences.append(ticker_data.iloc[start_index:end_index][feature_columns].values)
            y_values.append(ticker_data.iloc[index][target_column])

    if not X_sequences:
        raise ValueError("No sequences created for the final Outperformance-LSTM export.")

    return np.array(X_sequences, dtype=np.float32), np.array(y_values, dtype=np.float32)


def create_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_probabilities(model, loader, device):
    model.eval()
    targets = []
    probabilities = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(y_batch.numpy())

    return np.array(targets).astype(int), np.array(probabilities)


def train_final_model(X_train, y_train, X_validation, y_validation):
    train_loader = create_loader(X_train, y_train, PARAMS["MODEL"]["BATCH_SIZE"], shuffle=True)
    validation_loader = create_loader(X_validation, y_validation, PARAMS["MODEL"]["BATCH_SIZE"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_size=X_train.shape[2],
        hidden_size=PARAMS["MODEL"]["HIDDEN_SIZE"],
        num_layers=PARAMS["MODEL"]["NUM_LAYERS"],
        dropout=PARAMS["MODEL"]["DROPOUT"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["MODEL"]["LEARNING_RATE"])

    best_validation_f1 = -1.0
    best_state_dict = None

    for epoch in range(1, PARAMS["MODEL"]["EPOCHS"] + 1):
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

        y_true, probabilities = evaluate_probabilities(model, validation_loader, device)
        predictions = (probabilities >= PARAMS["MODEL"]["PREDICTION_THRESHOLD"]).astype(int)
        validation_f1 = f1_score(y_true, predictions, zero_division=0)

        print(
            f"Epoch {epoch:02d}/{PARAMS['MODEL']['EPOCHS']} | "
            f"Loss: {total_loss / len(train_loader):.5f} | "
            f"Validation F1: {validation_f1:.5f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state_dict = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_validation_f1


def main():
    print("Export final Outperformance-LSTM for Alpaca")
    print("===========================================")
    set_random_seed(PARAMS["MODEL"]["RANDOM_STATE"])

    feature_columns = load_feature_columns()
    print("Final feature set:")
    for feature in feature_columns:
        print(f"- {feature}")

    train_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["TRAIN_FILE"], "train")
    validation_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["VALIDATION_FILE"], "validation")
    test_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"], "test")
    all_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

    start_date = all_data["Date"].min() - timedelta(days=60)
    end_date = all_data["Date"].max() + timedelta(days=10)
    qqq_data = download_qqq_data(start_date.date().isoformat(), end_date.date().isoformat())
    data = add_outperformance_columns(all_data, qqq_data)

    required_columns = feature_columns + [TARGET_COLUMN, "Future_Return"]
    data = data.dropna(subset=required_columns).copy()

    train_data = data[data["Split"] == "train"].copy()
    validation_data = data[data["Split"] == "validation"].copy()
    test_data = data[data["Split"] == "test"].copy()

    train_data, validation_data, test_data, scaler = scale_data(
        train_data,
        validation_data,
        test_data,
        feature_columns,
    )

    sequence_length = PARAMS["MODEL"]["SEQUENCE_LENGTH"]
    X_train, y_train = create_sequences(train_data, feature_columns, TARGET_COLUMN, sequence_length)
    X_validation, y_validation = create_sequences(validation_data, feature_columns, TARGET_COLUMN, sequence_length)
    X_test, y_test = create_sequences(test_data, feature_columns, TARGET_COLUMN, sequence_length)

    print("\nShapes:")
    print(f"X_train:      {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test:       {X_test.shape}")

    model, best_validation_f1 = train_final_model(X_train, y_train, X_validation, y_validation)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "outperformance_lstm",
            "target": TARGET_COLUMN,
            "benchmark": "QQQ",
            "feature_group": "Technical + Relative Strength",
            "feature_columns": feature_columns,
            "sequence_length": sequence_length,
            "hidden_size": PARAMS["MODEL"]["HIDDEN_SIZE"],
            "num_layers": PARAMS["MODEL"]["NUM_LAYERS"],
            "dropout": PARAMS["MODEL"]["DROPOUT"],
            "learning_rate": PARAMS["MODEL"]["LEARNING_RATE"],
            "best_validation_f1": best_validation_f1,
        },
        MODEL_FILE,
    )

    with open(SCALER_FILE, "wb") as file:
        pickle.dump(scaler, file)

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_type": "outperformance_lstm",
                "feature_group": "Technical + Relative Strength",
                "benchmark": "QQQ",
                "strategy": "Top-K",
                "top_k_default": 1,
                "target": TARGET_COLUMN,
                "feature_count": len(feature_columns),
                "sequence_length": sequence_length,
                "best_validation_f1": best_validation_f1,
                "model_file": str(MODEL_FILE.relative_to(EXPERIMENT_ROOT)),
                "scaler_file": str(SCALER_FILE.relative_to(EXPERIMENT_ROOT)),
                "feature_file": str(FEATURE_FILE.relative_to(EXPERIMENT_ROOT)),
            }
        ]
    ).to_csv(SUMMARY_FILE, index=False)

    print("\nSaved final Alpaca inference files:")
    print(f"- {MODEL_FILE}")
    print(f"- {SCALER_FILE}")
    print(f"- {FEATURE_FILE}")
    print(f"- {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
