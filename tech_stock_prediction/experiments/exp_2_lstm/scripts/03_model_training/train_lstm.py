"""
Train LSTM model.

The LSTM receives sequences of past stock features and predicts whether the
stock will go up or not.

Input shape:
(batch_size, sequence_length, number_of_features)

Example:
(32, 30, 19)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


class LSTMClassifier(nn.Module):
    """
    Simple LSTM model for binary classification.

    The model reads a sequence of stock features and outputs one value.
    This value is later converted into a probability with sigmoid.
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

        # Take the last hidden state from the last LSTM layer
        last_hidden_state = hidden_state[-1]

        logits = self.output_layer(last_hidden_state)

        return logits.squeeze(1)


def set_random_seed(seed):
    """
    Set random seeds to make results more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_data_loader(X, y, batch_size, shuffle):
    """
    Convert numpy arrays to PyTorch DataLoader.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on validation data.
    """
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).int().cpu().numpy()

            all_predictions.extend(predictions)
            all_targets.extend(y_batch.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)

    return accuracy, precision, recall, f1


def main():
    print("Training LSTM model")
    print("===================")

    seed = PARAMS["MODEL"]["RANDOM_STATE"]
    set_random_seed(seed)

    sequence_file = EXPERIMENT_ROOT / PARAMS["DATA"]["SEQUENCE_FILE"]
    model_file = EXPERIMENT_ROOT / PARAMS["MODEL"]["MODEL_FILE"]

    hidden_size = PARAMS["MODEL"]["HIDDEN_SIZE"]
    num_layers = PARAMS["MODEL"]["NUM_LAYERS"]
    dropout = PARAMS["MODEL"]["DROPOUT"]
    epochs = PARAMS["MODEL"]["EPOCHS"]
    batch_size = PARAMS["MODEL"]["BATCH_SIZE"]
    learning_rate = PARAMS["MODEL"]["LEARNING_RATE"]

    data = np.load(sequence_file, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_validation = data["X_validation"]
    y_validation = data["y_validation"]

    feature_columns = data["feature_columns"]

    input_size = X_train.shape[2]

    print(f"Train shape:      {X_train.shape}")
    print(f"Validation shape: {X_validation.shape}")
    print(f"Input size:       {input_size}")
    print(f"Features:         {len(feature_columns)}")
    print(f"Hidden size:      {hidden_size}")
    print(f"LSTM layers:      {num_layers}")

    if num_layers == 1:
        print("Dropout:          disabled inside LSTM because NUM_LAYERS = 1")
    else:
        print(f"Dropout:          {dropout}")

    train_loader = create_data_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = create_data_loader(
        X_validation,
        y_validation,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start with -1.0 so that the first model is always saved,
    # even if the first validation F1-score is 0.0.
    best_validation_f1 = -1.0

    training_history = []

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

            # Prevent exploding gradients, which can happen in LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        average_train_loss = total_loss / len(train_loader)

        validation_accuracy, validation_precision, validation_recall, validation_f1 = evaluate_model(
            model,
            validation_loader,
            device,
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "validation_accuracy": validation_accuracy,
                "validation_precision": validation_precision,
                "validation_recall": validation_recall,
                "validation_f1": validation_f1,
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {average_train_loss:.5f} | "
            f"Val Acc: {validation_accuracy:.5f} | "
            f"Val Precision: {validation_precision:.5f} | "
            f"Val Recall: {validation_recall:.5f} | "
            f"Val F1: {validation_f1:.5f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1

            model_file.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "feature_columns": feature_columns.tolist(),
                    "best_validation_f1": best_validation_f1,
                },
                model_file,
            )

            print(f"Saved new best model with validation F1: {best_validation_f1:.5f}")

    history_file = EXPERIMENT_ROOT / "data" / "processed" / "lstm_training_history.csv"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(training_history).to_csv(history_file, index=False)

    print("\nTraining completed.")
    print(f"Best validation F1: {best_validation_f1:.5f}")
    print(f"Model saved to: {model_file}")
    print(f"Training history saved to: {history_file}")


if __name__ == "__main__":
    main()
