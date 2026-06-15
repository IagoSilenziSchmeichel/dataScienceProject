"""
Evaluate LSTM model.

This script loads the best saved LSTM model and evaluates it on the test set.
It calculates the same classification metrics as the Random Forest experiment:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

It also compares the LSTM to a simple Always-Yes baseline.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


class LSTMClassifier(nn.Module):
    """
    Same model structure as in train_lstm.py.

    We need the same class here so that we can load the saved model weights.
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


def load_checkpoint(model_file, device):
    """
    Load the saved PyTorch checkpoint.

    The try/except keeps it compatible with different PyTorch versions.
    """
    try:
        checkpoint = torch.load(
            model_file,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            model_file,
            map_location=device,
        )

    return checkpoint


def create_test_loader(X_test, y_test, batch_size):
    """
    Convert test arrays into a PyTorch DataLoader.
    """
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return loader


def predict(model, test_loader, device, threshold):
    """
    Create predictions and probabilities for the test set.
    """
    model.eval()

    all_targets = []
    all_probabilities = []
    all_predictions = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).int()

            all_targets.extend(y_batch.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_targets = np.array(all_targets).astype(int)
    all_probabilities = np.array(all_probabilities)
    all_predictions = np.array(all_predictions).astype(int)

    return all_targets, all_probabilities, all_predictions


def print_metrics(title, y_true, y_pred):
    """
    Print classification metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(title)
    print("-" * len(title))
    print(f"Accuracy:  {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"F1-Score:  {f1:.5f}")

    return accuracy, precision, recall, f1


def main():
    print("Evaluating LSTM model")
    print("=====================")

    sequence_file = EXPERIMENT_ROOT / PARAMS["DATA"]["SEQUENCE_FILE"]
    test_metadata_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_METADATA_FILE"]
    model_file = EXPERIMENT_ROOT / PARAMS["MODEL"]["MODEL_FILE"]
    predictions_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["PREDICTIONS_FILE"]

    batch_size = PARAMS["MODEL"]["BATCH_SIZE"]
    threshold = PARAMS["MODEL"].get("PREDICTION_THRESHOLD", 0.50)

    if not sequence_file.exists():
        raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

    if not test_metadata_file.exists():
        raise FileNotFoundError(f"Test metadata file not found: {test_metadata_file}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    data = np.load(sequence_file, allow_pickle=True)

    X_test = data["X_test"]
    y_test = data["y_test"]

    test_metadata = pd.read_csv(test_metadata_file)

    if len(test_metadata) != len(y_test):
        raise ValueError(
            "Test metadata length does not match number of test labels."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Prediction threshold: {threshold:.2f}")

    checkpoint = load_checkpoint(model_file, device)

    model = LSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model from: {model_file}")
    print(f"LSTM layers: {checkpoint['num_layers']}")
    if checkpoint["num_layers"] == 1:
        print("Dropout: disabled inside LSTM because num_layers = 1")
    else:
        print(f"Dropout: {checkpoint['dropout']}")
    print(f"Best validation F1 from training: {checkpoint['best_validation_f1']:.5f}")

    test_loader = create_test_loader(
        X_test,
        y_test,
        batch_size=batch_size,
    )

    y_true, probabilities, predictions = predict(
        model,
        test_loader,
        device,
        threshold,
    )

    print()
    lstm_metrics = print_metrics(
        "LSTM Test Results",
        y_true,
        predictions,
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    always_yes_predictions = np.ones_like(y_true)

    print()
    always_yes_metrics = print_metrics(
        "Always-Yes Baseline",
        y_true,
        always_yes_predictions,
    )

    accuracy_difference = lstm_metrics[0] - always_yes_metrics[0]
    f1_difference = lstm_metrics[3] - always_yes_metrics[3]

    print("\nComparison:")
    print(f"Accuracy difference LSTM - Always-Yes: {accuracy_difference:.5f}")
    print(f"F1 difference LSTM - Always-Yes:       {f1_difference:.5f}")

    if accuracy_difference > 0:
        print("=> LSTM beats Always-Yes in Accuracy.")
    else:
        print("=> LSTM does NOT beat Always-Yes in Accuracy.")

    if f1_difference > 0:
        print("=> LSTM beats Always-Yes in F1-Score.")
    else:
        print("=> LSTM does NOT beat Always-Yes in F1-Score.")

    results = test_metadata.copy()
    results["Probability"] = probabilities
    results["Prediction"] = predictions
    results["Actual"] = y_true

    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(predictions_file, index=False)

    print(f"\nPredictions saved to: {predictions_file}")


if __name__ == "__main__":
    main()
