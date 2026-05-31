from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_baseline.pkl"

FEATURE_COLUMNS = [
    "Daily_Return",
    "Lag_1_Return",
    "Lag_3_Return",
    "Lag_7_Return",
    "RollingMean_7",
    "RollingMean_30",
    "RollingVolatility_7",
    "RollingVolatility_30",
    "RSI_14",
    "MACD",
    "MACD_Signal",
]


def print_metrics(y_true, y_pred):
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")


def main():
    train_data = pd.read_csv(PROCESSED_DATA_DIR / "train.csv", parse_dates=["Date"])
    validation_data = pd.read_csv(PROCESSED_DATA_DIR / "validation.csv", parse_dates=["Date"])

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["Target"]

    x_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data["Target"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        class_weight="balanced",
    )

    print("Trainiere RandomForestClassifier...")
    model.fit(x_train, y_train)

    validation_predictions = model.predict(x_validation)

    print("\nValidierungs-Ergebnisse:")
    print_metrics(y_validation, validation_predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModell gespeichert unter: {MODEL_PATH}")


if __name__ == "__main__":
    main()
