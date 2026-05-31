from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_baseline.pkl"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "test_predictions.csv"

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


def main():
    test_data = pd.read_csv(PROCESSED_DATA_DIR / "test.csv", parse_dates=["Date"])
    model = joblib.load(MODEL_PATH)

    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["Target"]

    test_predictions = model.predict(x_test)

    print("Test-Ergebnisse:")
    print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, test_predictions, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    results = test_data.copy()
    results["Prediction"] = test_predictions
    results.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nPredictions gespeichert unter: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
