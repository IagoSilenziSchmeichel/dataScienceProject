from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "tech_stocks_features.csv"


def main():
    data = pd.read_csv(FEATURE_DATA_PATH, parse_dates=["Date"])

    # Chronologisch sortieren, damit keine zukuenftigen Daten im Training landen.
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    unique_dates = sorted(data["Date"].unique())
    train_date_end = int(len(unique_dates) * 0.70)
    validation_date_end = int(len(unique_dates) * 0.85)

    train_dates = unique_dates[:train_date_end]
    validation_dates = unique_dates[train_date_end:validation_date_end]
    test_dates = unique_dates[validation_date_end:]

    train_data = data[data["Date"].isin(train_dates)].copy()
    validation_data = data[data["Date"].isin(validation_dates)].copy()
    test_data = data[data["Date"].isin(test_dates)].copy()

    train_data.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    validation_data.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test_data.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print("Chronologischer Split gespeichert.")
    print(f"Training: {len(train_data)} Zeilen")
    print(f"Validation: {len(validation_data)} Zeilen")
    print(f"Test: {len(test_data)} Zeilen")

    print("\nZeitraeume:")
    print(f"Training: {train_data['Date'].min().date()} bis {train_data['Date'].max().date()}")
    print(f"Validation: {validation_data['Date'].min().date()} bis {validation_data['Date'].max().date()}")
    print(f"Test: {test_data['Date'].min().date()} bis {test_data['Date'].max().date()}")


if __name__ == "__main__":
    main()
