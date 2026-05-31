from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tech_stocks_raw.csv"


def main():
    data = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])

    print("DATA UNDERSTANDING")
    print("==================")

    print(f"\nAnzahl Zeilen: {len(data)}")
    print(f"Anzahl Spalten: {len(data.columns)}")

    print("\nZeitraum:")
    print(f"Von: {data['Date'].min().date()}")
    print(f"Bis: {data['Date'].max().date()}")

    print("\nEnthaltene Ticker:")
    print(sorted(data["Ticker"].unique()))

    print("\nFehlende Werte pro Spalte:")
    print(data.isna().sum())

    print("\nDeskriptive Statistiken:")
    print(data.describe())

    print("\nErste Zeilen:")
    print(data.head())


if __name__ == "__main__":
    main()
