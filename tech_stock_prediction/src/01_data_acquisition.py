from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tech_stocks_raw.csv"

TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "TSLA", "INTC", "ADBE"]
START_DATE = "2019-01-01"


def main():
    all_data = []

    for ticker in TICKERS:
        print(f"Lade Daten fuer {ticker}...")

        data = yf.download(
            ticker,
            start=START_DATE,
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            print(f"Keine Daten fuer {ticker} gefunden.")
            continue

        # Manche yfinance-Versionen liefern verschachtelte Spaltennamen.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Ticker"] = ticker
        all_data.append(data)

    if not all_data:
        raise ValueError("Es konnten keine Kursdaten geladen werden.")

    raw_data = pd.concat(all_data, ignore_index=True)
    raw_data = raw_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_data.to_csv(RAW_DATA_PATH, index=False)

    print(f"\nRohdaten gespeichert unter: {RAW_DATA_PATH}")
    print(f"Anzahl Zeilen: {len(raw_data)}")


if __name__ == "__main__":
    main()
