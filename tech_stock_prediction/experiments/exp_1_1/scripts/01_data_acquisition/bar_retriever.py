"""
Data Acquisition.

Downloads daily Yahoo Finance data for selected large tech stocks and stores one
combined CSV file. Each row contains the stock ticker.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def main():
    tickers = PARAMS["DATA_ACQUISITION"]["TICKERS"]
    start_date = PARAMS["DATA_ACQUISITION"]["START_DATE"]
    raw_data_file = EXPERIMENT_ROOT / PARAMS["DATA_ACQUISITION"]["RAW_DATA_FILE"]

    all_data = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")

        data = yf.download(
            ticker,
            start=start_date,
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            print(f"No data found for {ticker}.")
            continue

        # Some yfinance versions return MultiIndex columns.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Ticker"] = ticker
        all_data.append(data)

    if not all_data:
        raise ValueError("No stock data could be downloaded.")

    raw_data = pd.concat(all_data, ignore_index=True)
    raw_data = raw_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    raw_data_file.parent.mkdir(parents=True, exist_ok=True)
    raw_data.to_csv(raw_data_file, index=False)

    print(f"\nRaw data saved to: {raw_data_file}")
    print(f"Rows: {len(raw_data)}")


if __name__ == "__main__":
    main()
