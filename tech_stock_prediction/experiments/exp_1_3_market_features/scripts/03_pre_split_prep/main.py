"""
Pre-Split Data Preparation.

Creates technical features and the prediction target for each ticker separately.
The final feature dataset is stored before train/validation/test splitting.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml
import features3 as features
from formatting import save_csv

import targets


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def build_market_features(data):
    market_tickers = PARAMS["DATA_ACQUISITION"].get("MARKET_TICKERS", [])
    market_frames = []

    for ticker in market_tickers:
        ticker_data = data[data["Ticker"] == ticker].copy()
        if ticker_data.empty:
            raise ValueError(f"Missing market ticker data for {ticker}.")

        ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)
        ticker_data[f"{ticker}_Return"] = ticker_data["Close"].pct_change()

        market_frames.append(ticker_data[["Date", f"{ticker}_Return"]])

    if not market_frames:
        raise ValueError("No market tickers configured for market features.")

    market_features = market_frames[0]
    for frame in market_frames[1:]:
        market_features = market_features.merge(frame, on="Date", how="inner")

    return market_features


def main():
    raw_data_file = EXPERIMENT_ROOT / PARAMS["DATA_ACQUISITION"]["RAW_DATA_FILE"]
    feature_data_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["FEATURE_DATA_FILE"]

    data = pd.read_csv(raw_data_file, parse_dates=["Date"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    tradable_tickers = PARAMS["DATA_ACQUISITION"]["TICKERS"]
    market_features = build_market_features(data)
    stock_data = data[data["Ticker"].isin(tradable_tickers)].copy()

    prepared_parts = []

    for ticker, ticker_data in stock_data.groupby("Ticker"):
        print(f"Creating features for {ticker}...")

        ticker_data = features.add_features_for_ticker(ticker_data)
        ticker_data = targets.add_next_day_target(ticker_data)
        prepared_parts.append(ticker_data)

    feature_data = pd.concat(prepared_parts, ignore_index=True)
    feature_data = feature_data.merge(market_features, on="Date", how="left")
    feature_data["Stock_vs_QQQ_Return"] = (
        feature_data["Daily_Return"] - feature_data["QQQ_Return"]
    )
    feature_data["Stock_vs_SPY_Return"] = (
        feature_data["Daily_Return"] - feature_data["SPY_Return"]
    )

    # Drop rows created by rolling windows and the last row per ticker.
    feature_data = feature_data.dropna().drop(columns=["Next_Close"])
    feature_data = feature_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feature_data_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(feature_data, feature_data_file)

    print(f"\nFeature data saved to: {feature_data_file}")
    print(f"Rows after feature engineering: {len(feature_data)}")
    print(f"Tradable tickers in feature data: {sorted(feature_data['Ticker'].unique())}")
    print("QQQ and SPY were used only as market feature sources.")


if __name__ == "__main__":
    main()
