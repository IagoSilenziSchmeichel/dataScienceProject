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
import features2 as features

import targets


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def main():
    raw_data_file = EXPERIMENT_ROOT / PARAMS["DATA_ACQUISITION"]["RAW_DATA_FILE"]
    feature_data_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["FEATURE_DATA_FILE"]

    data = pd.read_csv(raw_data_file, parse_dates=["Date"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    prepared_parts = []

    for ticker, ticker_data in data.groupby("Ticker"):
        print(f"Creating features for {ticker}...")

        ticker_data = features.add_features_for_ticker(ticker_data)
        ticker_data = targets.add_next_day_target(ticker_data)
        prepared_parts.append(ticker_data)

    feature_data = pd.concat(prepared_parts, ignore_index=True)

    # Drop rows created by rolling windows and the last row per ticker.
    feature_data = feature_data.dropna().drop(columns=["Next_Close"])
    feature_data = feature_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feature_data_file.parent.mkdir(parents=True, exist_ok=True)
    feature_data.to_csv(feature_data_file, index=False)

    print(f"\nFeature data saved to: {feature_data_file}")
    print(f"Rows after feature engineering: {len(feature_data)}")


if __name__ == "__main__":
    main()
