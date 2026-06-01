"""
Data Understanding.

Loads the raw CSV file and prints basic information about the dataset.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml
from formatting import set_pandas_display_options


set_pandas_display_options(pd)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def main():
    raw_data_file = EXPERIMENT_ROOT / PARAMS["DATA_ACQUISITION"]["RAW_DATA_FILE"]
    data = pd.read_csv(raw_data_file, parse_dates=["Date"])

    print("DATA UNDERSTANDING")
    print("==================")

    print(f"\nRows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    print("\nTime period:")
    print(f"From: {data['Date'].min().date()}")
    print(f"To:   {data['Date'].max().date()}")

    print("\nTickers:")
    print(sorted(data["Ticker"].unique()))

    print("\nMissing values:")
    print(data.isna().sum())

    print("\nDescriptive statistics:")
    print(data.describe())

    print("\nRows per ticker:")
    print(data["Ticker"].value_counts().sort_index())


if __name__ == "__main__":
    main()
