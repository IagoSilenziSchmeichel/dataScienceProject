"""
Prepare outperformance target data.

This experiment reuses the chronological train/validation/test files from
exp_1_1 and replaces the absolute direction target with a cross-sectional
outperformance target.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
import yaml
from formatting import save_csv


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))


def add_outperformance_target(data):
    data = data.copy()
    data["Daily_Average_Future_Return"] = (
        data.groupby("Date")["Future_Return"].transform("mean")
    )
    data["Target"] = (
        data["Future_Return"] > data["Daily_Average_Future_Return"]
    ).astype(int)
    return data


def prepare_split(split_name, source_key, output_key):
    source_file = EXPERIMENT_ROOT / PARAMS["SOURCE"][source_key]
    output_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"][output_key]

    if not source_file.exists():
        raise FileNotFoundError(f"{split_name} source file not found: {source_file}")

    data = pd.read_csv(source_file, parse_dates=["Date"])
    data = add_outperformance_target(data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_csv(data, output_file)

    print(f"{split_name} rows: {len(data)}")
    print(f"{split_name} saved to: {output_file}")


def main():
    print("Preparing outperformance data")
    print("=============================")

    prepare_split("Train", "TRAIN_FILE", "TRAIN_FILE")
    prepare_split("Validation", "VALIDATION_FILE", "VALIDATION_FILE")
    prepare_split("Test", "TEST_FILE", "TEST_FILE")

    print("\nOutperformance target:")
    print("Target = 1 if Future_Return is above the same-day stock average.")
    print("Daily_Average_Future_Return is kept for analysis, not used as a feature.")


if __name__ == "__main__":
    main()
