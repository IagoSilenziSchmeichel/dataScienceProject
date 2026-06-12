"""
Run LSTM Pipeline.

This script runs all steps of the LSTM experiment in the correct order.
The existing Random Forest experiment remains unchanged.
"""

from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parent


SCRIPTS = [
    "scripts/01_data_preparation/prepare_lstm_data.py",
    "scripts/02_sequence_creation/create_sequences.py",
    "scripts/03_model_training/train_lstm.py",
    "scripts/04_model_testing/evaluate_lstm.py",
    "scripts/05_backtesting/backtest_lstm.py",
]


def main():
    for script in SCRIPTS:
        script_path = EXPERIMENT_ROOT / script

        print("\n" + "=" * 80)
        print(f"Running {script}")
        print("=" * 80)

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=EXPERIMENT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()