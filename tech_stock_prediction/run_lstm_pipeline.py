"""
Run LSTM Pipeline.

This script runs the LSTM experiment pipeline from the project root.
The actual LSTM experiment is stored in:

experiments/exp_2_lstm/
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "exp_2_lstm"


SCRIPTS = [
    "scripts/01_data_preparation/prepare_lstm_data.py",
    "scripts/02_sequence_creation/create_sequences.py",
    "scripts/03_model_training/train_lstm.py",
    "scripts/04_model_testing/evaluate_lstm.py",
    "scripts/05_backtesting/backtest_lstm.py",
]


def main():
    if not EXPERIMENT_ROOT.exists():
        raise FileNotFoundError(f"LSTM experiment folder not found: {EXPERIMENT_ROOT}")

    for script in SCRIPTS:
        script_path = EXPERIMENT_ROOT / script

        print("\n" + "=" * 80)
        print(f"Running {script}")
        print("=" * 80)

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=EXPERIMENT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()