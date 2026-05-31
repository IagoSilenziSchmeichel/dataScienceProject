"""
Run the complete experiment pipeline from one file.

This is useful when IntelliJ does not show Python run configurations correctly.
"""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "exp_1_1"

SCRIPTS = [
    "scripts/01_data_acquisition/bar_retriever.py",
    "scripts/02_data_understanding/plotter.py",
    "scripts/03_pre_split_prep/main.py",
    "scripts/04_split_data/split.py",
    "scripts/05_model_training/train_random_forest.py",
    "scripts/06_model_testing/evaluate_random_forest.py",
    "scripts/07_backtesting/simple_backtest.py",
]


def main():
    for script in SCRIPTS:
        print("\n" + "=" * 80)
        print(f"Running {script}")
        print("=" * 80)

        script_path = EXPERIMENT_ROOT / script
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
