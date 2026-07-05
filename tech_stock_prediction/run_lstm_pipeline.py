"""
Run the complete LSTM experiment pipeline from one file.

This is useful when IntelliJ run configurations are confusing or when the LSTM
steps should be executed in the correct order.
"""

from pathlib import Path
import subprocess
import sys

from ensure_venv import restart_with_project_venv

restart_with_project_venv()


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "exp_2_lstm"

SCRIPTS = [
    "scripts/01_data_preparation/prepare_lstm_data.py",
    "scripts/02_sequence_creation/create_sequences.py",
    "scripts/03_model_training/train_lstm.py",
    "scripts/04_model_testing/evaluate_lstm.py",
    "scripts/05_backtesting/backtest_lstm.py",
    "scripts/10_top_k_backtest/top_k_backtest.py",
    "scripts/14_outperformance_lstm/outperformance_lstm.py",
    "scripts/15_feature_ablation/lstm_feature_ablation.py",
    "scripts/16_robustness/robustness_check.py",
    "scripts/11_visualization/generate_lstm_plots.py",
]


def main():
    scripts_to_run = SCRIPTS.copy()

    for script in scripts_to_run:
        print("\n" + "=" * 80, flush=True)
        print(f"Running {script}", flush=True)
        print("=" * 80, flush=True)

        script_path = EXPERIMENT_ROOT / script
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
