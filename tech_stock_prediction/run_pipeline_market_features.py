"""
Run the complete exp_1_3_market_features pipeline.
"""

from pathlib import Path
import subprocess
import sys

from ensure_venv import restart_with_project_venv

restart_with_project_venv()


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "exp_1_3_market_features"

SCRIPTS = [
    "scripts/01_data_acquisition/bar_retriever.py",
    "scripts/02_data_understanding/plotter.py",
    "scripts/03_pre_split_prep/main.py",
    "scripts/04_split_data/split.py",
    "scripts/05_model_training/train_random_forest.py",
    "scripts/06_model_testing/evaluate_random_forest.py",
    "scripts/07_backtesting/simple_backtest.py",
    "scripts/10_threshold_tuning/tune_random_forest_threshold.py",
    "scripts/11_percentile_threshold_backtest/tune_random_forest_percentile_threshold.py",
    "scripts/12_top_k_backtest/top_k_random_forest_backtest.py",
    "scripts/08_baseline/dummy_baseline.py",
    "scripts/09_visualization/generate_all_plots.py",
]


def main():
    for script in SCRIPTS:
        print("\n" + "=" * 80, flush=True)
        print(f"Running {script}", flush=True)
        print("=" * 80, flush=True)

        script_path = EXPERIMENT_ROOT / script
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
