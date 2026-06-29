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

# Tuning trains several LSTM models and can take much longer.
# Keep it False for the normal team pipeline.
RUN_TUNING = False
RUN_TUNED_FINAL_TEST = True
RUN_FEATURE_ABLATION = False
RUN_ROBUSTNESS_CHECK = False

TUNING_RESULTS_FILE = (
    EXPERIMENT_ROOT / "data" / "processed" / "lstm_tuning_results.csv"
)

SCRIPTS = [
    "scripts/01_data_preparation/prepare_lstm_data.py",
    "scripts/02_sequence_creation/create_sequences.py",
    "scripts/03_model_training/train_lstm.py",
    "scripts/04_model_testing/evaluate_lstm.py",
    "scripts/05_backtesting/backtest_lstm.py",
    "scripts/08_threshold_backtest/threshold_backtest.py",
    "scripts/10_top_k_backtest/top_k_backtest.py",
    "scripts/14_outperformance_lstm/outperformance_lstm.py",
    "scripts/11_visualization/generate_lstm_plots.py",
]

TUNING_SCRIPTS = [
    "scripts/09_lstm_tuning/lstm_tuning.py",
]

TUNED_FINAL_TEST_SCRIPT = "scripts/12_tuned_final_test/tuned_final_test.py"
COMPARISON_SCRIPT = "scripts/13_comparison/compare_lstm_results.py"
FEATURE_ABLATION_SCRIPT = "scripts/15_feature_ablation/lstm_feature_ablation.py"
ROBUSTNESS_CHECK_SCRIPT = "scripts/16_robustness/robustness_check.py"


def main():
    scripts_to_run = SCRIPTS.copy()

    if RUN_TUNING:
        insert_position = 3
        scripts_to_run[insert_position:insert_position] = TUNING_SCRIPTS

    if RUN_TUNED_FINAL_TEST:
        if RUN_TUNING or TUNING_RESULTS_FILE.exists():
            scripts_to_run.insert(-1, TUNED_FINAL_TEST_SCRIPT)
            scripts_to_run.insert(-1, COMPARISON_SCRIPT)
        else:
            print(
                "Skipping tuned final test because lstm_tuning_results.csv is missing. "
                "Run scripts/09_lstm_tuning/lstm_tuning.py first or set RUN_TUNING = True.",
                flush=True,
            )

    if RUN_FEATURE_ABLATION:
        scripts_to_run.insert(-1, FEATURE_ABLATION_SCRIPT)

    if RUN_ROBUSTNESS_CHECK:
        scripts_to_run.insert(-1, ROBUSTNESS_CHECK_SCRIPT)

    for script in scripts_to_run:
        print("\n" + "=" * 80, flush=True)
        print(f"Running {script}", flush=True)
        print("=" * 80, flush=True)

        script_path = EXPERIMENT_ROOT / script
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
