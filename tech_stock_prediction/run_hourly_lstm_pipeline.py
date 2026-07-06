"""
Run the Hourly Outperformance-LSTM pipeline for Alpaca Paper Trading.

This entry point is intentionally separate from run_lstm_pipeline.py. The
existing daily research pipeline remains unchanged and continues to produce the
scientific results used for backtests, ablations and robustness checks.
"""

from pathlib import Path
import subprocess
import sys

from ensure_venv import restart_with_project_venv

restart_with_project_venv()


PROJECT_ROOT = Path(__file__).resolve().parent
HOURLY_ROOT = PROJECT_ROOT / "experiments" / "exp_2_lstm" / "hourly"

SCRIPTS = [
    "scripts/hourly_outperformance_lstm.py",
]


def main():
    for script in SCRIPTS:
        print("\n" + "=" * 80, flush=True)
        print(f"Running hourly pipeline step: {script}", flush=True)
        print("=" * 80, flush=True)

        script_path = HOURLY_ROOT / script
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
