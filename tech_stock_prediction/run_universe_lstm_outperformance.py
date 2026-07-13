"""
Run the final Outperformance-LSTM + Top-K pipeline for one or all four stock
universes (see config/stock_universes.py), and archive the results so runs
for different universes do not overwrite each other.

Usage:
    python run_universe_lstm_outperformance.py --universe defensive_non_tech
    python run_universe_lstm_outperformance.py --all

For each requested universe this script:
  1. Points TICKERS/UNIVERSE at the requested universe (see set_universe.py).
  2. Re-downloads raw prices and rebuilds features/split
     (experiments/exp_1_randomforest, steps 01-04). Random Forest itself is
     NOT trained here - that experiment folder is only reused for its shared
     data-acquisition/feature/split steps, which the LSTM pipeline also
     depends on.
  3. Runs the daily LSTM chain (experiments/exp_2_lstm): the standard LSTM
     first (needed as the comparison baseline inside the outperformance
     script), then the final Outperformance-LSTM + Top-K backtest, then
     plots.
  4. Archives data/models/plots/reports into
     universe_results/<universe_name>/ so the next universe run does not
     overwrite this one.

After all requested universes are done, a combined comparison CSV
(universe_results/all_universes_comparison.csv) and a small comparison plot
are written so all universes can be shown on a single slide.

Feature-ablation and robustness checks (steps 15/16 in exp_2_lstm) are
intentionally skipped here for speed, because they retrain several extra
LSTMs per universe. Run them separately per
experiments/exp_2_lstm/README.md if there is time left before the
presentation.
"""

from pathlib import Path
import shutil
import subprocess
import sys

from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config.stock_universes import get_universe, list_available_universes
from formatting import save_csv
from set_universe import set_universe_in_lstm_params, set_universe_in_rf_params


PROJECT_ROOT = Path(__file__).resolve().parent
RF_ROOT = PROJECT_ROOT / "experiments" / "exp_1_randomforest"
LSTM_ROOT = PROJECT_ROOT / "experiments" / "exp_2_lstm"
RESULTS_ROOT = PROJECT_ROOT / "universe_results"

# Shared data-acquisition/feature/split steps. Random Forest training itself
# (steps 05+ in exp_1_randomforest) is skipped on purpose.
RF_DATA_SCRIPTS = [
    "scripts/01_data_acquisition/bar_retriever.py",
    "scripts/02_data_understanding/plotter.py",
    "scripts/03_pre_split_prep/main.py",
    "scripts/04_split_data/split.py",
]

# Standard LSTM (needed as comparison baseline) + final Outperformance-LSTM
# Top-K pipeline + plots. Ablation/robustness intentionally excluded.
LSTM_SCRIPTS = [
    "scripts/01_data_preparation/prepare_lstm_data.py",
    "scripts/02_sequence_creation/create_sequences.py",
    "scripts/03_model_training/train_lstm.py",
    "scripts/04_model_testing/evaluate_lstm.py",
    "scripts/05_backtesting/backtest_lstm.py",
    "scripts/10_top_k_backtest/top_k_backtest.py",
    "scripts/14_outperformance_lstm/outperformance_lstm.py",
    "scripts/11_visualization/generate_lstm_plots.py",
]


def run_script(experiment_root, relative_path):
    script_path = experiment_root / relative_path
    print("\n" + "=" * 80, flush=True)
    print(f"Running {script_path.relative_to(PROJECT_ROOT)}", flush=True)
    print("=" * 80, flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


def archive_universe_results(universe_name):
    universe_dir = RESULTS_ROOT / universe_name

    rf_target = universe_dir / "exp_1_randomforest"
    for folder in ["data", "models", "plots"]:
        source = RF_ROOT / folder
        if source.exists():
            shutil.copytree(source, rf_target / folder, dirs_exist_ok=True)

    lstm_target = universe_dir / "exp_2_lstm"
    for folder in ["data", "models", "plots", "reports"]:
        source = LSTM_ROOT / folder
        if source.exists():
            shutil.copytree(source, lstm_target / folder, dirs_exist_ok=True)

    print(f"\nArchived results to: {universe_dir}")


def load_universe_summary(universe_name):
    results_file = LSTM_ROOT / "data" / "processed" / "lstm_outperformance_results.csv"
    top_k_file = LSTM_ROOT / "data" / "processed" / "lstm_outperformance_top_k_results.csv"

    if not results_file.exists() or not top_k_file.exists():
        print(f"Warning: missing outperformance result files for {universe_name}, skipping summary row.")
        return None

    results = pd.read_csv(results_file)
    top_k_results = pd.read_csv(top_k_file)
    best_top_k_row = top_k_results.sort_values("strategy_return", ascending=False).iloc[0]
    result_row = results.iloc[0]

    return {
        "universe": universe_name,
        "benchmark": result_row.get("benchmark", ""),
        "test_start": best_top_k_row["test_start"],
        "test_end": best_top_k_row["test_end"],
        "accuracy": result_row["accuracy"],
        "f1_score": result_row["f1_score"],
        "best_top_k": int(best_top_k_row["top_k"]),
        "top_k_strategy_return": best_top_k_row["strategy_return"],
        "top_k_buy_and_hold_return": best_top_k_row["buy_and_hold_return"],
        "top_k_difference": best_top_k_row["difference"],
    }


def save_comparison(summary_rows):
    if not summary_rows:
        print("No universe summaries collected; skipping combined comparison output.")
        return

    summary = pd.DataFrame(summary_rows)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_file = RESULTS_ROOT / "all_universes_comparison.csv"
    save_csv(summary, comparison_file)
    print(f"\nSaved combined comparison: {comparison_file}")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        summary["universe"],
        summary["top_k_strategy_return"] * 100,
        color="#4F8F6F",
        label="Top-K Strategie",
    )
    ax.scatter(
        summary["universe"],
        summary["top_k_buy_and_hold_return"] * 100,
        color="#D08B2C",
        marker="D",
        s=80,
        label="Buy-and-Hold (gleicher Zeitraum)",
        zorder=3,
    )
    ax.set_title("Outperformance-LSTM Top-K: Vergleich aller Universen", fontsize=15, fontweight="bold")
    ax.set_ylabel("Rendite (%)")
    ax.axhline(0, color="#111111", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in summary["top_k_strategy_return"] * 100],
        padding=4,
    )
    fig.autofmt_xdate(rotation=20)
    plot_file = RESULTS_ROOT / "all_universes_comparison.png"
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined comparison plot: {plot_file}")


def run_universe(universe_name):
    print("\n" + "#" * 80)
    print(f"# Universe: {universe_name}")
    print("#" * 80)

    tickers = get_universe(universe_name)
    set_universe_in_rf_params(universe_name, tickers)
    set_universe_in_lstm_params(universe_name)

    for relative_path in RF_DATA_SCRIPTS:
        run_script(RF_ROOT, relative_path)

    for relative_path in LSTM_SCRIPTS:
        run_script(LSTM_ROOT, relative_path)

    archive_universe_results(universe_name)
    return load_universe_summary(universe_name)


def main():
    args = sys.argv[1:]
    available = list_available_universes()

    if args == ["--all"]:
        universe_names = available
    elif len(args) == 2 and args[0] == "--universe" and args[1] in available:
        universe_names = [args[1]]
    else:
        print("Usage:")
        print("  python run_universe_lstm_outperformance.py --universe <name>")
        print("  python run_universe_lstm_outperformance.py --all")
        print(f"Available universes: {', '.join(available)}")
        sys.exit(1)

    summary_rows = []
    for universe_name in universe_names:
        summary_row = run_universe(universe_name)
        if summary_row is not None:
            summary_rows.append(summary_row)

    save_comparison(summary_rows)


if __name__ == "__main__":
    main()
