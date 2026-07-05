"""
Create a fair walk-forward presentation plot using equal test periods.

For each walk-forward window, both final model variants are evaluated only on
their overlapping dates. This keeps Buy-and-Hold directly comparable.
"""

from pathlib import Path
import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_SCRIPT = EXPERIMENT_ROOT / "scripts" / "16_robustness" / "robustness_check.py"
OUTPUT_CSV = (
    EXPERIMENT_ROOT
    / "data"
    / "processed"
    / "lstm_robustness_walk_forward_common_period_summary.csv"
)
OUTPUT_PLOT = (
    EXPERIMENT_ROOT
    / "plots"
    / "12_walk_forward_performance_comparison.png"
)

WINDOW_LABELS = {
    "2025_H1": "2025 H1",
    "2025_H2": "2025 H2",
    "2026_H1": "2026 H1",
}

MODEL_STYLES = {
    "standard_lstm": {
        "label": "Standard-LSTM\nTechnical + Market",
        "color": "#2563eb",
    },
    "outperformance_lstm": {
        "label": "Outperformance-LSTM\nTechnical + Relative Strength",
        "color": "#7c3aed",
    },
}


def load_robustness_module():
    spec = importlib.util.spec_from_file_location("robustness_check", ROBUSTNESS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def date_range(predictions):
    dates = pd.to_datetime(predictions["Date"])
    return dates.min(), dates.max()


def filter_to_period(predictions, start, end):
    dates = pd.to_datetime(predictions["Date"])
    return predictions[(dates >= start) & (dates <= end)].copy()


def calculate_common_period_results(robustness):
    data = robustness.prepare_data()
    rows = []

    for window in robustness.WINDOWS:
        print("\n" + "=" * 80)
        print(f"Common-period walk-forward window: {window['window']}")
        print("=" * 80)

        predictions_by_model = {}

        for variant in robustness.MODEL_VARIANTS:
            predictions, _ = robustness.train_and_predict(data, variant, window)
            predictions_by_model[variant["model_name"]] = {
                "variant": variant,
                "predictions": predictions,
            }

        starts = []
        ends = []
        for model_data in predictions_by_model.values():
            start, end = date_range(model_data["predictions"])
            starts.append(start)
            ends.append(end)

        common_start = max(starts)
        common_end = min(ends)

        for model_name, model_data in predictions_by_model.items():
            variant = model_data["variant"]
            common_predictions = filter_to_period(
                model_data["predictions"],
                common_start,
                common_end,
            )

            metrics, _, _ = robustness.calculate_backtest_metrics(
                common_predictions,
                variant["default_top_k"],
                robustness.DEFAULT_TRANSACTION_COST,
            )

            rows.append(
                {
                    "model_name": variant["model_name"],
                    "feature_group": variant["feature_group"],
                    "window": window["window"],
                    "top_k": variant["default_top_k"],
                    "transaction_cost": robustness.DEFAULT_TRANSACTION_COST,
                    "common_start": common_start.date().isoformat(),
                    "common_end": common_end.date().isoformat(),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def format_bar(axis, bar, value, model_color, is_best):
    if value < 0:
        bar.set_facecolor("#fecaca")
        bar.set_edgecolor(model_color)
        bar.set_hatch("//")
        bar.set_linewidth(2)
    else:
        bar.set_facecolor(model_color)
        bar.set_edgecolor(model_color)
        bar.set_linewidth(1.5)

    if is_best:
        bar.set_edgecolor("#f59e0b")
        bar.set_linewidth(4)

    offset = 2.0 if value >= 0 else -2.6
    va = "bottom" if value >= 0 else "top"

    axis.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:+.1f} pp",
        ha="center",
        va=va,
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )

    if is_best:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 8,
            "Best",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#92400e",
        )


def create_plot(results):
    windows = list(WINDOW_LABELS.keys())
    x_positions = np.arange(len(windows))
    bar_width = 0.34

    all_values = (results["difference"] * 100).tolist()
    best_value = max(all_values)

    fig, axis = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor("white")

    for model_index, model_name in enumerate(MODEL_STYLES.keys()):
        model_rows = []

        for window in windows:
            match = results[
                (results["model_name"] == model_name)
                & (results["window"] == window)
            ]
            if match.empty:
                raise ValueError(f"Missing result for {model_name} / {window}")
            model_rows.append(match.iloc[0])

        values = [row["difference"] * 100 for row in model_rows]
        offsets = x_positions + (model_index - 0.5) * bar_width

        bars = axis.bar(
            offsets,
            values,
            width=bar_width,
            label=MODEL_STYLES[model_name]["label"],
        )

        for bar, value in zip(bars, values):
            format_bar(
                axis,
                bar,
                value,
                MODEL_STYLES[model_name]["color"],
                is_best=value == best_value,
            )

    axis.axhline(0, color="#111827", linewidth=1.6)
    axis.text(
        x_positions[-1] + 0.18,
        1.5,
        "0% = Buy-and-Hold",
        fontsize=12,
        color="#374151",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([WINDOW_LABELS[window] for window in windows], fontsize=15, fontweight="bold")
    axis.set_ylabel("Difference to Buy-and-Hold (%)", fontsize=15)
    axis.tick_params(axis="y", labelsize=13)
    axis.set_ylim(-25, 76)
    axis.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Walk-Forward Performance Across Market Phases",
        fontsize=24,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "Difference to Buy-and-Hold over three independent market periods",
        ha="center",
        fontsize=15,
        color="#374151",
    )

    fig.text(
        0.5,
        0.07,
        "Feature Engineering  ->  Top-K Backtest  ->  Robustness Check  ->  Model Selection",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.5,
        0.035,
        "Gleiche Testzeitraeume je Fenster: Nicht die hoechste Einzelrendite entscheidet, sondern Stabilitaet.",
        ha="center",
        fontsize=14,
        color="#374151",
    )

    legend_handles = [
        mpatches.Patch(
            color=MODEL_STYLES["standard_lstm"]["color"],
            label=MODEL_STYLES["standard_lstm"]["label"],
        ),
        mpatches.Patch(
            color=MODEL_STYLES["outperformance_lstm"]["color"],
            label=MODEL_STYLES["outperformance_lstm"]["label"],
        ),
        mpatches.Patch(
            facecolor="#fecaca",
            edgecolor="#991b1b",
            hatch="//",
            label="Underperformed Buy-and-Hold",
        ),
    ]

    axis.legend(handles=legend_handles, loc="upper left", fontsize=12)

    plt.tight_layout(rect=[0.04, 0.11, 1, 0.88])
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    robustness = load_robustness_module()
    results = calculate_common_period_results(robustness)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    create_plot(results)

    print(f"Common-period summary saved to: {OUTPUT_CSV}")
    print(f"Presentation plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
