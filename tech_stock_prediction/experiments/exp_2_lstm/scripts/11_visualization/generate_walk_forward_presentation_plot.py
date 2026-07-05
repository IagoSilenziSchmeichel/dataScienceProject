"""
Create a presentation plot for walk-forward robustness.

The plot connects the best backtest candidates with the robustness check:
Feature Engineering -> Top-K Backtest -> Robustness Check -> Model Selection.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    EXPERIMENT_ROOT
    / "data"
    / "processed"
    / "lstm_robustness_walk_forward_summary.csv"
)
OUTPUT_FILE = (
    EXPERIMENT_ROOT
    / "plots"
    / "12_walk_forward_performance_comparison.png"
)

WINDOW_ORDER = ["2025_H1", "2025_H2", "2026_H1"]
WINDOW_LABELS = ["2025 H1", "2025 H2", "2026 H1"]

MODELS = [
    {
        "model_name": "standard_lstm",
        "feature_group": "Technical + Market",
        "label": "Standard-LSTM\nTechnical + Market",
        "color": "#2563eb",
    },
    {
        "model_name": "outperformance_lstm",
        "feature_group": "Technical + Relative Strength",
        "label": "Outperformance-LSTM\nTechnical + Relative Strength",
        "color": "#7c3aed",
    },
]


def get_difference(data, model, window):
    """
    Return Difference vs. Buy-and-Hold as percentage points.
    """
    match = data[
        (data["model_name"] == model["model_name"])
        & (data["feature_group"] == model["feature_group"])
        & (data["window"] == window)
    ]

    if match.empty:
        raise ValueError(
            "Missing walk-forward row for "
            f"{model['model_name']} / {model['feature_group']} / {window}"
        )

    return float(match.iloc[0]["difference"]) * 100


def format_bar(axis, bar, value, model_color, is_best):
    """
    Positive bars keep the model color. Negative bars are shown as muted red.
    The highest bar gets a gold outline.
    """
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
    vertical_alignment = "bottom" if value >= 0 else "top"

    axis.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:+.1f} pp",
        ha="center",
        va=vertical_alignment,
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


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE)

    values_by_model = []
    all_values = []

    for model in MODELS:
        model_values = [get_difference(data, model, window) for window in WINDOW_ORDER]
        values_by_model.append(model_values)
        all_values.extend(model_values)

    best_value = max(all_values)

    fig, axis = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")

    x_positions = np.arange(len(WINDOW_ORDER))
    bar_width = 0.34

    for model_index, model in enumerate(MODELS):
        offsets = x_positions + (model_index - 0.5) * bar_width
        bars = axis.bar(
            offsets,
            values_by_model[model_index],
            width=bar_width,
            label=model["label"],
        )

        for bar, value in zip(bars, values_by_model[model_index]):
            format_bar(
                axis,
                bar,
                value,
                model["color"],
                is_best=value == best_value,
            )

    axis.axhline(0, color="#111827", linewidth=1.6)
    axis.text(
        x_positions[-1] + 0.22,
        1.5,
        "0% = Buy-and-Hold",
        fontsize=12,
        color="#374151",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(WINDOW_LABELS, fontsize=15, fontweight="bold")
    axis.set_ylabel("Difference to Buy-and-Hold (%)", fontsize=15)
    axis.tick_params(axis="y", labelsize=13)
    axis.set_ylim(-25, 76)
    axis.grid(axis="y", alpha=0.2)

    title = "Walk-Forward Performance Across Market Phases"
    subtitle = "Difference to Buy-and-Hold over three independent market periods"

    fig.suptitle(title, fontsize=24, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=15, color="#374151")

    method_text = (
        "Feature Engineering  ->  Top-K Backtest  ->  Robustness Check  ->  Model Selection"
    )
    fig.text(
        0.5,
        0.07,
        method_text,
        ha="center",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.5,
        0.035,
        "Nicht die hoechste Einzelrendite entscheidet, sondern Stabilitaet ueber mehrere Marktphasen.",
        ha="center",
        fontsize=14,
        color="#374151",
    )

    legend_handles = [
        mpatches.Patch(color=MODELS[0]["color"], label=MODELS[0]["label"]),
        mpatches.Patch(color=MODELS[1]["color"], label=MODELS[1]["label"]),
        mpatches.Patch(
            facecolor="#fecaca",
            edgecolor="#991b1b",
            hatch="//",
            label="Underperformed Buy-and-Hold",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=12,
        frameon=True,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0.04, 0.11, 1, 0.88])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Presentation plot saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
