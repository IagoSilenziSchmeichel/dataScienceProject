"""
Create a simple walk-forward robustness plot for the presentation.

The plot compares only:
- Standard-LSTM Top-K
- Outperformance-LSTM Top-K

Feature group names are intentionally not shown, so the slide stays easy to
understand. The y-axis shows Difference vs. Buy-and-Hold.
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
    / "14_walk_forward_top_k_vs_buy_and_hold.png"
)

WINDOW_ORDER = ["2025_H1", "2025_H2", "2026_H1"]
WINDOW_LABELS = ["2025 H1", "2025 H2", "2026 H1"]

MODEL_LABELS = {
    "standard_lstm": "Standard-LSTM Top-K",
    "outperformance_lstm": "Outperformance-LSTM Top-K",
}

MODEL_COLORS = {
    "standard_lstm": "#2563eb",
    "outperformance_lstm": "#7c3aed",
}


def get_values(data, model_name):
    values = []

    for window in WINDOW_ORDER:
        match = data[
            (data["model_name"] == model_name)
            & (data["window"] == window)
        ]

        if match.empty:
            raise ValueError(f"Missing row for {model_name} / {window}")

        values.append(float(match.iloc[0]["difference"]) * 100)

    return values


def style_bar(bar, value, base_color, is_best):
    if value < 0:
        bar.set_facecolor("#fecaca")
        bar.set_edgecolor("#991b1b")
        bar.set_hatch("//")
        bar.set_linewidth(2)
    else:
        bar.set_facecolor(base_color)
        bar.set_edgecolor(base_color)
        bar.set_linewidth(1.5)

    if is_best:
        bar.set_edgecolor("#f59e0b")
        bar.set_linewidth(4)


def add_value_label(axis, bar, value):
    offset = 1.8 if value >= 0 else -2.2
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


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE)

    standard_values = get_values(data, "standard_lstm")
    outperformance_values = get_values(data, "outperformance_lstm")
    all_values = standard_values + outperformance_values
    best_value = max(all_values)

    x_positions = np.arange(len(WINDOW_ORDER))
    width = 0.36

    fig, axis = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    standard_bars = axis.bar(
        x_positions - width / 2,
        standard_values,
        width,
        label=MODEL_LABELS["standard_lstm"],
    )
    outperformance_bars = axis.bar(
        x_positions + width / 2,
        outperformance_values,
        width,
        label=MODEL_LABELS["outperformance_lstm"],
    )

    for bars, model_name, values in [
        (standard_bars, "standard_lstm", standard_values),
        (outperformance_bars, "outperformance_lstm", outperformance_values),
    ]:
        for bar, value in zip(bars, values):
            style_bar(
                bar,
                value,
                MODEL_COLORS[model_name],
                is_best=value == best_value,
            )
            add_value_label(axis, bar, value)

    axis.axhline(0, color="#111827", linewidth=1.8)
    axis.text(
        x_positions[-1] + 0.2,
        1.4,
        "0 = Buy-and-Hold",
        fontsize=12,
        color="#374151",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(WINDOW_LABELS, fontsize=15, fontweight="bold")
    axis.set_ylabel("Difference vs. Buy-and-Hold (percentage points)", fontsize=15)
    axis.tick_params(axis="y", labelsize=13)
    axis.grid(axis="y", alpha=0.2)
    axis.set_ylim(min(all_values) - 8, max(all_values) + 12)

    legend_handles = [
        mpatches.Patch(color=MODEL_COLORS["standard_lstm"], label=MODEL_LABELS["standard_lstm"]),
        mpatches.Patch(color=MODEL_COLORS["outperformance_lstm"], label=MODEL_LABELS["outperformance_lstm"]),
        mpatches.Patch(
            facecolor="#fecaca",
            edgecolor="#991b1b",
            hatch="//",
            label="Schlechter als Buy-and-Hold",
        ),
    ]
    axis.legend(handles=legend_handles, loc="upper left", fontsize=13)

    fig.suptitle(
        "Walk-Forward Robustness vs. Buy-and-Hold",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Standard-LSTM Top-K vs. Outperformance-LSTM Top-K across market phases",
        ha="center",
        fontsize=14,
        color="#374151",
    )
    fig.text(
        0.5,
        0.035,
        "Alles ueber 0 schlaegt Buy-and-Hold. Alles unter 0 ist schlechter als Buy-and-Hold.",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )

    plt.tight_layout(rect=[0.04, 0.08, 1, 0.9])
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Presentation plot saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
