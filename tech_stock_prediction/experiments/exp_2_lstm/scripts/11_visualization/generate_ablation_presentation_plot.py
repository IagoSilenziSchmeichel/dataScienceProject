"""
Create a simple presentation plot for the LSTM feature ablation.

The goal is not a full scientific overview. The plot should be readable in a
presentation within a few seconds.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = EXPERIMENT_ROOT / "data" / "processed" / "lstm_feature_ablation_summary.csv"
OUTPUT_FILE = EXPERIMENT_ROOT / "plots" / "lstm_feature_ablation_presentation.png"


FEATURE_LABELS = {
    "Technical only": "Technical",
    "Technical + Volume": "Volume",
    "Technical + Market": "Market",
    "Technical + Relative Strength": "Relative\nStrength",
    "Final Feature Set": "Final Set",
}

MODEL_LABELS = {
    "standard_lstm": "Standard-LSTM",
    "outperformance_lstm": "Outperformance-LSTM",
}


def prepare_plot_data(data, model_name):
    """
    Keep only the feature groups that are easy to explain in a presentation.
    """
    rows = []

    for full_name, short_name in FEATURE_LABELS.items():
        match = data[
            (data["model_name"] == model_name)
            & (data["feature_group"] == full_name)
        ]

        if match.empty:
            continue

        row = match.iloc[0]

        rows.append(
            {
                "feature_group": short_name,
                "difference_pct": row["difference"] * 100,
                "best_top_k": int(row["best_top_k"]),
            }
        )

    return pd.DataFrame(rows)


def bar_colors(values):
    """
    Positive values are green, negative values are red.
    The best bar is highlighted in blue.
    """
    colors = []
    best_index = values.idxmax()

    for index, value in values.items():
        if index == best_index:
            colors.append("#2563eb")
        elif value >= 0:
            colors.append("#16a34a")
        else:
            colors.append("#dc2626")

    return colors


def add_labels(axis, values):
    """
    Add short percentage labels above or below each bar.
    """
    for index, value in enumerate(values):
        offset = 2.2 if value >= 0 else -3.5
        va = "bottom" if value >= 0 else "top"
        axis.text(
            index,
            value + offset,
            f"{value:+.1f} pp",
            ha="center",
            va=va,
            fontsize=13,
            fontweight="bold",
        )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE)

    standard_data = prepare_plot_data(data, "standard_lstm")
    outperformance_data = prepare_plot_data(data, "outperformance_lstm")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.patch.set_facecolor("white")

    for axis, model_name, plot_data in [
        (axes[0], "standard_lstm", standard_data),
        (axes[1], "outperformance_lstm", outperformance_data),
    ]:
        values = plot_data["difference_pct"]
        axis.bar(
            plot_data["feature_group"],
            values,
            color=bar_colors(values),
            width=0.65,
        )

        axis.axhline(0, color="#111827", linewidth=1.5)
        axis.set_title(MODEL_LABELS[model_name], fontsize=20, fontweight="bold")
        axis.set_xlabel("")
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelsize=13)
        axis.tick_params(axis="y", labelsize=13)

        add_labels(axis, values)

        best_row = plot_data.loc[values.idxmax()]
        axis.text(
            0.02,
            0.93,
            f"Best: {best_row['feature_group'].replace(chr(10), ' ')} "
            f"(Top {best_row['best_top_k']})",
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            color="#2563eb",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#eff6ff",
                "edgecolor": "#bfdbfe",
            },
        )

    axes[0].set_ylabel("Difference vs. Buy-and-Hold (percentage points)", fontsize=15)
    axes[0].set_ylim(-55, 65)

    fig.suptitle(
        "Feature Ablation: Which Feature Groups Help?",
        fontsize=24,
        fontweight="bold",
        y=0.97,
    )

    fig.text(
        0.5,
        0.035,
        "Mehr Features sind nicht automatisch besser. Market Features verbessern das "
        "Standard-LSTM, Relative-Strength-Features verbessern das Outperformance-LSTM.",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )

    plt.tight_layout(rect=[0.03, 0.08, 1, 0.92])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Presentation plot saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
