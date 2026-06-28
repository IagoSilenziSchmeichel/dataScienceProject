"""
Create comparison plots for RF tuning presentation.

The plots compare exp_1_1 against exp_1_4 and show why the
validation-selected threshold is different from the default 0.50 threshold.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
BASELINE_ROOT = EXPERIMENTS_ROOT / "exp_1_1"
RF_TUNING_ROOT = EXPERIMENTS_ROOT / "exp_1_randomforest"
PLOTS_DIR = RF_TUNING_ROOT / "plots"

SCORE_PLOT_FILE = PLOTS_DIR / "18_rf_tuning_score_comparison.png"
THRESHOLD_PLOT_FILE = PLOTS_DIR / "19_threshold_comparison.png"
TOP3_THRESHOLD_PLOT_FILE = PLOTS_DIR / "20_top3_thresholds.png"
EXP11_EXP14_SCORE_PLOT_FILE = PLOTS_DIR / "21_exp1_1_vs_exp1_4_scores.png"


def load_predictions(experiment_root):
    return pd.read_csv(experiment_root / "data" / "processed" / "test_predictions.csv")


def prediction_metrics(predictions):
    return {
        "Accuracy": accuracy_score(predictions["Actual"], predictions["Prediction"]),
        "F1-Score": f1_score(predictions["Actual"], predictions["Prediction"], zero_division=0),
    }


def add_bar_labels(ax, bars, values, suffix="%"):
    ax.bar_label(
        bars,
        labels=[f"{value:.1f}{suffix}" for value in values],
        padding=4,
        fontsize=9,
        fontweight="bold",
    )


def plot_score_comparison():
    baseline_metrics = prediction_metrics(load_predictions(BASELINE_ROOT))
    tuned_metrics = prediction_metrics(load_predictions(RF_TUNING_ROOT))

    metric_names = ["Accuracy", "F1-Score"]
    labels = ["Exp 1\nBasis-Features", "RF Tuning\nMarket Features"]
    x_positions = range(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Modellguete: Exp 1 vs. RF Tuning", fontsize=16, fontweight="bold")

    for offset, metric_name, color in [
        (-width / 2, "Accuracy", "#2F6B9A"),
        (width / 2, "F1-Score", "#D08B2C"),
    ]:
        values = [
            baseline_metrics[metric_name] * 100,
            tuned_metrics[metric_name] * 100,
        ]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=metric_name,
            color=color,
        )
        add_bar_labels(ax, bars, values)

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 70)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.5,
        0.02,
        "RF Tuning wird gegen die urspruengliche Exp-1-Pipeline verglichen.",
        ha="center",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCORE_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {SCORE_PLOT_FILE}")


def plot_exp11_exp14_score_comparison():
    exp11_predictions = load_predictions(BASELINE_ROOT)
    exp14_predictions = load_predictions(RF_TUNING_ROOT)

    exp11_metrics = prediction_metrics(exp11_predictions)
    exp14_metrics = prediction_metrics(exp14_predictions)

    metric_names = ["Accuracy", "F1-Score"]
    labels = ["exp_1_1", "exp_1_4"]
    x_positions = range(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 6))
    fig.suptitle("Accuracy und F1-Score: exp_1_1 vs. exp_1_4", fontsize=16, fontweight="bold")

    for offset, metric_name, color in [
        (-width / 2, "Accuracy", "#2F6B9A"),
        (width / 2, "F1-Score", "#D08B2C"),
    ]:
        values = [
            exp11_metrics[metric_name] * 100,
            exp14_metrics[metric_name] * 100,
        ]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=metric_name,
            color=color,
        )
        add_bar_labels(ax, bars, values)

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 75)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.5,
        0.02,
        "exp_1_4 verbessert vor allem den F1-Score, waehrend die Accuracy nahezu gleich bleibt.",
        ha="center",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP11_EXP14_SCORE_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {EXP11_EXP14_SCORE_PLOT_FILE}")


def plot_threshold_comparison():
    processed_dir = RF_TUNING_ROOT / "data" / "processed"
    validation_results = pd.read_csv(
        processed_dir / "validation_selected_threshold_validation_results.csv"
    ).sort_values("Threshold")
    selected_test = pd.read_csv(
        processed_dir / "validation_selected_threshold_test_result.csv"
    ).iloc[0]
    direct_test_results = pd.read_csv(
        processed_dir / "random_forest_threshold_results.csv"
    ).sort_values("Threshold")

    selected_threshold = float(selected_test["Threshold"])
    standard_threshold = 0.50

    comparison = validation_results[
        ["Threshold", "Difference", "Signal_Rate"]
    ].rename(
        columns={
            "Difference": "Validation_Difference",
            "Signal_Rate": "Validation_Signal_Rate",
        }
    )
    comparison = comparison.merge(
        direct_test_results[["Threshold", "Difference", "Signal_Rate"]].rename(
            columns={
                "Difference": "Test_Difference",
                "Signal_Rate": "Test_Signal_Rate",
            }
        ),
        on="Threshold",
        how="inner",
    )

    fig, (ax_difference, ax_signal) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Threshold-Analyse: Validation vs. Test", fontsize=16, fontweight="bold")

    x_labels = comparison["Threshold"].map(lambda value: f"{value:.2f}")
    x_positions = range(len(comparison))

    ax_difference.plot(
        x_positions,
        comparison["Validation_Difference"] * 100,
        marker="o",
        linewidth=2.2,
        label="Validation",
        color="#2F6B9A",
    )
    ax_difference.plot(
        x_positions,
        comparison["Test_Difference"] * 100,
        marker="o",
        linewidth=2.2,
        label="Test",
        color="#C9504D",
    )
    ax_difference.axhline(0, color="#111111", linewidth=1)
    selected_matches = comparison.index[
        comparison["Threshold"].round(2) == round(selected_threshold, 2)
    ]
    standard_index = comparison.index[
        comparison["Threshold"].round(2) == standard_threshold
    ][0]
    if len(selected_matches) > 0:
        ax_difference.axvline(
            selected_matches[0],
            color="#2F7D5F",
            linestyle="--",
            linewidth=1.8,
            label=f"Selected {selected_threshold:.2f}",
        )
    ax_difference.axvline(
        standard_index,
        color="#666666",
        linestyle=":",
        linewidth=1.8,
        label="Standard 0.50",
    )
    ax_difference.set_title("Mehr-/Minderrendite gegen Buy and Hold", fontweight="bold")
    ax_difference.set_xlabel("Threshold")
    ax_difference.set_ylabel("Difference vs. Buy and Hold (%)")
    ax_difference.set_xticks(list(x_positions))
    ax_difference.set_xticklabels(x_labels, rotation=45)
    ax_difference.legend()
    ax_difference.grid(axis="y", alpha=0.25)

    width = 0.34
    validation_signal = comparison["Validation_Signal_Rate"] * 100
    test_signal = comparison["Test_Signal_Rate"] * 100
    validation_bars = ax_signal.bar(
        [x - width / 2 for x in x_positions],
        validation_signal,
        width=width,
        label="Validation",
        color="#2F6B9A",
    )
    test_bars = ax_signal.bar(
        [x + width / 2 for x in x_positions],
        test_signal,
        width=width,
        label="Test",
        color="#C9504D",
    )
    add_bar_labels(ax_signal, validation_bars, validation_signal)
    add_bar_labels(ax_signal, test_bars, test_signal)
    ax_signal.set_title("Wie viele Signale gehandelt werden", fontweight="bold")
    ax_signal.set_xlabel("Threshold")
    ax_signal.set_ylabel("Signal Rate (%)")
    ax_signal.set_ylim(0, 110)
    ax_signal.set_xticks(list(x_positions))
    ax_signal.set_xticklabels(x_labels, rotation=45)
    ax_signal.legend()
    ax_signal.grid(axis="y", alpha=0.25)

    fig.text(
        0.5,
        0.02,
        "Der Threshold 0.53 war auf Validation am besten, zeigt auf Test aber keine stabile Verbesserung gegen Buy and Hold.",
        ha="center",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(THRESHOLD_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {THRESHOLD_PLOT_FILE}")


def plot_top3_thresholds():
    processed_dir = RF_TUNING_ROOT / "data" / "processed"
    validation_results = pd.read_csv(
        processed_dir / "validation_selected_threshold_validation_results.csv"
    )

    top3 = validation_results.sort_values(
        ["Difference", "Strategy_Total_Return"],
        ascending=False,
    ).head(3)
    top3 = top3[[
        "Threshold",
        "Strategy_Total_Return",
        "Buy_Hold_Total_Return",
        "Difference",
        "Strategy_Sharpe",
        "Signal_Rate",
    ]]

    labels = top3["Threshold"].map(lambda value: f"{value:.2f}")
    x_positions = range(len(top3))

    fig, (ax_strategy, ax_context) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle("Top 3 Thresholds aus Validation", fontsize=16, fontweight="bold")

    strategy_values = top3["Strategy_Total_Return"] * 100
    bars = ax_strategy.bar(
        labels,
        strategy_values,
        color="#2F6B9A",
    )
    add_bar_labels(ax_strategy, bars, strategy_values)
    ax_strategy.axhline(
        top3["Buy_Hold_Total_Return"].iloc[0] * 100,
        color="#C9504D",
        linestyle="--",
        linewidth=1.8,
        label="Buy and Hold",
    )
    ax_strategy.set_title("Strategie-Rendite der besten 3 Thresholds", fontweight="bold")
    ax_strategy.set_xlabel("Threshold")
    ax_strategy.set_ylabel("Strategy Return (%)")
    ax_strategy.legend()
    ax_strategy.grid(axis="y", alpha=0.25)

    width = 0.34
    difference_values = top3["Difference"] * 100
    signal_values = top3["Signal_Rate"] * 100
    difference_bars = ax_context.bar(
        [x - width / 2 for x in x_positions],
        difference_values,
        width=width,
        label="Difference vs. Buy and Hold",
        color="#2F7D5F",
    )
    signal_bars = ax_context.bar(
        [x + width / 2 for x in x_positions],
        signal_values,
        width=width,
        label="Signal Rate",
        color="#D08B2C",
    )
    add_bar_labels(ax_context, difference_bars, difference_values)
    add_bar_labels(ax_context, signal_bars, signal_values)
    ax_context.axhline(0, color="#111111", linewidth=1)
    ax_context.set_title("Warum diese Thresholds vorne liegen", fontweight="bold")
    ax_context.set_xlabel("Threshold")
    ax_context.set_ylabel("Prozent (%)")
    ax_context.set_xticks(list(x_positions))
    ax_context.set_xticklabels(labels)
    ax_context.legend()
    ax_context.grid(axis="y", alpha=0.25)

    fig.text(
        0.5,
        0.02,
        "Die besten drei Thresholds wurden auf Validation bestimmt; niedrige Thresholds lassen hier mehr Signale zu.",
        ha="center",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TOP3_THRESHOLD_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {TOP3_THRESHOLD_PLOT_FILE}")


def main():
    plot_score_comparison()
    plot_exp11_exp14_score_comparison()
    plot_threshold_comparison()
    plot_top3_thresholds()


if __name__ == "__main__":
    main()
