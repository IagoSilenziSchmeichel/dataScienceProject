"""
Generate plots for the LSTM experiment.

The plots are built from the CSV result files that are created by the LSTM
pipeline. They are meant for the project presentation and make the model
results easier to explain.
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
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

PROCESSED_DIR = EXPERIMENT_ROOT / "data" / "processed"
PLOTS_DIR = EXPERIMENT_ROOT / "plots"
DPI = 300


def load_csv(file_name, parse_dates=False):
    path = PROCESSED_DIR / file_name

    if not path.exists():
        print(f"Warning: missing file, plot skipped: {path}")
        return None

    if parse_dates:
        return pd.read_csv(path, parse_dates=["Date"])

    return pd.read_csv(path)


def save_plot(fig, file_name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / file_name
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def format_percent_axis(ax):
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")


def plot_training_history(training_history):
    if training_history is None:
        return

    required_columns = {"epoch", "train_loss", "validation_f1"}
    if not required_columns.issubset(training_history.columns):
        print("Warning: training history plot skipped because columns are missing.")
        return

    fig, first_axis = plt.subplots(figsize=(11, 6))
    second_axis = first_axis.twinx()

    first_axis.plot(
        training_history["epoch"],
        training_history["train_loss"],
        marker="o",
        color="#2F6B9A",
        label="Train Loss",
    )
    second_axis.plot(
        training_history["epoch"],
        training_history["validation_f1"],
        marker="o",
        color="#D08B2C",
        label="Validation F1",
    )

    first_axis.set_title("LSTM Training Verlauf", fontsize=15, fontweight="bold")
    first_axis.set_xlabel("Epoch")
    first_axis.set_ylabel("Train Loss")
    second_axis.set_ylabel("Validation F1")
    first_axis.grid(True, alpha=0.25)

    lines = first_axis.get_lines() + second_axis.get_lines()
    labels = [line.get_label() for line in lines]
    first_axis.legend(lines, labels, loc="best")

    save_plot(fig, "01_lstm_training_history.png")


def plot_test_metrics(predictions):
    if predictions is None:
        return

    required_columns = {"Actual", "Prediction"}
    if not required_columns.issubset(predictions.columns):
        print("Warning: test metrics plot skipped because columns are missing.")
        return

    y_true = predictions["Actual"]
    y_pred = predictions["Prediction"]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(metrics.keys(), metrics.values(), color=["#365C7D", "#4F8F6F", "#D08B2C", "#8A5A83"])

    ax.set_title("LSTM Testmetriken", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in metrics.values()], padding=4)

    save_plot(fig, "02_lstm_test_metrics.png")


def plot_confusion_matrix(predictions):
    if predictions is None:
        return

    required_columns = {"Actual", "Prediction"}
    if not required_columns.issubset(predictions.columns):
        print("Warning: confusion matrix plot skipped because columns are missing.")
        return

    matrix = confusion_matrix(predictions["Actual"], predictions["Prediction"])

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("LSTM Confusion Matrix", fontsize=15, fontweight="bold")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0: nicht steigend", "1: steigend"])
    ax.set_yticklabels(["0: nicht steigend", "1: steigend"])

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="#111111",
                fontweight="bold",
            )

    save_plot(fig, "03_lstm_confusion_matrix.png")


def plot_backtest_cumulative(backtest_results):
    if backtest_results is None:
        return

    required_columns = {"Date", "Strategy_Cumulative", "Buy_Hold_Cumulative"}
    if not required_columns.issubset(backtest_results.columns):
        print("Warning: cumulative backtest plot skipped because columns are missing.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        backtest_results["Date"],
        backtest_results["Strategy_Cumulative"],
        color="#2F6B9A",
        linewidth=2,
        label="LSTM Strategie",
    )
    ax.plot(
        backtest_results["Date"],
        backtest_results["Buy_Hold_Cumulative"],
        color="#D08B2C",
        linewidth=2,
        label="Buy-and-Hold",
    )

    ax.set_title("Kumulierte Rendite: LSTM vs. Buy-and-Hold", fontsize=15, fontweight="bold")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Kapitalfaktor")
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_plot(fig, "04_lstm_cumulative_backtest.png")


def plot_threshold_results(threshold_results):
    if threshold_results is None:
        return

    required_columns = {"Threshold", "Strategy_Return", "Buy_And_Hold_Return", "Predicted_Up_Share"}
    if not required_columns.issubset(threshold_results.columns):
        print("Warning: threshold plot skipped because columns are missing.")
        return

    fig, first_axis = plt.subplots(figsize=(11, 6))
    second_axis = first_axis.twinx()

    first_axis.plot(
        threshold_results["Threshold"],
        threshold_results["Strategy_Return"],
        marker="o",
        linewidth=2,
        color="#2F6B9A",
        label="Strategy Return",
    )
    first_axis.plot(
        threshold_results["Threshold"],
        threshold_results["Buy_And_Hold_Return"],
        linestyle="--",
        linewidth=2,
        color="#D08B2C",
        label="Buy-and-Hold",
    )
    second_axis.plot(
        threshold_results["Threshold"],
        threshold_results["Predicted_Up_Share"],
        marker="s",
        color="#8A5A83",
        label="Predicted-Up-Share",
    )

    first_axis.set_title("Threshold Backtest", fontsize=15, fontweight="bold")
    first_axis.set_xlabel("Threshold")
    first_axis.set_ylabel("Rendite")
    second_axis.set_ylabel("Anteil Kaufsignale")
    first_axis.grid(True, alpha=0.25)
    format_percent_axis(first_axis)
    format_percent_axis(second_axis)

    lines = first_axis.get_lines() + second_axis.get_lines()
    labels = [line.get_label() for line in lines]
    first_axis.legend(lines, labels, loc="best")

    save_plot(fig, "05_lstm_threshold_backtest.png")


def plot_top_k_results(top_k_results):
    if top_k_results is None:
        return

    required_columns = {"top_k", "strategy_return", "buy_and_hold_return", "difference"}
    if not required_columns.issubset(top_k_results.columns):
        print("Warning: Top-K plot skipped because columns are missing.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        top_k_results["top_k"].astype(str),
        top_k_results["strategy_return"],
        color="#4F8F6F",
        label="Top-K Strategie",
    )
    buy_and_hold = top_k_results["buy_and_hold_return"].iloc[0]
    ax.axhline(
        buy_and_hold,
        color="#D08B2C",
        linestyle="--",
        linewidth=2,
        label="Buy-and-Hold",
    )

    ax.set_title("Top-K Backtest", fontsize=15, fontweight="bold")
    ax.set_xlabel("Top K Aktien pro Tag")
    ax.set_ylabel("Gesamtrendite")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    format_percent_axis(ax)
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in top_k_results["strategy_return"]], padding=4)

    save_plot(fig, "06_lstm_top_k_backtest.png")


def plot_tuning_results(tuning_results):
    if tuning_results is None:
        return

    required_columns = {"sequence_length", "hidden_size", "dropout", "learning_rate", "f1_score"}
    if not required_columns.issubset(tuning_results.columns):
        print("Warning: tuning plot skipped because columns are missing.")
        return

    data = tuning_results.copy().head(8)
    data["Configuration"] = [
        f"SL {row.sequence_length}, H {row.hidden_size}, D {row.dropout}, LR {row.learning_rate}"
        for row in data.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(data["Configuration"], data["f1_score"], color="#365C7D")
    ax.invert_yaxis()
    ax.set_title("LSTM Tuning: Validation F1", fontsize=15, fontweight="bold")
    ax.set_xlabel("Validation F1")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in data["f1_score"]], padding=4)

    save_plot(fig, "07_lstm_tuning_f1.png")


def plot_model_comparison_summary(summary):
    if summary is None:
        return

    required_columns = {
        "model_name",
        "period_type",
        "decision_rule",
        "strategy_return",
        "buy_and_hold_return",
    }
    if not required_columns.issubset(summary.columns):
        print("Warning: model comparison plot skipped because columns are missing.")
        return

    data = summary[
        (summary["period_type"] == "native_period")
        & (summary["decision_rule"].isin(["prediction_1", "best_top_k_probability"]))
    ].copy()

    if data.empty:
        print("Warning: model comparison plot skipped because no rows matched.")
        return

    data["Label"] = data["model_name"] + " | " + data["decision_rule"]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(data["Label"], data["strategy_return"], color="#4F8F6F", label="Strategie")
    ax.scatter(
        data["buy_and_hold_return"],
        data["Label"],
        color="#D08B2C",
        marker="D",
        s=70,
        label="Buy-and-Hold gleicher Zeitraum",
        zorder=3,
    )

    ax.set_title("Gesamtvergleich LSTM-Modelle", fontsize=15, fontweight="bold")
    ax.set_xlabel("Rendite")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    format_percent_axis(ax)
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in data["strategy_return"]], padding=4)

    save_plot(fig, "08_lstm_model_comparison_summary.png")


def main():
    print("Generating LSTM plots")
    print("=====================")

    training_history = load_csv("lstm_training_history.csv")
    predictions = load_csv("lstm_test_predictions.csv", parse_dates=True)
    backtest_results = load_csv("lstm_backtest_results.csv", parse_dates=True)
    threshold_results = load_csv(Path(PARAMS["RESULTS"]["THRESHOLD_RESULTS_FILE"]).name)
    top_k_results = load_csv(Path(PARAMS["RESULTS"]["STANDARD_TOP_K_FILE"]).name)
    tuning_results = load_csv(Path(PARAMS["RESULTS"]["TUNING_RESULTS_FILE"]).name)
    comparison_summary = load_csv(Path(PARAMS["RESULTS"]["COMPARISON_SUMMARY_FILE"]).name)

    plot_training_history(training_history)
    plot_test_metrics(predictions)
    plot_confusion_matrix(predictions)
    plot_backtest_cumulative(backtest_results)
    plot_threshold_results(threshold_results)
    plot_top_k_results(top_k_results)
    plot_tuning_results(tuning_results)
    plot_model_comparison_summary(comparison_summary)

    print("\nPlot generation completed.")


if __name__ == "__main__":
    main()
