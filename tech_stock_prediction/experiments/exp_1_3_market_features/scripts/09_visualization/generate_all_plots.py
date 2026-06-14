

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from project_paths import (
    BACKTEST_RESULTS_FILE,
    FEATURE_COLUMNS_FILE,
    FEATURE_DATA_FILE,
    MODEL_FILE,
    PLOTS_DIR,
    RAW_DATA_FILE,
    TEST_FILE,
    TEST_PREDICTIONS_FILE,
    TRAIN_FILE,
    VALIDATION_FILE,
)


DPI = 300
TARGET_COLUMN = "Target"
PREDICTION_COLUMN = "Prediction"


def warn(message):
    print(f"WARNING: {message}")


def save_plot(fig, filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def load_csv(path, parse_dates=True):
    if not path.exists():
        warn(f"Missing input file: {path}")
        return None

    date_cols = ["Date"] if parse_dates else None
    try:
        return pd.read_csv(path, parse_dates=date_cols)
    except Exception as exc:
        warn(f"Could not load {path}: {exc}")
        return None


def load_model():
    if not MODEL_FILE.exists():
        warn(f"Missing input file: {MODEL_FILE}")
        return None

    try:
        return joblib.load(MODEL_FILE)
    except Exception as exc:
        warn(f"Could not load model {MODEL_FILE}: {exc}")
        return None


def load_feature_columns(data=None):
    if FEATURE_COLUMNS_FILE.exists():
        columns = [line.strip() for line in FEATURE_COLUMNS_FILE.read_text().splitlines() if line.strip()]
        if columns:
            return columns

    warn(f"Missing or empty feature list: {FEATURE_COLUMNS_FILE}")
    if data is None:
        return []

    excluded = {
        "Date",
        "Ticker",
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
        TARGET_COLUMN,
        PREDICTION_COLUMN,
    }
    return [col for col in data.select_dtypes(include=[np.number]).columns if col not in excluded]


def metric_dict(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_backtest_totals(backtest_data):
    if backtest_data is None:
        return None

    required = {"Date", "Strategy_Return", "Buy_And_Hold_Return"}
    if not required.issubset(backtest_data.columns):
        warn("Backtest totals skipped because required return columns are missing.")
        return None

    daily_returns = backtest_data.groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]].mean()
    return {
        "ML Strategy": (1 + daily_returns["Strategy_Return"]).prod() - 1,
        "Buy and Hold": (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1,
    }


def plot_stock_prices(raw_data):
    if raw_data is None:
        return
    required = {"Date", "Ticker", "Close"}
    if not required.issubset(raw_data.columns):
        warn("01_stock_prices.png skipped because Date, Ticker or Close is missing.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    for ticker, group in raw_data.sort_values("Date").groupby("Ticker"):
        ax.plot(group["Date"], group["Close"], linewidth=1.6, label=ticker)

    ax.set_title("Historical Closing Prices of 10 Tech Stocks", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    ax.legend(title="Ticker", ncol=2, frameon=True)
    ax.grid(True, alpha=0.25)
    save_plot(fig, "01_stock_prices.png")


def plot_target_distribution(feature_data):
    if feature_data is None:
        return
    if TARGET_COLUMN not in feature_data.columns:
        warn("02_target_distribution.png skipped because Target is missing.")
        return

    counts = feature_data[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    labels = ["0: Next day not up", "1: Next day up"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, counts.values, color=["#C9504D", "#2F7D5F"])
    ax.set_title("Target Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Number of rows")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, padding=4)
    save_plot(fig, "02_target_distribution.png")


def plot_dataset_split_timeline(train_data, validation_data, test_data):
    datasets = [
        ("Train", train_data, "#2F6B9A"),
        ("Validation", validation_data, "#D08B2C"),
        ("Test", test_data, "#6B9A4A"),
    ]
    if any(data is None for _, data, _ in datasets):
        warn("03_dataset_split_timeline.png skipped because at least one split file is missing.")
        return
    if any("Date" not in data.columns for _, data, _ in datasets):
        warn("03_dataset_split_timeline.png skipped because Date is missing in a split file.")
        return

    fig, ax = plt.subplots(figsize=(14, 6.2))
    timeline_start = min(data["Date"].min() for _, data, _ in datasets)
    timeline_end = max(data["Date"].max() for _, data, _ in datasets)
    label_box = {
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "#DDDDDD",
        "alpha": 0.95,
    }

    for y_pos, (name, data, color) in enumerate(datasets):
        start = data["Date"].min()
        end = data["Date"].max()
        start_offset = (8, 18)
        start_ha = "left"
        start_va = "bottom"
        if name in {"Validation", "Test"}:
            start_offset = (-8, -34)
            start_ha = "right"
            start_va = "top"

        ax.hlines(y=y_pos, xmin=start, xmax=end, linewidth=14, color=color, label=name)
        ax.scatter([start, end], [y_pos, y_pos], color=color, s=70, zorder=3)
        ax.annotate(
            f"Start\n{start.date().isoformat()}",
            xy=(start, y_pos),
            xytext=start_offset,
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha=start_ha,
            va=start_va,
            color="#222222",
            bbox=label_box,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2},
        )
        ax.annotate(
            f"Ende\n{end.date().isoformat()}",
            xy=(end, y_pos),
            xytext=(8, 18),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#222222",
            bbox=label_box,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2},
        )

    ax.set_title("Dataset Split Timeline", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([name for name, _, _ in datasets])
    ax.set_xlim(timeline_start - pd.Timedelta(days=180), timeline_end + pd.Timedelta(days=180))
    ax.set_ylim(-0.65, len(datasets) - 0.05)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, "03_dataset_split_timeline.png")


def plot_daily_return_distribution(feature_data):
    if feature_data is None:
        return
    if "Daily_Return" not in feature_data.columns:
        warn("04_daily_return_distribution.png skipped because Daily_Return is missing.")
        return

    returns = feature_data["Daily_Return"].dropna() * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(returns, bins=80, color="#365C7D", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#111111", linewidth=1.2, label="0%")
    ax.set_title("Distribution of Daily Returns", fontsize=16, fontweight="bold")
    ax.set_xlabel("Daily return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "04_daily_return_distribution.png")


def plot_feature_correlation_heatmap(feature_data):
    if feature_data is None:
        return

    feature_columns = [col for col in load_feature_columns(feature_data) if col in feature_data.columns]
    if len(feature_columns) < 2:
        warn("05_feature_correlation_heatmap.png skipped because fewer than two feature columns are available.")
        return

    corr = feature_data[feature_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    save_plot(fig, "05_feature_correlation_heatmap.png")


def plot_feature_importance(feature_data):
    model = load_model()
    if model is None:
        return

    feature_columns = load_feature_columns(feature_data)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        warn("06_feature_importance.png skipped because the model has no feature_importances_.")
        return
    if len(feature_columns) != len(importances):
        warn("06_feature_importance.png skipped because feature count does not match model importances.")
        return

    data = pd.DataFrame({"Feature": feature_columns, "Importance": importances}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(data["Feature"], data["Importance"], color="#4C78A8")
    ax.set_title("Random Forest Feature Importances", fontsize=16, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    save_plot(fig, "06_feature_importance.png")


def plot_confusion_matrix(predictions):
    if predictions is None:
        return
    required = {TARGET_COLUMN, PREDICTION_COLUMN}
    if not required.issubset(predictions.columns):
        warn("07_confusion_matrix.png skipped because Target or Prediction is missing.")
        return

    matrix = confusion_matrix(predictions[TARGET_COLUMN], predictions[PREDICTION_COLUMN], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix - Test Set", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0: not up", "1: up"])
    ax.set_yticklabels(["0: not up", "1: up"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_plot(fig, "07_confusion_matrix.png")


def get_test_metric_comparison(predictions):
    if predictions is None:
        return None
    required = {TARGET_COLUMN, PREDICTION_COLUMN}
    if not required.issubset(predictions.columns):
        warn("Metric comparison unavailable because Target or Prediction is missing.")
        return None

    y_true = predictions[TARGET_COLUMN]
    return {
        "Random Forest": metric_dict(y_true, predictions[PREDICTION_COLUMN]),
        "Always-Up Baseline": metric_dict(y_true, pd.Series(1, index=predictions.index)),
    }


def plot_metric_comparison(predictions):
    metrics_by_model = get_test_metric_comparison(predictions)
    if metrics_by_model is None:
        warn("08_metrics_random_forest_vs_always_up.png skipped.")
        return

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    rf_values = [metrics_by_model["Random Forest"][name] for name in metric_names]
    baseline_values = [metrics_by_model["Always-Up Baseline"][name] for name in metric_names]
    ax.bar(x - width / 2, rf_values, width, label="Random Forest", color="#2F6B9A")
    ax.bar(x + width / 2, baseline_values, width, label="Always-Up Baseline", color="#C9504D")
    ax.set_title("Random Forest vs Always-Up Baseline", fontsize=16, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "08_metrics_random_forest_vs_always_up.png")


def plot_backtest_bar_comparison(backtest_data):
    totals = compute_backtest_totals(backtest_data)
    if totals is None:
        warn("09_backtest_bar_comparison.png skipped.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = list(totals.keys())
    values = [totals[label] * 100 for label in labels]
    bars = ax.bar(labels, values, color=["#2F6B9A", "#C9504D"])
    ax.set_title("Backtest Total Return Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Total return (%)")
    ax.axhline(0, color="#111111", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
    save_plot(fig, "09_backtest_bar_comparison.png")


def plot_cumulative_returns(backtest_data):
    if backtest_data is None:
        return
    required = {"Date", "Strategy_Return", "Buy_And_Hold_Return"}
    if not required.issubset(backtest_data.columns):
        warn("10_cumulative_returns.png skipped because required return columns are missing.")
        return

    daily_returns = backtest_data.groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]].mean()
    cumulative = (1 + daily_returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(cumulative.index, cumulative["Strategy_Return"] * 100, label="ML Strategy", linewidth=2.2, color="#2F6B9A")
    ax.plot(
        cumulative.index,
        cumulative["Buy_And_Hold_Return"] * 100,
        label="Buy and Hold",
        linewidth=2.2,
        color="#C9504D",
    )
    ax.set_title("Cumulative Returns Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    save_plot(fig, "10_cumulative_returns.png")


def plot_prediction_distribution(predictions):
    if predictions is None:
        return
    if PREDICTION_COLUMN not in predictions.columns:
        warn("11_prediction_distribution.png skipped because Prediction is missing.")
        return

    counts = predictions[PREDICTION_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Predicted 0", "Predicted 1"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts.values, color=["#C9504D", "#2F7D5F"])
    ax.set_title("Prediction Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Number of predictions")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, padding=4)
    save_plot(fig, "11_prediction_distribution.png")


def plot_accuracy_by_ticker(predictions):
    if predictions is None:
        return
    required = {"Ticker", TARGET_COLUMN, PREDICTION_COLUMN}
    if not required.issubset(predictions.columns):
        warn("12_accuracy_by_ticker.png skipped because Ticker, Target or Prediction is missing.")
        return

    accuracy = predictions.groupby("Ticker").apply(
        lambda group: accuracy_score(group[TARGET_COLUMN], group[PREDICTION_COLUMN]),
        include_groups=False,
    )
    accuracy = accuracy.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(accuracy.index, accuracy.values, color="#4C78A8")
    ax.set_title("Test Accuracy by Ticker", fontsize=16, fontweight="bold")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in accuracy.values], padding=3, fontsize=8)
    save_plot(fig, "12_accuracy_by_ticker.png")


def plot_returns_by_ticker(backtest_data):
    if backtest_data is None:
        return
    required = {"Ticker", "Strategy_Return"}
    if not required.issubset(backtest_data.columns):
        warn("13_returns_by_ticker.png skipped because Ticker or Strategy_Return is missing.")
        return

    returns = backtest_data.groupby("Ticker")["Strategy_Return"].apply(lambda series: (1 + series).prod() - 1)
    returns = returns.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(returns.index, returns.values * 100, color="#2F6B9A")
    ax.set_title("ML Strategy Return by Ticker", fontsize=16, fontweight="bold")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Strategy return (%)")
    ax.axhline(0, color="#111111", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value * 100:.1f}%" for value in returns.values], padding=3, fontsize=8)
    save_plot(fig, "13_returns_by_ticker.png")


def plot_dashboard(predictions, backtest_data):
    metrics_by_model = get_test_metric_comparison(predictions)
    totals = compute_backtest_totals(backtest_data)

    if predictions is None or metrics_by_model is None or totals is None:
        warn("14_model_summary_dashboard.png skipped because required metrics or backtest data are unavailable.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Summary Dashboard", fontsize=18, fontweight="bold")

    models = ["Random Forest", "Always-Up Baseline"]
    accuracy_values = [metrics_by_model[model]["Accuracy"] for model in models]
    f1_values = [metrics_by_model[model]["F1-Score"] for model in models]
    colors = ["#2F6B9A", "#C9504D"]

    axes[0, 0].bar(models, accuracy_values, color=colors)
    axes[0, 0].set_title("Accuracy Comparison")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(models, f1_values, color=colors)
    axes[0, 1].set_title("F1 Comparison")
    axes[0, 1].set_ylabel("F1-Score")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(list(totals.keys()), [value * 100 for value in totals.values()], color=colors)
    axes[1, 0].set_title("Backtest Total Return")
    axes[1, 0].set_ylabel("Total return (%)")
    axes[1, 0].axhline(0, color="#111111", linewidth=1)
    axes[1, 0].grid(axis="y", alpha=0.25)

    counts = predictions[PREDICTION_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    axes[1, 1].bar(["Predicted 0", "Predicted 1"], counts.values, color=["#C9504D", "#2F7D5F"])
    axes[1, 1].set_title("Prediction Distribution")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(axis="y", alpha=0.25)

    save_plot(fig, "14_model_summary_dashboard.png")


def plot_validation_vs_test_metrics(validation_data, predictions):
    model = load_model()
    if model is None or validation_data is None or predictions is None:
        warn("15_validation_vs_test_metrics.png skipped because model, validation or test predictions are missing.")
        return

    feature_columns = load_feature_columns(validation_data)
    required_validation = set(feature_columns + [TARGET_COLUMN])
    if not required_validation.issubset(validation_data.columns):
        warn("15_validation_vs_test_metrics.png skipped because validation feature columns are missing.")
        return
    if not {TARGET_COLUMN, PREDICTION_COLUMN}.issubset(predictions.columns):
        warn("15_validation_vs_test_metrics.png skipped because test Target or Prediction is missing.")
        return

    validation_predictions = model.predict(validation_data[feature_columns])
    validation_metrics = metric_dict(validation_data[TARGET_COLUMN], validation_predictions)
    test_metrics = metric_dict(predictions[TARGET_COLUMN], predictions[PREDICTION_COLUMN])

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, [validation_metrics[name] for name in metric_names], width, label="Validation", color="#D08B2C")
    ax.bar(x + width / 2, [test_metrics[name] for name in metric_names], width, label="Test", color="#2F6B9A")
    ax.set_title("Validation vs Test Metrics - Random Forest", fontsize=16, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "15_validation_vs_test_metrics.png")


def plot_always_buy_vs_model(backtest_data):
    if backtest_data is None:
        warn("17_always_buy_vs_model.png skipped because backtest results are missing.")
        return

    required = {"Date", "Strategy_Return", "Buy_And_Hold_Return"}
    if not required.issubset(backtest_data.columns):
        warn("17_always_buy_vs_model.png skipped because required return columns are missing.")
        return

    daily_returns = backtest_data.groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]].mean()
    cumulative = (1 + daily_returns).cumprod() - 1
    totals = {
        "Unser Modell": (1 + daily_returns["Strategy_Return"]).prod() - 1,
        "baseline": (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1,
    }

    fig, (ax_line, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [1.7, 1]},
        facecolor="white",
    )
    fig.suptitle("Vergleich: baseline vs. unser Modell", fontsize=18, fontweight="bold")

    ax_line.plot(
        cumulative.index,
        cumulative["Strategy_Return"] * 100,
        label="Unser Modell",
        linewidth=2.4,
        color="#2F6B9A",
    )
    ax_line.plot(
        cumulative.index,
        cumulative["Buy_And_Hold_Return"] * 100,
        label="baseline",
        linewidth=2.4,
        color="#C9504D",
    )
    ax_line.set_title("Kumulierte Rendite im Testzeitraum", fontsize=14, fontweight="bold")
    ax_line.set_xlabel("Datum")
    ax_line.set_ylabel("Kumulierte Rendite (%)")
    ax_line.legend()
    ax_line.grid(True, alpha=0.25)

    labels = list(totals.keys())
    values = [totals[label] * 100 for label in labels]
    bars = ax_bar.bar(labels, values, color=["#2F6B9A", "#C9504D"], width=0.55)
    ax_bar.set_title("Gesamtrendite", fontsize=14, fontweight="bold")
    ax_bar.set_xlabel("Strategie")
    ax_bar.set_ylabel("Gesamtrendite (%)")
    ax_bar.axhline(0, color="#111111", linewidth=1)
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4, fontweight="bold")

    save_plot(fig, "17_always_buy_vs_model.png")


def main():
    print("Generating presentation plots")
    print("============================")

    raw_data = load_csv(RAW_DATA_FILE)
    feature_data = load_csv(FEATURE_DATA_FILE)
    train_data = load_csv(TRAIN_FILE)
    validation_data = load_csv(VALIDATION_FILE)
    test_data = load_csv(TEST_FILE)
    predictions = load_csv(TEST_PREDICTIONS_FILE)
    backtest_data = load_csv(BACKTEST_RESULTS_FILE)

    plot_stock_prices(raw_data)
    plot_target_distribution(feature_data)
    plot_dataset_split_timeline(train_data, validation_data, test_data)
    plot_daily_return_distribution(feature_data)
    plot_feature_correlation_heatmap(feature_data)
    plot_feature_importance(feature_data)
    plot_confusion_matrix(predictions)
    plot_metric_comparison(predictions)
    plot_backtest_bar_comparison(backtest_data)
    plot_cumulative_returns(backtest_data)
    plot_prediction_distribution(predictions)
    plot_accuracy_by_ticker(predictions)
    plot_returns_by_ticker(backtest_data)
    plot_dashboard(predictions, backtest_data)
    plot_validation_vs_test_metrics(validation_data, predictions)
    plot_always_buy_vs_model(backtest_data)


if __name__ == "__main__":
    main()
