"""
Three-year walk-forward robustness for the two final LSTM Top-K strategies.

This script retrains both models for each market phase using only about three
years of historical data before the test window. The plot is intentionally
simple for presentations:

- Standard-LSTM Top-K
- Outperformance-LSTM Top-K
- Difference vs. Buy-and-Hold
"""

from datetime import timedelta
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
    / "lstm_three_year_walk_forward_top_k_summary.csv"
)
OUTPUT_PLOT = (
    EXPERIMENT_ROOT
    / "plots"
    / "15_three_year_walk_forward_top_k_vs_buy_and_hold.png"
)

VALIDATION_DAYS = 126
TRAINING_YEARS = 3
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


def load_robustness_module():
    spec = importlib.util.spec_from_file_location("robustness_check", ROBUSTNESS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_three_year_data(robustness, data, feature_columns, target_column, window):
    """
    Use only about three years before the test window for train + validation.

    The last validation dates are separated from the training dates. The last
    historical date is removed so its next-day target cannot include the first
    test day.
    """
    required_columns = feature_columns + [target_column, "Future_Return"]

    if target_column == "Outperform_QQQ_Target":
        required_columns.append("Next_Day_QQQ_Return")

    data = data.dropna(subset=required_columns).copy()
    dates = pd.Series(sorted(data["Date"].drop_duplicates()))

    test_start = pd.Timestamp(window["start"])
    test_end = pd.Timestamp(window["end"])
    train_start = test_start - pd.DateOffset(years=TRAINING_YEARS)

    historical_dates = dates[(dates >= train_start) & (dates < test_start)].reset_index(drop=True)

    if len(historical_dates) <= VALIDATION_DAYS + robustness.PARAMS["MODEL"]["SEQUENCE_LENGTH"] + 30:
        raise ValueError(f"Not enough three-year history before {window['window']}.")

    historical_dates = historical_dates.iloc[:-1]
    validation_dates = set(historical_dates.iloc[-VALIDATION_DAYS:])
    train_dates = set(historical_dates.iloc[:-VALIDATION_DAYS])
    context_dates = set(historical_dates.iloc[-(robustness.PARAMS["MODEL"]["SEQUENCE_LENGTH"] - 1):])
    test_dates = set(dates[(dates >= test_start) & (dates <= test_end)])

    train_data = data[data["Date"].isin(train_dates)].copy()
    validation_data = data[data["Date"].isin(validation_dates)].copy()
    test_context_data = data[data["Date"].isin(context_dates | test_dates)].copy()

    return train_data, validation_data, test_context_data


def create_window_test_sequences(robustness, test_context_data, feature_columns, target_column, window):
    X_all, y_all, metadata_all = robustness.ablation.create_sequences(
        test_context_data,
        feature_columns,
        target_column,
        robustness.PARAMS["MODEL"]["SEQUENCE_LENGTH"],
    )

    start = pd.Timestamp(window["start"])
    end = pd.Timestamp(window["end"])
    mask = (metadata_all["Date"] >= start) & (metadata_all["Date"] <= end)

    return X_all[mask.to_numpy()], y_all[mask.to_numpy()], metadata_all[mask].reset_index(drop=True)


def train_and_predict_three_years(robustness, data, variant, window):
    feature_columns = variant["feature_columns"]
    target_column = variant["target"]

    train_data, validation_data, test_context_data = split_three_year_data(
        robustness,
        data,
        feature_columns,
        target_column,
        window,
    )

    train_data, validation_data, test_context_data = robustness.ablation.scale_data(
        train_data,
        validation_data,
        test_context_data,
        feature_columns,
    )

    X_train, y_train, _ = robustness.ablation.create_sequences(
        train_data,
        feature_columns,
        target_column,
        robustness.PARAMS["MODEL"]["SEQUENCE_LENGTH"],
    )
    X_validation, y_validation, _ = robustness.ablation.create_sequences(
        validation_data,
        feature_columns,
        target_column,
        robustness.PARAMS["MODEL"]["SEQUENCE_LENGTH"],
    )
    X_test, y_test, metadata = create_window_test_sequences(
        robustness,
        test_context_data,
        feature_columns,
        target_column,
        window,
    )

    print("\n" + "=" * 80)
    print(f"3-year training | {window['window']} | {variant['model_name']}")
    print("=" * 80)
    print(f"X_train: {X_train.shape} | X_validation: {X_validation.shape} | X_test: {X_test.shape}")

    robustness.ablation.set_random_seed(robustness.PARAMS["MODEL"]["RANDOM_STATE"])
    model, device, _ = robustness.ablation.train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    test_loader = robustness.ablation.create_loader(
        X_test,
        y_test,
        robustness.PARAMS["MODEL"]["BATCH_SIZE"],
        shuffle=False,
    )
    _, probabilities = robustness.ablation.evaluate_probabilities(model, test_loader, device)

    predictions = metadata.copy()
    predictions["Probability"] = probabilities
    predictions["window"] = window["window"]
    predictions["model_name"] = variant["model_name"]

    return predictions


def filter_common_dates(predictions_by_model):
    date_sets = []

    for predictions in predictions_by_model.values():
        date_sets.append(set(pd.to_datetime(predictions["Date"])))

    common_dates = sorted(set.intersection(*date_sets))

    if not common_dates:
        raise ValueError("No common dates found for both models.")

    common_start = common_dates[0]
    common_end = common_dates[-1]

    filtered = {}
    for model_name, predictions in predictions_by_model.items():
        dates = pd.to_datetime(predictions["Date"])
        filtered[model_name] = predictions[(dates >= common_start) & (dates <= common_end)].copy()

    return filtered, common_start, common_end


def calculate_results(robustness, data):
    rows = []

    for window in robustness.WINDOWS:
        predictions_by_model = {}

        for variant in robustness.MODEL_VARIANTS:
            predictions_by_model[variant["model_name"]] = {
                "variant": variant,
                "predictions": train_and_predict_three_years(robustness, data, variant, window),
            }

        common_predictions, common_start, common_end = filter_common_dates(
            {
                model_name: model_data["predictions"]
                for model_name, model_data in predictions_by_model.items()
            }
        )

        for model_name, predictions in common_predictions.items():
            variant = predictions_by_model[model_name]["variant"]
            metrics, _, _ = robustness.calculate_backtest_metrics(
                predictions,
                variant["default_top_k"],
                robustness.DEFAULT_TRANSACTION_COST,
            )

            rows.append(
                {
                    "model_name": variant["model_name"],
                    "window": window["window"],
                    "top_k": variant["default_top_k"],
                    "common_start": common_start.date().isoformat(),
                    "common_end": common_end.date().isoformat(),
                    "training_years": TRAINING_YEARS,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def style_bar(bar, value, color, is_best):
    if value < 0:
        bar.set_facecolor("#fecaca")
        bar.set_edgecolor("#991b1b")
        bar.set_hatch("//")
        bar.set_linewidth(2)
    else:
        bar.set_facecolor(color)
        bar.set_edgecolor(color)
        bar.set_linewidth(1.5)

    if is_best:
        bar.set_edgecolor("#f59e0b")
        bar.set_linewidth(4)


def create_plot(results):
    standard_values = (
        results[results["model_name"] == "standard_lstm"]
        .set_index("window")
        .loc[WINDOW_ORDER, "difference"]
        .to_numpy()
        * 100
    )
    outperformance_values = (
        results[results["model_name"] == "outperformance_lstm"]
        .set_index("window")
        .loc[WINDOW_ORDER, "difference"]
        .to_numpy()
        * 100
    )

    all_values = list(standard_values) + list(outperformance_values)
    best_value = max(all_values)

    x = np.arange(len(WINDOW_ORDER))
    width = 0.36

    fig, axis = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    standard_bars = axis.bar(x - width / 2, standard_values, width)
    outperformance_bars = axis.bar(x + width / 2, outperformance_values, width)

    for bars, model_name, values in [
        (standard_bars, "standard_lstm", standard_values),
        (outperformance_bars, "outperformance_lstm", outperformance_values),
    ]:
        for bar, value in zip(bars, values):
            style_bar(bar, value, MODEL_COLORS[model_name], value == best_value)
            offset = 1.8 if value >= 0 else -2.2
            va = "bottom" if value >= 0 else "top"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:+.1f} pp",
                ha="center",
                va=va,
                fontsize=14,
                fontweight="bold",
            )

    axis.axhline(0, color="#111827", linewidth=1.8)
    axis.text(x[-1] + 0.2, 1.4, "0 = Buy-and-Hold", fontsize=12, color="#374151")
    axis.set_xticks(x)
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
    axis.legend(handles=legend_handles, loc="upper right", fontsize=13)

    fig.suptitle(
        "Walk-Forward Robustness with 3-Year Training Windows",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Standard-LSTM Top-K vs. Outperformance-LSTM Top-K",
        ha="center",
        fontsize=14,
        color="#374151",
    )
    fig.text(
        0.5,
        0.035,
        "Jedes Fenster wird mit ca. 3 Jahren Historie davor trainiert. 0 bedeutet Buy-and-Hold.",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )

    plt.tight_layout(rect=[0.04, 0.08, 1, 0.9])
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    robustness = load_robustness_module()
    data = robustness.prepare_data()
    results = calculate_results(robustness, data)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    create_plot(results)

    print(f"3-year walk-forward summary saved to: {OUTPUT_CSV}")
    print(f"Presentation plot saved to: {OUTPUT_PLOT}")
    print(
        results[
            [
                "model_name",
                "window",
                "top_k",
                "common_start",
                "common_end",
                "strategy_return",
                "buy_and_hold_return",
                "difference",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
