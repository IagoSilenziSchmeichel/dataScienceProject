"""
Robustness checks for the two best LSTM variants.

No new model architecture is introduced here. The script retrains the same LSTM
logic on walk-forward windows and checks whether the results are stable.
"""

from datetime import timedelta
from pathlib import Path
import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from formatting import format_decimal, format_percent, save_csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))
ABLATION_SCRIPT = EXPERIMENT_ROOT / "scripts" / "15_feature_ablation" / "lstm_feature_ablation.py"

spec = importlib.util.spec_from_file_location("lstm_feature_ablation", ABLATION_SCRIPT)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)

TOP_K_VALUES = [1, 2, 3, 4, 5]
TRANSACTION_COSTS = [0.0, 0.0005, 0.001, 0.002]
DEFAULT_TRANSACTION_COST = 0.001
VALIDATION_DAYS = 126
TRADING_DAYS_PER_YEAR = 252

WINDOWS = [
    {"window": "2025_H1", "start": "2025-01-28", "end": "2025-06-30"},
    {"window": "2025_H2", "start": "2025-07-01", "end": "2025-12-31"},
    {"window": "2026_H1", "start": "2026-01-01", "end": "2026-06-12"},
]

MODEL_VARIANTS = [
    {
        "model_name": "standard_lstm",
        "feature_group": "Technical + Market",
        "target": "Target",
        "feature_columns": ablation.TECHNICAL_FEATURES + ablation.MARKET_FEATURES,
        "default_top_k": 2,
    },
    {
        "model_name": "outperformance_lstm",
        "feature_group": "Technical + Relative Strength",
        "target": "Outperform_QQQ_Target",
        "feature_columns": ablation.TECHNICAL_FEATURES + ablation.RELATIVE_STRENGTH_FEATURES,
        "default_top_k": 1,
    },
]


def to_percent(value):
    return f"{value:.2%}"


def to_decimal(value):
    return f"{value:.4f}"


def markdown_table(data, columns):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if "return" in column or "difference" in column or "drawdown" in column or "cost" in column:
                    values.append(to_percent(value))
                else:
                    values.append(to_decimal(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def prepare_data():
    train_data = ablation.load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["TRAIN_FILE"], "train")
    validation_data = ablation.load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["VALIDATION_FILE"], "validation")
    test_data = ablation.load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"], "test")
    all_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

    start_date = all_data["Date"].min() - timedelta(days=450)
    end_date = all_data["Date"].max() + timedelta(days=10)
    market_data = ablation.download_market_data(start_date.date().isoformat(), end_date.date().isoformat())
    all_data = ablation.add_market_and_relative_features(all_data, market_data)

    return all_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def split_walk_forward_data(data, feature_columns, target_column, window_start, window_end):
    required_columns = feature_columns + [target_column, "Future_Return"]
    if target_column == "Outperform_QQQ_Target":
        required_columns.append("Next_Day_QQQ_Return")

    data = data.dropna(subset=required_columns).copy()
    dates = pd.Series(sorted(data["Date"].drop_duplicates()))
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)

    historical_dates = dates[dates < start].reset_index(drop=True)
    if len(historical_dates) <= VALIDATION_DAYS + 30:
        raise ValueError(f"Not enough historical data before {window_start}.")

    # Drop the last historical date so its next-day target cannot include the
    # first test day.
    historical_dates = historical_dates.iloc[:-1]
    validation_dates = set(historical_dates.iloc[-VALIDATION_DAYS:])
    train_dates = set(historical_dates.iloc[:-VALIDATION_DAYS])

    context_dates = set(historical_dates.iloc[-(PARAMS["MODEL"]["SEQUENCE_LENGTH"] - 1):])
    test_dates = set(dates[(dates >= start) & (dates <= end)])

    train_data = data[data["Date"].isin(train_dates)].copy()
    validation_data = data[data["Date"].isin(validation_dates)].copy()
    test_context_data = data[data["Date"].isin(context_dates | test_dates)].copy()

    return train_data, validation_data, test_context_data


def create_window_test_sequences(test_context_data, feature_columns, target_column, window_start, window_end):
    X_all, y_all, metadata_all = ablation.create_sequences(
        test_context_data,
        feature_columns,
        target_column,
        PARAMS["MODEL"]["SEQUENCE_LENGTH"],
    )
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    mask = (metadata_all["Date"] >= start) & (metadata_all["Date"] <= end)

    return X_all[mask.to_numpy()], y_all[mask.to_numpy()], metadata_all[mask].reset_index(drop=True)


def train_and_predict(data, variant, window):
    feature_columns = variant["feature_columns"]
    target_column = variant["target"]

    train_data, validation_data, test_context_data = split_walk_forward_data(
        data,
        feature_columns,
        target_column,
        window["start"],
        window["end"],
    )
    train_data, validation_data, test_context_data = ablation.scale_data(
        train_data,
        validation_data,
        test_context_data,
        feature_columns,
    )

    X_train, y_train, _ = ablation.create_sequences(
        train_data,
        feature_columns,
        target_column,
        PARAMS["MODEL"]["SEQUENCE_LENGTH"],
    )
    X_validation, y_validation, _ = ablation.create_sequences(
        validation_data,
        feature_columns,
        target_column,
        PARAMS["MODEL"]["SEQUENCE_LENGTH"],
    )
    X_test, y_test, metadata = create_window_test_sequences(
        test_context_data,
        feature_columns,
        target_column,
        window["start"],
        window["end"],
    )

    print("\n" + "=" * 80)
    print(f"Walk-forward window: {window['window']} | {variant['model_name']} | {variant['feature_group']}")
    print("=" * 80)
    print(f"X_train: {X_train.shape} | X_validation: {X_validation.shape} | X_test: {X_test.shape}")

    ablation.set_random_seed(PARAMS["MODEL"]["RANDOM_STATE"])
    model, device, best_validation_f1 = ablation.train_model(X_train, y_train, X_validation, y_validation)
    test_loader = ablation.create_loader(X_test, y_test, PARAMS["MODEL"]["BATCH_SIZE"], shuffle=False)
    y_true, probabilities = ablation.evaluate_probabilities(model, test_loader, device)
    predictions = (probabilities >= PARAMS["MODEL"]["PREDICTION_THRESHOLD"]).astype(int)

    prediction_data = metadata.copy()
    prediction_data["Probability"] = probabilities
    prediction_data["Prediction"] = predictions
    prediction_data["window"] = window["window"]
    prediction_data["model_name"] = variant["model_name"]
    prediction_data["feature_group"] = variant["feature_group"]

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "best_validation_f1": best_validation_f1,
    }

    return prediction_data, metrics


def calculate_daily_top_k_details(predictions, top_k, transaction_cost):
    data = predictions.copy()
    data["Probability_Rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    data["Selected"] = data["Probability_Rank"] <= top_k

    daily_rows = []
    trade_rows = []
    previous_selected = set()

    for date, day_data in data.groupby("Date"):
        selected = day_data[day_data["Selected"]].copy()
        selected_tickers = set(selected["Ticker"])

        if selected.empty:
            strategy_return_before_cost = 0.0
        else:
            strategy_return_before_cost = selected["Future_Return"].mean()

        buy_and_hold_return = day_data["Future_Return"].mean()
        new_buys = len(selected_tickers - previous_selected)
        sells = len(previous_selected - selected_tickers)
        number_of_trades = new_buys + sells
        turnover = number_of_trades / (2 * len(selected_tickers)) if selected_tickers else 0.0
        cost = turnover * transaction_cost
        strategy_return_after_cost = strategy_return_before_cost - cost

        daily_rows.append(
            {
                "Date": date,
                "Strategy_Return_Before_Cost": strategy_return_before_cost,
                "Strategy_Return_After_Cost": strategy_return_after_cost,
                "Buy_And_Hold_Return": buy_and_hold_return,
                "Number_Of_Positions": len(selected_tickers),
                "Number_Of_Trades": number_of_trades,
                "Turnover": turnover,
                "Transaction_Cost": cost,
            }
        )

        position_count = max(len(selected), 1)
        for _, row in selected.iterrows():
            trade_rows.append(
                {
                    "Date": date,
                    "Ticker": row["Ticker"],
                    "Future_Return": row["Future_Return"],
                    "Contribution": row["Future_Return"] / position_count,
                    "Probability": row["Probability"],
                }
            )

        previous_selected = selected_tickers

    daily = pd.DataFrame(daily_rows).set_index("Date")
    trades = pd.DataFrame(trade_rows)

    return daily, trades


def calculate_backtest_metrics(predictions, top_k, transaction_cost):
    daily, trades = calculate_daily_top_k_details(predictions, top_k, transaction_cost)
    strategy_return = (1 + daily["Strategy_Return_After_Cost"]).prod() - 1
    strategy_return_before_cost = (1 + daily["Strategy_Return_Before_Cost"]).prod() - 1
    buy_and_hold_return = (1 + daily["Buy_And_Hold_Return"]).prod() - 1

    return {
        "top_k": top_k,
        "transaction_cost": transaction_cost,
        "test_start": daily.index.min().date().isoformat(),
        "test_end": daily.index.max().date().isoformat(),
        "number_of_trading_days": len(daily),
        "strategy_return": strategy_return,
        "strategy_return_before_cost": strategy_return_before_cost,
        "buy_and_hold_return": buy_and_hold_return,
        "difference": strategy_return - buy_and_hold_return,
        "sharpe_ratio": ablation.calculate_sharpe(daily["Strategy_Return_After_Cost"]),
        "max_drawdown": ablation.calculate_drawdown(daily["Strategy_Return_After_Cost"]),
        "volatility": daily["Strategy_Return_After_Cost"].std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "number_of_trades": daily["Number_Of_Trades"].sum(),
        "average_turnover": daily["Turnover"].mean(),
        "average_number_of_positions": daily["Number_Of_Positions"].mean(),
    }, daily, trades


def run_sensitivity(predictions, variant, window, classification_metrics):
    rows = []

    for top_k in TOP_K_VALUES:
        for transaction_cost in TRANSACTION_COSTS:
            metrics, _, _ = calculate_backtest_metrics(predictions, top_k, transaction_cost)
            rows.append(
                {
                    "model_name": variant["model_name"],
                    "feature_group": variant["feature_group"],
                    "window": window["window"],
                    "default_top_k": variant["default_top_k"],
                    **classification_metrics,
                    **metrics,
                }
            )

    return rows


def concentration_summary(predictions, variant):
    metrics, daily, trades = calculate_backtest_metrics(
        predictions,
        variant["default_top_k"],
        DEFAULT_TRANSACTION_COST,
    )

    positive_daily_total = daily["Strategy_Return_After_Cost"].clip(lower=0).sum()
    top_5_days = daily["Strategy_Return_After_Cost"].clip(lower=0).sort_values(ascending=False).head(5).sum()
    top_5_day_share = top_5_days / positive_daily_total if positive_daily_total > 0 else 0.0

    if trades.empty:
        largest_ticker = ""
        largest_ticker_share = 0.0
        largest_ticker_contribution = 0.0
    else:
        ticker_contribution = trades.groupby("Ticker")["Contribution"].sum().sort_values(ascending=False)
        positive_ticker_total = ticker_contribution.clip(lower=0).sum()
        largest_ticker = ticker_contribution.index[0]
        largest_ticker_contribution = ticker_contribution.iloc[0]
        largest_ticker_share = (
            ticker_contribution.clip(lower=0).iloc[0] / positive_ticker_total
            if positive_ticker_total > 0
            else 0.0
        )

    return {
        "model_name": variant["model_name"],
        "feature_group": variant["feature_group"],
        "top_k": variant["default_top_k"],
        "transaction_cost": DEFAULT_TRANSACTION_COST,
        "strategy_return": metrics["strategy_return"],
        "buy_and_hold_return": metrics["buy_and_hold_return"],
        "difference": metrics["difference"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "volatility": metrics["volatility"],
        "number_of_trades": metrics["number_of_trades"],
        "average_turnover": metrics["average_turnover"],
        "largest_ticker": largest_ticker,
        "largest_ticker_contribution": largest_ticker_contribution,
        "largest_ticker_positive_share": largest_ticker_share,
        "top_5_day_positive_share": top_5_day_share,
        "largest_single_day_return": daily["Strategy_Return_After_Cost"].max(),
        "worst_single_day_return": daily["Strategy_Return_After_Cost"].min(),
    }


def plot_walk_forward(walk_forward):
    plot_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["ROBUSTNESS_WALK_FORWARD_PLOT"]
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    labels = walk_forward["model_name"] + "\n" + walk_forward["window"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, walk_forward["difference"], color="#4F8F6F")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Walk-Forward Difference vs Buy-and-Hold", fontweight="bold")
    ax.set_ylabel("Strategy Return - Buy-and-Hold")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in walk_forward["difference"]], padding=4)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_line(summary, group_column, value_column, plot_key, title, xlabel):
    plot_file = EXPERIMENT_ROOT / PARAMS["RESULTS"][plot_key]
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, model_data in summary.groupby("model_name"):
        grouped = model_data.groupby(group_column)[value_column].mean()
        ax.plot(grouped.index, grouped.values, marker="o", linewidth=2, label=model_name)

    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Difference vs Buy-and-Hold")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_concentration(concentration):
    plot_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["ROBUSTNESS_CONCENTRATION_PLOT"]
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    labels = concentration["model_name"] + "\n" + concentration["feature_group"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, concentration["largest_ticker_positive_share"], width, label="Largest ticker share")
    ax.bar(x + width / 2, concentration["top_5_day_positive_share"], width, label="Top 5 day share")
    ax.set_title("Return Concentration Check", fontweight="bold")
    ax.set_ylabel("Share of positive return")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(walk_forward, sensitivity, concentration):
    report_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["ROBUSTNESS_REPORT_FILE"]
    report_file.parent.mkdir(parents=True, exist_ok=True)

    selected_cost = sensitivity[
        (sensitivity["transaction_cost"] == DEFAULT_TRANSACTION_COST)
        & (sensitivity["top_k"] == sensitivity["default_top_k"])
    ].copy()

    top_k_sensitivity = (
        sensitivity[sensitivity["transaction_cost"] == DEFAULT_TRANSACTION_COST]
        .groupby(["model_name", "feature_group", "top_k"], as_index=False)["difference"]
        .mean()
    )
    cost_sensitivity = (
        sensitivity[sensitivity["top_k"] == sensitivity["default_top_k"]]
        .groupby(["model_name", "feature_group", "transaction_cost"], as_index=False)["difference"]
        .mean()
    )

    lines = [
        "# LSTM Robustness Report",
        "",
        "This report checks whether the two best LSTM variants are stable across time windows, Top-K values, transaction costs and return concentration.",
        "",
        "## Tested Variants",
        "",
        "- `standard_lstm` with `Technical + Market`, default Top-K = 2",
        "- `outperformance_lstm` with `Technical + Relative Strength`, default Top-K = 1",
        "",
        "## Walk-Forward Summary",
        "",
        markdown_table(
            selected_cost[
                [
                    "model_name",
                    "feature_group",
                    "window",
                    "top_k",
                    "transaction_cost",
                    "strategy_return",
                    "buy_and_hold_return",
                    "difference",
                    "sharpe_ratio",
                    "max_drawdown",
                    "number_of_trades",
                    "average_turnover",
                ]
            ],
            [
                "model_name",
                "feature_group",
                "window",
                "top_k",
                "transaction_cost",
                "strategy_return",
                "buy_and_hold_return",
                "difference",
                "sharpe_ratio",
                "max_drawdown",
                "number_of_trades",
                "average_turnover",
            ],
        ),
        "",
        "## Top-K Sensitivity",
        "",
        markdown_table(
            top_k_sensitivity,
            ["model_name", "feature_group", "top_k", "difference"],
        ),
        "",
        "## Transaction Cost Sensitivity",
        "",
        markdown_table(
            cost_sensitivity,
            ["model_name", "feature_group", "transaction_cost", "difference"],
        ),
        "",
        "## Concentration Check",
        "",
        markdown_table(
            concentration[
                [
                    "model_name",
                    "feature_group",
                    "largest_ticker",
                    "largest_ticker_positive_share",
                    "top_5_day_positive_share",
                    "largest_single_day_return",
                    "worst_single_day_return",
                ]
            ],
            [
                "model_name",
                "feature_group",
                "largest_ticker",
                "largest_ticker_positive_share",
                "top_5_day_positive_share",
                "largest_single_day_return",
                "worst_single_day_return",
            ],
        ),
        "",
        "## Plots",
        "",
        "- `plots/lstm_robustness_walk_forward_difference.png`",
        "- `plots/lstm_robustness_top_k_sensitivity.png`",
        "- `plots/lstm_robustness_cost_sensitivity.png`",
        "- `plots/lstm_robustness_concentration.png`",
        "",
    ]

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {report_file}")


def main():
    print("LSTM Robustness Check")
    print("=====================")
    data = prepare_data()

    sensitivity_rows = []
    walk_forward_rows = []
    all_predictions = {variant["model_name"]: [] for variant in MODEL_VARIANTS}

    for variant in MODEL_VARIANTS:
        for window in WINDOWS:
            predictions, classification_metrics = train_and_predict(data, variant, window)
            all_predictions[variant["model_name"]].append(predictions)
            sensitivity_rows.extend(run_sensitivity(predictions, variant, window, classification_metrics))

    sensitivity = pd.DataFrame(sensitivity_rows)
    walk_forward = sensitivity[
        (sensitivity["transaction_cost"] == DEFAULT_TRANSACTION_COST)
        & (sensitivity["top_k"] == sensitivity["default_top_k"])
    ].copy()

    concentration_rows = []
    for variant in MODEL_VARIANTS:
        combined_predictions = pd.concat(all_predictions[variant["model_name"]], ignore_index=True)
        concentration_rows.append(concentration_summary(combined_predictions, variant))

    concentration = pd.DataFrame(concentration_rows)

    walk_forward_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["ROBUSTNESS_WALK_FORWARD_FILE"]
    sensitivity_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["ROBUSTNESS_SENSITIVITY_FILE"]
    concentration_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["ROBUSTNESS_CONCENTRATION_FILE"]

    save_csv(walk_forward, walk_forward_file)
    save_csv(sensitivity, sensitivity_file)
    save_csv(concentration, concentration_file)

    plot_walk_forward(walk_forward)
    plot_line(
        sensitivity[sensitivity["transaction_cost"] == DEFAULT_TRANSACTION_COST],
        "top_k",
        "difference",
        "ROBUSTNESS_TOP_K_PLOT",
        "Top-K Sensitivity at 0.10% Transaction Cost",
        "Top-K",
    )
    plot_line(
        sensitivity[sensitivity["top_k"] == sensitivity["default_top_k"]],
        "transaction_cost",
        "difference",
        "ROBUSTNESS_COST_PLOT",
        "Transaction Cost Sensitivity at Default Top-K",
        "Transaction Cost",
    )
    plot_concentration(concentration)
    write_report(walk_forward, sensitivity, concentration)

    print("\nWalk-forward summary")
    print("====================")
    print(
        walk_forward[
            [
                "model_name",
                "feature_group",
                "window",
                "top_k",
                "strategy_return",
                "buy_and_hold_return",
                "difference",
                "sharpe_ratio",
                "max_drawdown",
                "number_of_trades",
                "average_turnover",
            ]
        ].to_string(
            index=False,
            formatters={
                "strategy_return": format_percent,
                "buy_and_hold_return": format_percent,
                "difference": format_percent,
                "sharpe_ratio": format_decimal,
                "max_drawdown": format_percent,
                "average_turnover": format_decimal,
            },
        )
    )

    print("\nConcentration summary")
    print("=====================")
    print(
        concentration[
            [
                "model_name",
                "feature_group",
                "largest_ticker",
                "largest_ticker_positive_share",
                "top_5_day_positive_share",
                "largest_single_day_return",
                "worst_single_day_return",
            ]
        ].to_string(
            index=False,
            formatters={
                "largest_ticker_positive_share": format_percent,
                "top_5_day_positive_share": format_percent,
                "largest_single_day_return": format_percent,
                "worst_single_day_return": format_percent,
            },
        )
    )

    print(f"\nWalk-forward saved to: {walk_forward_file}")
    print(f"Sensitivity saved to: {sensitivity_file}")
    print(f"Concentration saved to: {concentration_file}")


if __name__ == "__main__":
    main()
