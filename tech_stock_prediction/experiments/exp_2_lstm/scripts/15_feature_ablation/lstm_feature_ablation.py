"""
Feature group ablation for the LSTM experiment.

This script answers one question:
Which feature groups really improve the LSTM backtest?

It keeps the model architecture fixed and only changes the input features.
The scaler is fitted only on train data to avoid data leakage.
"""

from datetime import timedelta
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import yaml
from formatting import format_decimal, format_percent, save_csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

TRANSACTION_COST = 0.001
TRADING_DAYS_PER_YEAR = 252

TECHNICAL_FEATURES = [
    "Daily_Return",
    "Lag_1_Return",
    "Lag_3_Return",
    "Lag_7_Return",
    "RSI_14",
    "MACD",
]

VOLATILITY_FEATURES = [
    "RollingVolatility_7",
    "RollingVolatility_30",
    "High_Low_Range",
]

VOLUME_FEATURES = [
    "Volume_Change",
    "Volume_Ratio_20",
]

MOMENTUM_TREND_FEATURES = [
    "Momentum_5",
    "Momentum_20",
    "Distance_to_MA_200",
]

MARKET_FEATURES = [
    "QQQ_Return",
    "VIX_Change",
    "QQQ_Momentum_20",
    "QQQ_Distance_to_MA200",
]

RELATIVE_STRENGTH_FEATURES = [
    "Relative_Return_QQQ",
    "Relative_Momentum_20_QQQ",
]

FEATURE_GROUPS = {
    "Technical only": TECHNICAL_FEATURES,
    "Technical + Volatility": TECHNICAL_FEATURES + VOLATILITY_FEATURES,
    "Technical + Volume": TECHNICAL_FEATURES + VOLUME_FEATURES,
    "Technical + Momentum/Trend": TECHNICAL_FEATURES + MOMENTUM_TREND_FEATURES,
    "Technical + Market": TECHNICAL_FEATURES + MARKET_FEATURES,
    "Technical + Relative Strength": TECHNICAL_FEATURES + RELATIVE_STRENGTH_FEATURES,
    "Final Feature Set": (
        TECHNICAL_FEATURES
        + VOLATILITY_FEATURES
        + VOLUME_FEATURES
        + MOMENTUM_TREND_FEATURES
        + MARKET_FEATURES
        + RELATIVE_STRENGTH_FEATURES
    ),
}

TARGETS = {
    "standard_lstm": "Target",
    "outperformance_lstm": "Outperform_QQQ_Target",
}


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_output, (hidden_state, cell_state) = self.lstm(x)
        last_hidden_state = hidden_state[-1]
        logits = self.output_layer(last_hidden_state)

        return logits.squeeze(1)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(file_path, split_name):
    data = pd.read_csv(file_path, parse_dates=["Date"])
    data["Split"] = split_name

    return data.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def download_one_market_ticker(ticker, start_date, end_date):
    print(f"Downloading market data for {ticker}...")
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        print(f"Warning: no market data found for {ticker}.")
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])

    return data[["Date", "Close"]].rename(columns={"Close": ticker})


def download_market_data(start_date, end_date):
    qqq_ticker = PARAMS["MARKET"]["QQQ_TICKER"]
    vix_ticker = PARAMS["MARKET"]["VIX_TICKER"]

    qqq = download_one_market_ticker(qqq_ticker, start_date, end_date)
    vix = download_one_market_ticker(vix_ticker, start_date, end_date)

    if qqq.empty:
        raise ValueError("QQQ market data is required for the ablation.")

    market = qqq.copy()

    if vix.empty:
        market["VIX"] = np.nan
    else:
        market = market.merge(vix.rename(columns={vix_ticker: "VIX"}), on="Date", how="outer")

    market = market.rename(columns={qqq_ticker: "QQQ_Close"})
    market = market.sort_values("Date").reset_index(drop=True)

    market["QQQ_Return"] = market["QQQ_Close"].pct_change()
    market["VIX_Change"] = market["VIX"].pct_change().fillna(0.0)
    market["QQQ_Momentum_20"] = market["QQQ_Close"] / market["QQQ_Close"].shift(20) - 1

    qqq_ma200 = market["QQQ_Close"].rolling(200).mean()
    market["QQQ_Distance_to_MA200"] = market["QQQ_Close"] / qqq_ma200 - 1

    # These next-day returns are used only for targets, never as features.
    market["Next_Day_QQQ_Return"] = market["QQQ_Close"].shift(-1) / market["QQQ_Close"] - 1

    return market


def add_market_and_relative_features(data, market):
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    data["Stock_Daily_Return_Raw"] = data.groupby("Ticker")["Close"].pct_change()
    data["Stock_Momentum_20_Raw"] = data.groupby("Ticker")["Close"].transform(
        lambda values: values / values.shift(20) - 1
    )

    merged = data.merge(market, on="Date", how="left")
    merged["Relative_Return_QQQ"] = merged["Stock_Daily_Return_Raw"] - merged["QQQ_Return"]
    merged["Relative_Momentum_20_QQQ"] = merged["Stock_Momentum_20_Raw"] - merged["QQQ_Momentum_20"]
    merged["Outperform_QQQ_Target"] = np.where(
        merged["Next_Day_QQQ_Return"].notna(),
        (merged["Future_Return"] > merged["Next_Day_QQQ_Return"]).astype(int),
        np.nan,
    )

    return merged


def scale_data(train_data, validation_data, test_data, feature_columns):
    scaler = StandardScaler()
    train_scaled = train_data.copy()
    validation_scaled = validation_data.copy()
    test_scaled = test_data.copy()

    train_scaled[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    validation_scaled[feature_columns] = scaler.transform(validation_data[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test_data[feature_columns])

    return train_scaled, validation_scaled, test_scaled


def create_sequences(data, feature_columns, target_column, sequence_length):
    X_sequences = []
    y_values = []
    metadata_rows = []

    for ticker, ticker_data in data.groupby("Ticker"):
        ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)

        for index in range(sequence_length - 1, len(ticker_data)):
            start_index = index - sequence_length + 1
            end_index = index + 1
            X_sequences.append(ticker_data.iloc[start_index:end_index][feature_columns].values)
            y_values.append(ticker_data.iloc[index][target_column])
            metadata_rows.append(
                {
                    "Date": ticker_data.iloc[index]["Date"],
                    "Ticker": ticker,
                    "Close": ticker_data.iloc[index]["Close"],
                    "Actual": ticker_data.iloc[index][target_column],
                    "Future_Return": ticker_data.iloc[index]["Future_Return"],
                }
            )

    if not X_sequences:
        raise ValueError(f"No sequences created for target {target_column}.")

    return (
        np.array(X_sequences, dtype=np.float32),
        np.array(y_values, dtype=np.float32),
        pd.DataFrame(metadata_rows),
    )


def create_loader(X, y, batch_size, shuffle):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_probabilities(model, loader, device):
    model.eval()
    targets = []
    probabilities = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            probabilities.extend(batch_probabilities)
            targets.extend(y_batch.numpy())

    return np.array(targets).astype(int), np.array(probabilities)


def train_model(X_train, y_train, X_validation, y_validation):
    train_loader = create_loader(X_train, y_train, PARAMS["MODEL"]["BATCH_SIZE"], shuffle=True)
    validation_loader = create_loader(
        X_validation,
        y_validation,
        PARAMS["MODEL"]["BATCH_SIZE"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_size=X_train.shape[2],
        hidden_size=PARAMS["MODEL"]["HIDDEN_SIZE"],
        num_layers=PARAMS["MODEL"]["NUM_LAYERS"],
        dropout=PARAMS["MODEL"]["DROPOUT"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["MODEL"]["LEARNING_RATE"])
    best_validation_f1 = -1.0
    best_state_dict = None

    for epoch in range(1, PARAMS["MODEL"]["EPOCHS"] + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        y_validation_true, validation_probabilities = evaluate_probabilities(
            model,
            validation_loader,
            device,
        )
        validation_predictions = (validation_probabilities >= 0.5).astype(int)
        validation_f1 = f1_score(y_validation_true, validation_predictions, zero_division=0)

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state_dict = {key: value.cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"    Epoch {epoch:02d}/{PARAMS['MODEL']['EPOCHS']} | "
            f"Loss: {total_loss / len(train_loader):.5f} | "
            f"Val F1: {validation_f1:.5f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, device, best_validation_f1


def calculate_drawdown(daily_returns):
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1

    return drawdown.min()


def calculate_sharpe(daily_returns):
    if daily_returns.std() == 0:
        return 0.0

    return daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_daily_top_k_returns(predictions, top_k, transaction_cost):
    data = predictions.copy()
    data["Probability_Rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    data["Selected"] = data["Probability_Rank"] <= top_k

    rows = []
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

        if selected_tickers:
            turnover = number_of_trades / (2 * len(selected_tickers))
        else:
            turnover = 0.0

        cost = turnover * transaction_cost
        strategy_return_after_cost = strategy_return_before_cost - cost

        rows.append(
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

        previous_selected = selected_tickers

    return pd.DataFrame(rows).set_index("Date")


def calculate_top_k_metrics(predictions, top_k, transaction_cost):
    daily = calculate_daily_top_k_returns(predictions, top_k, transaction_cost)
    strategy_return = (1 + daily["Strategy_Return_After_Cost"]).prod() - 1
    strategy_return_before_cost = (1 + daily["Strategy_Return_Before_Cost"]).prod() - 1
    buy_and_hold_return = (1 + daily["Buy_And_Hold_Return"]).prod() - 1

    return {
        "top_k": top_k,
        "test_start": daily.index.min().date().isoformat(),
        "test_end": daily.index.max().date().isoformat(),
        "number_of_trading_days": len(daily),
        "transaction_cost": transaction_cost,
        "strategy_return": strategy_return,
        "strategy_return_before_cost": strategy_return_before_cost,
        "buy_and_hold_return": buy_and_hold_return,
        "difference": strategy_return - buy_and_hold_return,
        "sharpe_ratio": calculate_sharpe(daily["Strategy_Return_After_Cost"]),
        "buy_and_hold_sharpe": calculate_sharpe(daily["Buy_And_Hold_Return"]),
        "max_drawdown": calculate_drawdown(daily["Strategy_Return_After_Cost"]),
        "buy_and_hold_max_drawdown": calculate_drawdown(daily["Buy_And_Hold_Return"]),
        "volatility": daily["Strategy_Return_After_Cost"].std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "buy_and_hold_volatility": daily["Buy_And_Hold_Return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "number_of_trades": daily["Number_Of_Trades"].sum(),
        "average_turnover": daily["Turnover"].mean(),
        "average_number_of_positions": daily["Number_Of_Positions"].mean(),
    }


def run_one_experiment(data, feature_group_name, feature_columns, model_name, target_column):
    print("\n" + "=" * 80)
    print(f"Feature group: {feature_group_name}")
    print(f"Model target:   {model_name} ({target_column})")
    print("=" * 80)

    required_columns = feature_columns + [target_column, "Future_Return"]
    if target_column == "Outperform_QQQ_Target":
        required_columns.append("Next_Day_QQQ_Return")
    experiment_data = data.dropna(subset=required_columns).copy()

    train_data = experiment_data[experiment_data["Split"] == "train"].copy()
    validation_data = experiment_data[experiment_data["Split"] == "validation"].copy()
    test_data = experiment_data[experiment_data["Split"] == "test"].copy()

    train_data, validation_data, test_data = scale_data(
        train_data,
        validation_data,
        test_data,
        feature_columns,
    )

    sequence_length = PARAMS["MODEL"]["SEQUENCE_LENGTH"]
    X_train, y_train, _ = create_sequences(train_data, feature_columns, target_column, sequence_length)
    X_validation, y_validation, _ = create_sequences(
        validation_data,
        feature_columns,
        target_column,
        sequence_length,
    )
    X_test, y_test, test_metadata = create_sequences(test_data, feature_columns, target_column, sequence_length)

    print(f"Features: {len(feature_columns)}")
    print(f"X_train: {X_train.shape} | X_validation: {X_validation.shape} | X_test: {X_test.shape}")

    set_random_seed(PARAMS["MODEL"]["RANDOM_STATE"])
    model, device, best_validation_f1 = train_model(X_train, y_train, X_validation, y_validation)

    test_loader = create_loader(X_test, y_test, PARAMS["MODEL"]["BATCH_SIZE"], shuffle=False)
    y_true, probabilities = evaluate_probabilities(model, test_loader, device)
    predictions = (probabilities >= PARAMS["MODEL"]["PREDICTION_THRESHOLD"]).astype(int)

    prediction_data = test_metadata.copy()
    prediction_data["Probability"] = probabilities
    prediction_data["Prediction"] = predictions

    classification_metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "predicted_positive_share": predictions.mean(),
        "best_validation_f1": best_validation_f1,
    }

    rows = []
    for top_k in PARAMS["MODEL"].get("TOP_K_VALUES", [1, 2, 3, 4, 5]):
        row = calculate_top_k_metrics(prediction_data, top_k, TRANSACTION_COST)
        row.update(
            {
                "feature_group": feature_group_name,
                "model_name": model_name,
                "target": target_column,
                "feature_count": len(feature_columns),
                **classification_metrics,
            }
        )
        rows.append(row)

    return rows


def select_best_results(all_top_k_results):
    best_results = (
        all_top_k_results.sort_values("difference", ascending=False)
        .groupby(["model_name", "feature_group"], as_index=False)
        .first()
        .sort_values("difference", ascending=False)
        .reset_index(drop=True)
    )
    best_results = best_results.rename(columns={"top_k": "best_top_k"})

    return best_results


def make_labels(results):
    return results["model_name"] + "\n" + results["feature_group"]


def save_bar_plot(results, value_column, plot_key, title, ylabel, percent_axis=True):
    plot_file = EXPERIMENT_ROOT / PARAMS["RESULTS"][plot_key]
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plot_data = results.copy().sort_values(value_column, ascending=False)
    labels = make_labels(plot_data)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#4F8F6F" if value >= 0 else "#B85C5C" for value in plot_data[value_column]]
    bars = ax.bar(labels, plot_data[value_column], color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    if percent_axis:
        ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    if percent_axis:
        bar_labels = [f"{value:.1%}" for value in plot_data[value_column]]
    else:
        bar_labels = [f"{value:.2f}" for value in plot_data[value_column]]
    ax.bar_label(bars, labels=bar_labels, padding=4)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_file}")


def plot_ablation_results(best_results):
    save_bar_plot(
        best_results,
        "difference",
        "FEATURE_ABLATION_RETURNS_PLOT",
        "Feature Ablation: Best Top-K Difference vs Buy-and-Hold",
        "Strategy Return - Buy-and-Hold",
        percent_axis=True,
    )
    save_bar_plot(
        best_results,
        "sharpe_ratio",
        "FEATURE_ABLATION_SHARPE_PLOT",
        "Feature Ablation: Sharpe Ratio",
        "Sharpe Ratio",
        percent_axis=False,
    )
    save_bar_plot(
        best_results,
        "max_drawdown",
        "FEATURE_ABLATION_DRAWDOWN_PLOT",
        "Feature Ablation: Max Drawdown",
        "Max Drawdown",
        percent_axis=True,
    )
    save_bar_plot(
        best_results,
        "best_top_k",
        "FEATURE_ABLATION_TOP_K_PLOT",
        "Feature Ablation: Best Top-K",
        "Best Top-K",
        percent_axis=False,
    )


def get_result(best_results, model_name, feature_group):
    match = best_results[
        (best_results["model_name"] == model_name)
        & (best_results["feature_group"] == feature_group)
    ]
    if match.empty:
        return None

    return match.iloc[0]


def print_decision_summary(best_results):
    best_difference = best_results.sort_values("difference", ascending=False).iloc[0]
    best_sharpe = best_results.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_drawdown = best_results.sort_values("max_drawdown", ascending=False).iloc[0]

    print("\nDecision summary")
    print("================")
    print(
        "Best feature group by Difference: "
        f"{best_difference['model_name']} | {best_difference['feature_group']} | "
        f"Top {int(best_difference['best_top_k'])} | {format_percent(best_difference['difference'])}"
    )
    print(
        "Best feature group by Sharpe Ratio: "
        f"{best_sharpe['model_name']} | {best_sharpe['feature_group']} | "
        f"{format_decimal(best_sharpe['sharpe_ratio'])}"
    )
    print(
        "Best feature group by Max Drawdown: "
        f"{best_drawdown['model_name']} | {best_drawdown['feature_group']} | "
        f"{format_percent(best_drawdown['max_drawdown'])}"
    )

    for model_name in TARGETS:
        final_result = get_result(best_results, model_name, "Final Feature Set")
        technical_result = get_result(best_results, model_name, "Technical only")
        relative_result = get_result(best_results, model_name, "Technical + Relative Strength")

        if final_result is not None and technical_result is not None:
            final_delta = final_result["difference"] - technical_result["difference"]
            print(
                f"Final Feature Set vs Technical only ({model_name}): "
                f"{format_percent(final_delta)}"
            )

        if relative_result is not None and technical_result is not None:
            relative_delta = relative_result["difference"] - technical_result["difference"]
            helped = "yes" if relative_delta > 0 else "no"
            print(
                f"Relative Strength helped ({model_name}): {helped} "
                f"({format_percent(relative_delta)})"
            )

    if best_difference["difference"] > 0:
        print(
            "Recommended next feature group: "
            f"{best_difference['feature_group']} with {best_difference['model_name']}."
        )
    else:
        print(
            "Recommended next feature group: "
            f"{best_difference['feature_group']} with {best_difference['model_name']} "
            "as the best current variant, but it still does not beat Buy-and-Hold after costs."
        )


def print_summary(best_results):
    display_columns = [
        "model_name",
        "feature_group",
        "best_top_k",
        "f1_score",
        "strategy_return",
        "buy_and_hold_return",
        "difference",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_trades",
        "average_turnover",
    ]

    print("\nBest Top-K result per model and feature group")
    print("=============================================")
    print(
        best_results[display_columns].to_string(
            index=False,
            formatters={
                "f1_score": format_decimal,
                "strategy_return": format_percent,
                "buy_and_hold_return": format_percent,
                "difference": format_percent,
                "sharpe_ratio": format_decimal,
                "max_drawdown": format_percent,
                "average_turnover": format_decimal,
            },
        )
    )
    print_decision_summary(best_results)


def main():
    print("LSTM Feature Group Ablation")
    print("===========================")
    print(f"Transaction cost per turnover: {TRANSACTION_COST:.3%}")

    train_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["TRAIN_FILE"], "train")
    validation_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["VALIDATION_FILE"], "validation")
    test_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"], "test")
    all_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

    start_date = all_data["Date"].min() - timedelta(days=450)
    end_date = all_data["Date"].max() + timedelta(days=10)
    market_data = download_market_data(start_date.date().isoformat(), end_date.date().isoformat())
    all_data = add_market_and_relative_features(all_data, market_data)

    all_rows = []

    for feature_group_name, feature_columns in FEATURE_GROUPS.items():
        missing_features = [column for column in feature_columns if column not in all_data.columns]
        if missing_features:
            raise ValueError(f"{feature_group_name} has missing features: {missing_features}")

        for model_name, target_column in TARGETS.items():
            rows = run_one_experiment(
                all_data,
                feature_group_name,
                feature_columns,
                model_name,
                target_column,
            )
            all_rows.extend(rows)

    all_top_k_results = pd.DataFrame(all_rows)
    summary = select_best_results(all_top_k_results)
    output_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["FEATURE_ABLATION_SUMMARY_FILE"]
    save_csv(summary, output_file)
    plot_ablation_results(summary)
    print_summary(summary)

    print(f"\nFeature ablation summary saved to: {output_file}")


if __name__ == "__main__":
    main()
