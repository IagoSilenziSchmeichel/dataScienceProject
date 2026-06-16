"""
Train and evaluate an LSTM for market outperformance.

The standard LSTM predicts whether a stock goes up tomorrow.
This variant predicts whether a stock beats QQQ tomorrow.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

TARGET_COLUMN = "Outperform_QQQ_Target"
TOP_K_VALUES = PARAMS["MODEL"].get("TOP_K_VALUES", [1, 2, 3, 4, 5])

MARKET_FEATURES = [
    "QQQ_Return",
    "SPY_Return",
    "VIX_Change",
    "QQQ_Momentum_20",
    "SPY_Momentum_20",
    "QQQ_Distance_to_MA200",
    "SPY_Distance_to_MA200",
]

RELATIVE_FEATURES = [
    "Relative_Return_QQQ",
    "Relative_Return_SPY",
    "Relative_Momentum_20_QQQ",
    "Relative_Momentum_20_SPY",
]


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


def load_feature_columns(feature_path):
    return [line.strip() for line in open(feature_path, encoding="utf-8") if line.strip()]


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
    spy_ticker = PARAMS["MARKET"]["SPY_TICKER"]
    vix_ticker = PARAMS["MARKET"]["VIX_TICKER"]

    qqq = download_one_market_ticker(qqq_ticker, start_date, end_date)
    spy = download_one_market_ticker(spy_ticker, start_date, end_date)
    vix = download_one_market_ticker(vix_ticker, start_date, end_date)

    if qqq.empty or spy.empty:
        raise ValueError("QQQ and SPY market data are required for outperformance.")

    market = qqq.merge(spy, on="Date", how="outer")

    if vix.empty:
        market["VIX"] = np.nan
    else:
        market = market.merge(vix.rename(columns={vix_ticker: "VIX"}), on="Date", how="outer")

    market = market.rename(columns={qqq_ticker: "QQQ_Close", spy_ticker: "SPY_Close"})
    market = market.sort_values("Date").reset_index(drop=True)

    market["QQQ_Return"] = market["QQQ_Close"].pct_change()
    market["SPY_Return"] = market["SPY_Close"].pct_change()
    market["VIX_Change"] = market["VIX"].pct_change().fillna(0.0)

    market["QQQ_Momentum_20"] = market["QQQ_Close"] / market["QQQ_Close"].shift(20) - 1
    market["SPY_Momentum_20"] = market["SPY_Close"] / market["SPY_Close"].shift(20) - 1

    qqq_ma200 = market["QQQ_Close"].rolling(200).mean()
    spy_ma200 = market["SPY_Close"].rolling(200).mean()
    market["QQQ_Distance_to_MA200"] = market["QQQ_Close"] / qqq_ma200 - 1
    market["SPY_Distance_to_MA200"] = market["SPY_Close"] / spy_ma200 - 1

    # Next-day market returns are only used for the target, not as features.
    market["Next_Day_QQQ_Return"] = market["QQQ_Close"].shift(-1) / market["QQQ_Close"] - 1
    market["Next_Day_SPY_Return"] = market["SPY_Close"].shift(-1) / market["SPY_Close"] - 1

    return market


def add_outperformance_features(data, market):
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    data["Stock_Daily_Return_Raw"] = data.groupby("Ticker")["Close"].pct_change()
    data["Stock_Momentum_20_Raw"] = data.groupby("Ticker")["Close"].transform(
        lambda values: values / values.shift(20) - 1
    )

    merged = data.merge(market, on="Date", how="left")

    merged["Relative_Return_QQQ"] = merged["Stock_Daily_Return_Raw"] - merged["QQQ_Return"]
    merged["Relative_Return_SPY"] = merged["Stock_Daily_Return_Raw"] - merged["SPY_Return"]
    merged["Relative_Momentum_20_QQQ"] = merged["Stock_Momentum_20_Raw"] - merged["QQQ_Momentum_20"]
    merged["Relative_Momentum_20_SPY"] = merged["Stock_Momentum_20_Raw"] - merged["SPY_Momentum_20"]

    merged["Outperform_QQQ_Target"] = (
        merged["Future_Return"] > merged["Next_Day_QQQ_Return"]
    ).astype(int)
    merged["Outperform_SPY_Target"] = (
        merged["Future_Return"] > merged["Next_Day_SPY_Return"]
    ).astype(int)

    return merged


def scale_features(train_data, validation_data, test_data, feature_columns):
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
                    "Next_Day_QQQ_Return": ticker_data.iloc[index]["Next_Day_QQQ_Return"],
                    "Next_Day_SPY_Return": ticker_data.iloc[index]["Next_Day_SPY_Return"],
                }
            )

    if len(X_sequences) == 0:
        raise ValueError("No outperformance sequences were created.")

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


def evaluate_on_loader(model, loader, device):
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
    validation_loader = create_loader(X_validation, y_validation, PARAMS["MODEL"]["BATCH_SIZE"], shuffle=False)

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

        y_validation_true, validation_probabilities = evaluate_on_loader(model, validation_loader, device)
        validation_predictions = (validation_probabilities >= 0.5).astype(int)
        validation_f1 = f1_score(y_validation_true, validation_predictions, zero_division=0)

        print(
            f"Epoch {epoch:02d}/{PARAMS['MODEL']['EPOCHS']} | "
            f"Loss: {total_loss / len(train_loader):.5f} | "
            f"Val F1: {validation_f1:.5f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state_dict = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, device, best_validation_f1


def calculate_metrics(y_true, probabilities):
    threshold = PARAMS["MODEL"].get("PREDICTION_THRESHOLD", 0.5)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "predicted_outperform_share": predictions.mean(),
        "confusion_true_0_pred_0": matrix[0, 0],
        "confusion_true_0_pred_1": matrix[0, 1],
        "confusion_true_1_pred_0": matrix[1, 0],
        "confusion_true_1_pred_1": matrix[1, 1],
    }, predictions


def calculate_simple_backtest(predictions):
    data = predictions.copy()
    data["Strategy_Return"] = np.where(data["Prediction"] == 1, data["Future_Return"], 0.0)
    data["Buy_And_Hold_Return"] = data["Future_Return"]
    daily_returns = data.groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]].mean()

    strategy_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
    buy_and_hold_return = (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1

    return strategy_return, buy_and_hold_return, daily_returns


def calculate_top_k_results(predictions):
    rows = []

    for top_k in TOP_K_VALUES:
        data = predictions.copy()
        data["Probability_Rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
        selected = data[data["Probability_Rank"] <= top_k].copy()

        daily_strategy_returns = selected.groupby("Date")["Future_Return"].mean()
        daily_buy_hold_returns = data.groupby("Date")["Future_Return"].mean()
        daily_positions = selected.groupby("Date")["Ticker"].count()
        daily_results = pd.concat(
            [
                daily_strategy_returns.rename("Strategy_Return"),
                daily_buy_hold_returns.rename("Buy_And_Hold_Return"),
                daily_positions.rename("Number_Of_Positions"),
            ],
            axis=1,
        ).fillna(0)

        strategy_return = (1 + daily_results["Strategy_Return"]).prod() - 1
        buy_and_hold_return = (1 + daily_results["Buy_And_Hold_Return"]).prod() - 1

        rows.append(
            {
                "model_name": "outperformance_lstm",
                "top_k": top_k,
                "test_start": daily_results.index.min().date().isoformat(),
                "test_end": daily_results.index.max().date().isoformat(),
                "strategy_return": strategy_return,
                "buy_and_hold_return": buy_and_hold_return,
                "difference": strategy_return - buy_and_hold_return,
                "average_number_of_positions": daily_positions.mean(),
                "number_of_trading_days": len(daily_results),
            }
        )

    return pd.DataFrame(rows)


def calculate_top_k_daily_returns(predictions, top_k):
    data = predictions.copy()
    data["Probability_Rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    selected = data[data["Probability_Rank"] <= top_k].copy()

    daily_strategy_returns = selected.groupby("Date")["Future_Return"].mean()
    daily_buy_hold_returns = data.groupby("Date")["Future_Return"].mean()

    return pd.concat(
        [
            daily_strategy_returns.rename("Strategy_Return"),
            daily_buy_hold_returns.rename("Buy_And_Hold_Return"),
        ],
        axis=1,
    ).fillna(0)


def calculate_standard_top_k_results_on_dates(dates):
    standard_predictions = load_standard_predictions_for_plot()
    common_dates = pd.to_datetime(pd.Series(dates)).drop_duplicates()
    standard_predictions = standard_predictions[standard_predictions["Date"].isin(common_dates)].copy()

    rows = []
    for top_k in TOP_K_VALUES:
        daily_returns = calculate_top_k_daily_returns(standard_predictions, top_k)
        strategy_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
        buy_and_hold_return = (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1

        rows.append(
            {
                "model_name": "standard_lstm",
                "top_k": top_k,
                "test_start": daily_returns.index.min().date().isoformat(),
                "test_end": daily_returns.index.max().date().isoformat(),
                "strategy_return": strategy_return,
                "buy_and_hold_return": buy_and_hold_return,
                "difference": strategy_return - buy_and_hold_return,
                "average_number_of_positions": top_k,
                "number_of_trading_days": len(daily_returns),
            }
        )

    return pd.DataFrame(rows)


def load_standard_predictions_for_plot():
    predictions_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["PREDICTIONS_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"]
    predictions = pd.read_csv(predictions_file, parse_dates=["Date"])
    test_data = pd.read_csv(test_file, parse_dates=["Date"])

    test_data = test_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    test_data["Next_Close"] = test_data.groupby("Ticker")["Close"].shift(-1)

    predictions = predictions.merge(test_data[["Date", "Ticker", "Next_Close"]], on=["Date", "Ticker"], how="left")
    predictions = predictions.dropna(subset=["Next_Close"]).copy()
    predictions["Future_Return"] = predictions["Next_Close"] / predictions["Close"] - 1

    return predictions


def plot_metrics(results):
    plot_file = EXPERIMENT_ROOT / "plots" / "09_outperformance_lstm_metrics.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    metric_values = [results[name] for name in metric_names]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(metric_names, metric_values, color=["#365C7D", "#4F8F6F", "#D08B2C", "#8A5A83"])
    ax.set_title("Outperformance LSTM Metrics", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in metric_values], padding=4)
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_file}")


def plot_top_k(top_k_results):
    plot_file = EXPERIMENT_ROOT / "plots" / "10_outperformance_top_k_return.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        top_k_results["top_k"].astype(str),
        top_k_results["strategy_return"],
        color="#4F8F6F",
        label="Outperformance Top-K",
    )
    buy_and_hold = top_k_results["buy_and_hold_return"].iloc[0]
    ax.axhline(buy_and_hold, color="#D08B2C", linestyle="--", linewidth=2, label="Buy-and-Hold")
    ax.set_title("Outperformance LSTM Top-K Backtest", fontsize=15, fontweight="bold")
    ax.set_xlabel("Top K Aktien pro Tag")
    ax.set_ylabel("Gesamtrendite")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in top_k_results["strategy_return"]], padding=4)
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_file}")


def plot_cumulative_comparison(outperformance_predictions, outperformance_top_k_results, standard_top_k_results):
    plot_file = EXPERIMENT_ROOT / "plots" / "11_outperformance_cumulative_comparison.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    standard_best_top_k = int(standard_top_k_results.sort_values("strategy_return", ascending=False).iloc[0]["top_k"])
    outperformance_best_top_k = int(
        outperformance_top_k_results.sort_values("strategy_return", ascending=False).iloc[0]["top_k"]
    )

    standard_predictions = load_standard_predictions_for_plot()
    standard_daily = calculate_top_k_daily_returns(standard_predictions, standard_best_top_k)
    outperformance_daily = calculate_top_k_daily_returns(outperformance_predictions, outperformance_best_top_k)

    common_dates = standard_daily.index.intersection(outperformance_daily.index)
    standard_daily = standard_daily.loc[common_dates]
    outperformance_daily = outperformance_daily.loc[common_dates]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        common_dates,
        (1 + standard_daily["Strategy_Return"]).cumprod(),
        label=f"Standard LSTM Top {standard_best_top_k}",
        color="#2F6B9A",
        linewidth=2,
    )
    ax.plot(
        common_dates,
        (1 + outperformance_daily["Strategy_Return"]).cumprod(),
        label=f"Outperformance LSTM Top {outperformance_best_top_k}",
        color="#4F8F6F",
        linewidth=2,
    )
    ax.plot(
        common_dates,
        (1 + outperformance_daily["Buy_And_Hold_Return"]).cumprod(),
        label="Buy-and-Hold",
        color="#D08B2C",
        linestyle="--",
        linewidth=2,
    )
    ax.set_title("Cumulative Return Comparison", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital factor")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_file}")


def print_final_comparison(top_k_results, standard_top_k_results):
    standard_best = standard_top_k_results.sort_values("strategy_return", ascending=False).iloc[0]
    outperformance_best = top_k_results.sort_values("strategy_return", ascending=False).iloc[0]

    print("\nStandard LSTM vs Outperformance LSTM on same dates")
    print("==================================================")
    print(f"Period: {outperformance_best['test_start']} to {outperformance_best['test_end']}")
    print(f"Standard LSTM Top {int(standard_best['top_k'])}: {format_percent(standard_best['strategy_return'])}")
    print(
        f"Outperformance LSTM Top {int(outperformance_best['top_k'])}: "
        f"{format_percent(outperformance_best['strategy_return'])}"
    )
    print(f"Buy-and-Hold: {format_percent(outperformance_best['buy_and_hold_return'])}")
    print(
        "Difference Outperformance - Standard: "
        f"{format_percent(outperformance_best['strategy_return'] - standard_best['strategy_return'])}"
    )


def main():
    print("Outperformance LSTM")
    print("===================")
    set_random_seed(PARAMS["MODEL"]["RANDOM_STATE"])

    train_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["TRAIN_FILE"], "train")
    validation_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["VALIDATION_FILE"], "validation")
    test_data = load_split(EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"], "test")
    all_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

    start_date = all_data["Date"].min() - timedelta(days=450)
    end_date = all_data["Date"].max() + timedelta(days=10)
    market = download_market_data(start_date.date().isoformat(), end_date.date().isoformat())
    data = add_outperformance_features(all_data, market)

    base_features = load_feature_columns(EXPERIMENT_ROOT / PARAMS["DATA"]["FEATURE_PATH"])
    feature_columns = base_features + MARKET_FEATURES + RELATIVE_FEATURES
    required_columns = feature_columns + [TARGET_COLUMN, "Future_Return", "Next_Day_QQQ_Return"]
    data = data.dropna(subset=required_columns).copy()

    train_data = data[data["Split"] == "train"].copy()
    validation_data = data[data["Split"] == "validation"].copy()
    test_data = data[data["Split"] == "test"].copy()
    train_data, validation_data, test_data = scale_features(train_data, validation_data, test_data, feature_columns)

    sequence_length = PARAMS["MODEL"]["SEQUENCE_LENGTH"]
    X_train, y_train, _ = create_sequences(train_data, feature_columns, TARGET_COLUMN, sequence_length)
    X_validation, y_validation, _ = create_sequences(validation_data, feature_columns, TARGET_COLUMN, sequence_length)
    X_test, y_test, test_metadata = create_sequences(test_data, feature_columns, TARGET_COLUMN, sequence_length)

    print("\nShapes:")
    print(f"X_train:      {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test:       {X_test.shape}")
    print(f"Features:     {len(feature_columns)}")

    model, device, best_validation_f1 = train_model(X_train, y_train, X_validation, y_validation)
    test_loader = create_loader(X_test, y_test, PARAMS["MODEL"]["BATCH_SIZE"], shuffle=False)
    y_true, probabilities = evaluate_on_loader(model, test_loader, device)
    metrics, predictions = calculate_metrics(y_true, probabilities)

    prediction_data = test_metadata.copy()
    prediction_data["Probability"] = probabilities
    prediction_data["Prediction"] = predictions

    strategy_return, buy_and_hold_return, daily_returns = calculate_simple_backtest(prediction_data)
    top_k_results = calculate_top_k_results(prediction_data)
    standard_common_top_k_results = calculate_standard_top_k_results_on_dates(prediction_data["Date"])

    result_row = {
        "model_name": "outperformance_lstm",
        "target": TARGET_COLUMN,
        "sequence_length": sequence_length,
        "hidden_size": PARAMS["MODEL"]["HIDDEN_SIZE"],
        "num_layers": PARAMS["MODEL"]["NUM_LAYERS"],
        "dropout": PARAMS["MODEL"]["DROPOUT"],
        "learning_rate": PARAMS["MODEL"]["LEARNING_RATE"],
        "best_validation_f1": best_validation_f1,
        "test_start": daily_returns.index.min().date().isoformat(),
        "test_end": daily_returns.index.max().date().isoformat(),
        "number_of_trading_days": len(daily_returns),
        **metrics,
        "simple_strategy_return": strategy_return,
        "buy_and_hold_return": buy_and_hold_return,
        "difference": strategy_return - buy_and_hold_return,
    }
    results = pd.DataFrame([result_row])

    best_top_k = top_k_results.sort_values("strategy_return", ascending=False).iloc[0]
    best_standard_top_k = standard_common_top_k_results.sort_values("strategy_return", ascending=False).iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "model_name": "standard_lstm",
                "decision_rule": "best_top_k_probability_same_dates",
                "top_k": best_standard_top_k["top_k"],
                "strategy_return": best_standard_top_k["strategy_return"],
                "buy_and_hold_return": best_standard_top_k["buy_and_hold_return"],
                "difference": best_standard_top_k["difference"],
            },
            {
                "model_name": "outperformance_lstm",
                "decision_rule": "prediction_1",
                "top_k": np.nan,
                "strategy_return": strategy_return,
                "buy_and_hold_return": buy_and_hold_return,
                "difference": strategy_return - buy_and_hold_return,
            },
            {
                "model_name": "outperformance_lstm",
                "decision_rule": "best_top_k_probability",
                "top_k": best_top_k["top_k"],
                "strategy_return": best_top_k["strategy_return"],
                "buy_and_hold_return": best_top_k["buy_and_hold_return"],
                "difference": best_top_k["difference"],
            },
        ]
    )

    predictions_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["OUTPERFORMANCE_PREDICTIONS_FILE"]
    results_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["OUTPERFORMANCE_RESULTS_FILE"]
    top_k_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["OUTPERFORMANCE_TOP_K_FILE"]
    comparison_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["OUTPERFORMANCE_COMPARISON_FILE"]

    save_csv(prediction_data, predictions_file)
    save_csv(results, results_file)
    save_csv(top_k_results, top_k_file)
    save_csv(comparison, comparison_file)

    print("\nOutperformance test results")
    print("===========================")
    print(
        results.to_string(
            index=False,
            formatters={
                "best_validation_f1": format_decimal,
                "accuracy": format_decimal,
                "precision": format_decimal,
                "recall": format_decimal,
                "f1_score": format_decimal,
                "predicted_outperform_share": format_decimal,
                "simple_strategy_return": format_percent,
                "buy_and_hold_return": format_percent,
                "difference": format_percent,
            },
        )
    )
    print("\nOutperformance Top-K results")
    print("============================")
    print(
        top_k_results.to_string(
            index=False,
            formatters={
                "strategy_return": format_percent,
                "buy_and_hold_return": format_percent,
                "difference": format_percent,
                "average_number_of_positions": format_decimal,
            },
        )
    )

    plot_metrics(result_row)
    plot_top_k(top_k_results)
    plot_cumulative_comparison(prediction_data, top_k_results, standard_common_top_k_results)
    print_final_comparison(top_k_results, standard_common_top_k_results)

    print(f"\nPredictions saved to: {predictions_file}")
    print(f"Results saved to: {results_file}")
    print(f"Top-K results saved to: {top_k_file}")
    print(f"Comparison saved to: {comparison_file}")


if __name__ == "__main__":
    main()
