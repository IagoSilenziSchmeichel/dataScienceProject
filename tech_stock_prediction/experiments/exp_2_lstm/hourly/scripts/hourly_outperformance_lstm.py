"""
Hourly Outperformance-LSTM pipeline for Alpaca Paper Trading.

This script is deliberately separated from the daily LSTM research pipeline.
It trains an hourly model on Yahoo Finance 1h bars and stores all artifacts
under experiments/exp_2_lstm/hourly/ so existing daily results stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.stock_universes import BENCHMARKS, UNIVERSES
from formatting import format_decimal, format_percent, save_csv


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
HOURLY_ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml", encoding="utf-8"))

TARGET_COLUMN = "Outperform_Benchmark_Target"
TOP_K_VALUES = [1, 3, 5]
HOURLY_PERIOD = "730d"  # Yahoo Finance's practical maximum for 1h bars.
TRADING_HOURS_PER_YEAR = 252 * 6.5
DEFAULT_TRANSACTION_COST = 0.001

FEATURE_FILE = HOURLY_ROOT / "conf" / "hourly_features.txt"
MODEL_FILE = HOURLY_ROOT / "models" / "hourly_outperformance_lstm_model.pth"
SCALER_FILE = HOURLY_ROOT / "models" / "hourly_outperformance_scaler.pkl"
METADATA_FILE = HOURLY_ROOT / "models" / "hourly_outperformance_metadata.json"
PREDICTIONS_FILE = HOURLY_ROOT / "data" / "hourly_outperformance_predictions.csv"
RESULTS_FILE = HOURLY_ROOT / "data" / "hourly_outperformance_results.csv"
TOP_K_FILE = HOURLY_ROOT / "data" / "hourly_outperformance_top_k_results.csv"
REPORT_FILE = HOURLY_ROOT / "reports" / "hourly_outperformance_report.md"


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
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


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_feature_columns() -> list[str]:
    return [line.strip() for line in FEATURE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_training_ticker_benchmarks() -> dict[str, str]:
    ticker_to_benchmark: dict[str, str] = {}
    for universe_name, tickers in UNIVERSES.items():
        benchmark = BENCHMARKS[universe_name]
        for ticker in tickers:
            ticker_to_benchmark.setdefault(ticker, benchmark)
    return ticker_to_benchmark


def _extract_downloaded_ticker(data: pd.DataFrame, ticker: str, all_tickers: list[str]) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return pd.DataFrame()
        ticker_data = data[ticker].copy()
    elif len(all_tickers) == 1:
        ticker_data = data.copy()
    else:
        return pd.DataFrame()

    ticker_data = ticker_data.reset_index()
    first_column = ticker_data.columns[0]
    if first_column != "Date":
        ticker_data = ticker_data.rename(columns={first_column: "Date"})
    ticker_data["Ticker"] = ticker
    return ticker_data


def download_hourly_prices(tickers: list[str], benchmarks: list[str]) -> pd.DataFrame:
    all_tickers = sorted(set(tickers + benchmarks + [PARAMS["MARKET"]["VIX_TICKER"]]))
    print(f"Downloading Yahoo Finance hourly data for {len(all_tickers)} tickers...")

    data = yf.download(
        all_tickers,
        period=HOURLY_PERIOD,
        interval="1h",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if data.empty:
        raise ValueError("No hourly Yahoo Finance data downloaded.")

    rows = []
    for ticker in all_tickers:
        ticker_data = _extract_downloaded_ticker(data, ticker, all_tickers)
        if ticker_data.empty:
            print(f"Warning: missing hourly data for {ticker}; skipping.")
            continue
        rows.append(ticker_data)

    prices = pd.concat(rows, ignore_index=True)
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_convert(None)

    required_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in required_columns if column not in prices.columns]
    if missing_columns:
        raise ValueError(f"Hourly data missing required columns: {missing_columns}")

    return prices[required_columns].dropna(subset=["Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.rolling(window).mean()
    average_loss = loss.rolling(window).mean()
    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, ticker_data in data.groupby("Ticker"):
        ticker_data = ticker_data.sort_values("Date").copy()
        close = ticker_data["Close"]

        ticker_data["Daily_Return"] = close.pct_change()
        ticker_data["Lag_1_Return"] = ticker_data["Daily_Return"].shift(1)
        ticker_data["Lag_3_Return"] = ticker_data["Daily_Return"].shift(3)
        ticker_data["Lag_7_Return"] = ticker_data["Daily_Return"].shift(7)
        ticker_data["RollingMean_7"] = close / close.rolling(7).mean() - 1
        ticker_data["RollingMean_30"] = close / close.rolling(30).mean() - 1
        ticker_data["RollingVolatility_7"] = ticker_data["Daily_Return"].rolling(7).std()
        ticker_data["RollingVolatility_30"] = ticker_data["Daily_Return"].rolling(30).std()
        ticker_data["RSI_14"] = calculate_rsi(close, 14)

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        ticker_data["MACD"] = ema_12 - ema_26
        ticker_data["MACD_Signal"] = ticker_data["MACD"].ewm(span=9, adjust=False).mean()
        ticker_data["Volume_Change"] = ticker_data["Volume"].pct_change()
        ticker_data["Volume_Ratio_20"] = ticker_data["Volume"] / ticker_data["Volume"].rolling(20).mean()
        ticker_data["Distance_to_MA_200"] = close / close.rolling(200).mean() - 1
        ticker_data["Momentum_5"] = close / close.shift(5) - 1
        ticker_data["Momentum_10"] = close / close.shift(10) - 1
        ticker_data["Momentum_20"] = close / close.shift(20) - 1
        ticker_data["Price_Position_20"] = (close - ticker_data["Low"].rolling(20).min()) / (
            ticker_data["High"].rolling(20).max() - ticker_data["Low"].rolling(20).min()
        )
        ticker_data["High_Low_Range"] = ticker_data["High"] / ticker_data["Low"] - 1

        frames.append(ticker_data)

    result = pd.concat(frames, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def build_market_features(feature_data: pd.DataFrame) -> pd.DataFrame:
    qqq_ticker = PARAMS["MARKET"]["QQQ_TICKER"]
    spy_ticker = PARAMS["MARKET"]["SPY_TICKER"]
    vix_ticker = PARAMS["MARKET"]["VIX_TICKER"]

    qqq = feature_data[feature_data["Ticker"] == qqq_ticker][["Date", "Open", "Close"]].rename(
        columns={"Open": "QQQ_Open", "Close": "QQQ_Close"}
    )
    spy = feature_data[feature_data["Ticker"] == spy_ticker][["Date", "Open", "Close"]].rename(
        columns={"Open": "SPY_Open", "Close": "SPY_Close"}
    )
    vix = feature_data[feature_data["Ticker"] == vix_ticker][["Date", "Close"]].rename(columns={"Close": "VIX_Close"})

    if qqq.empty or spy.empty:
        raise ValueError("QQQ and SPY hourly data are required for hourly outperformance training.")

    market = qqq.merge(spy, on="Date", how="outer").merge(vix, on="Date", how="left")
    market = market.sort_values("Date").reset_index(drop=True)

    market["QQQ_Return"] = market["QQQ_Close"].pct_change()
    market["SPY_Return"] = market["SPY_Close"].pct_change()
    market["VIX_Change"] = market["VIX_Close"].pct_change().fillna(0.0)
    market["QQQ_Momentum_20"] = market["QQQ_Close"] / market["QQQ_Close"].shift(20) - 1
    market["SPY_Momentum_20"] = market["SPY_Close"] / market["SPY_Close"].shift(20) - 1
    market["QQQ_Distance_to_MA200"] = market["QQQ_Close"] / market["QQQ_Close"].rolling(200).mean() - 1
    market["SPY_Distance_to_MA200"] = market["SPY_Close"] / market["SPY_Close"].rolling(200).mean() - 1
    market["Next_Hour_QQQ_Return"] = market["QQQ_Close"].shift(-1) / market["QQQ_Close"] - 1
    market["Next_Hour_SPY_Return"] = market["SPY_Close"].shift(-1) / market["SPY_Close"] - 1
    market["Next_Hour_QQQ_Tradable_Return"] = market["QQQ_Close"].shift(-1) / market["QQQ_Open"].shift(-1) - 1
    market["Next_Hour_SPY_Tradable_Return"] = market["SPY_Close"].shift(-1) / market["SPY_Open"].shift(-1) - 1

    return market


def add_outperformance_columns(feature_data: pd.DataFrame, ticker_to_benchmark: dict[str, str]) -> pd.DataFrame:
    market = build_market_features(feature_data)
    stock_data = feature_data[feature_data["Ticker"].isin(ticker_to_benchmark)].copy()
    stock_data = stock_data.merge(market, on="Date", how="left")
    stock_data["Benchmark"] = stock_data["Ticker"].map(ticker_to_benchmark)
    stock_data["Future_Return"] = stock_data.groupby("Ticker")["Close"].shift(-1) / stock_data["Close"] - 1
    stock_data["Tradable_Return"] = stock_data.groupby("Ticker")["Close"].shift(-1) / stock_data.groupby("Ticker")["Open"].shift(-1) - 1
    stock_data["Benchmark_Future_Return"] = np.where(
        stock_data["Benchmark"] == "SPY",
        stock_data["Next_Hour_SPY_Return"],
        stock_data["Next_Hour_QQQ_Return"],
    )
    stock_data["Benchmark_Tradable_Return"] = np.where(
        stock_data["Benchmark"] == "SPY",
        stock_data["Next_Hour_SPY_Tradable_Return"],
        stock_data["Next_Hour_QQQ_Tradable_Return"],
    )

    stock_data["Relative_Return_QQQ"] = stock_data["Daily_Return"] - stock_data["QQQ_Return"]
    stock_data["Relative_Return_SPY"] = stock_data["Daily_Return"] - stock_data["SPY_Return"]
    stock_data["Relative_Momentum_20_QQQ"] = stock_data["Momentum_20"] - stock_data["QQQ_Momentum_20"]
    stock_data["Relative_Momentum_20_SPY"] = stock_data["Momentum_20"] - stock_data["SPY_Momentum_20"]
    stock_data[TARGET_COLUMN] = (stock_data["Future_Return"] > stock_data["Benchmark_Future_Return"]).astype(int)

    return stock_data.replace([np.inf, -np.inf], np.nan)


def chronological_split(data: pd.DataFrame) -> SplitData:
    timestamps = pd.Series(sorted(data["Date"].dropna().unique()))
    if len(timestamps) < 10:
        raise ValueError("Not enough hourly timestamps for train/validation/test split.")

    train_end = timestamps.iloc[int(len(timestamps) * 0.70)]
    validation_end = timestamps.iloc[int(len(timestamps) * 0.85)]

    train = data[data["Date"] <= train_end].copy()
    validation = data[(data["Date"] > train_end) & (data["Date"] <= validation_end)].copy()
    test = data[data["Date"] > validation_end].copy()

    train["Split"] = "train"
    validation["Split"] = "validation"
    test["Split"] = "test"
    return SplitData(train=train, validation=validation, test=test)


def scale_features(split_data: SplitData, feature_columns: list[str]) -> tuple[SplitData, StandardScaler]:
    scaler = StandardScaler()
    train = split_data.train.copy()
    validation = split_data.validation.copy()
    test = split_data.test.copy()

    train[feature_columns] = scaler.fit_transform(train[feature_columns])
    validation[feature_columns] = scaler.transform(validation[feature_columns])
    test[feature_columns] = scaler.transform(test[feature_columns])

    return SplitData(train=train, validation=validation, test=test), scaler


def create_sequences(data: pd.DataFrame, feature_columns: list[str], sequence_length: int):
    X_sequences = []
    y_values = []
    metadata_rows = []

    for ticker, ticker_data in data.groupby("Ticker"):
        ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)
        for index in range(sequence_length - 1, len(ticker_data)):
            start_index = index - sequence_length + 1
            end_index = index + 1
            row = ticker_data.iloc[index]
            X_sequences.append(ticker_data.iloc[start_index:end_index][feature_columns].values)
            y_values.append(row[TARGET_COLUMN])
            metadata_rows.append(
                {
                    "Date": row["Date"],
                    "Ticker": ticker,
                    "Benchmark": row["Benchmark"],
                    "Close": row["Close"],
                    "Actual": row[TARGET_COLUMN],
                    "Future_Return": row["Future_Return"],
                    "Benchmark_Future_Return": row["Benchmark_Future_Return"],
                    "Tradable_Return": row["Tradable_Return"],
                    "Benchmark_Tradable_Return": row["Benchmark_Tradable_Return"],
                }
            )

    if not X_sequences:
        raise ValueError("No hourly LSTM sequences were created.")

    return (
        np.array(X_sequences, dtype=np.float32),
        np.array(y_values, dtype=np.float32),
        pd.DataFrame(metadata_rows),
    )


def create_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets = []
    probabilities = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
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
        validation_predictions = (validation_probabilities >= PARAMS["MODEL"]["PREDICTION_THRESHOLD"]).astype(int)
        validation_f1 = f1_score(y_validation_true, validation_predictions, zero_division=0)
        print(
            f"Epoch {epoch:02d}/{PARAMS['MODEL']['EPOCHS']} | "
            f"Loss: {total_loss / len(train_loader):.5f} | Val F1: {validation_f1:.5f}"
        )
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state_dict = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, device, best_validation_f1


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict, np.ndarray]:
    predictions = (probabilities >= PARAMS["MODEL"].get("PREDICTION_THRESHOLD", 0.5)).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
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


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def annualized_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(TRADING_HOURS_PER_YEAR))


def turnover_for_selected(selected: pd.DataFrame) -> float:
    previous: set[str] | None = None
    turnovers = []
    for _, group in selected.groupby("Date"):
        current = set(group["Ticker"])
        if previous is not None:
            changes = len(current.symmetric_difference(previous))
            turnovers.append(changes / max(len(current), 1))
        previous = current
    return float(np.mean(turnovers)) if turnovers else 0.0


def transaction_cost_by_hour(selected: pd.DataFrame, top_k: int, cost_rate: float) -> pd.Series:
    costs = {}
    previous: set[str] = set()
    for timestamp, group in selected.groupby("Date"):
        current = set(group["Ticker"])
        buys = len(current - previous)
        sells = len(previous - current)
        costs[timestamp] = cost_rate * (buys + sells) / max(top_k, 1)
        previous = current
    return pd.Series(costs, name="Transaction_Cost")


def calculate_top_k_results(predictions: pd.DataFrame, cost_rate: float = DEFAULT_TRANSACTION_COST) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows = []
    hourly_returns_by_top_k = {}
    for top_k in TOP_K_VALUES:
        data = predictions.copy()
        data["Probability_Rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
        selected = data[data["Probability_Rank"] <= top_k].copy()

        strategy_return_column = "Tradable_Return" if "Tradable_Return" in selected.columns else "Future_Return"
        benchmark_return_column = "Benchmark_Tradable_Return" if "Benchmark_Tradable_Return" in selected.columns else "Benchmark_Future_Return"

        hourly_returns = selected.groupby("Date").agg(
            Gross_Strategy_Return=(strategy_return_column, "mean"),
            Benchmark_Return=(benchmark_return_column, "mean"),
            Number_Of_Positions=("Ticker", "count"),
        )
        hourly_returns = hourly_returns.fillna(0)
        hourly_returns["Transaction_Cost"] = transaction_cost_by_hour(selected, top_k, cost_rate)
        hourly_returns["Transaction_Cost"] = hourly_returns["Transaction_Cost"].fillna(0)
        hourly_returns["Strategy_Return"] = hourly_returns["Gross_Strategy_Return"] - hourly_returns["Transaction_Cost"]
        hourly_returns_by_top_k[top_k] = hourly_returns

        strategy_return = (1 + hourly_returns["Strategy_Return"]).prod() - 1
        gross_return = (1 + hourly_returns["Gross_Strategy_Return"]).prod() - 1
        benchmark_return = (1 + hourly_returns["Benchmark_Return"]).prod() - 1
        rows.append(
            {
                "model_name": "hourly_outperformance_lstm",
                "top_k": top_k,
                "test_start": hourly_returns.index.min().isoformat(),
                "test_end": hourly_returns.index.max().isoformat(),
                "return": strategy_return,
                "gross_return": gross_return,
                "benchmark_return": benchmark_return,
                "difference": strategy_return - benchmark_return,
                "transaction_cost": cost_rate,
                "sharpe": annualized_sharpe(hourly_returns["Strategy_Return"]),
                "max_drawdown": max_drawdown(hourly_returns["Strategy_Return"]),
                "volatility": float(hourly_returns["Strategy_Return"].std() * np.sqrt(TRADING_HOURS_PER_YEAR)),
                "trades": int(selected.shape[0]),
                "turnover": turnover_for_selected(selected),
                "average_number_of_positions": hourly_returns["Number_Of_Positions"].mean(),
                "number_of_hours": len(hourly_returns),
            }
        )
    return pd.DataFrame(rows), hourly_returns_by_top_k


def plot_equity_curve(hourly_returns_by_top_k: dict[int, pd.DataFrame]) -> None:
    plot_file = HOURLY_ROOT / "plots" / "hourly_equity_curve.png"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for top_k, returns in hourly_returns_by_top_k.items():
        ax.plot(returns.index, (1 + returns["Strategy_Return"]).cumprod(), label=f"Top {top_k}")
    first_returns = next(iter(hourly_returns_by_top_k.values()))
    ax.plot(first_returns.index, (1 + first_returns["Benchmark_Return"]).cumprod(), "--", label="Benchmark")
    ax.set_title("Hourly Outperformance LSTM Equity Curve")
    ax.set_ylabel("Capital factor")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_k_returns(top_k_results: pd.DataFrame) -> None:
    plot_file = HOURLY_ROOT / "plots" / "hourly_top_k_returns.png"
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(top_k_results["top_k"].astype(str), top_k_results["return"], color="#4F8F6F")
    ax.axhline(top_k_results["benchmark_return"].iloc[0], color="#D08B2C", linestyle="--", label="Benchmark")
    ax.set_title("Hourly Top-K Returns")
    ax.set_xlabel("Top K")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in top_k_results["return"]], padding=4)
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(hourly_returns_by_top_k: dict[int, pd.DataFrame]) -> None:
    plot_file = HOURLY_ROOT / "plots" / "hourly_drawdown.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    for top_k, returns in hourly_returns_by_top_k.items():
        equity = (1 + returns["Strategy_Return"]).cumprod()
        drawdown = equity / equity.cummax() - 1
        ax.plot(drawdown.index, drawdown, label=f"Top {top_k}")
    ax.set_title("Hourly Drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_comparison(hourly_returns_by_top_k: dict[int, pd.DataFrame], top_k_results: pd.DataFrame) -> None:
    plot_file = HOURLY_ROOT / "plots" / "hourly_benchmark_comparison.png"
    best_top_k = int(top_k_results.sort_values("return", ascending=False).iloc[0]["top_k"])
    returns = hourly_returns_by_top_k[best_top_k]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(returns.index, (1 + returns["Strategy_Return"]).cumprod(), label=f"Hourly LSTM Top {best_top_k}")
    ax.plot(returns.index, (1 + returns["Benchmark_Return"]).cumprod(), "--", label="Benchmark Buy & Hold")
    ax.set_title("Hourly Benchmark Comparison")
    ax.set_ylabel("Capital factor")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    columns = list(data.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.5f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def json_safe_metrics(metrics: dict) -> dict:
    safe = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, int)):
            safe[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            safe[key] = float(value)
        else:
            safe[key] = value
    return safe


def write_report(
    clean_data: pd.DataFrame,
    split_data: SplitData,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    metrics: dict,
    top_k_results: pd.DataFrame,
    best_validation_f1: float,
) -> None:
    label_distribution = pd.Series(np.concatenate([y_train, y_validation, y_test])).value_counts(normalize=True).sort_index()
    top_k_markdown = dataframe_to_markdown(top_k_results)
    content = f"""# Hourly Outperformance LSTM Report

## Purpose

This hourly model is used only for Alpaca Paper Trading live validation. The daily LSTM pipeline remains the scientific research pipeline.

## Training Period

- First hourly bar: {clean_data['Date'].min()}
- Last hourly bar: {clean_data['Date'].max()}
- Train rows: {len(split_data.train)}
- Validation rows: {len(split_data.validation)}
- Test rows: {len(split_data.test)}

## Samples

- Train sequences: {len(y_train)}
- Validation sequences: {len(y_validation)}
- Test sequences: {len(y_test)}

## Label Distribution

- Class 0: {label_distribution.get(0.0, 0.0):.2%}
- Class 1: {label_distribution.get(1.0, 0.0):.2%}

## Test Metrics

- Best validation F1: {best_validation_f1:.5f}
- Accuracy: {metrics['accuracy']:.5f}
- Precision: {metrics['precision']:.5f}
- Recall: {metrics['recall']:.5f}
- F1: {metrics['f1_score']:.5f}

## Top-K Results

{top_k_markdown}

## Benchmark Comparison

The hourly Top-K strategy is compared against the matching benchmark return for the selected universe rows: QQQ for tech universes and SPY for defensive_non_tech.

## Alpaca Recommendation

Use `hourly_outperformance_lstm_model.pth`, `hourly_outperformance_scaler.pkl` and `hourly_features.txt` for Alpaca Paper Trading. Keep `run_lstm_pipeline.py` and the daily research artifacts unchanged.
"""
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(content, encoding="utf-8")


def save_artifacts(model, scaler, feature_columns, clean_data, split_data, best_validation_f1, metrics) -> None:
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "hourly_outperformance_lstm",
            "training_timeframe": "1Hour",
            "target": TARGET_COLUMN,
            "feature_group": "Technical + Market + Relative Strength",
            "feature_columns": feature_columns,
            "sequence_length": PARAMS["MODEL"]["SEQUENCE_LENGTH"],
            "hidden_size": PARAMS["MODEL"]["HIDDEN_SIZE"],
            "num_layers": PARAMS["MODEL"]["NUM_LAYERS"],
            "dropout": PARAMS["MODEL"]["DROPOUT"],
            "learning_rate": PARAMS["MODEL"]["LEARNING_RATE"],
            "best_validation_f1": best_validation_f1,
        },
        MODEL_FILE,
    )
    with open(SCALER_FILE, "wb") as file:
        pickle.dump(scaler, file)

    metadata = {
        "model_type": "hourly_outperformance_lstm",
        "training_timeframe": "1Hour",
        "target_definition": "1 if next-hour close-to-close stock return is greater than next-hour close-to-close benchmark return, else 0.",
        "backtest_execution_assumption": "Signals are formed after the current bar. Backtests use next-hour open-to-close tradable returns plus transaction costs.",
        "benchmarks": {"tech_universes": "QQQ", "defensive_non_tech": "SPY"},
        "feature_group": "Technical + Market + Relative Strength",
        "feature_list": feature_columns,
        "sequence_length": PARAMS["MODEL"]["SEQUENCE_LENGTH"],
        "training_start": str(clean_data["Date"].min()),
        "training_end": str(clean_data["Date"].max()),
        "train_rows": len(split_data.train),
        "validation_rows": len(split_data.validation),
        "test_rows": len(split_data.test),
        "metrics": json_safe_metrics(metrics),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    print("Hourly Outperformance LSTM")
    print("==========================")
    set_random_seed(PARAMS["MODEL"]["RANDOM_STATE"])

    feature_columns = load_feature_columns()
    ticker_to_benchmark = get_training_ticker_benchmarks()
    prices = download_hourly_prices(sorted(ticker_to_benchmark), sorted(set(ticker_to_benchmark.values())))
    features = add_technical_features(prices)
    data = add_outperformance_columns(features, ticker_to_benchmark)

    required_columns = feature_columns + [
        TARGET_COLUMN,
        "Future_Return",
        "Benchmark_Future_Return",
        "Tradable_Return",
        "Benchmark_Tradable_Return",
    ]
    clean_data = data.dropna(subset=required_columns).copy()
    if clean_data.empty:
        raise ValueError("No usable hourly rows after feature and target calculation.")

    split_data = chronological_split(clean_data)
    scaled_split, scaler = scale_features(split_data, feature_columns)

    sequence_length = PARAMS["MODEL"]["SEQUENCE_LENGTH"]
    X_train, y_train, _ = create_sequences(scaled_split.train, feature_columns, sequence_length)
    X_validation, y_validation, _ = create_sequences(scaled_split.validation, feature_columns, sequence_length)
    X_test, y_test, test_metadata = create_sequences(scaled_split.test, feature_columns, sequence_length)

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

    top_k_results, hourly_returns_by_top_k = calculate_top_k_results(prediction_data)
    result_row = {
        "model_name": "hourly_outperformance_lstm",
        "target": TARGET_COLUMN,
        "sequence_length": sequence_length,
        "hidden_size": PARAMS["MODEL"]["HIDDEN_SIZE"],
        "num_layers": PARAMS["MODEL"]["NUM_LAYERS"],
        "dropout": PARAMS["MODEL"]["DROPOUT"],
        "learning_rate": PARAMS["MODEL"]["LEARNING_RATE"],
        "best_validation_f1": best_validation_f1,
        "test_start": prediction_data["Date"].min().isoformat(),
        "test_end": prediction_data["Date"].max().isoformat(),
        "number_of_test_samples": len(prediction_data),
        **metrics,
    }

    save_artifacts(model, scaler, feature_columns, clean_data, split_data, best_validation_f1, metrics)
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_csv(prediction_data, PREDICTIONS_FILE)
    save_csv(pd.DataFrame([result_row]), RESULTS_FILE)
    save_csv(top_k_results, TOP_K_FILE)

    plot_equity_curve(hourly_returns_by_top_k)
    plot_top_k_returns(top_k_results)
    plot_drawdown(hourly_returns_by_top_k)
    plot_benchmark_comparison(hourly_returns_by_top_k, top_k_results)
    write_report(clean_data, split_data, y_train, y_validation, y_test, metrics, top_k_results, best_validation_f1)

    print("\nHourly test results")
    print("===================")
    print(
        pd.DataFrame([result_row]).to_string(
            index=False,
            formatters={
                "best_validation_f1": format_decimal,
                "accuracy": format_decimal,
                "precision": format_decimal,
                "recall": format_decimal,
                "f1_score": format_decimal,
                "predicted_outperform_share": format_decimal,
            },
        )
    )
    print("\nHourly Top-K results")
    print("====================")
    print(
        top_k_results.to_string(
            index=False,
            formatters={
                "return": format_percent,
                "benchmark_return": format_percent,
                "difference": format_percent,
                "sharpe": format_decimal,
                "max_drawdown": format_percent,
                "volatility": format_percent,
                "turnover": format_decimal,
            },
        )
    )
    print("\nSaved hourly Alpaca artifacts:")
    print(f"- {MODEL_FILE}")
    print(f"- {SCALER_FILE}")
    print(f"- {FEATURE_FILE}")
    print(f"- {REPORT_FILE}")


if __name__ == "__main__":
    main()
