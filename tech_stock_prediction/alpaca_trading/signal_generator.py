"""
Generate Outperformance-LSTM signals for Alpaca Paper Trading.

This module performs inference only. It does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import yaml

from config.stock_universes import get_benchmark_for_universe, get_universe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LSTM_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "exp_2_lstm"
LSTM_HOURLY_ROOT = LSTM_EXPERIMENT_ROOT / "hourly"
PARAMS_FILE = LSTM_EXPERIMENT_ROOT / "conf" / "params.yaml"

# Alpaca Paper Trading must use the separately trained hourly model. The daily
# research model remains untouched and is intentionally not loaded here.
EXPECTED_MODEL_FILE = LSTM_HOURLY_ROOT / "models" / "hourly_outperformance_lstm_model.pth"
EXPECTED_SCALER_FILE = LSTM_HOURLY_ROOT / "models" / "hourly_outperformance_scaler.pkl"
EXPECTED_METADATA_FILE = LSTM_HOURLY_ROOT / "models" / "hourly_outperformance_metadata.json"
EXPECTED_FEATURE_FILE = LSTM_HOURLY_ROOT / "conf" / "hourly_features.txt"

FALLBACK_FEATURE_COLUMNS = [
    "Daily_Return",
    "Lag_1_Return",
    "Lag_3_Return",
    "Lag_7_Return",
    "RollingMean_7",
    "RollingMean_30",
    "RollingVolatility_7",
    "RollingVolatility_30",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Volume_Change",
    "Volume_Ratio_20",
    "Distance_to_MA_200",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Price_Position_20",
    "High_Low_Range",
    "QQQ_Return",
    "SPY_Return",
    "VIX_Change",
    "QQQ_Momentum_20",
    "SPY_Momentum_20",
    "QQQ_Distance_to_MA200",
    "SPY_Distance_to_MA200",
    "Relative_Return_QQQ",
    "Relative_Return_SPY",
    "Relative_Momentum_20_QQQ",
    "Relative_Momentum_20_SPY",
]


class SignalGenerationError(RuntimeError):
    """Raised when signals cannot be generated safely."""


class LSTMClassifier(nn.Module):
    """Same simple LSTM architecture used in the research experiment."""

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


@dataclass
class SignalGeneratorConfig:
    universe_name: str
    top_k: int = 5
    timeframe: str = "1Hour"
    lookback_days: int = 450


def load_params() -> dict:
    if not PARAMS_FILE.exists():
        raise SignalGenerationError(f"LSTM params file not found: {PARAMS_FILE}")

    return yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))


def check_inference_files() -> None:
    missing_files = []

    if not EXPECTED_MODEL_FILE.exists():
        missing_files.append(str(EXPECTED_MODEL_FILE.relative_to(PROJECT_ROOT)))
    if not EXPECTED_SCALER_FILE.exists():
        missing_files.append(str(EXPECTED_SCALER_FILE.relative_to(PROJECT_ROOT)))
    if not EXPECTED_FEATURE_FILE.exists():
        missing_files.append(str(EXPECTED_FEATURE_FILE.relative_to(PROJECT_ROOT)))

    if missing_files:
        raise SignalGenerationError(
            "Final Outperformance-LSTM inference files are missing.\n"
            "Expected files:\n- "
            + "\n- ".join(missing_files)
            + "\n\nDo not use the standard LSTM model by accident. "
            "Save the final Outperformance-LSTM model and scaler to the paths above first."
        )


def load_model_metadata() -> dict:
    if not EXPECTED_METADATA_FILE.exists():
        return {
            "model_type": "hourly_outperformance_lstm",
            "training_timeframe": "unknown",
            "target_definition": "Unknown; metadata file is missing.",
        }

    return json.loads(EXPECTED_METADATA_FILE.read_text(encoding="utf-8"))


def warn_if_timeframe_mismatch(timeframe: str) -> None:
    metadata = load_model_metadata()
    training_timeframe = metadata.get("training_timeframe", "unknown")

    if training_timeframe != "unknown" and timeframe != training_timeframe:
        print(
            f"WARNING: Model was trained for {training_timeframe}, but signals "
            f"were requested for {timeframe}."
        )
        print(
            "Alpaca now loads the hourly model artifacts. Use --timeframe 1Hour "
            "for the intended Paper-Trading validation setup."
        )


def load_feature_columns() -> list[str]:
    if not EXPECTED_FEATURE_FILE.exists():
        return FALLBACK_FEATURE_COLUMNS

    return [
        line.strip()
        for line in EXPECTED_FEATURE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def get_download_settings(timeframe: str) -> tuple[str, int]:
    if timeframe == "1Hour":
        return "1h", 120
    if timeframe == "1Day":
        return "1d", 450

    raise SignalGenerationError("Timeframe must be either '1Hour' or '1Day'.")


def download_price_data(tickers: list[str], benchmark: str, timeframe: str) -> pd.DataFrame:
    interval, lookback_days = get_download_settings(timeframe)
    end_timestamp = pd.Timestamp.now(tz="UTC")
    start_timestamp = end_timestamp - pd.Timedelta(days=lookback_days)
    all_tickers = sorted(set(tickers + [benchmark, "QQQ", "SPY", "^VIX"]))

    print(f"Downloading {timeframe} data with yfinance interval '{interval}'...")

    data = yf.download(
        all_tickers,
        start=start_timestamp.date().isoformat(),
        end=(end_timestamp + pd.Timedelta(days=1)).date().isoformat(),
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise SignalGenerationError("No Yahoo Finance data downloaded.")

    rows = []

    for ticker in all_tickers:
        if len(all_tickers) == 1:
            ticker_data = data.copy()
        else:
            if ticker not in data.columns.get_level_values(0):
                raise SignalGenerationError(f"Missing downloaded data for ticker: {ticker}")
            ticker_data = data[ticker].copy()

        ticker_data = ticker_data.reset_index()
        first_column = ticker_data.columns[0]
        if first_column != "Date":
            ticker_data = ticker_data.rename(columns={first_column: "Date"})
        ticker_data["Ticker"] = ticker
        rows.append(ticker_data)

    prices = pd.concat(rows, ignore_index=True)
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_convert(None)

    required_columns = ["Date", "Ticker", "Close", "High", "Low", "Volume"]
    missing_columns = [column for column in required_columns if column not in prices.columns]

    if missing_columns:
        raise SignalGenerationError(f"Downloaded data missing columns: {missing_columns}")

    return prices[required_columns].dropna(subset=["Close"]).copy()


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(window).mean()
    average_loss = loss.rolling(window).mean()
    relative_strength = average_gain / average_loss

    return 100 - (100 / (1 + relative_strength))


def add_stock_features(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for ticker, ticker_data in data.groupby("Ticker"):
        ticker_data = ticker_data.sort_values("Date").copy()
        ticker_data["Daily_Return"] = ticker_data["Close"].pct_change()
        ticker_data["Lag_1_Return"] = ticker_data["Daily_Return"].shift(1)
        ticker_data["Lag_3_Return"] = ticker_data["Daily_Return"].shift(3)
        ticker_data["Lag_7_Return"] = ticker_data["Daily_Return"].shift(7)
        ticker_data["RollingMean_7"] = ticker_data["Close"] / ticker_data["Close"].rolling(7).mean() - 1
        ticker_data["RollingMean_30"] = ticker_data["Close"] / ticker_data["Close"].rolling(30).mean() - 1
        ticker_data["RollingVolatility_7"] = ticker_data["Daily_Return"].rolling(7).std()
        ticker_data["RollingVolatility_30"] = ticker_data["Daily_Return"].rolling(30).std()
        ticker_data["RSI_14"] = calculate_rsi(ticker_data["Close"], 14)

        ema_12 = ticker_data["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = ticker_data["Close"].ewm(span=26, adjust=False).mean()
        ticker_data["MACD"] = ema_12 - ema_26
        ticker_data["MACD_Signal"] = ticker_data["MACD"].ewm(span=9, adjust=False).mean()
        ticker_data["Volume_Change"] = ticker_data["Volume"].pct_change()
        ticker_data["Volume_Ratio_20"] = ticker_data["Volume"] / ticker_data["Volume"].rolling(20).mean()
        ticker_data["Distance_to_MA_200"] = ticker_data["Close"] / ticker_data["Close"].rolling(200).mean() - 1
        ticker_data["Momentum_5"] = ticker_data["Close"] / ticker_data["Close"].shift(5) - 1
        ticker_data["Momentum_10"] = ticker_data["Close"] / ticker_data["Close"].shift(10) - 1
        ticker_data["Momentum_20"] = ticker_data["Close"] / ticker_data["Close"].shift(20) - 1
        ticker_data["Price_Position_20"] = (
            ticker_data["Close"] - ticker_data["Low"].rolling(20).min()
        ) / (ticker_data["High"].rolling(20).max() - ticker_data["Low"].rolling(20).min())
        ticker_data["High_Low_Range"] = ticker_data["High"] / ticker_data["Low"] - 1

        rows.append(ticker_data)

    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def add_relative_features(data: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    qqq = data[data["Ticker"] == "QQQ"][["Date", "Daily_Return", "Momentum_20", "Distance_to_MA_200"]].rename(
        columns={
            "Daily_Return": "QQQ_Return",
            "Momentum_20": "QQQ_Momentum_20",
            "Distance_to_MA_200": "QQQ_Distance_to_MA200",
        }
    )
    spy = data[data["Ticker"] == "SPY"][["Date", "Daily_Return", "Momentum_20", "Distance_to_MA_200"]].rename(
        columns={
            "Daily_Return": "SPY_Return",
            "Momentum_20": "SPY_Momentum_20",
            "Distance_to_MA_200": "SPY_Distance_to_MA200",
        }
    )
    vix = data[data["Ticker"] == "^VIX"][["Date", "Daily_Return"]].rename(columns={"Daily_Return": "VIX_Change"})

    stock_data = data[~data["Ticker"].isin(["QQQ", "SPY", "^VIX"])].copy()
    stock_data = stock_data.merge(qqq, on="Date", how="left").merge(spy, on="Date", how="left").merge(vix, on="Date", how="left")
    stock_data["VIX_Change"] = stock_data["VIX_Change"].fillna(0.0)

    stock_data["Relative_Return_QQQ"] = stock_data["Daily_Return"] - stock_data["QQQ_Return"]
    stock_data["Relative_Return_SPY"] = stock_data["Daily_Return"] - stock_data["SPY_Return"]
    stock_data["Relative_Momentum_20_QQQ"] = stock_data["Momentum_20"] - stock_data["QQQ_Momentum_20"]
    stock_data["Relative_Momentum_20_SPY"] = stock_data["Momentum_20"] - stock_data["SPY_Momentum_20"]

    return stock_data.replace([np.inf, -np.inf], np.nan)


def load_scaler():
    with open(EXPECTED_SCALER_FILE, "rb") as file:
        return pickle.load(file)


def load_model(params: dict, input_size: int) -> LSTMClassifier:
    model_params = params["MODEL"]
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=model_params["HIDDEN_SIZE"],
        num_layers=model_params["NUM_LAYERS"],
        dropout=model_params["DROPOUT"],
    )

    state = torch.load(EXPECTED_MODEL_FILE, map_location="cpu")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model


def build_sequences(
    feature_data: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, list[str], pd.Timestamp, pd.Timestamp]:
    sequences = []
    tickers = []
    window_start_dates = []
    latest_dates = []

    for ticker, ticker_data in feature_data.groupby("Ticker"):
        ticker_data = ticker_data.sort_values("Date").dropna(subset=feature_columns)

        if len(ticker_data) < sequence_length:
            print(f"Skipping {ticker}: not enough feature rows for one sequence.")
            continue

        latest_sequence = ticker_data.tail(sequence_length)[feature_columns].to_numpy()
        sequences.append(latest_sequence)
        tickers.append(ticker)
        window_start_dates.append(ticker_data.tail(sequence_length)["Date"].min())
        latest_dates.append(ticker_data["Date"].max())

    if not sequences:
        raise SignalGenerationError(
            "No valid LSTM sequences created. Check downloaded data and feature calculation."
        )

    return np.array(sequences, dtype=np.float32), tickers, min(window_start_dates), max(latest_dates)


def generate_signals(universe_name: str, top_k: int = 5, timeframe: str = "1Hour") -> pd.DataFrame:
    """
    Generate Top-K outperformance probabilities for one stock universe.
    """
    check_inference_files()
    warn_if_timeframe_mismatch(timeframe)

    params = load_params()
    tickers = get_universe(universe_name)
    benchmark = get_benchmark_for_universe(universe_name)
    sequence_length = int(params["MODEL"]["SEQUENCE_LENGTH"])
    final_feature_columns = load_feature_columns()

    print(f"Signal timeframe: {timeframe}")
    prices = download_price_data(tickers, benchmark, timeframe)
    benchmark_prices = prices[prices["Ticker"] == benchmark].sort_values("Date")
    latest_benchmark_close = float(benchmark_prices["Close"].dropna().iloc[-1])
    features = add_stock_features(prices)
    feature_data = add_relative_features(features, benchmark)

    missing_features = [
        feature for feature in final_feature_columns if feature not in feature_data.columns
    ]
    if missing_features:
        raise SignalGenerationError(f"Missing inference features: {missing_features}")

    valid_feature_data = feature_data.dropna(subset=final_feature_columns).copy()

    scaler = load_scaler()
    scaled_data = valid_feature_data.copy()
    scaled_data[final_feature_columns] = scaler.transform(
        valid_feature_data[final_feature_columns]
    )

    X, sequence_tickers, window_start_timestamp, signal_timestamp = build_sequences(
        scaled_data,
        final_feature_columns,
        sequence_length,
    )

    model = load_model(params, input_size=len(final_feature_columns))

    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.tensor(X))).numpy()

    generated_at = datetime.now(timezone.utc).isoformat()
    run_id = (
        f"{universe_name}_{timeframe}_"
        f"{generated_at.replace(':', '').replace('-', '').replace('.', '')}"
    )
    signal_data = pd.DataFrame(
        {
            "date": signal_timestamp.date().isoformat(),
            "timeframe": timeframe,
            "bar_timestamp": signal_timestamp.isoformat(),
            "test_run_id": run_id,
            "test_universe": universe_name,
            "test_period_start": window_start_timestamp.isoformat(),
            "test_period_end": signal_timestamp.isoformat(),
            "feature_window_end": signal_timestamp.isoformat(),
            "universe": universe_name,
            "benchmark": benchmark,
            "benchmark_close": latest_benchmark_close,
            "ticker": sequence_tickers,
            "probability": probabilities,
        }
    )
    signal_data = signal_data.sort_values("probability", ascending=False).reset_index(drop=True)
    signal_data["rank"] = signal_data.index + 1
    signal_data["selected"] = signal_data["rank"] <= top_k
    signal_data["top_k"] = top_k
    signal_data["generated_at"] = generated_at

    return signal_data[
        [
            "date",
            "generated_at",
            "timeframe",
            "bar_timestamp",
            "test_run_id",
            "test_universe",
            "test_period_start",
            "test_period_end",
            "feature_window_end",
            "universe",
            "benchmark",
            "benchmark_close",
            "ticker",
            "probability",
            "rank",
            "selected",
            "top_k",
        ]
    ]
