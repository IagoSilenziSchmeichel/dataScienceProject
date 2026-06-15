"""
Feature engineering utilities - Version 2.

Adds baseline features plus extra momentum, trend and risk features.
"""

import pandas as pd


def calculate_rsi(close_prices, window=14):
    price_change = close_prices.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(window=window).mean()
    average_loss = losses.rolling(window=window).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


def add_features_for_ticker(ticker_data):
    ticker_data = ticker_data.sort_values("Date").copy()

    ticker_data["Daily_Return"] = ticker_data["Close"].pct_change()

    ticker_data["Lag_1_Return"] = ticker_data["Daily_Return"].shift(1)
    ticker_data["Lag_3_Return"] = ticker_data["Daily_Return"].shift(3)
    ticker_data["Lag_7_Return"] = ticker_data["Daily_Return"].shift(7)

    ticker_data["RollingMean_7"] = ticker_data["Daily_Return"].rolling(7).mean()
    ticker_data["RollingMean_30"] = ticker_data["Daily_Return"].rolling(30).mean()

    ticker_data["RollingVolatility_7"] = ticker_data["Daily_Return"].rolling(7).std()
    ticker_data["RollingVolatility_30"] = ticker_data["Daily_Return"].rolling(30).std()

    ticker_data["RSI_14"] = calculate_rsi(ticker_data["Close"], window=14)

    ema_12 = ticker_data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = ticker_data["Close"].ewm(span=26, adjust=False).mean()

    ticker_data["MACD"] = ema_12 - ema_26
    ticker_data["MACD_Signal"] = ticker_data["MACD"].ewm(span=9, adjust=False).mean()

    ma_20 = ticker_data["Close"].rolling(20).mean()
    ma_50 = ticker_data["Close"].rolling(50).mean()
    ma_200 = ticker_data["Close"].rolling(200).mean()

    ticker_data["Distance_to_MA_20"] = ticker_data["Close"] / ma_20 - 1
    ticker_data["Distance_to_MA_50"] = ticker_data["Close"] / ma_50 - 1
    ticker_data["Distance_to_MA_200"] = ticker_data["Close"] / ma_200 - 1

    ticker_data["Momentum_5"] = ticker_data["Close"] / ticker_data["Close"].shift(5) - 1
    ticker_data["Momentum_10"] = ticker_data["Close"] / ticker_data["Close"].shift(10) - 1
    ticker_data["Momentum_20"] = ticker_data["Close"] / ticker_data["Close"].shift(20) - 1

    ticker_data["Volume_Change"] = ticker_data["Volume"].pct_change()

    average_volume_20 = ticker_data["Volume"].rolling(20).mean()
    ticker_data["Volume_Ratio_20"] = ticker_data["Volume"] / average_volume_20

    rolling_high_20 = ticker_data["High"].rolling(20).max()
    rolling_low_20 = ticker_data["Low"].rolling(20).min()

    ticker_data["Price_Position_20"] = (
            (ticker_data["Close"] - rolling_low_20)
            / (rolling_high_20 - rolling_low_20)
    )

    ticker_data["High_Low_Range"] = (
            (ticker_data["High"] - ticker_data["Low"])
            / ticker_data["Close"]
    )

    return ticker_data