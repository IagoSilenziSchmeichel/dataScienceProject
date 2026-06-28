"""
Feature engineering utilities.

The functions in this file are intentionally simple. They create technical,
volume-based and time-based features for one stock at a time.
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

    # 1. Daily return
    ticker_data["Daily_Return"] = ticker_data["Close"].pct_change()

    # 2. Lag features
    ticker_data["Lag_1_Return"] = ticker_data["Daily_Return"].shift(1)
    ticker_data["Lag_3_Return"] = ticker_data["Daily_Return"].shift(3)
    ticker_data["Lag_7_Return"] = ticker_data["Daily_Return"].shift(7)

    # 3. Rolling return statistics
    ticker_data["RollingMean_7"] = ticker_data["Daily_Return"].rolling(window=7).mean()
    ticker_data["RollingMean_30"] = ticker_data["Daily_Return"].rolling(window=30).mean()

    ticker_data["RollingVolatility_7"] = ticker_data["Daily_Return"].rolling(window=7).std()
    ticker_data["RollingVolatility_30"] = ticker_data["Daily_Return"].rolling(window=30).std()

    # 4. RSI
    ticker_data["RSI_14"] = calculate_rsi(ticker_data["Close"], window=14)

    # 5. MACD
    ema_12 = ticker_data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = ticker_data["Close"].ewm(span=26, adjust=False).mean()

    ticker_data["MACD"] = ema_12 - ema_26
    ticker_data["MACD_Signal"] = ticker_data["MACD"].ewm(span=9, adjust=False).mean()

    # 6. Distance to moving averages
    moving_average_20 = ticker_data["Close"].rolling(window=20).mean()
    moving_average_50 = ticker_data["Close"].rolling(window=50).mean()

    ticker_data["Distance_to_MA_20"] = ticker_data["Close"] / moving_average_20 - 1
    ticker_data["Distance_to_MA_50"] = ticker_data["Close"] / moving_average_50 - 1

    # 7. Volume features
    ticker_data["Volume_Change"] = ticker_data["Volume"].pct_change()

    average_volume_20 = ticker_data["Volume"].rolling(window=20).mean()
    ticker_data["Volume_Ratio_20"] = ticker_data["Volume"] / average_volume_20

    return ticker_data