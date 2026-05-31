from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "tech_stocks_raw.csv"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tech_stocks_features.csv"


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

    ticker_data["RollingMean_7"] = ticker_data["Daily_Return"].rolling(window=7).mean()
    ticker_data["RollingMean_30"] = ticker_data["Daily_Return"].rolling(window=30).mean()
    ticker_data["RollingVolatility_7"] = ticker_data["Daily_Return"].rolling(window=7).std()
    ticker_data["RollingVolatility_30"] = ticker_data["Daily_Return"].rolling(window=30).std()

    ticker_data["RSI_14"] = calculate_rsi(ticker_data["Close"], window=14)

    ema_12 = ticker_data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = ticker_data["Close"].ewm(span=26, adjust=False).mean()
    ticker_data["MACD"] = ema_12 - ema_26
    ticker_data["MACD_Signal"] = ticker_data["MACD"].ewm(span=9, adjust=False).mean()

    next_close = ticker_data["Close"].shift(-1)
    ticker_data["Target"] = (next_close > ticker_data["Close"]).astype(int)

    return ticker_data


def main():
    data = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    feature_data = data.groupby("Ticker", group_keys=False).apply(add_features_for_ticker)

    # Die letzte Zeile je Ticker hat kein echtes naechstes Close und wird entfernt.
    feature_data["Next_Close"] = feature_data.groupby("Ticker")["Close"].shift(-1)
    feature_data = feature_data.dropna().drop(columns=["Next_Close"])

    feature_data = feature_data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_data.to_csv(FEATURE_DATA_PATH, index=False)

    print(f"Feature-Daten gespeichert unter: {FEATURE_DATA_PATH}")
    print(f"Anzahl Zeilen nach Feature Engineering: {len(feature_data)}")
    print("\nVerwendete Features:")
    print(
        [
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
        ]
    )


if __name__ == "__main__":
    main()
