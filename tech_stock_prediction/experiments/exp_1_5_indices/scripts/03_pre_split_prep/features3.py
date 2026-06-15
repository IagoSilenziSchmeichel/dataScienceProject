import pandas as pd
import yfinance as yf


FUNDAMENTAL_FEATURES = [
    "revenue_growth",
    "eps",
    "profit_margin",
    "operating_margin",
    "free_cash_flow",
    "debt_to_equity",
    "roe",
]

EARNINGS_FEATURES = [
    "actual_eps",
    "expected_eps",
    "eps_surprise",
    "revenue_surprise",
]

NEW_FEATURES = FUNDAMENTAL_FEATURES + EARNINGS_FEATURES


def calculate_rsi(close_prices, window=14):
    price_change = close_prices.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(window=window).mean()
    average_loss = losses.rolling(window=window).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


def get_statement_row(statement, possible_names):
    """Return the first matching row from a yfinance statement."""
    if statement is None or statement.empty:
        return None

    for name in possible_names:
        if name in statement.index:
            return pd.to_numeric(statement.loc[name], errors="coerce")

    return None


def add_series(table, column_name, series):
    if series is not None:
        table[column_name] = series


def safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, pd.NA)


def build_quarterly_fundamentals(stock):
    """Create point-in-time quarterly fundamental features where possible."""
    income = stock.quarterly_income_stmt
    balance = stock.quarterly_balance_sheet
    cashflow = stock.quarterly_cashflow

    dates = set()
    for statement in [income, balance, cashflow]:
        if statement is not None and not statement.empty:
            dates.update(statement.columns)

    if not dates:
        return pd.DataFrame()

    quarterly = pd.DataFrame(index=pd.to_datetime(sorted(dates)))

    revenue = get_statement_row(income, ["Total Revenue", "Operating Revenue"])
    net_income = get_statement_row(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing Operation Net Minority Interest",
        ],
    )
    operating_income = get_statement_row(
        income,
        ["Operating Income", "Total Operating Income As Reported", "EBIT"],
    )
    eps = get_statement_row(income, ["Diluted EPS", "Basic EPS"])
    free_cash_flow = get_statement_row(cashflow, ["Free Cash Flow"])
    total_debt = get_statement_row(balance, ["Total Debt"])
    equity = get_statement_row(
        balance,
        ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    )

    add_series(quarterly, "revenue", revenue)
    add_series(quarterly, "net_income", net_income)
    add_series(quarterly, "operating_income", operating_income)
    add_series(quarterly, "eps", eps)
    add_series(quarterly, "free_cash_flow", free_cash_flow)
    add_series(quarterly, "total_debt", total_debt)
    add_series(quarterly, "equity", equity)

    quarterly = quarterly.sort_index()
    quarterly["revenue_growth"] = quarterly["revenue"].pct_change() if "revenue" in quarterly else pd.NA
    quarterly["profit_margin"] = safe_divide(quarterly["net_income"], quarterly["revenue"]) if {"net_income", "revenue"}.issubset(quarterly.columns) else pd.NA
    quarterly["operating_margin"] = safe_divide(quarterly["operating_income"], quarterly["revenue"]) if {"operating_income", "revenue"}.issubset(quarterly.columns) else pd.NA
    quarterly["debt_to_equity"] = safe_divide(quarterly["total_debt"], quarterly["equity"]) if {"total_debt", "equity"}.issubset(quarterly.columns) else pd.NA
    quarterly["roe"] = safe_divide(quarterly["net_income"], quarterly["equity"]) if {"net_income", "equity"}.issubset(quarterly.columns) else pd.NA

    quarterly = quarterly[FUNDAMENTAL_FEATURES].copy()

    # Quarterly reports are not known on the quarter end date.
    # A 45-day delay is a simple beginner-friendly approximation.
    quarterly["Date"] = quarterly.index + pd.Timedelta(days=45)

    return quarterly.reset_index(drop=True).sort_values("Date")


def build_earnings_features(stock):
    """Load EPS estimate/actual data if yfinance can provide it."""
    try:
        earnings = stock.get_earnings_dates(limit=60)
    except Exception:
        return pd.DataFrame()

    if earnings is None or earnings.empty:
        return pd.DataFrame()

    earnings = earnings.reset_index()
    date_column = earnings.columns[0]

    result = pd.DataFrame()
    result["Date"] = pd.to_datetime(earnings[date_column], errors="coerce").dt.tz_localize(None)
    result["actual_eps"] = pd.to_numeric(earnings.get("Reported EPS", 0.0), errors="coerce")
    result["expected_eps"] = pd.to_numeric(earnings.get("EPS Estimate", 0.0), errors="coerce")

    if "Surprise(%)" in earnings:
        result["eps_surprise"] = pd.to_numeric(earnings["Surprise(%)"], errors="coerce")
    else:
        result["eps_surprise"] = result["actual_eps"] - result["expected_eps"]

    # yfinance earnings dates usually contain EPS surprises, not revenue surprises.
    result["revenue_surprise"] = 0.0

    return result.dropna(subset=["Date"]).sort_values("Date")


def merge_time_based_features(ticker_data, feature_data, columns):
    if feature_data.empty:
        for column in columns:
            ticker_data[column] = pd.NA
        return ticker_data

    merged = pd.merge_asof(
        ticker_data.sort_values("Date"),
        feature_data[["Date"] + columns].sort_values("Date"),
        on="Date",
        direction="backward",
    )

    return merged


def add_fundamental_and_earnings_features(ticker_data):
    ticker = ticker_data["Ticker"].iloc[0]
    stock = yf.Ticker(ticker)

    try:
        fundamentals = build_quarterly_fundamentals(stock)
    except Exception:
        fundamentals = pd.DataFrame()

    ticker_data = merge_time_based_features(ticker_data, fundamentals, FUNDAMENTAL_FEATURES)

    for column in FUNDAMENTAL_FEATURES:
        ticker_data[column] = pd.to_numeric(ticker_data[column], errors="coerce")
        # Missing historical statements stay neutral instead of using today's values.
        ticker_data[column] = ticker_data[column].fillna(0.0)

    earnings = build_earnings_features(stock)
    ticker_data = merge_time_based_features(ticker_data, earnings, EARNINGS_FEATURES)

    for column in EARNINGS_FEATURES:
        ticker_data[column] = pd.to_numeric(ticker_data[column], errors="coerce").fillna(0.0)

    return ticker_data


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

    ticker_data = add_fundamental_and_earnings_features(ticker_data)

    return ticker_data
