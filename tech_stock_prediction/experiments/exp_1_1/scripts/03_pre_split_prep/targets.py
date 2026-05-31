"""
Target calculation.

Target = 1 if tomorrow's close is higher than today's close, otherwise 0.
"""


def add_next_day_target(ticker_data):
    ticker_data = ticker_data.sort_values("Date").copy()

    ticker_data["Next_Close"] = ticker_data["Close"].shift(-1)
    ticker_data["Target"] = (ticker_data["Next_Close"] > ticker_data["Close"]).astype(int)

    return ticker_data
