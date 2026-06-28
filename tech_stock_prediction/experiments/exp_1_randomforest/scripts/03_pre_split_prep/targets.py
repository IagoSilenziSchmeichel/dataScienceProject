"""
Target calculation.

Target = 1 if the stock price is higher by at least a minimum percentage
after a defined number of trading days.
"""


def add_next_day_target(ticker_data):
    ticker_data = ticker_data.sort_values("Date").copy()

    # How many trading days into the future we look
    target_days = 1

    # Minimum required increase
    # 0.02 means +2%
    min_return = 0.00

    # Closing price in 20 trading days
    ticker_data["Next_Close"] = ticker_data["Close"].shift(-target_days)

    # Calculate future return
    # Example: today 100, future 103 -> return = 0.03 = +3%
    ticker_data["Future_Return"] = (
            ticker_data["Next_Close"] / ticker_data["Close"] - 1
    )

    # Target:
    # 1 = stock increased by more than 2% after 20 trading days
    # 0 = stock did not increase by more than 2%
    ticker_data["Target"] = (
            ticker_data["Future_Return"] > min_return
    ).astype(int)

    return ticker_data