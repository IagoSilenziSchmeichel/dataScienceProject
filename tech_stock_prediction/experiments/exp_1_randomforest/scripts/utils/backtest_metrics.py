"""
Shared backtest metric utilities.
"""

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_backtest_metrics(daily_returns, return_column):
    """
    Calculate total and annualized metrics for a daily return column.
    """
    returns = pd.to_numeric(daily_returns[return_column], errors="coerce").dropna()

    if len(returns) == 0:
        return {
            "Total_Return": 0.0,
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "Volatility": 0.0,
            "Max_Drawdown": 0.0,
        }

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    number_of_days = len(returns)
    years = number_of_days / TRADING_DAYS_PER_YEAR
    if years > 0:
        cagr = cumulative.iloc[-1] ** (1 / years) - 1
    else:
        cagr = 0.0

    return_std = returns.std()
    if pd.notna(return_std) and return_std > 0:
        sharpe = returns.mean() / return_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        volatility = return_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0
        volatility = 0.0

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()
    if pd.isna(max_drawdown):
        max_drawdown = 0.0

    return {
        "Total_Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Volatility": volatility,
        "Max_Drawdown": max_drawdown,
    }
