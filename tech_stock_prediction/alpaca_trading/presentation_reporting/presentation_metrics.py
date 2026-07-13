"""
Derived metrics for the presentation reporting system.

Every function here computes a number from an existing, already-loaded
DataFrame (see presentation_data_loader.py). Nothing is hardcoded and
nothing retrains or re-runs any model. Where a metric (Sharpe, max
drawdown, trade count) does not already exist as a column in the source
CSVs, it is derived here from the raw return series in that CSV - this is
"berechnen, nicht hardcoden" in practice.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from reporting_config import TRADING_DAYS_PER_YEAR


def normalize_to_100(values):
    values = pd.Series(values).astype(float)
    if values.empty or values.iloc[0] == 0:
        return values
    return values / values.iloc[0] * 100.0


def annualized_sharpe(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    returns = pd.Series(returns).dropna()
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(periods_per_year))


def max_drawdown_from_returns(returns):
    returns = pd.Series(returns).fillna(0.0)
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    if drawdown.empty:
        return 0.0
    return float(drawdown.min())


def max_drawdown_from_index(index_values):
    index_values = pd.Series(index_values).astype(float)
    if index_values.empty:
        return 0.0
    running_max = index_values.cummax()
    drawdown = index_values / running_max - 1
    return float(drawdown.min())


def total_return_from_returns(returns):
    returns = pd.Series(returns).fillna(0.0)
    if returns.empty:
        return 0.0
    return float((1 + returns).prod() - 1)


def build_cumulative_index(returns):
    """
    Turn a per-period return series into an index starting exactly at 100.
    The first observation is the baseline (index = 100, no return applied
    yet); every following point compounds the realized return for that
    period. Used so strategy and benchmark lines are normalized the same,
    identical way in every plot.
    """
    returns = pd.Series(returns).fillna(0.0).reset_index(drop=True)
    factors = (1 + returns)
    if len(factors) > 0:
        factors.iloc[0] = 1.0
    return 100 * factors.cumprod()


# ---------------------------------------------------------------------------
# Daily Outperformance-LSTM Top-K backtest reconstruction
# ---------------------------------------------------------------------------

def select_best_top_k(source_path, top_k_candidates=range(1, 6)):
    """
    Read the sibling lstm_outperformance_top_k_results.csv next to a
    predictions file (if it exists) and return the top_k value with the
    highest strategy_return, matching what the research pipeline itself
    already reported as best. Falls back to the full candidate range if
    that file is missing, in which case the caller must search itself.
    """
    if source_path is None:
        return None, list(top_k_candidates)

    sibling = Path(source_path).parent / "lstm_outperformance_top_k_results.csv"
    if not sibling.exists():
        return None, list(top_k_candidates)

    try:
        top_k_results = pd.read_csv(sibling)
    except Exception:
        return None, list(top_k_candidates)

    if not {"top_k", "strategy_return"}.issubset(top_k_results.columns):
        return None, list(top_k_candidates)

    best_row = top_k_results.sort_values("strategy_return", ascending=False).iloc[0]
    return int(best_row["top_k"]), list(top_k_candidates)


def reconstruct_daily_topk_backtest(predictions, benchmark_ticker, top_k):
    """
    Rebuild the day-by-day Top-K strategy return series and the matching
    benchmark (QQQ or SPY) return series from the raw per-ticker prediction
    file. This mirrors calculate_top_k_daily_returns() in
    experiments/exp_2_lstm/scripts/14_outperformance_lstm/outperformance_lstm.py
    exactly, so the totals here should match that script's own reported
    numbers - it is a recomputation, not a new methodology.

    Returns a DataFrame indexed by Date with columns:
      Strategy_Return, Benchmark_Return, Number_Of_Positions, Selected_Tickers
    """
    data = predictions.copy()
    data["Probability_Rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    selected = data[data["Probability_Rank"] <= top_k].copy()

    benchmark_return_column = f"Next_Day_{benchmark_ticker}_Return"
    if benchmark_return_column not in data.columns:
        raise ValueError(f"Missing column {benchmark_return_column} in predictions data.")

    daily_strategy_returns = selected.groupby("Date")["Future_Return"].mean()
    daily_benchmark_returns = data.groupby("Date")[benchmark_return_column].first()
    daily_positions = selected.groupby("Date")["Ticker"].count()
    daily_selected_tickers = selected.groupby("Date")["Ticker"].apply(lambda tickers: sorted(tickers))

    daily = pd.concat(
        [
            daily_strategy_returns.rename("Strategy_Return"),
            daily_benchmark_returns.rename("Benchmark_Return"),
            daily_positions.rename("Number_Of_Positions"),
            daily_selected_tickers.rename("Selected_Tickers"),
        ],
        axis=1,
    )
    daily["Strategy_Return"] = daily["Strategy_Return"].fillna(0.0)
    return daily.sort_index()


def count_position_changes(daily_selected_tickers):
    """Number of buy/sell events implied by day-to-day Top-K set changes."""
    previous = None
    trades = 0
    for tickers in daily_selected_tickers:
        current = set(tickers)
        if previous is not None:
            trades += len(current.symmetric_difference(previous))
        else:
            trades += len(current)  # opening positions count as trades too
        previous = current
    return int(trades)


def summarize_backtest(daily, benchmark_ticker):
    """
    Build the compact metric block required next to Plot 1: strategy
    return, benchmark return, difference, Sharpe, max drawdown, number of
    trades, period. No transaction costs are modeled anywhere in the daily
    Outperformance-LSTM backtest, so gross and net return are identical -
    this is stated explicitly rather than silently assumed.
    """
    strategy_returns = daily["Strategy_Return"]
    benchmark_returns = daily["Benchmark_Return"]

    strategy_total = total_return_from_returns(strategy_returns)
    benchmark_total = total_return_from_returns(benchmark_returns)

    return {
        "period_start": daily.index.min(),
        "period_end": daily.index.max(),
        "number_of_days": int(len(daily)),
        "strategy_return_gross": strategy_total,
        "strategy_return_net": strategy_total,  # no transaction costs modeled
        "benchmark_return": benchmark_total,
        "difference": strategy_total - benchmark_total,
        "sharpe_ratio": annualized_sharpe(strategy_returns),
        "max_drawdown": max_drawdown_from_returns(strategy_returns),
        "number_of_trades": count_position_changes(daily["Selected_Tickers"]),
        "transaction_cost_assumption": 0.0,
        "average_number_of_positions": float(daily["Number_Of_Positions"].mean()),
    }


# ---------------------------------------------------------------------------
# Alpaca paper trading (daily and hourly) summaries
# ---------------------------------------------------------------------------

def summarize_alpaca_performance(data, index_col_prefix="portfolio", benchmark_prefix="benchmark", date_column="date"):
    """
    Summarize a per-universe Alpaca mark-to-market log that already has
    <prefix>_index columns normalized to 100 at the first observation.
    Returns are recomputed from the index series (pct_change) rather than
    trusting any pre-existing return column, to avoid unit ambiguities
    between the various log files.
    """
    portfolio_index = data[f"{index_col_prefix}_index"].astype(float)
    benchmark_index = data[f"{benchmark_prefix}_index"].astype(float)

    portfolio_returns = portfolio_index.pct_change().fillna(0.0)
    benchmark_returns = benchmark_index.pct_change().fillna(0.0)

    portfolio_total_return = portfolio_index.iloc[-1] / portfolio_index.iloc[0] - 1
    benchmark_total_return = benchmark_index.iloc[-1] / benchmark_index.iloc[0] - 1

    summary = {
        "observations": int(len(data)),
        "portfolio_return": float(portfolio_total_return),
        "benchmark_return": float(benchmark_total_return),
        "difference": float(portfolio_total_return - benchmark_total_return),
        "sharpe_ratio": annualized_sharpe(portfolio_returns),
        "max_drawdown": max_drawdown_from_index(portfolio_index),
        "period_start": None,
        "period_end": None,
    }

    if date_column in data.columns:
        summary["period_start"] = data[date_column].min()
        summary["period_end"] = data[date_column].max()

    return summary


def summarize_orders(orders_data):
    if orders_data is None or orders_data.empty:
        return {"buys": 0, "sells": 0, "total_trades": 0}

    action = orders_data["action"].str.lower()
    buys = int((action == "buy").sum())
    sells = int((action == "sell").sum())
    return {"buys": buys, "sells": sells, "total_trades": buys + sells}


# ---------------------------------------------------------------------------
# Signal / Top-K selection behaviour
# ---------------------------------------------------------------------------

def summarize_signal_selection(signal_data, universe_tickers):
    """
    For each ticker in the universe, compute the share of (deduplicated)
    signal dates on which it was selected for the Top-K, its average
    predicted probability, its average rank, and how many times it newly
    entered the Top-K compared to the previous signal date.

    Deduplication: the Alpaca scheduler occasionally logs more than one run
    for the same calendar date (retries/skips); we keep only the latest run
    per (date, ticker) using generated_at/timestamp, so each calendar date
    counts once per ticker.
    """
    data = signal_data.copy()
    order_column = "generated_at" if "generated_at" in data.columns else "timestamp"
    if order_column in data.columns:
        data = data.sort_values(order_column)
    data = data.drop_duplicates(subset=["date", "ticker"], keep="last")

    distinct_dates = sorted(data["date"].unique())
    rows = []
    previous_selected = set()

    selection_by_date = {}
    for date in distinct_dates:
        day_data = data[data["date"] == date]
        selection_by_date[date] = set(day_data.loc[day_data["selected"] == True, "ticker"])  # noqa: E712

    for ticker in universe_tickers:
        ticker_rows = data[data["ticker"] == ticker]
        if ticker_rows.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "selection_share": 0.0,
                    "average_probability": np.nan,
                    "average_rank": np.nan,
                    "new_entries": 0,
                    "observations": 0,
                }
            )
            continue

        selection_share = float((ticker_rows["selected"] == True).mean())  # noqa: E712
        average_probability = float(ticker_rows["probability"].mean())
        average_rank = float(ticker_rows["rank"].mean())

        new_entries = 0
        was_selected_previous_date = False
        for date in distinct_dates:
            is_selected_today = ticker in selection_by_date[date]
            if is_selected_today and not was_selected_previous_date:
                new_entries += 1
            was_selected_previous_date = is_selected_today

        rows.append(
            {
                "ticker": ticker,
                "selection_share": selection_share,
                "average_probability": average_probability,
                "average_rank": average_rank,
                "new_entries": new_entries,
                "observations": int(len(ticker_rows)),
            }
        )

    result = pd.DataFrame(rows).sort_values("selection_share", ascending=False).reset_index(drop=True)
    return result, len(distinct_dates)
