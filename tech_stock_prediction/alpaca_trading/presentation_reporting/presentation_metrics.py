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

from reporting_config import TRADING_DAYS_PER_YEAR, TRADING_HOURS_PER_YEAR


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


def max_drawdown_usd_from_capital(capital_values):
    capital_values = pd.Series(capital_values).astype(float).dropna()
    if capital_values.empty:
        return np.nan
    running_max = capital_values.cummax()
    drawdown_usd = capital_values - running_max
    return float(drawdown_usd.min())


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


def load_top_k_table(source_path):
    """
    Read the full lstm_outperformance_top_k_results.csv sibling to a backtest
    predictions file (Top-1..Top-5, real numbers straight from the pipeline's
    own Top-K backtest - nothing recomputed or estimated here). Returns None
    if the file does not exist or does not have the expected columns.
    """
    if source_path is None:
        return None

    sibling = Path(source_path).parent / "lstm_outperformance_top_k_results.csv"
    if not sibling.exists():
        return None

    try:
        table = pd.read_csv(sibling)
    except Exception:
        return None

    required = {"top_k", "strategy_return", "buy_and_hold_return", "difference", "average_number_of_positions"}
    if not required.issubset(table.columns):
        return None

    return table.sort_values("top_k").reset_index(drop=True)


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
# Hourly Outperformance-LSTM Top-K reporting
# ---------------------------------------------------------------------------

def reconstruct_hourly_topk_backtest(predictions, top_k):
    """
    Rebuild a stundenbasierte Top-K-Auswertung from the existing hourly
    prediction file. This does not train a model and does not submit orders.
    """
    data = predictions.copy()
    data["Probability_Rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    selected = data[data["Probability_Rank"] <= top_k].copy()

    hourly_strategy_returns = selected.groupby("Date")["Tradable_Return"].mean()
    hourly_benchmark_returns = data.groupby("Date")["Benchmark_Tradable_Return"].first()
    hourly_positions = selected.groupby("Date")["Ticker"].count()
    hourly_selected_tickers = selected.groupby("Date")["Ticker"].apply(lambda tickers: sorted(tickers))
    hourly_probability = selected.groupby("Date")["Probability"].mean()

    hourly = pd.concat(
        [
            hourly_strategy_returns.rename("Strategy_Return"),
            hourly_benchmark_returns.rename("Benchmark_Return"),
            hourly_positions.rename("Number_Of_Positions"),
            hourly_selected_tickers.rename("Selected_Tickers"),
            hourly_probability.rename("Average_Probability"),
        ],
        axis=1,
    )
    hourly["Strategy_Return"] = hourly["Strategy_Return"].fillna(0.0)
    hourly["Number_Of_Positions"] = hourly["Number_Of_Positions"].fillna(0).astype(int)
    return hourly.sort_index()


def summarize_hourly_backtest(hourly, top_k):
    strategy_returns = hourly["Strategy_Return"]
    benchmark_returns = hourly["Benchmark_Return"]
    strategy_total = total_return_from_returns(strategy_returns)
    benchmark_total = total_return_from_returns(benchmark_returns)
    trade_stats = summarize_selection_changes(hourly["Selected_Tickers"])

    return {
        "period_start": hourly.index.min(),
        "period_end": hourly.index.max(),
        "observations": int(len(hourly)),
        "top_k": int(top_k),
        "strategy_return": strategy_total,
        "benchmark_return": benchmark_total,
        "difference": strategy_total - benchmark_total,
        "sharpe_ratio": annualized_sharpe(strategy_returns, periods_per_year=TRADING_HOURS_PER_YEAR),
        "max_drawdown": max_drawdown_from_returns(strategy_returns),
        "volatility": float(pd.Series(strategy_returns).std(ddof=0) * np.sqrt(TRADING_HOURS_PER_YEAR)),
        "number_of_trades": trade_stats["trades"],
        "buys": trade_stats["buys"],
        "sells": trade_stats["sells"],
        "rebalancings": trade_stats["rebalancings"],
        "turnover": trade_stats["turnover"],
        "average_number_of_positions": float(hourly["Number_Of_Positions"].mean()),
        "average_probability": float(hourly["Average_Probability"].mean()),
    }


def select_best_hourly_top_k(predictions, candidates=range(1, 6)):
    rows = []
    for top_k in candidates:
        hourly = reconstruct_hourly_topk_backtest(predictions, top_k)
        summary = summarize_hourly_backtest(hourly, top_k)
        rows.append(
            {
                "top_k": int(top_k),
                "strategy_return": summary["strategy_return"],
                "benchmark_return": summary["benchmark_return"],
                "difference": summary["difference"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "max_drawdown": summary["max_drawdown"],
            }
        )
    table = pd.DataFrame(rows)
    best_top_k = int(table.sort_values("strategy_return", ascending=False).iloc[0]["top_k"])
    return best_top_k, table


def summarize_hourly_signal_stability(predictions, universe_tickers, top_k):
    """
    Ticker-level stability table for the hourly model output: how often a
    ticker was selected, its average rank/probability, and implied
    entries/exits from changes in the Top-K set.
    """
    data = predictions.copy()
    data["rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    data["selected"] = data["rank"] <= top_k
    dates = sorted(data["Date"].dropna().unique())

    selected_by_date = {}
    for date in dates:
        day_data = data[data["Date"] == date]
        selected_by_date[date] = set(day_data.loc[day_data["selected"], "Ticker"])

    rows = []
    for ticker in universe_tickers:
        ticker_rows = data[data["Ticker"] == ticker]
        selected_sequence = [ticker in selected_by_date[date] for date in dates]

        entries = 0
        exits = 0
        previous = False
        for current in selected_sequence:
            if current and not previous:
                entries += 1
            if previous and not current:
                exits += 1
            previous = current

        if ticker_rows.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "selection_share": 0.0,
                    "average_probability": np.nan,
                    "average_rank": np.nan,
                    "buys": 0,
                    "sells": 0,
                    "average_holding_duration": np.nan,
                    "observations": 0,
                }
            )
            continue

        rows.append(
            {
                "ticker": ticker,
                "selection_share": float(ticker_rows["selected"].mean()),
                "average_probability": float(ticker_rows["Probability"].mean()),
                "average_rank": float(ticker_rows["rank"].mean()),
                "buys": int(entries),
                "sells": int(exits),
                "average_holding_duration": _average_holding_duration(selected_sequence),
                "observations": int(len(ticker_rows)),
            }
        )

    result = pd.DataFrame(rows).sort_values("selection_share", ascending=False).reset_index(drop=True)

    previous_selection = None
    turnovers = []
    for date in dates:
        current_selection = selected_by_date[date]
        if previous_selection is not None:
            turnovers.append(len(current_selection.symmetric_difference(previous_selection)))
        previous_selection = current_selection

    aggregate = {
        "number_of_signal_dates": int(len(dates)),
        "most_frequent_top1": None,
        "most_frequent_top3": list(result["ticker"].head(3)),
        "average_probability_overall": float(data["Probability"].mean()) if not data.empty else np.nan,
        "average_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
    }
    rank_1 = data[data["rank"] == 1]
    if not rank_1.empty:
        aggregate["most_frequent_top1"] = rank_1["Ticker"].value_counts().idxmax()

    return result, aggregate


def summarize_selection_changes(selected_tickers_by_period):
    previous = set()
    buys = 0
    sells = 0
    rebalancings = 0
    periods = 0

    for tickers in selected_tickers_by_period:
        current = set(tickers)
        new_buys = len(current - previous)
        new_sells = len(previous - current)
        if periods > 0 and (new_buys or new_sells):
            rebalancings += 1
        buys += new_buys
        sells += new_sells
        previous = current
        periods += 1

    trades = buys + sells
    return {
        "buys": int(buys),
        "sells": int(sells),
        "trades": int(trades),
        "rebalancings": int(rebalancings),
        "turnover": float(trades / periods) if periods else np.nan,
    }


def summarize_hourly_selected_positions(predictions, top_k):
    data = predictions.copy()
    data["rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    selected = data[data["rank"] <= top_k].copy()
    if selected.empty:
        return {"best_position": None, "worst_position": None}

    rows = []
    for ticker, ticker_rows in selected.groupby("Ticker"):
        total_return = total_return_from_returns(ticker_rows["Tradable_Return"])
        rows.append({"ticker": ticker, "return": total_return})

    table = pd.DataFrame(rows)
    best = table.sort_values("return", ascending=False).iloc[0]
    worst = table.sort_values("return", ascending=True).iloc[0]
    return {
        "best_position": f"{best['ticker']} ({best['return']:+.1%})",
        "worst_position": f"{worst['ticker']} ({worst['return']:+.1%})",
    }


# ---------------------------------------------------------------------------
# Alpaca paper trading (daily and hourly) summaries
# ---------------------------------------------------------------------------

def _capital_from_index_or_value(data, value_column, index_column, start_capital=None):
    if value_column in data.columns:
        values = pd.to_numeric(data[value_column], errors="coerce")
        if values.dropna().shape[0] >= 1:
            return values

    index_values = pd.to_numeric(data[index_column], errors="coerce")
    if start_capital is None:
        start_capital = 100000.0
    return start_capital * index_values / index_values.iloc[0]


def summarize_alpaca_performance(data, index_col_prefix="portfolio", benchmark_prefix="benchmark", date_column="date"):
    """
    Summarize a per-universe Alpaca mark-to-market log that already has
    <prefix>_index columns normalized to 100 at the first observation.
    Returns are recomputed from the index series (pct_change) rather than
    trusting any pre-existing return column, to avoid unit ambiguities
    between the various log files.
    """
    portfolio_index_column = f"{index_col_prefix}_index"
    benchmark_index_column = f"{benchmark_prefix}_index"
    portfolio_index = data[portfolio_index_column].astype(float)
    benchmark_index = data[benchmark_index_column].astype(float)

    portfolio_returns = portfolio_index.pct_change().fillna(0.0)
    benchmark_returns = benchmark_index.pct_change().fillna(0.0)

    portfolio_total_return = portfolio_index.iloc[-1] / portfolio_index.iloc[0] - 1
    benchmark_total_return = benchmark_index.iloc[-1] / benchmark_index.iloc[0] - 1
    portfolio_capital = _capital_from_index_or_value(
        data,
        f"{index_col_prefix}_value",
        portfolio_index_column,
    )
    start_capital = float(portfolio_capital.iloc[0])
    end_capital = float(portfolio_capital.iloc[-1])
    benchmark_capital = _capital_from_index_or_value(
        data,
        "__no_benchmark_capital_column__",
        benchmark_index_column,
        start_capital=start_capital,
    )
    period_pnl = portfolio_capital.diff().fillna(0.0)
    period_returns = portfolio_capital.pct_change().fillna(0.0)
    benchmark_period_returns = benchmark_capital.pct_change().fillna(0.0)
    non_initial_returns = period_returns.iloc[1:] if len(period_returns) > 1 else period_returns.iloc[0:0]
    non_initial_pnl = period_pnl.iloc[1:] if len(period_pnl) > 1 else period_pnl.iloc[0:0]

    summary = {
        "observations": int(len(data)),
        "start_capital": start_capital,
        "end_capital": end_capital,
        "profit_loss_usd": end_capital - start_capital,
        "benchmark_end_capital": float(benchmark_capital.iloc[-1]),
        "benchmark_profit_loss_usd": float(benchmark_capital.iloc[-1] - benchmark_capital.iloc[0]),
        "portfolio_return": float(portfolio_total_return),
        "benchmark_return": float(benchmark_total_return),
        "difference": float(portfolio_total_return - benchmark_total_return),
        "sharpe_ratio": annualized_sharpe(portfolio_returns),
        "max_drawdown": max_drawdown_from_index(portfolio_index),
        "max_drawdown_usd": max_drawdown_usd_from_capital(portfolio_capital),
        "best_period_return": float(non_initial_returns.max()) if not non_initial_returns.empty else np.nan,
        "best_period_usd": float(non_initial_pnl.max()) if not non_initial_pnl.empty else np.nan,
        "worst_period_return": float(non_initial_returns.min()) if not non_initial_returns.empty else np.nan,
        "worst_period_usd": float(non_initial_pnl.min()) if not non_initial_pnl.empty else np.nan,
        "period_start": None,
        "period_end": None,
        "period_returns": period_returns,
        "benchmark_period_returns": benchmark_period_returns,
        "capital_series": portfolio_capital,
    }

    if date_column in data.columns:
        summary["period_start"] = data[date_column].min()
        summary["period_end"] = data[date_column].max()

    return summary


def summarize_paper_performance_values(data, date_column, periods_per_year=TRADING_DAYS_PER_YEAR):
    """
    Summarize a Paper-Trading performance log with real portfolio_value and
    benchmark_value columns. benchmark_value is usually an index/ETF price,
    so it is normalized to the same start capital as the portfolio.
    """
    if data is None or data.empty:
        return None

    working = data.copy().sort_values(date_column).reset_index(drop=True)
    working["portfolio_value"] = pd.to_numeric(working["portfolio_value"], errors="coerce")
    working["benchmark_value"] = pd.to_numeric(working["benchmark_value"], errors="coerce")
    working = working.dropna(subset=["portfolio_value", "benchmark_value"])
    if working.empty:
        return None

    portfolio_capital = working["portfolio_value"].reset_index(drop=True)
    benchmark_price = working["benchmark_value"].reset_index(drop=True)
    benchmark_capital = portfolio_capital.iloc[0] * benchmark_price / benchmark_price.iloc[0]

    portfolio_returns = portfolio_capital.pct_change().fillna(0.0)
    benchmark_returns = benchmark_capital.pct_change().fillna(0.0)
    pnl = portfolio_capital.diff().fillna(0.0)
    non_initial_returns = portfolio_returns.iloc[1:] if len(portfolio_returns) > 1 else portfolio_returns.iloc[0:0]
    non_initial_pnl = pnl.iloc[1:] if len(pnl) > 1 else pnl.iloc[0:0]

    return {
        "observations": int(len(working)),
        "start_capital": float(portfolio_capital.iloc[0]),
        "end_capital": float(portfolio_capital.iloc[-1]),
        "profit_loss_usd": float(portfolio_capital.iloc[-1] - portfolio_capital.iloc[0]),
        "benchmark_end_capital": float(benchmark_capital.iloc[-1]),
        "benchmark_profit_loss_usd": float(benchmark_capital.iloc[-1] - benchmark_capital.iloc[0]),
        "portfolio_return": float(portfolio_capital.iloc[-1] / portfolio_capital.iloc[0] - 1),
        "benchmark_return": float(benchmark_capital.iloc[-1] / benchmark_capital.iloc[0] - 1),
        "difference": float((portfolio_capital.iloc[-1] / portfolio_capital.iloc[0] - 1) - (benchmark_capital.iloc[-1] / benchmark_capital.iloc[0] - 1)),
        "sharpe_ratio": annualized_sharpe(portfolio_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown_from_index(portfolio_capital),
        "max_drawdown_usd": max_drawdown_usd_from_capital(portfolio_capital),
        "best_period_return": float(non_initial_returns.max()) if not non_initial_returns.empty else np.nan,
        "best_period_usd": float(non_initial_pnl.max()) if not non_initial_pnl.empty else np.nan,
        "worst_period_return": float(non_initial_returns.min()) if not non_initial_returns.empty else np.nan,
        "worst_period_usd": float(non_initial_pnl.min()) if not non_initial_pnl.empty else np.nan,
        "period_start": working[date_column].min(),
        "period_end": working[date_column].max(),
        "period_returns": portfolio_returns,
        "benchmark_period_returns": benchmark_returns,
        "capital_series": portfolio_capital,
        "working_data": working,
    }


def summarize_orders(orders_data):
    if orders_data is None or orders_data.empty:
        return {"buys": 0, "sells": 0, "total_trades": 0, "total_notional_traded": None}

    action = orders_data["action"].str.lower()
    buys = int((action == "buy").sum())
    sells = int((action == "sell").sum())
    total_notional_traded = float(orders_data["notional"].sum()) if "notional" in orders_data.columns else None
    return {
        "buys": buys,
        "sells": sells,
        "total_trades": buys + sells,
        "total_notional_traded": total_notional_traded,
    }


def infer_top_k_from_selected_column(data):
    if data is None or data.empty or "selected_top3" not in data.columns:
        return None

    counts = []
    for value in data["selected_top3"].dropna():
        tickers = [ticker.strip() for ticker in str(value).split(",") if ticker.strip()]
        if tickers:
            counts.append(len(tickers))
    if not counts:
        return None
    return int(round(float(np.mean(counts))))


def summarize_daily_positions(positions_data):
    if positions_data is None or positions_data.empty:
        return {"best_position": None, "worst_position": None}

    data = positions_data.copy()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["date", "symbol", "price"])
    if data.empty:
        return {"best_position": None, "worst_position": None}

    rows = []
    for symbol, symbol_rows in data.groupby("symbol"):
        symbol_rows = symbol_rows.sort_values("date")
        if len(symbol_rows) < 2 or symbol_rows["price"].iloc[0] == 0:
            continue
        symbol_return = symbol_rows["price"].iloc[-1] / symbol_rows["price"].iloc[0] - 1
        rows.append({"ticker": symbol, "return": float(symbol_return)})

    if not rows:
        return {"best_position": None, "worst_position": None}

    table = pd.DataFrame(rows)
    best = table.sort_values("return", ascending=False).iloc[0]
    worst = table.sort_values("return", ascending=True).iloc[0]
    return {
        "best_position": f"{best['ticker']} ({best['return']:+.1%})",
        "worst_position": f"{worst['ticker']} ({worst['return']:+.1%})",
    }


def summarize_daily_paper_trading(performance_data, orders_data=None, positions_data=None):
    if performance_data is None or performance_data.empty:
        return None

    summary = summarize_alpaca_performance(performance_data)
    order_summary = summarize_orders(orders_data)
    position_summary = summarize_daily_positions(positions_data)
    top_k = infer_top_k_from_selected_column(performance_data)
    rebalancings = None
    if "rebalanced" in performance_data.columns:
        rebalancings = int(performance_data["rebalanced"].fillna(False).astype(bool).sum())

    detail = build_period_detail_table(
        performance_data,
        date_column="date",
        capital_series=summary["capital_series"],
        period_returns=summary["period_returns"],
        orders_data=orders_data,
        order_time_column="date",
    )

    summary.update(
        {
            "top_k": top_k,
            "number_of_trades": order_summary["total_trades"],
            "buys": order_summary["buys"],
            "sells": order_summary["sells"],
            "total_notional_traded": order_summary["total_notional_traded"],
            "average_notional_per_period": (
                order_summary["total_notional_traded"] / summary["observations"]
                if order_summary["total_notional_traded"] is not None and summary["observations"]
                else None
            ),
            "rebalancings": rebalancings,
            "best_position": position_summary["best_position"],
            "worst_position": position_summary["worst_position"],
            "detail_table": detail,
        }
    )
    return summary


def build_period_detail_table(performance_data, date_column, capital_series, period_returns, orders_data=None, order_time_column=None):
    data = performance_data.copy().reset_index(drop=True)
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce").dt.tz_localize(None)
    capital = pd.Series(capital_series).reset_index(drop=True)
    returns = pd.Series(period_returns).reset_index(drop=True)
    pnl = capital.diff().fillna(0.0)

    trade_counts = {}
    if orders_data is not None and not orders_data.empty and order_time_column in orders_data.columns:
        orders = orders_data.copy()
        orders[order_time_column] = pd.to_datetime(orders[order_time_column], errors="coerce").dt.tz_localize(None)
        if date_column == "date":
            keys = orders[order_time_column].dt.date
        else:
            keys = orders[order_time_column]
        trade_counts = keys.value_counts().to_dict()

    rows = []
    for index, row in data.iterrows():
        timestamp = row[date_column]
        key = timestamp.date() if date_column == "date" else timestamp
        rows.append(
            {
                "period": timestamp,
                "period_return": float(returns.iloc[index]) if index < len(returns) else np.nan,
                "pnl_usd": float(pnl.iloc[index]) if index < len(pnl) else np.nan,
                "end_capital": float(capital.iloc[index]) if index < len(capital) else np.nan,
                "trades": int(trade_counts.get(key, 0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_hourly_paper_trading(performance_data, orders_data=None, signals_data=None):
    if performance_data is None or performance_data.empty:
        signal_observations = 0 if signals_data is None or signals_data.empty else int(signals_data["bar_timestamp"].nunique())
        top_k = None
        if signals_data is not None and not signals_data.empty and "top_k" in signals_data.columns:
            top_k = int(pd.to_numeric(signals_data["top_k"], errors="coerce").dropna().mode().iloc[0])
        order_summary = summarize_orders(orders_data)
        return {
            "observations": signal_observations,
            "top_k": top_k,
            "start_capital": None,
            "end_capital": None,
            "profit_loss_usd": None,
            "portfolio_return": None,
            "benchmark_return": None,
            "difference": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "max_drawdown_usd": None,
            "number_of_trades": order_summary["total_trades"],
            "buys": order_summary["buys"],
            "sells": order_summary["sells"],
            "rebalancings": None,
            "total_notional_traded": order_summary["total_notional_traded"],
            "average_notional_per_period": None,
            "detail_table": pd.DataFrame(),
            "period_start": None,
            "period_end": None,
            "data_limitation": "Keine vollstaendige Hourly-Portfolio-Zeitreihe im Paper-Trading-Zeitraum.",
        }

    performance = performance_data.copy()
    time_column = "bar_timestamp" if "bar_timestamp" in performance.columns else "timestamp"
    performance[time_column] = pd.to_datetime(performance[time_column], errors="coerce").dt.tz_localize(None)
    performance = performance.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)

    summary = summarize_paper_performance_values(performance, time_column, periods_per_year=TRADING_HOURS_PER_YEAR)
    if summary is None:
        return summarize_hourly_paper_trading(None, orders_data=orders_data, signals_data=signals_data)

    order_summary = summarize_orders(orders_data)
    top_k = None
    if signals_data is not None and not signals_data.empty and "top_k" in signals_data.columns:
        top_k_values = pd.to_numeric(signals_data["top_k"], errors="coerce").dropna()
        if not top_k_values.empty:
            top_k = int(top_k_values.mode().iloc[0])

    detail = build_period_detail_table(
        performance,
        date_column=time_column,
        capital_series=summary["capital_series"],
        period_returns=summary["period_returns"],
        orders_data=orders_data,
        order_time_column="bar_timestamp" if orders_data is not None and "bar_timestamp" in orders_data.columns else "timestamp",
    )

    summary.update(
        {
            "top_k": top_k,
            "number_of_trades": order_summary["total_trades"],
            "buys": order_summary["buys"],
            "sells": order_summary["sells"],
            "rebalancings": int(detail["trades"].gt(0).sum()) if not detail.empty else 0,
            "total_notional_traded": order_summary["total_notional_traded"],
            "average_notional_per_period": (
                order_summary["total_notional_traded"] / summary["observations"]
                if order_summary["total_notional_traded"] is not None and summary["observations"]
                else None
            ),
            "detail_table": detail,
            "data_limitation": None,
        }
    )
    return summary


def summarize_hourly_hybrid_scenario(hybrid_series, start_capital=100000.0):
    """
    Summarize the honest real+reproducible-simulation Hourly series produced
    by hourly_hybrid_reporting.build_hourly_hybrid_series().

    The series always has a continuous model_index/benchmark_index (both
    normalized to 100 at the first observation, real or simulated), but the
    dollar portfolio_value/benchmark_value columns are only populated for
    real rows. We therefore rebuild a capital series from the index columns
    using a single fixed, documented notional start_capital ($100,000,
    matching the real dry-run's notional base) rather than inventing a
    capital trajectory. Trade-level fields (number_of_trades, buys, sells,
    rebalancings, total_notional_traded) are deliberately left as None: we
    have no real order log for the simulated hours, and fabricating discrete
    trade counts would violate the no-invention rule. This limitation must be
    called out wherever the scenario is shown (dashboard subtitle, inventory,
    Q&A notes).
    """
    if hybrid_series is None or hybrid_series.empty:
        return None

    working = hybrid_series.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp", "model_index", "benchmark_index"])
    working = working.sort_values("timestamp").reset_index(drop=True)
    if working.empty:
        return None

    portfolio_capital = start_capital * working["model_index"].astype(float) / working["model_index"].astype(float).iloc[0]
    benchmark_capital = start_capital * working["benchmark_index"].astype(float) / working["benchmark_index"].astype(float).iloc[0]

    portfolio_returns = portfolio_capital.pct_change().fillna(0.0)
    benchmark_returns = benchmark_capital.pct_change().fillna(0.0)
    pnl = portfolio_capital.diff().fillna(0.0)
    non_initial_returns = portfolio_returns.iloc[1:] if len(portfolio_returns) > 1 else portfolio_returns.iloc[0:0]
    non_initial_pnl = pnl.iloc[1:] if len(pnl) > 1 else pnl.iloc[0:0]

    real_mask = working["source_type"] == "real"
    simulated_mask = working["source_type"] == "simulated"

    detail_rows = []
    for index, row in working.iterrows():
        detail_rows.append(
            {
                "period": row["timestamp"],
                "period_return": float(portfolio_returns.iloc[index]),
                "pnl_usd": float(pnl.iloc[index]),
                "end_capital": float(portfolio_capital.iloc[index]),
                "trades": np.nan,
                "source_type": row["source_type"],
            }
        )
    detail = pd.DataFrame(detail_rows)

    simulation_methods = sorted({m for m in working.loc[simulated_mask, "simulation_method"].dropna().unique() if m})
    seeds = sorted({str(s) for s in working.loc[simulated_mask, "seed"].dropna().unique() if str(s)})

    return {
        "observations": int(len(working)),
        "real_observations": int(real_mask.sum()),
        "simulated_observations": int(simulated_mask.sum()),
        "top_k": None,
        "start_capital": float(portfolio_capital.iloc[0]),
        "end_capital": float(portfolio_capital.iloc[-1]),
        "profit_loss_usd": float(portfolio_capital.iloc[-1] - portfolio_capital.iloc[0]),
        "benchmark_end_capital": float(benchmark_capital.iloc[-1]),
        "benchmark_profit_loss_usd": float(benchmark_capital.iloc[-1] - benchmark_capital.iloc[0]),
        "portfolio_return": float(portfolio_capital.iloc[-1] / portfolio_capital.iloc[0] - 1),
        "benchmark_return": float(benchmark_capital.iloc[-1] / benchmark_capital.iloc[0] - 1),
        "difference": float(
            (portfolio_capital.iloc[-1] / portfolio_capital.iloc[0] - 1)
            - (benchmark_capital.iloc[-1] / benchmark_capital.iloc[0] - 1)
        ),
        "sharpe_ratio": annualized_sharpe(portfolio_returns, periods_per_year=TRADING_HOURS_PER_YEAR),
        "max_drawdown": max_drawdown_from_index(portfolio_capital),
        "max_drawdown_usd": max_drawdown_usd_from_capital(portfolio_capital),
        "best_period_return": float(non_initial_returns.max()) if not non_initial_returns.empty else np.nan,
        "best_period_usd": float(non_initial_pnl.max()) if not non_initial_pnl.empty else np.nan,
        "worst_period_return": float(non_initial_returns.min()) if not non_initial_returns.empty else np.nan,
        "worst_period_usd": float(non_initial_pnl.min()) if not non_initial_pnl.empty else np.nan,
        "period_start": working["timestamp"].min(),
        "period_end": working["timestamp"].max(),
        "period_returns": portfolio_returns,
        "benchmark_period_returns": benchmark_returns,
        "capital_series": portfolio_capital,
        "number_of_trades": None,
        "buys": None,
        "sells": None,
        "rebalancings": None,
        "total_notional_traded": None,
        "average_notional_per_period": None,
        "detail_table": detail,
        "data_limitation": (
            "Hourly-Paper-Trading-Szenario: nur "
            f"{int(real_mask.sum())} echte Beobachtung(en), "
            f"{int(simulated_mask.sum())} reproduzierbar ergaenzt "
            f"(Methode: {', '.join(simulation_methods) if simulation_methods else 'n. v.'}, "
            f"Seed: {', '.join(seeds) if seeds else 'n. v.'}). "
            "Trades/Rebalancings fuer den simulierten Anteil nicht ausgewiesen (nicht erfunden)."
        ),
    }


def per_ticker_order_counts(orders_data, ticker_column="symbol"):
    """
    Buy/sell counts per ticker from a daily/hourly order log. Returns a dict
    {ticker: {"buys": n, "sells": n}}; tickers with no orders are simply
    absent (caller fills in 0, never a guess).
    """
    if orders_data is None or orders_data.empty or ticker_column not in orders_data.columns:
        return {}

    data = orders_data.copy()
    data["action"] = data["action"].str.lower()
    counts = data.groupby([ticker_column, "action"]).size().unstack(fill_value=0)
    result = {}
    for ticker, row in counts.iterrows():
        result[ticker] = {
            "buys": int(row.get("buy", 0)),
            "sells": int(row.get("sell", 0)),
        }
    return result


# ---------------------------------------------------------------------------
# Signal / Top-K selection behaviour
# ---------------------------------------------------------------------------

def _average_holding_duration(is_selected_by_date):
    """
    Given a chronological list of booleans (selected on that signal date or
    not), return the average length (in signal periods) of consecutive True
    streaks. NaN if the ticker was never selected.
    """
    streaks = []
    current = 0
    for is_selected in is_selected_by_date:
        if is_selected:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    if not streaks:
        return np.nan
    return float(np.mean(streaks))


def summarize_signal_selection(signal_data, universe_tickers, order_counts=None):
    """
    For each ticker in the universe, compute the share of (deduplicated)
    signal dates on which it was selected for the Top-K, its average
    predicted probability, its average rank, how many times it newly
    entered the Top-K compared to the previous signal date, its average
    holding duration (consecutive signal periods once selected), and - if
    an order log is available - its real buy/sell counts.

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
    order_counts = order_counts or {}

    selection_by_date = {}
    for date in distinct_dates:
        day_data = data[data["date"] == date]
        selection_by_date[date] = set(day_data.loc[day_data["selected"] == True, "ticker"])  # noqa: E712

    for ticker in universe_tickers:
        ticker_rows = data[data["ticker"] == ticker]
        ticker_order_counts = order_counts.get(ticker, {"buys": 0, "sells": 0})

        if ticker_rows.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "selection_share": 0.0,
                    "average_probability": np.nan,
                    "average_rank": np.nan,
                    "new_entries": 0,
                    "observations": 0,
                    "average_holding_duration": np.nan,
                    "buys": ticker_order_counts["buys"],
                    "sells": ticker_order_counts["sells"],
                }
            )
            continue

        selection_share = float((ticker_rows["selected"] == True).mean())  # noqa: E712
        average_probability = float(ticker_rows["probability"].mean())
        average_rank = float(ticker_rows["rank"].mean())

        is_selected_sequence = []
        new_entries = 0
        was_selected_previous_date = False
        for date in distinct_dates:
            is_selected_today = ticker in selection_by_date[date]
            is_selected_sequence.append(is_selected_today)
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
                "average_holding_duration": _average_holding_duration(is_selected_sequence),
                "buys": ticker_order_counts["buys"],
                "sells": ticker_order_counts["sells"],
            }
        )

    result = pd.DataFrame(rows).sort_values("selection_share", ascending=False).reset_index(drop=True)
    return result, len(distinct_dates)


def summarize_signal_aggregate(signal_data, selection_df, distinct_dates):
    """
    Universe-level aggregate stats for the signal-selection plot's footer:
    most frequent Top-1 ticker, most frequent Top-3 tickers (by selection
    share), average predicted probability across all evaluated tickers/
    dates, average turnover (Top-K set changes per signal date), average
    holding duration across tickers that were ever selected, and the number
    of distinct signal timestamps.
    """
    data = signal_data.copy()
    order_column = "generated_at" if "generated_at" in data.columns else "timestamp"
    if order_column in data.columns:
        data = data.sort_values(order_column)
    data = data.drop_duplicates(subset=["date", "ticker"], keep="last")

    rank_1_rows = data[data["rank"] == 1]
    most_frequent_top1 = None
    if not rank_1_rows.empty:
        most_frequent_top1 = rank_1_rows["ticker"].value_counts().idxmax()

    most_frequent_top3 = list(selection_df.sort_values("selection_share", ascending=False)["ticker"].head(3))

    distinct_date_values = sorted(data["date"].unique())
    selection_sequence = []
    for date in distinct_date_values:
        day_data = data[data["date"] == date]
        selection_sequence.append(set(day_data.loc[day_data["selected"] == True, "ticker"]))  # noqa: E712
    average_turnover = None
    if len(selection_sequence) > 1:
        transitions = [
            len(current.symmetric_difference(previous))
            for previous, current in zip(selection_sequence[:-1], selection_sequence[1:])
        ]
        average_turnover = float(np.mean(transitions)) if transitions else 0.0

    holding_durations = selection_df["average_holding_duration"].dropna()

    return {
        "most_frequent_top1": most_frequent_top1,
        "most_frequent_top3": most_frequent_top3,
        "average_probability_overall": float(data["probability"].mean()) if not data.empty else np.nan,
        "average_turnover": average_turnover,
        "average_holding_duration_overall": float(holding_durations.mean()) if not holding_durations.empty else np.nan,
        "number_of_signal_dates": distinct_dates,
    }
