"""
Convert model signals into simple long-only paper-trading actions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def positions_by_ticker(positions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return Alpaca positions keyed by ticker."""
    return {position["ticker"]: position for position in positions}


def build_rebalance_orders(
    *,
    signals: pd.DataFrame,
    positions: list[dict[str, Any]],
    portfolio_value: float,
    universe_count: int,
    order_type: str,
    time_in_force: str,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Build target orders for one universe.

    The strategy is long only. Capital is split equally across universes and
    then equally across the selected Top-K tickers inside the universe.
    """
    if signals.empty:
        raise ValueError("No signals provided for rebalancing.")

    selected = signals[signals["selected"]].copy()
    universe = signals["universe"].iloc[0]
    generated_at = datetime.now(timezone.utc).isoformat()

    if selected.empty:
        raise ValueError(f"No selected Top-K tickers for universe: {universe}")

    selected_tickers = set(selected["ticker"])
    universe_tickers = set(signals["ticker"])
    current_positions = positions_by_ticker(positions)

    universe_budget = portfolio_value / max(universe_count, 1)
    target_value_per_position = universe_budget / len(selected)

    orders = []
    position_rows = []

    for _, row in selected.iterrows():
        ticker = row["ticker"]
        current_position = current_positions.get(ticker, {})
        current_value = float(current_position.get("market_value", 0.0))
        difference = target_value_per_position - current_value

        if abs(difference) < 1:
            action = "hold"
            reason = "Ticker remains selected and current value is close to target."
            notional = 0.0
            quantity = None
        elif difference > 0:
            action = "buy"
            reason = "Ticker is selected in Top-K."
            notional = round(difference, 2)
            quantity = None
        else:
            action = "hold"
            reason = "Ticker remains selected; no unnecessary sell order."
            notional = None
            quantity = None

        orders.append(
            {
                "date": row["date"],
                "generated_at": generated_at,
                "timeframe": row.get("timeframe", ""),
                "bar_timestamp": row.get("bar_timestamp", ""),
                "test_run_id": row.get("test_run_id", ""),
                "test_universe": row.get("test_universe", ""),
                "test_period_start": row.get("test_period_start", ""),
                "test_period_end": row.get("test_period_end", ""),
                "universe": universe,
                "ticker": ticker,
                "action": action,
                "quantity": quantity,
                "notional": notional,
                "order_type": order_type,
                "time_in_force": time_in_force,
                "dry_run": dry_run,
                "alpaca_order_id": "",
                "reason": reason,
            }
        )

        position_rows.append(
            {
                "date": row["date"],
                "generated_at": generated_at,
                "timeframe": row.get("timeframe", ""),
                "bar_timestamp": row.get("bar_timestamp", ""),
                "test_run_id": row.get("test_run_id", ""),
                "test_universe": row.get("test_universe", ""),
                "test_period_start": row.get("test_period_start", ""),
                "test_period_end": row.get("test_period_end", ""),
                "universe": universe,
                "ticker": ticker,
                "quantity": current_position.get("quantity", 0.0),
                "market_value": current_value,
                "target_value": target_value_per_position,
                "unrealized_pl": current_position.get("unrealized_pl", 0.0),
                "unrealized_plpc": current_position.get("unrealized_plpc", 0.0),
                "action": action,
                "reason": reason,
            }
        )

    for ticker, current_position in current_positions.items():
        if ticker not in universe_tickers or ticker in selected_tickers:
            continue

        orders.append(
            {
                "date": signals["date"].iloc[0],
                "generated_at": generated_at,
                "timeframe": signals["timeframe"].iloc[0],
                "bar_timestamp": signals["bar_timestamp"].iloc[0],
                "test_run_id": signals["test_run_id"].iloc[0],
                "test_universe": signals["test_universe"].iloc[0],
                "test_period_start": signals["test_period_start"].iloc[0],
                "test_period_end": signals["test_period_end"].iloc[0],
                "universe": universe,
                "ticker": ticker,
                "action": "sell",
                "quantity": current_position.get("quantity"),
                "notional": None,
                "order_type": order_type,
                "time_in_force": time_in_force,
                "dry_run": dry_run,
                "alpaca_order_id": "",
                "reason": "Ticker is no longer selected in Top-K.",
            }
        )

        position_rows.append(
            {
                "date": signals["date"].iloc[0],
                "generated_at": generated_at,
                "timeframe": signals["timeframe"].iloc[0],
                "bar_timestamp": signals["bar_timestamp"].iloc[0],
                "test_run_id": signals["test_run_id"].iloc[0],
                "test_universe": signals["test_universe"].iloc[0],
                "test_period_start": signals["test_period_start"].iloc[0],
                "test_period_end": signals["test_period_end"].iloc[0],
                "universe": universe,
                "ticker": ticker,
                "quantity": current_position.get("quantity", 0.0),
                "market_value": current_position.get("market_value", 0.0),
                "target_value": 0.0,
                "unrealized_pl": current_position.get("unrealized_pl", 0.0),
                "unrealized_plpc": current_position.get("unrealized_plpc", 0.0),
                "action": "sell",
                "reason": "Ticker is no longer selected in Top-K.",
            }
        )

    return orders, pd.DataFrame(position_rows)
