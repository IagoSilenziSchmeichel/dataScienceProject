"""
Paper Trading engine for one Alpaca workflow cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpaca_trading.alpaca_client import AlpacaPaperClient
from alpaca_trading.alpaca_config import load_alpaca_settings
from alpaca_trading.portfolio_rebalancer import build_rebalance_orders
from alpaca_trading.signal_generator import generate_signals
from config.stock_universes import get_benchmark_for_universe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "alpaca_trading" / "logs"

SIGNALS_LOG = LOG_DIR / "paper_signals.csv"
ORDERS_LOG = LOG_DIR / "paper_orders.csv"
POSITIONS_LOG = LOG_DIR / "paper_positions.csv"
PERFORMANCE_LOG = LOG_DIR / "paper_performance.csv"


@dataclass
class DailyCycleResult:
    universe: str
    benchmark: str
    top_k: int
    timeframe: str
    selected_tickers: list[str]
    mode: str
    portfolio_value: float | None
    cash: float | None
    planned_orders: list[dict]


def append_csv(rows: pd.DataFrame, file_path: Path) -> None:
    """Append rows to a CSV log file."""
    if rows.empty:
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        existing_header = file_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        new_header = list(rows.columns)

        if existing_header != new_header:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            legacy_path = file_path.with_name(f"{file_path.stem}_legacy_{timestamp}{file_path.suffix}")
            file_path.rename(legacy_path)
            print(f"Archived old log schema: {legacy_path}")

    write_header = not file_path.exists()
    rows.to_csv(file_path, mode="a", header=write_header, index=False)


class PaperTradingEngine:
    """Coordinate signal generation, rebalancing and Alpaca orders."""

    def __init__(
        self,
        *,
        universe_name: str,
        top_k: int = 5,
        timeframe: str = "1Hour",
        dry_run: bool = True,
        signals_only: bool = False,
        universe_count: int = 1,
    ):
        self.universe_name = universe_name
        self.top_k = top_k
        self.timeframe = timeframe
        self.dry_run = dry_run
        self.signals_only = signals_only
        self.universe_count = universe_count

    def run_daily_cycle(self) -> DailyCycleResult:
        """Run one signal and paper-trading cycle."""
        benchmark = get_benchmark_for_universe(self.universe_name)
        settings = load_alpaca_settings(
            require_keys=not self.signals_only,
            dry_run_override=self.dry_run,
            top_k_override=self.top_k,
            timeframe_override=self.timeframe,
        )

        signals = generate_signals(self.universe_name, self.top_k, settings.timeframe)
        append_csv(signals, SIGNALS_LOG)

        selected = signals[signals["selected"]].copy()
        selected_tickers = selected["ticker"].tolist()

        if self.signals_only:
            self.print_summary(
                benchmark=benchmark,
                selected=selected,
                account=None,
                orders=[],
                mode="SIGNALS_ONLY",
            )
            return DailyCycleResult(
                universe=self.universe_name,
                benchmark=benchmark,
                top_k=self.top_k,
                timeframe=settings.timeframe,
                selected_tickers=selected_tickers,
                mode="SIGNALS_ONLY",
                portfolio_value=None,
                cash=None,
                planned_orders=[],
            )

        client = AlpacaPaperClient(settings)
        account = client.get_account_summary()
        positions = client.get_positions()

        orders, position_log = build_rebalance_orders(
            signals=signals,
            positions=positions,
            portfolio_value=account["portfolio_value"],
            universe_count=self.universe_count,
            order_type=settings.order_type,
            time_in_force=settings.time_in_force,
            dry_run=settings.dry_run,
        )

        submitted_orders = []
        for order in orders:
            if order["action"] == "hold":
                submitted_orders.append(order)
                continue

            submitted = client.submit_order(
                ticker=order["ticker"],
                side=order["action"],
                quantity=order["quantity"],
                notional=order["notional"],
                reason=order["reason"],
            )
            order["alpaca_order_id"] = submitted.get("alpaca_order_id", "")
            submitted_orders.append(order)

        append_csv(pd.DataFrame(submitted_orders), ORDERS_LOG)
        append_csv(position_log, POSITIONS_LOG)
        append_csv(self.build_performance_row(signals, account), PERFORMANCE_LOG)

        mode = "DRY_RUN" if settings.dry_run else "EXECUTE"
        self.print_summary(
            benchmark=benchmark,
            selected=selected,
            account=account,
            orders=submitted_orders,
            mode=mode,
        )

        return DailyCycleResult(
            universe=self.universe_name,
            benchmark=benchmark,
            top_k=self.top_k,
            timeframe=settings.timeframe,
            selected_tickers=selected_tickers,
            mode=mode,
            portfolio_value=account["portfolio_value"],
            cash=account["cash"],
            planned_orders=submitted_orders,
        )

    def build_performance_row(self, signals: pd.DataFrame, account: dict) -> pd.DataFrame:
        selected_tickers = signals[signals["selected"]]["ticker"].tolist()
        generated_at = datetime.now(timezone.utc).isoformat()

        return pd.DataFrame(
            [
                {
                    "date": signals["date"].iloc[0],
                    "generated_at": generated_at,
                    "timeframe": signals["timeframe"].iloc[0],
                    "bar_timestamp": signals["bar_timestamp"].iloc[0],
                    "universe": self.universe_name,
                    "benchmark": signals["benchmark"].iloc[0],
                    "portfolio_value": account["portfolio_value"],
                    "cash": account["cash"],
                    "selected_tickers": ",".join(selected_tickers),
                    "benchmark_close": "",
                    "notes": "Paper Trading daily cycle",
                }
            ]
        )

    def print_summary(
        self,
        *,
        benchmark: str,
        selected: pd.DataFrame,
        account: dict | None,
        orders: list[dict],
        mode: str,
    ) -> None:
        print("\nPaper Trading Daily Cycle")
        print("=========================")
        print(f"Universe: {self.universe_name}")
        print(f"Benchmark: {benchmark}")
        print(f"Top-K: {self.top_k}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Mode: {mode}")

        if account:
            print(f"Portfolio Value: {account['portfolio_value']:.2f}")
            print(f"Cash: {account['cash']:.2f}")
            print(f"Buying Power: {account['buying_power']:.2f}")

        print("\nSelected tickers:")
        for _, row in selected.iterrows():
            print(f"- {row['ticker']} | Probability: {row['probability']:.4f}")

        print("\nPlanned orders:")
        if not orders:
            print("- No orders planned.")
        for order in orders:
            print(
                f"- {order['action'].upper()} {order['ticker']} "
                f"notional={order['notional']} quantity={order['quantity']} "
                f"reason={order['reason']}"
            )
