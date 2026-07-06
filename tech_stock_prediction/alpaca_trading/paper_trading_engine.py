"""
Paper Trading engine for one Alpaca workflow cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import uuid

import pandas as pd

from alpaca_trading.alpaca_client import AlpacaPaperClient
from alpaca_trading.alpaca_config import load_alpaca_settings
from alpaca_trading.portfolio_rebalancer import build_rebalance_orders
from alpaca_trading.signal_generator import generate_signals
from alpaca_trading.trading_documentation import record_trading_documentation
from config.stock_universes import get_benchmark_for_universe
from config.stock_universes import get_universe


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
    buying_power: float | None
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
            legacy_path = file_path.with_name(
                f"{file_path.stem}_legacy_{timestamp}{file_path.suffix}"
            )
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
            allow_existing_positions: bool = False,
            run_id: str | None = None,
    ):
        self.universe_name = universe_name
        self.top_k = top_k
        self.timeframe = timeframe
        self.dry_run = dry_run
        self.signals_only = signals_only
        self.universe_count = universe_count
        self.allow_existing_positions = allow_existing_positions
        self.run_id = run_id or f"single_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    def run_daily_cycle(self) -> DailyCycleResult:
        """Run one signal and paper-trading cycle."""
        benchmark = get_benchmark_for_universe(self.universe_name)
        settings = load_alpaca_settings(
            require_keys=not self.signals_only and not self.dry_run,
            dry_run_override=self.dry_run,
            top_k_override=self.top_k,
            timeframe_override=self.timeframe,
        )

        signals = generate_signals(self.universe_name, self.top_k, settings.timeframe)
        signals = self.add_run_context(signals, benchmark)
        append_csv(signals, SIGNALS_LOG)

        selected = signals[signals["selected"]].copy()
        selected_tickers = selected["ticker"].tolist()

        if self.signals_only:
            record_trading_documentation(
                run_id=self.run_id,
                universe=self.universe_name,
                timeframe=settings.timeframe,
                benchmark=benchmark,
                top_k=self.top_k,
                mode="signals_only",
                signals=signals,
                account=None,
                positions=[],
                orders=[],
            )
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
                buying_power=None,
                planned_orders=[],
            )

        client = AlpacaPaperClient(settings)
        account = client.get_account_summary()
        positions = client.get_positions()

        # In multi-universe server mode, positions from other universes are expected.
        # In single-universe mode, we keep the safety check to avoid mixed tests.
        if self.universe_count == 1:
            self.warn_or_stop_for_existing_positions(positions, settings.dry_run)

        orders, position_log = build_rebalance_orders(
            signals=signals,
            positions=positions,
            portfolio_value=account["portfolio_value"],
            universe_count=self.universe_count,
            order_type=settings.order_type,
            time_in_force=settings.time_in_force,
            dry_run=settings.dry_run,
        )
        position_log = self.add_position_log_context(position_log, benchmark)

        submitted_orders = []
        for order in orders:
            self.add_order_context(order, benchmark)
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
            order["alpaca_order_status"] = submitted.get("alpaca_order_status", "")
            submitted_orders.append(order)

        latest_positions = client.get_positions()
        append_csv(pd.DataFrame(submitted_orders), ORDERS_LOG)
        append_csv(position_log, POSITIONS_LOG)
        append_csv(
            self.build_performance_row(signals, account, submitted_orders),
            PERFORMANCE_LOG,
        )

        mode = "DRY_RUN" if settings.dry_run else "EXECUTE"
        record_trading_documentation(
            run_id=self.run_id,
            universe=self.universe_name,
            timeframe=settings.timeframe,
            benchmark=benchmark,
            top_k=self.top_k,
            mode=mode.lower(),
            signals=signals,
            account=account,
            positions=latest_positions,
            orders=submitted_orders,
        )
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
            buying_power=account["buying_power"],
            planned_orders=submitted_orders,
        )

    def add_run_context(self, signals: pd.DataFrame, benchmark: str) -> pd.DataFrame:
        """Add common server-run columns to the signal log."""
        signals = signals.copy()
        signals["run_id"] = self.run_id
        signals["timestamp"] = datetime.now(timezone.utc).isoformat()
        signals["benchmark"] = benchmark
        signals["top_k"] = self.top_k
        return signals

    def add_order_context(self, order: dict, benchmark: str) -> None:
        """Add common server-run columns to each order row."""
        order["run_id"] = self.run_id
        order["timestamp"] = datetime.now(timezone.utc).isoformat()
        order["benchmark"] = benchmark
        order["top_k"] = self.top_k

    def add_position_log_context(self, position_log: pd.DataFrame, benchmark: str) -> pd.DataFrame:
        """Add common server-run columns to the rebalancing position log."""
        if position_log.empty:
            return position_log

        position_log = position_log.copy()
        position_log["run_id"] = self.run_id
        position_log["timestamp"] = datetime.now(timezone.utc).isoformat()
        position_log["benchmark"] = benchmark
        position_log["top_k"] = self.top_k
        return position_log

    def build_performance_row(
            self,
            signals: pd.DataFrame,
            account: dict,
            orders: list[dict],
    ) -> pd.DataFrame:
        selected_tickers = signals[signals["selected"]]["ticker"].tolist()
        generated_at = datetime.now(timezone.utc).isoformat()
        bought_tickers = [order["ticker"] for order in orders if order["action"] == "buy"]
        sold_tickers = [order["ticker"] for order in orders if order["action"] == "sell"]
        held_tickers = [order["ticker"] for order in orders if order["action"] == "hold"]
        benchmark_value = signals["benchmark_close"].iloc[0] if "benchmark_close" in signals.columns else ""

        portfolio_return = ""
        benchmark_return = ""
        outperformance = ""
        if PERFORMANCE_LOG.exists():
            try:
                previous = pd.read_csv(PERFORMANCE_LOG)
                previous = previous[
                    (previous["universe"] == self.universe_name)
                    & (previous["timeframe"] == signals["timeframe"].iloc[0])
                    ]
                if not previous.empty and "benchmark_value" in previous.columns:
                    last = previous.iloc[-1]
                    previous_portfolio = float(last["portfolio_value"])
                    previous_benchmark = float(last["benchmark_value"])
                    if previous_portfolio > 0 and previous_benchmark > 0:
                        portfolio_return = account["portfolio_value"] / previous_portfolio - 1
                        benchmark_return = float(benchmark_value) / previous_benchmark - 1
                        outperformance = portfolio_return - benchmark_return
            except (ValueError, KeyError):
                pass

        return pd.DataFrame(
            [
                {
                    "date": signals["date"].iloc[0],
                    "run_id": self.run_id,
                    "generated_at": generated_at,
                    "timestamp": generated_at,
                    "timeframe": signals["timeframe"].iloc[0],
                    "bar_timestamp": signals["bar_timestamp"].iloc[0],
                    "test_run_id": signals["test_run_id"].iloc[0],
                    "test_universe": signals["test_universe"].iloc[0],
                    "test_period_start": signals["test_period_start"].iloc[0],
                    "test_period_end": signals["test_period_end"].iloc[0],
                    "universe": self.universe_name,
                    "benchmark": signals["benchmark"].iloc[0],
                    "top_k": self.top_k,
                    "portfolio_value": account["portfolio_value"],
                    "cash": account["cash"],
                    "benchmark_value": benchmark_value,
                    "selected_tickers": ",".join(selected_tickers),
                    "bought_tickers": ",".join(bought_tickers),
                    "sold_tickers": ",".join(sold_tickers),
                    "held_tickers": ",".join(held_tickers),
                    "portfolio_return_since_last": portfolio_return,
                    "benchmark_return_since_last": benchmark_return,
                    "outperformance_since_last": outperformance,
                    "notes": "Paper Trading daily cycle",
                }
            ]
        )

    def warn_or_stop_for_existing_positions(self, positions: list[dict], dry_run: bool) -> None:
        """Protect a shared Paper Trading account from mixed universe tests."""
        if not positions:
            return

        universe_tickers = set(get_universe(self.universe_name))
        outside_positions = [
            position
            for position in positions
            if position["ticker"] not in universe_tickers
               and abs(float(position.get("quantity", 0.0))) > 0
        ]

        if not outside_positions:
            return

        tickers = ", ".join(position["ticker"] for position in outside_positions)
        message = (
            "WARNING: Existing positions outside the selected universe were found: "
            f"{tickers}. This is a shared Paper Trading account. Close old positions "
            "before testing another universe, or explicitly acknowledge this risk."
        )

        if dry_run:
            print(message)
            return

        if not self.allow_existing_positions:
            raise RuntimeError(
                message
                + " For --execute, close these positions first or rerun with "
                  "--allow-existing-positions."
            )

        print(message)

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
