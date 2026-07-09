"""
Automatic documentation for Alpaca Paper Trading runs.

This module only records and visualizes run data. It does not change the model,
training pipeline or trading logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config.stock_universes import get_benchmark_for_universe
from config.stock_universes import get_universe
from config.stock_universes import list_available_universes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "alpaca_trading" / "logs"
REPORT_DIR = PROJECT_ROOT / "alpaca_trading" / "reports"

SCHEDULER_LOG_FILE = LOG_DIR / "scheduler_log.csv"
UNIVERSE_POSITION_REGISTRY_FILE = LOG_DIR / "universe_position_registry.csv"
TRADING_SNAPSHOTS_FILE = LOG_DIR / "trading_snapshots.csv"
POSITIONS_HISTORY_FILE = LOG_DIR / "positions_history.csv"
ORDERS_HISTORY_FILE = LOG_DIR / "orders_history.csv"
SIGNAL_HISTORY_FILE = LOG_DIR / "signal_history.csv"
PERFORMANCE_HISTORY_FILE = LOG_DIR / "performance_history.csv"
DAILY_SUMMARY_FILE = REPORT_DIR / "daily_summary.md"
MULTI_UNIVERSE_SUMMARY_FILE = REPORT_DIR / "multi_universe_summary.md"

SCHEDULER_LOG_COLUMNS = [
    "timestamp",
    "run_id",
    "mode",
    "command",
    "status",
    "return_code",
    "message",
]

UNIVERSE_POSITION_REGISTRY_COLUMNS = [
    "run_id",
    "timestamp",
    "universe",
    "benchmark",
    "timeframe",
    "top_k",
    "ticker",
    "opened_at",
    "last_seen_at",
    "target_weight",
    "current_status",
    "alpaca_order_id",
]

TRADING_SNAPSHOT_COLUMNS = [
    "run_id",
    "timestamp",
    "universe",
    "timeframe",
    "benchmark",
    "top_k",
    "portfolio_value",
    "cash",
    "buying_power",
    "number_of_positions",
    "selected_top_k",
    "average_probability",
    "highest_probability",
    "lowest_probability",
    "mode",
]

POSITIONS_HISTORY_COLUMNS = [
    "run_id",
    "timestamp",
    "universe",
    "benchmark",
    "timeframe",
    "top_k",
    "symbol",
    "quantity",
    "entry_price",
    "current_price",
    "market_value",
    "unrealized_pl",
    "unrealized_pl_percent",
]

ORDERS_HISTORY_COLUMNS = [
    "run_id",
    "timestamp",
    "universe",
    "benchmark",
    "timeframe",
    "top_k",
    "symbol",
    "action",
    "quantity",
    "notional",
    "reason",
    "alpaca_order_id",
    "status",
]

SIGNAL_HISTORY_COLUMNS = [
    "run_id",
    "timestamp",
    "universe",
    "benchmark",
    "timeframe",
    "top_k",
    "symbol",
    "predicted_probability",
    "selected_for_top_k",
    "rank",
]

PERFORMANCE_HISTORY_COLUMNS = [
    "run_id",
    "timestamp",
    "universe",
    "benchmark",
    "timeframe",
    "top_k",
    "portfolio_return",
    "benchmark_return",
    "outperformance",
    "portfolio_value",
    "benchmark_value",
]


def ensure_csv_file(file_path: Path, columns: list[str]) -> None:
    """Create a CSV file with headers if it does not exist yet."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)


def ensure_documentation_files() -> None:
    """Create all append-only documentation CSVs before writing a run."""
    ensure_csv_file(SCHEDULER_LOG_FILE, SCHEDULER_LOG_COLUMNS)
    ensure_csv_file(UNIVERSE_POSITION_REGISTRY_FILE, UNIVERSE_POSITION_REGISTRY_COLUMNS)
    ensure_csv_file(TRADING_SNAPSHOTS_FILE, TRADING_SNAPSHOT_COLUMNS)
    ensure_csv_file(POSITIONS_HISTORY_FILE, POSITIONS_HISTORY_COLUMNS)
    ensure_csv_file(ORDERS_HISTORY_FILE, ORDERS_HISTORY_COLUMNS)
    ensure_csv_file(SIGNAL_HISTORY_FILE, SIGNAL_HISTORY_COLUMNS)
    ensure_csv_file(PERFORMANCE_HISTORY_FILE, PERFORMANCE_HISTORY_COLUMNS)


def append_rows(rows: pd.DataFrame, file_path: Path) -> None:
    """Append rows to a CSV file and create it if needed."""
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
            print(f"Archived old documentation schema: {legacy_path}")

    write_header = not file_path.exists()
    rows.to_csv(file_path, mode="a", header=write_header, index=False)


def safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def find_universe_for_ticker(ticker: str, fallback_universe: str) -> str:
    """Map an open position back to the configured stock universe."""
    for universe_name in list_available_universes():
        if ticker in get_universe(universe_name):
            return universe_name
    return fallback_universe


def build_trading_snapshot(
    *,
    run_id: str,
    timestamp: str,
    universe: str,
    timeframe: str,
    benchmark: str,
    top_k: int,
    signals: pd.DataFrame,
    account: dict | None,
    positions: list[dict],
    mode: str,
) -> pd.DataFrame:
    selected = signals[signals["selected"]].copy()
    probabilities = selected["probability"] if not selected.empty else signals["probability"]

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "universe": universe,
        "timeframe": timeframe,
        "benchmark": benchmark,
        "top_k": top_k,
        "portfolio_value": account.get("portfolio_value") if account else "",
        "cash": account.get("cash") if account else "",
        "buying_power": account.get("buying_power") if account else "",
        "number_of_positions": len(positions),
        "selected_top_k": ",".join(selected["ticker"].tolist()),
        "average_probability": probabilities.mean() if not probabilities.empty else "",
        "highest_probability": signals["probability"].max() if not signals.empty else "",
        "lowest_probability": signals["probability"].min() if not signals.empty else "",
        "mode": mode,
    }

    return pd.DataFrame([row], columns=TRADING_SNAPSHOT_COLUMNS)


def build_positions_history(
    *,
    run_id: str,
    timestamp: str,
    universe: str,
    benchmark: str,
    timeframe: str,
    top_k: int,
    positions: list[dict],
) -> pd.DataFrame:
    rows = []

    for position in positions:
        symbol = position.get("ticker", "")
        position_universe = find_universe_for_ticker(symbol, universe)
        position_benchmark = get_benchmark_for_universe(position_universe)
        rows.append(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "universe": position_universe,
                "benchmark": position_benchmark,
                "timeframe": timeframe,
                "top_k": top_k,
                "symbol": symbol,
                "quantity": position.get("quantity", ""),
                "entry_price": position.get("entry_price", ""),
                "current_price": position.get("current_price", ""),
                "market_value": position.get("market_value", ""),
                "unrealized_pl": position.get("unrealized_pl", ""),
                "unrealized_pl_percent": position.get("unrealized_plpc", ""),
            }
        )

    return pd.DataFrame(rows, columns=POSITIONS_HISTORY_COLUMNS)


def order_status(order: dict) -> str:
    if order.get("action") == "hold":
        return "held"
    if order.get("dry_run"):
        return "dry_run"
    if order.get("alpaca_order_status"):
        return order["alpaca_order_status"]
    if order.get("alpaca_order_id"):
        return "submitted"
    return "planned"


def build_orders_history(
    *,
    run_id: str,
    timestamp: str,
    universe: str,
    benchmark: str,
    timeframe: str,
    top_k: int,
    orders: list[dict],
) -> pd.DataFrame:
    rows = []

    for order in orders:
        order_universe = order.get("universe", universe)
        rows.append(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "universe": order_universe,
                "benchmark": get_benchmark_for_universe(order_universe),
                "timeframe": order.get("timeframe", timeframe),
                "top_k": top_k,
                "symbol": order.get("ticker", ""),
                "action": order.get("action", ""),
                "quantity": order.get("quantity", ""),
                "notional": order.get("notional", ""),
                "reason": order.get("reason", ""),
                "alpaca_order_id": order.get("alpaca_order_id", ""),
                "status": order_status(order),
            }
        )

    return pd.DataFrame(rows, columns=ORDERS_HISTORY_COLUMNS)


def build_signal_history(
    *,
    run_id: str,
    timestamp: str,
    universe: str,
    benchmark: str,
    timeframe: str,
    top_k: int,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": run_id,
            "timestamp": timestamp,
            "universe": universe,
            "benchmark": benchmark,
            "timeframe": timeframe,
            "top_k": top_k,
            "symbol": signals["ticker"],
            "predicted_probability": signals["probability"],
            "selected_for_top_k": signals["selected"],
            "rank": signals["rank"],
        },
        columns=SIGNAL_HISTORY_COLUMNS,
    )


def latest_previous_performance(universe: str, timeframe: str) -> pd.Series | None:
    if not PERFORMANCE_HISTORY_FILE.exists():
        return None

    data = pd.read_csv(PERFORMANCE_HISTORY_FILE)
    if data.empty or "universe" not in data.columns or "timeframe" not in data.columns:
        return None

    subset = data[(data["universe"] == universe) & (data["timeframe"] == timeframe)].copy()
    if subset.empty:
        return None

    return subset.iloc[-1]


def build_performance_history(
    *,
    run_id: str,
    timestamp: str,
    universe: str,
    benchmark: str,
    timeframe: str,
    top_k: int,
    account: dict | None,
    benchmark_value: float | None,
) -> pd.DataFrame:
    portfolio_value = safe_float(account.get("portfolio_value")) if account else None
    previous = latest_previous_performance(universe, timeframe)
    portfolio_return = ""
    benchmark_return = ""
    outperformance = ""

    if previous is not None and portfolio_value is not None and benchmark_value is not None:
        previous_portfolio = safe_float(previous.get("portfolio_value"))
        previous_benchmark = safe_float(previous.get("benchmark_value"))
        if previous_portfolio and previous_benchmark:
            portfolio_return = portfolio_value / previous_portfolio - 1
            benchmark_return = benchmark_value / previous_benchmark - 1
            outperformance = portfolio_return - benchmark_return

    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "universe": universe,
                "benchmark": benchmark,
                "timeframe": timeframe,
                "top_k": top_k,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "outperformance": outperformance,
                "portfolio_value": portfolio_value if portfolio_value is not None else "",
                "benchmark_value": benchmark_value if benchmark_value is not None else "",
            }
        ],
        columns=PERFORMANCE_HISTORY_COLUMNS,
    )


def build_universe_position_registry(
    *,
    run_id: str,
    timestamp: str,
    universe: str,
    benchmark: str,
    timeframe: str,
    top_k: int,
    orders: list[dict],
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Track which selected or traded ticker belongs to which universe."""
    selected_tickers = set(signals[signals["selected"]]["ticker"].tolist())
    order_by_ticker = {order.get("ticker", ""): order for order in orders}
    target_weight = 1 / max(len(selected_tickers), 1)
    rows = []

    for ticker in sorted(selected_tickers | set(order_by_ticker.keys())):
        order = order_by_ticker.get(ticker, {})
        action = order.get("action", "selected")
        if action == "sell":
            status = "closed"
        elif action == "buy":
            status = "opened"
        elif action == "hold":
            status = "held"
        else:
            status = "selected_signal"

        rows.append(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "universe": universe,
                "benchmark": benchmark,
                "timeframe": timeframe,
                "top_k": top_k,
                "ticker": ticker,
                "opened_at": timestamp if status in {"opened", "selected_signal"} else "",
                "last_seen_at": timestamp,
                "target_weight": target_weight if ticker in selected_tickers else 0.0,
                "current_status": status,
                "alpaca_order_id": order.get("alpaca_order_id", ""),
            }
        )

    return pd.DataFrame(rows, columns=UNIVERSE_POSITION_REGISTRY_COLUMNS)


def append_scheduler_log(
    *,
    run_id: str,
    mode: str,
    command: str,
    status: str,
    return_code: int | None,
    message: str,
) -> None:
    """Record one scheduler decision or subprocess result."""
    ensure_csv_file(SCHEDULER_LOG_FILE, SCHEDULER_LOG_COLUMNS)
    rows = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "mode": mode,
                "command": command,
                "status": status,
                "return_code": return_code if return_code is not None else "",
                "message": message,
            }
        ],
        columns=SCHEDULER_LOG_COLUMNS,
    )
    append_rows(rows, SCHEDULER_LOG_FILE)


def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


def latest_portfolio_value() -> float | None:
    snapshots = load_csv(TRADING_SNAPSHOTS_FILE)
    if snapshots.empty or "portfolio_value" not in snapshots.columns:
        return None

    values = pd.to_numeric(snapshots["portfolio_value"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def add_normalized_series(group: pd.DataFrame, value_column: str, normalized_column: str) -> pd.DataFrame:
    group = group.copy()
    values = pd.to_numeric(group[value_column], errors="coerce")
    first_valid = values.dropna()
    if first_valid.empty or first_valid.iloc[0] == 0:
        group[normalized_column] = pd.NA
        return group

    group[normalized_column] = values / first_valid.iloc[0] * 100
    return group


def save_line_plot(data: pd.DataFrame, x: str, y_columns: list[str], title: str, file_name: str) -> None:
    plot_file = REPORT_DIR / file_name
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    has_line = False
    for column in y_columns:
        if column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce")
            if values.notna().any():
                ax.plot(pd.to_datetime(data[x]), values, marker="o", linewidth=2, label=column)
                has_line = True

    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.25)
    if has_line:
        ax.legend()
        fig.autofmt_xdate()
    else:
        ax.text(0.5, 0.5, "Not enough data yet", ha="center", va="center")
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_portfolio_vs_benchmark_plot() -> None:
    data = load_csv(PERFORMANCE_HISTORY_FILE)
    plot_file = REPORT_DIR / "portfolio_vs_benchmark.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    if not data.empty:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        grouped = []
        for _, group in data.groupby(["universe", "timeframe"], dropna=False):
            group = add_normalized_series(group, "portfolio_value", "portfolio_index")
            group = add_normalized_series(group, "benchmark_value", "benchmark_index")
            grouped.append(group)

        normalized = pd.concat(grouped, ignore_index=True) if grouped else pd.DataFrame()
        if not normalized.empty:
            portfolio = normalized.groupby("timestamp", as_index=False)["portfolio_index"].mean()
            benchmark = normalized.groupby("timestamp", as_index=False)["benchmark_index"].mean()

            if portfolio["portfolio_index"].notna().any():
                ax.plot(
                    portfolio["timestamp"],
                    portfolio["portfolio_index"],
                    marker="o",
                    linewidth=2,
                    label="Portfolio index",
                )
            if benchmark["benchmark_index"].notna().any():
                ax.plot(
                    benchmark["timestamp"],
                    benchmark["benchmark_index"],
                    marker="o",
                    linestyle="--",
                    linewidth=2,
                    label="Benchmark index",
                )

        if ax.has_data():
            ax.axhline(100, color="black", linewidth=1, alpha=0.35)
            ax.set_title("Portfolio vs Benchmark (Start = 100)", fontweight="bold")
            ax.set_ylabel("Index value")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.autofmt_xdate()
        else:
            ax.text(0.5, 0.5, "Not enough performance data yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No performance data yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_position_allocation() -> None:
    data = load_csv(POSITIONS_HISTORY_FILE)
    plot_file = REPORT_DIR / "position_allocation.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    if not data.empty:
        latest_timestamp = data["timestamp"].max()
        latest = data[data["timestamp"] == latest_timestamp].copy()
        latest["market_value"] = pd.to_numeric(latest["market_value"], errors="coerce").fillna(0)
        latest = latest[latest["market_value"] > 0].copy()
        if not latest.empty:
            portfolio_value = latest_portfolio_value()
            invested_value = float(latest["market_value"].sum())
            denominator = portfolio_value if portfolio_value and portfolio_value > 0 else invested_value

            allocation = latest.groupby("symbol", as_index=False)["market_value"].sum()
            allocation["portfolio_weight"] = allocation["market_value"] / denominator * 100

            cash_value = max(denominator - invested_value, 0.0)
            if cash_value > 1:
                allocation = pd.concat(
                    [
                        allocation,
                        pd.DataFrame(
                            [{"symbol": "Cash", "market_value": cash_value, "portfolio_weight": cash_value / denominator * 100}]
                        ),
                    ],
                    ignore_index=True,
                )

            allocation = allocation.sort_values("portfolio_weight", ascending=True)
            colors = ["#8A8F98" if symbol == "Cash" else "#4F8F6F" for symbol in allocation["symbol"]]
            ax.barh(allocation["symbol"], allocation["portfolio_weight"], color=colors)
            ax.set_title("Current Position Allocation (% of Portfolio)", fontweight="bold")
            ax.set_xlabel("Portfolio weight (%)")
            ax.grid(axis="x", alpha=0.25)

            for index, value in enumerate(allocation["portfolio_weight"]):
                ax.text(value + 0.15, index, f"{value:.1f}%", va="center", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No open positions", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No position history yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_probability_history() -> None:
    data = load_csv(SIGNAL_HISTORY_FILE)
    plot_file = REPORT_DIR / "probability_history.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    if not data.empty:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        selected = data[data["selected_for_top_k"] == True].copy()
        if selected.empty:
            ax.text(0.5, 0.5, "No selected signals yet", ha="center", va="center")
        else:
            for symbol, group in selected.groupby("symbol"):
                ax.plot(group["timestamp"], group["predicted_probability"], marker="o", linewidth=1.8, label=symbol)
            ax.set_title("Selected Signal Probability History", fontweight="bold")
            ax.set_ylabel("Predicted probability")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.autofmt_xdate()
    else:
        ax.text(0.5, 0.5, "No signal history yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_daily_return_distribution() -> None:
    data = load_csv(PERFORMANCE_HISTORY_FILE)
    plot_file = REPORT_DIR / "daily_return_distribution.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    if not data.empty and "portfolio_return" in data.columns:
        returns = pd.to_numeric(data["portfolio_return"], errors="coerce").dropna()
        if not returns.empty:
            ax.hist(returns, bins=20, color="#4F8F6F", edgecolor="white")
            ax.set_title("Portfolio Return Distribution", fontweight="bold")
            ax.set_xlabel("Return per run")
            ax.grid(axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "Not enough return data yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No performance history yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_trade_activity() -> None:
    data = load_csv(ORDERS_HISTORY_FILE)
    plot_file = REPORT_DIR / "trade_activity.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    if not data.empty:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["date"] = data["timestamp"].dt.date
        counts = data[data["action"].isin(["buy", "sell"])].groupby(["date", "action"]).size().unstack(fill_value=0)
        if not counts.empty:
            counts.plot(kind="bar", ax=ax, color=["#4F8F6F", "#B85C5C"])
            ax.set_title("Trade Activity", fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of orders")
            ax.grid(axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No buy/sell orders yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No order history yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_multi_universe_portfolio_vs_benchmark() -> None:
    data = load_csv(PERFORMANCE_HISTORY_FILE)
    plot_file = REPORT_DIR / "multi_universe_portfolio_vs_benchmark.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    if not data.empty:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp")
        for universe, group in data.groupby("universe"):
            group = group.sort_values("timestamp")
            group = add_normalized_series(group, "portfolio_value", "portfolio_index")
            group = add_normalized_series(group, "benchmark_value", "benchmark_index")
            if group["portfolio_index"].notna().any():
                ax.plot(group["timestamp"], group["portfolio_index"], marker="o", linewidth=2, label=f"{universe} portfolio")
            if group["benchmark_index"].notna().any():
                ax.plot(group["timestamp"], group["benchmark_index"], linestyle="--", alpha=0.65, label=f"{universe} benchmark")
        if ax.has_data():
            ax.axhline(100, color="black", linewidth=1, alpha=0.35)
            ax.set_title("Multi-Universe Portfolio vs Benchmark (Start = 100)", fontweight="bold")
            ax.set_ylabel("Index value")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.autofmt_xdate()
        else:
            ax.text(0.5, 0.5, "Not enough performance data yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No performance data yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_multi_universe_outperformance() -> None:
    data = load_csv(PERFORMANCE_HISTORY_FILE)
    plot_file = REPORT_DIR / "multi_universe_outperformance.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    if not data.empty:
        data["outperformance"] = pd.to_numeric(data["outperformance"], errors="coerce")
        latest = data.dropna(subset=["outperformance"]).groupby("universe").tail(1)
        if not latest.empty:
            colors = ["#4F8F6F" if value >= 0 else "#B85C5C" for value in latest["outperformance"]]
            ax.bar(latest["universe"], latest["outperformance"], color=colors)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title("Latest Outperformance by Universe", fontweight="bold")
            ax.set_ylabel("Outperformance since last run")
            ax.grid(axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "Not enough return data yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No performance data yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_trade_activity_by_universe() -> None:
    data = load_csv(ORDERS_HISTORY_FILE)
    plot_file = REPORT_DIR / "trade_activity_by_universe.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    if not data.empty:
        trades = data[data["action"].isin(["buy", "sell"])].copy()
        if not trades.empty:
            counts = trades.groupby(["universe", "action"]).size().unstack(fill_value=0)
            counts.plot(kind="bar", ax=ax, color=["#4F8F6F", "#B85C5C"])
            ax.set_title("Trade Activity by Universe", fontweight="bold")
            ax.set_xlabel("Universe")
            ax.set_ylabel("Number of orders")
            ax.grid(axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No buy/sell orders yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No order history yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_current_allocation_by_universe() -> None:
    data = load_csv(POSITIONS_HISTORY_FILE)
    plot_file = REPORT_DIR / "current_allocation_by_universe.png"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    if not data.empty:
        data["market_value"] = pd.to_numeric(data["market_value"], errors="coerce").fillna(0)
        latest_rows = data.sort_values("timestamp").groupby(["universe", "symbol"]).tail(1)
        allocation = latest_rows.groupby("universe")["market_value"].sum()
        allocation = allocation[allocation > 0]
        if not allocation.empty:
            ax.pie(allocation, labels=allocation.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Current Allocation by Universe", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No open allocation yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No position history yet", ha="center", va="center")

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_multi_universe_plots() -> None:
    save_multi_universe_portfolio_vs_benchmark()
    save_multi_universe_outperformance()
    save_trade_activity_by_universe()
    save_current_allocation_by_universe()


def update_plots() -> None:
    performance = load_csv(PERFORMANCE_HISTORY_FILE)
    if not performance.empty:
        save_portfolio_vs_benchmark_plot()
        save_line_plot(
            performance,
            "timestamp",
            ["portfolio_value"],
            "Portfolio Value",
            "portfolio_value.png",
        )
    else:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    save_position_allocation()
    save_probability_history()
    save_daily_return_distribution()
    save_trade_activity()
    update_multi_universe_plots()


def update_daily_summary(timestamp: str) -> None:
    snapshots = load_csv(TRADING_SNAPSHOTS_FILE)
    positions = load_csv(POSITIONS_HISTORY_FILE)
    orders = load_csv(ORDERS_HISTORY_FILE)
    signals = load_csv(SIGNAL_HISTORY_FILE)
    performance = load_csv(PERFORMANCE_HISTORY_FILE)

    latest_snapshot = snapshots.iloc[-1] if not snapshots.empty else {}
    latest_performance = performance.iloc[-1] if not performance.empty else {}
    current_day = str(pd.to_datetime(timestamp).date())

    buys = 0
    sells = 0
    if not orders.empty:
        orders["timestamp"] = pd.to_datetime(orders["timestamp"])
        today_orders = orders[orders["timestamp"].dt.date.astype(str) == current_day]
        buys = int((today_orders["action"] == "buy").sum())
        sells = int((today_orders["action"] == "sell").sum())

    latest_positions = pd.DataFrame()
    if not positions.empty:
        latest_timestamp = positions["timestamp"].max()
        latest_positions = positions[positions["timestamp"] == latest_timestamp].copy()

    top_signals = pd.DataFrame()
    if not signals.empty:
        latest_signal_timestamp = signals["timestamp"].max()
        top_signals = signals[signals["timestamp"] == latest_signal_timestamp].sort_values("rank").head(5)

    best_position = ""
    worst_position = ""
    if not latest_positions.empty and "unrealized_pl" in latest_positions.columns:
        latest_positions["unrealized_pl"] = pd.to_numeric(latest_positions["unrealized_pl"], errors="coerce")
        if latest_positions["unrealized_pl"].notna().any():
            best = latest_positions.sort_values("unrealized_pl", ascending=False).iloc[0]
            worst = latest_positions.sort_values("unrealized_pl", ascending=True).iloc[0]
            best_position = f"{best['symbol']} ({best['unrealized_pl']})"
            worst_position = f"{worst['symbol']} ({worst['unrealized_pl']})"

    positions_text = "Keine offenen Positionen"
    if not latest_positions.empty:
        positions_text = "\n".join(
            f"- {row['symbol']}: {row['quantity']} shares, value {row['market_value']}"
            for _, row in latest_positions.iterrows()
        )

    top_signals_text = "Keine Signale"
    if not top_signals.empty:
        top_signals_text = "\n".join(
            f"- {row['symbol']}: rank {row['rank']}, probability {float(row['predicted_probability']):.4f}"
            for _, row in top_signals.iterrows()
        )

    content = f"""# Daily Alpaca Paper Trading Summary

## Date

{current_day}

## Portfolio

- Portfolio value: {latest_snapshot.get('portfolio_value', '')}
- Daily return: {latest_performance.get('portfolio_return', '')}
- Benchmark: {latest_snapshot.get('benchmark', '')}
- Benchmark value: {latest_performance.get('benchmark_value', '')}
- Outperformance: {latest_performance.get('outperformance', '')}

## Trading Activity

- Buys: {buys}
- Sells: {sells}

## Current Positions

{positions_text}

## Top-5 Signals

{top_signals_text}

## Best / Worst Position

- Best position: {best_position}
- Worst position: {worst_position}
"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_SUMMARY_FILE.write_text(content, encoding="utf-8")


def format_order_list(orders: list[dict]) -> str:
    if not orders:
        return "Keine Orders"

    return "\n".join(
        f"- {order.get('action', '').upper()} {order.get('ticker', '')} "
        f"notional={order.get('notional', '')} status={order_status(order)}"
        for order in orders
    )


def update_multi_universe_summary(
    *,
    run_id: str,
    timestamp: str,
    mode: str,
    timeframe: str,
    top_k: int,
    results: list[Any],
    errors: list[dict],
) -> None:
    """Write a readable report for the latest all-universes run."""
    sections = []

    for result in results:
        orders_text = format_order_list(result.planned_orders)
        selected = ", ".join(result.selected_tickers) if result.selected_tickers else "Keine"
        portfolio_value = result.portfolio_value if result.portfolio_value is not None else ""
        cash = result.cash if result.cash is not None else ""
        buying_power = result.buying_power if result.buying_power is not None else ""
        sections.append(
            f"""## {result.universe}

- Benchmark: {result.benchmark}
- Top-K: {result.top_k}
- Timeframe: {result.timeframe}
- Selected stocks: {selected}
- Portfolio Value: {portfolio_value}
- Cash: {cash}
- Buying Power: {buying_power}

### Orders

{orders_text}
"""
        )

    error_text = "Keine Fehler"
    if errors:
        error_text = "\n".join(
            f"- {error.get('universe', '')}: {error.get('message', '')}"
            for error in errors
        )

    configured_universes = "\n".join(
        f"- {universe}: benchmark {get_benchmark_for_universe(universe)}"
        for universe in list_available_universes()
    )

    content = f"""# Multi-Universe Paper Trading Summary

## Run

- Zeitpunkt: {timestamp}
- Run ID: {run_id}
- Mode: {mode}
- Timeframe: {timeframe}
- Top-K: {top_k}

## Configured Universes

{configured_universes}

## Universe Results

{chr(10).join(sections)}

## Fehler / offene Hinweise

{error_text}
"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MULTI_UNIVERSE_SUMMARY_FILE.write_text(content, encoding="utf-8")
    update_multi_universe_plots()


def record_trading_documentation(
    *,
    run_id: str,
    universe: str,
    timeframe: str,
    benchmark: str,
    top_k: int,
    mode: str,
    signals: pd.DataFrame,
    account: dict | None,
    positions: list[dict],
    orders: list[dict],
) -> None:
    """Write all scientific documentation artifacts for one run."""
    ensure_documentation_files()
    timestamp = datetime.now(timezone.utc).isoformat()
    benchmark_value = safe_float(signals["benchmark_close"].iloc[0]) if "benchmark_close" in signals.columns else None

    append_rows(
        build_trading_snapshot(
            run_id=run_id,
            timestamp=timestamp,
            universe=universe,
            timeframe=timeframe,
            benchmark=benchmark,
            top_k=top_k,
            signals=signals,
            account=account,
            positions=positions,
            mode=mode,
        ),
        TRADING_SNAPSHOTS_FILE,
    )
    append_rows(
        build_positions_history(
            run_id=run_id,
            timestamp=timestamp,
            universe=universe,
            benchmark=benchmark,
            timeframe=timeframe,
            top_k=top_k,
            positions=positions,
        ),
        POSITIONS_HISTORY_FILE,
    )
    append_rows(
        build_orders_history(
            run_id=run_id,
            timestamp=timestamp,
            universe=universe,
            benchmark=benchmark,
            timeframe=timeframe,
            top_k=top_k,
            orders=orders,
        ),
        ORDERS_HISTORY_FILE,
    )
    append_rows(
        build_signal_history(
            run_id=run_id,
            timestamp=timestamp,
            universe=universe,
            benchmark=benchmark,
            timeframe=timeframe,
            top_k=top_k,
            signals=signals,
        ),
        SIGNAL_HISTORY_FILE,
    )
    append_rows(
        build_universe_position_registry(
            run_id=run_id,
            timestamp=timestamp,
            universe=universe,
            benchmark=benchmark,
            timeframe=timeframe,
            top_k=top_k,
            orders=orders,
            signals=signals,
        ),
        UNIVERSE_POSITION_REGISTRY_FILE,
    )
    append_rows(
        build_performance_history(
            run_id=run_id,
            timestamp=timestamp,
            universe=universe,
            benchmark=benchmark,
            timeframe=timeframe,
            top_k=top_k,
            account=account,
            benchmark_value=benchmark_value,
        ),
        PERFORMANCE_HISTORY_FILE,
    )

    update_plots()
    update_daily_summary(timestamp)
