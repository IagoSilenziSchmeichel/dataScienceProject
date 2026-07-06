"""
Simple hourly scheduler for Alpaca Paper Trading.

The scheduler does not change the model or trading logic. It only decides
whether a Paper-Trading run should be started during US market hours.
"""

from __future__ import annotations

import argparse
from datetime import datetime, time
from pathlib import Path
import subprocess
import sys
import time as time_module
import uuid
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca_trading.trading_documentation import append_scheduler_log


MARKET_TIMEZONE = ZoneInfo("Europe/Berlin")
MARKET_OPEN = time(15, 30)
MARKET_CLOSE = time(22, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the hourly Alpaca Paper-Trading scheduler.")

    universe_group = parser.add_mutually_exclusive_group(required=True)
    universe_group.add_argument("--universe", help="Universe name to run.")
    universe_group.add_argument("--all-universes", action="store_true", help="Run all configured universes.")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true")
    mode_group.add_argument("--execute", action="store_true")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeframe", choices=["1Hour", "1Day"], default="1Hour")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running and check once per minute. Without this flag, one scheduler check is executed.",
    )

    return parser.parse_args()


def is_market_time(now: datetime | None = None) -> bool:
    """Return True during Monday-Friday, 15:30-22:00 German time."""
    current_time = now or datetime.now(MARKET_TIMEZONE)

    if current_time.weekday() >= 5:
        return False

    return MARKET_OPEN <= current_time.time() <= MARKET_CLOSE


def build_command(args) -> list[str]:
    command = [
        sys.executable,
        "alpaca_trading/run_paper_trading.py",
    ]

    if args.all_universes:
        command.append("--all-universes")
    else:
        command.extend(["--universe", args.universe])

    if args.execute:
        command.append("--execute")
    else:
        command.append("--dry-run")

    command.extend(["--top-k", str(args.top_k), "--timeframe", args.timeframe])
    return command


def run_one_scheduler_check(args, last_run_hour: str | None) -> str | None:
    """Run at most one Paper-Trading cycle for the current hour."""
    now = datetime.now(MARKET_TIMEZONE)
    run_id = f"scheduler_{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    mode = "execute" if args.execute else "dry_run"
    command = build_command(args)
    command_text = " ".join(command)
    current_hour = now.strftime("%Y-%m-%d %H")

    if args.execute:
        print("\nWARNING: Scheduler is in EXECUTE mode and will send Paper-Trading orders.")
        print("This is still Alpaca Paper Trading only, not live trading.")

    if not is_market_time(now):
        message = "Skipped: outside configured US market hours in German time."
        print(message)
        append_scheduler_log(
            run_id=run_id,
            mode=mode,
            command=command_text,
            status="skipped",
            return_code=None,
            message=message,
        )
        return last_run_hour

    if last_run_hour == current_hour:
        message = "Skipped: scheduler already ran during this hour."
        print(message)
        append_scheduler_log(
            run_id=run_id,
            mode=mode,
            command=command_text,
            status="skipped",
            return_code=None,
            message=message,
        )
        return last_run_hour

    print(f"Starting Paper-Trading run: {command_text}")
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        status = "success" if completed.returncode == 0 else "failed"
        message = "Paper-Trading command finished."
        append_scheduler_log(
            run_id=run_id,
            mode=mode,
            command=command_text,
            status=status,
            return_code=completed.returncode,
            message=message,
        )
        return current_hour if completed.returncode == 0 else last_run_hour
    except Exception as error:
        message = f"Scheduler error: {error}"
        print(message)
        append_scheduler_log(
            run_id=run_id,
            mode=mode,
            command=command_text,
            status="failed",
            return_code=None,
            message=message,
        )
        return last_run_hour


def main() -> None:
    args = parse_args()
    last_run_hour = None

    print("Hourly Alpaca scheduler started.")
    print(f"Timeframe: {args.timeframe}")
    print(f"Top-K: {args.top_k}")
    print("Market window: Monday-Friday, 15:30-22:00 German time")

    try:
        while True:
            last_run_hour = run_one_scheduler_check(args, last_run_hour)
            if not args.loop:
                break
            time_module.sleep(60)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")


if __name__ == "__main__":
    main()
