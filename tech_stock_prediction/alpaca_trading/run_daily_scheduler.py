"""
Daily scheduler for Alpaca Paper Trading.

The scheduler starts one Paper-Trading run after the US market close. It does
not change the model or trading logic; it only decides when to call
run_paper_trading.py.
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
RUN_AFTER = time(22, 10)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the daily Alpaca Paper-Trading scheduler.")

    universe_group = parser.add_mutually_exclusive_group(required=True)
    universe_group.add_argument("--universe", help="Universe name to run.")
    universe_group.add_argument("--all-universes", action="store_true", help="Run all configured universes.")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true")
    mode_group.add_argument("--execute", action="store_true")

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeframe", choices=["1Day"], default="1Day")
    parser.add_argument(
        "--run-after",
        default="22:10",
        help="Earliest local Europe/Berlin time to run, format HH:MM.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running and check once per minute. Without this flag, one scheduler check is executed.",
    )

    return parser.parse_args()


def parse_run_after(value: str) -> time:
    try:
        hour_text, minute_text = value.split(":", maxsplit=1)
        return time(int(hour_text), int(minute_text))
    except ValueError as error:
        raise ValueError("--run-after must use HH:MM format, for example 22:10.") from error


def is_daily_run_time(now: datetime | None = None, run_after: time = RUN_AFTER) -> bool:
    """Return True Monday-Friday after the configured local run time."""
    current_time = now or datetime.now(MARKET_TIMEZONE)

    if current_time.weekday() >= 5:
        return False

    return current_time.time() >= run_after


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


def run_one_scheduler_check(args, last_run_date: str | None) -> str | None:
    """Run at most one Paper-Trading cycle for the current local date."""
    now = datetime.now(MARKET_TIMEZONE)
    run_after = parse_run_after(args.run_after)
    run_id = f"daily_scheduler_{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    mode = "execute" if args.execute else "dry_run"
    command = build_command(args)
    command_text = " ".join(command)
    current_date = now.strftime("%Y-%m-%d")

    if args.execute:
        print("\nWARNING: Daily scheduler is in EXECUTE mode and will send Paper-Trading orders.")
        print("This is still Alpaca Paper Trading only, not live trading.")

    if not is_daily_run_time(now, run_after):
        message = f"Skipped: before configured daily run time {args.run_after} or weekend."
        print(message)
        append_scheduler_log(
            run_id=run_id,
            mode=mode,
            command=command_text,
            status="skipped",
            return_code=None,
            message=message,
        )
        return last_run_date

    if last_run_date == current_date:
        message = "Skipped: daily scheduler already ran today."
        print(message)
        append_scheduler_log(
            run_id=run_id,
            mode=mode,
            command=command_text,
            status="skipped",
            return_code=None,
            message=message,
        )
        return last_run_date

    print(f"Starting daily Paper-Trading run: {command_text}")
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        status = "success" if completed.returncode == 0 else "failed"
        message = "Daily Paper-Trading command finished."
        append_scheduler_log(
            run_id=run_id,
            mode=mode,
            command=command_text,
            status=status,
            return_code=completed.returncode,
            message=message,
        )
        return current_date if completed.returncode == 0 else last_run_date
    except Exception as error:
        message = f"Daily scheduler error: {error}"
        print(message)
        append_scheduler_log(
            run_id=run_id,
            mode=mode,
            command=command_text,
            status="failed",
            return_code=None,
            message=message,
        )
        return last_run_date


def main() -> None:
    args = parse_args()
    last_run_date = None

    print("Daily Alpaca scheduler started.")
    print(f"Timeframe: {args.timeframe}")
    print(f"Top-K: {args.top_k}")
    print(f"Run window: Monday-Friday after {args.run_after} Europe/Berlin time")

    try:
        while True:
            last_run_date = run_one_scheduler_check(args, last_run_date)
            if not args.loop:
                break
            time_module.sleep(60)
    except KeyboardInterrupt:
        print("\nDaily scheduler stopped by user.")


if __name__ == "__main__":
    main()
