"""
Hourly scheduler for Alpaca Paper Trading.

Runs the paper-trading workflow once per hour during US market hours.
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
import argparse
import csv
import subprocess
import sys
import time as sleep_time
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_ROOT / "alpaca_trading" / "logs" / "scheduler_log.csv"

GERMAN_TZ = ZoneInfo("Europe/Berlin")

MARKET_OPEN = time(15, 30)
MARKET_CLOSE = time(22, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run hourly Alpaca scheduler.")

    universe_group = parser.add_mutually_exclusive_group(required=True)
    universe_group.add_argument("--universe")
    universe_group.add_argument("--all-universes", action="store_true")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true")
    mode_group.add_argument("--execute", action="store_true")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeframe", choices=["1Hour", "1Day"], default="1Hour")
    parser.add_argument("--poll-seconds", type=int, default=60)

    return parser.parse_args()


def is_market_open(now: datetime) -> bool:
    current_time = now.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE and now.weekday() < 5


def append_scheduler_log(
        *,
        timestamp: str,
        universe: str,
        command: str,
        status: str,
        return_code: int | str,
        message: str,
) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "universe",
                "command",
                "status",
                "return_code",
                "message",
            ],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": timestamp,
                "universe": universe,
                "command": command,
                "status": status,
                "return_code": return_code,
                "message": message,
            }
        )


def build_command(args) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "alpaca_trading" / "run_paper_trading.py"),
        "--top-k",
        str(args.top_k),
        "--timeframe",
        args.timeframe,
    ]

    if args.all_universes:
        command.append("--all-universes")
        universe_label = "all_universes"
    else:
        command.extend(["--universe", args.universe])
        universe_label = args.universe

    if args.execute:
        command.append("--execute")
    else:
        command.append("--dry-run")

    return command, universe_label


def main():
    args = parse_args()

    if args.execute:
        print("\nTHIS SCHEDULER WILL SEND PAPER TRADING ORDERS EVERY HOUR.")
        print("Mode: EXECUTE")
    else:
        print("\nScheduler running in DRY-RUN mode.")

    print("Stop with CTRL+C.")

    last_run_hour: str | None = None

    try:
        while True:
            now = datetime.now(GERMAN_TZ)
            timestamp = now.isoformat(timespec="seconds")

            if not is_market_open(now):
                print(f"{timestamp} | Market closed. Waiting for next run.")
                sleep_time.sleep(args.poll_seconds)
                continue

            current_hour_key = now.strftime("%Y-%m-%d-%H")

            if current_hour_key == last_run_hour:
                sleep_time.sleep(args.poll_seconds)
                continue

            command, universe_label = build_command(args)
            command_string = " ".join(command)

            print(f"{timestamp} | Running: {command_string}")

            try:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                )

                status = "success" if result.returncode == 0 else "error"
                message = result.stdout[-1000:] if result.stdout else result.stderr[-1000:]

                print(result.stdout)
                if result.stderr:
                    print(result.stderr)

                append_scheduler_log(
                    timestamp=timestamp,
                    universe=universe_label,
                    command=command_string,
                    status=status,
                    return_code=result.returncode,
                    message=message,
                )

            except Exception as error:
                append_scheduler_log(
                    timestamp=timestamp,
                    universe=universe_label,
                    command=command_string,
                    status="exception",
                    return_code="",
                    message=str(error),
                )

            last_run_hour = current_hour_key
            sleep_time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")


if __name__ == "__main__":
    main()