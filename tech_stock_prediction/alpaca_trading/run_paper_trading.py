"""
CLI entry point for Alpaca Paper Trading.
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca_trading.paper_trading_engine import PaperTradingEngine
from alpaca_trading.signal_generator import SignalGenerationError
from config.stock_universes import list_available_universes


def parse_args():
    parser = argparse.ArgumentParser(description="Run Alpaca Paper Trading workflow.")

    universe_group = parser.add_mutually_exclusive_group(required=True)
    universe_group.add_argument("--universe", help="Universe name to run.")
    universe_group.add_argument(
        "--all-universes",
        action="store_true",
        help="Run all configured universes.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--signals-only",
        action="store_true",
        help="Generate signals only. No Alpaca keys or connection required.",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect to Alpaca if keys exist, but do not send orders.",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="Send Paper Trading orders to Alpaca.",
    )

    parser.add_argument("--top-k", type=int, default=1, help="Number of tickers to select.")

    return parser.parse_args()


def resolve_universes(args) -> list[str]:
    if args.all_universes:
        return list_available_universes()

    return [args.universe]


def main():
    args = parse_args()
    universes = resolve_universes(args)

    if args.execute:
        print("\nTHIS WILL SEND PAPER TRADING ORDERS TO ALPACA.")
        print("Mode: EXECUTE")
        print("Endpoint: https://paper-api.alpaca.markets")
        dry_run = False
        signals_only = False
    elif args.signals_only:
        dry_run = True
        signals_only = True
    else:
        dry_run = True
        signals_only = False

    for universe_name in universes:
        engine = PaperTradingEngine(
            universe_name=universe_name,
            top_k=args.top_k,
            dry_run=dry_run,
            signals_only=signals_only,
            universe_count=len(universes),
        )
        try:
            engine.run_daily_cycle()
        except (RuntimeError, ValueError, SignalGenerationError) as error:
            print("\nPaper Trading run stopped:")
            print(error)
            sys.exit(1)


if __name__ == "__main__":
    main()
