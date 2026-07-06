"""
CLI entry point for Alpaca Paper Trading.
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca_trading.paper_trading_engine import PaperTradingEngine
from alpaca_trading.signal_generator import SignalGenerationError
from alpaca_trading.trading_documentation import update_multi_universe_summary
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
    mode_group.add_argument("--signals-only", action="store_true")
    mode_group.add_argument("--dry-run", action="store_true")
    mode_group.add_argument("--execute", action="store_true")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeframe", choices=["1Hour", "1Day"], default="1Hour")
    parser.add_argument(
        "--allow-existing-positions",
        action="store_true",
        help="Only relevant for single-universe execute mode.",
    )

    args = parser.parse_args()

    if args.allow_existing_positions and not args.execute:
        parser.error("--allow-existing-positions is only useful together with --execute.")

    return args


def resolve_universes(args) -> list[str]:
    if args.all_universes:
        return list_available_universes()
    return [args.universe]


def main():
    args = parse_args()
    universes = resolve_universes(args)
    run_id = f"paper_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    results = []
    errors = []

    if args.execute:
        print("\nTHIS WILL SEND PAPER TRADING ORDERS TO ALPACA.")
        print("Mode: EXECUTE")
        print("Endpoint: https://paper-api.alpaca.markets")
        if args.all_universes:
            print("Universes: all configured universes")
        else:
            print(f"Universe: {args.universe}")

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
            timeframe=args.timeframe,
            dry_run=dry_run,
            signals_only=signals_only,
            universe_count=len(universes),
            allow_existing_positions=args.allow_existing_positions,
            run_id=run_id,
        )

        try:
            results.append(engine.run_daily_cycle())
        except (RuntimeError, ValueError, SignalGenerationError) as error:
            print("\nPaper Trading run stopped:")
            print(error)
            errors.append({"universe": universe_name, "message": str(error)})
            if not args.all_universes:
                sys.exit(1)

    if args.all_universes:
        if args.execute:
            mode = "execute"
        elif args.signals_only:
            mode = "signals_only"
        else:
            mode = "dry_run"

        update_multi_universe_summary(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode=mode,
            timeframe=args.timeframe,
            top_k=args.top_k,
            results=results,
            errors=errors,
        )

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
