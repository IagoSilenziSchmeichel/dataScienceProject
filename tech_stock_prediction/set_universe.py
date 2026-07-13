"""
Set the active stock universe for the shared research pipeline.

Usage:
    python set_universe.py original_tech
    python set_universe.py tech_no_nvda
    python set_universe.py new_tech
    python set_universe.py defensive_non_tech

This writes the correct TICKERS list into
experiments/exp_1_randomforest/conf/params.yaml (data acquisition / feature /
split steps are shared with the LSTM pipeline) and the correct UNIVERSE name
into both experiments/exp_1_randomforest/conf/params.yaml and
experiments/exp_2_lstm/conf/params.yaml, using config/stock_universes.py as
the single source of truth for tickers and benchmark.

Run this once before re-running the pipeline for a different universe. See
run_universe_lstm_outperformance.py for the full per-universe pipeline runner.
"""

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.stock_universes import get_universe, list_available_universes


RF_PARAMS_FILE = PROJECT_ROOT / "experiments" / "exp_1_randomforest" / "conf" / "params.yaml"
LSTM_PARAMS_FILE = PROJECT_ROOT / "experiments" / "exp_2_lstm" / "conf" / "params.yaml"


def format_ticker_list(tickers):
    return "[" + ", ".join(f"'{ticker}'" for ticker in tickers) + "]"


def replace_universe_line(text, universe_name, file_path):
    updated_text, count = re.subn(
        r'^UNIVERSE:\s*".*"\s*$',
        f'UNIVERSE: "{universe_name}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise ValueError(f"Could not find a 'UNIVERSE: \"...\"' line in {file_path}")
    return updated_text


def set_universe_in_rf_params(universe_name, tickers):
    text = RF_PARAMS_FILE.read_text(encoding="utf-8")
    text = replace_universe_line(text, universe_name, RF_PARAMS_FILE)

    text, tickers_count = re.subn(
        r"^(\s*TICKERS:\s*)\[.*\]\s*$",
        lambda match: f"{match.group(1)}{format_ticker_list(tickers)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if tickers_count == 0:
        raise ValueError(f"Could not find a 'TICKERS: [...]' line in {RF_PARAMS_FILE}")

    RF_PARAMS_FILE.write_text(text, encoding="utf-8")


def set_universe_in_lstm_params(universe_name):
    text = LSTM_PARAMS_FILE.read_text(encoding="utf-8")
    text = replace_universe_line(text, universe_name, LSTM_PARAMS_FILE)
    LSTM_PARAMS_FILE.write_text(text, encoding="utf-8")


def main():
    if len(sys.argv) != 2:
        available = ", ".join(list_available_universes())
        print("Usage: python set_universe.py <universe_name>")
        print(f"Available universes: {available}")
        sys.exit(1)

    universe_name = sys.argv[1].strip().lower()

    try:
        tickers = get_universe(universe_name)
    except ValueError as error:
        print(error)
        sys.exit(1)

    set_universe_in_rf_params(universe_name, tickers)
    set_universe_in_lstm_params(universe_name)

    print(f"Universe set to: {universe_name}")
    print(f"Tickers: {tickers}")
    print(f"Updated: {RF_PARAMS_FILE}")
    print(f"Updated: {LSTM_PARAMS_FILE}")


if __name__ == "__main__":
    main()
