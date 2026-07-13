"""
Central configuration for the presentation reporting system.

This module is the single place that defines:
  - which universes exist and their benchmark (re-exported from
    config/stock_universes.py, not duplicated)
  - where output goes
  - the shared plot style (colors, fonts, figure size, DPI) so that all
    plots for all four universes look identical and only need to change in
    one place

This module does not read or write any pipeline/trading files itself and
does not change any existing training or trading logic. It is purely a
styling/paths helper for the reporting scripts in this folder.
"""

from pathlib import Path
import sys

REPORTING_ROOT = Path(__file__).resolve().parent
ALPACA_ROOT = REPORTING_ROOT.parent
PROJECT_ROOT = ALPACA_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from config.stock_universes import BENCHMARKS, UNIVERSES  # noqa: E402

UNIVERSE_NAMES = list(UNIVERSES.keys())

OUTPUT_ROOT = REPORTING_ROOT / "output"
COMPARISON_OUTPUT_DIR = OUTPUT_ROOT / "comparison"

# Existing experiment / trading folders this reporting layer only reads from.
EXP_1_RANDOMFOREST_ROOT = PROJECT_ROOT / "experiments" / "exp_1_randomforest"
EXP_2_LSTM_ROOT = PROJECT_ROOT / "experiments" / "exp_2_lstm"
UNIVERSE_RESULTS_ROOT = PROJECT_ROOT / "universe_results"
ALPACA_LOGS_DIR = ALPACA_ROOT / "logs"
PER_UNIVERSE_DAILY_ROOT = ALPACA_ROOT / "per_universe_daily_mark_to_market"

# ---------------------------------------------------------------------------
# Shared plot style. Change values here to restyle every plot at once.
# ---------------------------------------------------------------------------

FIGURE_SIZE_WIDE = (12.8, 7.2)  # 16:9, matches a PowerPoint slide
FIGURE_SIZE_SQUARE = (9.6, 7.2)
DPI = 220  # spec requires >= 200 DPI

FONT_FAMILY = "DejaVu Sans"
TITLE_FONT_SIZE = 17
SUBTITLE_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 10

COLOR_STRATEGY = "#2F6B9A"      # model / strategy: always this blue
COLOR_BENCHMARK = "#D08B2C"     # benchmark index: always this orange, dashed
COLOR_ALWAYS_BUY = "#8A5A83"    # always-buy baseline, where shown
COLOR_POSITIVE = "#2F7D5F"      # positive outperformance / bars
COLOR_NEGATIVE = "#C9504D"      # negative outperformance / bars
COLOR_NEUTRAL = "#6B7280"       # missing/preliminary status graphics
COLOR_GRID = "#B0B0B0"

STATUS_READY = "READY"
STATUS_PRELIMINARY = "PRELIMINARY"
STATUS_MISSING = "MISSING"
STATUS_INVALID = "INVALID"
# Used only for the supplementary simulated paper-trading extension (Plot 5).
# Never applied to real Alpaca data - marks a series as "computed from real
# prices/predictions via the backtest methodology, not an actual broker
# result" so it can never be confused with READY/PRELIMINARY real data.
STATUS_SIMULATION = "SIMULATION"

# Below this many observations, a time series is only ever shown as
# "preliminary", never presented as a settled/robust result.
MIN_OBSERVATIONS_FOR_ROBUST = 30

# Below this many observations, we do not draw a line/bar chart at all and
# instead render a plain status graphic explaining what is missing.
MIN_OBSERVATIONS_FOR_ANY_PLOT = 2

TRADING_DAYS_PER_YEAR = 252


def universe_output_dir(universe_name):
    return OUTPUT_ROOT / universe_name


def ensure_output_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for universe_name in UNIVERSE_NAMES:
        universe_output_dir(universe_name).mkdir(parents=True, exist_ok=True)


UNIVERSE_TITLE_DE = {
    "original_tech": "Original Tech",
    "tech_no_nvda": "Tech ohne Nvidia",
    "new_tech": "Neue Tech-Aktien",
    "defensive_non_tech": "Defensive Non-Tech",
}

RESEARCH_QUESTION_DE = {
    "original_tech": "Wie funktioniert das Modell auf dem urspruenglichen Tech-Universum?",
    "tech_no_nvda": "Wie stark haengen die Ergebnisse von Nvidia ab?",
    "new_tech": "Generalisiert das Modell auf weitere bzw. kleinere Tech-Unternehmen ausserhalb des urspruenglichen Universums?",
    "defensive_non_tech": "Ist die Strategie Tech-spezifisch oder funktioniert sie auch bei defensiven Nicht-Tech-Aktien?",
}
