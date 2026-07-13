"""
Data loading + validation for the presentation reporting system.

This module only reads existing files. It never trains a model, never
writes to any trading log, and never touches the Random Forest / LSTM /
Alpaca pipelines. Every loader function returns a DataSource object that
records where the data came from, how many observations it has, and a
READY / PRELIMINARY / MISSING / INVALID status - so the plotting layer
never has to guess whether a result is trustworthy.

Design rule (per project instructions): if a number cannot be honestly
computed from an existing CSV, we do not invent it. Missing data is always
reported as missing, never silently turned into a 0 or an empty-but-present
line.
"""

from pathlib import Path

import pandas as pd

from reporting_config import (
    ALPACA_LOGS_DIR,
    BENCHMARKS,
    EXP_2_LSTM_ROOT,
    MIN_OBSERVATIONS_FOR_ANY_PLOT,
    MIN_OBSERVATIONS_FOR_ROBUST,
    PER_UNIVERSE_DAILY_ROOT,
    STATUS_INVALID,
    STATUS_MISSING,
    STATUS_PRELIMINARY,
    STATUS_READY,
    UNIVERSE_RESULTS_ROOT,
    UNIVERSES,
)


class DataSource:
    """Container describing one loaded (or missing/invalid) data source."""

    def __init__(self, name, status, data=None, source_path=None, note="", observations=0):
        self.name = name
        self.status = status
        self.data = data
        self.source_path = str(source_path) if source_path else None
        self.note = note
        self.observations = observations

    @property
    def is_usable(self):
        return self.status in (STATUS_READY, STATUS_PRELIMINARY) and self.data is not None

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "source_path": self.source_path,
            "observations": self.observations,
            "note": self.note,
        }


def infer_universe_from_tickers(tickers):
    ticker_set = set(tickers)
    for universe_name, universe_tickers in UNIVERSES.items():
        if set(universe_tickers) == ticker_set:
            return universe_name
    return None


_READ_ERRORS = []


def get_read_errors():
    """Files that were found but could not be parsed as CSV, for the validation report."""
    return list(_READ_ERRORS)


def _read_csv_safe(path, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as error:
        message = f"{path}: {error}"
        print(f"Warning: could not read {message}")
        _READ_ERRORS.append(message)
        return None


# ---------------------------------------------------------------------------
# 1. Daily backtest (Outperformance-LSTM + Top-K), from experiments/exp_2_lstm
# ---------------------------------------------------------------------------

def load_backtest_predictions(universe_name):
    """
    Locate the daily Outperformance-LSTM test predictions for one universe.

    Preferred source: an archived run under universe_results/<universe>/...
    (written by run_universe_lstm_outperformance.py). Falls back to the
    live experiments/exp_2_lstm/data/processed/ file, but only if its
    ticker set actually matches the requested universe - that file gets
    overwritten by whichever universe was run last in the shared pipeline,
    so it can not be trusted without checking its contents first.
    """
    archived_path = (
        UNIVERSE_RESULTS_ROOT
        / universe_name
        / "exp_2_lstm"
        / "data"
        / "processed"
        / "lstm_outperformance_predictions.csv"
    )
    live_path = EXP_2_LSTM_ROOT / "data" / "processed" / "lstm_outperformance_predictions.csv"

    mismatch_notes = []

    for path, is_archived in [(archived_path, True), (live_path, False)]:
        if not path.exists():
            continue

        data = _read_csv_safe(path, parse_dates=["Date"])
        if data is None or data.empty:
            mismatch_notes.append(f"{path.name} ist leer oder nicht lesbar.")
            continue

        required = {"Date", "Ticker", "Probability", "Future_Return"}
        if not required.issubset(data.columns):
            mismatch_notes.append(f"{path.name}: Pflichtspalten fehlen ({sorted(required - set(data.columns))}).")
            continue

        tickers_in_file = set(data["Ticker"].unique())
        expected_tickers = set(UNIVERSES[universe_name])
        if tickers_in_file != expected_tickers:
            matched_universe = infer_universe_from_tickers(tickers_in_file) or "unbekanntes Universum"
            mismatch_notes.append(
                f"{path.name} enthaelt Ticker von '{matched_universe}', nicht von '{universe_name}' - uebersprungen."
            )
            continue

        if data["Date"].isna().any():
            mismatch_notes.append(f"{path.name}: ungueltige Datumswerte gefunden.")
            continue

        observations = int(data["Date"].nunique())
        status = STATUS_READY if observations >= MIN_OBSERVATIONS_FOR_ROBUST else STATUS_PRELIMINARY
        note = (
            "Archivierter Lauf aus universe_results/."
            if is_archived
            else "Live-Ergebnis aus experiments/exp_2_lstm/data/processed (aktuell aktives Universum in der Pipeline)."
        )
        return DataSource(
            "daily_backtest",
            status,
            data=data,
            source_path=path,
            note=note,
            observations=observations,
        )

    note = f"Kein Outperformance-LSTM-Backtest fuer Universum '{universe_name}' gefunden."
    if mismatch_notes:
        note += " Details: " + " | ".join(mismatch_notes)
    note += (
        " Reproduzierbar mit: python run_universe_lstm_outperformance.py --universe "
        f"{universe_name}"
    )
    return DataSource("daily_backtest", STATUS_MISSING, note=note)


# ---------------------------------------------------------------------------
# 2. Daily Alpaca paper trading, from the isolated per-universe simulation
# ---------------------------------------------------------------------------

def load_daily_alpaca_performance(universe_name):
    path = PER_UNIVERSE_DAILY_ROOT / universe_name / "logs" / "daily_mark_to_market_with_baselines.csv"
    if not path.exists():
        return DataSource("daily_alpaca", STATUS_MISSING, note=f"Datei nicht gefunden: {path}")

    data = _read_csv_safe(path, parse_dates=["date"])
    if data is None or data.empty:
        return DataSource("daily_alpaca", STATUS_MISSING, note=f"Datei leer oder nicht lesbar: {path}")

    required = {"date", "universe", "benchmark", "portfolio_index", "benchmark_index"}
    missing_columns = required - set(data.columns)
    if missing_columns:
        return DataSource(
            "daily_alpaca", STATUS_INVALID, source_path=path,
            note=f"Erforderliche Spalten fehlen in {path.name}: {sorted(missing_columns)}",
        )

    data = data.sort_values("date").reset_index(drop=True)

    duplicate_dates = int(data["date"].duplicated().sum())
    if duplicate_dates:
        return DataSource(
            "daily_alpaca", STATUS_INVALID, data=data, source_path=path,
            note=f"{duplicate_dates} doppelte Datumswerte in {path.name}.",
        )

    foreign_universes = set(data["universe"].unique()) - {universe_name}
    if foreign_universes:
        return DataSource(
            "daily_alpaca", STATUS_INVALID, data=data, source_path=path,
            note=f"Datei enthaelt fremde Universen: {foreign_universes}",
        )

    expected_benchmark = BENCHMARKS[universe_name]
    wrong_benchmark = set(data["benchmark"].unique()) - {expected_benchmark}
    if wrong_benchmark:
        return DataSource(
            "daily_alpaca", STATUS_INVALID, data=data, source_path=path,
            note=f"Unerwarteter Benchmark {wrong_benchmark} in Datei, erwartet {expected_benchmark}.",
        )

    observations = len(data)
    if observations < MIN_OBSERVATIONS_FOR_ANY_PLOT:
        return DataSource(
            "daily_alpaca", STATUS_MISSING, data=data, source_path=path,
            observations=observations, note="Zu wenige Beobachtungen fuer einen Plot (mindestens 2 Tage noetig).",
        )

    status = STATUS_READY if observations >= MIN_OBSERVATIONS_FOR_ROBUST else STATUS_PRELIMINARY
    note = f"{observations} Handelstage ({data['date'].min().date()} bis {data['date'].max().date()})."
    if status == STATUS_PRELIMINARY:
        note += " Vorlaeufig: deutlich weniger als 30 Beobachtungen."
    return DataSource("daily_alpaca", status, data=data, source_path=path, observations=observations, note=note)


def load_daily_alpaca_orders(universe_name):
    path = PER_UNIVERSE_DAILY_ROOT / universe_name / "logs" / "daily_mark_to_market_orders.csv"
    if not path.exists():
        return DataSource("daily_alpaca_orders", STATUS_MISSING, note=f"Datei nicht gefunden: {path}")

    data = _read_csv_safe(path, parse_dates=["date"])
    if data is None or data.empty:
        return DataSource("daily_alpaca_orders", STATUS_MISSING, note=f"Datei leer oder nicht lesbar: {path}")

    if "action" not in data.columns:
        return DataSource("daily_alpaca_orders", STATUS_INVALID, source_path=path, note="Spalte 'action' fehlt.")

    foreign_universes = set(data.get("universe", pd.Series(dtype=str)).unique()) - {universe_name}
    if foreign_universes:
        return DataSource(
            "daily_alpaca_orders", STATUS_INVALID, data=data, source_path=path,
            note=f"Datei enthaelt fremde Universen: {foreign_universes}",
        )

    return DataSource(
        "daily_alpaca_orders", STATUS_READY if len(data) else STATUS_MISSING,
        data=data, source_path=path, observations=len(data),
    )


# ---------------------------------------------------------------------------
# 3. Hourly Alpaca paper trading - scanned from alpaca_trading/logs, including
#    legacy files, filtered strictly to timeframe == "1Hour"
# ---------------------------------------------------------------------------

def _scan_hourly_rows(universe_name):
    matched_frames = []
    scanned_files = []

    if not ALPACA_LOGS_DIR.exists():
        return None, []

    for path in sorted(ALPACA_LOGS_DIR.glob("*.csv")):
        data = _read_csv_safe(path)
        if data is None or data.empty:
            continue
        if "timeframe" not in data.columns or "universe" not in data.columns:
            continue

        subset = data[(data["timeframe"] == "1Hour") & (data["universe"] == universe_name)]
        if subset.empty:
            continue

        subset = subset.copy()
        subset["__source_file"] = path.name
        matched_frames.append(subset)
        scanned_files.append(path.name)

    if not matched_frames:
        return None, []

    combined = pd.concat(matched_frames, ignore_index=True, sort=False)
    return combined, scanned_files


def load_hourly_alpaca_performance(universe_name):
    combined, source_files = _scan_hourly_rows(universe_name)
    if combined is None:
        return DataSource(
            "hourly_alpaca", STATUS_MISSING,
            note=(
                "Keine 1Hour-Zeilen fuer dieses Universum in alpaca_trading/logs/ "
                "(inklusive Legacy-Dateien) gefunden."
            ),
        )

    timestamp_column = None
    for candidate in ["timestamp", "bar_timestamp"]:
        if candidate in combined.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        return DataSource(
            "hourly_alpaca", STATUS_INVALID, data=combined,
            note=f"1Hour-Zeilen gefunden ({', '.join(sorted(set(source_files)))}), aber keine Zeitstempelspalte erkannt.",
        )

    distinct_timestamps = int(combined[timestamp_column].nunique())
    has_portfolio_columns = {"portfolio_value", "benchmark_value"}.issubset(combined.columns) or {
        "portfolio_return",
        "benchmark_return",
    }.issubset(combined.columns)

    note = (
        f"{distinct_timestamps} eindeutige(r) 1Hour-Zeitstempel gefunden in: "
        f"{', '.join(sorted(set(source_files)))}. Diese Zeilen stammen aus dem einmaligen "
        "Alpaca-Dry-Run vor der Umstellung auf den Daily-Scheduler; die gleichen Logdateien "
        "wurden danach mit 1Day-Zeilen weitergeschrieben (Schema-Wechsel, alte Zeilen als "
        "Legacy-Datei archiviert)."
    )

    if distinct_timestamps < MIN_OBSERVATIONS_FOR_ANY_PLOT or not has_portfolio_columns:
        return DataSource(
            "hourly_alpaca", STATUS_MISSING, data=combined, observations=distinct_timestamps,
            note=note + " Keine belastbare universenspezifische Performance-Attribution ueber die Zeit verfuegbar.",
        )

    status = STATUS_READY if distinct_timestamps >= MIN_OBSERVATIONS_FOR_ROBUST else STATUS_PRELIMINARY
    return DataSource("hourly_alpaca", status, data=combined, observations=distinct_timestamps, note=note)


# ---------------------------------------------------------------------------
# 4. Signal / selection history (Alpaca), from alpaca_trading/logs
# ---------------------------------------------------------------------------

def load_signal_history(universe_name, timeframe="1Day"):
    path = ALPACA_LOGS_DIR / "paper_signals.csv"
    if not path.exists():
        return DataSource("signal_history", STATUS_MISSING, note=f"Datei nicht gefunden: {path}")

    data = _read_csv_safe(path, parse_dates=["date"])
    if data is None or data.empty:
        return DataSource("signal_history", STATUS_MISSING, note=f"Datei leer oder nicht lesbar: {path}")

    required = {"date", "universe", "benchmark", "ticker", "probability", "rank", "selected", "top_k", "timeframe"}
    missing_columns = required - set(data.columns)
    if missing_columns:
        return DataSource(
            "signal_history", STATUS_INVALID, source_path=path,
            note=f"Erforderliche Spalten fehlen: {sorted(missing_columns)}",
        )

    subset = data[(data["universe"] == universe_name) & (data["timeframe"] == timeframe)].copy()
    if subset.empty:
        return DataSource(
            "signal_history", STATUS_MISSING, source_path=path,
            note=f"Keine {timeframe}-Signale fuer Universum '{universe_name}' gefunden.",
        )

    expected_benchmark = BENCHMARKS[universe_name]
    wrong_benchmark = set(subset["benchmark"].unique()) - {expected_benchmark}
    if wrong_benchmark:
        return DataSource(
            "signal_history", STATUS_INVALID, data=subset, source_path=path,
            note=f"Unerwarteter Benchmark {wrong_benchmark} in Signaldatei, erwartet {expected_benchmark}.",
        )

    expected_tickers = set(UNIVERSES[universe_name])
    foreign_tickers = set(subset["ticker"].unique()) - expected_tickers
    if foreign_tickers:
        return DataSource(
            "signal_history", STATUS_INVALID, data=subset, source_path=path,
            note=f"Signaldatei enthaelt Ticker ausserhalb des Universums: {foreign_tickers}",
        )

    distinct_signal_dates = int(subset["date"].nunique())
    status = STATUS_MISSING
    if distinct_signal_dates >= MIN_OBSERVATIONS_FOR_ANY_PLOT:
        status = STATUS_READY if distinct_signal_dates >= MIN_OBSERVATIONS_FOR_ROBUST else STATUS_PRELIMINARY

    note = (
        f"{distinct_signal_dates} {timeframe}-Signalzeitpunkt(e) "
        f"({subset['date'].min().date()} bis {subset['date'].max().date()})."
    )
    if status == STATUS_PRELIMINARY:
        note += " Vorlaeufig: deutlich weniger als 30 Beobachtungen."
    elif status == STATUS_MISSING:
        note += " Zu wenige Beobachtungen fuer eine Auswertung."

    return DataSource(
        "signal_history", status, data=subset, source_path=path,
        observations=distinct_signal_dates, note=note,
    )
