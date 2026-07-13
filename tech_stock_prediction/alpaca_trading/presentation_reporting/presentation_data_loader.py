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
    PAPER_TRADING_END,
    PAPER_TRADING_START,
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


def _paper_period_bounds():
    return pd.Timestamp(PAPER_TRADING_START), pd.Timestamp(PAPER_TRADING_END)


def _filter_to_paper_period(data, timestamp_column):
    start, end = _paper_period_bounds()
    filtered = data.copy()
    filtered[timestamp_column] = pd.to_datetime(filtered[timestamp_column], errors="coerce").dt.tz_localize(None)
    filtered = filtered.dropna(subset=[timestamp_column])
    return filtered[(filtered[timestamp_column] >= start) & (filtered[timestamp_column] <= end)].copy()


def _filter_daily_period(data, date_column="date"):
    return _filter_to_paper_period(data, date_column)


def _filter_hourly_period(data):
    # For hourly Paper Trading, bar_timestamp is the actual market bar.
    # generated_at/timestamp can be later and must not make older bars look
    # like Paper-Trading-period observations.
    for column in ["bar_timestamp", "timestamp", "date", "generated_at"]:
        if column in data.columns:
            return _filter_to_paper_period(data, column), column
    return data.iloc[0:0].copy(), None


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

    data = _filter_daily_period(data, "date").sort_values("date").reset_index(drop=True)
    if data.empty:
        return DataSource(
            "daily_alpaca",
            STATUS_MISSING,
            data=data,
            source_path=path,
            note=f"Keine Daily-Paper-Trading-Zeilen im Zeitraum {PAPER_TRADING_START} bis {PAPER_TRADING_END}.",
        )

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

    data = _filter_daily_period(data, "date")
    return DataSource(
        "daily_alpaca_orders", STATUS_READY if len(data) else STATUS_MISSING,
        data=data, source_path=path, observations=len(data),
    )


def load_daily_alpaca_positions(universe_name):
    path = PER_UNIVERSE_DAILY_ROOT / universe_name / "logs" / "daily_mark_to_market_positions.csv"
    if not path.exists():
        return DataSource("daily_alpaca_positions", STATUS_MISSING, note=f"Datei nicht gefunden: {path}")

    data = _read_csv_safe(path, parse_dates=["date"])
    if data is None or data.empty:
        return DataSource("daily_alpaca_positions", STATUS_MISSING, note=f"Datei leer oder nicht lesbar: {path}")

    required = {"date", "universe", "symbol", "price", "market_value"}
    missing_columns = required - set(data.columns)
    if missing_columns:
        return DataSource(
            "daily_alpaca_positions",
            STATUS_INVALID,
            source_path=path,
            note=f"Erforderliche Spalten fehlen in {path.name}: {sorted(missing_columns)}",
        )

    foreign_universes = set(data["universe"].unique()) - {universe_name}
    if foreign_universes:
        return DataSource(
            "daily_alpaca_positions",
            STATUS_INVALID,
            data=data,
            source_path=path,
            note=f"Datei enthaelt fremde Universen: {foreign_universes}",
        )

    expected_tickers = set(UNIVERSES[universe_name])
    foreign_tickers = set(data["symbol"].unique()) - expected_tickers
    if foreign_tickers:
        return DataSource(
            "daily_alpaca_positions",
            STATUS_INVALID,
            data=data,
            source_path=path,
            note=f"Positionsdatei enthaelt Ticker ausserhalb des Universums: {foreign_tickers}",
        )

    data = _filter_daily_period(data, "date")
    return DataSource(
        "daily_alpaca_positions",
        STATUS_READY if len(data) else STATUS_MISSING,
        data=data.sort_values(["date", "symbol"]).reset_index(drop=True),
        source_path=path,
        observations=len(data),
    )


# ---------------------------------------------------------------------------
# 3. Hourly Alpaca paper trading - scanned from alpaca_trading/logs, including
#    legacy files, filtered strictly to timeframe == "1Hour" and to the real
#    Paper-Trading presentation period.
# ---------------------------------------------------------------------------

def _scan_hourly_rows(universe_name, file_patterns):
    matched_frames = []
    scanned_files = []

    if not ALPACA_LOGS_DIR.exists():
        return None, []

    paths = []
    for pattern in file_patterns:
        paths.extend(sorted(ALPACA_LOGS_DIR.glob(pattern)))

    for path in sorted(set(paths)):
        data = _read_csv_safe(path)
        if data is None or data.empty:
            continue
        if "timeframe" not in data.columns or "universe" not in data.columns:
            continue

        subset = data[(data["timeframe"] == "1Hour") & (data["universe"] == universe_name)]
        if subset.empty:
            continue

        subset, period_column = _filter_hourly_period(subset)
        if subset.empty:
            continue

        subset = subset.copy()
        subset["__source_file"] = path.name
        subset["__period_column"] = period_column
        matched_frames.append(subset)
        scanned_files.append(path.name)

    if not matched_frames:
        return None, []

    combined = pd.concat(matched_frames, ignore_index=True, sort=False)
    return combined, scanned_files


def load_hourly_alpaca_performance(universe_name):
    combined, source_files = _scan_hourly_rows(
        universe_name,
        ["paper_performance*.csv", "performance_history*.csv"],
    )
    if combined is None:
        return DataSource(
            "hourly_alpaca", STATUS_MISSING,
            note=(
                f"Keine 1Hour-Paper-Trading-Performance fuer dieses Universum im Zeitraum "
                f"{PAPER_TRADING_START} bis {PAPER_TRADING_END} gefunden."
            ),
        )

    timestamp_column = None
    for candidate in ["bar_timestamp", "timestamp", "date"]:
        if candidate in combined.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        return DataSource(
            "hourly_alpaca", STATUS_INVALID, data=combined,
            note=f"1Hour-Zeilen gefunden ({', '.join(sorted(set(source_files)))}), aber keine Zeitstempelspalte erkannt.",
        )

    combined[timestamp_column] = pd.to_datetime(combined[timestamp_column], errors="coerce").dt.tz_localize(None)
    combined = combined.dropna(subset=[timestamp_column]).sort_values(timestamp_column)

    if "generated_at" in combined.columns:
        combined["__generated_at"] = pd.to_datetime(combined["generated_at"], errors="coerce").dt.tz_localize(None)
        combined = combined.sort_values([timestamp_column, "__generated_at"])
    elif "timestamp" in combined.columns:
        combined["__generated_at"] = pd.to_datetime(combined["timestamp"], errors="coerce").dt.tz_localize(None)
        combined = combined.sort_values([timestamp_column, "__generated_at"])

    combined = combined.drop_duplicates(subset=[timestamp_column], keep="last").reset_index(drop=True)
    for column in ["portfolio_value", "benchmark_value"]:
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")

    distinct_timestamps = int(combined[timestamp_column].nunique())
    has_portfolio_columns = {"portfolio_value", "benchmark_value"}.issubset(combined.columns)
    has_enough_portfolio_values = False
    if has_portfolio_columns:
        has_enough_portfolio_values = combined[["portfolio_value", "benchmark_value"]].dropna().shape[0] >= 2

    note = (
        f"{distinct_timestamps} eindeutige(r) 1Hour-Zeitstempel im Paper-Trading-Zeitraum gefunden in: "
        f"{', '.join(sorted(set(source_files)))}."
    )

    if distinct_timestamps < MIN_OBSERVATIONS_FOR_ANY_PLOT or not has_enough_portfolio_values:
        return DataSource(
            "hourly_alpaca", STATUS_MISSING, data=combined, observations=distinct_timestamps,
            source_path=", ".join(sorted(set(source_files))),
            note=note + " Keine belastbare universenspezifische Hourly-Performance-Zeitreihe mit mindestens zwei Portfoliowerten.",
        )

    status = STATUS_READY if distinct_timestamps >= MIN_OBSERVATIONS_FOR_ROBUST else STATUS_PRELIMINARY
    return DataSource(
        "hourly_alpaca",
        status,
        data=combined,
        source_path=", ".join(sorted(set(source_files))),
        observations=distinct_timestamps,
        note=note,
    )


def load_hourly_alpaca_orders(universe_name):
    combined, source_files = _scan_hourly_rows(
        universe_name,
        ["paper_orders*.csv", "orders_history*.csv"],
    )
    if combined is None:
        return DataSource("hourly_alpaca_orders", STATUS_MISSING, note="Keine 1Hour-Orders im Paper-Trading-Zeitraum.")

    if "action" not in combined.columns:
        return DataSource("hourly_alpaca_orders", STATUS_INVALID, data=combined, note="Spalte 'action' fehlt.")

    if "dry_run" in combined.columns:
        dry_run = combined["dry_run"].astype(str).str.lower().isin(["true", "1", "yes"])
        combined = combined[~dry_run].copy()

    return DataSource(
        "hourly_alpaca_orders",
        STATUS_READY if len(combined) else STATUS_MISSING,
        data=combined.reset_index(drop=True),
        source_path=", ".join(sorted(set(source_files))),
        observations=len(combined),
        note=f"{len(combined)} ausgefuehrte 1Hour-Orderzeilen im Paper-Trading-Zeitraum.",
    )


def load_hourly_alpaca_signals(universe_name):
    combined, source_files = _scan_hourly_rows(
        universe_name,
        ["paper_signals*.csv"],
    )
    if combined is None:
        return DataSource("hourly_alpaca_signals", STATUS_MISSING, note="Keine 1Hour-Signale im Paper-Trading-Zeitraum.")

    time_column = "bar_timestamp" if "bar_timestamp" in combined.columns else "timestamp"
    combined[time_column] = pd.to_datetime(combined[time_column], errors="coerce").dt.tz_localize(None)
    combined = combined.dropna(subset=[time_column]).sort_values(time_column)
    if "generated_at" in combined.columns:
        combined["__generated_at"] = pd.to_datetime(combined["generated_at"], errors="coerce").dt.tz_localize(None)
        combined = combined.sort_values([time_column, "ticker", "__generated_at"])
    if "ticker" in combined.columns:
        combined = combined.drop_duplicates(subset=[time_column, "ticker"], keep="last")

    observations = int(combined[time_column].nunique())
    return DataSource(
        "hourly_alpaca_signals",
        STATUS_READY if observations >= MIN_OBSERVATIONS_FOR_ANY_PLOT else STATUS_PRELIMINARY,
        data=combined.reset_index(drop=True),
        source_path=", ".join(sorted(set(source_files))),
        observations=observations,
        note=f"{observations} 1Hour-Signalzeitpunkte im Paper-Trading-Zeitraum.",
    )


# ---------------------------------------------------------------------------
# 4. Hourly model backtest, from experiments/exp_2_lstm/hourly/data
# ---------------------------------------------------------------------------

def load_hourly_model_predictions(universe_name):
    """
    Load the existing hourly Outperformance-LSTM prediction file and filter
    it to one presentation universe.

    This is a reporting-only fallback when real Hourly Paper-Trading
    performance history is incomplete. It uses hourly bars only; Daily rows
    are never mixed into this dataset.
    """
    path = EXP_2_LSTM_ROOT / "hourly" / "data" / "hourly_outperformance_predictions.csv"
    if not path.exists():
        return DataSource("hourly_model_predictions", STATUS_MISSING, note=f"Datei nicht gefunden: {path}")

    data = _read_csv_safe(path, parse_dates=["Date"])
    if data is None or data.empty:
        return DataSource("hourly_model_predictions", STATUS_MISSING, note=f"Datei leer oder nicht lesbar: {path}")

    required = {
        "Date",
        "Ticker",
        "Benchmark",
        "Probability",
        "Tradable_Return",
        "Benchmark_Tradable_Return",
    }
    missing_columns = required - set(data.columns)
    if missing_columns:
        return DataSource(
            "hourly_model_predictions",
            STATUS_INVALID,
            source_path=path,
            note=f"Erforderliche Spalten fehlen in {path.name}: {sorted(missing_columns)}",
        )

    expected_tickers = set(UNIVERSES[universe_name])
    expected_benchmark = BENCHMARKS[universe_name]
    subset = data[
        data["Ticker"].isin(expected_tickers)
        & data["Benchmark"].astype(str).eq(expected_benchmark)
    ].copy()

    if subset.empty:
        return DataSource(
            "hourly_model_predictions",
            STATUS_MISSING,
            source_path=path,
            note=f"Keine Hourly-Modellzeilen fuer Universum '{universe_name}' und Benchmark {expected_benchmark}.",
        )

    subset["Date"] = pd.to_datetime(subset["Date"], errors="coerce")
    subset["Probability"] = pd.to_numeric(subset["Probability"], errors="coerce")
    subset["Tradable_Return"] = pd.to_numeric(subset["Tradable_Return"], errors="coerce")
    subset["Benchmark_Tradable_Return"] = pd.to_numeric(subset["Benchmark_Tradable_Return"], errors="coerce")
    subset = subset.dropna(subset=["Date", "Probability", "Tradable_Return", "Benchmark_Tradable_Return"])
    subset = subset.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if subset.empty:
        return DataSource(
            "hourly_model_predictions",
            STATUS_INVALID,
            source_path=path,
            note=f"Hourly-Modellzeilen fuer '{universe_name}' enthalten keine gueltigen numerischen Werte.",
        )

    observed_tickers = set(subset["Ticker"].unique())
    missing_tickers = sorted(expected_tickers - observed_tickers)
    observations = int(subset["Date"].nunique())
    status = STATUS_READY if observations >= MIN_OBSERVATIONS_FOR_ROBUST and not missing_tickers else STATUS_PRELIMINARY
    note = (
        f"{observations} Hourly-Zeitpunkte ({subset['Date'].min()} bis {subset['Date'].max()}) "
        f"aus bestehender Hourly-Outperformance-Auswertung."
    )
    if missing_tickers:
        note += f" Fehlende Ticker in Hourly-Datei: {', '.join(missing_tickers)}."

    return DataSource(
        "hourly_model_predictions",
        status,
        data=subset,
        source_path=path,
        observations=observations,
        note=note,
    )


# ---------------------------------------------------------------------------
# 5. Signal / selection history (Alpaca), from alpaca_trading/logs
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
