"""
Hybrid hourly reporting for presentation plots.

This module is read-only with respect to trading/model files. It normalizes
real 1Hour Paper-Trading observations when they exist and, if there are too
few points for a presentation curve, appends a clearly marked reproducible
simulation. Simulated rows are never labeled as real Alpaca data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import reporting_config as cfg


SEED = 42
MIN_REAL_POINTS_WITHOUT_SIMULATION = cfg.MIN_OBSERVATIONS_FOR_ROBUST
TARGET_HYBRID_POINTS = cfg.MIN_OBSERVATIONS_FOR_ROBUST
HOURLY_PREDICTIONS_FILE = (
    cfg.EXP_2_LSTM_ROOT
    / "hourly"
    / "data"
    / "hourly_outperformance_predictions.csv"
)


def target_columns():
    return [
        "timestamp",
        "universe",
        "benchmark",
        "timeframe",
        "portfolio_value",
        "benchmark_value",
        "model_index",
        "benchmark_index",
        "source_type",
        "source_file",
        "simulation_method",
        "seed",
    ]


def read_csv_lenient(path):
    """Read current and legacy CSV files, skipping malformed legacy lines."""
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return None


def normalize_to_100(values):
    values = pd.Series(values, dtype="float64")
    if values.empty or pd.isna(values.iloc[0]) or values.iloc[0] == 0:
        return values
    return values / values.iloc[0] * 100.0


def parse_timestamp(data):
    """
    Prefer the market bar timestamp over the run timestamp.

    bar_timestamp is the market data point used for the signal; timestamp or
    generated_at is when the script ran.
    """
    for column in ["bar_timestamp", "timestamp", "generated_at", "date"]:
        if column in data.columns:
            parsed = pd.to_datetime(data[column], errors="coerce", utc=True).dt.tz_convert(None)
            if parsed.notna().any():
                return parsed
    return pd.Series(pd.NaT, index=data.index)


def has_hourly_rows(data):
    return "timeframe" in data.columns and data["timeframe"].astype(str).eq("1Hour").any()


def normalize_real_hourly_observations(universe_name):
    """
    Return real Hourly observations in the shared target schema.

    Only rows with explicit timeframe == 1Hour and universe == universe_name
    are used. Daily rows are ignored, even if they share the same file.
    """
    rows = []
    source_files = []

    for path in sorted(cfg.ALPACA_LOGS_DIR.glob("*.csv")):
        data = read_csv_lenient(path)
        if data is None or data.empty or "universe" not in data.columns or not has_hourly_rows(data):
            continue

        subset = data[
            data["timeframe"].astype(str).eq("1Hour")
            & data["universe"].astype(str).eq(universe_name)
        ].copy()
        if subset.empty:
            continue

        subset["timestamp"] = parse_timestamp(subset)
        subset["portfolio_value"] = pd.to_numeric(subset.get("portfolio_value"), errors="coerce")

        if "benchmark_value" in subset.columns:
            subset["benchmark_value"] = pd.to_numeric(subset["benchmark_value"], errors="coerce")
        elif "benchmark_close" in subset.columns:
            subset["benchmark_value"] = pd.to_numeric(subset["benchmark_close"], errors="coerce")
        else:
            subset["benchmark_value"] = np.nan

        subset["benchmark"] = subset.get("benchmark", cfg.BENCHMARKS[universe_name])
        subset["source_file"] = path.name
        subset = subset.dropna(subset=["timestamp", "portfolio_value", "benchmark_value"])
        if subset.empty:
            continue

        rows.append(
            subset[
                [
                    "timestamp",
                    "universe",
                    "benchmark",
                    "timeframe",
                    "portfolio_value",
                    "benchmark_value",
                    "source_file",
                ]
            ]
        )
        source_files.append(path.name)

    if not rows:
        return pd.DataFrame(columns=target_columns()), source_files

    real = pd.concat(rows, ignore_index=True, sort=False)
    real["timestamp"] = pd.to_datetime(real["timestamp"], errors="coerce")
    real = real.dropna(subset=["timestamp"])
    real = real.sort_values(["timestamp", "source_file"]).drop_duplicates("timestamp", keep="last")
    real = real.sort_values("timestamp").reset_index(drop=True)
    real["model_index"] = normalize_to_100(real["portfolio_value"])
    real["benchmark_index"] = normalize_to_100(real["benchmark_value"])
    real["source_type"] = "real"
    real["simulation_method"] = ""
    real["seed"] = ""

    return real[target_columns()], sorted(set(source_files))


def real_return_distribution(real):
    """Use real Hourly Paper-Trading changes if at least two real points exist."""
    if len(real) < cfg.MIN_OBSERVATIONS_FOR_ANY_PLOT:
        return None, "not_enough_real_hourly_returns"

    distribution = pd.DataFrame(
        {
            "model_return": real["model_index"].astype(float).pct_change(),
            "benchmark_return": real["benchmark_index"].astype(float).pct_change(),
        }
    ).dropna()
    if distribution.empty:
        return None, "real_hourly_returns_empty"

    return distribution.reset_index(drop=True), "real_hourly_return_bootstrap"


def load_hourly_backtest_return_distribution(universe_name, top_k=5):
    """
    Load an hourly return distribution without using Daily returns.

    The current project contains one hourly backtest prediction file with QQQ
    and SPY rows. For each universe we keep only the matching benchmark.
    """
    if not HOURLY_PREDICTIONS_FILE.exists():
        return None, "missing_hourly_backtest_predictions"

    data = read_csv_lenient(HOURLY_PREDICTIONS_FILE)
    if data is None or data.empty:
        return None, "empty_hourly_backtest_predictions"

    required = {"Date", "Benchmark", "Probability", "Tradable_Return", "Benchmark_Tradable_Return"}
    if not required.issubset(data.columns):
        return None, f"hourly_predictions_missing_columns:{sorted(required - set(data.columns))}"

    expected_benchmark = cfg.BENCHMARKS[universe_name]
    data = data[data["Benchmark"].astype(str).eq(expected_benchmark)].copy()
    if data.empty:
        return None, f"no_hourly_backtest_rows_for_benchmark:{expected_benchmark}"

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Probability"] = pd.to_numeric(data["Probability"], errors="coerce")
    data["Tradable_Return"] = pd.to_numeric(data["Tradable_Return"], errors="coerce")
    data["Benchmark_Tradable_Return"] = pd.to_numeric(data["Benchmark_Tradable_Return"], errors="coerce")
    data = data.dropna(subset=["Date", "Probability", "Tradable_Return", "Benchmark_Tradable_Return"])
    if data.empty:
        return None, "hourly_backtest_rows_not_numeric"

    data["rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    selected = data[data["rank"] <= top_k].copy()
    strategy_returns = selected.groupby("Date")["Tradable_Return"].mean()
    benchmark_returns = data.groupby("Date")["Benchmark_Tradable_Return"].first()
    distribution = pd.concat(
        [
            strategy_returns.rename("model_return"),
            benchmark_returns.rename("benchmark_return"),
        ],
        axis=1,
    ).dropna()

    if len(distribution) < cfg.MIN_OBSERVATIONS_FOR_ANY_PLOT:
        return None, "hourly_backtest_distribution_too_small"

    return distribution.reset_index(drop=True), f"hourly_backtest_distribution_top_{top_k}"


def next_market_hours(start_timestamp, count):
    """Generate future hourly timestamps inside the configured US market window."""
    timestamps = []
    current = pd.Timestamp(start_timestamp) + pd.Timedelta(hours=1)
    while len(timestamps) < count:
        # Summer 2026: German 15:30-22:00 roughly maps to UTC 13:30-20:00.
        if current.weekday() < 5 and 13 <= current.hour <= 20:
            timestamps.append(current)
        current += pd.Timedelta(hours=1)
    return pd.to_datetime(timestamps)


def simulate_missing_hourly_points(real, universe_name, top_k=5):
    """
    Append simulated points only when a valid hourly basis exists.

    Priority:
    1. bootstrap from real Hourly Paper-Trading returns
    2. bootstrap from hourly backtest returns for the same benchmark
    """
    if len(real) >= MIN_REAL_POINTS_WITHOUT_SIMULATION:
        return pd.DataFrame(columns=target_columns()), "not_needed", ""

    distribution, method = real_return_distribution(real)
    if distribution is None:
        distribution, method = load_hourly_backtest_return_distribution(universe_name, top_k=top_k)

    if distribution is None or distribution.empty:
        return pd.DataFrame(columns=target_columns()), "not_possible", method

    if real.empty:
        # No real anchor means we cannot honestly connect simulation to Paper-Trading history.
        return pd.DataFrame(columns=target_columns()), "not_possible", "no_real_anchor_for_simulation"

    missing_count = max(TARGET_HYBRID_POINTS - len(real), 0)
    if missing_count == 0:
        return pd.DataFrame(columns=target_columns()), method, ""

    rng = np.random.default_rng(SEED)
    sample_indices = rng.integers(0, len(distribution), size=missing_count)
    sampled = distribution.iloc[sample_indices].reset_index(drop=True)
    timestamps = next_market_hours(real["timestamp"].iloc[-1], missing_count)

    last_model = float(real["model_index"].iloc[-1])
    last_benchmark = float(real["benchmark_index"].iloc[-1])
    model_index = []
    benchmark_index = []

    for _, row in sampled.iterrows():
        last_model *= 1 + float(row["model_return"])
        last_benchmark *= 1 + float(row["benchmark_return"])
        model_index.append(last_model)
        benchmark_index.append(last_benchmark)

    simulated = pd.DataFrame(
        {
            "timestamp": timestamps,
            "universe": universe_name,
            "benchmark": cfg.BENCHMARKS[universe_name],
            "timeframe": "1Hour",
            "portfolio_value": np.nan,
            "benchmark_value": np.nan,
            "model_index": model_index,
            "benchmark_index": benchmark_index,
            "source_type": "simulated",
            "source_file": f"simulation:{method}",
            "simulation_method": method,
            "seed": SEED,
        }
    )
    return simulated[target_columns()], method, ""


def build_metadata(universe_name, source_files, real_count, simulated_count, simulation_method, note):
    return {
        "universe": universe_name,
        "benchmark": cfg.BENCHMARKS[universe_name],
        "real_source_files": source_files,
        "real_observations": real_count,
        "simulated_observations": simulated_count,
        "simulation_method": simulation_method,
        "seed": SEED if simulated_count else "",
        "note": note,
    }


def build_hourly_hybrid_series(universe_name, top_k=5):
    real, source_files = normalize_real_hourly_observations(universe_name)
    simulated, simulation_method, simulation_note = simulate_missing_hourly_points(real, universe_name, top_k=top_k)

    if real.empty and simulated.empty:
        note = "Keine realen Hourly-Paper-Trading-Punkte mit Portfolio- und Benchmark-Wert gefunden."
        if simulation_note:
            note += f" Simulation nicht moeglich: {simulation_note}."
        return pd.DataFrame(columns=target_columns()), build_metadata(
            universe_name, source_files, 0, 0, "not_possible", note
        )

    series = pd.concat([real, simulated], ignore_index=True, sort=False)
    series = series.sort_values("timestamp").drop_duplicates(["timestamp", "source_type"], keep="last")
    series = series.reset_index(drop=True)

    metadata = build_metadata(
        universe_name=universe_name,
        source_files=source_files,
        real_count=int((series["source_type"] == "real").sum()),
        simulated_count=int((series["source_type"] == "simulated").sum()),
        simulation_method=simulation_method,
        note=simulation_note,
    )
    return series[target_columns()], metadata


def period_text(data):
    if data.empty:
        return "n/a"
    return f"{pd.to_datetime(data['timestamp']).min()} bis {pd.to_datetime(data['timestamp']).max()}"


def summarize_hybrid_series(series, metadata):
    if series.empty:
        return {
            "observations": 0,
            "real_observations": 0,
            "simulated_observations": 0,
            "model_return": None,
            "benchmark_return": None,
            "difference": None,
            "real_period": "n/a",
            "simulated_period": "n/a",
            "simulation_method": metadata["simulation_method"],
        }

    real = series[series["source_type"] == "real"]
    simulated = series[series["source_type"] == "simulated"]
    model_return = float(series["model_index"].iloc[-1] / series["model_index"].iloc[0] - 1)
    benchmark_return = float(series["benchmark_index"].iloc[-1] / series["benchmark_index"].iloc[0] - 1)

    return {
        "observations": int(len(series)),
        "real_observations": int(len(real)),
        "simulated_observations": int(len(simulated)),
        "model_return": model_return,
        "benchmark_return": benchmark_return,
        "difference": model_return - benchmark_return,
        "real_period": period_text(real),
        "simulated_period": period_text(simulated),
        "simulation_method": metadata["simulation_method"],
    }


def build_methodology_text(metadata_by_universe):
    lines = [
        "# Hourly Hybrid Methodology",
        "",
        "Diese Datei dokumentiert die Hybrid-Visualisierung fuer das stundenbasierte Paper Trading.",
        "",
        "Wichtig: Simulierte Daten sind keine echten Alpaca-Paper-Trading-Ergebnisse.",
        "",
        f"Random Seed: {SEED}",
        "",
        "## Methode",
        "",
        "- Echte Beobachtungen werden nur verwendet, wenn `timeframe == 1Hour` eindeutig gesetzt ist.",
        "- Legacy- und aktuelle CSV-Schemas werden auf ein gemeinsames Format normalisiert.",
        "- Daily-Zeilen werden nicht als Hourly-Zeilen interpretiert.",
        "- Modell- und Benchmark-Reihe werden gemeinsam auf Startwert 100 normiert.",
        "- Wenn weniger als 30 echte Hourly-Punkte vorhanden sind, wird eine reproduzierbare Simulation ergaenzt.",
        "- Simulation nutzt zuerst reale Hourly-Renditen; falls zu wenige vorhanden sind, die Hourly-Backtest-Renditeverteilung fuer denselben Benchmark.",
        "- Wenn keine belastbare Simulationsbasis vorhanden ist, wird nicht simuliert.",
        "",
        "## Universen",
        "",
    ]

    for universe_name, metadata in metadata_by_universe.items():
        lines.extend(
            [
                f"### {universe_name}",
                "",
                f"- Benchmark: {metadata['benchmark']}",
                f"- Reale Beobachtungen: {metadata['real_observations']}",
                f"- Simulierte Beobachtungen: {metadata['simulated_observations']}",
                f"- Verwendete reale Dateien: {', '.join(metadata['real_source_files']) if metadata['real_source_files'] else 'keine'}",
                f"- Simulationsmethode: {metadata['simulation_method']}",
                f"- Seed: {metadata['seed']}",
                f"- Hinweis: {metadata['note'] or 'keine'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Einschraenkungen",
            "",
            "- Die echte Hourly-Historie ist unvollstaendig, weil fruehe Dry-Run-Logs spaeter durch Schemawechsel archiviert wurden.",
            "- Die Hybrid-Kurve ist ein Szenario fuer den stundenbasierten Serverbetrieb, kein vollstaendiges Paper-Trading-Ergebnis.",
            "- Die Simulation wird nicht auf ein positives Ergebnis optimiert.",
            "- Fehlende Werte werden nicht als 0 interpretiert.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_hybrid_outputs(series_by_universe, metadata_by_universe):
    """Write combined/per-universe CSVs and the methodology document."""
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    non_empty = [series for series in series_by_universe.values() if not series.empty]
    if non_empty:
        combined = pd.concat(non_empty, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(columns=target_columns())

    combined_csv = cfg.OUTPUT_ROOT / "hourly_hybrid_series.csv"
    combined.to_csv(combined_csv, index=False)

    for universe_name, series in series_by_universe.items():
        output_dir = cfg.universe_output_dir(universe_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        series.to_csv(output_dir / "hourly_hybrid_series.csv", index=False)

    methodology_path = cfg.OUTPUT_ROOT / "hourly_hybrid_methodology.md"
    methodology_path.write_text(build_methodology_text(metadata_by_universe), encoding="utf-8")
    print(f"Saved: {combined_csv}")
    print(f"Saved: {methodology_path}")
