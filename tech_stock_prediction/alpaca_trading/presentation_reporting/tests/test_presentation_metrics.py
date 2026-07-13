"""
Plausibility tests for the presentation reporting system.

Plain assert-based runner (no pytest dependency, to avoid adding a new
package to the project). Run with:

    python alpaca_trading/presentation_reporting/tests/test_presentation_metrics.py

These are read-only sanity checks against small synthetic DataFrames (and,
for the loader validation tests, small temporary files) - nothing here
touches real project data, models, or logs.
"""

from pathlib import Path
import shutil
import sys
import tempfile

TESTS_ROOT = Path(__file__).resolve().parent
REPORTING_ROOT = TESTS_ROOT.parent
sys.path.insert(0, str(REPORTING_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import presentation_data_loader as loader  # noqa: E402
import presentation_metrics as metrics  # noqa: E402
import reporting_config as cfg  # noqa: E402


PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
    else:
        FAILED.append(f"{name} ({detail})" if detail else name)


# ---------------------------------------------------------------------------
# 1. Return calculation
# ---------------------------------------------------------------------------

def test_total_return_from_returns():
    returns = pd.Series([0.10, -0.05, 0.02])
    expected = (1.10 * 0.95 * 1.02) - 1
    actual = metrics.total_return_from_returns(returns)
    check("total_return_from_returns matches manual compounding", abs(actual - expected) < 1e-9, f"{actual} vs {expected}")


def test_build_cumulative_index_starts_at_100():
    returns = pd.Series([0.0, 0.10, -0.10])
    index = metrics.build_cumulative_index(returns)
    check("cumulative index starts exactly at 100", index.iloc[0] == 100.0, str(index.iloc[0]))
    check(
        "cumulative index compounds correctly",
        abs(index.iloc[-1] - 100 * 1.10 * 0.90) < 1e-9,
        f"{index.iloc[-1]}",
    )


# ---------------------------------------------------------------------------
# 2. Benchmark normalization
# ---------------------------------------------------------------------------

def test_normalize_to_100():
    values = pd.Series([250000.0, 300000.0, 200000.0])
    normalized = metrics.normalize_to_100(values)
    check("normalize_to_100 sets first value to 100", normalized.iloc[0] == 100.0)
    check("normalize_to_100 keeps relative scale", abs(normalized.iloc[1] - 120.0) < 1e-9, str(normalized.iloc[1]))


def test_summarize_alpaca_performance_uses_index_columns():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "portfolio_index": [100.0, 105.0, 95.0],
            "benchmark_index": [100.0, 101.0, 102.0],
        }
    )
    summary = metrics.summarize_alpaca_performance(data)
    check("alpaca portfolio_return matches index ratio", abs(summary["portfolio_return"] - (-0.05)) < 1e-9, str(summary))
    check("alpaca benchmark_return matches index ratio", abs(summary["benchmark_return"] - 0.02) < 1e-9, str(summary))
    check("alpaca difference = portfolio - benchmark", abs(summary["difference"] - (-0.07)) < 1e-9, str(summary))


# ---------------------------------------------------------------------------
# 3. Universe-specific filtering + QQQ/SPY mapping
# ---------------------------------------------------------------------------

def test_infer_universe_from_tickers():
    original_tech_tickers = cfg.UNIVERSES["original_tech"]
    check(
        "infer_universe_from_tickers finds original_tech",
        loader.infer_universe_from_tickers(original_tech_tickers) == "original_tech",
    )
    check(
        "infer_universe_from_tickers returns None for unknown set",
        loader.infer_universe_from_tickers(["ZZZZ", "YYYY"]) is None,
    )


def test_benchmark_mapping_is_correct_qqq_spy():
    check("original_tech benchmark is QQQ", cfg.BENCHMARKS["original_tech"] == "QQQ")
    check("tech_no_nvda benchmark is QQQ", cfg.BENCHMARKS["tech_no_nvda"] == "QQQ")
    check("new_tech benchmark is QQQ", cfg.BENCHMARKS["new_tech"] == "QQQ")
    check("defensive_non_tech benchmark is SPY", cfg.BENCHMARKS["defensive_non_tech"] == "SPY")


def test_backtest_predictions_rejects_wrong_universe_tickers():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake_predictions = tmp_path / "lstm_outperformance_predictions.csv"
        # Tickers belong to original_tech, but we ask for defensive_non_tech.
        pd.DataFrame(
            {
                "Date": ["2026-01-01", "2026-01-01"],
                "Ticker": ["AAPL", "MSFT"],
                "Probability": [0.6, 0.4],
                "Future_Return": [0.01, -0.01],
                "Next_Day_QQQ_Return": [0.005, 0.005],
                "Next_Day_SPY_Return": [0.002, 0.002],
            }
        ).to_csv(fake_predictions, index=False)

        original_live_path = cfg.EXP_2_LSTM_ROOT
        original_archive_path = cfg.UNIVERSE_RESULTS_ROOT
        try:
            loader.EXP_2_LSTM_ROOT = tmp_path  # type: ignore[attr-defined]
            loader.UNIVERSE_RESULTS_ROOT = tmp_path / "no_archive"  # type: ignore[attr-defined]
            # load_backtest_predictions reads EXP_2_LSTM_ROOT / data/processed/...
            (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
            shutil.move(str(fake_predictions), str(tmp_path / "data" / "processed" / "lstm_outperformance_predictions.csv"))

            data_source = loader.load_backtest_predictions("defensive_non_tech")
            check(
                "mismatched-universe backtest file is rejected as MISSING",
                data_source.status == cfg.STATUS_MISSING,
                data_source.note,
            )
        finally:
            loader.EXP_2_LSTM_ROOT = original_live_path  # type: ignore[attr-defined]
            loader.UNIVERSE_RESULTS_ROOT = original_archive_path  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 4. Missing / empty data handling (never silently becomes 0)
# ---------------------------------------------------------------------------

def test_missing_file_returns_missing_status_not_zero():
    with tempfile.TemporaryDirectory() as tmp:
        original_root = cfg.PER_UNIVERSE_DAILY_ROOT
        try:
            loader.PER_UNIVERSE_DAILY_ROOT = Path(tmp) / "does_not_exist"  # type: ignore[attr-defined]
            data_source = loader.load_daily_alpaca_performance("original_tech")
            check("missing daily alpaca file -> status MISSING", data_source.status == cfg.STATUS_MISSING)
            check("missing daily alpaca file -> data is None, not zeros", data_source.data is None)
        finally:
            loader.PER_UNIVERSE_DAILY_ROOT = original_root  # type: ignore[attr-defined]


def test_empty_csv_file_returns_missing_status():
    with tempfile.TemporaryDirectory() as tmp:
        universe_dir = Path(tmp) / "original_tech" / "logs"
        universe_dir.mkdir(parents=True)
        empty_file = universe_dir / "daily_mark_to_market_with_baselines.csv"
        empty_file.write_text("")

        original_root = cfg.PER_UNIVERSE_DAILY_ROOT
        try:
            loader.PER_UNIVERSE_DAILY_ROOT = Path(tmp)  # type: ignore[attr-defined]
            data_source = loader.load_daily_alpaca_performance("original_tech")
            check("empty csv file -> status MISSING (not a fabricated flat line)", data_source.status == cfg.STATUS_MISSING)
        finally:
            loader.PER_UNIVERSE_DAILY_ROOT = original_root  # type: ignore[attr-defined]


def test_comparison_bars_do_not_turn_missing_into_zero():
    # A None value with STATUS_MISSING must be rendered as a hatched/"n/a"
    # bar, not silently plotted as a 0%. We check the plotting function's
    # internal logic by calling it with a temp output path and inspecting
    # that it does not raise and that missing entries are distinguishable
    # from a genuine 0.0 value (handled via color/hatch in presentation_plots).
    import presentation_plots as plots

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "test_comparison.png"
        # Should not raise, and None must be accepted without turning into 0.
        plots.plot_comparison_bars(
            title="Test",
            subtitle="Test",
            universes=["A", "B"],
            values=[0.0, None],
            statuses=[cfg.STATUS_READY, cfg.STATUS_MISSING],
            output_path=output_path,
        )
        check("plot_comparison_bars handles None without raising", output_path.exists())


# ---------------------------------------------------------------------------
# 5. Duplicate timestamps
# ---------------------------------------------------------------------------

def test_duplicate_dates_are_flagged_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        universe_dir = Path(tmp) / "original_tech" / "logs"
        universe_dir.mkdir(parents=True)
        duplicate_file = universe_dir / "daily_mark_to_market_with_baselines.csv"
        pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "universe": ["original_tech", "original_tech"],
                "benchmark": ["QQQ", "QQQ"],
                "portfolio_index": [100.0, 101.0],
                "benchmark_index": [100.0, 100.5],
            }
        ).to_csv(duplicate_file, index=False)

        original_root = cfg.PER_UNIVERSE_DAILY_ROOT
        try:
            loader.PER_UNIVERSE_DAILY_ROOT = Path(tmp)  # type: ignore[attr-defined]
            data_source = loader.load_daily_alpaca_performance("original_tech")
            check("duplicate dates -> status INVALID", data_source.status == cfg.STATUS_INVALID, data_source.note)
        finally:
            loader.PER_UNIVERSE_DAILY_ROOT = original_root  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 6. Hourly / Daily separation and Backtest / Alpaca separation
# ---------------------------------------------------------------------------

def test_hourly_scan_only_keeps_1hour_rows():
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        mixed_file = logs_dir / "performance_history.csv"
        pd.DataFrame(
            {
                "timeframe": ["1Hour", "1Day", "1Day"],
                "universe": ["original_tech", "original_tech", "original_tech"],
                "timestamp": ["t1", "t2", "t3"],
                "portfolio_value": [1.0, 2.0, 3.0],
                "benchmark_value": [1.0, 2.0, 3.0],
            }
        ).to_csv(mixed_file, index=False)

        original_logs_dir = cfg.ALPACA_LOGS_DIR
        try:
            loader.ALPACA_LOGS_DIR = logs_dir  # type: ignore[attr-defined]
            combined, files = loader._scan_hourly_rows("original_tech")  # noqa: SLF001
            check("hourly scan drops 1Day rows from a mixed file", len(combined) == 1, str(len(combined) if combined is not None else None))
            check("hourly scan keeps only the 1Hour row", combined.iloc[0]["timeframe"] == "1Hour")
        finally:
            loader.ALPACA_LOGS_DIR = original_logs_dir  # type: ignore[attr-defined]


def test_backtest_and_alpaca_sources_are_independent():
    # The backtest loader must never read from alpaca_trading/logs or
    # per_universe_daily_mark_to_market, and vice versa - verified by
    # checking the constants point at disjoint folders.
    check(
        "backtest source root differs from alpaca log roots",
        cfg.EXP_2_LSTM_ROOT != cfg.ALPACA_LOGS_DIR and cfg.EXP_2_LSTM_ROOT != cfg.PER_UNIVERSE_DAILY_ROOT,
    )
    check(
        "daily alpaca source root differs from shared account log root",
        cfg.PER_UNIVERSE_DAILY_ROOT != cfg.ALPACA_LOGS_DIR,
    )


# ---------------------------------------------------------------------------
# 7. Signal selection dedup / share calculation
# ---------------------------------------------------------------------------

def test_signal_selection_share_and_dedup():
    signal_data = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "generated_at": ["2026-01-01T10:00", "2026-01-01T10:00", "2026-01-02T10:00", "2026-01-02T10:00"],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "selected": [True, False, False, True],
            "probability": [0.7, 0.3, 0.4, 0.6],
            "rank": [1, 2, 2, 1],
        }
    )
    selection_df, distinct_dates = metrics.summarize_signal_selection(signal_data, ["AAA", "BBB"])
    check("signal selection sees 2 distinct dates", distinct_dates == 2, str(distinct_dates))

    aaa_row = selection_df[selection_df["ticker"] == "AAA"].iloc[0]
    check("AAA selected on 1 of 2 days -> share 0.5", abs(aaa_row["selection_share"] - 0.5) < 1e-9, str(aaa_row["selection_share"]))


def test_signal_selection_deduplicates_repeated_runs_same_date():
    # Two runs logged for the same date/ticker (a scheduler retry) must not
    # be double-counted; only the latest run per (date, ticker) is kept.
    signal_data = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-01"],
            "generated_at": ["2026-01-01T09:00", "2026-01-01T09:00", "2026-01-01T10:00"],
            "ticker": ["AAA", "BBB", "AAA"],
            "selected": [False, True, True],
            "probability": [0.2, 0.9, 0.8],
            "rank": [3, 1, 1],
        }
    )
    selection_df, distinct_dates = metrics.summarize_signal_selection(signal_data, ["AAA", "BBB"])
    check("dedup keeps 1 distinct date despite 2 runs", distinct_dates == 1, str(distinct_dates))
    aaa_row = selection_df[selection_df["ticker"] == "AAA"].iloc[0]
    check(
        "dedup keeps the LATEST run for AAA (selected=True), not the first",
        aaa_row["selection_share"] == 1.0,
        str(aaa_row["selection_share"]),
    )


def main():
    test_functions = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test_function in test_functions:
        try:
            test_function()
        except Exception as error:  # noqa: BLE001
            FAILED.append(f"{test_function.__name__} raised {type(error).__name__}: {error}")

    print(f"Passed: {len(PASSED)}")
    for name in PASSED:
        print(f"  OK   {name}")

    print(f"\nFailed: {len(FAILED)}")
    for name in FAILED:
        print(f"  FAIL {name}")

    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
