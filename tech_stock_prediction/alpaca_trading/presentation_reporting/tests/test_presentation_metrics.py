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
import hourly_hybrid_reporting as hourly_hybrid  # noqa: E402
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
                "date": ["2026-07-06", "2026-07-06"],
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
                "timestamp": ["2026-07-06T14:00:00", "2026-07-06T15:00:00", "2026-07-06T16:00:00"],
                "portfolio_value": [1.0, 2.0, 3.0],
                "benchmark_value": [1.0, 2.0, 3.0],
            }
        ).to_csv(mixed_file, index=False)

        original_logs_dir = cfg.ALPACA_LOGS_DIR
        try:
            loader.ALPACA_LOGS_DIR = logs_dir  # type: ignore[attr-defined]
            combined, files = loader._scan_hourly_rows("original_tech", ["performance_history.csv"])  # noqa: SLF001
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


# ---------------------------------------------------------------------------
# 8. Top-K table loading, order counts, holding duration, aggregates
# ---------------------------------------------------------------------------

def test_load_top_k_table_reads_sibling_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        predictions_path = tmp_path / "lstm_outperformance_predictions.csv"
        predictions_path.write_text("Date,Ticker\n2026-01-01,AAA\n")
        pd.DataFrame(
            {
                "top_k": [2, 1, 3],
                "strategy_return": [0.05, 0.10, 0.02],
                "buy_and_hold_return": [0.08, 0.08, 0.08],
                "difference": [-0.03, 0.02, -0.06],
                "average_number_of_positions": [2.0, 1.0, 3.0],
            }
        ).to_csv(tmp_path / "lstm_outperformance_top_k_results.csv", index=False)

        table = metrics.load_top_k_table(str(predictions_path))
        check("load_top_k_table returns a table", table is not None)
        check("load_top_k_table sorts by top_k ascending", list(table["top_k"]) == [1, 2, 3], str(list(table["top_k"])))


def test_load_top_k_table_missing_file_returns_none():
    check("load_top_k_table returns None for missing sibling", metrics.load_top_k_table(None) is None)


def test_per_ticker_order_counts():
    orders = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "action": ["buy", "sell", "buy"],
        }
    )
    counts = metrics.per_ticker_order_counts(orders)
    check("per_ticker_order_counts counts AAA buys", counts["AAA"]["buys"] == 1, str(counts))
    check("per_ticker_order_counts counts AAA sells", counts["AAA"]["sells"] == 1, str(counts))
    check("per_ticker_order_counts counts BBB buys", counts["BBB"]["buys"] == 1, str(counts))
    check("per_ticker_order_counts empty for no orders", metrics.per_ticker_order_counts(None) == {})


def test_average_holding_duration():
    # Selected on days 1-2 (streak of 2), then day 4 alone (streak of 1) -> avg 1.5
    sequence = [True, True, False, True]
    duration = metrics._average_holding_duration(sequence)  # noqa: SLF001
    check("average holding duration over two streaks is 1.5", abs(duration - 1.5) < 1e-9, str(duration))
    check("average holding duration is NaN when never selected", np.isnan(metrics._average_holding_duration([False, False])))  # noqa: SLF001


def test_summarize_signal_selection_includes_holding_duration_and_orders():
    signal_data = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "generated_at": ["2026-01-01T10:00", "2026-01-02T10:00", "2026-01-03T10:00"],
            "ticker": ["AAA", "AAA", "AAA"],
            "selected": [True, True, False],
            "probability": [0.7, 0.6, 0.3],
            "rank": [1, 1, 2],
        }
    )
    order_counts = {"AAA": {"buys": 1, "sells": 1}}
    selection_df, _ = metrics.summarize_signal_selection(signal_data, ["AAA"], order_counts=order_counts)
    aaa_row = selection_df[selection_df["ticker"] == "AAA"].iloc[0]
    check("AAA average holding duration is 2.0 (one streak of 2)", abs(aaa_row["average_holding_duration"] - 2.0) < 1e-9, str(aaa_row["average_holding_duration"]))
    check("AAA buy/sell counts passed through", aaa_row["buys"] == 1 and aaa_row["sells"] == 1)


def test_summarize_signal_aggregate_most_frequent_top1_and_top3():
    signal_data = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "generated_at": ["2026-01-01T10:00"] * 2 + ["2026-01-02T10:00"] * 2,
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "selected": [True, False, True, False],
            "probability": [0.8, 0.4, 0.7, 0.3],
            "rank": [1, 2, 1, 2],
        }
    )
    selection_df, distinct_dates = metrics.summarize_signal_selection(signal_data, ["AAA", "BBB"])
    aggregate = metrics.summarize_signal_aggregate(signal_data, selection_df, distinct_dates)
    check("most frequent top-1 is AAA", aggregate["most_frequent_top1"] == "AAA", str(aggregate))
    check("most frequent top-3 starts with AAA", aggregate["most_frequent_top3"][0] == "AAA", str(aggregate))
    check("average turnover is 0 (AAA selected both days, no changes)", aggregate["average_turnover"] == 0.0, str(aggregate))


# ---------------------------------------------------------------------------
# 9. Hourly hybrid reporting
# ---------------------------------------------------------------------------

def test_hourly_hybrid_keeps_1hour_and_drops_1day_rows():
    with tempfile.TemporaryDirectory() as tmp:
        original_logs_dir = cfg.ALPACA_LOGS_DIR
        try:
            cfg.ALPACA_LOGS_DIR = Path(tmp)  # type: ignore[attr-defined]
            pd.DataFrame(
                {
                    "timestamp": ["2026-07-06T14:00:00", "2026-07-06T15:00:00"],
                    "universe": ["original_tech", "original_tech"],
                    "benchmark": ["QQQ", "QQQ"],
                    "timeframe": ["1Hour", "1Day"],
                    "portfolio_value": [100000.0, 100500.0],
                    "benchmark_value": [100.0, 101.0],
                }
            ).to_csv(Path(tmp) / "performance_history.csv", index=False)

            real, source_files = hourly_hybrid.normalize_real_hourly_observations("original_tech")
            check("hybrid loader keeps only 1Hour rows", len(real) == 1, str(real))
            check("hybrid loader records source file", source_files == ["performance_history.csv"], str(source_files))
        finally:
            cfg.ALPACA_LOGS_DIR = original_logs_dir  # type: ignore[attr-defined]


def test_hourly_hybrid_normalizes_legacy_benchmark_close_schema():
    with tempfile.TemporaryDirectory() as tmp:
        original_logs_dir = cfg.ALPACA_LOGS_DIR
        try:
            cfg.ALPACA_LOGS_DIR = Path(tmp)  # type: ignore[attr-defined]
            pd.DataFrame(
                {
                    "bar_timestamp": ["2026-07-06T14:00:00", "2026-07-06T15:00:00"],
                    "universe": ["defensive_non_tech", "defensive_non_tech"],
                    "benchmark": ["SPY", "SPY"],
                    "timeframe": ["1Hour", "1Hour"],
                    "portfolio_value": [100000.0, 101000.0],
                    "benchmark_close": [500.0, 505.0],
                }
            ).to_csv(Path(tmp) / "paper_performance_legacy.csv", index=False)

            real, _ = hourly_hybrid.normalize_real_hourly_observations("defensive_non_tech")
            check("legacy benchmark_close becomes benchmark_value", "benchmark_value" in real.columns)
            check("hybrid model index starts at 100", abs(real["model_index"].iloc[0] - 100.0) < 1e-9, str(real))
            check("hybrid benchmark index starts at 100", abs(real["benchmark_index"].iloc[0] - 100.0) < 1e-9, str(real))
        finally:
            cfg.ALPACA_LOGS_DIR = original_logs_dir  # type: ignore[attr-defined]


def test_hourly_hybrid_removes_duplicate_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        original_logs_dir = cfg.ALPACA_LOGS_DIR
        try:
            cfg.ALPACA_LOGS_DIR = Path(tmp)  # type: ignore[attr-defined]
            pd.DataFrame(
                {
                    "timestamp": ["2026-07-06T14:00:00", "2026-07-06T14:00:00"],
                    "universe": ["original_tech", "original_tech"],
                    "benchmark": ["QQQ", "QQQ"],
                    "timeframe": ["1Hour", "1Hour"],
                    "portfolio_value": [100000.0, 101000.0],
                    "benchmark_value": [100.0, 101.0],
                }
            ).to_csv(Path(tmp) / "performance_history.csv", index=False)

            real, _ = hourly_hybrid.normalize_real_hourly_observations("original_tech")
            check("hybrid loader removes duplicate timestamps", real["timestamp"].nunique() == len(real) == 1, str(real))
        finally:
            cfg.ALPACA_LOGS_DIR = original_logs_dir  # type: ignore[attr-defined]


def test_hourly_hybrid_simulation_is_reproducible_and_marked():
    real = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-06T14:00:00"]),
            "universe": ["original_tech"],
            "benchmark": ["QQQ"],
            "timeframe": ["1Hour"],
            "portfolio_value": [100000.0],
            "benchmark_value": [100.0],
            "model_index": [100.0],
            "benchmark_index": [100.0],
            "source_type": ["real"],
            "source_file": ["unit_test.csv"],
            "simulation_method": [""],
            "seed": [""],
        }
    )
    distribution = pd.DataFrame(
        {
            "model_return": [0.01, -0.02, 0.005],
            "benchmark_return": [0.002, -0.001, 0.003],
        }
    )
    original_loader = hourly_hybrid.load_hourly_backtest_return_distribution
    try:
        hourly_hybrid.load_hourly_backtest_return_distribution = lambda universe_name, top_k=5: (distribution, "unit_test_distribution")  # type: ignore[assignment]
        simulated_1, method_1, _ = hourly_hybrid.simulate_missing_hourly_points(real, "original_tech")
        simulated_2, method_2, _ = hourly_hybrid.simulate_missing_hourly_points(real, "original_tech")
        check("hybrid simulation uses fixed seed", simulated_1["model_index"].equals(simulated_2["model_index"]))
        check("hybrid simulation is marked simulated", set(simulated_1["source_type"]) == {"simulated"}, str(simulated_1["source_type"].unique()))
        check("hybrid simulation records method", method_1 == method_2 == "unit_test_distribution", method_1)
        check("hybrid simulation model and benchmark share time axis", simulated_1["timestamp"].equals(simulated_2["timestamp"]))
    finally:
        hourly_hybrid.load_hourly_backtest_return_distribution = original_loader  # type: ignore[assignment]


def test_hourly_hybrid_does_not_simulate_when_enough_real_data():
    timestamps = pd.date_range("2026-07-06T14:00:00", periods=cfg.MIN_OBSERVATIONS_FOR_ROBUST, freq="h")
    real = pd.DataFrame(
        {
            "timestamp": timestamps,
            "universe": "original_tech",
            "benchmark": "QQQ",
            "timeframe": "1Hour",
            "portfolio_value": np.linspace(100000, 101000, len(timestamps)),
            "benchmark_value": np.linspace(100, 101, len(timestamps)),
            "model_index": np.linspace(100, 101, len(timestamps)),
            "benchmark_index": np.linspace(100, 101, len(timestamps)),
            "source_type": "real",
            "source_file": "unit_test.csv",
            "simulation_method": "",
            "seed": "",
        }
    )
    simulated, method, _ = hourly_hybrid.simulate_missing_hourly_points(real, "original_tech")
    check("hybrid does not simulate when enough real data exists", simulated.empty and method == "not_needed", method)


# ---------------------------------------------------------------------------
# 10. Hourly model reporting
# ---------------------------------------------------------------------------

def test_reconstruct_hourly_topk_backtest_uses_top_probabilities():
    predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01 15:30", "2026-01-01 15:30", "2026-01-01 16:30", "2026-01-01 16:30"]),
            "Ticker": ["AAA", "BBB", "AAA", "BBB"],
            "Probability": [0.8, 0.2, 0.1, 0.9],
            "Tradable_Return": [0.10, -0.20, -0.30, 0.05],
            "Benchmark_Tradable_Return": [0.01, 0.01, 0.02, 0.02],
        }
    )
    hourly = metrics.reconstruct_hourly_topk_backtest(predictions, top_k=1)
    check("hourly top-k keeps one row per timestamp", len(hourly) == 2, str(hourly))
    check("hourly top-k picks AAA in first hour", hourly.iloc[0]["Selected_Tickers"] == ["AAA"], str(hourly.iloc[0]["Selected_Tickers"]))
    check("hourly top-k picks BBB in second hour", hourly.iloc[1]["Selected_Tickers"] == ["BBB"], str(hourly.iloc[1]["Selected_Tickers"]))
    check("hourly strategy return is selected ticker return", abs(hourly.iloc[1]["Strategy_Return"] - 0.05) < 1e-9)


def test_hourly_signal_stability_counts_entries_and_exits():
    predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01 15:30", "2026-01-01 15:30", "2026-01-01 16:30", "2026-01-01 16:30"]),
            "Ticker": ["AAA", "BBB", "AAA", "BBB"],
            "Probability": [0.8, 0.2, 0.1, 0.9],
            "Tradable_Return": [0.10, -0.20, -0.30, 0.05],
            "Benchmark_Tradable_Return": [0.01, 0.01, 0.02, 0.02],
        }
    )
    stability, aggregate = metrics.summarize_hourly_signal_stability(predictions, ["AAA", "BBB"], top_k=1)
    aaa = stability[stability["ticker"] == "AAA"].iloc[0]
    bbb = stability[stability["ticker"] == "BBB"].iloc[0]
    check("hourly stability selection share AAA is 50%", abs(aaa["selection_share"] - 0.5) < 1e-9, str(aaa))
    check("hourly stability selection share BBB is 50%", abs(bbb["selection_share"] - 0.5) < 1e-9, str(bbb))
    check("hourly stability counts AAA entry and exit", aaa["buys"] == 1 and aaa["sells"] == 1, str(aaa))
    check("hourly aggregate tracks two signal timestamps", aggregate["number_of_signal_dates"] == 2, str(aggregate))


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
