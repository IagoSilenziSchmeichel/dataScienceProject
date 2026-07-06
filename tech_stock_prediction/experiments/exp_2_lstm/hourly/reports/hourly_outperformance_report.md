# Hourly Outperformance LSTM Report

## Purpose

This hourly model is used only for Alpaca Paper Trading live validation. The daily LSTM pipeline remains the scientific research pipeline.

## Training Period

- First hourly bar: 2023-09-15 16:30:00
- Last hourly bar: 2026-07-02 19:30:00
- Train rows: 102390
- Validation rows: 21905
- Test rows: 21858

## Samples

- Train sequences: 101820
- Validation sequences: 21335
- Test sequences: 21288

## Label Distribution

- Class 0: 50.26%
- Class 1: 49.74%

## Test Metrics

- Best validation F1: 0.50385
- Accuracy: 0.49930
- Precision: 0.49643
- Recall: 0.51252
- F1: 0.50435

## Top-K Results

| model_name | top_k | test_start | test_end | return | gross_return | benchmark_return | difference | transaction_cost | sharpe | max_drawdown | volatility | trades | turnover | average_number_of_positions | number_of_hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hourly_outperformance_lstm | 1 | 2026-02-05T16:30:00 | 2026-07-02T19:30:00 | -0.71412 | 0.00010 | 0.08674 | -0.80085 | 0.00100 | -6.63891 | -0.72404 | 0.42015 | 712 | 1.75809 | 1.00000 | 712 |
| hourly_outperformance_lstm | 3 | 2026-02-05T16:30:00 | 2026-07-02T19:30:00 | -0.66897 | 0.02201 | 0.07734 | -0.74631 | 0.00100 | -9.16828 | -0.66910 | 0.27312 | 2136 | 1.58275 | 3.00000 | 712 |
| hourly_outperformance_lstm | 5 | 2026-02-05T16:30:00 | 2026-07-02T19:30:00 | -0.63061 | 0.02796 | 0.06572 | -0.69633 | 0.00100 | -9.01098 | -0.63460 | 0.25059 | 3560 | 1.43685 | 5.00000 | 712 |

## Benchmark Comparison

The hourly Top-K strategy is compared against the matching benchmark return for the selected universe rows: QQQ for tech universes and SPY for defensive_non_tech.

## Alpaca Recommendation

Use `hourly_outperformance_lstm_model.pth`, `hourly_outperformance_scaler.pkl` and `hourly_features.txt` for Alpaca Paper Trading. Keep `run_lstm_pipeline.py` and the daily research artifacts unchanged.
