# LSTM Robustness Report

This report checks whether the two best LSTM variants are stable across time windows, Top-K values, transaction costs and return concentration.

## Tested Variants

- `standard_lstm` with `Technical + Market`, default Top-K = 2
- `outperformance_lstm` with `Technical + Relative Strength`, default Top-K = 1

## Walk-Forward Summary

| model_name | feature_group | window | top_k | transaction_cost | strategy_return | buy_and_hold_return | difference | sharpe_ratio | max_drawdown | number_of_trades | average_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_lstm | Technical + Market | 2025_H1 | 2 | 0.10% | -6.00% | 1.74% | -7.74% | -0.1102 | -26.61% | 288 | 0.6792 |
| standard_lstm | Technical + Market | 2025_H2 | 2 | 0.10% | 52.77% | 28.01% | 24.76% | 2.5788 | -11.31% | 292 | 0.5703 |
| standard_lstm | Technical + Market | 2026_H1 | 2 | 0.10% | 32.76% | 26.70% | 6.06% | 2.0783 | -13.25% | 244 | 0.6224 |
| outperformance_lstm | Technical + Relative Strength | 2025_H1 | 1 | 0.10% | 11.90% | 1.74% | 10.16% | 0.7991 | -27.19% | 185 | 0.8726 |
| outperformance_lstm | Technical + Relative Strength | 2025_H2 | 1 | 0.10% | 94.00% | 28.01% | 66.00% | 3.3287 | -13.33% | 173 | 0.6758 |
| outperformance_lstm | Technical + Relative Strength | 2026_H1 | 1 | 0.10% | 6.42% | 23.47% | -17.05% | 0.5335 | -29.80% | 123 | 0.5591 |

## Top-K Sensitivity

| model_name | feature_group | top_k | difference |
| --- | --- | --- | --- |
| outperformance_lstm | Technical + Relative Strength | 1 | 19.70% |
| outperformance_lstm | Technical + Relative Strength | 2 | 5.28% |
| outperformance_lstm | Technical + Relative Strength | 3 | -5.31% |
| outperformance_lstm | Technical + Relative Strength | 4 | -3.28% |
| outperformance_lstm | Technical + Relative Strength | 5 | -8.63% |
| standard_lstm | Technical + Market | 1 | 30.09% |
| standard_lstm | Technical + Market | 2 | 7.69% |
| standard_lstm | Technical + Market | 3 | -0.70% |
| standard_lstm | Technical + Market | 4 | -4.23% |
| standard_lstm | Technical + Market | 5 | -6.32% |

## Transaction Cost Sensitivity

| model_name | feature_group | transaction_cost | difference |
| --- | --- | --- | --- |
| outperformance_lstm | Technical + Relative Strength | 0.00% | 31.37% |
| outperformance_lstm | Technical + Relative Strength | 0.05% | 25.42% |
| outperformance_lstm | Technical + Relative Strength | 0.10% | 19.70% |
| outperformance_lstm | Technical + Relative Strength | 0.20% | 8.95% |
| standard_lstm | Technical + Market | 0.00% | 16.65% |
| standard_lstm | Technical + Market | 0.05% | 12.09% |
| standard_lstm | Technical + Market | 0.10% | 7.69% |
| standard_lstm | Technical + Market | 0.20% | -0.67% |

## Concentration Check

| model_name | feature_group | largest_ticker | largest_ticker_positive_share | top_5_day_positive_share | largest_single_day_return | worst_single_day_return |
| --- | --- | --- | --- | --- | --- | --- |
| standard_lstm | Technical + Market | INTC | 0.3281 | 0.1840 | 15.27% | -10.14% |
| outperformance_lstm | Technical + Relative Strength | NVDA | 0.2327 | 0.1228 | 11.83% | -10.86% |

## Plots

- `plots/lstm_robustness_walk_forward_difference.png`
- `plots/lstm_robustness_top_k_sensitivity.png`
- `plots/lstm_robustness_cost_sensitivity.png`
- `plots/lstm_robustness_concentration.png`
