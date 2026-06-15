# Experiment 1.5 - Index (QQQ)

This experiment runs the **exact same pipeline as exp_1_1** (same features, same
chronological split, same per-ticker normalization, same Random Forest, same
Always-Yes baseline and backtest), but on a single index ETF (**QQQ**, the
Nasdaq-100) instead of the 10 individual tech stocks.

## Why an index?

The lecturer noted that pooling 10 different stocks can blur the statistics
(different volatility / price regimes mixed together). An index already bundles
many stocks into **one homogeneous price series**, so the cross-sectional
pooling problem disappears - a kind of automatic aggregation. The question:
does the model work better on this cleaner series?

## Two time horizons

Only the `START_DATE` in `conf/params.yaml` changes between the two runs:

| Horizon | START_DATE | Purpose |
| --- | --- | --- |
| Short | `2019-01-01` | Direct, apples-to-apples comparison with exp_1_1 |
| Full  | `1999-01-01` | Maximum QQQ history (ETF inception was March 1999), more data and more diverse market regimes (dot-com crash, 2008, 2022) |

QQQ cannot go earlier than 1999. Only the raw `^NDX` index reaches back to 1985,
but index series have unreliable volume data in yfinance and our features use
volume, so QQQ is the practical maximum with working features.

To switch, change `START_DATE` in `conf/params.yaml` and re-run the pipeline.

## Run order

```bash
cd experiments/exp_1_5_indices
python scripts/01_data_acquisition/bar_retriever.py
python scripts/03_pre_split_prep/main.py
python scripts/04_split_data/split.py
python scripts/05_model_training/train_random_forest.py
python scripts/06_model_testing/evaluate_random_forest.py
python scripts/07_backtesting/simple_backtest.py
python scripts/08_baseline/dummy_baseline.py
```

(`02_data_understanding/plotter.py` can be run after step 1 for a data overview.)

## Result summary

In every setup the Random Forest does **not** beat the naive Always-Yes
baseline, and the strategy underperforms buy-and-hold:

| Setup | Test Accuracy | Always-Yes | Model - Baseline | Strategy vs. Buy-and-Hold |
| --- | --- | --- | --- | --- |
| 10 single stocks (2019) | ~50.6% | ~51.9% | ~-1.3 pp | +16% vs +58% |
| QQQ (2019, 1y test)     | ~52.6% | ~58.2% | ~-5.6 pp | +16% vs +34% |
| QQQ (1999, 4y test)     | ~49.6% | ~55.8% | ~-6.2 pp | 10.5% vs 27.4% CAGR |

**Conclusion:** Neither switching to the index nor using much more history gives
the model an edge. The lack of signal is fundamental, not a data-quantity
problem - daily price direction is not predictable from technical features alone.
(Exact numbers vary slightly per run because yfinance re-downloads fresh data.)
