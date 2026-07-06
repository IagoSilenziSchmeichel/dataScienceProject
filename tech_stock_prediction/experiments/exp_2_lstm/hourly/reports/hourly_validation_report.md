# Hourly Pipeline Validation Report

## Checklist

- ✔ Look-Ahead-Bias geprüft
- ✔ Data Leakage geprüft
- ✔ Backtest validiert
- ✔ Transaktionskosten geprüft
- ✔ Walk-Forward geprüft
- ✔ Top-K geprüft
- ✔ Alpaca geprüft
- ✔ Logging geprüft
- ✔ Scaler geprüft
- ✔ Code Review abgeschlossen

## Artifact Status

- Model: `experiments/exp_2_lstm/hourly/models/hourly_outperformance_lstm_model.pth`
- Scaler: `experiments/exp_2_lstm/hourly/models/hourly_outperformance_scaler.pkl`
- Features: `experiments/exp_2_lstm/hourly/conf/hourly_features.txt`
- Predictions: `experiments/exp_2_lstm/hourly/data/hourly_outperformance_predictions.csv`
- Validated predictions: `experiments/exp_2_lstm/hourly/data/hourly_outperformance_predictions_validated.csv`
- Validation return mode: `tradable_next_open_to_close`

## Look-Ahead / Leakage Review

- Feature engineering uses rolling, lagged and exponentially weighted indicators computed per ticker from current and past bars.
- Target is shifted by exactly one hour via next-hour returns.
- Scaling is fitted on train data only and then applied to validation/test data.
- Chronological split is timestamp based.
- Finding: the original saved predictions did not contain tradable return columns.
- Fix: validation reconstructs `Tradable_Return` and `Benchmark_Tradable_Return` from hourly Yahoo bars without retraining the model.

## Backtest Validation

Validation uses next-hour open-to-close tradable returns.

Top-K results with 0.10% buy/sell costs:

| period | top_k | transaction_cost | return | gross_return | benchmark_return | difference | sharpe | max_drawdown | volatility | number_of_hours | average_positions | trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 1 | 0.00100 | -0.77057 | -0.21447 | 0.08860 | -0.85917 | -7.08780 | -0.78391 | 0.46157 | 713 | 1.00000 | 615 |
| full | 3 | 0.00100 | -0.62446 | 0.20310 | 0.09755 | -0.72201 | -7.35605 | -0.63809 | 0.29954 | 713 | 3.00000 | 707 |
| full | 5 | 0.00100 | -0.60032 | 0.18729 | 0.08806 | -0.68838 | -8.19715 | -0.61181 | 0.25294 | 713 | 5.00000 | 712 |

## Transaction Costs

Cost sensitivity file: `experiments/exp_2_lstm/hourly/data/hourly_cost_sensitivity.csv`

| period | top_k | transaction_cost | return | gross_return | benchmark_return | difference | sharpe | max_drawdown | volatility | number_of_hours | average_positions | trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 1 | 0.00000 | -0.21447 | -0.21447 | 0.08860 | -0.30307 | -0.97304 | -0.39684 | 0.46049 | 713 | 1.00000 | 0 |
| full | 3 | 0.00000 | 0.20310 | 0.20310 | 0.09755 | 0.10555 | 1.56685 | -0.11222 | 0.29991 | 713 | 3.00000 | 0 |
| full | 5 | 0.00000 | 0.18729 | 0.18729 | 0.08806 | 0.09922 | 1.68460 | -0.10853 | 0.25320 | 713 | 5.00000 | 0 |
| full | 1 | 0.00050 | -0.57534 | -0.21447 | 0.08860 | -0.66394 | -4.03585 | -0.63188 | 0.46082 | 713 | 1.00000 | 615 |
| full | 3 | 0.00050 | -0.32766 | 0.20310 | 0.09755 | -0.42521 | -2.89320 | -0.37563 | 0.29958 | 713 | 3.00000 | 707 |
| full | 5 | 0.00050 | -0.31098 | 0.18729 | 0.08806 | -0.39905 | -3.25547 | -0.35607 | 0.25294 | 713 | 5.00000 | 712 |
| full | 1 | 0.00100 | -0.77057 | -0.21447 | 0.08860 | -0.85917 | -7.08780 | -0.78391 | 0.46157 | 713 | 1.00000 | 615 |
| full | 3 | 0.00100 | -0.62446 | 0.20310 | 0.09755 | -0.72201 | -7.35605 | -0.63809 | 0.29954 | 713 | 3.00000 | 707 |
| full | 5 | 0.00100 | -0.60032 | 0.18729 | 0.08806 | -0.68838 | -8.19715 | -0.61181 | 0.25294 | 713 | 5.00000 | 712 |
| full | 1 | 0.00200 | -0.93316 | -0.21447 | 0.08860 | -1.02176 | -13.12637 | -0.93513 | 0.46433 | 713 | 1.00000 | 615 |
| full | 3 | 0.00200 | -0.88302 | 0.20310 | 0.09755 | -0.98057 | -16.23954 | -0.88685 | 0.30030 | 713 | 3.00000 | 707 |
| full | 5 | 0.00200 | -0.86569 | 0.18729 | 0.08806 | -0.95376 | -18.02275 | -0.86917 | 0.25376 | 713 | 5.00000 | 712 |

## Walk-Forward

Walk-forward file: `experiments/exp_2_lstm/hourly/data/hourly_walk_forward_summary.csv`

| period | top_k | transaction_cost | return | gross_return | benchmark_return | difference | sharpe | max_drawdown | volatility | number_of_hours | average_positions | trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WF_1_2026-02-05 14:30:00_to_2026-03-25 19:30:00 | 1 | 0.00100 | -0.41341 | -0.07284 | 0.04102 | -0.45443 | -8.56589 | -0.44235 | 0.41787 | 238 | 1.00000 | 229 |
| WF_1_2026-02-05 14:30:00_to_2026-03-25 19:30:00 | 3 | 0.00100 | -0.28540 | 0.09259 | 0.04186 | -0.32726 | -7.42008 | -0.30500 | 0.30517 | 238 | 3.00000 | 237 |
| WF_1_2026-02-05 14:30:00_to_2026-03-25 19:30:00 | 5 | 0.00100 | -0.28202 | 0.06687 | 0.03648 | -0.31850 | -8.81867 | -0.29917 | 0.25471 | 238 | 5.00000 | 238 |
| WF_2_2026-03-26 13:30:00_to_2026-05-13 19:30:00 | 1 | 0.00100 | -0.39815 | -0.10960 | 0.10355 | -0.50169 | -7.64908 | -0.42520 | 0.44346 | 238 | 1.00000 | 196 |
| WF_2_2026-03-26 13:30:00_to_2026-05-13 19:30:00 | 3 | 0.00100 | -0.29949 | 0.01668 | 0.10318 | -0.40268 | -8.05239 | -0.30266 | 0.29847 | 238 | 3.00000 | 237 |
| WF_2_2026-03-26 13:30:00_to_2026-05-13 19:30:00 | 5 | 0.00100 | -0.26832 | 0.03326 | 0.10090 | -0.36922 | -8.79701 | -0.26923 | 0.24096 | 238 | 5.00000 | 237 |
| WF_3_2026-05-14 13:30:00_to_2026-07-02 18:30:00 | 1 | 0.00100 | -0.35014 | -0.04847 | -0.05242 | -0.29772 | -5.47078 | -0.35752 | 0.51934 | 237 | 1.00000 | 191 |
| WF_3_2026-05-14 13:30:00_to_2026-07-02 18:30:00 | 3 | 0.00100 | -0.24878 | 0.08308 | -0.04508 | -0.20371 | -6.53070 | -0.25943 | 0.29586 | 237 | 3.00000 | 233 |
| WF_3_2026-05-14 13:30:00_to_2026-07-02 18:30:00 | 5 | 0.00100 | -0.23796 | 0.07704 | -0.04645 | -0.19151 | -6.99072 | -0.24882 | 0.26355 | 237 | 5.00000 | 237 |

## Alpaca Validation

- Alpaca loads hourly model files only: yes
- Findings: none

## Logging Validation

Paper Trading logs include signal timestamp, universe, benchmark, top-k, selected tickers and order actions. Performance logging has been extended to include bought tickers, sold tickers, held tickers, benchmark value and outperformance fields.

## Scaler Validation

- Scaler load without `InconsistentVersionWarning`: yes
- Warnings: none

## Code Review

- Findings: none

## Known Scientific Weaknesses

- Classification metrics are close to random.
- At realistic 0.10% buy/sell costs, all tested Top-K variants underperform the benchmark.
- Turnover is very high, so transaction costs dominate the strategy.
- Walk-forward periods are consistently negative after costs.
- The model is useful for observing hourly signal behavior, but not yet for claiming a robust profitable hourly trading edge.

## Metadata

```json
{
  "model_type": "hourly_outperformance_lstm",
  "training_timeframe": "1Hour",
  "target_definition": "1 if next-hour stock return is greater than next-hour benchmark return, else 0.",
  "benchmarks": {
    "tech_universes": "QQQ",
    "defensive_non_tech": "SPY"
  },
  "feature_group": "Technical + Market + Relative Strength",
  "feature_list": [
    "Daily_Return",
    "Lag_1_Return",
    "Lag_3_Return",
    "Lag_7_Return",
    "RollingMean_7",
    "RollingMean_30",
    "RollingVolatility_7",
    "RollingVolatility_30",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Volume_Change",
    "Volume_Ratio_20",
    "Distance_to_MA_200",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Price_Position_20",
    "High_Low_Range",
    "QQQ_Return",
    "SPY_Return",
    "VIX_Change",
    "QQQ_Momentum_20",
    "SPY_Momentum_20",
    "QQQ_Distance_to_MA200",
    "SPY_Distance_to_MA200",
    "Relative_Return_QQQ",
    "Relative_Return_SPY",
    "Relative_Momentum_20_QQQ",
    "Relative_Momentum_20_SPY"
  ],
  "sequence_length": 20,
  "training_start": "2023-09-14 16:30:00",
  "training_end": "2026-07-02 18:30:00",
  "train_rows": 102510,
  "validation_rows": 21935,
  "test_rows": 21888,
  "metrics": {
    "accuracy": 0.5002345435781969,
    "precision": 0.49735961978524906,
    "recall": 0.5334151406456484,
    "f1_score": 0.5147567862998724,
    "predicted_outperform_share": 0.5329768270944741,
    "confusion_true_0_pred_0": 5013,
    "confusion_true_0_pred_1": 5711,
    "confusion_true_1_pred_0": 4943,
    "confusion_true_1_pred_1": 5651
  }
}
```

## Final Assessment

Not ready as a final trading strategy commit. It can be committed only as an experimental validation result, and Alpaca should be limited to signals-only or very cautious paper observation.

Best cost-adjusted variant at 0.10% costs:

- Top-K: 5
- Strategy return: -0.60032
- Benchmark return: 0.08806
- Difference: -0.68838
