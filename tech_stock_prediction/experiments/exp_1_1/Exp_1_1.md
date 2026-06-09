# Experiment 1.1

### Problem Definition:

**Target**

Prediction of the next trading day's price direction for large tech stocks.

For every selected stock and every trading day from `2019-01-01`, the target is:

- `1`: next trading day's close price is higher than today's close price
- `0`: next trading day's close price is lower than or equal to today's close price

**Input Features**

- Daily return
- Lagged returns over 1, 3 and 7 trading days
- Rolling mean over 7 and 30 trading days
- Rolling volatility over 7 and 30 trading days
- RSI over 14 trading days
- MACD and MACD signal line
- Volume change and volume ratio
- Trend and momentum indicators

Fundamental and earnings features were tested in the feature group analysis,
but they are not used in the final model because they did not improve the
non-trivial model performance.

### Procedure Overview:

- Collects daily price data for selected large tech stocks from Yahoo Finance.
- Engineers technical and time-based features separately for each ticker.
- Splits the data chronologically into train, validation and test sets.
- Trains a simple Random Forest baseline model.
- Evaluates the model on validation and test data.
- Runs a simple backtest that only holds a stock when the model predicts an upward movement.

**Hypothesis**

Technical indicators and time-based features can predict short-term price movements of large tech stocks better than a simple buy-and-hold or random baseline.

---

## Data Acquisition

Retrieves daily Yahoo Finance market data for the selected tech stocks.

**Script**

[scripts/01_data_acquisition/bar_retriever.py](scripts/01_data_acquisition/bar_retriever.py)

The script writes one combined CSV file to:

[data/raw/tech_stocks_raw.csv](data/raw/tech_stocks_raw.csv)

Each row contains a `Ticker` column so that the stocks can be separated again during feature engineering.

**Ticker Universe**

- AAPL
- MSFT
- NVDA
- AMD
- GOOGL
- META
- AMZN
- TSLA
- INTC
- ADBE

---

## Step 2 - Data Understanding

Loads the raw CSV file and prints basic information about the dataset.

**Script**

[scripts/02_data_understanding/plotter.py](scripts/02_data_understanding/plotter.py)

The script prints:

- number of rows and columns
- date range
- included tickers
- missing values
- descriptive statistics
- number of rows per ticker

---

## Step 3 - Pre-Split Preparation

Creates the features and the target before splitting the data. Feature engineering is done per ticker to avoid mixing information between different stocks.

**Main Script**

[scripts/03_pre_split_prep/main.py](scripts/03_pre_split_prep/main.py)

**Feature Engineering Script**

[scripts/03_pre_split_prep/features3.py](scripts/03_pre_split_prep/features3.py)

**Target Computation Script**

[scripts/03_pre_split_prep/targets.py](scripts/03_pre_split_prep/targets.py)

**Feature List**

[scripts/03_pre_split_prep/features.txt](scripts/03_pre_split_prep/features.txt)

**Data**

[data/processed/tech_stocks_features.csv](data/processed/tech_stocks_features.csv)

Rows with missing values from rolling windows and the last row per ticker are removed.

---

## Step 4 - Split Data

Splits the feature data chronologically into train, validation and test sets.

**Script**

[scripts/04_split_data/split.py](scripts/04_split_data/split.py)

The split is not random because stock price data is time series data. A random split could leak future information into the training set.

**Split**

- 70% training
- 15% validation
- 15% test

**Data**

- [data/processed/train.csv](data/processed/train.csv)
- [data/processed/validation.csv](data/processed/validation.csv)
- [data/processed/test.csv](data/processed/test.csv)

---

## Step 5 - Post-Split Preparation

The selected model features are normalized per ticker after the chronological
split. The mean and standard deviation are calculated only on the training set
and then applied to validation and test data. This avoids data leakage and helps
compare stocks with different volatility levels.

---

## Step 6 - Model Training

Trains a `RandomForestClassifier` on the training set and evaluates it on the validation set.

**Script**

[scripts/05_model_training/train_random_forest.py](scripts/05_model_training/train_random_forest.py)

**Model**

[models/random_forest_baseline.pkl](models/random_forest_baseline.pkl)

**Metrics**

- Accuracy
- Precision
- Recall
- F1-Score

---

## Step 7 - Model Testing

Loads the saved model and evaluates it on the test set.

**Script**

[scripts/06_model_testing/evaluate_random_forest.py](scripts/06_model_testing/evaluate_random_forest.py)

**Data**

[data/processed/test_predictions.csv](data/processed/test_predictions.csv)

**Metrics**

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Step 8 - Backtesting

Runs a simple strategy on the test predictions.

**Script**

[scripts/07_backtesting/simple_backtest.py](scripts/07_backtesting/simple_backtest.py)

**Strategy**

- If `Prediction = 1`, hold the stock for the next trading day.
- If `Prediction = 0`, do not hold the stock.

The strategy return is compared with a simple equal-weighted buy-and-hold return.

**Data**

[data/processed/backtest_results.csv](data/processed/backtest_results.csv)

---

## Step 8 - Dummy Baseline

Prints a naive classification baseline that always predicts `1` (stock rises).
Because the market rises slightly more often than it falls, this trivial
strategy already reaches an accuracy equal to the share of rising days
(about 52% on the test set).

**Script**

[scripts/08_baseline/dummy_baseline.py](scripts/08_baseline/dummy_baseline.py)

The script prints, at the end of every pipeline run, the Always-Yes baseline
metrics (Accuracy, Precision, Recall, F1-Score) directly next to the Random
Forest metrics on the same test set. This makes it obvious whether the model
actually beats a model that always says "up".

---

## Step 9 - Visualization

Creates presentation-ready plots from the generated data, predictions and
backtest results.

**Script**

[scripts/09_visualization/generate_all_plots.py](scripts/09_visualization/generate_all_plots.py)

The plots are saved to:

[plots](plots)

---

## Step 10 - Feature Group Analysis

Tests which feature groups improve the model and which ones make it worse.
The script tests all non-empty combinations of the five feature groups:
technical, volume, trend/momentum, fundamental and earnings features. This
creates 31 Random Forest runs with the same model settings, so the feature
sets can be compared fairly.

**Script**

[scripts/10_feature_analysis/feature_group_analysis.py](scripts/10_feature_analysis/feature_group_analysis.py)

**Outputs**

- [data/processed/feature_group_results.csv](data/processed/feature_group_results.csv)
- [data/processed/feature_importance_results.csv](data/processed/feature_importance_results.csv)
- [plots/16_feature_group_comparison.png](plots/16_feature_group_comparison.png)
- [plots/18_feature_importance_top15.png](plots/18_feature_importance_top15.png)

This makes it easier to explain whether new features actually improved the
model or only added noise. The result table also includes the share of
`Prediction = 1`, so we can detect models that look good only because they
almost always predict rising prices.

---

## Configuration

The experiment configuration is stored in:

[conf/params.yaml](conf/params.yaml)

The config contains:

- ticker list
- start date
- file paths
- split ratios
- model parameters
- target name

---

## Run Order

Run the scripts from the experiment folder:

```bash
cd experiments/exp_1_1
python scripts/01_data_acquisition/bar_retriever.py
python scripts/02_data_understanding/plotter.py
python scripts/03_pre_split_prep/main.py
python scripts/04_split_data/split.py
python scripts/05_model_training/train_random_forest.py
python scripts/06_model_testing/evaluate_random_forest.py
python scripts/07_backtesting/simple_backtest.py
python scripts/08_baseline/dummy_baseline.py
python scripts/09_visualization/generate_all_plots.py
python scripts/10_feature_analysis/feature_group_analysis.py
```

---

## Next Steps

- Evaluate performance separately for each ticker.
- Add transaction costs to the backtest.
- Compare Random Forest with Logistic Regression.
- Compare the final feature set with Logistic Regression or another simple model.
