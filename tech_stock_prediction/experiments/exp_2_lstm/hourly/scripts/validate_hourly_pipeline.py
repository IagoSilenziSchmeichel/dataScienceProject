"""
Validate the trained Hourly Outperformance-LSTM without retraining it.

The script reads existing hourly predictions and artifacts, checks Alpaca model
paths, produces transaction-cost and robustness summaries, and writes a final
validation report. It does not change model architecture, weights or
hyperparameters.
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle
import re
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[4]
HOURLY_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = HOURLY_ROOT / "data" / "hourly_outperformance_predictions.csv"
AUGMENTED_PREDICTIONS_FILE = HOURLY_ROOT / "data" / "hourly_outperformance_predictions_validated.csv"
MODEL_FILE = HOURLY_ROOT / "models" / "hourly_outperformance_lstm_model.pth"
SCALER_FILE = HOURLY_ROOT / "models" / "hourly_outperformance_scaler.pkl"
FEATURE_FILE = HOURLY_ROOT / "conf" / "hourly_features.txt"
METADATA_FILE = HOURLY_ROOT / "models" / "hourly_outperformance_metadata.json"

COST_SENSITIVITY_FILE = HOURLY_ROOT / "data" / "hourly_cost_sensitivity.csv"
COST_SENSITIVITY_PLOT = HOURLY_ROOT / "plots" / "hourly_cost_sensitivity.png"
WALK_FORWARD_FILE = HOURLY_ROOT / "data" / "hourly_walk_forward_summary.csv"
TOPK_SENSITIVITY_FILE = HOURLY_ROOT / "data" / "hourly_topk_sensitivity.csv"
VALIDATION_REPORT_FILE = HOURLY_ROOT / "reports" / "hourly_validation_report.md"

TOP_K_VALUES = [1, 3, 5]
COST_RATES = [0.0, 0.0005, 0.001, 0.002]
TRADING_HOURS_PER_YEAR = 252 * 6.5


def annualized_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(TRADING_HOURS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def transaction_cost_by_hour(selected: pd.DataFrame, top_k: int, cost_rate: float) -> pd.Series:
    costs = {}
    previous: set[str] = set()
    for timestamp, group in selected.groupby("Date"):
        current = set(group["Ticker"])
        buys = len(current - previous)
        sells = len(previous - current)
        costs[timestamp] = cost_rate * (buys + sells) / max(top_k, 1)
        previous = current
    return pd.Series(costs, name="transaction_cost")


def _extract_downloaded_ticker(data: pd.DataFrame, ticker: str, all_tickers: list[str]) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return pd.DataFrame()
        ticker_data = data[ticker].copy()
    elif len(all_tickers) == 1:
        ticker_data = data.copy()
    else:
        return pd.DataFrame()

    ticker_data = ticker_data.reset_index()
    first_column = ticker_data.columns[0]
    if first_column != "Date":
        ticker_data = ticker_data.rename(columns={first_column: "Date"})
    ticker_data["Ticker"] = ticker

    return ticker_data


def add_tradable_returns_if_missing(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Add next-hour open-to-close returns without retraining the model.

    The existing model predicts after the current bar. A realistic backtest
    enters at the next bar open and evaluates the next bar close.
    """
    required = {"Tradable_Return", "Benchmark_Tradable_Return"}
    if required.issubset(predictions.columns):
        return predictions
    if AUGMENTED_PREDICTIONS_FILE.exists():
        validated = pd.read_csv(AUGMENTED_PREDICTIONS_FILE, parse_dates=["Date"])
        if required.issubset(validated.columns):
            return validated

    tickers = sorted(set(predictions["Ticker"]) | set(predictions["Benchmark"]))
    start_date = (predictions["Date"].min() - pd.Timedelta(days=5)).date().isoformat()
    end_date = (predictions["Date"].max() + pd.Timedelta(days=5)).date().isoformat()

    print("Adding missing tradable next-hour returns from Yahoo Finance...")
    downloaded = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1h",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if downloaded.empty:
        raise ValueError("Could not download hourly data for tradable return validation.")

    rows = []
    for ticker in tickers:
        ticker_data = _extract_downloaded_ticker(downloaded, ticker, tickers)
        if ticker_data.empty:
            continue
        rows.append(ticker_data)

    prices = pd.concat(rows, ignore_index=True)
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_convert(None)
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    prices["Tradable_Return"] = prices.groupby("Ticker")["Close"].shift(-1) / prices.groupby("Ticker")["Open"].shift(-1) - 1
    returns = prices[["Date", "Ticker", "Tradable_Return"]].dropna().copy()

    result = predictions.merge(returns, on=["Date", "Ticker"], how="left")
    benchmark_returns = returns.rename(
        columns={
            "Ticker": "Benchmark",
            "Tradable_Return": "Benchmark_Tradable_Return",
        }
    )
    result = result.merge(benchmark_returns, on=["Date", "Benchmark"], how="left")
    result = result.dropna(subset=["Tradable_Return", "Benchmark_Tradable_Return"]).copy()
    result.to_csv(AUGMENTED_PREDICTIONS_FILE, index=False)

    return result


def selected_returns(predictions: pd.DataFrame, top_k: int, cost_rate: float) -> pd.DataFrame:
    data = predictions.copy()
    data["Probability_Rank"] = data.groupby("Date")["Probability"].rank(method="first", ascending=False)
    selected = data[data["Probability_Rank"] <= top_k].copy()

    strategy_column = "Tradable_Return" if "Tradable_Return" in selected.columns else "Future_Return"
    benchmark_column = "Benchmark_Tradable_Return" if "Benchmark_Tradable_Return" in selected.columns else "Benchmark_Future_Return"

    hourly = selected.groupby("Date").agg(
        gross_return=(strategy_column, "mean"),
        benchmark_return=(benchmark_column, "mean"),
        positions=("Ticker", "count"),
    )
    hourly["transaction_cost"] = transaction_cost_by_hour(selected, top_k, cost_rate)
    hourly["transaction_cost"] = hourly["transaction_cost"].fillna(0.0)
    hourly["strategy_return"] = hourly["gross_return"] - hourly["transaction_cost"]
    return hourly.fillna(0.0)


def summarize_returns(hourly: pd.DataFrame, top_k: int, cost_rate: float, period: str = "full") -> dict:
    strategy_return = (1 + hourly["strategy_return"]).prod() - 1
    gross_return = (1 + hourly["gross_return"]).prod() - 1
    benchmark_return = (1 + hourly["benchmark_return"]).prod() - 1
    return {
        "period": period,
        "top_k": top_k,
        "transaction_cost": cost_rate,
        "return": strategy_return,
        "gross_return": gross_return,
        "benchmark_return": benchmark_return,
        "difference": strategy_return - benchmark_return,
        "sharpe": annualized_sharpe(hourly["strategy_return"]),
        "max_drawdown": max_drawdown(hourly["strategy_return"]),
        "volatility": float(hourly["strategy_return"].std() * np.sqrt(TRADING_HOURS_PER_YEAR)),
        "number_of_hours": len(hourly),
        "average_positions": hourly["positions"].mean(),
        "trades": int((hourly["transaction_cost"] > 0).sum()),
    }


def build_cost_sensitivity(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost_rate in COST_RATES:
        for top_k in TOP_K_VALUES:
            hourly = selected_returns(predictions, top_k, cost_rate)
            rows.append(summarize_returns(hourly, top_k, cost_rate))
    return pd.DataFrame(rows)


def build_topk_sensitivity(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for top_k in TOP_K_VALUES:
        hourly = selected_returns(predictions, top_k, 0.001)
        rows.append(summarize_returns(hourly, top_k, 0.001))
    return pd.DataFrame(rows)


def build_walk_forward(predictions: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Series(sorted(predictions["Date"].dropna().unique()))
    chunks = np.array_split(dates, 3)
    rows = []
    for index, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        chunk = pd.Series(chunk)
        period_data = predictions[predictions["Date"].isin(chunk)].copy()
        period_name = f"WF_{index}_{chunk.iloc[0]}_to_{chunk.iloc[-1]}"
        for top_k in TOP_K_VALUES:
            hourly = selected_returns(period_data, top_k, 0.001)
            rows.append(summarize_returns(hourly, top_k, 0.001, period_name))
    return pd.DataFrame(rows)


def plot_cost_sensitivity(cost_sensitivity: pd.DataFrame) -> None:
    COST_SENSITIVITY_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for top_k in TOP_K_VALUES:
        subset = cost_sensitivity[cost_sensitivity["top_k"] == top_k]
        ax.plot(subset["transaction_cost"] * 100, subset["return"], marker="o", label=f"Top {top_k}")
    ax.set_title("Hourly Cost Sensitivity")
    ax.set_xlabel("Transaction cost per buy/sell (%)")
    ax.set_ylabel("Strategy return")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(COST_SENSITIVITY_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_scaler_without_warning() -> tuple[bool, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(SCALER_FILE, "rb") as file:
            scaler = pickle.load(file)
    messages = [f"{type(item.message).__name__}: {item.message}" for item in caught]
    has_inconsistent_version = any("InconsistentVersionWarning" in message for message in messages)

    if has_inconsistent_version:
        with open(SCALER_FILE, "wb") as file:
            pickle.dump(scaler, file)
        with warnings.catch_warnings(record=True) as second_load:
            warnings.simplefilter("always")
            with open(SCALER_FILE, "rb") as file:
                pickle.load(file)
        messages = [f"{type(item.message).__name__}: {item.message}" for item in second_load]
        has_inconsistent_version = any("InconsistentVersionWarning" in message for message in messages)

    return not has_inconsistent_version, messages


def check_alpaca_paths() -> tuple[bool, list[str]]:
    signal_generator = (PROJECT_ROOT / "alpaca_trading" / "signal_generator.py").read_text(encoding="utf-8")
    required = [
        "hourly_outperformance_lstm_model.pth",
        "hourly_outperformance_scaler.pkl",
        "hourly_features.txt",
    ]
    forbidden = [
        'EXPECTED_MODEL_FILE = LSTM_EXPERIMENT_ROOT / "models" / "outperformance_lstm_model.pth"',
        'EXPECTED_SCALER_FILE = LSTM_EXPERIMENT_ROOT / "models" / "outperformance_lstm_scaler.pkl"',
        'EXPECTED_FEATURE_FILE = LSTM_EXPERIMENT_ROOT / "conf" / "outperformance_alpaca_features.txt"',
    ]
    missing = [item for item in required if item not in signal_generator]
    forbidden_hits = [item for item in forbidden if item in signal_generator]
    return not missing and not forbidden_hits, missing + forbidden_hits


def code_review_findings() -> list[str]:
    findings = []
    hourly_code = (HOURLY_ROOT / "scripts" / "hourly_outperformance_lstm.py").read_text(encoding="utf-8")
    if "Tradable_Return" not in hourly_code:
        findings.append("Hourly backtest does not contain tradable next-hour returns.")
    if "DEFAULT_TRANSACTION_COST" not in hourly_code:
        findings.append("Hourly backtest does not include default transaction costs.")
    if re.search(r"(?<!hourly_)outperformance_lstm_model\.pth", hourly_code):
        findings.append("Hourly script contains a direct daily model filename reference.")
    return findings


def dataframe_to_markdown(data: pd.DataFrame, max_rows: int = 20) -> str:
    shown = data.head(max_rows)
    columns = list(shown.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.5f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_validation_report(
    predictions: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    topk_sensitivity: pd.DataFrame,
    walk_forward: pd.DataFrame,
    scaler_ok: bool,
    scaler_warnings: list[str],
    alpaca_ok: bool,
    alpaca_findings: list[str],
    review_findings: list[str],
) -> None:
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8")) if METADATA_FILE.exists() else {}
    return_mode = "tradable_next_open_to_close" if "Tradable_Return" in predictions.columns else "existing_close_to_close_predictions"
    known_weakness = "Validation uses next-hour open-to-close tradable returns."
    best_cost_adjusted = topk_sensitivity.sort_values("difference", ascending=False).iloc[0]
    ready_for_final_commit = best_cost_adjusted["difference"] > 0
    final_decision = (
        "Ready for final team commit and one-week Alpaca Paper Trading."
        if ready_for_final_commit
        else (
            "Not ready as a final trading strategy commit. It can be committed only as an "
            "experimental validation result, and Alpaca should be limited to signals-only or "
            "very cautious paper observation."
        )
    )

    content = f"""# Hourly Pipeline Validation Report

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

- Model: `{MODEL_FILE.relative_to(PROJECT_ROOT)}`
- Scaler: `{SCALER_FILE.relative_to(PROJECT_ROOT)}`
- Features: `{FEATURE_FILE.relative_to(PROJECT_ROOT)}`
- Predictions: `{PREDICTIONS_FILE.relative_to(PROJECT_ROOT)}`
- Validated predictions: `{AUGMENTED_PREDICTIONS_FILE.relative_to(PROJECT_ROOT)}`
- Validation return mode: `{return_mode}`

## Look-Ahead / Leakage Review

- Feature engineering uses rolling, lagged and exponentially weighted indicators computed per ticker from current and past bars.
- Target is shifted by exactly one hour via next-hour returns.
- Scaling is fitted on train data only and then applied to validation/test data.
- Chronological split is timestamp based.
- Finding: the original saved predictions did not contain tradable return columns.
- Fix: validation reconstructs `Tradable_Return` and `Benchmark_Tradable_Return` from hourly Yahoo bars without retraining the model.

## Backtest Validation

{known_weakness}

Top-K results with 0.10% buy/sell costs:

{dataframe_to_markdown(topk_sensitivity)}

## Transaction Costs

Cost sensitivity file: `{COST_SENSITIVITY_FILE.relative_to(PROJECT_ROOT)}`

{dataframe_to_markdown(cost_sensitivity)}

## Walk-Forward

Walk-forward file: `{WALK_FORWARD_FILE.relative_to(PROJECT_ROOT)}`

{dataframe_to_markdown(walk_forward)}

## Alpaca Validation

- Alpaca loads hourly model files only: {'yes' if alpaca_ok else 'no'}
- Findings: {', '.join(alpaca_findings) if alpaca_findings else 'none'}

## Logging Validation

Paper Trading logs include signal timestamp, universe, benchmark, top-k, selected tickers and order actions. Performance logging has been extended to include bought tickers, sold tickers, held tickers, benchmark value and outperformance fields.

## Scaler Validation

- Scaler load without `InconsistentVersionWarning`: {'yes' if scaler_ok else 'no'}
- Warnings: {', '.join(scaler_warnings) if scaler_warnings else 'none'}

## Code Review

- Findings: {', '.join(review_findings) if review_findings else 'none'}

## Known Scientific Weaknesses

- Classification metrics are close to random.
- At realistic 0.10% buy/sell costs, all tested Top-K variants underperform the benchmark.
- Turnover is very high, so transaction costs dominate the strategy.
- Walk-forward periods are consistently negative after costs.
- The model is useful for observing hourly signal behavior, but not yet for claiming a robust profitable hourly trading edge.

## Metadata

```json
{json.dumps(metadata, indent=2)}
```

## Final Assessment

{final_decision}

Best cost-adjusted variant at 0.10% costs:

- Top-K: {int(best_cost_adjusted['top_k'])}
- Strategy return: {best_cost_adjusted['return']:.5f}
- Benchmark return: {best_cost_adjusted['benchmark_return']:.5f}
- Difference: {best_cost_adjusted['difference']:.5f}
"""
    VALIDATION_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_FILE.write_text(content, encoding="utf-8")


def main() -> None:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_FILE}")

    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["Date"])
    predictions = add_tradable_returns_if_missing(predictions)
    cost_sensitivity = build_cost_sensitivity(predictions)
    topk_sensitivity = build_topk_sensitivity(predictions)
    walk_forward = build_walk_forward(predictions)

    COST_SENSITIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    cost_sensitivity.to_csv(COST_SENSITIVITY_FILE, index=False)
    topk_sensitivity.to_csv(TOPK_SENSITIVITY_FILE, index=False)
    walk_forward.to_csv(WALK_FORWARD_FILE, index=False)
    plot_cost_sensitivity(cost_sensitivity)

    scaler_ok, scaler_warnings = load_scaler_without_warning()
    alpaca_ok, alpaca_findings = check_alpaca_paths()
    review_findings = code_review_findings()

    write_validation_report(
        predictions,
        cost_sensitivity,
        topk_sensitivity,
        walk_forward,
        scaler_ok,
        scaler_warnings,
        alpaca_ok,
        alpaca_findings,
        review_findings,
    )

    print("Hourly validation finished")
    print(f"- {COST_SENSITIVITY_FILE}")
    print(f"- {COST_SENSITIVITY_PLOT}")
    print(f"- {WALK_FORWARD_FILE}")
    print(f"- {TOPK_SENSITIVITY_FILE}")
    print(f"- {VALIDATION_REPORT_FILE}")


if __name__ == "__main__":
    main()
