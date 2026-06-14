"""
Compare exp_1_3_market_features with the exp_1_1 baseline.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import pandas as pd
from formatting import format_decimal, format_percent


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = PROJECT_ROOT / "experiments" / "exp_1_1"
MARKET_ROOT = PROJECT_ROOT / "experiments" / "exp_1_3_market_features"


def load_first_row(path):
    if not path.exists():
        print(f"Missing file: {path}")
        return None
    data = pd.read_csv(path)
    if data.empty:
        print(f"Empty file: {path}")
        return None
    return data.iloc[0]


def print_metric_row(metric_name, baseline_value, market_value, is_percent=True):
    if pd.isna(baseline_value) or pd.isna(market_value):
        return

    if is_percent:
        baseline_text = format_percent(baseline_value)
        market_text = format_percent(market_value)
        diff_text = format_percent(market_value - baseline_value)
    else:
        baseline_text = format_decimal(baseline_value)
        market_text = format_decimal(market_value)
        diff_text = format_decimal(market_value - baseline_value)

    print(f"{metric_name:<28}{baseline_text:>16}{market_text:>16}{diff_text:>16}")


def main():
    print("Compare exp_1_1 vs exp_1_3_market_features")
    print("===========================================")

    baseline = load_first_row(BASELINE_ROOT / "data" / "processed" / "backtest_results.csv")
    market = load_first_row(MARKET_ROOT / "data" / "processed" / "backtest_results.csv")

    if baseline is None or market is None:
        print("Run both experiments before comparing.")
        return

    metrics = [
        ("Strategy_Total_Return", True),
        ("Buy_Hold_Total_Return", True),
        ("Difference", True),
        ("Strategy_CAGR", True),
        ("Buy_Hold_CAGR", True),
        ("Strategy_Sharpe", False),
        ("Buy_Hold_Sharpe", False),
        ("Strategy_Max_Drawdown", True),
        ("Buy_Hold_Max_Drawdown", True),
        ("Strategy_Volatility", True),
        ("Buy_Hold_Volatility", True),
    ]

    print(f"{'Metric':<28}{'exp_1_1':>16}{'exp_1_3':>16}{'Delta':>16}")
    for metric_name, is_percent in metrics:
        if metric_name not in baseline.index or metric_name not in market.index:
            continue
        print_metric_row(
            metric_name,
            float(baseline[metric_name]),
            float(market[metric_name]),
            is_percent=is_percent,
        )


if __name__ == "__main__":
    main()
