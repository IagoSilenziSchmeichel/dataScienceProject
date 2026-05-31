from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "test_predictions.csv"
BACKTEST_PATH = PROCESSED_DATA_DIR / "backtest_results.csv"


def main():
    data = pd.read_csv(PREDICTIONS_PATH, parse_dates=["Date"])
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Rendite vom heutigen Close zum naechsten Handelstag.
    data["Next_Day_Return"] = data.groupby("Ticker")["Close"].shift(-1) / data["Close"] - 1

    # Strategie: Nur investieren, wenn das Modell steigende Kurse erwartet.
    data["Strategy_Return"] = data["Prediction"] * data["Next_Day_Return"]
    data["Buy_And_Hold_Return"] = data["Next_Day_Return"]

    backtest_data = data.dropna(subset=["Next_Day_Return"]).copy()

    strategy_total_return = (1 + backtest_data["Strategy_Return"]).prod() - 1
    buy_and_hold_total_return = (1 + backtest_data["Buy_And_Hold_Return"]).prod() - 1

    print("Simple Backtest")
    print("===============")
    print(f"Strategy Return:      {strategy_total_return:.2%}")
    print(f"Buy-and-Hold Return:  {buy_and_hold_total_return:.2%}")
    print(f"Differenz:            {strategy_total_return - buy_and_hold_total_return:.2%}")

    backtest_data.to_csv(BACKTEST_PATH, index=False)
    print(f"\nBacktest-Ergebnisse gespeichert unter: {BACKTEST_PATH}")


if __name__ == "__main__":
    main()
