"""
Compare LSTM result files in one clean summary.

The summary contains native-period results and common-overlap results.
Native-period results compare each model with Buy-and-Hold on its own test
dates. Common-overlap results compare all models on the dates all models share.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import numpy as np
import pandas as pd
import yaml
from formatting import format_decimal, format_percent, save_csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))
TOP_K_VALUES = PARAMS["MODEL"].get("TOP_K_VALUES", [1, 2, 3, 4, 5])


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path)


def load_test_next_close():
    test_file = EXPERIMENT_ROOT / PARAMS["DATA"]["TEST_FILE"]
    test_data = pd.read_csv(test_file, parse_dates=["Date"])
    test_data = test_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    test_data["Next_Close"] = test_data.groupby("Ticker")["Close"].shift(-1)

    return test_data[["Date", "Ticker", "Next_Close"]]


def prepare_predictions(predictions, test_next_close):
    data = predictions.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    if "Actual" not in data.columns and "Target" in data.columns:
        data["Actual"] = data["Target"]

    if "Next_Close" not in data.columns:
        data = data.merge(test_next_close, on=["Date", "Ticker"], how="left")

    data = data.dropna(subset=["Next_Close"]).copy()
    data["Future_Return"] = data["Next_Close"] / data["Close"] - 1

    return data.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def calculate_simple_result(data, model_name, period_type):
    y_true = data["Actual"].astype(int)
    y_pred = data["Prediction"].astype(int)

    result_data = data[["Date", "Ticker", "Future_Return"]].copy()
    result_data["Strategy_Return"] = np.where(y_pred == 1, result_data["Future_Return"], 0.0)
    result_data["Buy_And_Hold_Return"] = result_data["Future_Return"]

    daily_returns = result_data.groupby("Date")[["Strategy_Return", "Buy_And_Hold_Return"]].mean()
    strategy_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
    buy_and_hold_return = (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1

    return {
        "model_name": model_name,
        "period_type": period_type,
        "decision_rule": "prediction_1",
        "top_k": np.nan,
        "test_start": daily_returns.index.min().date().isoformat(),
        "test_end": daily_returns.index.max().date().isoformat(),
        "number_of_trading_days": len(daily_returns),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "predicted_up_share": y_pred.mean(),
        "strategy_return": strategy_return,
        "buy_and_hold_return": buy_and_hold_return,
        "difference": strategy_return - buy_and_hold_return,
    }


def calculate_top_k_result(data, model_name, period_type, top_k):
    ranked = data.copy()
    ranked["Probability_Rank"] = (
        ranked
        .groupby("Date")["Probability"]
        .rank(method="first", ascending=False)
    )
    selected = ranked[ranked["Probability_Rank"] <= top_k].copy()

    daily_strategy_returns = selected.groupby("Date")["Future_Return"].mean()
    daily_buy_hold_returns = ranked.groupby("Date")["Future_Return"].mean()

    daily_returns = pd.concat(
        [
            daily_strategy_returns.rename("Strategy_Return"),
            daily_buy_hold_returns.rename("Buy_And_Hold_Return"),
        ],
        axis=1,
    ).fillna(0)

    strategy_return = (1 + daily_returns["Strategy_Return"]).prod() - 1
    buy_and_hold_return = (1 + daily_returns["Buy_And_Hold_Return"]).prod() - 1

    if strategy_return > 1.0:
        print(
            f"Warning: {model_name} Top {top_k} result is very high. "
            "Check whether this result is dominated by a short period or single stock."
        )

    return {
        "model_name": model_name,
        "period_type": period_type,
        "decision_rule": "top_k_probability",
        "top_k": top_k,
        "test_start": daily_returns.index.min().date().isoformat(),
        "test_end": daily_returns.index.max().date().isoformat(),
        "number_of_trading_days": len(daily_returns),
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": np.nan,
        "predicted_up_share": np.nan,
        "strategy_return": strategy_return,
        "buy_and_hold_return": buy_and_hold_return,
        "difference": strategy_return - buy_and_hold_return,
    }


def add_native_result_rows(rows, standard_results, standard_top_k, tuned_results, tuned_top_k):
    standard = standard_results.iloc[0]
    rows.append(
        {
            "model_name": standard["model_name"],
            "period_type": "native_period",
            "decision_rule": "prediction_1",
            "top_k": np.nan,
            "test_start": standard["test_start"],
            "test_end": standard["test_end"],
            "number_of_trading_days": standard["number_of_trading_days"],
            "accuracy": standard["accuracy"],
            "precision": standard["precision"],
            "recall": standard["recall"],
            "f1_score": standard["f1_score"],
            "predicted_up_share": standard["predicted_up_share"],
            "strategy_return": standard["simple_strategy_return"],
            "buy_and_hold_return": standard["buy_and_hold_return"],
            "difference": standard["difference"],
        }
    )

    best_standard_top_k = standard_top_k.sort_values("strategy_return", ascending=False).iloc[0]
    rows.append(
        {
            "model_name": "standard_lstm",
            "period_type": "native_period",
            "decision_rule": "best_top_k_probability",
            "top_k": best_standard_top_k["top_k"],
            "test_start": best_standard_top_k["test_start"],
            "test_end": best_standard_top_k["test_end"],
            "number_of_trading_days": best_standard_top_k["number_of_trading_days"],
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "predicted_up_share": np.nan,
            "strategy_return": best_standard_top_k["strategy_return"],
            "buy_and_hold_return": best_standard_top_k["buy_and_hold_return"],
            "difference": best_standard_top_k["difference"],
        }
    )

    for _, tuned in tuned_results.iterrows():
        rows.append(
            {
                "model_name": tuned["model_name"],
                "period_type": "native_period",
                "decision_rule": "prediction_1",
                "top_k": np.nan,
                "test_start": tuned["test_start"],
                "test_end": tuned["test_end"],
                "number_of_trading_days": tuned["number_of_trading_days"],
                "accuracy": tuned["test_accuracy"],
                "precision": tuned["test_precision"],
                "recall": tuned["test_recall"],
                "f1_score": tuned["test_f1_score"],
                "predicted_up_share": tuned["test_predicted_up_share"],
                "strategy_return": tuned["simple_strategy_return"],
                "buy_and_hold_return": tuned["buy_and_hold_return"],
                "difference": tuned["simple_difference"],
            }
        )

    for selection_method, group in tuned_top_k.groupby("selection_method"):
        best_top_k = group.sort_values("strategy_return", ascending=False).iloc[0]
        rows.append(
            {
                "model_name": f"tuned_lstm_{selection_method}",
                "period_type": "native_period",
                "decision_rule": "best_top_k_probability",
                "top_k": best_top_k["top_k"],
                "test_start": best_top_k["test_start"],
                "test_end": best_top_k["test_end"],
                "number_of_trading_days": best_top_k["number_of_trading_days"],
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1_score": np.nan,
                "predicted_up_share": np.nan,
                "strategy_return": best_top_k["strategy_return"],
                "buy_and_hold_return": best_top_k["buy_and_hold_return"],
                "difference": best_top_k["difference"],
            }
        )


def add_common_period_rows(rows, prediction_sets):
    common_dates = None

    for data in prediction_sets.values():
        dates = set(pd.to_datetime(data["Date"]).dt.date)
        common_dates = dates if common_dates is None else common_dates.intersection(dates)

    if not common_dates:
        print("Warning: no common overlap period found.")
        return

    print(f"Common overlap period: {min(common_dates)} to {max(common_dates)}")

    buy_and_hold_values = []

    for model_name, data in prediction_sets.items():
        filtered = data[pd.to_datetime(data["Date"]).dt.date.isin(common_dates)].copy()
        simple_row = calculate_simple_result(filtered, model_name, "common_overlap_period")
        rows.append(simple_row)
        buy_and_hold_values.append(round(simple_row["buy_and_hold_return"], 10))

        top_k_rows = [
            calculate_top_k_result(filtered, model_name, "common_overlap_period", top_k)
            for top_k in TOP_K_VALUES
        ]
        best_top_k = sorted(top_k_rows, key=lambda row: row["strategy_return"], reverse=True)[0]
        best_top_k["decision_rule"] = "best_top_k_probability"
        rows.append(best_top_k)

    if len(set(buy_and_hold_values)) > 1:
        print(
            "Warning: Buy-and-Hold differs inside common overlap comparison. "
            "Please check date alignment."
        )


def print_summary(summary):
    print("LSTM model comparison summary")
    print("=============================")
    print(
        summary.to_string(
            index=False,
            formatters={
                "accuracy": lambda value: "" if pd.isna(value) else format_decimal(float(value)),
                "precision": lambda value: "" if pd.isna(value) else format_decimal(float(value)),
                "recall": lambda value: "" if pd.isna(value) else format_decimal(float(value)),
                "f1_score": lambda value: "" if pd.isna(value) else format_decimal(float(value)),
                "predicted_up_share": lambda value: "" if pd.isna(value) else format_decimal(float(value)),
                "strategy_return": lambda value: format_percent(float(value)),
                "buy_and_hold_return": lambda value: format_percent(float(value)),
                "difference": lambda value: format_percent(float(value)),
            },
        )
    )


def main():
    print("Comparing LSTM results")
    print("======================")

    standard_results = load_csv(EXPERIMENT_ROOT / PARAMS["RESULTS"]["STANDARD_RESULTS_FILE"])
    standard_top_k = load_csv(EXPERIMENT_ROOT / PARAMS["RESULTS"]["STANDARD_TOP_K_FILE"])
    tuned_results = load_csv(EXPERIMENT_ROOT / PARAMS["RESULTS"]["TUNED_FINAL_RESULTS_FILE"])
    tuned_top_k = load_csv(EXPERIMENT_ROOT / PARAMS["RESULTS"]["TUNED_TOP_K_RESULTS_FILE"])
    output_file = EXPERIMENT_ROOT / PARAMS["RESULTS"]["COMPARISON_SUMMARY_FILE"]

    rows = []
    add_native_result_rows(rows, standard_results, standard_top_k, tuned_results, tuned_top_k)

    test_next_close = load_test_next_close()
    prediction_sets = {
        "standard_lstm": prepare_predictions(
            load_csv(EXPERIMENT_ROOT / PARAMS["RESULTS"]["PREDICTIONS_FILE"]),
            test_next_close,
        )
    }

    for _, tuned in tuned_results.iterrows():
        prediction_file = (
            EXPERIMENT_ROOT
            / "data"
            / "processed"
            / f"lstm_tuned_test_predictions_{tuned['selection_method']}.csv"
        )
        prediction_sets[tuned["model_name"]] = prepare_predictions(
            load_csv(prediction_file),
            test_next_close,
        )

    add_common_period_rows(rows, prediction_sets)

    summary = pd.DataFrame(rows)
    save_csv(summary, output_file)

    print_summary(summary)
    print(f"\nComparison summary saved to: {output_file}")


if __name__ == "__main__":
    main()
