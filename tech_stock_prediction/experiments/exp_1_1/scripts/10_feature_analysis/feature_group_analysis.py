"""
Feature group analysis.

This script tests which feature groups help the model and which ones do not.
It trains the same RandomForestClassifier several times with different feature
sets and compares the validation and test metrics.
"""

from pathlib import Path
from itertools import combinations
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ensure_venv import restart_with_project_venv

restart_with_project_venv()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from formatting import format_decimal, save_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load(open(EXPERIMENT_ROOT / "conf" / "params.yaml"))

TECHNICAL_FEATURES = [
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
]

VOLUME_FEATURES = [
    "Volume_Change",
    "Volume_Ratio_20",
]

TREND_MOMENTUM_FEATURES = [
    "Distance_to_MA_200",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Price_Position_20",
    "High_Low_Range",
]

FUNDAMENTAL_FEATURES = [
    "revenue_growth",
    "eps",
    "profit_margin",
    "operating_margin",
    "free_cash_flow",
    "debt_to_equity",
    "roe",
]

EARNINGS_FEATURES = [
    "actual_eps",
    "expected_eps",
    "eps_surprise",
    "revenue_surprise",
]

FEATURE_GROUPS = {
    "Technical": TECHNICAL_FEATURES,
    "Volume": VOLUME_FEATURES,
    "Trend/Momentum": TREND_MOMENTUM_FEATURES,
    "Fundamental": FUNDAMENTAL_FEATURES,
    "Earnings": EARNINGS_FEATURES,
}


def get_available_features(data, features):
    return [feature for feature in features if feature in data.columns]


def make_model():
    return RandomForestClassifier(
        n_estimators=PARAMS["MODELING"]["N_ESTIMATORS"],
        max_depth=PARAMS["MODELING"]["MAX_DEPTH"],
        random_state=PARAMS["MODELING"]["RANDOM_STATE"],
        class_weight="balanced",
    )


def calculate_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1_Score": f1_score(y_true, predictions, zero_division=0),
    }


def train_and_evaluate(name, features, train_data, validation_data, test_data):
    target = PARAMS["MODELING"]["TARGET"]
    model = make_model()

    x_train = train_data[features]
    y_train = train_data[target]
    x_validation = validation_data[features]
    y_validation = validation_data[target]
    x_test = test_data[features]
    y_test = test_data[target]

    model.fit(x_train, y_train)

    validation_predictions = model.predict(x_validation)
    test_predictions = model.predict(x_test)

    row = {
        "Experiment": name,
        "Feature_Groups": name,
        "Feature_Count": len(features),
        "Features": ", ".join(features),
    }

    for prefix, metrics in [
        ("Validation", calculate_metrics(y_validation, validation_predictions)),
        ("Test", calculate_metrics(y_test, test_predictions)),
    ]:
        for metric_name, metric_value in metrics.items():
            row[f"{prefix}_{metric_name}"] = metric_value

    row["Validation_Predicted_Up_Share"] = validation_predictions.mean()
    row["Test_Predicted_Up_Share"] = test_predictions.mean()

    return row, model


def build_experiments(all_features):
    experiments = []
    group_names = list(FEATURE_GROUPS)

    for group_count in range(1, len(group_names) + 1):
        for selected_groups in combinations(group_names, group_count):
            features = []
            for group_name in selected_groups:
                features.extend(FEATURE_GROUPS[group_name])

            name = " + ".join(selected_groups)
            experiments.append((name, features))

    return experiments


def add_baseline_differences(results):
    technical_row = results[results["Experiment"] == "Technical"].iloc[0]
    technical_accuracy = technical_row["Test_Accuracy"]
    technical_f1 = technical_row["Test_F1_Score"]

    results["Test_Accuracy_vs_Technical"] = results["Test_Accuracy"] - technical_accuracy
    results["Test_F1_vs_Technical"] = results["Test_F1_Score"] - technical_f1
    return results


def save_feature_group_plot(results, output_file):
    plot_data = results.sort_values("Test_F1_Score", ascending=False).head(12)
    plot_data = plot_data.sort_values("Test_F1_Score", ascending=True)
    colors = ["#2F7D5F" if value >= 0 else "#C9504D" for value in plot_data["Test_F1_vs_Technical"]]

    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(plot_data["Experiment"], plot_data["Test_F1_vs_Technical"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Top Feature Group Combinations: F1 Difference vs Technical Features", fontsize=15, fontweight="bold")
    ax.set_xlabel("Test F1 difference compared to technical-only model")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[format_decimal(value) for value in plot_data["Test_F1_vs_Technical"]], padding=4)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_importance_plot(importances, output_file):
    top_features = importances.head(15).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top_features["Feature"], top_features["Importance"], color="#2F6B9A")
    ax.set_title("Top 15 Feature Importances", fontsize=15, fontweight="bold")
    ax.set_xlabel("Random Forest importance")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[format_decimal(value) for value in top_features["Importance"]], padding=4)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_results(results):
    print("Feature Group Analysis")
    print("======================")
    top_results = results.sort_values("Test_F1_Score", ascending=False).head(10)
    print(top_results[[
        "Experiment",
        "Feature_Count",
        "Test_Accuracy",
        "Test_F1_Score",
        "Test_Predicted_Up_Share",
        "Test_Accuracy_vs_Technical",
        "Test_F1_vs_Technical",
    ]].to_string(index=False))

    best_row = top_results.iloc[0]
    print("\nBest feature set by Test F1:")
    print(f"{best_row['Experiment']} with F1 = {format_decimal(best_row['Test_F1_Score'])}")

    non_trivial = results[results["Test_Predicted_Up_Share"] < 0.95]
    if not non_trivial.empty:
        best_non_trivial = non_trivial.sort_values("Test_F1_Score", ascending=False).iloc[0]
        print("\nBest feature set that does not almost always predict up:")
        print(f"{best_non_trivial['Experiment']} with F1 = {format_decimal(best_non_trivial['Test_F1_Score'])}")


def main():
    train_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["TRAIN_FILE"]
    validation_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["VALIDATION_FILE"]
    test_file = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["TEST_FILE"]
    feature_path = EXPERIMENT_ROOT / PARAMS["DATA_PREP"]["FEATURE_PATH"]
    processed_dir = EXPERIMENT_ROOT / "data" / "processed"
    plots_dir = EXPERIMENT_ROOT / "plots"

    train_data = pd.read_csv(train_file, parse_dates=["Date"])
    validation_data = pd.read_csv(validation_file, parse_dates=["Date"])
    test_data = pd.read_csv(test_file, parse_dates=["Date"])
    configured_features = [line.strip() for line in open(feature_path) if line.strip()]
    all_features = get_available_features(train_data, configured_features)

    rows = []
    all_features_model = None
    all_features_model_columns = []
    all_features_experiment = " + ".join(FEATURE_GROUPS)
    experiments = build_experiments(all_features)

    for name, features in experiments:
        available_features = get_available_features(train_data, features)
        row, model = train_and_evaluate(name, available_features, train_data, validation_data, test_data)
        rows.append(row)

        if name == all_features_experiment:
            all_features_model = model
            all_features_model_columns = available_features

    results = add_baseline_differences(pd.DataFrame(rows))
    results = results.sort_values("Test_F1_Score", ascending=False).reset_index(drop=True)
    results.insert(0, "Rank_By_Test_F1", results.index + 1)
    results_file = processed_dir / "feature_group_results.csv"
    save_csv(results, results_file)
    save_feature_group_plot(results, plots_dir / "16_feature_group_comparison.png")

    importances = pd.DataFrame({
        "Feature": all_features_model_columns,
        "Importance": all_features_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    importances_file = processed_dir / "feature_importance_results.csv"
    save_csv(importances, importances_file)
    save_importance_plot(importances, plots_dir / "18_feature_importance_top15.png")

    print_results(results)
    print(f"\nSaved results to: {results_file}")
    print(f"Saved feature importances to: {importances_file}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
