"""
Local project paths for the market features experiment.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "exp_1_randomforest"

CONF_DIR = EXPERIMENT_ROOT / "conf"
DATA_DIR = EXPERIMENT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = EXPERIMENT_ROOT / "models"
PLOTS_DIR = EXPERIMENT_ROOT / "plots"
SCRIPTS_DIR = EXPERIMENT_ROOT / "scripts"

PARAMS_FILE = CONF_DIR / "params.yaml"
RAW_DATA_FILE = RAW_DATA_DIR / "tech_stocks_raw.csv"
FEATURE_DATA_FILE = PROCESSED_DATA_DIR / "tech_stocks_features.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
TEST_PREDICTIONS_FILE = PROCESSED_DATA_DIR / "test_predictions.csv"
BACKTEST_RESULTS_FILE = PROCESSED_DATA_DIR / "backtest_results.csv"
MODEL_FILE = MODELS_DIR / "random_forest_tuned_market_features.pkl"
FEATURE_COLUMNS_FILE = SCRIPTS_DIR / "03_pre_split_prep" / "features.txt"
