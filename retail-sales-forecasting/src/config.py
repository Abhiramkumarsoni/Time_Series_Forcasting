"""
config.py

Centralized configuration for the Retail Sales Forecasting project.
All paths, hyperparameters, and settings live here so every module
stays consistent.
"""

from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MODEL_DIR  = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIG_DIR    = REPORT_DIR / "figures"
METRIC_DIR = REPORT_DIR / "metrics"

# Ensure directories exist at import time
for _d in [RAW_DIR, PROC_DIR, MODEL_DIR, FIG_DIR, METRIC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Data Settings ────────────────────────────────────────────────────────────
RAW_FILE       = RAW_DIR   / "retail_sales_raw.csv"
PROCESSED_FILE = PROC_DIR  / "sales_processed.csv"

DATE_COL   = "Date"
TARGET_COL = "Sales"
FREQ       = "D"          # daily frequency

TEST_SIZE  = 0.20         # last 20 % used for testing
VAL_SIZE   = 0.10         # last 10 % of train used for validation

# ─── ARIMA ────────────────────────────────────────────────────────────────────
ARIMA_ORDER         = (5, 1, 2)
SARIMA_ORDER        = (1, 1, 1)
SARIMA_SEASONAL     = (1, 1, 1, 7)   # weekly seasonality

# ─── Prophet ──────────────────────────────────────────────────────────────────
PROPHET_YEARLY    = True
PROPHET_WEEKLY    = True
PROPHET_DAILY     = False
PROPHET_CHANGEPOINT_PRIOR = 0.05

# ─── XGBoost ──────────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":    300,
    "learning_rate":   0.05,
    "max_depth":       5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "random_state":    42,
}

# ─── LSTM ─────────────────────────────────────────────────────────────────────
LSTM_LOOK_BACK  = 30    # days of history fed as input
LSTM_UNITS      = 64
LSTM_EPOCHS     = 50
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE   = 5     # early-stopping patience

# ─── Feature Engineering ──────────────────────────────────────────────────────
LAG_DAYS     = [1, 7, 14, 30]
ROLLING_WINS = [7, 14, 30]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = ROOT_DIR / "logs" / "forecasting.log"
