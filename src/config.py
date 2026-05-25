"""
config.py — Central configuration for the fraud detection pipeline.
Change paths here once; everything else reads from this file.
"""

from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PARQUET_DIR   = ROOT_DIR / "data" / "parquet"

# Parquet outputs
PARQUET_TRAIN      = ROOT_DIR / "data" / "train_final.parquet"
PARQUET_TEST       = ROOT_DIR / "data" / "test_final.parquet"
DRIFT_REPORT_PATH  = ROOT_DIR / "data" / "drift_shap_report.csv"

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR          = ROOT_DIR / "models"
MODEL_PATH         = MODEL_DIR / "lgbm_final.pkl"

# ── Spark settings ────────────────────────────────────────────────────────────
SPARK_APP_NAME   = "FraudDetectionPipeline"
SPARK_MASTER     = "local[*]"
SPARK_DRIVER_MEM = "4g"
SPARK_EXECUTOR_MEM = "4g"

# ── Column settings ───────────────────────────────────────────────────────────
TARGET_COL  = "isFraud"
TIME_COL    = "TransactionDT"
AMOUNT_COL  = "TransactionAmt"
ID_COL      = "TransactionID"

# ── Model settings ────────────────────────────────────────────────────────────
THRESHOLD   = 0.35

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_DIR      = ROOT_DIR / "mlflow_runs"
MLFLOW_EXP_NAME = "fraud-detection"

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000