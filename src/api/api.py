"""
api.py — Fraud Detection FastAPI Server

Endpoints:
    POST /predict  — returns fraud probability and decision
    POST /explain  — returns SHAP values for a transaction
    GET  /health   — checks if the server is running
"""

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "lgbm_final.pkl"
TRAIN_PATH = ROOT / "data" / "train_final.parquet"

# ── Load model and training data ──────────────────────────────────────────────
print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading training data for SHAP...")
train_df = pd.read_parquet(TRAIN_PATH)
TARGET = "isFraud"
X_train = train_df.drop(columns=[TARGET])

print("Building SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Get feature names from training data
FEATURE_NAMES = X_train.columns.tolist()

print("API ready ✅")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring and explanation",
    version="1.0.0"
)

# ── Request schema ────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    features: dict

# ── Helper ────────────────────────────────────────────────────────────────────
def dict_to_df(features: dict) -> pd.DataFrame:
    """Convert incoming feature dict to dataframe matching training schema."""
    df = pd.DataFrame([features])
    # Add missing columns with 0
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    # Keep only training columns in correct order
    df = df[FEATURE_NAMES]
    return df

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "running", "model": "lgbm_final"}


@app.post("/predict")
def predict(transaction: Transaction):
    df = dict_to_df(transaction.features)
    
    prob = model.predict_proba(df)[0][1]
    decision = "flag" if prob >= 0.35 else "allow"
    
    return {
        "fraud_probability": round(float(prob), 4),
        "decision": decision,
        "threshold": 0.35
    }


@app.post("/explain")
def explain(transaction: Transaction):
    df = dict_to_df(transaction.features)
    
    prob = model.predict_proba(df)[0][1]
    decision = "flag" if prob >= 0.35 else "allow"
    
    shap_values = explainer.shap_values(df)
    
    # Get top 10 features by absolute SHAP value
    shap_series = pd.Series(
        shap_values[0],
        index=FEATURE_NAMES
    ).abs().sort_values(ascending=False)
    
    top_features = {
        feat: round(float(shap_values[0][FEATURE_NAMES.index(feat)]), 4)
        for feat in shap_series.head(10).index
    }
    
    return {
        "fraud_probability": round(float(prob), 4),
        "decision": decision,
        "top_features": top_features
    }