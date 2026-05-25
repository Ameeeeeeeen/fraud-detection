"""
dashboard.py — Fraud Detection Streamlit Dashboard

Sections:
    1. Model Performance Summary
    2. Live Transaction Scorer
    3. Drift Monitoring Panel

Run:
    streamlit run src/dashboard/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import requests
from pathlib import Path


st.write("Dashboard loading...")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "lgbm_final.pkl"
TRAIN_PATH = ROOT / "data" / "train_final.parquet"
TEST_PATH  = ROOT / "data" / "test_final.parquet"
DRIFT_PATH = ROOT / "data" / "drift_shap_report.csv"

API_URL = "http://127.0.0.1:8000"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    test  = pd.read_parquet(TEST_PATH)
    return train, test

@st.cache_data
def load_drift():
    return pd.read_csv(DRIFT_PATH)

model       = load_model()
train_df, test_df = load_data()
drift_df    = load_drift()

TARGET      = "isFraud"
X_test      = test_df.drop(columns=[TARGET])
y_test      = test_df[TARGET]
y_prob      = model.predict_proba(X_test)[:, 1]
y_pred      = (y_prob >= 0.35).astype(int)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection System")
st.markdown("Real-time fraud monitoring and model explainability")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Model Performance Summary
# ══════════════════════════════════════════════════════════════════════════════
st.header("📊 Model Performance")

from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

total_fraud     = int(y_test.sum())
caught_fraud    = int((y_pred * y_test).sum())
missed_fraud    = total_fraud - caught_fraud
false_alarms    = int(y_pred.sum()) - caught_fraud

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("AUC-ROC",         f"{roc_auc_score(y_test, y_prob):.4f}")
col2.metric("Recall",          f"{recall_score(y_test, y_pred):.2%}")
col3.metric("Precision",       f"{precision_score(y_test, y_pred):.2%}")
col4.metric("Fraud Caught",    f"{caught_fraud:,} / {total_fraud:,}")
col5.metric("False Alarms",    f"{false_alarms:,}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Live Transaction Scorer
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔍 Live Transaction Scorer")
st.markdown("Enter transaction details to get an instant fraud score and explanation.")

col_a, col_b, col_c = st.columns(3)

with col_a:
    amount      = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
    zscore      = st.number_input("Amount Z-Score", value=0.0)

with col_b:
    tx_count    = st.number_input("Transactions in last 7 days", min_value=0, value=1)
    has_id      = st.selectbox("Has Identity Info", [1, 0])

with col_c:
    high_vel    = st.selectbox("High Velocity Flag", [0, 1])
    device_info = st.number_input("DeviceInfo (encoded)", value=0.035)

if st.button("Score Transaction"):
    features = {
        "TransactionAmt": amount,
        "amt_zscore": zscore,
        "tx_count_7d": tx_count,
        "has_identity": has_id,
        "is_high_velocity": high_vel,
        "DeviceInfo": device_info
    }

    try:
        # Call predict endpoint
        pred_response = requests.post(f"{API_URL}/predict", json={"features": features})
        pred = pred_response.json()

        # Call explain endpoint
        exp_response = requests.post(f"{API_URL}/explain", json={"features": features})
        exp = exp_response.json()

        # Show result
        prob = pred["fraud_probability"]
        decision = pred["decision"]

        if decision == "flag":
            st.error(f"🚨 FRAUD FLAGGED — Probability: {prob:.2%}")
        else:
            st.success(f"✅ ALLOWED — Probability: {prob:.2%}")

        # Show SHAP explanation
        st.subheader("Why did the model decide this?")
        shap_df = pd.DataFrame({
            "Feature": list(exp["top_features"].keys()),
            "SHAP Value": list(exp["top_features"].values())
        }).sort_values("SHAP Value")

        import plotly.express as px
        fig = px.bar(shap_df, x="SHAP Value", y="Feature", orientation="h",
                     color="SHAP Value", color_continuous_scale="RdBu_r",
                     title="Feature Contributions (red = toward fraud, blue = away from fraud)")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"API error: {e}. Make sure the API is running.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Drift Monitoring
# ══════════════════════════════════════════════════════════════════════════════
st.header("📡 Drift Monitoring")

total_features  = len(drift_df)
drifted         = drift_df["drifted"].sum()
high_risk       = drift_df["high_risk"].sum() if "high_risk" in drift_df.columns else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Features Monitored", total_features)
col2.metric("Features Drifting",        f"{drifted} / {total_features}")
col3.metric("High Risk Features",       high_risk)

st.subheader("High Risk Features — Important AND Drifting")
if "high_risk" in drift_df.columns:
    high_risk_df = drift_df[drift_df["high_risk"] == True][["feature", "shap_importance", "p_value"]]\
                   .sort_values("shap_importance", ascending=False)
    st.dataframe(high_risk_df, use_container_width=True)

st.subheader("Full Drift Report")
st.dataframe(
    drift_df[["feature", "p_value", "drifted"]]\
    .sort_values("p_value"),
    use_container_width=True
)