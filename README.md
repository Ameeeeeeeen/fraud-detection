#  Real-Time Fraud Detection System

> An end-to-end machine learning pipeline that detects payment fraud in real time, from raw data ingestion to a live API and monitoring dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PySpark](https://img.shields.io/badge/PySpark-3.5-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-4.1-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

---

## 1. Problem

Payment fraud costs the financial industry billions of dollars annually. Detecting it is genuinely hard:

- Fraud is rare : in most datasets less than 5% of transactions are fraudulent, making standard accuracy a useless metric
- Fraud patterns evolve constantly : a model trained today can degrade silently over months as fraudsters adapt
- Decisions must be made in milliseconds at the moment of transaction
- Two types of mistakes carry different costs: missing a fraudster loses money, blocking a legitimate customer loses trust

This project builds a system that addresses all four challenges, not just the modeling part.

---

## 2. Solution

A five-phase pipeline that mirrors how fraud detection is built in production:

| Phase | What it does |
|---|---|
| **1. Data Pipeline** | Ingest raw data with Spark, join tables, compute behavioral features, store as Parquet |
| **2. Feature Engineering** | Encode categoricals, compress 292 anonymized columns with PCA, handle nulls, create fraud signals |
| **3. Modeling** | Train LightGBM with class-weight adjustment, tune with Optuna, optimize decision threshold |
| **4. MLOps** | Track experiments with MLflow, monitor feature drift with statistical tests |
| **5. Deployment** | Serve predictions via FastAPI, visualize results in a Streamlit dashboard |

---

## 3. Results

Evaluated on 104,637 held-out transactions from the most recent time period (time-based split):

| Metric | Value |
|---|---|
| AUC-ROC | **0.8897** |
| Fraud Recall | **84.53%** — catches 8 in 10 fraudsters |
| Fraud Precision | 10.85% |
| Fraud Caught | 3,011 / 3,562 |
| Decision Threshold | 0.35 (tuned to prioritize recall) |

**Model comparison:**

| Model | AUC-ROC | Fraud Recall |
|---|---|---|
| XGBoost Baseline | 0.8733 | 73.0% |
| LightGBM Baseline | 0.8819 | 75.4% |
| LightGBM Tuned  | **0.8897** | **84.5%** |

**Estimated financial impact** (average fraud transaction = $150, investigation cost = $5/false alarm):

| Item | Amount |
|---|---|
| Fraud prevented | +$451,650 |
| Fraud missed | -$82,650 |
| False alarm investigation | -$123,725 |
| **Net savings per 104k transactions** | **$327,925** |

---

## 4. Project Structure

```
fraud-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py                      ← central paths and settings
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── spark_session.py           ← reusable Spark factory
│   │   ├── batch_ingestion.py         ← Phase 1: CSV → join → features → Parquet
│   │   └── validate_ingestion.py      ← data quality gate
│   ├── api/
│   │   ├── __init__.py
│   │   └── api.py                     ← FastAPI: /predict and /explain endpoints
│   └── dashboard/
│       ├── __init__.py
│       └── dashboard.py               ← Streamlit monitoring dashboard
│
├── notebooks/
│   └── Fraud_Det.ipynb                ← full pipeline notebook (Google Colab)
│
└── data/
    └── .gitkeep                       ← directory tracked, data files excluded
```

---

## 5. Setup & Installation

### Prerequisites
- Python 3.10+
- Java 8+ (required for PySpark)
- Google Colab account (for Phase 1 & 2 notebook)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the dataset
Download from [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection).

You need these four files:
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

Place them in `data/raw/`.

### 4. Run the pipeline notebook
Open `notebooks/Fraud_Det.ipynb` in Google Colab. Run all phases in order — the notebook handles data ingestion, feature engineering, model training, and saves outputs to Google Drive.

Download the outputs from Drive and place them in:
- `data/train_final.parquet`
- `data/test_final.parquet`
- `data/drift_shap_report.csv`
- `models/lgbm_final.pkl`

---

## 6. Running the Application

You need two terminals running simultaneously.

**Terminal 1 — Start the API:**
```bash
uvicorn src.api.api:app --reload
```
API live at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

**Terminal 2 — Start the dashboard:**
```bash
streamlit run src/dashboard/dashboard.py
```
Dashboard live at `http://localhost:8501`

---

## 7. API Reference

### `GET /health`
Check if the server is running.

```json
{ "status": "running", "model": "lgbm_final" }
```

### `POST /predict`
Score a transaction and return a fraud decision.

**Request:**
```json
{
  "features": {
    "TransactionAmt": 500.0,
    "amt_zscore": 3.5,
    "tx_count_7d": 15,
    "has_identity": 0
  }
}
```

**Response:**
```json
{
  "fraud_probability": 0.8712,
  "decision": "flag",
  "threshold": 0.35
}
```

### `POST /explain`
Score a transaction and return SHAP feature contributions.

**Response:**
```json
{
  "fraud_probability": 0.8712,
  "decision": "flag",
  "top_features": {
    "DeviceInfo": 1.24,
    "amt_zscore": 0.87,
    "tx_count_7d": 0.54
  }
}
```

---

## 8. Technical Deep Dive

### Why time-based train/test split?
Splitting by time (not randomly) simulates real deployment. the model learns from the past and is tested on the future. Random splitting would allow the model to train on future patterns, inflating performance metrics artificially.

### Why target encoding for email domains?
Email domain has 60+ unique values which are too many for one-hot encoding. Target encoding replaces each domain with its historical fraud rate computed from training data only, preventing data leakage.

### Why threshold = 0.35 instead of 0.5?
The default 0.5 threshold optimizes for accuracy. For fraud detection, recall matters more, missing a fraudster is costlier than a false alarm. Lowering the threshold to 0.35 increased recall from 74% to 84.5%.

### Why PCA on V columns?
The dataset contains 292 anonymized V columns. PCA compressed them to 30 components capturing 99.9% of the variance — reducing noise, speeding up training, and preventing memory issues.

### How is drift detected?
- **Numerical features:** Kolmogorov-Smirnov test compares distributions between training and new data
- **Categorical features:** Chi-square test compares category frequencies
- Features with p-value < 0.05 are flagged as drifted
- Features that are both high SHAP importance AND drifting are flagged as retraining triggers

---

## 9. Known Limitations

- **DeviceInfo dominance** : the model relies heavily on device fingerprinting. Fraudsters using clean devices may evade detection. A feature ablation study is recommended before production deployment.
- **Low precision** : 10.85% precision means 9 in 10 flagged transactions are legitimate. Acceptable for high-recall fraud systems but requires sufficient analyst capacity for investigation.
- **Static threshold** : the 0.35 threshold was tuned on this dataset. Different business contexts (higher/lower fraud rates) would require re-tuning.
- **19 high-risk features drifting** : statistical tests indicate model retraining is needed within 60 days of the training cutoff.

---

##  Tech Stack

| Category | Tools |
|---|---|
| Data Processing | PySpark, Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM, XGBoost |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Drift Monitoring | SciPy (KS-test, Chi-square) |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Storage | Parquet, Joblib |

---

##  Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```

Key libraries: `pyspark`, `lightgbm`, `xgboost`, `optuna`, `shap`, `mlflow`, `fastapi`, `uvicorn`, `streamlit`, `scikit-learn`, `scipy`, `plotly`, `pyarrow`, `pandas`, `numpy`, `joblib`, `requests`

---

*Portfolio project — built to simulate a production ML system end to end.*
*Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle)*
