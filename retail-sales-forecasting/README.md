# 📈 Retail Sales Forecasting

> **Production-quality time-series forecasting using ARIMA, Prophet, XGBoost, and LSTM — with an interactive Streamlit dashboard.**

[![CI](https://github.com/your-username/retail-sales-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/retail-sales-forecasting/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io)

---

## 🎯 Project Overview

This project demonstrates a **complete, production-ready machine learning pipeline** for retail sales forecasting. It covers the full lifecycle from raw data generation through exploratory analysis, feature engineering, model training, evaluation, and interactive deployment.

The project uses **four complementary forecasting approaches** to show how different model families tackle the same problem:

| Model | Category | Key Strength |
|---|---|---|
| **ARIMA** | Statistical | Interpretable, handles trend & seasonality |
| **Prophet** | Decomposition | Robust to outliers, handles holiday effects |
| **XGBoost** | Gradient Boosting | High accuracy with engineered tabular features |
| **LSTM** | Deep Learning | Captures long-range temporal dependencies |

---

## 📂 Project Structure

```
retail-sales-forecasting/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
│
├── data/
│   ├── raw/                    # Raw CSV (auto-generated)
│   └── processed/              # Scaled train/test splits + scaler
│
├── models/                     # Saved model artifacts (.pkl, .keras)
│
├── notebooks/                  # Step-by-step Jupyter notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_arima_model.ipynb
│   ├── 05_prophet_model.ipynb
│   ├── 06_xgboost_model.ipynb
│   ├── 07_lstm_model.ipynb
│   └── 08_model_comparison.ipynb
│
├── reports/
│   ├── figures/                # Exported plots
│   └── metrics/                # Model metric JSON files
│
├── src/                        # Core Python package
│   ├── config.py               # All hyperparameters & paths
│   ├── logger.py               # Structured logging
│   ├── data_loader.py          # Data generation & loading
│   ├── preprocessing.py        # Cleaning, splitting, scaling
│   ├── feature_engineering.py  # Lag, rolling, calendar features
│   ├── visualization.py        # Plotly charts (shared)
│   ├── evaluation.py           # MAE, RMSE, MAPE, R²
│   ├── forecasting.py          # Inference dispatcher
│   └── models/
│       ├── arima_model.py
│       ├── prophet_model.py
│       ├── xgboost_model.py
│       ├── lstm_model.py
│       └── trainer.py          # End-to-end training orchestrator
│
├── streamlit_app/              # Interactive dashboard
│   ├── Home.py                 # Overview & KPIs
│   └── pages/
│       ├── 1_EDA.py            # Exploratory analysis
│       ├── 2_Forecast.py       # Interactive forecasting
│       ├── 3_Model_Comparison.py
│       └── 4_About.py
│
├── tests/                      # Pytest unit tests
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── .gitignore
```

---

## 📊 Dataset

This project uses a **synthetic retail sales dataset** generated automatically — no download needed.

| Property | Details |
|---|---|
| Period | 2021-01-01 to 2024-12-31 (4 years, daily) |
| Rows | 1,461 observations |
| Target | `Sales` (daily units sold) |
| Trend | Gradual upward drift |
| Seasonality | Weekly (weekends higher) + Yearly (Nov–Dec boost) |
| Noise | Gaussian noise + random promotional spikes |

The data is generated automatically on first run by `src.data_loader.generate_synthetic_data()`.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/retail-sales-forecasting.git
cd retail-sales-forecasting
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run streamlit_app/Home.py
```

Then open **http://localhost:8501** in your browser.

### 3. Train All Models (CLI)

```bash
python -m src.models.trainer
```

This trains ARIMA → Prophet → XGBoost → LSTM and saves all models to `models/`.

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 🧪 Features

### Data Pipeline
- ✅ Automatic synthetic data generation (no manual download)
- ✅ Missing date imputation (time-based interpolation)
- ✅ Outlier clipping (IQR × 3)
- ✅ Chronological train/test split (no data leakage)
- ✅ MinMax scaling with persisted scaler

### Feature Engineering
- ✅ **Lag features**: `lag_1`, `lag_7`, `lag_14`, `lag_30`
- ✅ **Rolling statistics**: 7/14/30-day mean & standard deviation
- ✅ **Calendar features**: day-of-week, month, quarter, is_weekend, is_holiday_season

### Models
- ✅ **ARIMA/SARIMA** – with confidence intervals
- ✅ **Prophet** – yearly + weekly seasonality
- ✅ **XGBoost** – feature importance included
- ✅ **LSTM** – with early stopping and dropout

### Evaluation
- ✅ MAE, RMSE, MAPE, R²
- ✅ JSON metric persistence for comparison
- ✅ Side-by-side model leaderboard

### Dashboard (Streamlit)
- ✅ **Home** – KPI cards, date filter, sales overview
- ✅ **EDA** – seasonal patterns, decomposition, correlation heatmap
- ✅ **Forecast** – choose model + horizon, view forecast with CI, download CSV
- ✅ **Model Comparison** – metrics table + overlay chart
- ✅ **About** – architecture and tech stack

---

## 🐳 Docker Deployment

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

The dashboard will be available at **http://localhost:8501**.

---

## ⚙️ Configuration

All settings are centralised in [`src/config.py`](src/config.py):

```python
# Key settings you can tune
ARIMA_ORDER      = (5, 1, 2)
PROPHET_YEARLY   = True
XGB_PARAMS       = {"n_estimators": 300, "learning_rate": 0.05, ...}
LSTM_LOOK_BACK   = 30     # days of history
LSTM_EPOCHS      = 50
TEST_SIZE        = 0.20   # 20% held out for testing
LAG_DAYS         = [1, 7, 14, 30]
ROLLING_WINS     = [7, 14, 30]
```

---

## 📐 Model Architecture

```
Raw Data
    │
    ▼
data_loader.py ──► Synthetic / CSV data
    │
    ▼
preprocessing.py ──► Clean → Split → Scale
    │
    ├──► feature_engineering.py ──► Lags, Rolling, Calendar
    │           │
    │           ▼
    │       XGBoost / LSTM (tabular/sequence)
    │
    ├──► ARIMA (on raw scaled series)
    │
    └──► Prophet (on date + target)
    │
    ▼
evaluation.py ──► MAE, RMSE, MAPE, R²
    │
    ▼
reports/metrics/*.json
```

---

## 🧠 Interview Talking Points

| Question | Answer |
|---|---|
| Why 4 models? | Shows breadth: statistical baseline → ML → deep learning |
| Why XGBoost needs feature engineering | Tree models need explicit temporal features (lags) |
| Why LSTM uses sequences | RNNs learn patterns from raw time-step sequences |
| Why MinMaxScaler? | Neural networks train better on normalised input |
| Why chronological split? | Prevents future data leakage in time series |
| How to improve? | Hyperparameter tuning (Optuna), stacking, MLflow tracking |

---

## 🛠 Makefile Commands

```bash
make install    # Install all dependencies
make data       # Generate synthetic dataset
make train      # Train all four models
make test       # Run pytest unit tests
make lint       # Lint with Ruff
make format     # Auto-format with Black + isort
make app        # Launch Streamlit dashboard
make docker     # Build and run with Docker Compose
make clean      # Remove cache and generated files
```

---

## 🔮 Future Improvements

- [ ] **Optuna** hyperparameter optimisation
- [ ] **MLflow** experiment tracking
- [ ] **SHAP** feature explainability for XGBoost
- [ ] **DVC** data version control
- [ ] **Ensemble** model combining all four forecasters
- [ ] **SARIMA** seasonal ARIMA variant
- [ ] **Transformer** (Temporal Fusion Transformer)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ using Python · Streamlit · Plotly · XGBoost · TensorFlow · Prophet</p>
