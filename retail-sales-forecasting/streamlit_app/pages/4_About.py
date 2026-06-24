"""
pages/4_About.py  –  Project information and architecture
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="About | Retail Forecast", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
## 🎯 Project Overview

**Retail Sales Forecasting** is a production-quality machine learning project that demonstrates
end-to-end time-series forecasting using four different modelling approaches:

| Model     | Type                       | Strengths                                    |
|-----------|----------------------------|----------------------------------------------|
| ARIMA     | Statistical                | Interpretable, handles trend & seasonality   |
| Prophet   | Decomposition-based        | Robust to outliers, handles holidays         |
| XGBoost   | Gradient Boosting (ML)     | High accuracy with engineered features       |
| LSTM      | Deep Learning (RNN)        | Captures long-range temporal dependencies    |

## 🗂 Project Architecture

```
retail-sales-forecasting/
├── src/                    # Core Python modules
│   ├── config.py           # Centralized config & hyperparams
│   ├── logger.py           # Structured logging
│   ├── data_loader.py      # Data generation & loading
│   ├── preprocessing.py    # Cleaning, splitting, scaling
│   ├── feature_engineering.py  # Lag, rolling, calendar features
│   ├── evaluation.py       # MAE, RMSE, MAPE, R²
│   ├── visualization.py    # Plotly charts
│   ├── forecasting.py      # Inference dispatcher
│   └── models/
│       ├── arima_model.py
│       ├── prophet_model.py
│       ├── xgboost_model.py
│       ├── lstm_model.py
│       └── trainer.py      # End-to-end training pipeline
├── streamlit_app/          # Interactive dashboard
├── notebooks/              # Step-by-step Jupyter notebooks
├── tests/                  # Pytest unit tests
├── Dockerfile              # Container deployment
└── .github/workflows/      # CI/CD pipeline
```

## 🔧 Key Features

- ✅ **Synthetic data generation** – no dataset download required
- ✅ **Modular design** – each model is a self-contained class
- ✅ **Consistent interface** – `fit()`, `predict()`, `save()`, `load()` for every model
- ✅ **Interactive dashboard** – Streamlit with dark theme and Plotly charts
- ✅ **CI/CD** – GitHub Actions runs tests on every push
- ✅ **Docker** – one-command deployment
- ✅ **Logging** – structured logs to both console and file

## 📊 Feature Engineering

| Feature Type   | Examples                                  |
|---------------|-------------------------------------------|
| Lag features  | lag_1, lag_7, lag_14, lag_30             |
| Rolling stats | rolling_mean_7, rolling_std_14, etc.     |
| Calendar      | day_of_week, month, quarter, is_weekend  |
| Domain        | is_holiday_season (Nov–Dec boost)        |
""")

with col2:
    st.markdown("### 📦 Tech Stack")

    tech = {
        "Python 3.10+":    "Core language",
        "Pandas / NumPy":  "Data manipulation",
        "Statsmodels":     "ARIMA / SARIMA",
        "Prophet":         "Facebook Prophet",
        "XGBoost":         "Gradient boosting",
        "TensorFlow/Keras":"LSTM deep learning",
        "Scikit-Learn":    "Preprocessing / metrics",
        "Plotly":          "Interactive charts",
        "Streamlit":       "Dashboard framework",
        "Docker":          "Containerization",
        "GitHub Actions":  "CI/CD pipeline",
        "Pytest":          "Unit testing",
    }

    for tech_name, desc in tech.items():
        st.markdown(f"**{tech_name}** — {desc}")

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.code("""
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app/Home.py

# Train all models
python -m src.models.trainer

# Run tests
pytest tests/ -v

# Docker
docker-compose up --build
""", language="bash")

    st.markdown("---")
    st.markdown("### 📌 Author Notes")
    st.info(
        "This project was built as a portfolio demonstration of "
        "production ML engineering skills including modular code design, "
        "model training pipelines, evaluation frameworks, "
        "and interactive dashboards."
    )
