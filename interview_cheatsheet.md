# 🎯 Interview Cheat Sheet — Retail Sales Forecasting

> Quick reference for explaining this project in a data science / ML interview.

---

## 1. Project in One Sentence

> *"I built a production-quality retail sales forecasting system that trains and compares four models — ARIMA, Prophet, XGBoost, and LSTM — on synthetic daily sales data, deployed as an interactive Streamlit dashboard with full CI/CD and Docker support."*

---

## 2. Why These 4 Models?

| Model | Why Chosen | When It Wins |
|---|---|---|
| **ARIMA** | Statistical baseline, interpretable | Short series, clear trend/seasonality |
| **Prophet** | Handles holidays, outlier-robust | Business datasets with seasonal events |
| **XGBoost** | Strong tabular learner | When you have rich feature engineering |
| **LSTM** | Sequence-aware deep learning | Long-range dependencies, non-linear patterns |

> **Interview tip**: "Starting with ARIMA gives a statistical baseline. XGBoost often outperforms it with good feature engineering. LSTM can learn complex patterns but needs more data."

---

## 3. Data Pipeline (walk the interviewer through it)

```
Raw CSV  →  clean_data()  →  train_test_split_ts()  →  scale_target()
                                                              │
                                                    feature_engineering()
                                                              │
                                                     Model Training
                                                              │
                                                     evaluate() → metrics JSON
```

**Key decisions to mention:**
- **Chronological split** (not random) — prevents future data leakage in time series
- **IQR outlier clipping** — removes extreme spikes without losing data
- **MinMaxScaler fitted only on train** — scaler never sees test data

---

## 4. Feature Engineering (XGBoost/LSTM)

| Feature | What It Captures |
|---|---|
| `lag_1` | Yesterday's sales (momentum) |
| `lag_7` | Same day last week (weekly pattern) |
| `lag_30` | Same day last month (monthly trend) |
| `rolling_mean_7` | 7-day moving average (trend smoothing) |
| `rolling_std_14` | 14-day volatility (uncertainty measure) |
| `is_weekend` | Weekend uplift effect |
| `is_holiday_season` | Nov–Dec boost |

---

## 5. Evaluation Metrics

| Metric | Formula | What It Tells You |
|---|---|---|
| **MAE** | mean(|y - ŷ|) | Average error in same units as sales |
| **RMSE** | √mean((y - ŷ)²) | Penalises large errors more than MAE |
| **MAPE** | mean(|y - ŷ|/y) × 100 | Scale-independent % error |
| **R²** | 1 - SS_res/SS_tot | % variance explained (1.0 = perfect) |

> **When asked "which metric to use?"**: *"MAPE is best for business reporting (easy to explain), RMSE is better for training (differentiable, penalises large errors)."*

---

## 6. Architecture Design Decisions

| Decision | Rationale |
|---|---|
| Separate model classes | Each has its own `fit/predict/save/load` — easy to test and swap |
| `config.py` for all settings | No magic numbers in code, single source of truth |
| `trainer.py` orchestrator | Clean separation of concerns, easy to add new models |
| `forecasting.py` dispatcher | Runtime model selection without changing calling code |
| Lazy TensorFlow import | LSTM module imports TF only when used — faster startup |

---

## 7. Streamlit App Pages

| Page | What You Show |
|---|---|
| **Home** | KPI cards (mean/min/max sales, date range), full time series |
| **EDA** | Monthly/weekly patterns, decomposition, correlation heatmap |
| **Forecast** | Pick model + horizon → live prediction + CI band + CSV download |
| **Model Comparison** | Train all 4, metrics table with green highlighting for best model |

---

## 8. CI/CD & DevOps

- **GitHub Actions**: runs Ruff lint + Black check + pytest on every push
- **Docker**: `docker-compose up --build` — one command to deploy
- **Makefile**: `make train`, `make test`, `make app` — developer-friendly shortcuts
- **Logging**: structured logs (timestamp | level | module | message) to console + file

---

## 9. Common Interview Questions & Answers

**Q: How do you prevent data leakage in time series?**
> "I use a chronological split — train gets the first 80%, test gets the last 20% in time order. The scaler is fit only on training data. Features only look backward using `.shift()` so no future information is used."

**Q: Why use MinMaxScaler for LSTM?**
> "Neural networks train poorly on unnormalized data because gradients explode or vanish. MinMaxScaler maps to [0,1] which keeps gradients stable."

**Q: How would you improve this project?**
> "Add Optuna for automated hyperparameter tuning, MLflow for experiment tracking, SHAP for XGBoost explainability, and a model ensemble that blends all four forecasts."

**Q: What does Prophet do differently from ARIMA?**
> "ARIMA fits a mathematical model (AR + I + MA terms) requiring stationarity. Prophet decomposes the series into trend + seasonality + holiday effects — more interpretable for business users and handles outliers gracefully."

**Q: How does LSTM capture seasonality?**
> "Through the look_back window (30 days here), the LSTM cell state learns to retain periodic patterns across sequences — no explicit feature engineering needed unlike XGBoost."

---

## 10. Quick Command Reference

```bash
# Run the dashboard
streamlit run streamlit_app/Home.py

# Train all models
python -m src.models.trainer

# Run tests
pytest tests/ -v

# One model training example
python -c "
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.models.arima_model import ARIMAModel

df = load_data()
train, test, scaler = preprocess(df)
model = ARIMAModel()
model.fit(train['Sales'])
print(model.predict(steps=7))
"
```
