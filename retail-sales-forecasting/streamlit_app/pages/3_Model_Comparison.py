"""
pages/3_Model_Comparison.py  –  Side-by-side model metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.feature_engineering import build_features, get_feature_columns
from src.evaluation import evaluate
from src.utils import create_lstm_sequences, inverse_scale
from src.config import TARGET_COL, DATE_COL, LSTM_LOOK_BACK

from src.models.arima_model   import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model    import LSTMModel

st.set_page_config(page_title="Model Comparison | Retail", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .best { color: #22c55e; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Model Comparison")
st.markdown("Train all models on the same data and compare their forecasting accuracy.")
st.markdown("---")

@st.cache_data
def load_and_preprocess():
    df = load_data()
    train, test, scaler = preprocess(df)
    return df, train, test, scaler

@st.cache_data
def get_features(_df, _train, _test):
    full = build_features(pd.concat([_train, _test]).reset_index(drop=True))
    cut  = len(_train)
    return full.iloc[:cut], full.iloc[cut:]

@st.cache_data
def run_comparison():
    df, train, test, scaler = load_and_preprocess()
    train_feat, test_feat   = get_features(df, train, test)
    feature_cols            = get_feature_columns(train_feat)

    results = {}

    # ARIMA
    m = ARIMAModel(); m.fit(train[TARGET_COL])
    pred = inverse_scale(m.predict(len(test))["Forecast"].values, scaler)
    true = inverse_scale(test[TARGET_COL].values, scaler)
    results["ARIMA"] = evaluate(true, pred, "arima", save=False)
    results["ARIMA"]["preds"] = pred

    # Prophet
    m2 = ProphetModel(); m2.fit(train)
    p2 = m2.predict(periods=len(test))["yhat"].values
    results["Prophet"] = evaluate(true, p2, "prophet", save=False)
    results["Prophet"]["preds"] = p2

    # XGBoost
    m3 = XGBoostModel()
    m3.fit(train_feat[feature_cols], train_feat[TARGET_COL])
    pred3 = inverse_scale(m3.predict(test_feat[feature_cols]), scaler)
    results["XGBoost"] = evaluate(true, pred3, "xgboost", save=False)
    results["XGBoost"]["preds"] = pred3

    # LSTM (fewer epochs for speed)
    series_train = train[TARGET_COL].values
    series_test  = test[TARGET_COL].values
    X_tr, y_tr   = create_lstm_sequences(series_train, LSTM_LOOK_BACK)
    combined     = np.concatenate([series_train[-LSTM_LOOK_BACK:], series_test])
    X_te, _      = create_lstm_sequences(combined, LSTM_LOOK_BACK)
    m4 = LSTMModel(epochs=10); m4.fit(X_tr, y_tr)
    pred4 = inverse_scale(m4.predict(X_te[:len(test)]), scaler)
    results["LSTM"] = evaluate(true, pred4, "lstm", save=False)
    results["LSTM"]["preds"] = pred4

    return results, true, test[DATE_COL].values

with st.spinner("Training all models … (this takes ~2 min the first time)"):
    try:
        results, y_true, dates = run_comparison()

        # ── Metrics Table ─────────────────────────────────────────────────────
        st.subheader("📋 Metrics Summary")
        rows = []
        for name, m in results.items():
            rows.append({
                "Model": name,
                "MAE":   round(m["mae"],  2),
                "RMSE":  round(m["rmse"], 2),
                "MAPE (%)": round(m["mape"], 2),
                "R²":    round(m["r2"],   4),
            })

        metrics_df = pd.DataFrame(rows).sort_values("RMSE")

        # Highlight best model
        best = metrics_df.iloc[0]["Model"]
        st.dataframe(
            metrics_df.style
                .highlight_min(subset=["MAE","RMSE","MAPE (%)"], color="#14532d")
                .highlight_max(subset=["R²"], color="#14532d")
                .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE (%)": "{:.2f}", "R²": "{:.4f}"}),
            width='stretch',
        )
        st.success(f"🏆 Best model: **{best}** (lowest RMSE)")

        st.markdown("---")

        # ── Bar Chart ─────────────────────────────────────────────────────────
        st.subheader("📊 Error Comparison")
        fig_bar = go.Figure()
        for metric in ["MAE", "RMSE", "MAPE (%)"]:
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=metrics_df["Model"],
                y=metrics_df[metric],
            ))
        fig_bar.update_layout(
            barmode="group",
            template="plotly_dark",
            height=350,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_bar, width='stretch')

        # ── Overlay Forecast Plot ─────────────────────────────────────────────
        st.subheader("🔮 Forecast vs Actual (Test Period)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_true, name="Actual",
                                 line=dict(color="#00C2FF", width=2)))

        colors = {"ARIMA": "orange", "Prophet": "#a78bfa",
                  "XGBoost": "#22c55e", "LSTM": "#f43f5e"}
        for name, m in results.items():
            n_preds = len(m["preds"])
            fig.add_trace(go.Scatter(
                x=dates[:n_preds], y=m["preds"],
                name=name,
                line=dict(color=colors[name], width=1.5, dash="dash"),
            ))

        fig.update_layout(
            title="All Models vs Actual Sales",
            xaxis_title="Date", yaxis_title="Sales",
            template="plotly_dark",
            height=500,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"⚠️ {e}")
        st.info("Ensure all packages are installed: `pip install -r requirements.txt`")
