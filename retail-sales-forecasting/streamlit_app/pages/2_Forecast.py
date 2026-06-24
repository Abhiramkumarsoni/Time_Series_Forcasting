"""
pages/2_Forecast.py  –  Interactive forecast page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.feature_engineering import build_features, get_feature_columns
from src.models.arima_model   import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model    import LSTMModel
from src.utils import create_lstm_sequences, inverse_scale
from src.config import MODEL_DIR, LSTM_LOOK_BACK, DATE_COL, TARGET_COL
from src.logger import logger

st.set_page_config(page_title="Forecast | Retail", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = load_data()
    train, test, scaler = preprocess(df)
    return df, train, test, scaler

@st.cache_data
def get_features(_df, _train, _test):
    full  = build_features(pd.concat([_train, _test]).reset_index(drop=True))
    cut   = len(_train)
    return full.iloc[:cut], full.iloc[cut:]

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("🔮 Sales Forecast")
st.markdown("Select a model and forecast horizon, then click **Generate Forecast**.")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Forecast Settings")
    model_choice = st.selectbox(
        "Choose Model",
        ["ARIMA", "Prophet", "XGBoost", "LSTM"],
        index=0,
    )
    horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
    run_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

df, train, test, scaler = load_and_preprocess()
train_feat, test_feat   = get_features(df, train, test)

# ── Quick info ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Training Rows", f"{len(train):,}")
col2.metric("Test Rows",     f"{len(test):,}")
col3.metric("Forecast Days", horizon)

st.markdown("---")

if run_btn or True:     # always show something on page load
    with st.spinner(f"Training {model_choice} model …"):

        try:
            # ── ARIMA ─────────────────────────────────────────────────────────
            if model_choice == "ARIMA":
                model = ARIMAModel()
                model.fit(train[TARGET_COL])
                result = model.predict(steps=horizon)

                future_dates = pd.date_range(
                    start=test[DATE_COL].iloc[-1] + pd.Timedelta(days=1),
                    periods=horizon, freq="D",
                )
                y_pred = inverse_scale(result["Forecast"].values, scaler)
                lower  = inverse_scale(result["Lower_CI"].values,  scaler)
                upper  = inverse_scale(result["Upper_CI"].values,  scaler)
                has_ci = True

            # ── Prophet ───────────────────────────────────────────────────────
            elif model_choice == "Prophet":
                model = ProphetModel()
                model.fit(train)
                result = model.predict(periods=horizon)

                future_dates = result["ds"]
                y_pred = result["yhat"].values
                # Prophet gives absolute values – no scaler needed
                lower  = result["yhat_lower"].values
                upper  = result["yhat_upper"].values
                has_ci = True

            # ── XGBoost ───────────────────────────────────────────────────────
            elif model_choice == "XGBoost":
                feature_cols = get_feature_columns(train_feat)
                model = XGBoostModel()
                model.fit(train_feat[feature_cols], train_feat[TARGET_COL])

                future_dates = test[DATE_COL].iloc[:horizon]
                y_pred = inverse_scale(
                    model.predict(test_feat[feature_cols].iloc[:horizon]), scaler
                )
                lower, upper, has_ci = None, None, False

            # ── LSTM ──────────────────────────────────────────────────────────
            elif model_choice == "LSTM":
                series_train = train[TARGET_COL].values
                series_test  = test[TARGET_COL].values
                X_train, y_train = create_lstm_sequences(series_train, LSTM_LOOK_BACK)
                combined = np.concatenate([series_train[-LSTM_LOOK_BACK:], series_test])
                X_test, _  = create_lstm_sequences(combined, LSTM_LOOK_BACK)

                model = LSTMModel(epochs=15)   # fewer epochs for demo
                model.fit(X_train, y_train)

                preds_scaled = model.predict(X_test[:horizon])
                y_pred = inverse_scale(preds_scaled, scaler)
                future_dates = test[DATE_COL].iloc[:horizon]
                lower, upper, has_ci = None, None, False

            # ── Plot ──────────────────────────────────────────────────────────
            actual_inv = inverse_scale(test[TARGET_COL].values, scaler)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train[DATE_COL], y=inverse_scale(train[TARGET_COL].values, scaler),
                name="Train", line=dict(color="#8b97c0", width=1),
            ))
            fig.add_trace(go.Scatter(
                x=test[DATE_COL], y=actual_inv,
                name="Actual", line=dict(color="#00C2FF", width=1.5),
            ))

            if has_ci and lower is not None:
                fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=list(upper) + list(lower)[::-1],
                    fill="toself", fillcolor="rgba(255,165,0,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name="95 % CI",
                ))

            fig.add_trace(go.Scatter(
                x=future_dates, y=y_pred,
                name=f"{model_choice} Forecast",
                line=dict(color="orange", width=2.5, dash="dash"),
            ))

            fig.update_layout(
                title=f"{model_choice} – {horizon}-Day Sales Forecast",
                xaxis_title="Date", yaxis_title="Sales (units)",
                template="plotly_dark",
                legend=dict(orientation="h", y=1.05),
                height=500,
            )
            st.plotly_chart(fig, width='stretch')

            # ── Download ──────────────────────────────────────────────────────
            dl_df = pd.DataFrame({
                "Date": future_dates.values if hasattr(future_dates, "values") else future_dates,
                "Forecast": y_pred,
            })
            csv = dl_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Forecast CSV",
                data=csv,
                file_name=f"{model_choice.lower()}_forecast_{horizon}d.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            st.info("Make sure all dependencies are installed: `pip install -r requirements.txt`")
