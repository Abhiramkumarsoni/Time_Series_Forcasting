"""
Home.py  –  Streamlit Multi-page App: Home / Overview

Run:
    streamlit run streamlit_app/Home.py
"""

import sys
from pathlib import Path

# ── Make src importable ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.utils import summarise_df
from src.visualization import plot_sales_overview

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background: #0e1117; }

    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border: 1px solid #2e3555;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-label { color: #8b97c0; font-size: 13px; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #ffffff; font-size: 30px; font-weight: 700; margin-top: 4px; }
    .kpi-sub   { color: #00C2FF; font-size: 12px; margin-top: 2px; }

    .hero-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(90deg, #00C2FF, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: #8b97c0; font-size: 1.1rem; margin-top: -8px; }

    .section-header {
        color: #ffffff; font-size: 1.3rem; font-weight: 600;
        border-left: 4px solid #00C2FF; padding-left: 12px;
        margin: 24px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()

df = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.markdown("## 📈 Sales Forecasting")
    st.markdown("---")

    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()

    start_date, end_date = st.date_input(
        "📅 Date Range",
        value=[date_min, date_max],
        min_value=date_min,
        max_value=date_max,
    )
    st.markdown("---")
    st.caption("Navigate using the **Pages** menu above.")

# ── Filter ────────────────────────────────────────────────────────────────────
mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
filtered = df[mask]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">Retail Sales Forecasting</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">AI-powered time-series forecasting using ARIMA, Prophet, XGBoost & LSTM</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
summary = summarise_df(filtered)

c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

kpi(c1, "Total Records",  f"{summary['rows']:,}",              "daily observations")
kpi(c2, "Mean Sales",     f"{summary['mean_sales']:.1f}",      "units / day")
kpi(c3, "Min Sales",      f"{summary['min_sales']:.1f}",       "units")
kpi(c4, "Max Sales",      f"{summary['max_sales']:.1f}",       "units")
kpi(c5, "Date Range",     f"{summary['rows'] // 365} yrs",     summary["date_range"])

st.markdown("---")

# ── Overview Chart ────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Sales Time Series</p>', unsafe_allow_html=True)
st.plotly_chart(plot_sales_overview(filtered), width='stretch')

# ── Dataset Preview ───────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)
st.dataframe(
    filtered.tail(30).style.format({"Sales": "{:.2f}"}),
    width='stretch',
)

st.markdown("---")
st.caption("📌 Use the sidebar to filter dates | Navigate to **EDA**, **Forecast**, or **Model Comparison** pages.")
