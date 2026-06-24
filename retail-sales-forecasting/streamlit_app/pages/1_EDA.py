"""
pages/1_EDA.py  –  Exploratory Data Analysis page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import build_features
from src.visualization import (
    plot_monthly_sales,
    plot_weekly_pattern,
    plot_seasonal_decomposition,
    plot_correlation_heatmap,
)

st.set_page_config(page_title="EDA | Retail Forecast", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .section-header {
        color: #ffffff; font-size: 1.3rem; font-weight: 600;
        border-left: 4px solid #a78bfa; padding-left: 12px;
        margin: 24px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_data()
    df = clean_data(df)
    return df

df      = get_data()
df_feat = build_features(df)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔍 Exploratory Data Analysis")
st.markdown("Understand trends, seasonality, and feature correlations before modelling.")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Seasonal Patterns",
    "📆 Weekly & Monthly",
    "📊 Decomposition",
    "🔗 Correlations",
])

with tab1:
    st.markdown('<p class="section-header">Seasonal Patterns</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_monthly_sales(df), width='stretch')
    with col2:
        st.plotly_chart(plot_weekly_pattern(df), width='stretch')

    st.markdown("**Key Observations:**")
    st.info(
        "- 📈 Sales peak in **November & December** (holiday season)\n"
        "- 🗓 **Weekends** consistently outperform weekdays\n"
        "- 📉 Lowest sales observed in **January & February**"
    )

with tab2:
    st.markdown('<p class="section-header">Monthly & Weekly Breakdown</p>', unsafe_allow_html=True)

    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    monthly_ts = df.groupby("YearMonth")["Sales"].mean().reset_index()

    import plotly.express as px
    fig = px.line(
        monthly_ts, x="YearMonth", y="Sales",
        title="Monthly Average Sales (Full History)",
        template="plotly_dark",
        markers=True,
    )
    fig.update_traces(line_color="#a78bfa")
    st.plotly_chart(fig, width='stretch')

    # Year-over-year
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    yoy = df.groupby(["Year", "Month"])["Sales"].mean().reset_index()
    fig2 = px.line(
        yoy, x="Month", y="Sales", color="Year",
        title="Year-over-Year Monthly Comparison",
        template="plotly_dark",
        markers=True,
    )
    st.plotly_chart(fig2, width='stretch')

with tab3:
    st.markdown('<p class="section-header">Seasonal Decomposition (Period = 7)</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_seasonal_decomposition(df, period=7), width='stretch')
    st.info(
        "The decomposition separates the series into:\n"
        "- **Trend** – long-term upward growth\n"
        "- **Seasonal** – repeating weekly / yearly patterns\n"
        "- **Residual** – unexplained random noise"
    )

with tab4:
    st.markdown('<p class="section-header">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_correlation_heatmap(df_feat), width='stretch')
    st.caption("Lag features and rolling means are highly correlated with Sales (as expected).")
