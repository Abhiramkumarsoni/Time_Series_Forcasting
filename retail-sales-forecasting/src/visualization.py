"""
visualization.py

Reusable Plotly-based charts for EDA and forecast display.

All functions return a plotly Figure so they can be rendered in
both Jupyter notebooks (fig.show()) and Streamlit (st.plotly_chart()).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import DATE_COL, TARGET_COL


# ─── EDA Charts ───────────────────────────────────────────────────────────────

def plot_sales_overview(df: pd.DataFrame) -> go.Figure:
    """Line chart showing the full sales time series."""
    fig = px.line(
        df,
        x=DATE_COL,
        y=TARGET_COL,
        title="Daily Retail Sales Over Time",
        labels={DATE_COL: "Date", TARGET_COL: "Sales (units)"},
        template="plotly_dark",
    )
    fig.update_traces(line_color="#00C2FF", line_width=1.5)
    return fig


def plot_monthly_sales(df: pd.DataFrame) -> go.Figure:
    """Bar chart of average sales per calendar month."""
    df = df.copy()
    df["Month"] = pd.to_datetime(df[DATE_COL]).dt.month
    monthly = df.groupby("Month")[TARGET_COL].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["Month_Name"] = monthly["Month"].apply(lambda m: month_names[m-1])

    fig = px.bar(
        monthly,
        x="Month_Name",
        y=TARGET_COL,
        title="Average Sales by Month",
        labels={"Month_Name": "Month", TARGET_COL: "Avg Sales"},
        color=TARGET_COL,
        color_continuous_scale="Blues",
        template="plotly_dark",
    )
    return fig


def plot_weekly_pattern(df: pd.DataFrame) -> go.Figure:
    """Bar chart of average sales by day-of-week."""
    df = df.copy()
    df["DOW"] = pd.to_datetime(df[DATE_COL]).dt.dayofweek
    dow_names  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    weekly = df.groupby("DOW")[TARGET_COL].mean().reset_index()
    weekly["Day"] = weekly["DOW"].apply(lambda d: dow_names[d])

    fig = px.bar(
        weekly,
        x="Day",
        y=TARGET_COL,
        title="Average Sales by Day of Week",
        labels={"Day": "Day", TARGET_COL: "Avg Sales"},
        color=TARGET_COL,
        color_continuous_scale="Teal",
        template="plotly_dark",
    )
    return fig


def plot_seasonal_decomposition(df: pd.DataFrame, period: int = 7) -> go.Figure:
    """
    Decompose the sales series into trend, seasonal, and residual.
    Uses a 4-panel subplot.
    """
    series = df.set_index(DATE_COL)[TARGET_COL].asfreq("D").ffill()
    decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        shared_xaxes=True,
    )
    for i, (component, name) in enumerate([
        (decomp.observed,  "Observed"),
        (decomp.trend,     "Trend"),
        (decomp.seasonal,  "Seasonal"),
        (decomp.resid,     "Residual"),
    ], start=1):
        fig.add_trace(
            go.Scatter(x=component.index, y=component.values, name=name,
                       line=dict(width=1.2)),
            row=i, col=1,
        )

    fig.update_layout(
        height=700,
        title="Seasonal Decomposition",
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of feature correlations."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmid=0,
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        template="plotly_dark",
        height=600,
    )
    return fig


# ─── Forecast Charts ──────────────────────────────────────────────────────────

def plot_forecast(
    actual_dates: pd.Series,
    actual_values: pd.Series,
    forecast_dates: pd.Series,
    forecast_values: pd.Series,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    model_name: str = "Model",
) -> go.Figure:
    """
    Overlay actual sales with forecast and optional confidence interval.
    """
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=actual_dates, y=actual_values,
        name="Actual", line=dict(color="#00C2FF", width=1.5),
    ))

    # Confidence band (if provided)
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_dates, forecast_dates[::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        name=f"{model_name} Forecast",
        line=dict(color="orange", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=f"{model_name} – Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing MAE / RMSE / MAPE across models.

    Parameters
    ----------
    metrics_df : DataFrame with columns [model, mae, rmse, mape, r2].
    """
    fig = go.Figure()
    for metric in ["mae", "rmse", "mape"]:
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=metrics_df["model"],
            y=metrics_df[metric],
        ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison",
        xaxis_title="Model",
        yaxis_title="Error",
        template="plotly_dark",
    )
    return fig
