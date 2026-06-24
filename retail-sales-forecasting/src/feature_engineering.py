"""
feature_engineering.py

Create ML-ready features from the cleaned time-series DataFrame.

Features Generated
------------------
- Lag features        : Sales at t-1, t-7, t-14, t-30
- Rolling statistics  : 7-day / 14-day / 30-day mean & std
- Calendar features   : day-of-week, day-of-month, month, quarter,
                        is_weekend, is_month_end, is_holiday_season
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET_COL, LAG_DAYS, ROLLING_WINS
from src.logger import logger


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features for each window defined in config.LAG_DAYS."""
    df = df.copy()
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)
        logger.debug(f"Added lag_{lag}")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and rolling std for each window in config.ROLLING_WINS."""
    df = df.copy()
    for win in ROLLING_WINS:
        df[f"rolling_mean_{win}"] = (
            df[TARGET_COL].shift(1).rolling(win).mean()
        )
        df[f"rolling_std_{win}"] = (
            df[TARGET_COL].shift(1).rolling(win).std()
        )
        logger.debug(f"Added rolling stats for window={win}")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar / date-based features."""
    df = df.copy()
    dt = pd.to_datetime(df[DATE_COL])

    df["day_of_week"]      = dt.dt.dayofweek          # 0 = Mon
    df["day_of_month"]     = dt.dt.day
    df["month"]            = dt.dt.month
    df["quarter"]          = dt.dt.quarter
    df["week_of_year"]     = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]       = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_end"]     = dt.dt.is_month_end.astype(int)
    df["is_holiday_season"] = dt.dt.month.isin([11, 12]).astype(int)

    logger.debug("Calendar features added.")
    return df


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Parameters
    ----------
    df       : Cleaned DataFrame with [Date, Sales].
    drop_na  : Drop rows that contain NaN (introduced by lags / rolling).

    Returns
    -------
    Feature-enriched DataFrame.
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)

    if drop_na:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Dropped {before - len(df)} NaN rows after feature engineering.")

    logger.info(f"Feature engineering complete → {df.shape[1]} columns, {len(df)} rows.")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (everything except Date and Sales)."""
    return [c for c in df.columns if c not in [DATE_COL, TARGET_COL]]
