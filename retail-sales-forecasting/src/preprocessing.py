"""
preprocessing.py

Clean and prepare the raw sales DataFrame for modelling.

Steps
-----
1. Parse / validate date column
2. Handle missing dates (fill gaps)
3. Cap extreme outliers (IQR method)
4. Scale the target (MinMaxScaler)
5. Train / test split (chronological, no shuffle)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    DATE_COL, TARGET_COL,
    TEST_SIZE, PROC_DIR,
)
from src.logger import logger


# ─── Public API ───────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing dates, fill gaps, and clip outliers.

    Parameters
    ----------
    df : Raw DataFrame with [Date, Sales] columns.

    Returns
    -------
    Cleaned DataFrame indexed by date.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(DATE_COL)

    # ── Fill missing calendar days ────────────────────────────────────────────
    full_range = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    df = df.set_index(DATE_COL).reindex(full_range)
    df.index.name = DATE_COL
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="time")
    df = df.reset_index()

    # ── Clip outliers (IQR × 3) ───────────────────────────────────────────────
    q1, q3 = df[TARGET_COL].quantile([0.25, 0.75])
    iqr     = q3 - q1
    lower   = max(0, q1 - 3 * iqr)
    upper   =      q3 + 3 * iqr
    clipped = df[TARGET_COL].clip(lower, upper)
    n_clipped = (df[TARGET_COL] != clipped).sum()
    if n_clipped:
        logger.info(f"Clipped {n_clipped} outlier values.")
    df[TARGET_COL] = clipped

    logger.info(f"clean_data complete → shape {df.shape}")
    return df


def train_test_split_ts(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split (no shuffle).

    Parameters
    ----------
    df        : Cleaned DataFrame.
    test_size : Fraction of rows reserved for the test set.

    Returns
    -------
    (train_df, test_df)
    """
    n      = len(df)
    cutoff = int(n * (1 - test_size))
    train  = df.iloc[:cutoff].copy()
    test   = df.iloc[cutoff:].copy()
    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


def scale_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str = TARGET_COL,
    save_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Fit MinMaxScaler on train, transform both train and test.

    Parameters
    ----------
    train     : Training DataFrame.
    test      : Test DataFrame.
    col       : Column to scale.
    save_path : Optional path to persist the fitted scaler.

    Returns
    -------
    (scaled_train, scaled_test, scaler)
    """
    scaler = MinMaxScaler()
    train  = train.copy()
    test   = test.copy()

    train[col] = scaler.fit_transform(train[[col]])
    test[col]  = scaler.transform(test[[col]])

    if save_path:
        joblib.dump(scaler, save_path)
        logger.info(f"Scaler saved → {save_path}")

    logger.info("Target scaled to [0, 1] using MinMaxScaler.")
    return train, test, scaler


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Full preprocessing pipeline: clean → split → scale.

    Returns
    -------
    (train_df, test_df, scaler)
    """
    df               = clean_data(df)
    train, test      = train_test_split_ts(df)
    scaler_path      = PROC_DIR / "scaler.pkl"
    train, test, scaler = scale_target(train, test, save_path=str(scaler_path))

    # Save processed datasets
    train.to_csv(PROC_DIR / "train.csv", index=False)
    test.to_csv(PROC_DIR  / "test.csv",  index=False)
    logger.info("Preprocessed data saved to data/processed/")

    return train, test, scaler
