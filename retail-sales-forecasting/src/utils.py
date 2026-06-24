"""
utils.py

Miscellaneous helpers used across the project.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import logger


@contextmanager
def timer(label: str = ""):
    """Context manager that logs elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{label} completed in {elapsed:.2f}s")


def create_lstm_sequences(
    series: np.ndarray,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D time series into (X, y) pairs for LSTM training.

    Parameters
    ----------
    series    : 1-D array of values (already scaled).
    look_back : Number of past time-steps used as input.

    Returns
    -------
    X : shape (n_samples, look_back, 1)
    y : shape (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i : i + look_back])
        y.append(series[i + look_back])
    return np.array(X).reshape(-1, look_back, 1), np.array(y)


def inverse_scale(
    values: np.ndarray,
    scaler,
    col_index: int = 0,
) -> np.ndarray:
    """
    Inverse-transform a 1-D array that was scaled with MinMaxScaler.

    The scaler may have been fitted on multiple columns; `col_index`
    selects the relevant one (default = 0 for single-column scalers).
    """
    dummy = np.zeros((len(values), scaler.n_features_in_))
    dummy[:, col_index] = values.flatten()
    return scaler.inverse_transform(dummy)[:, col_index]


def summarise_df(df: pd.DataFrame) -> dict:
    """Return a compact summary dict for a DataFrame (used in Streamlit)."""
    return {
        "rows":       len(df),
        "columns":    df.shape[1],
        "date_range": f"{df.iloc[0, 0].date()} → {df.iloc[-1, 0].date()}" if len(df) else "N/A",
        "min_sales":  round(float(df.iloc[:, 1].min()), 2),
        "max_sales":  round(float(df.iloc[:, 1].max()), 2),
        "mean_sales": round(float(df.iloc[:, 1].mean()), 2),
    }
