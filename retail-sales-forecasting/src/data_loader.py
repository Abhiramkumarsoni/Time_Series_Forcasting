"""
data_loader.py

Generates synthetic retail-sales data (if needed) and loads it
into a clean DataFrame ready for the rest of the pipeline.

The synthetic data mimics a real retail time-series:
  - Upward trend
  - Weekly seasonality (weekends sell more)
  - Yearly seasonality (Nov–Dec holiday boost)
  - Random Gaussian noise
  - Occasional sales spikes (promotions / events)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATE_COL, TARGET_COL, RAW_FILE
from src.logger import logger


# ─── Synthetic Data Generator ─────────────────────────────────────────────────

def generate_synthetic_data(
    start: str = "2021-01-01",
    periods: int = 1461,          # 4 years of daily data
    seed: int = 42,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic daily retail-sales time series.

    Parameters
    ----------
    start   : First date of the series (YYYY-MM-DD).
    periods : Number of daily observations (default = 4 years = 1461).
    seed    : Random seed for reproducibility.
    save    : Whether to persist the CSV to data/raw/.

    Returns
    -------
    pd.DataFrame with columns [Date, Sales].
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="D")

    base   = 200.0
    trend  = np.arange(periods) * 0.05          # slow upward drift

    # Weekly seasonality – weekends bump sales
    weekly = np.where(dates.dayofweek >= 5, 25, -5)

    # Yearly seasonality – Nov / Dec holiday season
    yearly = np.where(dates.month.isin([11, 12]), 40, 0)

    # Gaussian noise
    noise  = rng.normal(0, 10, size=periods)

    # Occasional promotional spikes (~1 % of days)
    spike_mask  = rng.random(periods) < 0.01
    spikes      = rng.normal(50, 15, size=periods) * spike_mask

    sales = np.clip(base + trend + weekly + yearly + noise + spikes, 20, None)
    sales = np.round(sales, 2)

    df = pd.DataFrame({DATE_COL: dates, TARGET_COL: sales})

    if save:
        RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_FILE, index=False)
        logger.info(f"Synthetic data saved → {RAW_FILE}")

    logger.info(f"Generated {len(df):,} rows of synthetic retail-sales data.")
    return df


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load sales data from CSV.  If the file does not exist, generate
    synthetic data on the fly.

    Parameters
    ----------
    filepath : Path to CSV.  Defaults to ``config.RAW_FILE``.

    Returns
    -------
    pd.DataFrame with a DatetimeIndex.
    """
    path = Path(filepath) if filepath else RAW_FILE

    if not path.exists():
        logger.warning(f"{path} not found – generating synthetic data.")
        df = generate_synthetic_data(save=True)
    else:
        df = pd.read_csv(path, parse_dates=[DATE_COL])
        logger.info(f"Loaded {len(df):,} rows from {path}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df
