"""
tests/test_preprocessing.py

Unit tests for src.preprocessing
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean_data, train_test_split_ts, scale_target


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """200-day clean DataFrame."""
    dates  = pd.date_range("2022-01-01", periods=200, freq="D")
    sales  = np.random.default_rng(0).uniform(100, 300, size=200)
    return pd.DataFrame({"Date": dates, "Sales": sales})


@pytest.fixture
def df_with_gaps(sample_df):
    """Same DataFrame but with some rows removed (simulated missing dates)."""
    idx = list(range(200))
    idx.pop(5)   # remove day 5
    idx.pop(50)  # remove day 50
    return sample_df.iloc[idx].reset_index(drop=True)


@pytest.fixture
def df_with_outliers(sample_df):
    """DataFrame with extreme outlier values."""
    df = sample_df.copy()
    df.loc[10, "Sales"] = 99999.0
    df.loc[50, "Sales"] = -500.0
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCleanData:

    def test_output_has_expected_columns(self, sample_df):
        cleaned = clean_data(sample_df)
        assert "Date"  in cleaned.columns
        assert "Sales" in cleaned.columns

    def test_no_missing_dates(self, df_with_gaps):
        cleaned = clean_data(df_with_gaps)
        full_range = pd.date_range(
            cleaned["Date"].min(), cleaned["Date"].max(), freq="D"
        )
        assert len(cleaned) == len(full_range)

    def test_no_nan_after_clean(self, df_with_gaps):
        cleaned = clean_data(df_with_gaps)
        assert cleaned["Sales"].isna().sum() == 0

    def test_outliers_clipped(self, df_with_outliers):
        cleaned = clean_data(df_with_outliers)
        # No value should exceed reasonable bounds
        assert cleaned["Sales"].max() < 99999.0
        assert cleaned["Sales"].min() >= 0.0

    def test_sorted_by_date(self, sample_df):
        shuffled = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
        cleaned  = clean_data(shuffled)
        assert cleaned["Date"].is_monotonic_increasing


class TestTrainTestSplit:

    def test_split_sizes(self, sample_df):
        cleaned       = clean_data(sample_df)
        train, test   = train_test_split_ts(cleaned, test_size=0.2)
        assert len(train) + len(test) == len(cleaned)

    def test_chronological_order(self, sample_df):
        cleaned     = clean_data(sample_df)
        train, test = train_test_split_ts(cleaned)
        assert train["Date"].max() < test["Date"].min()

    def test_no_data_leakage(self, sample_df):
        cleaned     = clean_data(sample_df)
        train, test = train_test_split_ts(cleaned)
        overlap = set(train.index) & set(test.index)
        assert len(overlap) == 0


class TestScaleTarget:

    def test_scaled_in_zero_one(self, sample_df):
        cleaned = clean_data(sample_df)
        train, test = train_test_split_ts(cleaned)
        s_train, s_test, scaler = scale_target(train, test)
        assert s_train["Sales"].min() >= 0.0
        assert s_train["Sales"].max() <= 1.0

    def test_scaler_returned(self, sample_df):
        cleaned = clean_data(sample_df)
        train, test = train_test_split_ts(cleaned)
        _, _, scaler = scale_target(train, test)
        assert hasattr(scaler, "inverse_transform")
