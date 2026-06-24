"""
tests/test_features.py

Unit tests for src.feature_engineering
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
    build_features,
    get_feature_columns,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    sales = np.random.default_rng(1).uniform(100, 300, size=100)
    return pd.DataFrame({"Date": dates, "Sales": sales})


class TestLagFeatures:

    def test_lag_columns_created(self, sample_df):
        result = add_lag_features(sample_df)
        for lag in [1, 7, 14, 30]:
            assert f"lag_{lag}" in result.columns

    def test_lag_shifts_correctly(self, sample_df):
        result = add_lag_features(sample_df)
        # lag_1 at row i == Sales at row i-1
        assert result["lag_1"].iloc[1] == pytest.approx(sample_df["Sales"].iloc[0])

    def test_no_original_column_modified(self, sample_df):
        original = sample_df["Sales"].copy()
        add_lag_features(sample_df)
        pd.testing.assert_series_equal(sample_df["Sales"], original)


class TestRollingFeatures:

    def test_rolling_columns_created(self, sample_df):
        result = add_rolling_features(sample_df)
        for win in [7, 14, 30]:
            assert f"rolling_mean_{win}" in result.columns
            assert f"rolling_std_{win}"  in result.columns

    def test_rolling_mean_not_negative(self, sample_df):
        result = add_rolling_features(sample_df)
        # means should be positive (sales are positive)
        assert result["rolling_mean_7"].dropna().min() > 0


class TestCalendarFeatures:

    def test_calendar_columns_present(self, sample_df):
        result = add_calendar_features(sample_df)
        for col in ["day_of_week", "month", "quarter", "is_weekend", "is_holiday_season"]:
            assert col in result.columns

    def test_is_weekend_values(self, sample_df):
        result = add_calendar_features(sample_df)
        assert result["is_weekend"].isin([0, 1]).all()

    def test_month_range(self, sample_df):
        result = add_calendar_features(sample_df)
        assert result["month"].between(1, 12).all()


class TestBuildFeatures:

    def test_no_nan_after_dropna(self, sample_df):
        result = build_features(sample_df, drop_na=True)
        assert result.isnull().sum().sum() == 0

    def test_more_columns_than_original(self, sample_df):
        result = build_features(sample_df)
        assert result.shape[1] > sample_df.shape[1]

    def test_get_feature_columns_excludes_target(self, sample_df):
        result = build_features(sample_df)
        feat_cols = get_feature_columns(result)
        assert "Sales" not in feat_cols
        assert "Date"  not in feat_cols
        assert len(feat_cols) > 0
