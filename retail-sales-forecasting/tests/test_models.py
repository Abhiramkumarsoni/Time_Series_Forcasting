"""
tests/test_models.py

Lightweight smoke tests for each model wrapper.
These tests check that the fit/predict/save/load interface works
without requiring a full training run.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_series():
    """100-point synthetic series for quick model tests."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.uniform(100, 300, size=100))


@pytest.fixture
def tiny_df(tiny_series):
    dates = pd.date_range("2023-01-01", periods=len(tiny_series), freq="D")
    return pd.DataFrame({"Date": dates, "Sales": tiny_series.values})


@pytest.fixture
def tiny_features(tiny_df):
    from src.feature_engineering import build_features, get_feature_columns
    feat = build_features(tiny_df, drop_na=True)
    cols = get_feature_columns(feat)
    return feat[cols], feat["Sales"]


# ── ARIMA ─────────────────────────────────────────────────────────────────────

class TestARIMAModel:

    def test_fit_predict(self, tiny_series):
        from src.models.arima_model import ARIMAModel
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(tiny_series)
        result = model.predict(steps=5)
        assert len(result) == 5
        assert "Forecast" in result.columns

    def test_save_load(self, tiny_series):
        from src.models.arima_model import ARIMAModel
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(tiny_series)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        model.save(path)
        loaded = ARIMAModel()
        loaded.load(path)
        result = loaded._fitted.forecast(steps=3)
        assert len(result) == 3

    def test_predict_before_fit_raises(self):
        from src.models.arima_model import ARIMAModel
        with pytest.raises(RuntimeError):
            ARIMAModel().predict()


# ── XGBoost ───────────────────────────────────────────────────────────────────

class TestXGBoostModel:

    def test_fit_predict(self, tiny_features):
        from src.models.xgboost_model import XGBoostModel
        X, y = tiny_features
        model = XGBoostModel(params={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_feature_importance(self, tiny_features):
        from src.models.xgboost_model import XGBoostModel
        X, y = tiny_features
        model = XGBoostModel(params={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        fi = model.feature_importance(feature_names=list(X.columns))
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == X.shape[1]

    def test_save_load(self, tiny_features):
        from src.models.xgboost_model import XGBoostModel
        X, y = tiny_features
        model = XGBoostModel(params={"n_estimators": 5, "random_state": 42})
        model.fit(X, y)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        model.save(path)
        loaded = XGBoostModel()
        loaded.load(path)
        assert len(loaded.predict(X)) == len(X)


# ── Data Loader ───────────────────────────────────────────────────────────────

class TestDataLoader:

    def test_generate_returns_correct_shape(self):
        from src.data_loader import generate_synthetic_data
        df = generate_synthetic_data(periods=50, save=False)
        assert len(df) == 50
        assert "Date" in df.columns
        assert "Sales" in df.columns

    def test_sales_all_positive(self):
        from src.data_loader import generate_synthetic_data
        df = generate_synthetic_data(periods=100, save=False)
        assert (df["Sales"] > 0).all()

    def test_dates_are_daily(self):
        from src.data_loader import generate_synthetic_data
        df = generate_synthetic_data(periods=30, save=False)
        diffs = df["Date"].diff().dropna()
        assert (diffs == pd.Timedelta("1D")).all()
