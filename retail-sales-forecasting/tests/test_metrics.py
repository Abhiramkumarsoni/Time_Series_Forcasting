"""
tests/test_metrics.py

Unit tests for src.evaluation
"""

import numpy as np
import pytest

from src.evaluation import mae, rmse, mape, r2, evaluate


@pytest.fixture
def perfect_prediction():
    y = np.array([100.0, 200.0, 150.0, 175.0])
    return y, y.copy()


@pytest.fixture
def noisy_prediction():
    y_true = np.array([100.0, 200.0, 150.0, 175.0, 120.0])
    y_pred = np.array([110.0, 190.0, 160.0, 165.0, 130.0])
    return y_true, y_pred


class TestIndividualMetrics:

    def test_mae_perfect(self, perfect_prediction):
        y_true, y_pred = perfect_prediction
        assert mae(y_true, y_pred) == pytest.approx(0.0)

    def test_rmse_perfect(self, perfect_prediction):
        y_true, y_pred = perfect_prediction
        assert rmse(y_true, y_pred) == pytest.approx(0.0)

    def test_mape_perfect(self, perfect_prediction):
        y_true, y_pred = perfect_prediction
        assert mape(y_true, y_pred) == pytest.approx(0.0)

    def test_r2_perfect(self, perfect_prediction):
        y_true, y_pred = perfect_prediction
        assert r2(y_true, y_pred) == pytest.approx(1.0)

    def test_mae_positive(self, noisy_prediction):
        y_true, y_pred = noisy_prediction
        assert mae(y_true, y_pred) > 0

    def test_rmse_geq_mae(self, noisy_prediction):
        y_true, y_pred = noisy_prediction
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)

    def test_mape_percentage(self, noisy_prediction):
        y_true, y_pred = noisy_prediction
        result = mape(y_true, y_pred)
        assert 0 < result < 100

    def test_r2_leq_one(self, noisy_prediction):
        y_true, y_pred = noisy_prediction
        assert r2(y_true, y_pred) <= 1.0

    def test_mape_no_divide_by_zero():
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 190.0])
        result = mape(y_true, y_pred)
        assert np.isfinite(result)


class TestEvaluate:

    def test_returns_dict_with_all_keys(self, noisy_prediction):
        y_true, y_pred = noisy_prediction
        result = evaluate(y_true, y_pred, "test_model", save=False)
        assert "mae"  in result
        assert "rmse" in result
        assert "mape" in result
        assert "r2"   in result

    def test_all_values_are_floats(self, noisy_prediction):
        y_true, y_pred = noisy_prediction
        result = evaluate(y_true, y_pred, "test_model", save=False)
        for v in result.values():
            assert isinstance(v, float)

    def test_accepts_pandas_series(self):
        import pandas as pd
        y_true = pd.Series([100.0, 200.0, 150.0])
        y_pred = pd.Series([105.0, 195.0, 155.0])
        result = evaluate(y_true, y_pred, "pandas_test", save=False)
        assert result["mae"] > 0
