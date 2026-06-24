"""
xgboost_model.py

XGBoost regression model for time-series forecasting.

The model is trained on engineered tabular features (lags, rolling
statistics, calendar features) produced by src.feature_engineering.

Interface
---------
    model = XGBoostModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model.save("models/xgboost.pkl")
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor

from src.config import XGB_PARAMS
from src.logger import logger


class XGBoostModel:
    """
    XGBoost wrapper for tabular retail-sales forecasting.

    Parameters
    ----------
    params : XGBRegressor keyword arguments.
             Defaults to config.XGB_PARAMS.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or XGB_PARAMS
        self._model = XGBRegressor(**self.params)

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val:   pd.DataFrame | np.ndarray | None = None,
        y_val:   pd.Series   | np.ndarray | None = None,
    ) -> "XGBoostModel":
        """
        Train the XGBoost regressor.

        Parameters
        ----------
        X_train : Feature matrix for training.
        y_train : Target vector for training.
        X_val   : Optional validation features (for early stopping).
        y_val   : Optional validation target.
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        logger.info(f"Fitting XGBoost with params={self.params} …")
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info("XGBoost fitting complete.")
        return self

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return predictions as a 1-D NumPy array."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    # ── feature importance ────────────────────────────────────────────────────
    def feature_importance(
        self,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted descending.
        """
        scores = self._model.feature_importances_
        names  = feature_names or [f"f{i}" for i in range(len(scores))]
        return (
            pd.DataFrame({"feature": names, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        joblib.dump(self._model, path)
        logger.info(f"XGBoost model saved → {path}")

    def load(self, path: str | Path) -> "XGBoostModel":
        self._model = joblib.load(path)
        logger.info(f"XGBoost model loaded ← {path}")
        return self
