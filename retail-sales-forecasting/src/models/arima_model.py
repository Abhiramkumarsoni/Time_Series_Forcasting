"""
arima_model.py

ARIMA / SARIMA forecasting model.

Interface
---------
    model = ARIMAModel(order=(5,1,2))
    model.fit(train_series)
    preds = model.predict(steps=30)
    model.save("models/arima.pkl")
    model.load("models/arima.pkl")
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import ARIMA_ORDER, SARIMA_ORDER, SARIMA_SEASONAL
from src.logger import logger


class ARIMAModel:
    """
    Wrapper around statsmodels ARIMA / SARIMA.

    Attributes
    ----------
    order   : (p, d, q) tuple for ARIMA.
    seasonal: If True, fits SARIMA with config.SARIMA_SEASONAL.
    """

    def __init__(
        self,
        order: tuple = ARIMA_ORDER,
        seasonal: bool = False,
        seasonal_order: tuple = SARIMA_SEASONAL,
    ):
        self.order          = order
        self.seasonal       = seasonal
        self.seasonal_order = seasonal_order
        self._fitted: ARIMAResults | None = None

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, series: pd.Series | np.ndarray) -> "ARIMAModel":
        """
        Fit ARIMA (or SARIMA) on the training series.

        Parameters
        ----------
        series : 1-D time series of sales values (unscaled or scaled).
        """
        logger.info(f"Fitting {'SARIMA' if self.seasonal else 'ARIMA'} "
                    f"order={self.order}...")

        if self.seasonal:
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            model = ARIMA(series, order=self.order)

        self._fitted = model.fit()
        logger.info("ARIMA fitting complete.")
        return self

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, steps: int = 30) -> pd.DataFrame:
        """
        Forecast `steps` periods ahead.

        Returns
        -------
        DataFrame with columns [Step, Forecast, Lower_CI, Upper_CI].
        """
        if self._fitted is None:
            raise RuntimeError("Call fit() before predict().")

        forecast  = self._fitted.get_forecast(steps=steps)
        mean      = forecast.predicted_mean
        ci        = forecast.conf_int(alpha=0.05)

        return pd.DataFrame({
            "Step":     np.arange(1, steps + 1),
            "Forecast": mean.values,
            "Lower_CI": ci.iloc[:, 0].values,
            "Upper_CI": ci.iloc[:, 1].values,
        })

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        joblib.dump(self._fitted, path)
        logger.info(f"ARIMA model saved → {path}")

    def load(self, path: str | Path) -> "ARIMAModel":
        self._fitted = joblib.load(path)
        logger.info(f"ARIMA model loaded ← {path}")
        return self

    @property
    def summary(self) -> str:
        if self._fitted is None:
            return "Model not fitted yet."
        return str(self._fitted.summary())
