"""
prophet_model.py

Facebook Prophet forecasting model wrapper.

Interface
---------
    model = ProphetModel()
    model.fit(train_df)          # DataFrame with [Date, Sales]
    preds = model.predict(60)    # 60-day forecast
    model.save("models/prophet.pkl")
"""

from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

from src.config import (
    DATE_COL, TARGET_COL,
    PROPHET_YEARLY, PROPHET_WEEKLY, PROPHET_DAILY,
    PROPHET_CHANGEPOINT_PRIOR,
)
from src.logger import logger


class ProphetModel:
    """
    Thin wrapper around Facebook Prophet.

    The fit() method expects a DataFrame with [Date, Sales] columns;
    it internally renames them to the ds / y columns Prophet requires.
    """

    def __init__(
        self,
        yearly_seasonality: bool = PROPHET_YEARLY,
        weekly_seasonality: bool = PROPHET_WEEKLY,
        daily_seasonality:  bool = PROPHET_DAILY,
        changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR,
    ):
        self.yearly_seasonality      = yearly_seasonality
        self.weekly_seasonality      = weekly_seasonality
        self.daily_seasonality       = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._model = None

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "ProphetModel":
        """
        Train Prophet on the provided DataFrame.

        Parameters
        ----------
        df : DataFrame with columns [Date, Sales].
        """
        from prophet import Prophet   # lazy import (heavy)

        prophet_df = df[[DATE_COL, TARGET_COL]].rename(
            columns={DATE_COL: "ds", TARGET_COL: "y"}
        )

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        logger.info("Fitting Prophet model …")
        self._model.fit(prophet_df)
        logger.info("Prophet fitting complete.")
        return self

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """
        Forecast `periods` days into the future.

        Returns
        -------
        DataFrame with columns [ds, yhat, yhat_lower, yhat_upper].
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        future   = self._model.make_future_dataframe(periods=periods, freq="D")
        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        joblib.dump(self._model, path)
        logger.info(f"Prophet model saved → {path}")

    def load(self, path: str | Path) -> "ProphetModel":
        self._model = joblib.load(path)
        logger.info(f"Prophet model loaded ← {path}")
        return self
