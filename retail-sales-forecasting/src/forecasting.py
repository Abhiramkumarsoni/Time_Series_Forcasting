"""
forecasting.py

High-level forecasting interface that loads a saved model and
returns future predictions.

Supported models: arima, prophet, xgboost, lstm
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import MODEL_DIR, DATE_COL, TARGET_COL, LSTM_LOOK_BACK
from src.logger import logger


class Forecaster:
    """
    Load a saved model by name and generate forecasts.

    Parameters
    ----------
    model_name : One of 'arima', 'prophet', 'xgboost', 'lstm'.
    model_dir  : Directory where model files are saved.
    """

    _FILE_MAP = {
        "arima":   "arima.pkl",
        "prophet": "prophet.pkl",
        "xgboost": "xgboost.pkl",
        "lstm":    "lstm.keras",
    }

    def __init__(self, model_name: str, model_dir: Path = MODEL_DIR):
        self.model_name = model_name.lower()
        self.model_dir  = Path(model_dir)
        self._model     = self._load_model()

    def _load_model(self):
        fname = self._FILE_MAP.get(self.model_name)
        if fname is None:
            raise ValueError(f"Unknown model '{self.model_name}'. "
                             f"Choose from: {list(self._FILE_MAP)}")

        path = self.model_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}\n"
                                    "Run the trainer first: python -m src.models.trainer")

        if self.model_name == "lstm":
            from tensorflow.keras.models import load_model
            model = load_model(path)
        else:
            model = joblib.load(path)

        logger.info(f"{self.model_name.upper()} model loaded ← {path}")
        return model

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def forecast(self, **kwargs) -> pd.DataFrame:
        """
        Generate a forecast.  Keyword arguments are model-specific:

        arima   : steps (int)
        prophet : periods (int)
        xgboost : features (pd.DataFrame)
        lstm    : series (np.ndarray), scaler (optional)
        """
        dispatch = {
            "arima":   self._forecast_arima,
            "prophet": self._forecast_prophet,
            "xgboost": self._forecast_xgboost,
            "lstm":    self._forecast_lstm,
        }
        return dispatch[self.model_name](**kwargs)

    def _forecast_arima(self, steps: int = 30) -> pd.DataFrame:
        fc    = self._model.forecast(steps=steps)
        ci    = self._model.get_forecast(steps=steps).conf_int()
        return pd.DataFrame({
            "Step":     np.arange(1, steps + 1),
            "Forecast": fc.values,
            "Lower_CI": ci.iloc[:, 0].values,
            "Upper_CI": ci.iloc[:, 1].values,
        })

    def _forecast_prophet(self, periods: int = 30) -> pd.DataFrame:
        future   = self._model.make_future_dataframe(periods=periods, freq="D")
        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)

    def _forecast_xgboost(self, features: pd.DataFrame) -> pd.DataFrame:
        preds = self._model.predict(features)
        return pd.DataFrame({"Prediction": preds})

    def _forecast_lstm(
        self,
        series: np.ndarray,
        scaler=None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        series : Last `look_back` scaled sales values, shape (look_back,).
        scaler : Optional fitted MinMaxScaler for inverse transform.
        """
        X     = series[-LSTM_LOOK_BACK:].reshape(1, LSTM_LOOK_BACK, 1)
        pred  = self._model.predict(X, verbose=0).flatten()
        if scaler is not None:
            pred = scaler.inverse_transform([[pred[0]]])[0]
        return pd.DataFrame({"Prediction": pred})
