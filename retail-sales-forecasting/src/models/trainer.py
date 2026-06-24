"""
trainer.py

Orchestrates training of all four models end-to-end.

Usage (CLI)
-----------
    python -m src.models.trainer

Usage (Python)
--------------
    from src.models.trainer import Trainer
    trainer = Trainer()
    trainer.run_all()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    MODEL_DIR, TARGET_COL, DATE_COL,
    LSTM_LOOK_BACK, PROC_DIR,
)
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.feature_engineering import build_features, get_feature_columns
from src.evaluation import evaluate
from src.utils import create_lstm_sequences, inverse_scale, timer
from src.logger import logger

from src.models.arima_model   import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model    import LSTMModel


class Trainer:
    """
    End-to-end training pipeline.

    Attributes
    ----------
    model_dir : Directory where trained models are persisted.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    def train_arima(self, train: pd.DataFrame, test: pd.DataFrame, scaler) -> dict:
        with timer("ARIMA training"):
            model = ARIMAModel()
            model.fit(train[TARGET_COL])
            model.save(self.model_dir / "arima.pkl")

        result_df = model.predict(steps=len(test))
        y_pred    = inverse_scale(result_df["Forecast"].values, scaler)
        y_true    = inverse_scale(test[TARGET_COL].values, scaler)
        metrics   = evaluate(y_true, y_pred, "arima")
        return metrics

    # ── Prophet ───────────────────────────────────────────────────────────────
    def train_prophet(self, train: pd.DataFrame, test: pd.DataFrame, scaler) -> dict:
        with timer("Prophet training"):
            model = ProphetModel()
            model.fit(train)
            model.save(self.model_dir / "prophet.pkl")

        result_df = model.predict(periods=len(test))
        y_pred    = inverse_scale(result_df["yhat"].values, scaler)
        y_true    = inverse_scale(test[TARGET_COL].values, scaler)
        metrics   = evaluate(y_true, y_pred, "prophet")
        return metrics

    # ── XGBoost ───────────────────────────────────────────────────────────────
    def train_xgboost(self, train_feat: pd.DataFrame, test_feat: pd.DataFrame, scaler) -> dict:
        feature_cols = get_feature_columns(train_feat)
        X_train = train_feat[feature_cols]
        y_train = train_feat[TARGET_COL]
        X_test  = test_feat[feature_cols]
        y_test  = test_feat[TARGET_COL]

        with timer("XGBoost training"):
            model = XGBoostModel()
            model.fit(X_train, y_train)
            model.save(self.model_dir / "xgboost.pkl")

        y_pred  = inverse_scale(model.predict(X_test), scaler)
        y_true  = inverse_scale(y_test.values, scaler)
        metrics = evaluate(y_true, y_pred, "xgboost")
        return metrics

    # ── LSTM ──────────────────────────────────────────────────────────────────
    def train_lstm(self, train: pd.DataFrame, test: pd.DataFrame, scaler) -> dict:
        series_train = train[TARGET_COL].values
        series_test  = test[TARGET_COL].values

        X_train, y_train = create_lstm_sequences(series_train, LSTM_LOOK_BACK)
        X_test,  y_test  = create_lstm_sequences(
            np.concatenate([series_train[-LSTM_LOOK_BACK:], series_test]),
            LSTM_LOOK_BACK,
        )

        with timer("LSTM training"):
            model = LSTMModel()
            model.fit(X_train, y_train)
            model.save(self.model_dir / "lstm.keras")

        y_pred  = inverse_scale(model.predict(X_test), scaler)
        y_true  = inverse_scale(y_test, scaler)
        metrics = evaluate(y_true, y_pred, "lstm")
        return metrics

    # ── run all ───────────────────────────────────────────────────────────────
    def run_all(self) -> dict[str, dict]:
        """
        Run the full training pipeline for all four models.

        Returns
        -------
        Dict mapping model name → metrics dict.
        """
        logger.info("=" * 60)
        logger.info("Starting full training pipeline …")
        logger.info("=" * 60)

        # 1. Load & preprocess
        raw_df = load_data()
        train, test, scaler = preprocess(raw_df)

        # 2. Feature-engineered versions (for XGBoost)
        full_feat   = build_features(
            pd.concat([train, test]).reset_index(drop=True)
        )
        cutoff      = len(train)
        train_feat  = full_feat.iloc[:cutoff]
        test_feat   = full_feat.iloc[cutoff:]

        all_metrics = {}

        all_metrics["arima"]   = self.train_arima(train, test, scaler)
        all_metrics["prophet"] = self.train_prophet(train, test, scaler)
        all_metrics["xgboost"] = self.train_xgboost(train_feat, test_feat, scaler)
        all_metrics["lstm"]    = self.train_lstm(train, test, scaler)

        logger.info("=" * 60)
        logger.info("All models trained successfully!")
        for name, m in all_metrics.items():
            logger.info(f"  {name:10s} RMSE={m['rmse']:.3f} | MAE={m['mae']:.3f}")
        logger.info("=" * 60)

        return all_metrics


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Trainer().run_all()
