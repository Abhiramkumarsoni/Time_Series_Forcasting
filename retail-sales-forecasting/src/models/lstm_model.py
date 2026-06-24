"""
lstm_model.py

LSTM (Long Short-Term Memory) model for time-series forecasting.

The model takes sequences of `look_back` past sales values and
predicts the next value.

Interface
---------
    model = LSTMModel(look_back=30)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model.save("models/lstm.keras")
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.config import (
    LSTM_LOOK_BACK, LSTM_UNITS,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_PATIENCE,
)
from src.logger import logger


class LSTMModel:
    """
    Keras LSTM wrapper for univariate time-series forecasting.

    The model architecture:
        LSTM(units) → Dense(32, relu) → Dense(1)

    Parameters
    ----------
    look_back   : Number of past time-steps used as input.
    units       : Number of LSTM hidden units.
    epochs      : Maximum training epochs.
    batch_size  : Mini-batch size.
    patience    : Early-stopping patience (in epochs).
    """

    def __init__(
        self,
        look_back:  int = LSTM_LOOK_BACK,
        units:      int = LSTM_UNITS,
        epochs:     int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        patience:   int = LSTM_PATIENCE,
    ):
        self.look_back  = look_back
        self.units      = units
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self._model     = None

    # ── build ─────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        """Construct the Keras model graph."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(self.units, input_shape=(self.look_back, 1), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        self._model = model
        logger.info(f"LSTM model built: look_back={self.look_back}, units={self.units}")

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LSTMModel":
        """
        Train the LSTM model.

        Parameters
        ----------
        X_train : shape (samples, look_back, 1)
        y_train : shape (samples,)
        X_val   : Optional validation set.
        y_val   : Optional validation labels.
        """
        from tensorflow.keras.callbacks import EarlyStopping

        if self._model is None:
            self._build()

        callbacks = [
            EarlyStopping(monitor="val_loss" if X_val is not None else "loss",
                          patience=self.patience,
                          restore_best_weights=True,
                          verbose=0)
        ]

        val_data = (X_val, y_val) if X_val is not None else None

        logger.info(f"Training LSTM for up to {self.epochs} epochs …")
        history = self._model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info(f"LSTM training finished at epoch {len(history.history['loss'])}.")
        return self

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as a 1-D NumPy array."""
        if self._model is None:
            raise RuntimeError("Call fit() (or load()) before predict().")
        return self._model.predict(X, verbose=0).flatten()

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        self._model.save(path)
        logger.info(f"LSTM model saved → {path}")

    def load(self, path: str | Path) -> "LSTMModel":
        from tensorflow.keras.models import load_model
        self._model = load_model(path)
        logger.info(f"LSTM model loaded ← {path}")
        return self
