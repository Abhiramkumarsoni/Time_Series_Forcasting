"""
evaluation.py

Model-agnostic evaluation utilities.

Metrics
-------
- MAE   Mean Absolute Error
- RMSE  Root Mean Squared Error
- MAPE  Mean Absolute Percentage Error
- R²    Coefficient of Determination
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import METRIC_DIR
from src.logger import logger


# ─── Core Metrics ─────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoids divide-by-zero)."""
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


# ─── Composite Evaluation ─────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "model",
    save: bool = True,
) -> dict[str, float]:
    """
    Compute all metrics and optionally save them as JSON.

    Parameters
    ----------
    y_true     : Ground-truth values.
    y_pred     : Predicted values.
    model_name : Label used for the output filename.
    save       : Persist metrics to reports/metrics/<model_name>.json.

    Returns
    -------
    Dict with keys: mae, rmse, mape, r2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    results = {
        "mae":  mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
    }

    logger.info(
        f"[{model_name}] MAE={results['mae']:.3f} | "
        f"RMSE={results['rmse']:.3f} | "
        f"MAPE={results['mape']:.2f}% | "
        f"R²={results['r2']:.4f}"
    )

    if save:
        out = METRIC_DIR / f"{model_name}_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved → {out}")

    return results


def load_all_metrics(metric_dir: Path | None = None) -> pd.DataFrame:
    """
    Load all saved metric JSON files into one comparison DataFrame.

    Returns
    -------
    DataFrame with one row per model.
    """
    metric_dir = metric_dir or METRIC_DIR
    rows = []
    for path in sorted(metric_dir.glob("*_metrics.json")):
        with open(path) as f:
            data = json.load(f)
        model = path.stem.replace("_metrics", "")
        rows.append({"model": model, **data})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("rmse").reset_index(drop=True)
    return df
