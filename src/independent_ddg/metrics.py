"""Regression metrics used consistently across all baselines."""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays with equal shape")
    error = y_pred - y_true
    pearson = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 1 else math.nan
    spearman = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 1 else math.nan
    result = {
        "n": int(len(y_true)),
        "pearson": pearson,
        "spearman": spearman,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }
    binary = y_true > 0.0
    result["auroc"] = (
        float(roc_auc_score(binary.astype(int), y_pred))
        if binary.any() and (~binary).any()
        else math.nan
    )
    return result
