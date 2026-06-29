"""Evaluation metrics for RIP prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC only when both classes are present."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC only when positives exist."""
    if np.sum(y_true) == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the primary classification metrics for RIP."""
    probs = probabilities.reshape(-1)
    y_true = labels.reshape(-1).astype(float)
    # Filter out NaN values from predictions
    valid = np.isfinite(probs) & np.isfinite(y_true)
    probs = probs[valid]
    y_true = y_true.astype(int)
    if len(probs) == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "brier": float("nan"), "accuracy": 0.0, "f1": 0.0,
                "calibration": {"prob_true": [], "prob_pred": []}}
    y_true = y_true[valid]
    preds = (probs >= threshold).astype(int)
    n_bins = min(5, max(2, len(np.unique(probs))))
    try:
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
    except ValueError:
        prob_true, prob_pred = np.array([]), np.array([])
    return {
        "roc_auc": safe_auc(y_true, probs),
        "pr_auc": safe_average_precision(y_true, probs),
        "brier": float(brier_score_loss(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "calibration": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }


def dice_coefficient(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """Compute region-level Dice as an optional secondary metric."""
    preds = (probabilities.reshape(-1) >= threshold).astype(int)
    truth = labels.reshape(-1).astype(int)
    denom = float(np.sum(preds) + np.sum(truth))
    if denom == 0.0:
        return 1.0
    return 2.0 * float(np.sum(preds * truth)) / denom
