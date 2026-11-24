"""Mondrian (class-conditional) conformal p-values and thresholds.

Supports both marginal and class-conditional conformal inference,
plus the per-sample p_wrong metric.
"""
from __future__ import annotations

import numpy as np


def marginal_pvalue(
    test_scores: np.ndarray, cal_scores: np.ndarray,
) -> np.ndarray:
    """Marginal conformal p-values.

    p(x*) = (|{i : s_cal_i >= s*}| + 1) / (n_cal + 1)

    Args:
        test_scores: (N_test,) anomaly scores for test samples.
        cal_scores: (N_cal,) anomaly scores for calibration samples.

    Returns:
        p-values of shape (N_test,).
    """
    n_cal = len(cal_scores)
    cal_sorted = np.sort(cal_scores)
    ranks = n_cal - np.searchsorted(cal_sorted, test_scores, side="left")
    return (ranks + 1) / (n_cal + 1)


def mondrian_pvalue(
    test_scores: np.ndarray,
    test_classes: np.ndarray,
    cal_scores: np.ndarray,
    cal_classes: np.ndarray,
) -> np.ndarray:
    """Mondrian (class-conditional) conformal p-values.

    For each test point with predicted class y_hat, computes the p-value
    against the calibration subset with the same predicted class.

    Args:
        test_scores: (N_test,) anomaly scores.
        test_classes: (N_test,) predicted class labels (int or str).
        cal_scores: (N_cal,) calibration anomaly scores.
        cal_classes: (N_cal,) calibration predicted class labels.

    Returns:
        p-values of shape (N_test,).
    """
    pvals = np.full(len(test_scores), 0.5)  # default for missing classes
    for y in np.unique(test_classes):
        cal_mask = cal_classes == y
        test_mask = test_classes == y
        if not cal_mask.any():
            continue
        pvals[test_mask] = marginal_pvalue(
            test_scores[test_mask], cal_scores[cal_mask]
        )
    return pvals


def mondrian_threshold(
    cal_scores: np.ndarray,
    cal_classes: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Per-class Mondrian threshold at level alpha.

    tau_c = s_(ceil((n_c+1)(1-alpha))) — the finite-sample-corrected
    (1-alpha) empirical quantile of per-class calibration scores.

    Args:
        cal_scores: (N_cal,) calibration anomaly scores.
        cal_classes: (N_cal,) predicted class labels.
        alpha: significance level.

    Returns:
        Dict mapping class label -> threshold.
    """
    thresholds = {}
    for cls in np.unique(cal_classes):
        mask = cal_classes == cls
        scores = np.sort(cal_scores[mask])
        n = len(scores)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        thresholds[cls] = float(scores[k - 1])
    return thresholds


def pwrong(
    test_confidence: np.ndarray,
    test_classes: np.ndarray,
    cal_confidence: np.ndarray,
    cal_classes: np.ndarray,
) -> np.ndarray:
    """Per-sample conformal probability that the prediction is wrong.

    p_wrong(x*) = 1 - p_M(s_clf(x*, y_hat))
    where s_clf = 1 - confidence.

    Args:
        test_confidence: (N_test,) softmax confidence of predicted class.
        test_classes: (N_test,) predicted class labels.
        cal_confidence: (N_cal,) calibration confidence values.
        cal_classes: (N_cal,) calibration predicted class labels.

    Returns:
        p_wrong values of shape (N_test,) in [0, 1].
    """
    test_nonconf = 1.0 - test_confidence
    cal_nonconf = 1.0 - cal_confidence
    pvals = mondrian_pvalue(test_nonconf, test_classes, cal_nonconf, cal_classes)
    return 1.0 - pvals
