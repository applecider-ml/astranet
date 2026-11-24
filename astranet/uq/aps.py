"""Adaptive Prediction Sets (APS) — numpy-native implementation.

References:
    Romano, Sesia, Candes (2020). "Classification with Valid Adaptive
    Coverage Guarantees".
    Angelopoulos et al. (2021). "Uncertainty Sets for Image Classifiers
    using Conformal Prediction" (ICLR 2021).
"""
from __future__ import annotations

import numpy as np


def aps_scores_batch(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute APS nonconformity scores for a labelled batch (calibration).

    For each sample i with true label y_i, the APS score is the cumulative
    softmax probability needed to cover y_i when classes are sorted by
    probability in descending order.

    Args:
        probs: (N, C) softmax probabilities (already normalised).
        labels: (N,) integer ground-truth class indices.

    Returns:
        scores: (N,) APS score for each sample, in [0, 1].
    """
    N, C = probs.shape
    sorted_idx = np.argsort(-probs, axis=1)  # descending order
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cumsum = sorted_probs.cumsum(axis=1)

    # Find rank of true label in sorted order
    labels_exp = labels[:, np.newaxis]  # (N, 1)
    match = sorted_idx == labels_exp  # (N, C)
    rank = match.argmax(axis=1)  # (N,) — first True position

    scores = cumsum[np.arange(N), rank]
    return scores


def aps_scores_allclasses(probs: np.ndarray) -> np.ndarray:
    """Compute APS nonconformity scores for every hypothetical class (inference).

    Args:
        probs: (N, C) softmax probabilities.

    Returns:
        scores: (N, C) APS score for each sample x hypothetical class.
    """
    N, C = probs.shape
    sorted_idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cumsum = sorted_probs.cumsum(axis=1)

    # Invert permutation: for each original class c, find its rank
    scores = np.empty_like(probs)
    for r in range(C):
        c = sorted_idx[:, r]  # (N,) — class at rank r
        scores[np.arange(N), c] = cumsum[:, r]
    return scores


def aps_prediction_sets(
    test_probs: np.ndarray,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Construct APS prediction sets at level alpha.

    Args:
        test_probs: (N_test, C) softmax probabilities for test data.
        cal_probs: (N_cal, C) softmax probabilities for calibration data.
        cal_labels: (N_cal,) integer ground-truth labels for calibration.
        alpha: miscoverage level (e.g. 0.10 for 90% coverage).

    Returns:
        in_set: (N_test, C) boolean mask — True if class is in prediction set.
        q: the threshold score.
    """
    n_cal = len(cal_labels)
    nonconf = 1.0 - cal_probs[np.arange(n_cal), cal_labels]
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q = float(np.quantile(nonconf, q_level, method="higher"))
    in_set = test_probs >= (1.0 - q)
    return in_set, q
