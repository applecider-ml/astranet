"""ConformalPredictor — numpy-native wrapper for APS-based conformal UQ."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax

from astranet.uq.aps import aps_scores_batch, aps_scores_allclasses


class ConformalPredictor:
    """Conformal prediction with Adaptive Prediction Sets (APS).

    Numpy-native (no PyTorch dependency). Works with softmax probabilities
    or raw logits from any classifier.

    Usage:
        cp = ConformalPredictor()
        cp.calibrate(cal_probs, cal_labels)     # fit on calibration set
        result = cp.predict(test_probs)          # predict with UQ
        metrics = cp.coverage_report(test_probs, test_labels)
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: temperature scaling factor for logits.
                Applied only when using logits (not pre-computed probs).
        """
        self.temperature = temperature
        self.cal_scores: np.ndarray | None = None

    def fit_temperature(
        self, cal_logits: np.ndarray, cal_labels: np.ndarray, verbose: bool = True,
    ) -> float:
        """Find optimal temperature T minimising NLL on calibration set.

        Args:
            cal_logits: (N, C) raw logits.
            cal_labels: (N,) integer labels.
            verbose: print temperature and NLL values.

        Returns:
            Optimal temperature (also stored in self.temperature).
        """
        from scipy.optimize import minimize_scalar

        def nll(T):
            probs = softmax(cal_logits / T, axis=1)
            log_p = np.log(probs[np.arange(len(cal_labels)), cal_labels] + 1e-20)
            return -log_p.mean()

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        T = float(result.x)
        self.temperature = T

        if verbose:
            print(f"[ConformalPredictor] Temperature scaling: T = {T:.4f}")
            print(f"  NLL: {nll(1.0):.4f} -> {nll(T):.4f}")
        return T

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        verbose: bool = True,
    ) -> None:
        """Compute and store APS calibration scores.

        Args:
            cal_probs: (N, C) softmax probabilities for calibration set.
            cal_labels: (N,) integer ground-truth labels.
            verbose: print summary statistics.
        """
        self.cal_scores = aps_scores_batch(cal_probs, cal_labels)
        if verbose:
            print(
                f"[ConformalPredictor] Calibrated on {len(self.cal_scores)} samples. "
                f"APS mean={self.cal_scores.mean():.4f}, "
                f"median={np.median(self.cal_scores):.4f}, "
                f"90th-pct={np.quantile(self.cal_scores, 0.90):.4f}"
            )

    def predict(
        self, test_probs: np.ndarray, alpha: float = 0.10,
    ) -> dict[str, np.ndarray]:
        """Run conformal prediction on test probabilities.

        Args:
            test_probs: (N, C) softmax probabilities.
            alpha: miscoverage level (default 0.10 -> 90% coverage).

        Returns:
            Dict with keys:
                "softmax_probs"   (N, C) — input probabilities
                "aps_scores"      (N, C) — APS score per hypothetical class
                "pvalues"         (N, C) — conformal p-values
                "conformal_probs" (N, C) — normalised p-values
                "conformal_sets"  (N, C) — binary mask for prediction set
                "set_sizes"       (N,)   — number of classes in set
                "pred_class"      (N,)   — argmax of conformal_probs
        """
        if self.cal_scores is None:
            raise RuntimeError("Call .calibrate() before .predict().")

        N, C = test_probs.shape
        aps_sc = aps_scores_allclasses(test_probs)  # (N, C)

        # p-value(y) = #{cal scores >= APS(y)} / (n_cal + 1)
        n_cal = len(self.cal_scores)
        cal = self.cal_scores  # (n_cal,)

        # Vectorized: for each (i, c), count cal scores >= aps_sc[i, c]
        # Using broadcasting: (N, C, 1) vs (1, 1, n_cal)
        pvalues = (
            cal[np.newaxis, np.newaxis, :] >= aps_sc[:, :, np.newaxis]
        ).sum(axis=2).astype(float) / (n_cal + 1)

        # Conformal probabilities: normalise p-values
        pval_sum = pvalues.sum(axis=1, keepdims=True)
        pval_sum = np.maximum(pval_sum, 1e-8)
        conf_probs = pvalues / pval_sum

        # Prediction sets: include class if p-value > alpha
        conf_sets = (pvalues > alpha).astype(float)
        # Non-empty guarantee: always include softmax argmax
        argmax = test_probs.argmax(axis=1)
        conf_sets[np.arange(N), argmax] = 1.0

        set_sizes = conf_sets.sum(axis=1).astype(int)
        pred_class = conf_probs.argmax(axis=1)

        return {
            "softmax_probs": test_probs,
            "aps_scores": aps_sc,
            "pvalues": pvalues,
            "conformal_probs": conf_probs,
            "conformal_sets": conf_sets,
            "set_sizes": set_sizes,
            "pred_class": pred_class,
        }

    def coverage_report(
        self,
        test_probs: np.ndarray,
        test_labels: np.ndarray,
        alpha: float = 0.10,
    ) -> dict[str, float]:
        """Evaluate marginal coverage and efficiency on a labelled test set.

        Returns:
            Dict with "coverage", "avg_set_size", "singleton_rate".
        """
        result = self.predict(test_probs, alpha=alpha)
        conf_sets = result["conformal_sets"]
        covered = conf_sets[np.arange(len(test_labels)), test_labels] == 1.0
        set_sizes = result["set_sizes"]
        return {
            "coverage": float(covered.mean()),
            "avg_set_size": float(set_sizes.mean()),
            "singleton_rate": float((set_sizes == 1).mean()),
        }
