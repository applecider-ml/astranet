# astra/conformal.py
# Conformal Prediction with Adaptive Prediction Sets (APS) for ASTRANet.
#
# Pipeline (per sample at inference):
#   1. Get softmax probs p ∈ R^C from the model.
#   2. For each class y, compute APS score:
#        sort classes by prob descending → cumulative sum up to rank(y) → APS(y).
#      This measures how much probability mass is "needed" to include y.
#   3. Compare each APS(y) against the calibration score distribution:
#        p-value(y) = #{cal scores ≥ APS(y)} / (n_cal + 1)
#      (marginal coverage guarantee: true class included with prob ≥ 1-α).
#   4. Normalize p-values across classes → conformal probabilities.
#
# References:
#   Angelopoulos et al., "Uncertainty Sets for Image Classifiers using
#   Conformal Prediction" (ICLR 2021).

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ============================================================
# APS SCORE FUNCTIONS
# ============================================================

def aps_scores_batch(probs: Tensor, labels: Tensor) -> Tensor:
    """
    Compute APS nonconformity scores for a labelled batch (calibration).

    For each sample i with true label y_i, the APS score is the cumulative
    softmax probability needed to cover y_i when classes are sorted by
    probability in descending order:

        s_i = sum_{j: rank(j) <= rank(y_i)} p_i[j]

    where rank is 1-indexed in descending probability order.

    Args:
        probs:  (B, C)  softmax probabilities (already normalised).
        labels: (B,)    integer ground-truth class indices.

    Returns:
        scores: (B,)    APS score for each sample, in [0, 1].
    """
    B, C = probs.shape
    # Sort classes by prob descending for each sample
    sorted_probs, sorted_idx = probs.sort(dim=1, descending=True)  # (B, C)
    cumsum = sorted_probs.cumsum(dim=1)  # (B, C)

    # Find rank of the true label in the sorted order
    # sorted_idx[b, r] = original class index at rank r
    # We need: for each b, find r such that sorted_idx[b, r] == labels[b]
    labels_exp = labels.unsqueeze(1).expand(B, C)   # (B, C)
    match = (sorted_idx == labels_exp)               # (B, C) bool
    # rank = argmax of match (first True position)
    rank = match.long().argmax(dim=1)               # (B,)

    # APS score = cumulative sum up to and including the true label's rank
    scores = cumsum[torch.arange(B), rank]           # (B,)
    return scores


def aps_scores_allclasses(probs: Tensor) -> Tensor:
    """
    Compute APS nonconformity scores for every hypothetical class (inference).

    For each sample i and each hypothetical true class y:
        APS(i, y) = sum_{j: rank(j) <= rank(y)} p_i[j]

    Args:
        probs: (B, C)  softmax probabilities.

    Returns:
        scores: (B, C)  APS score for each sample × hypothetical class.
    """
    B, C = probs.shape
    sorted_probs, sorted_idx = probs.sort(dim=1, descending=True)  # (B, C)
    cumsum = sorted_probs.cumsum(dim=1)  # (B, C)

    # For each original class c, find its rank in sorted order
    # Invert the permutation: rank_of[b, sorted_idx[b, r]] = r
    scores = torch.zeros_like(probs)  # (B, C)
    for r in range(C):
        c = sorted_idx[:, r]                   # (B,) — class at rank r
        val = cumsum[:, r]                     # (B,) — cumsum up to rank r
        scores[torch.arange(B), c] = val
    return scores


# ============================================================
# CONFORMAL PREDICTOR
# ============================================================

class ConformalPredictor:
    """
    Wraps one or more ASTRANet models and adds APS-based conformal UQ.

    Usage
    -----
    >>> cp = ConformalPredictor(models, device=device)
    >>> cp.calibrate(cal_loader)          # fit on held-out calibration set
    >>> out = cp.predict(batch)           # dict with conformal probs etc.

    The calibration set should be disjoint from training *and* test sets
    (a typical split is ~10-15% of training data held out as "cal").

    Attributes
    ----------
    cal_scores : (n_cal,) Tensor
        APS scores collected during calibration (stored on CPU).
    """

    def __init__(self, models, device: str | torch.device = "cpu"):
        """
        Args:
            models: a single nn.Module or list of nn.Modules (ensemble).
            device: torch device to run inference on.
        """
        if not isinstance(models, (list, tuple)):
            models = [models]
        self.models = models
        self.device = torch.device(device)
        self.cal_scores: Optional[Tensor] = None
        self.temperature: float = 1.0   # set by fit_temperature()

    # ----------------------------------------------------------
    def fit_temperature(self, cal_loader, verbose: bool = True) -> float:
        """
        Find the optimal temperature T that minimises NLL on the calibration set.

        Temperature scaling divides every ensemble-averaged logit by T before
        softmax.  T > 1 softens the distribution (reduces overconfidence);
        T < 1 sharpens it.  It is the simplest post-hoc calibration method
        and the only one that can genuinely re-rank borderline predictions,
        thereby improving AUPRC in addition to calibration.

        Call this BEFORE .calibrate() so that the APS scores are computed
        with the temperature-scaled probabilities.

        Returns:
            Optimal temperature (also stored in self.temperature).
        """
        from scipy.optimize import minimize_scalar

        # Collect raw averaged logits + labels
        all_logits, all_labels = [], []
        for model in self.models:
            model.eval()
        with torch.no_grad():
            for batch in cal_loader:
                x, phase, z, labels = _unpack_batch(batch)
                x      = x.to(self.device)
                phase  = phase.to(self.device) if phase is not None else None
                z      = z.to(self.device)     if z     is not None else None

                logit_list = []
                for model in self.models:
                    out = model(x, phase=phase, z=z)
                    logit_list.append(out["fine_logits"])
                avg_logits = torch.stack(logit_list, dim=0).mean(0)  # (B, C)
                all_logits.append(avg_logits.cpu())
                if labels is not None:
                    all_labels.append(labels.cpu())

        logits = torch.cat(all_logits, dim=0)   # (N, C)
        labels = torch.cat(all_labels, dim=0)   # (N,)

        def nll(T):
            scaled = F.softmax(logits / T, dim=-1)
            log_p  = scaled[torch.arange(len(labels)), labels].log().clamp(min=-20)
            return -log_p.mean().item()

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        T = float(result.x)
        self.temperature = T

        if verbose:
            nll_before = nll(1.0)
            nll_after  = nll(T)
            print(f"[ConformalPredictor] Temperature scaling:")
            print(f"  T = {T:.4f}")
            print(f"  NLL before: {nll_before:.4f}  →  after: {nll_after:.4f}")
        return T

    # ----------------------------------------------------------
    def calibrate(self, cal_loader, verbose: bool = True) -> None:
        """
        Compute and store APS scores on the calibration loader.

        Args:
            cal_loader: DataLoader yielding (x, meta, labels) tuples,
                        where meta = (phase, z) tensors.
            verbose:    print summary statistics when done.
        """
        all_scores = []
        for model in self.models:
            model.eval()
        with torch.no_grad():
            for batch in cal_loader:
                x, phase, z, labels = _unpack_batch(batch)
                x      = x.to(self.device)
                phase  = phase.to(self.device) if phase is not None else None
                z      = z.to(self.device)     if z     is not None else None
                labels = labels.to(self.device)

                probs = self._ensemble_probs(x, phase, z)       # (B, C)
                scores = aps_scores_batch(probs, labels)         # (B,)
                all_scores.append(scores.cpu())

        self.cal_scores = torch.cat(all_scores, dim=0)
        if verbose:
            print(f"[ConformalPredictor] Calibrated on {len(self.cal_scores)} samples.")
            print(f"  APS score  mean={self.cal_scores.mean():.4f} "
                  f" median={self.cal_scores.median():.4f} "
                  f" 90th-pct={self.cal_scores.quantile(0.90):.4f}")

    # ----------------------------------------------------------
    def predict(
        self,
        batch,
        alpha: float = 0.1,
    ) -> dict:
        """
        Run inference and return softmax + conformal uncertainty estimates.

        Args:
            batch: same format as cal_loader batches.
            alpha: miscoverage level; conformal sets cover true class with
                   probability ≥ 1-α (default 0.10 → 90% marginal coverage).

        Returns a dict with:
            "softmax_probs"    (B, C) — raw model softmax probabilities.
            "aps_scores"       (B, C) — APS score per class (lower = more likely).
            "pvalues"          (B, C) — marginal conformal p-values ∈ (0, 1].
            "conformal_probs"  (B, C) — p-values normalised to sum to 1.
            "conformal_sets"   (B, C) — binary mask: 1 if class in prediction set at α.
            "set_sizes"        (B,)   — number of classes in the prediction set.
            "pred_class"       (B,)   — argmax of conformal_probs.
        """
        if self.cal_scores is None:
            raise RuntimeError("Call .calibrate() before .predict().")

        for model in self.models:
            model.eval()
        with torch.no_grad():
            x, phase, z, _ = _unpack_batch(batch)
            x     = x.to(self.device)
            phase = phase.to(self.device) if phase is not None else None
            z     = z.to(self.device)     if z     is not None else None

            probs  = self._ensemble_probs(x, phase, z)          # (B, C)
            aps_sc = aps_scores_allclasses(probs)                # (B, C)

        # p-value(y) = #{cal scores >= APS(y)} / (n_cal + 1)
        cal = self.cal_scores.to(aps_sc.device)                  # (n_cal,)
        n_cal = cal.shape[0]
        # Broadcast: aps_sc (B, C, 1) vs cal (1, 1, n_cal)
        pvalues = (cal.unsqueeze(0).unsqueeze(0) >= aps_sc.unsqueeze(2)).float().sum(dim=2)
        pvalues = pvalues / (n_cal + 1)                          # (B, C)

        # Conformal probabilities: normalise p-values
        pval_sum = pvalues.sum(dim=1, keepdim=True).clamp(min=1e-8)
        conf_probs = pvalues / pval_sum                          # (B, C)

        # Prediction sets: include class if p-value > alpha
        conf_sets = (pvalues > alpha).float()                    # (B, C)
        # Guarantee non-empty: always include argmax of softmax
        argmax = probs.argmax(dim=1)                             # (B,)
        conf_sets[torch.arange(probs.shape[0]), argmax] = 1.0

        set_sizes  = conf_sets.sum(dim=1)                        # (B,)
        pred_class = conf_probs.argmax(dim=1)                    # (B,)

        return {
            "softmax_probs":   probs.cpu(),
            "aps_scores":      aps_sc.cpu(),
            "pvalues":         pvalues.cpu(),
            "conformal_probs": conf_probs.cpu(),
            "conformal_sets":  conf_sets.cpu(),
            "set_sizes":       set_sizes.cpu(),
            "pred_class":      pred_class.cpu(),
        }

    # ----------------------------------------------------------
    def coverage_and_efficiency(self, test_loader) -> dict:
        """
        Evaluate marginal coverage and average set size on a labelled test set.

        Returns dict with:
            "coverage"       — fraction of samples where true class is in pred set.
            "avg_set_size"   — mean number of classes per prediction set.
            "singleton_rate" — fraction of samples with a single-class set.
        """
        if self.cal_scores is None:
            raise RuntimeError("Call .calibrate() first.")

        covered, set_sizes_all = [], []
        for model in self.models:
            model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x, phase, z, labels = _unpack_batch(batch)
                x      = x.to(self.device)
                phase  = phase.to(self.device) if phase is not None else None
                z      = z.to(self.device)     if z     is not None else None
                labels = labels.to(self.device)

                probs  = self._ensemble_probs(x, phase, z)
                aps_sc = aps_scores_allclasses(probs)
                cal    = self.cal_scores.to(aps_sc.device)
                n_cal  = cal.shape[0]

                pvalues  = (cal.unsqueeze(0).unsqueeze(0) >= aps_sc.unsqueeze(2)).float().sum(2)
                pvalues /= (n_cal + 1)

                conf_sets = (pvalues > 0.1).float()
                argmax    = probs.argmax(dim=1)
                conf_sets[torch.arange(probs.shape[0]), argmax] = 1.0

                # Coverage
                in_set = conf_sets[torch.arange(len(labels)), labels.cpu()]
                covered.append(in_set.cpu())
                set_sizes_all.append(conf_sets.sum(dim=1).cpu())

        covered_t   = torch.cat(covered)
        set_sizes_t = torch.cat(set_sizes_all)
        return {
            "coverage":       covered_t.mean().item(),
            "avg_set_size":   set_sizes_t.mean().item(),
            "singleton_rate": (set_sizes_t == 1).float().mean().item(),
        }

    # ----------------------------------------------------------
    def infer_spectra(
        self,
        flux: "np.ndarray",
        phase: float = float("nan"),
        z: float = float("nan"),
        alpha: float = 0.10,
        class_names: list = None,
        savgol_window: int = 7,
        savgol_poly: int = 3,
    ) -> list:
        """
        Run uncertainty-aware inference on one or more raw flux arrays.

        This is the end-to-end entry point for new, unlabelled spectra.
        It handles preprocessing (Savitzky-Golay derivative), batching,
        and returns human-readable results.

        Args:
            flux:  numpy array of shape (L,) for a single spectrum, or
                   (B, L) for a batch.  Should be normalised flux values.
            phase: observation phase in days (NaN if unknown).
            z:     redshift (NaN if unknown).
            alpha: miscoverage level (default 0.10 → 90% coverage sets).
            class_names: list of class name strings for display.
                         If None, returns integer indices.
            savgol_window: SG derivative window (must match training config).
            savgol_poly:   SG derivative polynomial order.

        Returns:
            List of dicts, one per spectrum:
                "pred_class"       — predicted class name (or index)
                "softmax_probs"    — dict of {class: prob}
                "conformal_probs"  — dict of {class: conf_prob}
                "prediction_set"   — list of class names in the set at α
                "set_size"         — number of classes in the set
                "uncertainty"      — "low" / "medium" / "high" based on set_size
                "top_conf_prob"    — conformal probability of the top class
        """
        if self.cal_scores is None:
            raise RuntimeError("Call .calibrate() before .infer_spectra().")

        import numpy as np

        flux = np.asarray(flux, dtype=np.float32)
        if flux.ndim == 1:
            flux = flux[np.newaxis]          # (1, L)
        B, L = flux.shape

        # ── Savitzky-Golay derivative ──────────────────────────
        from scipy.signal import savgol_filter
        deriv = savgol_filter(
            flux, window_length=savgol_window,
            polyorder=savgol_poly, deriv=1, axis=-1,
        ).astype(np.float32)

        # Stack flux + derivative → (B, 2, L)
        x = torch.tensor(
            np.stack([flux, deriv], axis=1), dtype=torch.float32
        ).to(self.device)

        # ── Metadata ──────────────────────────────────────────
        ph_t = torch.full((B,), phase, dtype=torch.float32).to(self.device)
        z_t  = torch.full((B,), z,     dtype=torch.float32).to(self.device)

        # ── Inference ─────────────────────────────────────────
        for m in self.models:
            m.eval()
        with torch.no_grad():
            probs  = self._ensemble_probs(x, ph_t, z_t)         # (B, C)
            aps_sc = aps_scores_allclasses(probs)                # (B, C)

        cal   = self.cal_scores.to(aps_sc.device)
        n_cal = cal.shape[0]
        pvals = (
            cal.unsqueeze(0).unsqueeze(0) >= aps_sc.unsqueeze(2)
        ).float().sum(2) / (n_cal + 1)                          # (B, C)

        pval_sum   = pvals.sum(1, keepdim=True).clamp(min=1e-8)
        conf_probs = pvals / pval_sum                            # (B, C)

        conf_sets = (pvals > alpha).float()
        argmax    = probs.argmax(1)
        conf_sets[torch.arange(B), argmax] = 1.0

        # ── Format results ────────────────────────────────────
        C = probs.shape[1]
        labels = class_names if class_names else list(range(C))

        results = []
        for i in range(B):
            pred_idx = conf_probs[i].argmax().item()
            set_idx  = conf_sets[i].nonzero(as_tuple=True)[0].tolist()
            sz       = len(set_idx)
            uncertainty = "low" if sz == 1 else ("medium" if sz == 2 else "high")

            results.append({
                "pred_class":      labels[pred_idx],
                "softmax_probs":   {labels[c]: round(probs[i, c].item(), 4)
                                    for c in range(C)},
                "conformal_probs": {labels[c]: round(conf_probs[i, c].item(), 4)
                                    for c in range(C)},
                "prediction_set":  [labels[c] for c in set_idx],
                "set_size":        sz,
                "uncertainty":     uncertainty,
                "top_conf_prob":   round(conf_probs[i, pred_idx].item(), 4),
            })
        return results

    # ----------------------------------------------------------
    def _ensemble_probs(
        self,
        x: Tensor,
        phase: Optional[Tensor],
        z: Optional[Tensor],
    ) -> Tensor:
        """Average logits across ensemble, apply temperature, then softmax."""
        logit_list = []
        for model in self.models:
            out = model(x, phase=phase, z=z)
            logit_list.append(out["fine_logits"])
        avg_logits = torch.stack(logit_list, dim=0).mean(dim=0)  # (B, C)
        return F.softmax(avg_logits / self.temperature, dim=-1)   # (B, C)


# ============================================================
# BATCH UNPACKING HELPER
# ============================================================

def _unpack_batch(batch):
    """
    Unpack a DataLoader batch into (x, phase, z, labels).

    Supports:
      - dict batch (ASTRANetDataset / collate_fn):
            keys: "x", "phase", "z", "y"  (y optional)
      - 4-tuple  (x, phase, z, labels)
      - 3-tuple  (x, meta_dict_or_tensor, labels)

    Returns None for missing phase/z/labels.
    """
    if isinstance(batch, dict):
        x      = batch["x"]
        phase  = batch.get("phase", None)
        z      = batch.get("z",     None)
        labels = batch.get("y",     None)
        return x, phase, z, labels
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 4:
            x, phase, z, labels = batch
            return x, phase, z, labels
        elif len(batch) == 3:
            x, meta, labels = batch
            if isinstance(meta, dict):
                phase = meta.get("phase", None)
                z     = meta.get("z",     None)
            else:
                phase, z = None, meta
            return x, phase, z, labels
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}")
    else:
        raise ValueError(f"Unexpected batch type {type(batch)}")
