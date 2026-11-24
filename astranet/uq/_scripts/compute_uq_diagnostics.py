"""
compute_uq_diagnostics.py
=========================

Companion script for ASTRAnet anomaly detection framework.  Computes
the UQ diagnostics needed for Section 5 of the reshaped paper
("From Anomaly Scores to Calibrated Prediction Sets"):

  1. Mondrian (class-conditional) conformal calibration of the
     LightGBM anomaly score against the in-distribution validation
     calibration set.
  2. Adaptive Prediction Sets (APS) for the fine-class classifier
     at multiple coverage levels.
  3. (Optional) Aleatoric/epistemic decomposition via deep-ensemble
     mutual information.  Skipped if per-model logits are not
     available in the cache.

Inputs (cached .npz files produced by anomaly_detection.ipynb):
  - {CACHE_DIR}/train.npz : training-set features (for fitting
    per-class statistics)
  - {CACHE_DIR}/val.npz   : validation embeddings + fine_logits +
    labels (calibration set)
  - {CACHE_DIR}/rare.npz  : rare-set embeddings + fine_logits + meta
    (test set)

Outputs:
  - {OUT_DIR}/uq_coverage.pdf      : empirical vs nominal coverage
  - {OUT_DIR}/uq_pvalues_ecdf.pdf  : ECDF of p-values ID vs OOD
  - {OUT_DIR}/uq_predset_size.pdf  : prediction set size histograms
  - {OUT_DIR}/uq_summary.md        : markdown numerical summary
  - {OUT_DIR}/uq_per_spectrum.csv  : per-spectrum UQ outputs

Runtime: ~3-5 minutes on a laptop.  No GPU required.

Usage:
  python compute_uq_diagnostics.py \
      --cache-dir <path to>/cache \
      --out-dir   ./uq_out
"""

# ---------------------------------------------------------------------
# Thread limits — MUST be set before numpy/scipy/sklearn imports.
# On shared clusters with RLIMIT_NPROC saturation, OpenBLAS will try to
# spawn one thread per core (often 40+) and fail.  Capping these
# env vars to a small number lets the script run anywhere.
# Override at the shell level if you want more threads.
# ---------------------------------------------------------------------
import os
for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS"):
    os.environ.setdefault(var, "4")

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.linewidth": 0.8, "lines.linewidth": 1.4, "axes.unicode_minus": False})
from scipy.special import softmax as sp_softmax
from scipy.stats import entropy as sp_entropy
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed.  Falling back to LR ensemble.")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


# ============================================================
# Configuration
# ============================================================
CLASS_NAMES = ["Ia", "Ibc", "SLSN", "II", "AGN", "CV", "TDE"]
N_CLASSES = len(CLASS_NAMES)

# Conformal alpha grid for coverage diagnostic
ALPHA_GRID = np.linspace(0.005, 0.20, 30)
APS_ALPHAS = [0.01, 0.05, 0.10, 0.20]  # coverage levels to report

# Score families (matching the notebook's PAPER_METHODS ordering)
SCORE_FAMILIES = {
    "uncertainty": ["msp", "entropy", "energy", "odin"],
    "distance":    ["mahal_global", "mahal_class", "mahal_within",
                    "knn", "cosine", "typicality"],
    "density":     ["gmm", "iforest", "lof", "pca_recon"],
    "hybrid":      ["hec", "mrs"],
}
ALL_SCORES = sum(SCORE_FAMILIES.values(), [])
assert len(ALL_SCORES) == 16, ALL_SCORES


# ============================================================
# Data loading
# ============================================================
def load_cache(cache_dir: Path, name: str) -> dict:
    """Load a cached split (matches save_cache from the notebook)."""
    npz = np.load(cache_dir / f"{name}.npz")
    data = {k: npz[k] for k in npz.files}
    meta_path = cache_dir / f"{name}_meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as fp:
            data["meta"] = pickle.load(fp)
    return data


def maybe_filter_tde(rare: dict) -> dict:
    """The notebook drops TDE from the rare set (it is a training class)."""
    if "meta" not in rare:
        return rare
    classes = np.array([m.get("class", "") for m in rare["meta"]])
    keep = classes != "Tidal Disruption Event"
    if keep.all():
        return rare
    rare["meta"] = [m for m, k in zip(rare["meta"], keep) if k]
    for key, val in list(rare.items()):
        if isinstance(val, np.ndarray):
            rare[key] = val[keep]
    return rare


# ============================================================
# Anomaly score computation (mirrors notebook cell that defines
# the 16 scores; included here so the script is standalone)
# ============================================================
class ScoringPipeline:
    """Fits the unsupervised score models on the training embeddings.

    All scores use the convention: higher = more anomalous.
    """

    def __init__(self, tr_emb: np.ndarray, tr_proj: np.ndarray,
                 tr_lab: np.ndarray):
        self.n_classes = N_CLASSES
        self.tr_emb = tr_emb
        self.tr_proj = tr_proj
        self.tr_lab = tr_lab

        # Per-class statistics
        self.class_means = {}
        self.class_cov_inv = {}
        self.class_proj_means = {}
        self.class_radii = {}
        for c in range(N_CLASSES):
            mask = tr_lab == c
            if mask.sum() < 5:
                continue
            self.class_means[c] = tr_emb[mask].mean(axis=0)
            cov = LedoitWolf().fit(tr_emb[mask]).covariance_
            self.class_cov_inv[c] = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
            self.class_proj_means[c] = tr_proj[mask].mean(axis=0)
            dists = np.linalg.norm(tr_emb[mask] - self.class_means[c], axis=1)
            self.class_radii[c] = np.percentile(dists, 95)

        # Global Mahalanobis
        self.global_mean = tr_emb.mean(axis=0)
        global_cov = LedoitWolf().fit(tr_emb).covariance_
        self.global_cov_inv = np.linalg.inv(
            global_cov + 1e-6 * np.eye(global_cov.shape[0])
        )

        # PCA for reconstruction score
        self.pca = PCA(n_components=32).fit(tr_emb)

        # Per-class PCA for MRS
        self.class_pca = {}
        for c, mean in self.class_means.items():
            mask = tr_lab == c
            if mask.sum() >= 32:
                self.class_pca[c] = PCA(n_components=12).fit(tr_emb[mask])

        # GMM (per-class, fitted with full covariance)
        self.gmm = {}
        for c, mean in self.class_means.items():
            mask = tr_lab == c
            if mask.sum() >= 32:
                self.gmm[c] = GaussianMixture(
                    n_components=1, covariance_type="full", reg_covar=1e-4
                ).fit(tr_emb[mask])

        # kNN, LOF, isolation forest
        self.knn = NearestNeighbors(n_neighbors=10).fit(tr_emb)
        self.iforest = IsolationForest(
            n_estimators=200, contamination="auto", random_state=42
        ).fit(tr_emb)
        self.lof = LocalOutlierFactor(
            n_neighbors=20, novelty=True
        ).fit(tr_emb)

        # Class projection norms (for cosine)
        proj_norm = tr_proj / (np.linalg.norm(tr_proj, axis=1, keepdims=True) + 1e-8)
        self.class_proj_norm = {
            c: (m / (np.linalg.norm(m) + 1e-8))
            for c, m in self.class_proj_means.items()
        }

    def compute_scores(self, emb: np.ndarray, proj: np.ndarray,
                       fine_logits: np.ndarray,
                       coarse_logits: np.ndarray) -> dict:
        """Return all 16 scores as a dict of arrays."""
        n = len(emb)
        scores = {}
        fine_sm = sp_softmax(fine_logits, axis=1)
        coarse_sm = sp_softmax(coarse_logits, axis=1)
        pred = fine_sm.argmax(axis=1)

        # 1-4: uncertainty
        scores["msp"] = -fine_sm.max(axis=1)
        scores["entropy"] = sp_entropy(fine_sm, axis=1)
        scores["energy"] = -np.log(np.exp(fine_logits).sum(axis=1) + 1e-12)
        T = 1.5
        scores["odin"] = -sp_softmax(fine_logits / T, axis=1).max(axis=1)

        # 5: Mahalanobis global
        d = emb - self.global_mean
        scores["mahal_global"] = np.sqrt(
            np.einsum("ij,jk,ik->i", d, self.global_cov_inv, d)
        )

        # 6: Mahalanobis min-class
        m_class = np.full(n, np.inf)
        for c, mean in self.class_means.items():
            dc = emb - mean
            mh = np.sqrt(np.einsum("ij,jk,ik->i", dc, self.class_cov_inv[c], dc))
            m_class = np.minimum(m_class, mh)
        scores["mahal_class"] = m_class

        # 7: Mahalanobis within (predicted class)
        m_within = np.zeros(n)
        for c in self.class_means:
            mask = pred == c
            if not mask.any():
                continue
            dc = emb[mask] - self.class_means[c]
            m_within[mask] = np.sqrt(
                np.einsum("ij,jk,ik->i", dc, self.class_cov_inv[c], dc)
            )
        scores["mahal_within"] = m_within

        # 8: kNN
        dists, _ = self.knn.kneighbors(emb)
        scores["knn"] = dists.mean(axis=1)

        # 9: cosine distance to nearest class projection centroid
        proj_norm = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)
        cos_sims = np.stack(
            [proj_norm @ self.class_proj_norm[c] for c in self.class_proj_norm],
            axis=1,
        )
        scores["cosine"] = 1.0 - cos_sims.max(axis=1)

        # 10: typicality
        typ = np.zeros(n)
        for c in self.class_means:
            mask = pred == c
            if not mask.any():
                continue
            d = np.linalg.norm(emb[mask] - self.class_means[c], axis=1)
            typ[mask] = d / max(self.class_radii[c], 1e-8)
        scores["typicality"] = typ

        # 11: GMM negative log-density (best class)
        gmm_nll = np.full(n, np.inf)
        for c, gmm in self.gmm.items():
            gmm_nll = np.minimum(gmm_nll, -gmm.score_samples(emb))
        scores["gmm"] = gmm_nll

        # 12: PCA reconstruction error
        recon = self.pca.inverse_transform(self.pca.transform(emb))
        scores["pca_recon"] = np.linalg.norm(emb - recon, axis=1)

        # 13: Isolation forest (sklearn returns +score for inliers; flip)
        scores["iforest"] = -self.iforest.score_samples(emb)

        # 14: LOF
        scores["lof"] = -self.lof.score_samples(emb)

        # 15: HEC
        Hf = sp_entropy(fine_sm, axis=1)
        Hc = sp_entropy(coarse_sm, axis=1)
        # Hierarchy: fine_class -> coarse_parent
        FINE_TO_COARSE = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
        parent = np.array([FINE_TO_COARSE[p] for p in pred])
        hier_incon = 1.0 - coarse_sm[np.arange(n), parent]
        scores["hec"] = 1.0 * Hf + 0.5 * Hc + 1.0 * hier_incon

        # 16: MRS (manifold residual score)
        mrs = np.full(n, np.inf)
        for c, pca in self.class_pca.items():
            r = pca.inverse_transform(pca.transform(emb))
            mrs = np.minimum(mrs, np.linalg.norm(emb - r, axis=1))
        scores["mrs"] = mrs

        return scores


# ============================================================
# Conformal prediction primitives
# ============================================================
def conformal_pvalues_marginal(test_scores: np.ndarray,
                               cal_scores: np.ndarray) -> np.ndarray:
    """Marginal conformal p-values.

    p(x*) = (|{i : s_cal_i >= s*}| + 1) / (n_cal + 1)
    """
    n_cal = len(cal_scores)
    cal_sorted = np.sort(cal_scores)
    # Count of cal_scores >= test_score = n_cal - searchsorted(test_score)
    ranks = n_cal - np.searchsorted(cal_sorted, test_scores, side="left")
    return (ranks + 1) / (n_cal + 1)


def conformal_pvalues_mondrian(test_scores: np.ndarray, test_preds: np.ndarray,
                               cal_scores: np.ndarray, cal_preds: np.ndarray
                               ) -> np.ndarray:
    """Mondrian (class-conditional) conformal p-values.

    For each test point with predicted class y_hat, computes the p-value
    against the calibration subset with the same predicted class.
    """
    pvals = np.zeros(len(test_scores))
    for y in np.unique(test_preds):
        cal_y = cal_scores[cal_preds == y]
        mask = test_preds == y
        if len(cal_y) == 0:
            pvals[mask] = 0.5  # no calibration data for this class
            continue
        pvals[mask] = conformal_pvalues_marginal(test_scores[mask], cal_y)
    return pvals


def aps_prediction_sets(test_softmax: np.ndarray, cal_softmax: np.ndarray,
                        cal_labels: np.ndarray, alpha: float
                        ) -> tuple[np.ndarray, float]:
    """Adaptive Prediction Sets (Romano et al. 2020).

    Calibration nonconformity: s_i = 1 - p_hat(y_i | x_i)
    Threshold: q = empirical (1-alpha)-quantile of s_i (finite-sample
               corrected)
    Prediction set: C(x) = {y : p_hat(y | x) >= 1 - q}
    """
    n = len(cal_labels)
    nonconf = 1.0 - cal_softmax[np.arange(n), cal_labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q = np.quantile(nonconf, q_level, method="higher")
    in_set = test_softmax >= (1.0 - q)
    return in_set, q


# ============================================================
# Deep-ensemble mutual information (optional)
# ============================================================
def mutual_information(per_model_softmax: np.ndarray) -> np.ndarray:
    """
    Entropy decomposition for a deep ensemble.

    per_model_softmax: shape (M, n, K) — M models, n samples, K classes.

    Returns mutual information MI(y; theta | x) = H[mean] - mean(H),
    a non-negative quantity that is high when models disagree.
    """
    mean_sm = per_model_softmax.mean(axis=0)
    H_mean = sp_entropy(mean_sm, axis=1)  # total
    H_per = sp_entropy(per_model_softmax, axis=2).mean(axis=0)  # aleatoric
    mi = H_mean - H_per
    return mi, H_mean, H_per


# ============================================================
# Plotting
# ============================================================
def plot_coverage(rare_pvals_mondrian: np.ndarray,
                  rare_pvals_marginal: np.ndarray,
                  val_pvals_mondrian: np.ndarray,
                  val_pvals_marginal: np.ndarray,
                  out_path: Path) -> None:
    """Empirical coverage on ID validation (held-out) vs nominal."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (pv_id, pv_ood, name) in zip(axes, [
        (val_pvals_marginal, rare_pvals_marginal, "Marginal"),
        (val_pvals_mondrian, rare_pvals_mondrian, "Mondrian (class-cond.)"),
    ]):
        # Empirical coverage of ID at each alpha = fraction of ID where p > alpha
        emp_cov = np.array([(pv_id > a).mean() for a in ALPHA_GRID])
        nom_cov = 1.0 - ALPHA_GRID
        ax.plot(nom_cov, emp_cov, "o-", lw=1.8, color="#2c7fb8",
                label="empirical (ID held-out)")
        ax.plot([0.7, 1.0], [0.7, 1.0], "k--", lw=1.0, alpha=0.5,
                label="perfect calibration")
        # Detection rate at each alpha = fraction of OOD where p < alpha
        det = np.array([(pv_ood < a).mean() for a in ALPHA_GRID])
        ax2 = ax.twinx()
        ax2.plot(nom_cov, det, "s-", lw=1.5, color="#d7301f", alpha=0.8,
                 label="OOD detection rate")
        ax2.set_ylabel("OOD detection rate", color="#d7301f")
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(axis="y", labelcolor="#d7301f")
        ax.set_xlabel("Nominal coverage  $1 - \\alpha$")
        ax.set_ylabel("Empirical ID coverage")
        ax.set_xlim(0.78, 1.005)
        ax.set_ylim(0.78, 1.005)
        ax.set_title(f"{name} conformal calibration")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pvalue_ecdfs(val_p_mondrian: np.ndarray, rare_p_mondrian: np.ndarray,
                      out_path: Path) -> None:
    """ECDF of Mondrian conformal p-values, ID vs OOD."""
    fig, ax = plt.subplots(figsize=(3.5, 2.55))   # paper single-column final size
    for vals, name, color in [
        (val_p_mondrian,  "in-distribution (val)", "#2c7fb8"),
        (rare_p_mondrian, "OOD (rare set)",        "#d7301f"),
    ]:
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, lw=2.0, label=name, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5,
            label="uniform (calibration target)")
    ax.set_xlabel("Mondrian conformal p-value")
    ax.set_ylabel("ECDF")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_predset_sizes(sizes_id: dict[float, np.ndarray],
                       sizes_ood: dict[float, np.ndarray],
                       out_path: Path) -> None:
    """Prediction set size distributions across alpha values."""
    alphas = sorted(sizes_id)
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.4 * len(alphas), 4),
                              sharey=True)
    for ax, a in zip(axes, alphas):
        bins = np.arange(0.5, N_CLASSES + 1.5, 1)
        ax.hist(sizes_id[a], bins=bins, alpha=0.65, density=True,
                label="ID (val)", color="#2c7fb8", edgecolor="white")
        ax.hist(sizes_ood[a], bins=bins, alpha=0.65, density=True,
                label="OOD (rare)", color="#d7301f", edgecolor="white")
        ax.set_xlabel("Prediction set size")
        ax.set_title(f"$\\alpha = {a:.2f}$, target cov.\\ ${1-a:.2f}$")
        ax.set_xticks(range(1, N_CLASSES + 1))
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("density")
            ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Reporting
# ============================================================
def write_markdown_summary(
    out_path: Path, summary: dict
) -> None:
    """Human-readable markdown summary."""
    lines = []
    lines.append("# UQ diagnostics summary\n")
    lines.append("Generated by `compute_uq_diagnostics.py`.\n")

    lines.append("## 1. Conformal anomaly detection\n")
    lines.append(f"- Calibration set size: {summary['n_cal']}\n")
    lines.append(f"- Test set size (rare): {summary['n_rare']}\n")
    lines.append(f"- LightGBM AUROC on rare: **{summary['lgb_auroc']:.4f}**\n")
    lines.append("\n### Coverage at nominal 1 - alpha (held-out ID)\n")
    lines.append("| alpha | nominal cov | marginal emp cov | Mondrian emp cov | OOD detection |\n")
    lines.append("|-------|-------------|------------------|------------------|----------------|\n")
    for a in [0.01, 0.025, 0.05, 0.10]:
        ec_marg = summary["coverage"][a]["marginal_emp"]
        ec_mond = summary["coverage"][a]["mondrian_emp"]
        det     = summary["coverage"][a]["mondrian_det"]
        lines.append(
            f"| {a:.3f} | {1-a:.3f}       | {ec_marg:.3f}            | {ec_mond:.3f}            | {det:.3f}        |\n"
        )

    lines.append("\n## 2. Adaptive Prediction Sets (APS)\n")
    lines.append("| alpha | cov target | mean size (ID) | mean size (OOD) | singleton % ID | singleton % OOD |\n")
    lines.append("|-------|-----------|---------------|----------------|----------------|------------------|\n")
    for a in APS_ALPHAS:
        s = summary["aps"][a]
        lines.append(
            f"| {a:.2f}  | {1-a:.2f}      | {s['mean_id']:.2f}          | {s['mean_ood']:.2f}           | {s['singleton_id']:.1f}%          | {s['singleton_ood']:.1f}%            |\n"
        )

    if summary.get("mi_available"):
        lines.append("\n## 3. Aleatoric / epistemic decomposition\n")
        m = summary["mi"]
        lines.append(f"- MI AUROC (epistemic-only AD on rare): **{m['mi_auroc']:.3f}**\n")
        lines.append(f"- Mean MI on ID (val): {m['mi_id_mean']:.3f}\n")
        lines.append(f"- Mean MI on OOD (rare): {m['mi_ood_mean']:.3f} ({m['mi_ratio']:.1f}× ID)\n")
    else:
        lines.append("\n## 3. Aleatoric / epistemic decomposition\n")
        lines.append("Skipped — per-model logits not found in cache "
                     "(see end of script for extraction instructions).\n")

    lines.append("\n## 4. Per-class Mondrian coverage at alpha = 0.025\n")
    lines.append("| Class | n_cal | n_test | empirical coverage | det rate OOD-routed-here |\n")
    lines.append("|-------|-------|--------|-------------------|--------------------------|\n")
    for c, row in summary["per_class"].items():
        lines.append(
            f"| {c} | {row['n_cal']} | {row['n_test']} | {row['cov']:.3f} | {row['det']:.3f} |\n"
        )

    out_path.write_text("".join(lines))


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="Directory containing train.npz, val.npz, rare.npz")
    ap.add_argument("--out-dir", type=Path, default=Path("./uq_out"),
                    help="Output directory for figures and reports")
    ap.add_argument("--per-model-cache", type=Path, default=None,
                    help="Optional .npz with per-model softmax outputs "
                         "(keys: 'val_softmax', 'rare_softmax', "
                         "each shape (M, n, K))")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------
    print("Loading cached splits...")
    train = load_cache(args.cache_dir, "train")
    val   = load_cache(args.cache_dir, "val")
    rare  = load_cache(args.cache_dir, "rare_unknown_combined")
    rare  = maybe_filter_tde(rare)

    print(f"  train: N={len(train['emb'])}")
    print(f"  val:   N={len(val['emb'])}")
    print(f"  rare:  N={len(rare['emb'])}")

    # ---- Fit scoring pipeline --------------------------------------
    print("Fitting scoring pipeline on training embeddings...")
    pipe = ScoringPipeline(
        tr_emb=train["emb"], tr_proj=train["proj_fine"],
        tr_lab=train["labels"],
    )

    # ---- Compute scores --------------------------------------------
    print("Computing scores on val and rare...")
    val_scores = pipe.compute_scores(
        emb=val["emb"], proj=val["proj_fine"],
        fine_logits=val["fine_logits"],
        coarse_logits=val["coarse_logits"],
    )
    rare_scores = pipe.compute_scores(
        emb=rare["emb"], proj=rare["proj_fine"],
        fine_logits=rare["fine_logits"],
        coarse_logits=rare["coarse_logits"],
    )

    # Stack into matrices in canonical order
    val_X = np.stack([val_scores[m] for m in ALL_SCORES], axis=1)
    rare_X = np.stack([rare_scores[m] for m in ALL_SCORES], axis=1)

    # Rank-normalise val_X on validation, apply same map to rare_X
    print("Rank-normalising scores against validation...")
    from scipy.stats import rankdata
    val_X_rn = np.empty_like(val_X)
    rare_X_rn = np.empty_like(rare_X)
    sort_indices = []
    for j in range(val_X.shape[1]):
        val_X_rn[:, j] = rankdata(val_X[:, j]) / len(val_X)
        sort_indices.append(np.sort(val_X[:, j]))
        # rare: interpolate rank in the validation distribution
        rare_X_rn[:, j] = np.searchsorted(sort_indices[-1], rare_X[:, j]) / len(val_X)

    # ---- Fit LightGBM ensemble (negatives = val, positives = rare) -
    # We use 5-fold CV on val to produce calibration scores, then full-fit
    # for evaluating rare.  This matches the notebook's setup.
    print("Training LightGBM ensemble...")
    X_all = np.vstack([val_X_rn, rare_X_rn])
    y_all = np.concatenate([
        np.zeros(len(val_X_rn)), np.ones(len(rare_X_rn))
    ])

    if HAS_LGB:
        clf_class = lambda: lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=-1,
            num_leaves=31, class_weight="balanced", verbosity=-1,
            random_state=args.seed,
        )
    else:
        clf_class = lambda: LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced",
            random_state=args.seed,
        )

    # 5-fold CV to get OOF scores on val (these are the conformal
    # calibration scores)
    val_lgb_scores = np.zeros(len(val_X_rn))
    rare_lgb_scores_folds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
        clf = clf_class()
        clf.fit(X_all[tr_idx], y_all[tr_idx])
        # OOF scores for val samples in this fold
        val_mask_in_fold = te_idx[te_idx < len(val_X_rn)]
        val_lgb_scores[val_mask_in_fold] = clf.predict_proba(X_all[val_mask_in_fold])[:, 1]
        # rare scores from this fold (will average)
        rare_lgb_scores_folds.append(clf.predict_proba(rare_X_rn)[:, 1])
    rare_lgb_scores = np.mean(rare_lgb_scores_folds, axis=0)
    np.save(args.out_dir / 'val_lgb_scores.npy', val_lgb_scores)
    np.save(args.out_dir / 'rare_lgb_scores.npy', rare_lgb_scores)
    # Also fit on the full data for the production score on rare
    clf_full = clf_class()
    clf_full.fit(X_all, y_all)
    rare_lgb_full = clf_full.predict_proba(rare_X_rn)[:, 1]

    # AUROC sanity check
    y_eval = np.concatenate([np.zeros(len(val_lgb_scores)),
                              np.ones(len(rare_lgb_full))])
    s_eval = np.concatenate([val_lgb_scores, rare_lgb_full])
    auroc = roc_auc_score(y_eval, s_eval)
    print(f"  LightGBM AUROC (val vs rare): {auroc:.4f}")

    # ---- Predicted classes -----------------------------------------
    val_pred  = val["fine_logits"].argmax(axis=1)
    rare_pred = rare["fine_logits"].argmax(axis=1)

    # ---- Split val into calibration / held-out for coverage tests --
    # We use a 50/50 random split so that the held-out portion gives
    # us an unbiased estimate of empirical ID coverage.
    idx = rng.permutation(len(val_lgb_scores))
    n_cal = len(idx) // 2
    cal_idx, ho_idx = idx[:n_cal], idx[n_cal:]

    cal_scores = val_lgb_scores[cal_idx]
    cal_preds  = val_pred[cal_idx]
    ho_scores  = val_lgb_scores[ho_idx]
    ho_preds   = val_pred[ho_idx]
    
    # ===== Export ID cal/holdout CSVs for conformal_novelty.py =====
    val_softmax_full = sp_softmax(val["fine_logits"], axis=1)
    val_conf = val_softmax_full.max(axis=1)

    pd.DataFrame({
        "pred_class": [CLASS_NAMES[p] for p in val_pred[cal_idx]],
        "confidence": val_conf[cal_idx],
        "ens_score":  val_lgb_scores[cal_idx],
    }).to_csv(args.out_dir / "id_cal.csv", index=False)

    pd.DataFrame({
        "pred_class": [CLASS_NAMES[p] for p in val_pred[ho_idx]],
        "confidence": val_conf[ho_idx],
        "ens_score":  val_lgb_scores[ho_idx],
    }).to_csv(args.out_dir / "id_holdout.csv", index=False)
    print(f"[ok] wrote id_cal.csv (n={len(cal_idx)}) and id_holdout.csv (n={len(ho_idx)})")

    # ---- Conformal p-values ----------------------------------------
    print("Computing conformal p-values...")
    val_p_marg = conformal_pvalues_marginal(ho_scores, cal_scores)
    val_p_mond = conformal_pvalues_mondrian(ho_scores, ho_preds,
                                              cal_scores, cal_preds)
    rare_p_marg = conformal_pvalues_marginal(rare_lgb_full, cal_scores)
    rare_p_mond = conformal_pvalues_mondrian(rare_lgb_full, rare_pred,
                                               cal_scores, cal_preds)

    # ---- APS prediction sets ---------------------------------------
    print("Computing APS prediction sets...")
    val_softmax  = sp_softmax(val["fine_logits"], axis=1)
    rare_softmax = sp_softmax(rare["fine_logits"], axis=1)
    cal_sm = val_softmax[cal_idx]
    cal_lb = val["labels"][cal_idx]
    ho_sm  = val_softmax[ho_idx]
    ho_lb  = val["labels"][ho_idx]

    sizes_id, sizes_ood, singleton_id, singleton_ood, q_per_alpha = (
        {}, {}, {}, {}, {}
    )
    for a in APS_ALPHAS:
        in_id, q = aps_prediction_sets(ho_sm, cal_sm, cal_lb, a)
        in_ood, _ = aps_prediction_sets(rare_softmax, cal_sm, cal_lb, a)
        sizes_id[a]  = in_id.sum(axis=1)
        sizes_ood[a] = in_ood.sum(axis=1)
        singleton_id[a]  = (sizes_id[a]  == 1).mean() * 100
        singleton_ood[a] = (sizes_ood[a] == 1).mean() * 100
        q_per_alpha[a] = q

    # ---- Optional MI decomposition ---------------------------------
    mi_available = False
    mi_summary = {}
    if args.per_model_cache is not None and args.per_model_cache.exists():
        print(f"Loading per-model softmax from {args.per_model_cache}...")
        pm = np.load(args.per_model_cache)
        if "val_softmax" in pm and "rare_softmax" in pm:
            pm_val = pm["val_softmax"]  # (M, n_val, K)
            pm_rare = pm["rare_softmax"]
            mi_val, _, _ = mutual_information(pm_val)
            mi_rare, _, _ = mutual_information(pm_rare)
            mi_auroc = roc_auc_score(
                np.concatenate([np.zeros(len(mi_val)), np.ones(len(mi_rare))]),
                np.concatenate([mi_val, mi_rare]),
            )
            mi_summary = {
                "mi_auroc": mi_auroc,
                "mi_id_mean":  float(mi_val.mean()),
                "mi_ood_mean": float(mi_rare.mean()),
                "mi_ratio":    float(mi_rare.mean() / max(mi_val.mean(), 1e-8)),
            }
            mi_available = True
            print(f"  MI AUROC on rare: {mi_auroc:.4f}")
        else:
            print("  WARNING: per-model cache missing 'val_softmax' or "
                  "'rare_softmax' keys; skipping MI.")

    # ---- Per-class Mondrian coverage at alpha = 0.025 --------------
    per_class = {}
    alpha_pc = 0.025
    for k in range(N_CLASSES):
        cm = cal_preds == k
        ho_m = ho_preds == k
        rm = rare_pred == k
        if cm.sum() < 10:
            continue
        cal_k = cal_scores[cm]
        # threshold = (1-alpha) quantile of cal_k
        q = np.quantile(cal_k, 1 - alpha_pc)
        ho_below = (ho_scores[ho_m] <= q).mean() if ho_m.sum() else float("nan")
        rare_above = (rare_lgb_full[rm] > q).mean() if rm.sum() else float("nan")
        per_class[CLASS_NAMES[k]] = {
            "n_cal": int(cm.sum()),
            "n_test": int(rm.sum()),
            "cov": float(ho_below),  # ID samples below threshold = ID coverage
            "det": float(rare_above),  # OOD samples above threshold = detection rate
        }

    # ---- Assemble summary ------------------------------------------
    summary = {
        "n_cal": int(n_cal),
        "n_rare": int(len(rare_lgb_full)),
        "lgb_auroc": float(auroc),
        "coverage": {},
        "aps": {},
        "per_class": per_class,
        "mi_available": mi_available,
        "mi": mi_summary,
    }
    for a in [0.01, 0.025, 0.05, 0.10]:
        summary["coverage"][a] = {
            "marginal_emp": float((val_p_marg > a).mean()),
            "mondrian_emp": float((val_p_mond > a).mean()),
            "mondrian_det": float((rare_p_mond < a).mean()),
        }
    for a in APS_ALPHAS:
        summary["aps"][a] = {
            "mean_id":  float(sizes_id[a].mean()),
            "mean_ood": float(sizes_ood[a].mean()),
            "singleton_id":  float(singleton_id[a]),
            "singleton_ood": float(singleton_ood[a]),
        }

    # ---- Plots and outputs -----------------------------------------
    print("Generating plots and reports...")
    plot_coverage(rare_p_mond, rare_p_marg, val_p_mond, val_p_marg,
                  args.out_dir / "uq_coverage.pdf")
    plot_pvalue_ecdfs(val_p_mond, rare_p_mond,
                      args.out_dir / "uq_pvalues_ecdf.pdf")
    plot_predset_sizes(sizes_id, sizes_ood,
                        args.out_dir / "uq_predset_size.pdf")
    write_markdown_summary(args.out_dir / "uq_summary.md", summary)

    # Per-spectrum CSV (for the paper's per-spectrum table)
    rare_meta = rare.get("meta", [{}] * len(rare_lgb_full))
    df_out = pd.DataFrame({
        "obj_id": [m.get("obj_id", f"row_{i}") for i, m in enumerate(rare_meta)],
        "true_class": [m.get("class", "?") for m in rare_meta],
        "pred_class": [CLASS_NAMES[p] for p in rare_pred],
        "lgb_score": rare_lgb_full,
        "p_marginal": rare_p_marg,
        "p_mondrian": rare_p_mond,
        **{f"in_C_alpha_{a}": (rare_softmax >= 1 - q_per_alpha[a]).sum(axis=1)
           for a in APS_ALPHAS},
    })
    if mi_available:
        df_out["mi"] = mi_rare
    df_out.to_csv(args.out_dir / "uq_per_spectrum.csv", index=False)

    # JSON summary for downstream paper-writing
    (args.out_dir / "uq_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone.  See {args.out_dir} for outputs:")
    for p in sorted(args.out_dir.iterdir()):
        print(f"  {p.name}")


# ============================================================
# Notebook helper — how to obtain per-model softmax for MI
# ============================================================
"""
To enable the MI decomposition (Section 3 of the report), the
extraction loop in the notebook needs to retain per-model softmax
outputs.  The notebook's ensemble_forward(...) currently averages
across models inside the function.  Replace that with the snippet
below in a fresh notebook cell, then save with np.savez:

    @torch.no_grad()
    def extract_per_model(models, loader):
        per_model_sm = []
        for m in models:
            m.eval()
            per_sample = []
            for batch in tqdm(loader, leave=False):
                x  = batch['x'].to(device)
                ph = batch['phase'].to(device)
                z  = batch['z'].to(device)
                out = m(x, ph, z)
                per_sample.append(
                    torch.softmax(out['fine_logits'], dim=-1).cpu().numpy()
                )
            per_model_sm.append(np.concatenate(per_sample, axis=0))
        return np.stack(per_model_sm, axis=0)  # (M, N, K)

    val_pm  = extract_per_model(models, val_loader)
    rare_pm = extract_per_model(models, rare_loader)
    np.savez(CACHE_DIR / 'per_model_softmax.npz',
             val_softmax=val_pm, rare_softmax=rare_pm)

Then run:
    python compute_uq_diagnostics.py \\
        --cache-dir <cache_dir> \\
        --per-model-cache <cache_dir>/per_model_softmax.npz
"""

if __name__ == "__main__":
    main()