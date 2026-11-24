"""
conformal_novelty.py

Empirical support for the two new methodological claims of the
ASTRANet confidence-aware paper:

  (A) AD-score-conditional Mondrian conformal calibration.
      Standard Mondrian gives per-class FDR control on average.
      We additionally verify that the calibration holds within
      strata of the LightGBM anomaly score — including the
      high-AD-score regime where the framework actually fires.

  (B) Per-sample p_wrong = 1 - p_M as the conformal analogue of
      evidential deep learning's per-sample uncertainty.

Required data:
  id_cal.csv       — in-distribution calibration spectra (n_cal ≈ 1631)
  id_holdout.csv   — in-distribution held-out spectra   (n_ho  ≈ 1631)
  anomaly_report.csv — OOD/rare set                     (N = 289)

Required columns in every CSV (rename as appropriate to your notebook):
  pred_class : str    fine-head predicted class label
  confidence : float  softmax p_fine(ŷ|x) of the PREDICTED class
  ens_score  : float  LightGBM cross-family OOF anomaly score in [0,1]

Outputs:
  per_class_thresholds.csv         per-class Mondrian thresholds
  ad_conditional_coverage.csv      one row per AD-score quintile (A)
  pwrong_per_sample.csv            anomaly_report.csv + p_wrong column (B)
  figures/uq_ad_conditional_coverage.pdf    diagnostic figure for §6.1

Cite as Eq.(2) of paper for the Mondrian formula:
  p_M(x*) = (#{i : ŷ_i = ŷ(x*) ∧ s_i ≥ s(x*)} + 1) / (n_class + 1)
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Core conformal primitives
# ----------------------------------------------------------------------

def mondrian_threshold(cal_df: pd.DataFrame,
                       alpha: float = 0.05,
                       score_col: str = "ens_score",
                       class_col: str = "pred_class") -> dict[str, float]:
    """Per-class Mondrian threshold τ_c such that
       P(s(x) ≥ τ_c | ŷ(x) = c) ≤ α   on the calibration set.

    Uses the standard finite-sample-corrected (1−α) empirical quantile:
       τ_c = s_(⌈(n_c+1)(1−α)⌉)
    where the order statistics are of the per-class calibration scores.
    """
    thresholds: dict[str, float] = {}
    for cls, group in cal_df.groupby(class_col):
        scores = np.sort(group[score_col].values)
        n = len(scores)
        # 1-indexed: rank k = ceil((n+1)(1-α)), clipped to n
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        thresholds[cls] = float(scores[k - 1])
    return thresholds


def mondrian_pvalue(test_df: pd.DataFrame,
                     cal_df: pd.DataFrame,
                     score_col: str = "ens_score",
                     class_col: str = "pred_class") -> np.ndarray:
    """Per-sample Mondrian conformal p-value (Eq. 2 of paper).

    p_M(x*) = (#{i in cal : ŷ_i = ŷ(x*) ∧ s_i ≥ s(x*)} + 1) / (n_class + 1)
    """
    p_vals = np.full(len(test_df), np.nan)
    for cls, group in cal_df.groupby(class_col):
        cal_scores = group[score_col].values
        n = len(cal_scores)
        mask = (test_df[class_col].values == cls)
        s_star = test_df.loc[mask, score_col].values
        # Vectorised tail probability
        ge_counts = np.sum(cal_scores[None, :] >= s_star[:, None], axis=1)
        p_vals[mask] = (ge_counts + 1.0) / (n + 1.0)
    if np.any(np.isnan(p_vals)):
        n_orphan = int(np.isnan(p_vals).sum())
        print(f"[warn] {n_orphan} test samples have a predicted class not "
              f"present in the calibration set; p-value left as NaN.")
    return p_vals


# ----------------------------------------------------------------------
# (A) AD-score-conditional coverage diagnostic
# ----------------------------------------------------------------------

def ad_conditional_coverage(holdout_df: pd.DataFrame,
                              cal_df: pd.DataFrame,
                              alpha: float = 0.05,
                              n_quantiles: int = 5,
                              score_col: str = "ens_score",
                              class_col: str = "pred_class") -> pd.DataFrame:
    """Stratify the in-distribution held-out set into n_quantiles bins
    of the AD score, compute the empirical Mondrian coverage within each
    stratum, and return a tidy summary table.

    A passing diagnostic shows empirical coverage ≈ (1 - α) within each
    stratum, including the top quintile (where survey-mode triage fires).
    """
    thresholds = mondrian_threshold(cal_df, alpha=alpha,
                                     score_col=score_col, class_col=class_col)

    h = holdout_df.copy()
    h["tau"] = h[class_col].map(thresholds)
    if h["tau"].isna().any():
        n_orphan = int(h["tau"].isna().sum())
        print(f"[warn] {n_orphan} holdout samples have a predicted class "
              f"absent from calibration — dropped from diagnostic.")
        h = h.dropna(subset=["tau"]).copy()
    # "Covered" ↔ NOT rejected as anomalous, i.e. s(x) < τ.
    # By construction P(s ≥ τ) ≤ α  ⇒  P(covered) ≥ 1 − α.
    h["covered"] = h[score_col].values < h["tau"].values

    # Stratify by AD score quintile (within the held-out distribution,
    # so the bins are equal-size regardless of α)
    h["quintile"] = pd.qcut(h[score_col].values, q=n_quantiles,
                              labels=[f"Q{i+1}" for i in range(n_quantiles)],
                              duplicates="drop")

    rows = []
    for q, g in h.groupby("quintile", observed=True):
        n = len(g)
        cov = float(g["covered"].mean())
        # Wilson 95% CI for a binomial proportion
        from math import sqrt
        z = 1.96
        denom = 1 + z**2 / n
        centre = (cov + z**2 / (2 * n)) / denom
        half = z * sqrt(cov * (1 - cov) / n + z**2 / (4 * n**2)) / denom
        rows.append({
            "quintile": str(q),
            "n": n,
            "ad_min": float(g[score_col].min()),
            "ad_max": float(g[score_col].max()),
            "empirical_coverage": cov,
            "nominal_coverage": 1.0 - alpha,
            "ci_lo": centre - half,
            "ci_hi": centre + half,
            "in_tolerance": abs(cov - (1.0 - alpha)) <= 0.02,
        })
    return pd.DataFrame(rows)


def plot_ad_conditional_coverage(coverage_df: pd.DataFrame,
                                   alpha: float = 0.05,
                                   savepath: str | None = None) -> plt.Figure:
    """Bar chart with 95% Wilson CIs and ±0.02 tolerance band.

    This is figure fig:ad_conditional_coverage referenced in §6.1 of the
    paper.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    q = coverage_df["quintile"].values
    cov = coverage_df["empirical_coverage"].values
    ci_lo = coverage_df["ci_lo"].values
    ci_hi = coverage_df["ci_hi"].values
    err = np.clip(np.vstack([cov - ci_lo, ci_hi - cov]), 0.0, None)   # CI may sit off-centre at the edges
    target = 1.0 - alpha

    ax.axhspan(target - 0.02, target + 0.02, color="tab:red", alpha=0.10,
                label=r"$\pm 0.02$ tolerance band")
    ax.axhline(target, color="tab:red", linestyle="--", lw=1.5,
                label=fr"Nominal $1-\alpha = {target:.2f}$")

    bars = ax.bar(q, cov, color="steelblue", alpha=0.80,
                   edgecolor="navy", linewidth=1.2)
    ax.errorbar(q, cov, yerr=err, fmt="none", ecolor="navy",
                 capsize=4, lw=1.2)

    for bar, c in zip(bars, cov):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.006,
                 f"{c:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("AD-score quintile (Q1: low AD  →  Q5: high AD)")
    ax.set_ylabel(fr"Empirical Mondrian coverage at $\alpha = {alpha:.2f}$")
    ax.set_ylim(max(0.80, target - 0.08), 1.005)
    ax.set_title("AD-conditional conformal coverage diagnostic")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        print(f"[ok] figure saved: {savepath}")
    return fig


# ----------------------------------------------------------------------
# (B) Per-sample p_wrong  =  1 - p_M(s_clf)
# ----------------------------------------------------------------------

def pwrong_per_sample(test_df: pd.DataFrame,
                       cal_df: pd.DataFrame,
                       conf_col: str = "confidence",
                       class_col: str = "pred_class") -> np.ndarray:
    """Conformal probability that the framework's predicted class is wrong
    for each test sample.

        s_clf(x, y) = 1 - p_fine(y | x)
        p_M(x*)     = Mondrian p-value of s_clf(x*, ŷ(x*))
                       against the calibration distribution of
                       s_clf(x_i, y_i) WITHIN the same predicted-class
                       partition (cal samples that the classifier mapped
                       to the same class).
        p_wrong(x*) = 1 - p_M(x*)

    Interpretation: p_wrong ∈ [0,1] is a finite-sample, distribution-free
    calibrated probability that the spectrum is at least as `un-confident'
    on its predicted class as a typical calibration spectrum of that
    class. This is the conformal analogue of the per-sample uncertainty
    output of evidential deep learning (Sensoy 2018; Amini 2020).
    """
    t = test_df.copy()
    c = cal_df.copy()
    t["s_clf"] = 1.0 - t[conf_col].values
    c["s_clf"] = 1.0 - c[conf_col].values

    p_vals = np.full(len(t), np.nan)
    for cls, group in c.groupby(class_col):
        cal_s = group["s_clf"].values
        n = len(cal_s)
        mask = (t[class_col].values == cls)
        s_star = t.loc[mask, "s_clf"].values
        ge_counts = np.sum(cal_s[None, :] >= s_star[:, None], axis=1)
        p_vals[mask] = (ge_counts + 1.0) / (n + 1.0)

    return 1.0 - p_vals


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cal",     default="id_cal.csv",
                    help="in-distribution calibration set")
    ap.add_argument("--holdout", default="id_holdout.csv",
                    help="in-distribution held-out set")
    ap.add_argument("--ood",     default="anomaly_report.csv",
                    help="OOD/rare-set CSV")
    ap.add_argument("--alpha",   default=0.05, type=float,
                    help="conformal level for coverage diagnostic")
    ap.add_argument("--n-quantiles", default=5, type=int,
                    help="number of AD-score strata")
    ap.add_argument("--outdir", default=".",
                    help="output directory")
    ap.add_argument("--score-col", default="ens_score")
    ap.add_argument("--conf-col",  default="confidence")
    ap.add_argument("--class-col", default="pred_class")
    args = ap.parse_args()

    for path in [args.cal, args.holdout, args.ood]:
        if not os.path.exists(path):
            print(f"[error] file not found: {path}", file=sys.stderr)
            return 1

    cal     = pd.read_csv(args.cal)
    holdout = pd.read_csv(args.holdout)
    ood     = pd.read_csv(args.ood)
    print(f"[ok] loaded  cal={len(cal)}  holdout={len(holdout)}  ood={len(ood)}")

    os.makedirs(args.outdir, exist_ok=True)

    # --- per-class Mondrian thresholds -------------------------------------
    thresholds = mondrian_threshold(cal, alpha=args.alpha,
                                      score_col=args.score_col,
                                      class_col=args.class_col)
    pd.DataFrame([{"pred_class": c, "tau": t}
                  for c, t in thresholds.items()]).to_csv(
        os.path.join(args.outdir, "per_class_thresholds.csv"), index=False)
    print(f"[ok] Mondrian thresholds: {thresholds}")

    # --- (A) AD-conditional coverage --------------------------------------
    cov_df = ad_conditional_coverage(holdout, cal, alpha=args.alpha,
                                       n_quantiles=args.n_quantiles,
                                       score_col=args.score_col,
                                       class_col=args.class_col)
    cov_df.to_csv(os.path.join(args.outdir, "ad_conditional_coverage.csv"),
                   index=False)
    print("\n=== AD-conditional Mondrian coverage at alpha "
          f"= {args.alpha:.3f} (target = {1-args.alpha:.3f}) ===")
    print(cov_df.to_string(index=False))

    fig_path = os.path.join(args.outdir, "figures",
                              "uq_ad_conditional_coverage.pdf")
    plot_ad_conditional_coverage(cov_df, alpha=args.alpha, savepath=fig_path)

    # --- (B) Per-sample p_wrong on the OOD set ----------------------------
    ood["p_wrong"] = pwrong_per_sample(ood, cal,
                                         conf_col=args.conf_col,
                                         class_col=args.class_col)
    out_csv = os.path.join(args.outdir, "pwrong_per_sample.csv")
    ood.to_csv(out_csv, index=False)
    print(f"\n[ok] p_wrong written to {out_csv}")

    # Quick summary on the five case-study spectra
    case_ids = ["ZTF18acyvlkx",   # Regime I  - Blazar
                "ZTF21acddxti",   # Regime II - Ca-rich
                "ZTF21aayokph",   # Regime III - afterglow
                "ZTF24abjjpbo",   # Regime IV - FBOT
                "ZTF22aabjpxh"]   # Regime V  - long GRB
    obj_col = "obj_id" if "obj_id" in ood.columns else ood.columns[0]
    sub = ood[ood[obj_col].isin(case_ids)][
        [obj_col, "true_class", args.class_col, args.conf_col,
         args.score_col, "p_wrong"]
    ].copy()
    print("\n=== Case-study spectra: per-sample p_wrong ===")
    print(sub.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
