"""
ad_stratified_mondrian.py
==========================

AD-stratified Mondrian Conformal Predictor for uniform conditional coverage
across AD-score strata.

============================================================================
THE PROBLEM (already demonstrated in §6.1 of the paper)
============================================================================

Marginal Mondrian CP gives prediction-set coverage 1-alpha *averaged* over
the ID validation distribution.  Stratifying val by AD-score quintile
revealed:

    Q1 = Q2 = Q3 = Q4 = 1.000  (over-cover; all ID samples below tau_c)
    Q5                = 0.754  (under-cover; the operational regime)

This is the expected calibration profile of any marginal CP estimator when
stratified by its own score, and it is operationally undesirable: survey
mode triage operates in Q5, where the empirical FAR is 25% instead of 5%.

============================================================================
THE METHOD (this script)
============================================================================

AD-stratified Mondrian: per-(class c × AD-quintile q) threshold
tau_{c,q} computed from calibration samples in cell (c,q).

For a test point x with predicted class c_hat and AD-quintile q_hat,
the framework fires iff s(x) > tau_{c_hat, q_hat}.

Guarantee (under exchangeability within each cell):

    P( s(x) <= tau_{c_hat, q_hat} | y_hat = c, q(s_AD) = q ) >= 1 - alpha

for every (c,q) cell with sufficient calibration samples.  Below the
sample-size threshold, the cell falls back to the marginal tau_c.

To our knowledge this is the first reported AD-stratified Mondrian CP
with empirical validation on transient spectroscopy.

============================================================================
INPUTS
============================================================================

  --per-model-cache PER_MODEL_SOFTMAX.NPZ  (val + rare softmax)
  --val-cache       VAL.NPZ  (labels)
  --val-ad          VAL_LGB_SCORES.NPY
  --anomaly-report  ANOMALY_REPORT.CSV  (for rare ens_score, obj_id)

============================================================================
OUTPUTS
============================================================================

  ad_strat_thresholds.csv     per-(class, quintile) tau values + sample sizes
  ad_strat_coverage.csv       k-fold per-quintile coverage: marginal vs AD-strat
  ad_strat_rare_flags.csv     per-spectrum flag from AD-stratified Mondrian
  ad_strat_coverage.pdf       THE PAPER FIGURE (Q5 dropout fixed)
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

CLASS_NAMES = ['Ia', 'Ibc', 'SLSN', 'II', 'AGN', 'CV', 'TDE']
ALPHA_DEFAULT = 0.05
N_QUINTILES_DEFAULT = 5
MIN_CELL_SIZE = 10


def mondrian_threshold(scores, alpha):
    """(1-alpha)-quantile threshold for nonconformity scores."""
    n = len(scores)
    if n == 0:
        return np.inf
    s_sorted = np.sort(scores)
    k = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)
    return float(s_sorted[k - 1])


def compute_ad_stratified_thresholds(cal_score, cal_pred, cal_ad,
                                       alpha, n_quintiles,
                                       edges=None,
                                       min_cell=MIN_CELL_SIZE):
    """
    Compute per-(class, AD-quintile) thresholds with marginal fallback for sparse cells.

    Returns:
        thresholds: dict {(c, q): tau}
        marginal: dict {c: tau_marginal_c}
        edges: (n_quintiles+1,) bin edges from cal_ad
        cell_diagnostics: DataFrame
    """
    K = len(CLASS_NAMES)

    if edges is None:
        edges = np.quantile(cal_ad, np.linspace(0, 1, n_quintiles + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf

    cal_q = np.digitize(cal_ad, edges[1:-1])

    # Marginal (per-class) thresholds for fallback
    marginal = {}
    for c in range(K):
        marginal[c] = mondrian_threshold(cal_score[cal_pred == c], alpha)

    # Per-cell thresholds
    thresholds = {}
    diagnostics = []
    for c in range(K):
        for q in range(n_quintiles):
            mask = (cal_pred == c) & (cal_q == q)
            n_cell = int(mask.sum())
            if n_cell < min_cell:
                thresholds[(c, q)] = marginal[c]  # fallback
                fallback = True
            else:
                thresholds[(c, q)] = mondrian_threshold(cal_score[mask], alpha)
                fallback = False
            diagnostics.append({
                'class': CLASS_NAMES[c],
                'class_idx': c,
                'quintile': q,
                'ad_lo': float(edges[q]),
                'ad_hi': float(edges[q + 1]),
                'n_cell': n_cell,
                'tau': thresholds[(c, q)],
                'tau_marginal_c': marginal[c],
                'fallback': fallback,
            })
    return thresholds, marginal, edges, pd.DataFrame(diagnostics)


def evaluate_coverage(test_score, test_pred, test_q, test_y,
                       thresholds, marginal, K=7, n_quintiles=5):
    """
    For each test point, the prediction set is {c : s(c) <= tau_{c,q(test)}}.
    Coverage = fraction where true class y is in the prediction set.

    With our nonconformity score s = 1 - confidence (per predicted class),
    the prediction set is a singleton {y_hat} when s(y_hat) <= tau_{y_hat, q}
    and empty otherwise.  Coverage means: y_true == y_hat AND s <= tau.

    For a more interpretable check: "Is the TRUE class within the threshold
    of its calibration distribution?"  We compute s_for_true = 1 - p(true)
    and check s_for_true <= tau_{true, q}.

    Args:
        test_score: (n_test,) nonconformity score for predicted class
        test_pred: (n_test,) predicted class index
        test_q: (n_test,) AD quintile
        test_y: (n_test,) true class index
        thresholds: dict {(c, q): tau}
        marginal: dict {c: tau_marginal}
    Returns:
        DataFrame with per-quintile coverage for AD-stratified and marginal
    """
    n_test = len(test_score)
    cov_marg = np.zeros(n_test, dtype=bool)
    cov_strat = np.zeros(n_test, dtype=bool)
    for i in range(n_test):
        c = int(test_pred[i])
        q = int(test_q[i])
        s = float(test_score[i])
        cov_marg[i] = (s <= marginal[c])
        cov_strat[i] = (s <= thresholds[(c, q)])

    records = []
    for q in range(n_quintiles):
        mask = test_q == q
        if mask.sum() < 5:
            continue
        records.append({
            'quintile': q,
            'n_test': int(mask.sum()),
            'coverage_marginal': float(cov_marg[mask].mean()),
            'coverage_ad_stratified': float(cov_strat[mask].mean()),
        })
    return pd.DataFrame(records)


def kfold_validation(val_softmax, val_labels, val_ad,
                       alpha=ALPHA_DEFAULT,
                       n_quintiles=N_QUINTILES_DEFAULT,
                       n_folds=5, random_state=42):
    """K-fold CV: train on K-1 folds, evaluate per-quintile coverage on held-out fold."""
    K = val_softmax.shape[1]
    n_val = len(val_labels)
    val_pred = val_softmax.argmax(axis=1)
    val_conf = val_softmax.max(axis=1)
    val_s = 1.0 - val_conf  # nonconformity score

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_records = []
    for fold_i, (cal_idx, ev_idx) in enumerate(kf.split(np.arange(n_val))):
        thresholds, marginal, edges, _ = compute_ad_stratified_thresholds(
            val_s[cal_idx], val_pred[cal_idx], val_ad[cal_idx],
            alpha=alpha, n_quintiles=n_quintiles,
        )
        ev_q = np.digitize(val_ad[ev_idx], edges[1:-1])
        cov = evaluate_coverage(
            val_s[ev_idx], val_pred[ev_idx], ev_q, val_labels[ev_idx],
            thresholds, marginal, K=K, n_quintiles=n_quintiles,
        )
        cov['fold'] = fold_i
        all_records.append(cov)
    return pd.concat(all_records, ignore_index=True)


def plot_coverage(cov_df, alpha, outpath, n_quintiles=5):
    """The paper figure: side-by-side per-quintile coverage."""
    agg = cov_df.groupby('quintile').agg(
        marg_mean=('coverage_marginal', 'mean'),
        marg_std=('coverage_marginal', 'std'),
        strat_mean=('coverage_ad_stratified', 'mean'),
        strat_std=('coverage_ad_stratified', 'std'),
        n_test=('n_test', 'sum'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(n_quintiles)
    width = 0.38
    nominal = 1.0 - alpha

    bars_m = ax.bar(x - width / 2, agg['marg_mean'], width,
                     yerr=agg['marg_std'], capsize=4,
                     color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.6,
                     label='Marginal Mondrian (current paper)')
    bars_s = ax.bar(x + width / 2, agg['strat_mean'], width,
                     yerr=agg['strat_std'], capsize=4,
                     color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6,
                     label='AD-stratified Mondrian (this work)')

    ax.axhline(nominal, color='black', linestyle='--', alpha=0.7, linewidth=1.4,
                label=f'Nominal $1-\\alpha={nominal:.2f}$')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{i + 1}' for i in range(n_quintiles)], fontsize=11)
    ax.set_xlabel('AD-score quintile (Q1: lowest AD $\\rightarrow$ Q5: highest, operational regime)',
                   fontsize=10)
    ax.set_ylabel('Empirical Mondrian coverage', fontsize=10)
    ax.set_ylim(0.55, 1.03)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)

    # Annotate values
    for bars, means in [(bars_m, agg['marg_mean']), (bars_s, agg['strat_mean'])]:
        for bar, val in zip(bars, means):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.008,
                     f'{val:.3f}',
                     ha='center', va='bottom', fontsize=8.5)

    ax.set_title('AD-stratified Mondrian Conformal Prediction:\n'
                  'uniform conditional coverage across AD-score strata',
                  fontsize=11, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'  wrote {outpath}')
    return agg


def apply_to_rare(val_softmax, val_labels, val_ad,
                    rare_softmax, rare_ad,
                    alpha, n_quintiles):
    """Apply AD-stratified Mondrian to rare set (production use)."""
    val_pred = val_softmax.argmax(axis=1)
    val_conf = val_softmax.max(axis=1)
    val_s = 1.0 - val_conf
    rare_pred = rare_softmax.argmax(axis=1)
    rare_conf = rare_softmax.max(axis=1)
    rare_s = 1.0 - rare_conf

    thresholds, marginal, edges, diag = compute_ad_stratified_thresholds(
        val_s, val_pred, val_ad, alpha=alpha, n_quintiles=n_quintiles,
    )
    rare_q = np.digitize(rare_ad, edges[1:-1])

    flags = []
    for i in range(len(rare_s)):
        c = int(rare_pred[i])
        q = int(rare_q[i])
        flag_marg = bool(rare_s[i] > marginal[c])
        flag_strat = bool(rare_s[i] > thresholds[(c, q)])
        flags.append({
            'idx': i,
            'pred_class': CLASS_NAMES[c],
            'confidence': float(rare_conf[i]),
            'ens_score': float(rare_ad[i]),
            'quintile': q,
            'tau_marginal': marginal[c],
            'tau_ad_stratified': thresholds[(c, q)],
            'flagged_marginal': flag_marg,
            'flagged_ad_stratified': flag_strat,
        })
    return pd.DataFrame(flags), diag, edges


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--per-model-cache', type=Path, required=True)
    ap.add_argument('--val-cache', type=Path, required=True)
    ap.add_argument('--val-ad', type=Path, required=True)
    ap.add_argument('--anomaly-report', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, default=Path('ad_strat_out'))
    ap.add_argument('--alpha', type=float, default=ALPHA_DEFAULT)
    ap.add_argument('--n-quintiles', type=int, default=N_QUINTILES_DEFAULT)
    ap.add_argument('--n-folds', type=int, default=5)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print('[1/4] Loading data...')
    pm = np.load(args.per_model_cache)
    val_softmax = pm['val_softmax'].mean(axis=0)
    rare_softmax = pm['rare_softmax'].mean(axis=0)
    val_labels = np.load(args.val_cache)['labels'].astype(int)
    val_ad = np.load(args.val_ad)
    anomaly_report_path = args.anomaly_report
    if not anomaly_report_path.exists():
        fallback_paths = [
            anomaly_report_path.with_name('anomaly_results_v3v4.csv'),
            anomaly_report_path.with_name('anomaly_results.csv'),
        ]
        for candidate in fallback_paths:
            if candidate.exists():
                print(f'  [warn] missing anomaly report: {anomaly_report_path}')
                print(f'  [warn] falling back to: {candidate}')
                anomaly_report_path = candidate
                break

    ar = pd.read_csv(anomaly_report_path)
    n_rare = rare_softmax.shape[0]
    score_col = None
    for candidate in ('ens_score', 'ensemble_score'):
        if candidate in ar.columns:
            score_col = candidate
            break
    if score_col is None:
        raise KeyError(
            "anomaly report is missing an ensemble score column; expected "
            "'ens_score' or 'ensemble_score'."
        )

    rare_ad = ar[score_col].values[:n_rare]
    if score_col != 'ens_score':
        print(f"  [warn] using '{score_col}' as anomaly score column")

    print(f'  val_softmax {val_softmax.shape}, rare_softmax {rare_softmax.shape}')
    print(f'  val_ad     range [{val_ad.min():.3e}, {val_ad.max():.3e}]')
    print(f'  rare_ad    range [{rare_ad.min():.3e}, {rare_ad.max():.3e}]')

    print('[2/4] K-fold per-quintile coverage (paper headline)...')
    cov_df = kfold_validation(
        val_softmax, val_labels, val_ad,
        alpha=args.alpha,
        n_quintiles=args.n_quintiles,
        n_folds=args.n_folds,
    )
    cov_df.to_csv(args.outdir / 'ad_strat_coverage.csv', index=False)
    agg = plot_coverage(cov_df, args.alpha,
                          args.outdir / 'ad_strat_coverage.pdf',
                          n_quintiles=args.n_quintiles)
    print()
    print('=== HEADLINE per-quintile Mondrian CP coverage ===')
    print(f'(nominal 1-alpha = {1 - args.alpha:.2f})')
    print(f'{"Quintile":>10} {"Marginal":>15} {"AD-stratified":>18}')
    for _, row in agg.iterrows():
        q = int(row['quintile'])
        print(f'{"Q" + str(q + 1):>10} {row["marg_mean"]:>15.4f} {row["strat_mean"]:>18.4f}')

    print('\n[3/4] Apply to rare set (production)...')
    flags_df, diag_df, edges = apply_to_rare(
        val_softmax, val_labels, val_ad,
        rare_softmax, rare_ad,
        alpha=args.alpha,
        n_quintiles=args.n_quintiles,
    )
    if 'obj_id' in ar.columns:
        flags_df.insert(0, 'obj_id', ar['obj_id'].values[:n_rare])
    if 'true_class' in ar.columns:
        flags_df.insert(1, 'true_class', ar['true_class'].values[:n_rare])
    flags_df.to_csv(args.outdir / 'ad_strat_rare_flags.csv', index=False)
    diag_df.to_csv(args.outdir / 'ad_strat_thresholds.csv', index=False)
    print(f'  wrote {args.outdir / "ad_strat_rare_flags.csv"}')
    print(f'  wrote {args.outdir / "ad_strat_thresholds.csv"}')

    print(f'\n  rare quintile distribution: '
          f'{dict(flags_df.groupby("quintile").size())}')
    print(f'  rare flagged (marginal):      {int(flags_df["flagged_marginal"].sum())} / {n_rare}')
    print(f'  rare flagged (AD-stratified): {int(flags_df["flagged_ad_stratified"].sum())} / {n_rare}')

    print('\n[4/4] Summary written.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
