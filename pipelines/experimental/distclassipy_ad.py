#!/usr/bin/env python3
"""DistClassiPy-style robust distance anomaly detection (arXiv:2403.12120).

Idea: instead of a single distance (our Mahalanobis), score each spectrum by its
distance to the NEAREST class centroid under MANY distance metrics, with robust
per-class scaling. An anomaly is far from every known-class centroid in (ideally)
several metrics. Robustness comes from:
  - robust centroid (median) + scale (MAD) per class  -> outlier-resistant,
  - a panel of metrics incl. robust ones (canberra, braycurtis, lorentzian, clark),
  - combining metrics (per-metric min-distance -> LightGBM / mean-rank).

We evaluate, on the 289-anomaly set, the recovery @ target FAR for:
  - each metric alone (min-distance-to-centroid),
  - the DistClassiPy panel combined with LightGBM,
  - vs the existing LUNA ensemble (baseline).

Outputs -> results/<spec>/anomaly/distclassipy.csv
Usage: python pipelines/distclassipy_ad.py --spec 6model --target-far 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from _common import REPO, load_config, make_astra_config, load_cache

# scipy cdist metrics + custom robust ones
SCIPY_METRICS = ["euclidean", "cityblock", "chebyshev", "cosine",
                 "correlation", "canberra", "braycurtis"]


def _lorentzian(A, C):
    # sum log(1+|a-c|) — robust, heavy-tail friendly
    out = np.empty((len(A), len(C)))
    for j, c in enumerate(C):
        out[:, j] = np.log1p(np.abs(A - c)).sum(1)
    return out


def _clark(A, C):
    out = np.empty((len(A), len(C)))
    for j, c in enumerate(C):
        d = (A - c); s = np.abs(A) + np.abs(c) + 1e-8
        out[:, j] = np.sqrt(((d / s) ** 2).sum(1))
    return out


def min_dist_score(Xstd, cents, metric):
    """Min distance from each row of Xstd to any class centroid, given metric."""
    if metric == "lorentzian":
        D = _lorentzian(Xstd, cents)
    elif metric == "clark":
        D = _clark(Xstd, cents)
    else:
        D = cdist(Xstd, cents, metric=metric)
    return D.min(1)


def far_rec(rare_s, val_s, far):
    """Held-out FAR + recovery: threshold on val half A, FP on val half B."""
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(val_s)); A, B = idx[:len(idx)//2], idx[len(idx)//2:]
    thr = np.quantile(val_s[A], 1 - far)
    return (rare_s >= thr).mean() * 100, (val_s[B] >= thr).mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--robust", action="store_true", default=True)
    args = ap.parse_args()
    far = args.target_far
    cfg = load_config(); ac = make_astra_config(cfg)
    cache = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "anomaly"; out.mkdir(parents=True, exist_ok=True)

    tr = load_cache(cache, "train"); va = load_cache(cache, "val")
    ra = load_cache(cache, "rare_unknown_combined")
    Xtr, ytr = tr["emb"], np.asarray(tr["labels"])
    Xva, Xra = va["emb"], ra["emb"]

    # Robust global standardisation (median / MAD) -> outlier-resistant scaling
    med = np.median(Xtr, 0)
    mad = np.median(np.abs(Xtr - med), 0) * 1.4826 + 1e-6
    def std(X): return (X - med) / mad
    Xtr_s, Xva_s, Xra_s = std(Xtr), std(Xva), std(Xra)

    # Robust per-class centroids (median in standardised space)
    classes = [c for c in np.unique(ytr) if (ytr == c).sum() >= 10]
    cents = np.array([np.median(Xtr_s[ytr == c], 0) for c in classes])

    metrics = SCIPY_METRICS + ["lorentzian", "clark"]
    val_scores, rare_scores, rows = {}, {}, []
    for m in metrics:
        vs = min_dist_score(Xva_s, cents, m)
        rs = min_dist_score(Xra_s, cents, m)
        val_scores[m], rare_scores[m] = vs, rs
        rec, fp = far_rec(rs, vs, far)
        rows.append((m, rec, fp))

    # Combine the metric panel with LightGBM (val=ID vs rare=OOD, OOF threshold)
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    Xv = np.column_stack([val_scores[m] for m in metrics])
    Xr = np.column_stack([rare_scores[m] for m in metrics])
    X = np.vstack([Xv, Xr]); y = np.r_[np.zeros(len(Xv)), np.ones(len(Xr))]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    val_oof = np.zeros(len(Xv)); rare_f = []
    for tri, tei in skf.split(X, y):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                 num_leaves=31, class_weight="balanced", verbosity=-1)
        clf.fit(X[tri], y[tri])
        m_in = tei[tei < len(Xv)]
        val_oof[m_in] = clf.predict_proba(X[m_in])[:, 1]
        rare_f.append(clf.predict_proba(Xr)[:, 1])
    rare_comb = np.mean(rare_f, 0)
    rec, fp = far_rec(rare_comb, val_oof, far)
    rows.append(("DistClassiPy panel + LightGBM", rec, fp))

    import pandas as pd
    df = pd.DataFrame(rows, columns=["metric", "recovery_%", "realised_FAR_%"]) \
        .sort_values("recovery_%", ascending=False)
    df.to_csv(out / "distclassipy.csv", index=False)
    print(f"[{args.spec}] DistClassiPy-style robust distance AD  (rare N={len(Xra)}, FAR={far*100:.0f}%)\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))
    hit = df[df["recovery_%"] >= 80]
    print("\n>= 80%:", " / ".join(hit["metric"]) if len(hit) else "none")


if __name__ == "__main__":
    main()
