#!/usr/bin/env python3
"""DiMMAD — Distance Multi-Metric Anomaly Detection (arXiv:2510.23702).

Unsupervised multi-metric distance ensemble. For each of several distance metrics
(different feature-space geometries), score = min distance to a robust class centroid.
Each metric's scores are rank-normalised against the ID (val) distribution, then
aggregated (mean rank) into one anomaly score. Unsupervised -> no training on the
rare set, so NO leakage and an honest recovery estimate.

Evaluated at the target FAR (threshold on val, applied to rare).
Outputs -> results/<spec>/anomaly/dimmad.csv
Usage: python pipelines/dimmad_ad.py --spec 6model --target-far 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from _common import REPO, load_config, make_astra_config, load_cache

# Diverse geometries (signed-embedding-safe). braycurtis uses a non-negative map.
METRICS = ["euclidean", "cityblock", "chebyshev", "cosine", "correlation",
           ("minkowski", 3), "mahalanobis", "braycurtis_pos"]


def _centroids(Xs, y):
    classes = [c for c in np.unique(y) if (y == c).sum() >= 10]
    return np.array([np.median(Xs[y == c], 0) for c in classes])


def _scores(Xq, cents, metric, cov_inv=None):
    if metric == "mahalanobis":
        D = cdist(Xq, cents, metric="mahalanobis", VI=cov_inv)
    elif isinstance(metric, tuple):
        D = cdist(Xq, cents, metric=metric[0], p=metric[1])
    elif metric == "braycurtis_pos":
        D = cdist(np.log1p(np.abs(Xq)), np.log1p(np.abs(cents)), metric="braycurtis")
    else:
        D = cdist(Xq, cents, metric=metric)
    return D.min(1)


def rank_vs(ref, x):
    """Rank of each x within ref distribution -> [0,1] (1 = most anomalous)."""
    ref_sorted = np.sort(ref)
    return np.searchsorted(ref_sorted, x, side="right") / len(ref)


def held_out_far(rare_agg, val_agg, far):
    rng = np.random.default_rng(42); idx = rng.permutation(len(val_agg))
    A, B = idx[:len(idx)//2], idx[len(idx)//2:]
    thr = np.quantile(val_agg[A], 1 - far)
    return (rare_agg >= thr).mean()*100, (val_agg[B] >= thr).mean()*100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--target-far", type=float, default=0.01)
    args = ap.parse_args()
    far = args.target_far
    cfg = load_config(); ac = make_astra_config(cfg)
    cache = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "anomaly"; out.mkdir(parents=True, exist_ok=True)

    tr = load_cache(cache, "train"); va = load_cache(cache, "val")
    ra = load_cache(cache, "rare_unknown_combined")
    Xtr, y = tr["emb"], np.asarray(tr["labels"])

    # robust median/MAD standardisation
    med = np.median(Xtr, 0); mad = np.median(np.abs(Xtr - med), 0)*1.4826 + 1e-6
    s = lambda X: (X - med)/mad
    Xtr_s, Xva_s, Xra_s = s(Xtr), s(va["emb"]), s(ra["emb"])
    cents = _centroids(Xtr_s, y)
    cov_inv = np.linalg.inv(np.cov(Xtr_s, rowvar=False) + 1e-6*np.eye(Xtr_s.shape[1]))

    per_metric, vrank, rrank, rows = {}, [], [], []
    for m in METRICS:
        name = m[0]+str(m[1]) if isinstance(m, tuple) else m
        vs = _scores(Xva_s, cents, m, cov_inv); rs = _scores(Xra_s, cents, m, cov_inv)
        vr = rank_vs(vs, vs); rr = rank_vs(vs, rs)   # rank vs val distribution
        vrank.append(vr); rrank.append(rr)
        rec, fp = held_out_far(rs, vs, far)
        rows.append((name, rec, fp))

    # DiMMAD aggregations across metrics
    VR = np.column_stack(vrank); RR = np.column_stack(rrank)
    for agg_name, agg in [("mean-rank", np.mean), ("median-rank", np.median),
                          ("max-rank", np.max)]:
        va_a, ra_a = agg(VR, 1), agg(RR, 1)
        rec, fp = held_out_far(ra_a, va_a, far)
        rows.append((f"DiMMAD ({agg_name})", rec, fp))

    df = pd.DataFrame(rows, columns=["metric/agg", "recovery_%", "realised_FAR_%"]) \
        .sort_values("recovery_%", ascending=False)
    df.to_csv(out / "dimmad.csv", index=False)
    print(f"[{args.spec}] DiMMAD multi-metric distance ensemble  (rare N={len(Xra_s)}, FAR={far*100:.0f}%)\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))
    best = df.iloc[0]
    print(f"\n  best: {best['metric/agg']} = {best['recovery_%']:.1f}%  "
          f"({'>= 80%!' if best['recovery_%'] >= 80 else 'still < 80%'})")


if __name__ == "__main__":
    main()
