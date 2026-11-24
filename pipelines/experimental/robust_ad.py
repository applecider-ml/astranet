#!/usr/bin/env python3
"""Maximal robust anomaly detector for the hard 289-set.

Combines:
  - the 16 LUNA scorers (uncertainty / distance / density / hybrid),
  - a DistClassiPy-style distance panel (arXiv:2403.12120): distance to the nearest
    class centroid under several metrics, with ROBUST median/MAD scaling,
  - a "margin" feature: 2nd-nearest / nearest centroid distance (anomalies are
    ambiguous -> small margin),
into one LightGBM, evaluated honestly (held-out FAR) at the target FAR AND at the
F1-optimal threshold (the operating point your old notebook used).

Outputs -> results/<spec>/anomaly/robust_ad.csv
Usage: python pipelines/robust_ad.py --spec 6model --target-far 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from _common import REPO, load_config, make_astra_config, load_cache
from astranet.model import build_hierarchy_mappings
from luna import LUNAPipeline

# Metrics suited to signed neural embeddings (drop non-negative-only ones)
METRICS = ["euclidean", "cityblock", "chebyshev", "cosine", "correlation"]


def distclassipy_feats(Xtr, ytr, Xq):
    """Per-metric min-distance + margin to robust class centroids."""
    med = np.median(Xtr, 0); mad = np.median(np.abs(Xtr - med), 0) * 1.4826 + 1e-6
    s = lambda X: (X - med) / mad
    classes = [c for c in np.unique(ytr) if (ytr == c).sum() >= 10]
    cents = np.array([np.median(s(Xtr)[ytr == c], 0) for c in classes])
    Xqs = s(Xq)
    feats = {}
    for m in METRICS:
        D = cdist(Xqs, cents, metric=m)
        Ds = np.sort(D, 1)
        feats[f"dcp_{m}_min"] = Ds[:, 0]
        feats[f"dcp_{m}_margin"] = Ds[:, 0] / (Ds[:, 1] + 1e-8)   # near 1 => ambiguous
    return feats


def _to_luna(d):
    o = {"embeddings": d["emb"].astype(np.float32), "logits": d["fine_logits"].astype(np.float32),
         "coarse_logits": d["coarse_logits"].astype(np.float32), "projections": d["proj_fine"].astype(np.float32)}
    if d.get("labels") is not None:
        o["labels"] = np.asarray(d["labels"])
    return o


def held_out(rare_s, val_s, far):
    rng = np.random.default_rng(42); idx = rng.permutation(len(val_s))
    A, B = idx[:len(idx)//2], idx[len(idx)//2:]
    thr = np.quantile(val_s[A], 1 - far)
    return (rare_s >= thr).mean()*100, (val_s[B] >= thr).mean()*100


def f1_opt(rare_s, val_s):
    """F1-optimal threshold over the combined val(ID)+rare(OOD) scores."""
    s = np.r_[val_s, rare_s]; y = np.r_[np.zeros(len(val_s)), np.ones(len(rare_s))]
    best = (0, 0, 0, 0)
    for t in np.quantile(s, np.linspace(0.5, 0.999, 200)):
        pred = s >= t; tp = (pred & (y == 1)).sum()
        if tp == 0:
            continue
        prec = tp / pred.sum(); rec = tp / (y == 1).sum()
        f1 = 2*prec*rec/(prec+rec)
        if f1 > best[0]:
            fp = ((val_s >= t).mean())*100
            best = (f1, rec*100, fp, t)
    return best  # f1, recovery%, FP%, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--target-far", type=float, default=0.01)
    args = ap.parse_args()
    far = args.target_far
    cfg = load_config(); ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "anomaly"; out.mkdir(parents=True, exist_ok=True)

    tr = load_cache(cache, "train"); va = load_cache(cache, "val")
    ra = load_cache(cache, "rare_unknown_combined")
    ytr = np.asarray(tr["labels"])

    # LUNA 16 scorers
    pipe = LUNAPipeline(combiner="lightgbm", target_far=far, fine_to_coarse=f2c)
    print(f"[{args.spec}] fitting LUNA scorers ...", flush=True)
    pipe.fit(_to_luna(tr))
    vs = pipe.score(_to_luna(va)); rs = pipe.score(_to_luna(ra))
    methods = pipe.active_methods

    # DistClassiPy distance panel
    dv = distclassipy_feats(tr["emb"], ytr, va["emb"])
    dr = distclassipy_feats(tr["emb"], ytr, ra["emb"])

    def matrix(score_dict, dcp):
        cols = [score_dict[m] for m in methods] + [dcp[k] for k in dcp]
        return np.column_stack(cols)
    Xv = matrix(vs, dv); Xr = matrix(rs, dr)
    names = methods + list(dv)

    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    nval = len(Xv)
    X = np.vstack([Xv, Xr]); y = np.r_[np.zeros(nval), np.ones(len(Xr))]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    voof = np.zeros(nval); rcomb = np.zeros(len(Xr))   # clean OOF for BOTH val and rare
    for tri, tei in skf.split(X, y):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31,
                                 class_weight="balanced", verbosity=-1)
        clf.fit(X[tri], y[tri])
        p = clf.predict_proba(X[tei])[:, 1]
        v_mask = tei < nval
        voof[tei[v_mask]] = p[v_mask]
        rcomb[tei[~v_mask] - nval] = p[~v_mask]   # rare scored only when held out

    rec_far, fp_far = held_out(rcomb, voof, far)
    f1, rec_f1, fp_f1, _ = f1_opt(rcomb, voof)
    print(f"\nrare N = {len(Xr)}   features = {len(names)} (16 LUNA + {len(dv)} DistClassiPy)\n")
    print(f"  Robust combined @ {far*100:.0f}% FAR : recovery = {rec_far:5.1f}%  (FP {fp_far:.1f}%)")
    print(f"  Robust combined @ F1-optimal : recovery = {rec_f1:5.1f}%  (FP {fp_f1:.1f}%, F1 {f1:.3f})")

    # feature importance (which signals matter)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31,
                             class_weight="balanced", verbosity=-1).fit(X, y)
    imp = pd.DataFrame({"feature": names, "importance": clf.feature_importances_}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(out / "robust_ad_importance.csv", index=False)
    pd.DataFrame([{"strategy": "robust_combined@FAR", "recovery_%": rec_far, "FP_%": fp_far},
                  {"strategy": "robust_combined@F1opt", "recovery_%": rec_f1, "FP_%": fp_f1}]) \
        .to_csv(out / "robust_ad.csv", index=False)
    print("\n  top features:", ", ".join(imp.head(8)["feature"]))
    print(f"\n  {'>= 80% reached!' if max(rec_far, rec_f1) >= 80 else 'still < 80% at ~1% FAR'}")


if __name__ == "__main__":
    main()
