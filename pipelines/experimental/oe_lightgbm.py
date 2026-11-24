#!/usr/bin/env python3
"""Outlier-Exposure at the LightGBM detector, evaluated LEAVE-ONE-CLASS-OUT.

For each rare class C: train the LightGBM detector on val (ID) + all rare EXCEPT C,
then test detection on C (held out). This is honest (C never seen) and tests whether
exposing the detector to some anomaly types boosts detection of UNSEEN types @1% FAR.

Features: 16 LUNA scorers + DistClassiPy multi-metric distances (richest panel).
Threshold: 1% FAR on a held-out half of val.

Outputs -> results/<spec>/anomaly/oe_leave_class_out.csv
Usage: python pipelines/oe_lightgbm.py --spec 6model --target-far 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipelines/ for _common
from _common import REPO, load_config, make_astra_config, load_cache
from astranet.model import build_hierarchy_mappings
from luna import LUNAPipeline

METRICS = ["euclidean", "cityblock", "chebyshev", "cosine", "correlation"]


def dcp_feats(Xtr, ytr, Xq):
    med = np.median(Xtr, 0); mad = np.median(np.abs(Xtr - med), 0)*1.4826 + 1e-6
    s = lambda X: (X - med)/mad
    classes = [c for c in np.unique(ytr) if (ytr == c).sum() >= 10]
    cents = np.array([np.median(s(Xtr)[ytr == c], 0) for c in classes])
    Xs = s(Xq); cols = {}
    for m in METRICS:
        D = np.sort(cdist(Xs, cents, metric=m), 1)
        cols[f"dcp_{m}_min"] = D[:, 0]; cols[f"dcp_{m}_margin"] = D[:, 0]/(D[:, 1]+1e-8)
    return cols


def _to_luna(d):
    o = {"embeddings": d["emb"].astype(np.float32), "logits": d["fine_logits"].astype(np.float32),
         "coarse_logits": d["coarse_logits"].astype(np.float32), "projections": d["proj_fine"].astype(np.float32)}
    if d.get("labels") is not None:
        o["labels"] = np.asarray(d["labels"])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="6model")
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
    rare_cls = np.array([m.get("class", "?") for m in ra["meta"]])

    pipe = LUNAPipeline(combiner="lightgbm", target_far=far, fine_to_coarse=f2c)
    print(f"[{args.spec}] fitting LUNA scorers ...", flush=True)
    pipe.fit(_to_luna(tr))
    vs = pipe.score(_to_luna(va)); rs = pipe.score(_to_luna(ra))
    methods = pipe.active_methods
    dv = dcp_feats(tr["emb"], ytr, va["emb"]); dr = dcp_feats(tr["emb"], ytr, ra["emb"])
    Xv = np.column_stack([vs[m] for m in methods] + [dv[k] for k in dv])
    Xr = np.column_stack([rs[m] for m in methods] + [dr[k] for k in dr])

    import lightgbm as lgb
    rng = np.random.default_rng(42)
    vidx = rng.permutation(len(Xv)); A, B = vidx[:len(vidx)//2], vidx[len(vidx)//2:]

    classes = sorted(set(rare_cls), key=lambda c: -(rare_cls == c).sum())
    print(f"\n{'rare class':20s} {'OE leave-out det':>16s} {'(baseline det)':>15s}  N")
    print("-"*62)
    det_oe = np.zeros(len(Xr), bool)
    # baseline: train on val + ALL rare (in-sample), for reference
    base = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31,
                              class_weight="balanced", verbosity=-1)
    base.fit(np.vstack([Xv, Xr]), np.r_[np.zeros(len(Xv)), np.ones(len(Xr))])
    thr_b = np.quantile(base.predict_proba(Xv[A])[:, 1], 1-far)
    base_r = base.predict_proba(Xr)[:, 1] >= thr_b

    rows = []
    for C in classes:
        mask_C = rare_cls == C; n = mask_C.sum()
        Xr_train = Xr[~mask_C]                         # all rare except C
        X = np.vstack([Xv, Xr_train]); y = np.r_[np.zeros(len(Xv)), np.ones(len(Xr_train))]
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31,
                                 class_weight="balanced", verbosity=-1)
        clf.fit(X, y)
        thr = np.quantile(clf.predict_proba(Xv[A])[:, 1], 1-far)
        det = clf.predict_proba(Xr[mask_C])[:, 1] >= thr
        det_oe[mask_C] = det
        b = base_r[mask_C].mean()*100
        rows.append((C, det.mean()*100, b, n))
        print(f"{C:20s} {det.mean()*100:>15.0f}% {b:>14.0f}%  {n}")
    fp = (base.predict_proba(Xv[B])[:, 1] >= thr_b).mean()*100
    print("-"*62)
    print(f"{'OVERALL (leave-out)':20s} {det_oe.mean()*100:>15.0f}% {base_r.mean()*100:>14.0f}%  {len(Xr)}")
    print(f"\n  held-out FAR (baseline) ~ {fp:.1f}%")
    print(f"  Interpretation: 'OE leave-out' = detector trained on OTHER anomaly classes,")
    print(f"  tested on the held-out class (honest). 'baseline' = trained on all rare (in-sample).")

    import csv
    with open(out / "oe_leave_class_out.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rare_class", "oe_leaveout_det_%", "baseline_det_%", "N"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.1f}", f"{r[2]:.1f}", r[3]])
        w.writerow(["OVERALL", f"{det_oe.mean()*100:.1f}", f"{base_r.mean()*100:.1f}", len(Xr)])
    print(f"\nSaved -> {out/'oe_leave_class_out.csv'}")


if __name__ == "__main__":
    main()
