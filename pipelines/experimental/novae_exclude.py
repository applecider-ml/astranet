#!/usr/bin/env python3
"""Does excluding novae from the Sentinel LightGBM training give better weights?

Novae are CV-like (ambiguous). Training the combiner to flag them as OOD may pull
its weights toward flagging CV-like objects, hurting detection of genuine anomalies.
We compare, honestly (5-fold OOF so the test anomalies are never trained on):

  A) combiner trained on  val(ID) vs ALL 289 rare
  B) combiner trained on  val(ID) vs 214 rare (novae + classical-nova excluded)

Both evaluated on the SAME 214 genuine out-of-taxonomy anomalies (OOF), threshold @1%FAR.

Usage: python pipelines/novae_exclude.py --spec 6model
"""
from __future__ import annotations

import argparse
import numpy as np

from _common import REPO, load_config, make_astra_config, load_cache
from astranet.model import build_hierarchy_mappings

NOVAE = {"Novae", "Classical Nova"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="6model")
    ap.add_argument("--far", type=float, default=0.01)
    args = ap.parse_args()
    cfg = load_config(); ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    tr = load_cache(cache, "train"); va = load_cache(cache, "val")
    ra = load_cache(cache, "rare_unknown_combined")
    rare_cls = np.array([m.get("class", "?") for m in ra["meta"]])

    from luna import LUNAPipeline
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    def L(d):
        o = {"embeddings": d["emb"].astype(np.float32), "logits": d["fine_logits"].astype(np.float32),
             "coarse_logits": d["coarse_logits"].astype(np.float32), "projections": d["proj_fine"].astype(np.float32)}
        if d.get("labels") is not None: o["labels"] = np.asarray(d["labels"])
        return o
    pipe = LUNAPipeline(combiner="lightgbm", target_far=args.far, fine_to_coarse=f2c)
    pipe.fit(L(tr)); M = pipe.active_methods
    Xv = np.column_stack([pipe.score(L(va))[m] for m in M])
    rs = pipe.score(L(ra)); Xr = np.column_stack([rs[m] for m in M])
    far = args.far
    LGB = dict(n_estimators=200, num_leaves=31, max_depth=-1, learning_rate=0.05,
               class_weight="balanced", verbosity=-1)

    keep214 = ~np.isin(rare_cls, list(NOVAE))   # genuine anomalies (test set, both configs)

    def oof_detect(train_rare_mask):
        """5-fold OOF: train on val + rare[train_rare_mask]; honest val threshold;
        OOF scores for the 214 test anomalies; detection @ far."""
        nval = len(Xv)
        idx_r = np.where(train_rare_mask)[0]
        X = np.vstack([Xv, Xr[idx_r]]); y = np.r_[np.zeros(nval), np.ones(len(idx_r))]
        voof = np.zeros(nval); roof_full = np.full(len(Xr), np.nan)
        for tri, tei in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            c = lgb.LGBMClassifier(**LGB).fit(X[tri], y[tri])
            p = c.predict_proba(X[tei])[:, 1]
            vm = tei < nval; voof[tei[vm]] = p[vm]
            rloc = idx_r[tei[~vm] - nval]; roof_full[rloc] = p[~vm]
        # rare not in training set: score with full-fit (never trained on them)
        cfull = lgb.LGBMClassifier(**LGB).fit(X, y)
        missing = np.isnan(roof_full)
        if missing.any():
            roof_full[missing] = cfull.predict_proba(Xr[missing])[:, 1]
        thr = np.quantile(voof, 1 - far)
        det = roof_full[keep214] >= thr
        return det, (voof >= thr).mean() * 100

    detA, fpA = oof_detect(np.ones(len(Xr), bool))   # train on all 289
    detB, fpB = oof_detect(keep214)                  # train on 214 (no novae)
    print(f"Eval set = {keep214.sum()} genuine anomalies (novae excluded), @ {far*100:.0f}% FAR\n")
    print(f"  A) combiner trained on val vs ALL 289 : {detA.mean()*100:5.1f}%  (val FP {fpA:.1f}%)")
    print(f"  B) combiner trained on val vs 214 (no novae): {detB.mean()*100:5.1f}%  (val FP {fpB:.1f}%)")
    print(f"\n  per-class detection (A -> B):")
    import collections
    for c in sorted(set(rare_cls[keep214])):
        mk = rare_cls[keep214] == c
        print(f"    {c:20s} {detA[mk].mean()*100:4.0f}% -> {detB[mk].mean()*100:4.0f}%  (n={mk.sum()})")


if __name__ == "__main__":
    main()
