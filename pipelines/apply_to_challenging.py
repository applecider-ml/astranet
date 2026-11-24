#!/usr/bin/env python3
"""Comprehensive pass over the WHOLE challenging set (18,430 spectra).

For every spectrum: true (last) label -> predicted ASTRA class -> correct? plus the
anomaly score and 1%-FAR flag. Cross-references classification correctness with anomaly
flagging, and highlights `unknown` (unlabelled) and `Ia-pec` (the class this model was
trained WITHOUT).

Outputs -> results/challenging/challenging_full.csv  (one row per spectrum)
Usage: python pipelines/apply_to_challenging.py --spec 6model
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np
import torch

from _common import (REPO, load_config, make_astra_config, device_for,
                     load_ensemble, extract, load_cache)
from astranet.model import build_hierarchy_mappings
from astranet.config import get_raw_to_group
from astranet.dataset import make_loader

CHALLENGING = None  # resolved at runtime: REPO / "data/challenging.pt"


def luna_dict(f):
    o = {"embeddings": f["emb"].astype(np.float32), "logits": f["fine_logits"].astype(np.float32),
         "coarse_logits": f["coarse_logits"].astype(np.float32), "projections": f["proj_fine"].astype(np.float32)}
    if f.get("labels") is not None:
        o["labels"] = np.asarray(f["labels"])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v7fix6")
    ap.add_argument("--far", type=float, default=0.01)
    args = ap.parse_args()
    cfg = load_config(); ac = make_astra_config(cfg); device = device_for(cfg)
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names); nco = len(sc)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    raw2grp = get_raw_to_group(cfg.get("class_set", "with_ia_pec"))
    CN = ac.class_names

    d = torch.load(REPO / "data" / "challenging.pt", map_location="cpu")
    N = len(d["class"]); raw = np.array(d["class"]); obj = np.array(d["obj_id"])
    meta = [{"obj_id": d["obj_id"][i], "class": d["class"][i],
             "phase": d["phase"][i], "z": d["redshift"][i]} for i in range(N)]
    print(f"challenging spectra: {N}  objects: {len(set(obj))}", flush=True)
    loader = make_loader({"X": d["flux"].unsqueeze(1), "meta": meta}, ac.batch_size, shuffle=False,
                         labeled=False, augment=False, num_workers=ac.num_workers,
                         pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                         class_names=ac.class_names, savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)

    members, _ = load_ensemble(args.spec, REPO / "models" / cfg.get("weights_source", "paper"), ac, nco, device)
    print(f"ensemble={len(members)} models; extracting {N} spectra ...", flush=True)
    f = extract(members, loader, device, labeled=False)

    pred = f["fine_logits"].argmax(1)
    from scipy.special import softmax
    conf = softmax(f["fine_logits"], axis=1).max(1)
    mapped = np.array([raw2grp.get(r, None) for r in raw])          # ASTRA class or None
    pred_name = np.array([CN[p] for p in pred])
    labeled = mapped != None                                         # noqa: E711
    correct = labeled & (mapped == pred_name)

    # Sentinel anomaly score
    from luna import LUNAPipeline
    import lightgbm as lgb
    tr = load_cache(cache, "train"); va = load_cache(cache, "val"); ra = load_cache(cache, "rare_unknown_combined")
    pipe = LUNAPipeline(combiner="lightgbm", target_far=args.far, fine_to_coarse=f2c)
    pipe.fit(luna_dict(tr)); M = pipe.active_methods
    def X(g): s = pipe.score(luna_dict(g)); return np.column_stack([s[m] for m in M])
    Xv, Xr, Xc = X(va), X(ra), X(f)
    LGB = dict(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.05,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               class_weight="balanced", random_state=42, verbosity=-1)
    clf = lgb.LGBMClassifier(**LGB).fit(np.vstack([Xv, Xr]), np.r_[np.zeros(len(Xv)), np.ones(len(Xr))])
    thr = np.quantile(clf.predict_proba(Xv)[:, 1], 1 - args.far)
    score = clf.predict_proba(Xc)[:, 1]; flag = score >= thr

    # ---- summary ----
    print(f"\n=== CLASSIFICATION (labeled spectra: {labeled.sum()}) ===")
    print(f"  accuracy (mapped true == pred): {correct[labeled].mean()*100:.1f}%")
    print(f"\n=== ANOMALY @ {args.far*100:.0f}% FAR (all {N}) ===")
    print(f"  flagged: {flag.sum()}/{N} = {flag.mean()*100:.1f}%")
    unk = raw == "unknown"
    print(f"  unknown flagged     : {flag[unk].sum()}/{unk.sum()} = {flag[unk].mean()*100:.1f}%")
    print(f"  labeled flagged     : {flag[labeled].sum()}/{labeled.sum()} = {flag[labeled].mean()*100:.1f}%")
    print(f"\n  correctly-classified & flagged : {(correct & flag).sum()}  (in-dist that still trips AD)")
    print(f"  misclassified & flagged        : {(labeled & ~correct & flag).sum()}/{(labeled & ~correct).sum()} "
          f"= {(labeled & ~correct & flag).mean()*100 if (labeled&~correct).any() else 0:.0f}% of misclassified")
    # Ia-pec spotlight (dropped class)
    for special in ["Ia-pec", "Ia-91T", "Ia-CSM", "Ic-BL", "Ic-SLSN"]:
        m = raw == special
        if m.any():
            print(f"  [{special:8s}] n={m.sum():>4d}  pred-> {collections.Counter(pred_name[m]).most_common(3)}  "
                  f"flagged {flag[m].mean()*100:.0f}%")

    out = REPO / "results" / "challenging"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "challenging_full.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["obj_id", "true_label", "mapped_true", "pred_class", "confidence",
                    "correct", "anomaly_score", "flagged", "phase"])
        for i in range(N):
            w.writerow([obj[i], raw[i], mapped[i] if mapped[i] else "", pred_name[i],
                        f"{conf[i]:.3f}", int(bool(correct[i])), f"{score[i]:.4f}",
                        int(bool(flag[i])), meta[i]["phase"]])
    print(f"\nSaved per-spectrum table -> {out/'challenging_full.csv'}")


if __name__ == "__main__":
    main()
