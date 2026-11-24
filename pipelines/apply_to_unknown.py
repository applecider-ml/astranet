#!/usr/bin/env python3
"""Apply the trained Sentinel to the UNKNOWN (unclassified) challenging spectra.

Counts how many unknown objects the anomaly layer flags at 1% FAR — i.e. candidate
novel/interesting transients for follow-up. Uses the consistent ensemble; the 16-score
Sentinel LightGBM is trained on train(ID) vs the 289 labelled rare (OOD), threshold at
1% FAR on validation.

Usage: python pipelines/apply_to_unknown.py --spec 6model
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import torch

from _common import (REPO, load_config, make_astra_config, device_for,
                     load_ensemble, extract, load_cache, ENSEMBLE_SPECS)
from astranet.model import build_hierarchy_mappings
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

    # ---- unknown loader (phase + z included!) ----
    d = torch.load(REPO / "data" / "challenging.pt", map_location="cpu")
    cls = np.array(d["class"])
    umask = cls == "unknown"
    idx = np.where(umask)[0]
    flux = d["flux"][idx]
    meta = [{"obj_id": d["obj_id"][i], "class": "unknown",
             "phase": d["phase"][i], "z": d["redshift"][i]} for i in idx]
    print(f"unknown spectra: {len(idx)}  unique objects: {len(set(d['obj_id'][i] for i in idx))}", flush=True)
    uloader = make_loader({"X": flux.unsqueeze(1), "meta": meta}, ac.batch_size, shuffle=False,
                          labeled=False, augment=False, num_workers=ac.num_workers,
                          pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                          class_names=ac.class_names, savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)

    # ---- extract unknown with the ensemble ----
    members, labels = load_ensemble(args.spec, REPO / "models" / cfg.get("weights_source", "paper"),
                                    ac, nco, device)
    print(f"ensemble = {len(members)} models; extracting {len(idx)} unknown spectra ...", flush=True)
    uf = extract(members, uloader, device, labeled=False)

    # ---- Sentinel: LUNA scores + LightGBM (train vs 289 rare), threshold on val ----
    from luna import LUNAPipeline
    import lightgbm as lgb
    tr = load_cache(cache, "train"); va = load_cache(cache, "val"); ra = load_cache(cache, "rare_unknown_combined")
    pipe = LUNAPipeline(combiner="lightgbm", target_far=args.far, fine_to_coarse=f2c)
    pipe.fit(luna_dict(tr)); M = pipe.active_methods
    def X(f): sc = pipe.score(luna_dict(f)); return np.column_stack([sc[m] for m in M])
    Xv, Xr, Xu = X(va), X(ra), X(uf)
    LGB = dict(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.05,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               class_weight="balanced", random_state=42, verbosity=-1)
    Xtr = np.vstack([Xv, Xr]); y = np.r_[np.zeros(len(Xv)), np.ones(len(Xr))]
    clf = lgb.LGBMClassifier(**LGB).fit(Xtr, y)
    thr = np.quantile(clf.predict_proba(Xv)[:, 1], 1 - args.far)
    su = clf.predict_proba(Xu)[:, 1]
    flag = su >= thr

    # ---- per-spectrum + per-object counts ----
    uobj = np.array([m["obj_id"] for m in meta])
    obj_flag = collections.defaultdict(bool)
    for o, fl in zip(uobj, flag):
        obj_flag[o] |= bool(fl)
    n_obj = len(obj_flag); n_obj_flag = sum(obj_flag.values())
    print(f"\n=== Anomalies in the UNKNOWN set @ {args.far*100:.0f}% FAR ===")
    print(f"  per-spectrum flagged: {flag.sum()}/{len(flag)} = {flag.mean()*100:.1f}%")
    print(f"  per-OBJECT  flagged : {n_obj_flag}/{n_obj} = {n_obj_flag/n_obj*100:.1f}%")

    # save flagged candidates
    import csv
    out = REPO / "results" / "unknown"; out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-su)
    with open(out / "unknown_anomaly_scores.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["obj_id", "anomaly_score", "flagged", "pred_class", "phase"])
        pred = uf["fine_logits"].argmax(1)
        for i in order:
            w.writerow([uobj[i], f"{su[i]:.4f}", int(flag[i]), ac.class_names[pred[i]], meta[i]["phase"]])
    print(f"\nSaved ranked candidates -> {out/'unknown_anomaly_scores.csv'}")


if __name__ == "__main__":
    main()
