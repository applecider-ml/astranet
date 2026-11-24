#!/usr/bin/env python3
"""Test-set UQ diagnostics for the paper (Tables: test-error AUROCs, selective
classification, and per-object case-study rows).

Computes, on the held-out test set of the config's ensemble:
  * six classifier-internal UQ signals (top-1 conf, entropy, margin, MI,
    predicted-class ensemble spread, model disagreement) from per-model softmax
  * the operational cross-family AD score s_AD (LightGBM fit on val+rare)
  * error-detection AUROC of each signal (errors = misclassified test spectra)
  * selective-classification sweep (drop top-X% least confident) with the
    fraction of rejected spectra whose AD score is below the 95th-pct threshold
Also dumps per-spectrum diagnostics for the RARE set (for case-study rows):
  conf, entropy, typicality, mahal_within, iforest, hec, n_families, s_ens,
  Mondrian p-value, and per-class ensemble mean/std from per-model softmax.

Usage: python pipelines/uq_test_diagnostics.py --spec v7fix6 --config configs/v7fix.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from _common import (resolve_path, REPO, load_config, make_astra_config, device_for,
                     load_ensemble, extract, load_cache)
from astranet.model import build_hierarchy_mappings
from astranet.dataset import make_loader
from sklearn.metrics import roc_auc_score


def luna_dict(f):
    o = {"embeddings": f["emb"].astype(np.float32), "logits": f["fine_logits"].astype(np.float32),
         "coarse_logits": f["coarse_logits"].astype(np.float32), "projections": f["proj_fine"].astype(np.float32)}
    if f.get("labels") is not None:
        o["labels"] = np.asarray(f["labels"])
    return o


def uq_signals(pm):
    """pm: (M, N, K) per-model softmax -> dict of 6 signals (higher = more uncertain)."""
    mean = pm.mean(0)                                   # (N, K)
    top2 = np.sort(mean, 1)[:, -2:]
    conf = top2[:, 1]
    margin = top2[:, 1] - top2[:, 0]
    ent = -(mean * np.log(mean + 1e-12)).sum(1)
    ent_per = -(pm * np.log(pm + 1e-12)).sum(2)         # (M, N)
    mi = ent - ent_per.mean(0)
    pred = mean.argmax(1)
    spread = pm[:, np.arange(pm.shape[1]), pred].std(0)
    dis = (pm.argmax(2) != pred[None, :]).mean(0)
    return {"top1_conf": -conf, "margin": -margin, "entropy": ent,
            "mi": mi, "spread": spread, "disagreement": dis}, mean, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v7fix6")
    ap.add_argument("--config", default="configs/v7fix.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg); device = device_for(cfg)
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names); nco = len(sc)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    CN = ac.class_names
    cache = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "uq_diagnostics"; out.mkdir(parents=True, exist_ok=True)

    # ---- per-model softmax on TEST + RARE (extract fresh; cache npz drops it) ----
    pm_path = out / "per_model_softmax_test_rare.npz"
    te = load_cache(cache, "test"); ra = load_cache(cache, "rare_unknown_combined")
    if pm_path.exists():
        z = np.load(pm_path); pm_te, pm_ra = z["pm_te"], z["pm_ra"]
    else:
        members, _ = load_ensemble(args.spec, REPO / "models" / cfg.get("weights_source"), ac, nco, device)
        d = torch.load(f"{ac.data_dir}/test.pt", map_location="cpu")
        loader = make_loader(d, ac.batch_size, shuffle=False, labeled=True, augment=False,
                             num_workers=0, pin_memory=ac.pin_memory,
                             compute_derivative=ac.compute_derivative, class_names=CN,
                             savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)
        fte = extract(members, loader, device, labeled=True)
        rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu")
        meta = [{"obj_id": rd["obj_id"][i], "class": rd["class"][i],
                 "phase": rd["phase"][i], "z": rd["redshift"][i]} for i in range(len(rd["class"]))]
        rloader = make_loader({"X": rd["flux"].unsqueeze(1), "meta": meta}, ac.batch_size,
                              shuffle=False, labeled=False, augment=False, num_workers=0,
                              pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                              class_names=CN, savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)
        fra = extract(members, rloader, device, labeled=False)
        pm_te, pm_ra = fte["per_model_softmax"], fra["per_model_softmax"]
        np.savez(pm_path, pm_te=pm_te, pm_ra=pm_ra)
    y = np.asarray(te["labels"])

    # ---- operational AD score on test (LightGBM fit val+rare, as deployed) ----
    from luna import LUNAPipeline
    import lightgbm as lgb
    tr = load_cache(cache, "train"); va = load_cache(cache, "val")
    pipe = LUNAPipeline(combiner="lightgbm", target_far=0.01, fine_to_coarse=f2c)
    pipe.fit(luna_dict(tr)); M = pipe.active_methods
    def X(g): s = pipe.score(luna_dict(g)); return np.column_stack([s[m] for m in M])
    Xv, Xr, Xt = X(va), X(ra), X(te)
    LGB = dict(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.05,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               class_weight="balanced", random_state=42, verbosity=-1)
    clf = lgb.LGBMClassifier(**LGB).fit(np.vstack([Xv, Xr]),
                                        np.r_[np.zeros(len(Xv)), np.ones(len(Xr))])
    sad_te = clf.predict_proba(Xt)[:, 1]
    sad_va = clf.predict_proba(Xv)[:, 1]

    # ---- Table: error-detection AUROCs ----
    sig, mean_te, pred = uq_signals(pm_te)
    err = (pred != y).astype(int)
    res = {"n_test": int(len(y)), "n_err": int(err.sum())}
    print(f"N_test={len(y)}  N_err={err.sum()}  acc={(1-err.mean())*100:.1f}%")
    print("\n--- Error-detection AUROC (test) ---")
    aurocs = {}
    for name, v in sig.items():
        aurocs[name] = float(roc_auc_score(err, v))
    aurocs["s_AD"] = float(roc_auc_score(err, sad_te))
    for k, v in sorted(aurocs.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:.3f}")
    res["error_aurocs"] = aurocs

    # ---- Table: selective classification ----
    conf = -sig["top1_conf"]
    order = np.argsort(conf)                       # least confident first
    thr95 = np.quantile(sad_va, 0.95)
    print("\n--- Selective classification ---")
    sel = []
    base = float(1 - err.mean())
    print(f"  coverage 1.00: acc={base:.3f}")
    for drop in [0.05, 0.10, 0.20]:
        k = int(drop * len(y))
        rej = order[:k]; keep = order[k:]
        acc = float(1 - err[keep].mean())
        adlow = float((sad_te[rej] < thr95).mean() * 100)
        sel.append({"coverage": 1 - drop, "acc": acc, "gain_pp": (acc - base) * 100, "ad_low_pct": adlow})
        print(f"  coverage {1-drop:.2f}: acc={acc:.3f} (+{(acc-base)*100:.1f}pp)  AD-low={adlow:.1f}%")
    res["selective"] = sel; res["baseline_acc"] = base

    # ---- per-object case-study diagnostics (rare set) ----
    sr = pipe.score(luna_dict(ra))
    fam = pipe.family_analysis(luna_dict(va), luna_dict(ra))
    nfam = np.stack([fam[f]["flags"] for f in fam]).sum(0)
    sad_ra = clf.predict_proba(Xr)[:, 1]
    # Mondrian p-value vs val calibration (per predicted class)
    pred_ra = ra["fine_logits"].argmax(1)
    pred_va = va["fine_logits"].argmax(1)
    pM = np.zeros(len(sad_ra))
    for c in range(len(CN)):
        mv = pred_va == c; mr = pred_ra == c
        if mr.any():
            calv = np.sort(sad_va[mv]) if mv.sum() > 5 else np.sort(sad_va)
            pM[mr] = 1.0 - np.searchsorted(calv, sad_ra[mr], side="left") / (len(calv) + 1)
    mean_ra = pm_ra.mean(0); std_ra = pm_ra.std(0)
    conf_ra = mean_ra.max(1); ent_ra = -(mean_ra * np.log(mean_ra + 1e-12)).sum(1)
    rows = []
    for i, m in enumerate(ra["meta"]):
        top3 = np.argsort(-mean_ra[i])[:3]
        rows.append({"obj_id": m["obj_id"], "true": m["class"], "pred": CN[int(pred_ra[i])],
                     "conf": round(float(conf_ra[i]), 3), "entropy": round(float(ent_ra[i]), 3),
                     "typicality": round(float(sr["typicality"][i]), 3),
                     "mahal_within": round(float(sr["mahal_within"][i]), 1),
                     "iforest": round(float(sr["iforest"][i]), 3),
                     "hec": round(float(sr["hec"][i]), 3),
                     "n_fam": int(nfam[i]), "s_ens": round(float(sad_ra[i]), 3),
                     "p_M": round(float(pM[i]), 4),
                     "top3": [(CN[int(k)], round(float(mean_ra[i, k]) * 100, 1),
                               round(float(std_ra[i, k]) * 100, 1)) for k in top3]})
    (out / "case_study_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "uq_test_summary.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
