#!/usr/bin/env python3
"""PRODUCTION analysis of the challenging set (18,430 spectra / 9,918 objects).

Rigorous, paper-grade:
  * HONEST 1% FAR threshold — LightGBM trained on val_cal + 289 rare; threshold set on a
    DISJOINT val_holdout (no in-sample leak).
  * Robust multi-epoch anomaly criterion — an OBJECT is an anomaly only if a robust
    fraction (>=50%) of its >=2 spectra are flagged (single-epoch flags are unreliable).
  * Classification correctness with subtype->parent mapping (Ia-91T -> Ia, etc.).
  * KEY: classification errors that the AD independently flags as anomalies
    (embedding-AD catching the classifier's mistakes — the complementarity result).
  * Caches challenging embeddings so re-analysis is instant.

Outputs -> results/challenging/  (production_summary.txt, robust_anomaly_candidates.csv,
           misclassified_anomalies.csv)
Usage: python pipelines/production_challenging.py --spec 6model
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from _common import (REPO, load_config, make_astra_config, device_for,
                     load_ensemble, extract, load_cache)
from astranet.model import build_hierarchy_mappings
from astranet.config import get_raw_to_group
from astranet.dataset import make_loader

CHALLENGING = None  # resolved at runtime: REPO / "data/challenging.pt"
# SN subtype string variants missing from RAW_TO_GROUP (subtype -> ASTRA parent)
EXTRA_MAP = {"Ic-pec": "Ibc", "Ib-norm": "Ibc", "IIb-pec": "II",
             "Ia-00cx": "Ia", "Ia-18byg": "Ia"}
ROBUST_FRAC = 0.5   # >=50% of an object's epochs flagged
MIN_EPOCHS = 2


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
    CN = ac.class_names
    r2g = dict(get_raw_to_group(cfg.get("class_set", "with_ia_pec"))); r2g.update(EXTRA_MAP)
    out = REPO / "results" / "challenging"; out.mkdir(parents=True, exist_ok=True)
    ccache = REPO / "cache" / "challenging"

    d = torch.load(REPO / "data" / "challenging.pt", map_location="cpu")
    N = len(d["class"]); raw = np.array(d["class"]); obj = np.array(d["obj_id"])
    phase = np.array([float(x) if x is not None else np.nan for x in d["phase"]])

    # ---- extract (cached) ----
    npz = ccache / f"{args.spec}.npz"
    if npz.exists():
        z = np.load(npz)
        emb, fl, cl, pf = z["emb"], z["fine_logits"], z["coarse_logits"], z["proj_fine"]
        print(f"loaded cached challenging features {emb.shape}", flush=True)
    else:
        meta = [{"obj_id": d["obj_id"][i], "class": d["class"][i],
                 "phase": d["phase"][i], "z": d["redshift"][i]} for i in range(N)]
        loader = make_loader({"X": d["flux"].unsqueeze(1), "meta": meta}, ac.batch_size, shuffle=False,
                             labeled=False, augment=False, num_workers=ac.num_workers,
                             pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                             class_names=CN, savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)
        members, _ = load_ensemble(args.spec, REPO / "models" / cfg.get("weights_source", "paper"), ac, nco, device)
        print(f"extracting {N} challenging spectra with {len(members)} models ...", flush=True)
        f = extract(members, loader, device, labeled=False)
        emb, fl, cl, pf = f["emb"], f["fine_logits"], f["coarse_logits"], f["proj_fine"]
        ccache.mkdir(parents=True, exist_ok=True)
        np.savez(npz, emb=emb, fine_logits=fl, coarse_logits=cl, proj_fine=pf)
        print(f"cached -> {npz}", flush=True)

    fchal = {"emb": emb, "fine_logits": fl, "coarse_logits": cl, "proj_fine": pf}
    pred = fl.argmax(1); conf = softmax(fl, 1).max(1)
    mapped = np.array([r2g.get(r) for r in raw], dtype=object)
    pred_name = np.array([CN[p] for p in pred])
    labeled = np.array([m is not None for m in mapped])
    correct = np.array([labeled[i] and mapped[i] == pred_name[i] for i in range(N)])

    # ---- Sentinel with HONEST held-out threshold ----
    from luna import LUNAPipeline
    import lightgbm as lgb
    tr = load_cache(cache, "train"); va = load_cache(cache, "val"); ra = load_cache(cache, "rare_unknown_combined")
    pipe = LUNAPipeline(combiner="lightgbm", target_far=args.far, fine_to_coarse=f2c)
    pipe.fit(luna_dict(tr)); M = pipe.active_methods
    def X(g): s = pipe.score(luna_dict(g)); return np.column_stack([s[m] for m in M])
    Xv, Xr, Xc = X(va), X(ra), X(fchal)
    nval = len(Xv); rng = np.random.default_rng(42); vi = rng.permutation(nval)
    vcal, vhold = vi[:nval // 2], vi[nval // 2:]
    LGB = dict(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.05,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               class_weight="balanced", random_state=42, verbosity=-1)
    clf = lgb.LGBMClassifier(**LGB).fit(np.vstack([Xv[vcal], Xr]), np.r_[np.zeros(len(vcal)), np.ones(len(Xr))])
    thr = np.quantile(clf.predict_proba(Xv[vhold])[:, 1], 1 - args.far)   # HONEST threshold
    val_fp = (clf.predict_proba(Xv[vhold])[:, 1] >= thr).mean() * 100
    score = clf.predict_proba(Xc)[:, 1]; flag = score >= thr

    # ---- per-object robust multi-epoch ----
    by_obj = collections.defaultdict(list)
    for i in range(N):
        by_obj[obj[i]].append(i)
    robust = {}
    for o, idxs in by_obj.items():
        fs = [flag[i] for i in idxs]
        n = len(fs); fr = np.mean(fs)
        robust[o] = (n >= MIN_EPOCHS and fr >= ROBUST_FRAC)

    unk_objs = [o for o in by_obj if raw[by_obj[o][0]] == "unknown"]
    unk_multi = [o for o in unk_objs if len(by_obj[o]) >= MIN_EPOCHS]
    unk_robust = [o for o in unk_multi if robust[o]]

    lines = []
    def P(s): print(s, flush=True); lines.append(s)
    P("=" * 64); P("PRODUCTION CHALLENGING-SET ANALYSIS"); P("=" * 64)
    P(f"spectra {N} | objects {len(by_obj)} | honest threshold (val_holdout FP={val_fp:.1f}%)")
    P(f"\n--- CLASSIFICATION (labeled spectra {labeled.sum()}) ---")
    P(f"  accuracy (subtype->parent): {correct[labeled].mean()*100:.1f}%")
    P(f"\n--- ANOMALIES (do we find them?) ---")
    P(f"  per-spectrum flagged: {flag.sum()}/{N} = {flag.mean()*100:.1f}%")
    P(f"  UNKNOWN objects: {len(unk_objs)}  (multi-epoch {len(unk_multi)}, single {len(unk_objs)-len(unk_multi)})")
    P(f"  ROBUST unknown anomalies (>= {int(ROBUST_FRAC*100)}% of >= {MIN_EPOCHS} epochs): "
      f"{len(unk_robust)} objects")

    # ---- KEY: classification errors that are anomalies ----
    lab_mask = labeled & (raw != "unknown")
    mis = lab_mask & ~correct
    mis_objs = set(obj[mis])
    mis_anom = [o for o in mis_objs if robust.get(o, False)]
    P(f"\n--- CLASSIFICATION ERRORS CAUGHT BY THE AD ---")
    P(f"  misclassified labeled objects        : {len(mis_objs)}")
    P(f"  ...of which robustly flagged anomalous: {len(mis_anom)} "
      f"({len(mis_anom)/max(len(mis_objs),1)*100:.0f}%)")
    # what are they? (raw class of the misclassified-anomalous)
    cnt = collections.Counter(raw[by_obj[o][0]] for o in mis_anom)
    P(f"  their true (sub)types: {dict(cnt.most_common(10))}")

    # ---- save candidate lists ----
    with open(out / "robust_anomaly_candidates.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["obj_id", "true_label", "n_epochs", "frac_flagged", "max_score", "pred_class"])
        cand = sorted([o for o in by_obj if robust.get(o)],
                      key=lambda o: -max(score[i] for i in by_obj[o]))
        for o in cand:
            idxs = by_obj[o]
            w.writerow([o, raw[idxs[0]], len(idxs),
                        f"{np.mean([flag[i] for i in idxs]):.2f}",
                        f"{max(score[i] for i in idxs):.4f}", pred_name[idxs[np.argmax([score[i] for i in idxs])]]])
    with open(out / "misclassified_anomalies.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["obj_id", "true_subtype", "mapped_true", "pred_class", "max_score", "n_epochs"])
        for o in sorted(mis_anom, key=lambda o: -max(score[i] for i in by_obj[o])):
            idxs = by_obj[o]; i0 = idxs[0]
            w.writerow([o, raw[i0], mapped[i0], pred_name[i0],
                        f"{max(score[i] for i in idxs):.4f}", len(idxs)])
    (out / "production_summary.txt").write_text("\n".join(lines))
    P(f"\nSaved: robust_anomaly_candidates.csv ({len(unk_robust)}+ objects), misclassified_anomalies.csv, production_summary.txt")


if __name__ == "__main__":
    main()
