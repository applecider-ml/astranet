#!/usr/bin/env python3
"""Categorize the challenging-set anomaly-detection results into 3 CSVs.

Per-object rows (aggregated over an object's spectra) split into:
  1. variable_stars.csv     — stellar-variability / CV labels (non-SN)
  2. labeled_transients.csv — labeled SN/transient objects (the 'already-labeled,
                              ambiguous → challenging' ones)
  3. unlabeled.csv          — 'unknown' (no catalog label): the discovery pool

Each row: raw_class, n_spectra, predicted class (majority vote), confidence,
anomaly verdict (robust multi-epoch @1% FAR) + score. Mirrors the style of
ASTRAnet_final/notebooks/apply_to_challenging.py but categorized and per-object.

The AD threshold is HONEST (LightGBM on val_cal + rare, threshold on a disjoint
val_holdout) — identical protocol to production_challenging.py. Uses whatever
ensemble/weights the config points at, so re-running with --config configs/v7new.yaml
(after rebuilding cache/challenging on the v7new weights) refreshes every number.

Usage: python pipelines/categorize_challenging.py [--config configs/v7new.yaml] [--spec 6model]
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
EXTRA_MAP = {"Ic-pec": "Ibc", "Ib-norm": "Ibc", "IIb-pec": "II",
             "Ia-00cx": "Ia", "Ia-18byg": "Ia"}
ROBUST_FRAC = 0.5
MIN_EPOCHS = 2

# Non-SN stellar-variability + cataclysmic-variable labels -> "variable stars".
VARSTAR = {"Stellar variable", "S Doradus", "periodic", "flaring", "W Ursae Maj",
           "eclipsing", "Eruptive", "Cataclysmic", "Novae", "Compact Binary",
           "AM CVn", "U Gem", "Nova-like", "Luminous Red Nova"}
# Data-quality tag, not an astrophysical class — excluded from all categories.
DROP_RAW = {"duplicate"}


def luna_dict(f):
    o = {"embeddings": f["emb"].astype(np.float32), "logits": f["fine_logits"].astype(np.float32),
         "coarse_logits": f["coarse_logits"].astype(np.float32), "projections": f["proj_fine"].astype(np.float32)}
    if f.get("labels") is not None:
        o["labels"] = np.asarray(f["labels"])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--far", type=float, default=0.01)
    args = ap.parse_args()
    cfg = load_config(args.config)
    spec = args.spec or cfg.get("ensemble", "6model")
    ac = make_astra_config(cfg); device = device_for(cfg)
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names); nco = len(sc)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    CN = ac.class_names
    r2g = dict(get_raw_to_group(cfg.get("class_set", "with_ia_pec"))); r2g.update(EXTRA_MAP)
    cache = REPO / "cache" / spec
    ccache = REPO / "cache" / "challenging"
    # namespaced by spec so paper-weights and v7new runs never overwrite each other
    out = REPO / "results" / "challenging" / f"categorized_{spec}"; out.mkdir(parents=True, exist_ok=True)

    d = torch.load(REPO / "data" / "challenging.pt", map_location="cpu")
    N = len(d["class"]); raw = np.array(d["class"]); obj = np.array(d["obj_id"])
    phase = np.array([float(x) if x is not None else np.nan for x in d["phase"]])

    # ---- features (cached; extracted with the config's ensemble) ----
    npz = ccache / f"{spec}.npz"
    if npz.exists():
        z = np.load(npz); emb, fl, cl, pf = z["emb"], z["fine_logits"], z["coarse_logits"], z["proj_fine"]
        print(f"loaded cached challenging features {emb.shape} ({spec})", flush=True)
    else:
        meta = [{"obj_id": d["obj_id"][i], "class": d["class"][i],
                 "phase": d["phase"][i], "z": d["redshift"][i]} for i in range(N)]
        loader = make_loader({"X": d["flux"].unsqueeze(1), "meta": meta}, ac.batch_size, shuffle=False,
                             labeled=False, augment=False, num_workers=ac.num_workers,
                             pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                             class_names=CN, savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)
        members, _ = load_ensemble(spec, REPO / "models" / cfg.get("weights_source", "paper"), ac, nco, device)
        print(f"extracting {N} challenging spectra with {len(members)} models ({spec}) ...", flush=True)
        f = extract(members, loader, device, labeled=False)
        emb, fl, cl, pf = f["emb"], f["fine_logits"], f["coarse_logits"], f["proj_fine"]
        ccache.mkdir(parents=True, exist_ok=True)
        np.savez(npz, emb=emb, fine_logits=fl, coarse_logits=cl, proj_fine=pf)
        print(f"cached -> {npz}", flush=True)

    fchal = {"emb": emb, "fine_logits": fl, "coarse_logits": cl, "proj_fine": pf}
    pred = fl.argmax(1); conf = softmax(fl, 1).max(1)
    pred_name = np.array([CN[p] for p in pred])

    # ---- Sentinel with HONEST held-out threshold (same as production) ----
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
    thr = np.quantile(clf.predict_proba(Xv[vhold])[:, 1], 1 - args.far)
    val_fp = (clf.predict_proba(Xv[vhold])[:, 1] >= thr).mean() * 100
    score = clf.predict_proba(Xc)[:, 1]; flag = score >= thr
    print(f"honest threshold set on val_holdout (FP={val_fp:.1f}%)", flush=True)

    # ---- APS conformal, calibrated on val; applied at OBJECT level ----
    # A challenging object may have several spectra; average their softmax (each
    # spectrum is evidence about the same object) into ONE probability vector and
    # build a single conformal set. This keeps `uncertainty` a property of the
    # object, not of how many spectra it happens to have.
    import torch.nn.functional as F
    from astranet.conformal import aps_scores_batch, aps_scores_allclasses
    alpha = float(cfg.get("alpha", 0.05))
    vlog = torch.from_numpy(va["fine_logits"].astype(np.float32))
    vlab = torch.from_numpy(np.asarray(va["labels"])).long()
    cal_scores = aps_scores_batch(F.softmax(vlog, 1), vlab)
    n_cal = len(cal_scores)
    spec_probs = F.softmax(torch.from_numpy(fl.astype(np.float32)), 1).numpy()  # (N, C)

    # ---- aggregate per object ----
    by = collections.defaultdict(list)
    for i in range(N):
        by[obj[i]].append(i)

    # object-mean probabilities -> batched APS conformal
    objs_order = list(by.keys())
    obj_prob = np.stack([spec_probs[by[o]].mean(0) for o in objs_order])        # (O, C)
    op = torch.from_numpy(obj_prob.astype(np.float32))
    aps_all = aps_scores_allclasses(op)                                         # (O, C)
    pvalues = (cal_scores.view(1, 1, -1) >= aps_all.unsqueeze(2)).float().sum(2) / (n_cal + 1)
    oset = (pvalues > alpha)
    oset[torch.arange(len(op)), op.argmax(1)] = True                           # non-empty guarantee
    oset = oset.numpy()
    CONF = {}                                                                  # obj_id -> (pred_set, size, uncertainty)
    for k, o in enumerate(objs_order):
        cls_in = [CN[c] for c in range(len(CN)) if oset[k, c]]
        ss = len(cls_in)
        CONF[o] = ("|".join(cls_in), ss,
                   "low" if ss == 1 else ("medium" if ss == 2 else "high"))
    print(f"APS conformal: alpha={alpha} (coverage {100*(1-alpha):.0f}%), n_cal={n_cal}, object-level", flush=True)

    def row(o):
        idxs = by[o]
        n = len(idxs)
        rc = raw[idxs[0]]
        # majority-vote predicted class over the object's spectra
        pv = collections.Counter(pred_name[i] for i in idxs).most_common(1)[0][0]
        cf = float(np.mean([conf[i] for i in idxs if pred_name[i] == pv]))
        nfl = int(sum(flag[i] for i in idxs)); frac = nfl / n
        robust = (n >= MIN_EPOCHS and frac >= ROBUST_FRAC)
        mapped = r2g.get(rc)
        correct = (mapped is not None and mapped == pv)
        pset, ss, unc = CONF[o]                       # object-level conformal set
        return {
            "obj_id": o, "raw_class": rc, "n_spectra": n,
            "pred_class": pv, "confidence": round(cf, 4),
            "mapped_true": mapped if mapped is not None else "",
            "correct": int(bool(correct)) if mapped is not None else "",
            "pred_set": pset, "set_size": ss, "uncertainty": unc,
            "n_flagged": nfl, "frac_flagged": round(frac, 2),
            "is_anomaly": int(bool(robust)),
            "anomaly_single_epoch": int(n == 1 and nfl == 1),
            "max_ad_score": round(float(max(score[i] for i in idxs)), 4),
        }

    COLS = ["obj_id", "raw_class", "n_spectra", "pred_class", "confidence",
            "mapped_true", "correct", "pred_set", "set_size", "uncertainty",
            "n_flagged", "frac_flagged", "is_anomaly", "anomaly_single_epoch",
            "max_ad_score"]

    cats = {"variable_stars": [], "labeled_transients": [], "unlabeled": []}
    for o in by:
        rc = raw[by[o][0]]
        if rc in DROP_RAW:
            continue
        if rc == "unknown":
            cats["unlabeled"].append(o)
        elif rc in VARSTAR:
            cats["variable_stars"].append(o)
        else:
            cats["labeled_transients"].append(o)

    print("\n=== CATEGORIZED CHALLENGING (per object) ===")
    for name, objs in cats.items():
        rows = sorted((row(o) for o in objs), key=lambda r: -r["max_ad_score"])
        with open(out / f"{name}.csv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        n = len(rows)
        anom = sum(r["is_anomaly"] for r in rows)
        multi = sum(r["n_spectra"] >= MIN_EPOCHS for r in rows)
        if name != "unlabeled":
            lab = [r for r in rows if r["correct"] != ""]
            acc = 100 * np.mean([r["correct"] for r in lab]) if lab else float("nan")
            print(f"  {name:18s}: {n:5d} objects | robust anomaly {anom:4d} "
                  f"({100*anom/max(multi,1):.0f}% of multi-epoch) | class-acc {acc:.0f}%")
        else:
            print(f"  {name:18s}: {n:5d} objects | robust anomaly {anom:4d} "
                  f"({100*anom/max(multi,1):.0f}% of multi-epoch)")
    print(f"\nSaved 3 CSVs -> {out}")


if __name__ == "__main__":
    main()
