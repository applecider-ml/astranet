#!/usr/bin/env python3
"""FINAL: k-fold Outlier-Exposure -> honest full-289 detection, per class & per object.

Stratified K-fold over the 289 anomalies. For each fold:
  * OE fine-tune the encoder (entropy + embedding-margin) on the EXPOSURE folds,
  * extract embeddings, fit the 16 LUNA scores + Sentinel LightGBM on val + exposure,
  * set the 1% FAR threshold on a held-out val split,
  * detect the held-out fold (never seen in fine-tune OR combiner).
Aggregate held-out detections over all folds -> honest detection on ALL 289,
reported per-spectrum AND per-object, with a per-class breakdown.
Baseline (no OE) uses the same k-fold combiner protocol on the base encoder.

Stays within the paper's framework: same 16-score Sentinel LightGBM; OE only the encoder.

Usage: python pipelines/oe_kfold.py --folds 3 --epochs 15 --lam 1.0 --lam-emb 1.0
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import lightgbm as lgb

from _common import (resolve_path, REPO, load_config, make_astra_config, device_for, load_model, extract,
                     tiered_report)
from astranet.sentinel import (oe_finetune as sentinel_oe_finetune, detect_fold as sentinel_detect_fold,
                               detect_fold_heads, detector_heads)
from astranet.model import build_hierarchy_mappings
from astranet.dataset import build_loaders, make_loader

LGB = dict(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.05,
           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
           class_weight="balanced", random_state=42, verbosity=-1)
FAR = 0.01


def luna_dict(f):
    o = {"embeddings": f["emb"].astype(np.float32), "logits": f["fine_logits"].astype(np.float32),
         "coarse_logits": f["coarse_logits"].astype(np.float32), "projections": f["proj_fine"].astype(np.float32)}
    if f.get("labels") is not None:
        o["labels"] = np.asarray(f["labels"])
    return o


def feats(pipe, f, M):
    sc = pipe.score(luna_dict(f))            # score once
    return np.column_stack([sc[m] for m in M])


def detect_fold(train_f, val_f, rare_f, f2c, expo_idx, test_idx):
    """ASTRANet-Sentinel fold (see astranet.sentinel.detect_fold)."""
    return sentinel_detect_fold(train_f, val_f, rare_f, f2c, expo_idx, test_idx,
                                target_far=FAR)


def oe_finetune(ckpt, ac, nco, device, expo_loader, train_aug, epochs, lam, lam_emb, mmult=2.0, lr=1e-4):
    """Load the checkpoint and OE fine-tune it (see astranet.sentinel.oe_finetune)."""
    model = load_model(str(ckpt), ac, nco, device)
    return sentinel_oe_finetune(model, expo_loader, train_aug, device, epochs=epochs,
                                lam=lam, lam_emb=lam_emb, margin_mult=mmult, lr=lr,
                                num_classes=ac.num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lam-emb", type=float, default=1.0)
    ap.add_argument("--ckpt", default="models/trained_v7fix/seed42/checkpoint_best.pt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="", help="suffix for the results json")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers override (0 avoids thread-limit issues)")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--split-by", choices=["object", "spectrum", "tier"], default="object",
                    help="object: StratifiedGroupKFold on obj_id (no object in both exposure and "
                         "test); spectrum: legacy per-spectrum StratifiedKFold (object leakage); "
                         "tier: two folds — expose Tier 2 (rare members of trained families) and "
                         "test Tier 1 (genuinely novel phenomena), then the reverse: the detector "
                         "never sees any class of the tested tier (discovery protocol)")
    args = ap.parse_args()
    ckpt = REPO / args.ckpt
    cfg = load_config(args.config); ac = make_astra_config(cfg); device = device_for(cfg)
    if args.workers is not None:
        ac.num_workers = args.workers
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names); nco = len(sc)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)

    ac.augment_train = False; ac.use_balanced_sampling = False
    _, train_loader, val_loader, _, _ = build_loaders(ac)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu")
    cls = np.array(rd["class"]); flux = rd["flux"]
    objid = np.array(rd["obj_id"])
    def rmeta(idx): return [{"obj_id": rd["obj_id"][i], "class": rd["class"][i],
                             "phase": rd["phase"][i], "z": rd["redshift"][i]} for i in idx]
    rl = make_loader({"X": flux.unsqueeze(1), "meta": rmeta(range(len(cls)))}, ac.batch_size,
                     shuffle=False, labeled=False, augment=False, num_workers=ac.num_workers,
                     pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                     class_names=ac.class_names, savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)

    if args.split_by == "object":
        skf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=0)
        folds = list(skf.split(cls, cls, groups=objid))
        for k, (e, t) in enumerate(folds):
            assert not (set(objid[e]) & set(objid[t])), f"object leakage in fold {k}"
        print(f"object-grouped {args.folds}-fold: "
              f"{[len(set(objid[t])) for _, t in folds]} held-out objects per fold", flush=True)
    elif args.split_by == "tier":
        from _common import RARE_TIER1, RARE_TIER2
        t1 = np.where(np.isin(cls, list(RARE_TIER1)))[0]; t2 = np.where(np.isin(cls, list(RARE_TIER2)))[0]
        assert len(t1) + len(t2) == len(cls), "every rare class must belong to a tier"
        folds = [(t2, t1), (t1, t2)]      # (exposure, test): expose Tier 2 -> test Tier 1, and reverse
        args.folds = 2
        print(f"leave-tier-out: expose Tier2 ({len(set(objid[t2]))} obj) -> test Tier1 ({len(set(objid[t1]))} obj); "
              f"and the reverse", flush=True)
    else:
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=0)
        folds = list(skf.split(cls, cls))
    HEADS = list(detector_heads())
    # per detector head: flags / scores / per-fold thresholds, for the base and the OE encoder
    H = {enc: {h: {"det": np.zeros(len(cls), bool), "score": np.zeros(len(cls)), "thr": []} for h in HEADS}
         for enc in ("base", "oe")}

    # baseline encoder features (extract once)
    base = load_model(str(ckpt), ac, nco, device)
    btr = extract([base], train_loader, device, labeled=True)
    bva = extract([base], val_loader, device, labeled=True)
    bra = extract([base], rl, device, labeled=False)

    for fold, (expo_idx, test_idx) in enumerate(folds):
        print(f"\n=== fold {fold+1}/{args.folds}: expose {len(expo_idx)}, test {len(test_idx)} ===", flush=True)
        for h, (d, sc, t) in detect_fold_heads(btr, bva, bra, f2c, expo_idx, test_idx, target_far=FAR).items():
            H["base"][h]["det"][test_idx] = d; H["base"][h]["score"][test_idx] = sc; H["base"][h]["thr"].append(t)
        print("  baseline fold det: " + "  ".join(f"{h}={H['base'][h]['det'][test_idx].mean()*100:.0f}%" for h in HEADS), flush=True)
        if args.baseline_only:
            continue
        # OE fine-tune on this fold's exposure anomalies
        ac.augment_train = True; ac.use_balanced_sampling = True
        _, train_aug, _, _, _ = build_loaders(ac)
        ac.augment_train = False; ac.use_balanced_sampling = False
        expo_loader = make_loader({"X": flux[expo_idx].unsqueeze(1), "meta": rmeta(expo_idx)},
                                  ac.batch_size, shuffle=True, labeled=False, augment=False,
                                  num_workers=ac.num_workers, pin_memory=ac.pin_memory,
                                  compute_derivative=ac.compute_derivative, class_names=ac.class_names,
                                  savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)
        m = oe_finetune(ckpt, ac, nco, device, expo_loader, train_aug,
                        args.epochs, args.lam, args.lam_emb)
        otr = extract([m], train_loader, device, labeled=True)
        ova = extract([m], val_loader, device, labeled=True)
        ora = extract([m], rl, device, labeled=False)
        for h, (d, sc, t) in detect_fold_heads(otr, ova, ora, f2c, expo_idx, test_idx, target_far=FAR).items():
            H["oe"][h]["det"][test_idx] = d; H["oe"][h]["score"][test_idx] = sc; H["oe"][h]["thr"].append(t)
        print("  OE fold det      : " + "  ".join(f"{h}={H['oe'][h]['det'][test_idx].mean()*100:.0f}%" for h in HEADS), flush=True)
        del m; torch.cuda.empty_cache()

    def per_object(det):
        o = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(cls)):
            o[objid[i]][1] = cls[i]
            if det[i]: o[objid[i]][0] = True
        return o

    def report(name, det):
        ob = per_object(det)
        ps = det.mean() * 100
        od = sum(1 for v in ob.values() if v[0]); on = len(ob)
        print(f"\n[{name}] per-spectrum {det.sum()}/{len(det)}={ps:.1f}%   "
              f"per-object {od}/{on}={od/on*100:.1f}%")
        byc = collections.defaultdict(lambda: [0, 0])
        for d, c in ob.values():
            byc[c][1] += 1; byc[c][0] += d
        for c, (d, n) in sorted(byc.items(), key=lambda x: x[1][0] / max(x[1][1], 1)):
            print(f"    {c:20s} {d:>2d}/{n:<2d} = {d/n*100:3.0f}%")
        tiers = tiered_report((v[0], v[1]) for v in ob.values())
        return {"per_spectrum_%": ps, "per_object_%": od / on * 100, "tiers": tiers,
                "per_class": {c: [v[0], v[1]] for c, v in byc.items()}}

    # the Sentinel combiner (16 scores + LightGBM) keeps the legacy keys; every head is stored too
    det_base, det_oe = H["base"]["scores16_lgbm"]["det"], H["oe"]["scores16_lgbm"]["det"]
    res = {"split_by": args.split_by, "folds": args.folds, "heads": HEADS,
           "baseline": report("BASELINE (no OE)", det_base)}
    if not args.baseline_only:
        res["oe"] = report("OE k-fold", det_oe)
    res["per_head"] = {enc: {h: {"per_spectrum_%": 100 * float(H[enc][h]["det"].mean()),
                                 "per_object_%": 100 * sum(v[0] for v in per_object(H[enc][h]["det"]).values())
                                                 / len(set(objid)),
                                 "tiers": tiered_report((v[0], v[1]) for v in per_object(H[enc][h]["det"]).values())}
                             for h in HEADS} for enc in ("base", "oe")}
    print("\n[per head] per-object detection @1% FAR (base -> OE):")
    for h in HEADS:
        print(f"    {h:20s} {res['per_head']['base'][h]['per_object_%']:5.1f}% -> {res['per_head']['oe'][h]['per_object_%']:5.1f}%")
    out = REPO / "results" / "oe"; out.mkdir(parents=True, exist_ok=True)
    arrays = dict(score_base=H["base"]["scores16_lgbm"]["score"], score_oe=H["oe"]["scores16_lgbm"]["score"],
                  det_base=det_base, det_oe=det_oe,
                  thr_base=np.array(H["base"]["scores16_lgbm"]["thr"]), thr_oe=np.array(H["oe"]["scores16_lgbm"]["thr"]),
                  cls=cls, objid=objid)
    for enc in ("base", "oe"):
        for h in HEADS:
            arrays[f"score_{enc}_{h}"] = H[enc][h]["score"]; arrays[f"det_{enc}_{h}"] = H[enc][h]["det"]
            arrays[f"thr_{enc}_{h}"] = np.array(H[enc][h]["thr"])
    np.savez(out / f"oe_kfold_scores{('_'+args.tag) if args.tag else ''}.npz", **arrays)
    name = f"oe_kfold_results{('_'+args.tag) if args.tag else ''}.json"
    (out / name).write_text(json.dumps(res, indent=2, default=str))
    print(f"\nSaved -> {out/name}")


if __name__ == "__main__":
    main()
