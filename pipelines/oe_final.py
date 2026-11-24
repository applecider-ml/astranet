#!/usr/bin/env python3
"""Deployable ASTRANet-Sentinel encoders: outlier-exposure fine-tuning with a PERMANENT
object-level held-out split of the rare set.

Protocol (per seed):
  1. Split the rare set 80/20 at the OBJECT level, stratified by raw class (fixed seed,
     saved to data/rare_split.json the first time and reused afterwards).
  2. OE fine-tune the seed's best checkpoint on the EXPOSURE objects only
     (CE + entropy-OE + embedding-margin, exactly as astranet.sentinel.oe_finetune).
  3. Extract train/val/rare features with the fine-tuned encoder (cached).
  4. Fit the 16 LUNA scores on train, the LightGBM combiner on val-half + exposure,
     set the 1% FAR threshold on the other val half.
  5. Report detection on the HELD-OUT objects (never seen in fine-tune or combiner),
     the test-set false-alarm rate, and the same numbers for the baseline (no OE) encoder.
Across seeds: anchor regime (mean score, mean threshold) and per-seed numbers.

Usage:
  python pipelines/oe_final.py --seeds 42 7 123 --epochs 20            # train + evaluate
  python pipelines/oe_final.py --seeds 42 7 123 --report-only          # aggregate existing runs
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
import lightgbm as lgb

from _common import (REPO, resolve_path, load_config, make_astra_config, device_for, load_model,
                     extract, save_cache, load_cache, rare_tier)
from astranet.sentinel import oe_finetune, luna_dict, LGB_PARAMS
from astranet.model import build_hierarchy_mappings
from astranet.dataset import build_loaders, make_loader

FAR = 0.01
SPLIT_FILE = REPO / "data" / "rare_split.json"
OUT_MODELS = REPO / "models" / "sentinel_oe"
OUT_CACHE = REPO / "cache" / "sentinel_oe"
OUT_RES = REPO / "results" / "oe"


# ── permanent object-level split ─────────────────────────────────────────────
def make_or_load_split(obj: np.ndarray, cls: np.ndarray, holdout_frac=0.2, seed=0) -> dict:
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text())
    rng = np.random.default_rng(seed)
    obj_cls = {o: c for o, c in zip(obj, cls)}
    by_cls = collections.defaultdict(list)
    for o, c in obj_cls.items():
        by_cls[c].append(o)
    held = []
    for c in sorted(by_cls):
        objs = sorted(by_cls[c]); rng.shuffle(objs)
        n = len(objs)
        k = 0 if n < 3 else max(1, int(round(holdout_frac * n)))   # singletons/pairs: exposure only
        held += objs[:k]
    split = {"holdout_objects": sorted(held),
             "exposure_objects": sorted(set(obj_cls) - set(held)),
             "holdout_frac": holdout_frac, "seed": seed,
             "note": "object-level, stratified by raw class; classes with <3 objects are exposure-only"}
    SPLIT_FILE.write_text(json.dumps(split, indent=2))
    return split


# ── one Sentinel (scorers + combiner + threshold) on one encoder's features ───
def fit_sentinel(tr, va, ra, f2c, expo_mask, seed=42):
    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=FAR, fine_to_coarse=f2c)
    pipe.fit(luna_dict(tr)); M = pipe.active_methods
    X = lambda f: np.column_stack([pipe.score(luna_dict(f))[m] for m in M])  # noqa: E731
    Xv, Xr = X(va), X(ra)
    rng = np.random.default_rng(seed); vi = rng.permutation(len(Xv))
    vtr, vth = vi[:len(Xv) // 2], vi[len(Xv) // 2:]
    clf = lgb.LGBMClassifier(**LGB_PARAMS).fit(np.vstack([Xv[vtr], Xr[expo_mask]]),
                                               np.r_[np.zeros(len(vtr)), np.ones(expo_mask.sum())])
    thr = float(np.quantile(clf.predict_proba(Xv[vth])[:, 1], 1 - FAR))
    return pipe, clf, M, thr, clf.predict_proba(Xr)[:, 1], X


def summarize(flags, obj, cls, held_mask):
    """Per-spectrum / per-object / tier detection on the held-out objects."""
    f, o, c = flags[held_mask], obj[held_mask], cls[held_mask]
    per_obj = collections.defaultdict(bool); ocls = {}
    for oi, fi, ci in zip(o, f, c):
        per_obj[oi] |= bool(fi); ocls[oi] = ci
    tiers = {1: [], 2: []}
    for oi, det in per_obj.items():
        t = rare_tier(ocls[oi])
        if t: tiers[t].append(det)
    per_class = {}
    for ci in sorted(set(c)):
        objs = [per_obj[oi] for oi in per_obj if ocls[oi] == ci]
        per_class[ci] = {"objects": len(objs), "detected": int(sum(objs))}
    return {"spectra_pct": 100 * f.mean(), "spectra": f"{int(f.sum())}/{len(f)}",
            "objects_pct": 100 * np.mean(list(per_obj.values())), "objects": f"{sum(per_obj.values())}/{len(per_obj)}",
            "tier1_pct": 100 * np.mean(tiers[1]) if tiers[1] else None, "tier1_n": len(tiers[1]),
            "tier2_pct": 100 * np.mean(tiers[2]) if tiers[2] else None, "tier2_n": len(tiers[2]),
            "per_class": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lam-emb", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None, help="pipeline yaml (default configs/default.yaml)")
    ap.add_argument("--tag", default="", help="namespace suffix for models/cache/results "
                    "(e.g. v7fixmd -> models/sentinel_oe_v7fixmd, results/oe/oe_final_results_v7fixmd.json)")
    args = ap.parse_args()
    global OUT_MODELS, OUT_CACHE
    if args.tag:
        OUT_MODELS = REPO / "models" / f"sentinel_oe_{args.tag}"
        OUT_CACHE = REPO / "cache" / f"sentinel_oe_{args.tag}"
    cfg = load_config(args.config); ac = make_astra_config(cfg); device = device_for(cfg)
    ac.num_workers = args.workers
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names); nco = len(sc)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    weights_dir = REPO / "models" / cfg["weights_source"]
    for d in (OUT_MODELS, OUT_CACHE, OUT_RES): d.mkdir(parents=True, exist_ok=True)

    # rare set + permanent split
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu", weights_only=False)
    cls = np.array(rd["class"]); obj = np.array(rd["obj_id"]); flux = rd["flux"]
    split = make_or_load_split(obj, cls)
    held_mask = np.isin(obj, split["holdout_objects"]); expo_mask = ~held_mask
    print(f"rare set: {len(cls)} spectra / {len(set(obj))} objects | exposure {expo_mask.sum()} spectra "
          f"({len(split['exposure_objects'])} obj) | HELD-OUT {held_mask.sum()} spectra "
          f"({len(split['holdout_objects'])} obj)", flush=True)
    print("held-out classes:", dict(collections.Counter(cls[held_mask])), flush=True)

    def rmeta(idx): return [{"obj_id": rd["obj_id"][i], "class": rd["class"][i],
                             "phase": rd["phase"][i], "z": rd["redshift"][i]} for i in idx]
    lk = dict(labeled=False, augment=False, num_workers=ac.num_workers, pin_memory=ac.pin_memory,
              compute_derivative=ac.compute_derivative, class_names=ac.class_names,
              savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)
    rare_loader = make_loader({"X": flux.unsqueeze(1), "meta": rmeta(range(len(cls)))}, ac.batch_size, shuffle=False, **lk)
    ac.augment_train = False; ac.use_balanced_sampling = False
    _, train_loader, val_loader, test_loader, _ = build_loaders(ac)

    results = {"split": {k: split[k] for k in ("holdout_frac", "seed")},
               "n_holdout_spectra": int(held_mask.sum()), "n_holdout_objects": len(split["holdout_objects"]),
               "epochs": args.epochs, "lam": args.lam, "lam_emb": args.lam_emb, "far": FAR, "seeds": {}}
    scores = {"oe": [], "base": []}; thrs = {"oe": [], "base": []}

    for seed in args.seeds:
        ckpt = weights_dir / f"seed{seed}" / "checkpoint_best.pt"
        enc_path = OUT_MODELS / f"seed{seed}" / "encoder.pt"; enc_path.parent.mkdir(exist_ok=True)
        cdir = OUT_CACHE / f"seed{seed}"
        print(f"\n===== seed {seed} =====", flush=True)

        # --- baseline (no OE) features, cached ---
        bdir = OUT_CACHE / f"seed{seed}_base"
        if not (bdir / "rare.npz").exists():
            base = load_model(ckpt, ac, nco, device)
            for name, ld, lab in (("train", train_loader, True), ("val", val_loader, True),
                                  ("test", test_loader, True), ("rare", rare_loader, False)):
                if not (bdir / f"{name}.npz").exists():
                    save_cache(extract([base], ld, device, labeled=lab), bdir, name)
            del base
        # --- OE fine-tune (skip if encoder exists) ---
        if not args.report_only and (args.force or not enc_path.exists()):
            t0 = time.time()
            ac.augment_train = True; ac.use_balanced_sampling = True
            _, train_aug, _, _, _ = build_loaders(ac)
            ac.augment_train = False; ac.use_balanced_sampling = False
            expo_idx = np.where(expo_mask)[0]
            expo_loader = make_loader({"X": flux[expo_idx].unsqueeze(1), "meta": rmeta(expo_idx)},
                                      ac.batch_size, shuffle=True, **lk)
            model = load_model(ckpt, ac, nco, device)
            model = oe_finetune(model, expo_loader, train_aug, device, epochs=args.epochs, lam=args.lam,
                                lam_emb=args.lam_emb, num_classes=ac.num_classes)
            torch.save(model.state_dict(), enc_path)
            print(f"  fine-tuned in {(time.time() - t0) / 60:.1f} min -> {enc_path}", flush=True)
            for name, ld, lab in (("train", train_loader, True), ("val", val_loader, True),
                                  ("test", test_loader, True), ("rare", rare_loader, False)):
                save_cache(extract([model], ld, device, labeled=lab), cdir, name)
            del model; torch.cuda.empty_cache()
        if not (cdir / "rare.npz").exists():
            print("  no OE features for this seed (run without --report-only)"); continue

        # --- Sentinels: baseline vs OE, same split, same protocol ---
        res_seed = {}
        for tag, d in (("base", bdir), ("oe", cdir)):
            tr, va, ra, te = (load_cache(d, n) for n in ("train", "val", "rare", "test"))
            pipe, clf, M, thr, s_rare, X = fit_sentinel(tr, va, ra, f2c, expo_mask, seed=seed)
            test_far = float((clf.predict_proba(X(te))[:, 1] >= thr).mean())
            summ = summarize(s_rare >= thr, obj, cls, held_mask)
            summ.update({"threshold": thr, "test_far_pct": 100 * test_far,
                         "exposure_insample_pct": 100 * float((s_rare[expo_mask] >= thr).mean())})
            res_seed[tag] = summ; scores[tag].append(s_rare); thrs[tag].append(thr)
            print(f"  [{tag:4s}] held-out: spectra {summ['spectra_pct']:.1f}%  objects {summ['objects_pct']:.1f}% "
                  f"(tier1 {summ['tier1_pct']}, tier2 {summ['tier2_pct']}) | test FAR {100 * test_far:.2f}% "
                  f"| thr {thr:.3f}", flush=True)
            if tag == "oe":
                import pickle
                with open(OUT_MODELS / f"seed{seed}" / "sentinel.pkl", "wb") as f:
                    pickle.dump({"scorer": pipe, "combiner": clf, "methods": M, "threshold": thr,
                                 "false_alarm_rate": FAR}, f)
        results["seeds"][seed] = res_seed

    # --- anchor regime across seeds ---
    for tag in ("base", "oe"):
        if len(scores[tag]) == len(args.seeds) and scores[tag]:
            s = np.mean(scores[tag], 0); t = float(np.mean(thrs[tag]))
            summ = summarize(s >= t, obj, cls, held_mask); summ["threshold"] = t
            results[f"anchor_{tag}"] = summ
            print(f"\nANCHOR [{tag}] ({len(args.seeds)} seeds, mean score/threshold) held-out: "
                  f"spectra {summ['spectra_pct']:.1f}%  objects {summ['objects_pct']:.1f}%  "
                  f"tier1 {summ['tier1_pct']} (n={summ['tier1_n']})  tier2 {summ['tier2_pct']} (n={summ['tier2_n']})")
    out = OUT_RES / f"oe_final_results{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
