#!/usr/bin/env python3
"""Outlier exposure with SYNTHETIC spectral anomalies — no real anomaly is used for training.

Synthetic anomalies are built from training spectra in the raw flux domain (the loader applies the
standard scaling afterwards), five families of ~equal size:
  featureless  heavily smoothed continuum (lines removed) + noise at the original residual level
  mixup        50/50 mix of two spectra from different classes
  flip         wavelength-reversed spectrum (unphysical line pattern)
  shift        lines displaced by 200–400 pixels (wrong rest wavelengths)
  lineboost    line features amplified x3 over the continuum
The encoder is OE fine-tuned on them (same recipe as the real-anomaly Sentinel), then every
detector is evaluated ONCE on the full rare set, which is never touched before:
  unsupervised  a-priori score families (rank-average) on the base and the synthetic-OE encoder
  heads         detectors trained on validation-half (ID) vs synthetic anomalies, 1% FAR threshold

Usage: python pipelines/oe_synthetic.py --config configs/v7fixmd.yaml --ckpt models/.../checkpoint_best.pt --tag v7fixmd_s42
Outputs -> results/oe/oe_synthetic_<tag>.{json,md}
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from _common import (resolve_path, REPO, load_config, make_astra_config, device_for, load_model, extract, tiered_report)
from astranet.sentinel import oe_finetune, luna_dict, detector_heads
from astranet.model import build_hierarchy_mappings
from astranet.dataset import build_loaders, make_loader

FAR = 0.01
FAMILIES = {"featureless", "mixup", "flip", "shift", "lineboost"}


def synthesize(X: np.ndarray, y: np.ndarray, n_per: int, rng) -> tuple[np.ndarray, list[str]]:
    N, L = X.shape; out, fam = [], []
    for _ in range(n_per):
        i = rng.integers(N); x = X[i].copy()
        sm = gaussian_filter1d(x, 60); resid = x - sm
        out.append(sm + rng.normal(0, resid.std() * 0.5, L)); fam.append("featureless")
        j = rng.integers(N)
        while y[j] == y[i]: j = rng.integers(N)
        out.append(0.5 * (x + X[j])); fam.append("mixup")
        out.append(x[::-1].copy()); fam.append("flip")
        out.append(np.roll(x, int(rng.choice([-1, 1]) * rng.integers(200, 400)))); fam.append("shift")
        out.append(sm + 3.0 * resid); fam.append("lineboost")
    return np.stack(out).astype(np.float32), fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--n-per", type=int, default=120)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--lam-emb", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg); device = device_for(cfg); ac.num_workers = args.workers
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names); nco = len(sc)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    ac.augment_train = False; ac.use_balanced_sampling = False
    _, train_loader, val_loader, _, _ = build_loaders(ac)
    lk = dict(labeled=False, augment=False, num_workers=ac.num_workers, pin_memory=ac.pin_memory,
              compute_derivative=ac.compute_derivative, class_names=ac.class_names,
              savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu", weights_only=False)
    cls = np.array(rd["class"]); objid = np.array(rd["obj_id"])
    rmeta = [{"obj_id": rd["obj_id"][i], "class": rd["class"][i], "phase": rd["phase"][i], "z": rd["redshift"][i]} for i in range(len(cls))]
    rare_loader = make_loader({"X": rd["flux"].unsqueeze(1), "meta": rmeta}, ac.batch_size, shuffle=False, **lk)

    td = torch.load(str(Path(resolve_path(cfg["data"]["dir"])) / "train.pt"), map_location="cpu", weights_only=False)
    Xtr = td["X"].squeeze(1).numpy(); ytr = np.array([m.get("class", "?") for m in td["meta"]])
    rng = np.random.default_rng(0); S, fam = synthesize(Xtr, ytr, args.n_per, rng)
    smeta = [{"obj_id": f"synthetic_{k}", "class": f"synthetic:{f}", "phase": float("nan"), "z": float("nan")} for k, f in enumerate(fam)]
    syn_eval = make_loader({"X": torch.from_numpy(S).unsqueeze(1), "meta": smeta}, ac.batch_size, shuffle=False, **lk)
    syn_expo = make_loader({"X": torch.from_numpy(S).unsqueeze(1), "meta": smeta}, ac.batch_size, shuffle=True, **lk)
    print(f"synthetic anomalies: {len(S)} spectra ({args.n_per} per family x {len(FAMILIES)})", flush=True)

    # --- base encoder features
    base = load_model(args.ckpt, ac, nco, device)
    F = {"base": {n: extract([base], ld, device, labeled=lab) for n, ld, lab in
                  (("train", train_loader, True), ("val", val_loader, True), ("rare", rare_loader, False), ("syn", syn_eval, False))}}
    del base; torch.cuda.empty_cache()
    # --- OE fine-tune on synthetic anomalies only
    ac.augment_train = True; ac.use_balanced_sampling = True
    _, train_aug, _, _, _ = build_loaders(ac)
    ac.augment_train = False; ac.use_balanced_sampling = False
    m = oe_finetune(load_model(args.ckpt, ac, nco, device), syn_expo, train_aug, device, epochs=args.epochs,
                    lam=args.lam, lam_emb=args.lam_emb, num_classes=ac.num_classes)
    F["oe_syn"] = {n: extract([m], ld, device, labeled=lab) for n, ld, lab in
                   (("train", train_loader, True), ("val", val_loader, True), ("rare", rare_loader, False), ("syn", syn_eval, False))}
    del m; torch.cuda.empty_cache()

    # --- evaluation on the full rare set (never used above)
    from luna import LUNAPipeline
    def objsum(det):
        ob = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(cls)):
            ob[objid[i]][1] = cls[i]
            if det[i]: ob[objid[i]][0] = True
        t = tiered_report((v[0], v[1]) for v in ob.values())
        return {"spectra_pct": 100 * float(det.mean()), "objects_pct": 100 * sum(v[0] for v in ob.values()) / len(ob),
                "tier1_pct": t["tier1"]["pct"], "tier2_pct": t["tier2"]["pct"]}
    res = {"tag": args.tag, "n_synthetic": int(len(S)), "epochs": args.epochs, "encoders": {}}
    families = {"typicality": ["typicality"], "density_trio": ["typicality", "knn", "mahal_class"], "all16": None}
    for enc, f in F.items():
        pipe = LUNAPipeline(combiner="lightgbm", target_far=FAR, fine_to_coarse=f2c); pipe.fit(luna_dict(f["train"])); M = pipe.active_methods
        sv, sr, ss = (pipe.score(luna_dict(f[k])) for k in ("val", "rare", "syn"))
        vi = np.random.default_rng(42).permutation(len(f["val"]["emb"])); vtr, vth = vi[:len(vi) // 2], vi[len(vi) // 2:]
        r = {"unsupervised": {}, "heads_synthetic_exposure": {}}
        def rank(mm, x): ref = np.sort(sv[mm][vtr]); return np.searchsorted(ref, x) / len(ref)
        for name, combo in families.items():
            combo = combo or M
            v = np.mean([rank(mm, sv[mm][vth]) for mm in combo], 0); rr = np.mean([rank(mm, sr[mm]) for mm in combo], 0)
            thr = np.quantile(v, 1 - FAR); r["unsupervised"][name] = objsum(rr >= thr)
        X16 = {k: np.column_stack([s[mm] for mm in M]) for k, s in (("val", sv), ("rare", sr), ("syn", ss))}
        XE = {k: f[k]["emb"].astype(np.float32) for k in ("val", "rare", "syn")}
        XB = {k: np.hstack([XE[k], X16[k]]) for k in XE}
        for hname, (kind, make) in detector_heads().items():
            X = {"scores": X16, "emb": XE, "both": XB}[kind]
            clf = make().fit(np.vstack([X["val"][vtr], X["syn"]]), np.r_[np.zeros(len(vtr)), np.ones(len(X["syn"]))])
            pv = clf.predict_proba(X["val"][vth])[:, 1]; thr = np.quantile(pv, 1 - FAR)
            r["heads_synthetic_exposure"][hname] = objsum(clf.predict_proba(X["rare"])[:, 1] >= thr)
        res["encoders"][enc] = r
        print(f"\n[{enc}]"); [print(f"  unsup {k:14s} objects {v['objects_pct']:5.1f}%  tier1 {v['tier1_pct']:5.1f}%") for k, v in r["unsupervised"].items()]
        [print(f"  head  {k:20s} objects {v['objects_pct']:5.1f}%  tier1 {v['tier1_pct']:5.1f}%") for k, v in r["heads_synthetic_exposure"].items()]
    out = REPO / "results" / "oe"; out.mkdir(parents=True, exist_ok=True)
    (out / f"oe_synthetic_{args.tag}.json").write_text(json.dumps(res, indent=2, default=float)); print(f"\nSaved -> {out}/oe_synthetic_{args.tag}.json")


if __name__ == "__main__":
    main()
