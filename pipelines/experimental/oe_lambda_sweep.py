#!/usr/bin/env python3
"""Outlier-Exposure fine-tuning to boost the Sentinel LightGBM @ 1% FAR (honest).

Fine-tunes the encoder with  L = CE(train) + lambda * OE(exposure anomalies -> uniform),
which pushes anomalies toward high-entropy predictions and out of the class clusters.

HONEST eval (lessons learned):
  * extraction uses UN-AUGMENTED loaders for train/val/rare (augmenting val created a
    spurious val-vs-rare gap before),
  * the combiner is the paper's LIGHT LightGBM (can't over-fit to 100%),
  * stratified 70/30 split of the 289: fine-tune + combiner see the 70% EXPOSURE only;
    detection is measured on the held-out 30% (never in fine-tune or combiner training),
  * the 1% FAR threshold is set on a val split disjoint from the combiner's training.

Usage: python pipelines/oe_finetune.py --epochs 12 --lam 0.5 --holdout-frac 0.30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _common import (resolve_path, REPO, load_config, make_astra_config, device_for,
                     load_model, extract)
from astranet.model import build_hierarchy_mappings
from astranet.dataset import build_loaders, make_loader

LGB_KW = dict(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.05,
              min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
              class_weight="balanced", verbosity=-1)   # paper-light, regularized


def rare_loader(ac, flux, meta):
    return make_loader({"X": flux.unsqueeze(1), "meta": meta}, ac.batch_size, shuffle=False,
                       labeled=False, augment=False, num_workers=ac.num_workers,
                       pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                       class_names=ac.class_names, savgol_window=ac.savgol_window,
                       savgol_poly=ac.savgol_poly)


def luna_feats(pipe, f):
    o = {"embeddings": f["emb"].astype(np.float32), "logits": f["fine_logits"].astype(np.float32),
         "coarse_logits": f["coarse_logits"].astype(np.float32), "projections": f["proj_fine"].astype(np.float32)}
    if f.get("labels") is not None: o["labels"] = np.asarray(f["labels"])
    return o


def ad_detection(train_f, val_f, rare_f, expo_mask, hold_mask, f2c, far=0.01, tag=""):
    """Light LightGBM on 16 LUNA scores. Train: val + EXPOSURE rare. Test: held-out rare."""
    from luna import LUNAPipeline
    import lightgbm as lgb
    pipe = LUNAPipeline(combiner="lightgbm", target_far=far, fine_to_coarse=f2c)
    pipe.fit(luna_feats(pipe, train_f))
    M = pipe.active_methods
    Xv = np.column_stack([pipe.score(luna_feats(pipe, val_f))[m] for m in M])
    rs = pipe.score(luna_feats(pipe, rare_f)); Xr = np.column_stack([rs[m] for m in M])
    nval = len(Xv); rng = np.random.default_rng(42); vi = rng.permutation(nval)
    vtr, vth = vi[:nval // 2], vi[nval // 2:]
    X = np.vstack([Xv[vtr], Xr[expo_mask]]); y = np.r_[np.zeros(len(vtr)), np.ones(expo_mask.sum())]
    clf = lgb.LGBMClassifier(**LGB_KW).fit(X, y)
    thr = np.quantile(clf.predict_proba(Xv[vth])[:, 1], 1 - far)
    s_hold = clf.predict_proba(Xr[hold_mask])[:, 1]
    det = (s_hold >= thr).mean() * 100
    val_fp = (clf.predict_proba(Xv[vth])[:, 1] >= thr).mean() * 100
    sv = clf.predict_proba(Xv[vth])[:, 1]
    a = np.r_[sv, s_hold]; yy = np.r_[np.zeros(len(sv)), np.ones(len(s_hold))]
    o = np.argsort(a); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a) + 1)
    auroc = (rk[yy == 1].sum() - len(s_hold) * (len(s_hold) + 1) / 2) / (len(s_hold) * len(sv))
    print(f"    [{tag}] held-out det={det:.1f}%  val_FP={val_fp:.1f}%  AUROC(hold vs val)={auroc:.3f}", flush=True)
    return det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/paper/seed42/checkpoint_best.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lam", type=float, default=0.5, help="entropy-OE weight")
    ap.add_argument("--lam-emb", type=float, default=0.0, help="embedding-margin OE weight")
    ap.add_argument("--margin-mult", type=float, default=2.0, help="push anomalies > mult x median centroid dist")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    ap.add_argument("--tag", default="oe")
    args = ap.parse_args()
    ckpt = Path(args.ckpt); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    cfg = load_config(); ac = make_astra_config(cfg); device = device_for(cfg)
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names); nco = len(sc)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)

    # ---- CLEAN (un-augmented) loaders for extraction ----
    ac.augment_train = False; ac.use_balanced_sampling = False
    _, train_loader, val_loader, _, _ = build_loaders(ac)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu")
    rare_cls = np.array(rd["class"]); flux = rd["flux"]
    # MUST include phase + z (FiLM metadata) or rare embeddings are NaN-conditioned
    # and systematically differ from val -> spurious perfect AD separation.
    meta = [{"obj_id": rd["obj_id"][i], "class": rd["class"][i],
             "phase": rd["phase"][i], "z": rd["redshift"][i]} for i in range(len(rare_cls))]
    rl = rare_loader(ac, flux, meta)

    # ---- stratified 70/30 split of the 289 ----
    rng = np.random.default_rng(0)
    hold_mask = np.zeros(len(rare_cls), bool)
    for c in np.unique(rare_cls):
        idx = np.where(rare_cls == c)[0]; rng.shuffle(idx)
        k = max(1, int(round(len(idx) * args.holdout_frac)))
        hold_mask[idx[:k]] = True
    expo_mask = ~hold_mask
    print(f"289 split: exposure={expo_mask.sum()}  held-out(test)={hold_mask.sum()}", flush=True)

    def extract_all(model):
        return (extract([model], train_loader, device, labeled=True),
                extract([model], val_loader, device, labeled=True),
                extract([model], rl, device, labeled=False))

    # ---- baseline (no OE) ----
    print("\n[baseline] extract + AD ...", flush=True)
    base = load_model(str(ckpt), ac, nco, device)
    base_det = ad_detection(*extract_all(base), expo_mask, hold_mask, f2c, tag="baseline")

    # ---- OE fine-tune (augmented train loader + exposure anomalies) ----
    ac.augment_train = True; ac.use_balanced_sampling = True
    _, train_aug, _, _, _ = build_loaders(ac)
    expo_idx = np.where(expo_mask)[0]
    expo_meta = [meta[i] for i in expo_idx]
    expo_loader = make_loader({"X": flux[expo_idx].unsqueeze(1), "meta": expo_meta}, ac.batch_size,
                              shuffle=True, labeled=False, augment=False, num_workers=ac.num_workers,
                              pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                              class_names=ac.class_names, savgol_window=ac.savgol_window,
                              savgol_poly=ac.savgol_poly)
    model = load_model(str(ckpt), ac, nco, device); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ei = iter(expo_loader)
    K = ac.num_classes

    @torch.no_grad()
    def class_centroids():
        """Per-class mean embedding from a train subsample (current model)."""
        model.eval(); embs = []; ys = []
        for i, b in enumerate(train_loader):
            o = model(b["x"].to(device), b["phase"].to(device), b["z"].to(device))
            embs.append(o["embedding"]); ys.append(b["y"].to(device))
            if i >= 8: break          # ~8 batches subsample, cheap
        model.train()
        E = torch.cat(embs); Y = torch.cat(ys)
        cents = torch.stack([E[Y == c].mean(0) if (Y == c).any() else E.mean(0) for c in range(K)])
        med = torch.cdist(E, cents).min(1).values.median()
        return cents, med

    for ep in range(args.epochs):
        cents = med = margin = None
        if args.lam_emb > 0:
            cents, med = class_centroids()
            margin = args.margin_mult * med
        ce_t = oe_t = em_t = 0; nb = 0
        for b in train_aug:
            out = model(b["x"].to(device), b["phase"].to(device), b["z"].to(device))
            ce = F.cross_entropy(out["fine_logits"], b["y"].to(device))
            try: ob = next(ei)
            except StopIteration: ei = iter(expo_loader); ob = next(ei)
            oo = model(ob["x"].to(device), ob["phase"].to(device), ob["z"].to(device))
            oe = -(F.log_softmax(oo["fine_logits"], 1).mean(1)).mean()
            loss = ce + args.lam * oe
            em = torch.tensor(0.0, device=device)
            if args.lam_emb > 0:
                d_min = torch.cdist(oo["embedding"], cents).min(1).values   # dist to nearest centroid
                em = F.relu(margin - d_min).mean()                           # push anomalies > margin away
                loss = loss + args.lam_emb * em
            opt.zero_grad(); loss.backward(); opt.step()
            ce_t += ce.item(); oe_t += oe.item(); em_t += float(em); nb += 1
        if (ep + 1) % 3 == 0 or ep == 0:
            mtxt = f"{float(margin):.1f}" if margin is not None else "-"
            print(f"  epoch {ep+1:2d}  CE={ce_t/nb:.3f}  OE={oe_t/nb:.3f}  EMB={em_t/nb:.3f}  (margin={mtxt})", flush=True)

    print("\n[OE] extract + AD ...", flush=True)
    model.eval()
    oe_det = ad_detection(*extract_all(model), expo_mask, hold_mask, f2c, tag="OE")

    print(f"\n=== held-out (30%) detection @1% FAR  [{args.tag}: lam={args.lam} lam_emb={args.lam_emb} ep={args.epochs}] ===")
    print(f"  baseline (no OE): {base_det:.1f}%")
    print(f"  OE fine-tuned   : {oe_det:.1f}%   ({oe_det-base_det:+.1f} pp)")
    out = REPO / "results" / "oe"; out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{args.tag}.pt")
    import json
    (out / f"{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "lam": args.lam, "lam_emb": args.lam_emb, "epochs": args.epochs,
         "baseline_det": base_det, "oe_det": oe_det, "gain_pp": oe_det - base_det}, indent=2))


if __name__ == "__main__":
    main()
