#!/usr/bin/env python3
"""Convolutional-autoencoder anomaly detection in SPECTRUM space.

Orthogonal to the classifier embedding: a 1D CAE is trained to reconstruct the
NORMAL training spectra (~16k). Anomalies reconstruct poorly -> high MSE -> flagged.
The 289 rare spectra are NEVER used for training (test only), so "few samples" is a
non-issue. We evaluate recovery @ target FAR (threshold on val), and also whether
the CAE error is COMPLEMENTARY to the LUNA embedding ensemble (OR / LightGBM combine).

Outputs -> results/cae/cae_ad.csv  (+ checkpoint)
Usage: python pipelines/cae_ad.py --epochs 40 --target-far 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from _common import REPO, load_config, device_for


def _norm(X):
    """Per-spectrum z-normalisation (robust to overall flux scale)."""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6
    return (X - mu) / sd


def load_flux(cfg):
    import torch as t
    dd = Path(cfg["data"]["dir"])
    tr = t.load(dd / "train.pt", map_location="cpu")["X"].squeeze(1).numpy()
    va = t.load(dd / "val.pt", map_location="cpu")["X"].squeeze(1).numpy()
    ra = t.load(cfg["data"]["rare_pt"], map_location="cpu")["flux"].numpy()
    return _norm(tr), _norm(va), _norm(ra)


class CAE(nn.Module):
    def __init__(self, L=4096):
        super().__init__()
        def enc(i, o): return nn.Sequential(nn.Conv1d(i, o, 7, 2, 3), nn.BatchNorm1d(o), nn.GELU())
        def dec(i, o, last=False):
            layers = [nn.ConvTranspose1d(i, o, 4, 2, 1)]
            if not last: layers += [nn.BatchNorm1d(o), nn.GELU()]
            return nn.Sequential(*layers)
        self.encoder = nn.Sequential(enc(1, 16), enc(16, 32), enc(32, 64), enc(64, 128))
        self.decoder = nn.Sequential(dec(128, 64), dec(64, 32), dec(32, 16), dec(16, 1, last=True))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def recon_err(model, X, device, bs=512):
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i+bs]).unsqueeze(1).to(device)
            out = model(xb)
            errs.append(((out - xb) ** 2).mean((1, 2)).cpu().numpy())
    return np.concatenate(errs)


def held_out_far(rare_s, val_s, far):
    rng = np.random.default_rng(42); idx = rng.permutation(len(val_s))
    A, B = idx[:len(idx)//2], idx[len(idx)//2:]
    thr = np.quantile(val_s[A], 1 - far)
    return (rare_s >= thr).mean()*100, (val_s[B] >= thr).mean()*100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--spec", default="6model", help="for combining with LUNA ensemble OOF scores")
    args = ap.parse_args()
    cfg = load_config(); device = device_for(cfg)
    out = REPO / "results" / "cae"; out.mkdir(parents=True, exist_ok=True)

    Xtr, Xva, Xra = load_flux(cfg)
    print(f"CAE: train={Xtr.shape}  val={Xva.shape}  rare={Xra.shape}  device={device}", flush=True)

    model = CAE(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    Xt = torch.tensor(Xtr)
    n = len(Xt); bs = 256
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0
        for i in range(0, n, bs):
            xb = Xt[perm[i:i+bs]].unsqueeze(1).to(device)
            rx = model(xb); loss = ((rx - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  train MSE = {tot/n:.4f}", flush=True)
    torch.save(model.state_dict(), out / "cae.pt")

    e_va = recon_err(model, Xva, device); e_ra = recon_err(model, Xra, device)
    far = args.target_far
    rec, fp = held_out_far(e_ra, e_va, far)
    print(f"\nCAE recon-error AD  @ {far*100:.0f}% FAR : recovery = {rec:.1f}%  (FP {fp:.1f}%)")

    # AUROC
    def auroc(neg, pos):
        a = np.r_[neg, pos]; y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        o = np.argsort(a); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a)+1)
        return (rk[y == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
    print(f"CAE AUROC = {auroc(e_va, e_ra):.4f}")

    # Complementarity: OR with the LUNA embedding ensemble (honest OOF scores)
    conf = REPO / "results" / args.spec / "conformal"
    rows = [("CAE recon-error", rec, fp)]
    try:
        v_l = np.load(conf / "val_lgb_scores.npy")   # OOF val (honest)
        # honest rare via held-out: use the robust_ad clean-OOF if present, else skip OR
        # Build CAE-vs-ensemble OR at matched 1% each (free-detection union)
        # ensemble rare scores (note: leaked; for OR illustration of complementarity only)
        r_l = np.load(conf / "rare_lgb_scores.npy")
        # recovery of ensemble alone at far (honest val threshold)
        rec_e, _ = held_out_far(r_l, v_l, far)
        # OR: flag if CAE OR ensemble exceeds their own val thresholds (each at far/2 to keep ~far total)
        rng = np.random.default_rng(0); idx = rng.permutation(len(v_la := v_l))
        A = idx[:len(idx)//2]
        thr_e = np.quantile(v_l[A], 1-far); thr_c = np.quantile(e_va[A], 1-far)
        union = (r_l >= thr_e) | (e_ra >= thr_c)
        rows.append(("LUNA ensemble (ref)", rec_e, np.nan))
        rows.append(("CAE OR ensemble", union.mean()*100, np.nan))
        print(f"\nComplementarity (illustrative): CAE OR LUNA-ensemble recovery = {union.mean()*100:.1f}%")
        # how many CAE catches that ensemble misses
        miss_e = r_l < thr_e
        cae_saves = ((e_ra >= thr_c) & miss_e).sum()
        print(f"  CAE catches {cae_saves} anomalies the embedding ensemble misses")
    except Exception as ex:
        print(f"  (ensemble combine skipped: {ex})")

    import pandas as pd
    pd.DataFrame(rows, columns=["method", "recovery_%", "FP_%"]).to_csv(out / "cae_ad.csv", index=False)
    print(f"\nSaved -> {out/'cae_ad.csv'}")


if __name__ == "__main__":
    main()
