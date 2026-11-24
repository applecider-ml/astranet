#!/usr/bin/env python3
"""Self-supervised masked-spectrum pretraining of the ASTRANet trunk.

MAE-style: mask 2-4 contiguous wavelength chunks (~35% of the spectrum), run the
usual trunk (stem -> pool -> proj_in -> FiLM(meta) -> TCN), and reconstruct the
ORIGINAL flux from the feature map with a light conv decoder. Loss = MSE on the
masked positions (+0.1 on unmasked). Phase/z FiLM conditioning is kept so those
weights pretrain too.

Data: flux from the v7new train+val splits ONLY (labels unused). Test, challenging
and rare are deliberately excluded so downstream AD evaluation stays honest.

Output: models/ssl/encoder_pretrained.pt  — trunk state_dict (stem, proj_in,
        film_seq, meta_enc, tcn) to be loaded via config `init_from` before
        supervised training (see train_loop non-strict load).

Usage: python pipelines/ssl_pretrain.py --config configs/v7new.yaml --epochs 80
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import REPO, load_config, make_astra_config, device_for
from astranet.model import ASTRANet, build_hierarchy_mappings
from astranet.dataset import make_loader

MASK_FRAC = 0.35          # total fraction of the spectrum masked
MASK_CHUNKS = (2, 4)      # number of contiguous masked blocks


class Decoder(nn.Module):
    """Feature map (B, emb, L/ds) -> reconstructed flux (B, L)."""

    def __init__(self, emb_dim: int, ds: int):
        super().__init__()
        self.ds = ds
        self.net = nn.Sequential(
            nn.Conv1d(emb_dim, 96, 3, padding=1), nn.GELU(),
            nn.Conv1d(96, 96, 3, padding=1), nn.GELU(),
            nn.Conv1d(96, 1, 1),
        )

    def forward(self, feat, L):
        x = F.interpolate(feat, size=L, mode="linear", align_corners=False)
        return self.net(x).squeeze(1)


def mask_batch(x, rng):
    """Zero 2-4 contiguous chunks per sample (both channels). Returns (x_masked, mask[B,L])."""
    B, C, L = x.shape
    mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    for b in range(B):
        n = rng.integers(MASK_CHUNKS[0], MASK_CHUNKS[1] + 1)
        total = int(MASK_FRAC * L)
        sizes = rng.dirichlet(np.ones(n)) * total
        for s in sizes:
            s = max(8, int(s))
            start = rng.integers(0, max(1, L - s))
            mask[b, start:start + s] = True
    xm = x.clone()
    xm[mask.unsqueeze(1).expand(-1, C, -1)] = 0.0
    return xm, mask


def trunk_forward(model, x, phase, z):
    """The classifier's own trunk, up to the pre-pooling feature map."""
    meta = model._encode_meta(phase, z)
    feat = model.stem(x)
    if model.downsample > 1:
        feat = F.avg_pool1d(feat, model.downsample, model.downsample)
    feat = model.proj_in(feat)
    feat = model.film_seq(feat, meta)
    return model.tcn(feat)


TRUNK_PREFIXES = ("stem.", "proj_in.", "film_seq.", "meta_enc.", "tcn.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v7new.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names)

    # ---- data: train+val flux only (no labels, no test/challenging/rare) ----
    X, meta = [], []
    for split in ["train", "val"]:
        d = torch.load(Path(ac.data_dir) / f"{split}.pt", map_location="cpu")
        X.append(d["X"] if d["X"].dim() == 3 else d["X"].unsqueeze(1))
        meta += d["meta"]
    X = torch.cat(X)
    loader = make_loader({"X": X, "meta": meta}, args.batch, shuffle=True,
                         labeled=False, augment=False, num_workers=ac.num_workers,
                         pin_memory=ac.pin_memory, compute_derivative=ac.compute_derivative,
                         class_names=ac.class_names, savgol_window=ac.savgol_window,
                         savgol_poly=ac.savgol_poly)
    print(f"SSL corpus: {len(X)} spectra (train+val, labels unused)", flush=True)

    model = ASTRANet(num_fine=ac.num_classes, num_coarse=len(sc), in_channels=ac.in_channels,
                     base_c=ac.base_c, emb_dim=ac.emb_dim, tcn_depth=ac.tcn_depth,
                     tcn_kernel=ac.tcn_kernel, downsample=ac.downsample, dropout=ac.dropout,
                     drop_path_rate=ac.drop_path_rate, attn_heads=ac.attn_heads,
                     meta_hidden=ac.meta_hidden, meta_out_dim=ac.meta_out_dim,
                     emb_dropout=ac.emb_dropout, fine_head_stop_grad=ac.fine_head_stop_grad,
                     proj_dim=ac.proj_dim).to(device)
    dec = Decoder(ac.emb_dim, ac.downsample).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(dec.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(0)

    out = REPO / "models" / "ssl"; out.mkdir(parents=True, exist_ok=True)
    corpus_tag = Path(ac.data_dir).name          # e.g. object-split-v7fix
    ckpt_name = f"encoder_pretrained_{corpus_tag}.pt"
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train(); dec.train(); tot = n = 0; t0 = time.time()
        for b in loader:
            x = b["x"].to(device); phase = b["phase"].to(device); z = b["z"].to(device)
            # robust per-spectrum standardized target (a handful of pathological
            # high-amplitude spectra otherwise dominate / destabilize the MSE)
            t = x[:, 0]
            med = t.median(dim=1, keepdim=True).values
            scale = (t - med).abs().median(dim=1, keepdim=True).values * 1.4826 + 1e-3
            target = ((t - med) / scale).clamp(-8, 8)
            xm, mask = mask_batch(x, rng)
            feat = trunk_forward(model, xm, phase, z)
            rec = dec(feat, x.shape[-1])
            err = F.huber_loss(rec, target, reduction="none", delta=2.0)
            loss = err[mask].mean() + 0.1 * err[~mask].mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(x); n += len(x)
        sched.step()
        ep_loss = tot / n
        print(f"[{ep:3d}/{args.epochs}] {time.time()-t0:.0f}s  masked-MSE={ep_loss:.5f}", flush=True)
        if ep_loss < best:
            best = ep_loss
            trunk = {k: v for k, v in model.state_dict().items()
                     if k.startswith(TRUNK_PREFIXES)}
            torch.save({"state_dict": trunk, "epoch": ep, "loss": ep_loss,
                        "corpus": f"{corpus_tag} train+val", "mask_frac": MASK_FRAC},
                       out / ckpt_name)
    print(f"\nbest masked-MSE={best:.5f} -> {out/ckpt_name}", flush=True)


if __name__ == "__main__":
    main()
