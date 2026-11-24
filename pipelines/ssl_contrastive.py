#!/usr/bin/env python3
"""Self-supervised contrastive encoder for zero-knowledge novelty detection.

Trains the ASTRAnet backbone (stem + TCN + attention pooling, no FiLM, no heads) with an NT-Xent
objective on two augmented views of every spectrum in train + val + the unlabelled application set
(rare-set objects are excluded by object id; there is no overlap by construction).  No label and
no anomaly is used.  The embedding is then scored with LUNA's label-free scorers (density/distance),
thresholded at 1% FAR on a validation half, and evaluated once on the rare set — the zero-knowledge
protocol of pipelines/discovery_eval.py — with a per-instrument flag-rate check (an unsupervised
space can learn the instrument instead of the astrophysics).

Usage: python pipelines/ssl_contrastive.py --tag ssl_s0 [--epochs 60] [--batch 256] [--with-labels]
Outputs -> models/ssl_<tag>/encoder.pt, cache/ssl_<tag>/*.npz, results/oe/ssl_contrastive_<tag>.{json,md}
"""
from __future__ import annotations
import argparse, collections, json, math, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from _common import REPO, resolve_path, load_config, make_astra_config, device_for, tiered_report
from astranet.model import ASTRANet, build_hierarchy_mappings
from astranet.dataset import SpectraDataset

FAR = 0.01


class TwoViews(Dataset):
    def __init__(self, X: torch.Tensor, ac):
        meta = [{} for _ in range(len(X))]
        self.ds = SpectraDataset({"X": X, "meta": meta}, labeled=False, augment=True,
                                 compute_derivative=ac.compute_derivative, class_names=ac.class_names,
                                 savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly,
                                 augmentor_kwargs=dict(mask_prob=0.8, noise_prob=0.8, flux_scale_prob=0.6,
                                                       wave_shift_prob=0.5, continuum_warp_prob=0.6))
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return self.ds[i]["x"], self.ds[i]["x"]


class Encoder(nn.Module):
    """ASTRAnet backbone without metadata conditioning or classification heads."""
    def __init__(self, ac, nco):
        super().__init__()
        self.net = ASTRANet(num_fine=ac.num_classes, num_coarse=nco, in_channels=ac.in_channels, base_c=ac.base_c,
                            emb_dim=ac.emb_dim, tcn_depth=ac.tcn_depth, tcn_kernel=ac.tcn_kernel, downsample=ac.downsample,
                            dropout=ac.dropout, proj_dim=ac.proj_dim, drop_path_rate=ac.drop_path_rate, attn_heads=ac.attn_heads)
        self.head = nn.Sequential(nn.Linear(ac.emb_dim, ac.emb_dim), nn.ReLU(), nn.Linear(ac.emb_dim, 128))
    def embed(self, x):
        n = self.net; feat = n.stem(x)
        if n.downsample > 1: feat = F.avg_pool1d(feat, n.downsample, n.downsample)
        feat = n.proj_in(feat); feat = n.tcn(feat)
        emb, _ = n.attn_pool(feat)
        return n.late_norm(emb)
    def forward(self, x):
        return F.normalize(self.head(self.embed(x)), dim=-1)


def nt_xent(z1, z2, T=0.1):
    z = torch.cat([z1, z2]); n = z1.shape[0]
    sim = z @ z.t() / T; sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.no_grad()
def extract(enc, X, ac, device, bs=256):
    ds = SpectraDataset({"X": X, "meta": [{} for _ in range(len(X))]}, labeled=False, augment=False,
                        compute_derivative=ac.compute_derivative, class_names=ac.class_names,
                        savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly)
    out = []
    for i in range(0, len(ds), bs):
        x = torch.stack([ds[j]["x"] for j in range(i, min(i + bs, len(ds)))]).to(device)
        out.append(enc.embed(x).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--with-labels", action="store_true",
                    help="also give LUNA the training labels (enables class-conditional scores)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_config(args.config); ac = make_astra_config(cfg); device = device_for(cfg)
    _, _, sc = build_hierarchy_mappings(ac.hierarchy, ac.class_names); nco = len(sc)
    ddir = Path(resolve_path(cfg["data"]["dir"]))
    tr = torch.load(ddir / "train.pt", map_location="cpu", weights_only=False)
    va = torch.load(ddir / "val.pt", map_location="cpu", weights_only=False)
    te = torch.load(ddir / "test.pt", map_location="cpu", weights_only=False)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu", weights_only=False)
    ch = torch.load(str(REPO / "data" / "challenging.pt"), map_location="cpu", weights_only=False)
    rare_ids = set(rd["obj_id"]); keep = [i for i, o in enumerate(ch["obj_id"]) if str(o) not in rare_ids]
    pool = torch.cat([tr["X"], va["X"], ch["flux"][keep].unsqueeze(1)]).float()
    print(f"SSL pool: train {len(tr['X'])} + val {len(va['X'])} + unlabelled {len(keep)} "
          f"(excluded {len(ch['obj_id']) - len(keep)} rare-set spectra) = {len(pool)}", flush=True)

    mdir = REPO / "models" / f"ssl_{args.tag}"; mdir.mkdir(parents=True, exist_ok=True)
    enc = Encoder(ac, nco).to(device)
    if (mdir / "encoder.pt").exists():
        enc.load_state_dict(torch.load(mdir / "encoder.pt", map_location=device)); print("loaded existing encoder", flush=True)
    else:
        loader = DataLoader(TwoViews(pool, ac), batch_size=args.batch, shuffle=True, drop_last=True,
                            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
        opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = args.epochs * len(loader); sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda"); hist = []
        for ep in range(1, args.epochs + 1):
            enc.train(); t0 = time.time(); tot = 0.0
            for v1, v2 in loader:
                v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = nt_xent(enc(v1).float(), enc(v2).float())
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item()
            hist.append({"epoch": ep, "loss": tot / len(loader)})
            print(f"[{ep:03d}/{args.epochs}] {time.time() - t0:.0f}s  nt-xent {tot / len(loader):.4f}", flush=True)
        torch.save(enc.state_dict(), mdir / "encoder.pt"); (mdir / "history.json").write_text(json.dumps(hist))

    # ---- embeddings
    enc.eval(); cdir = REPO / "cache" / f"ssl_{args.tag}"; cdir.mkdir(parents=True, exist_ok=True)
    E = {"train": extract(enc, tr["X"], ac, device), "val": extract(enc, va["X"], ac, device),
         "test": extract(enc, te["X"], ac, device), "rare": extract(enc, rd["flux"].unsqueeze(1), ac, device)}
    l2i = {c: i for i, c in enumerate(ac.class_names)}
    labels = {k: np.array([l2i.get(str(c), -1) for c in d["y"]]) for k, d in (("train", tr), ("val", va), ("test", te))}
    for k, e in E.items():
        np.savez(cdir / f"{k}.npz", emb=e, **({"labels": labels[k]} if k in labels else {}))

    # ---- zero-knowledge evaluation with LUNA (label-free scorers unless --with-labels)
    from luna import LUNAPipeline
    cls = np.array(rd["class"]); objid = np.array(rd["obj_id"]); inst_rare = np.array(rd["instrument"])
    inst_val = np.array([m.get("instrument", "?") for m in va["meta"]])
    tr_in = {"embeddings": E["train"]}
    if args.with_labels:
        ok = labels["train"] >= 0; tr_in = {"embeddings": E["train"][ok], "labels": labels["train"][ok]}
    pipe = LUNAPipeline(combiner="lightgbm", target_far=FAR, fine_to_coarse=None); pipe.fit(tr_in); M = pipe.active_methods
    sv = pipe.score({"embeddings": E["val"]}); sr = pipe.score({"embeddings": E["rare"]})
    vi = np.random.default_rng(42).permutation(len(E["val"])); vtr, vth = vi[:len(vi) // 2], vi[len(vi) // 2:]
    def rank(m, x): ref = np.sort(sv[m][vtr]); return np.searchsorted(ref, x) / len(ref)
    def objsum(det):
        ob = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(cls)):
            ob[objid[i]][1] = cls[i]
            if det[i]: ob[objid[i]][0] = True
        t = tiered_report((v[0], v[1]) for v in ob.values())
        return {"spectra_pct": 100 * float(det.mean()), "objects_pct": 100 * sum(v[0] for v in ob.values()) / len(ob),
                "tier1_pct": t["tier1"]["pct"], "tier2_pct": t["tier2"]["pct"]}
    rows = {}
    combos = {m: [m] for m in M}; combos["mean_rank_all"] = list(M)
    for name, combo in combos.items():
        v = np.mean([rank(m, sv[m][vth]) for m in combo], 0); r = np.mean([rank(m, sr[m]) for m in combo], 0)
        thr = np.quantile(v, 1 - FAR); det = r >= thr; s = objsum(det)
        fv = v >= thr
        s["val_flag_rate_by_instrument"] = {i: round(100 * float(fv[inst_val[vth] == i].mean()), 2) for i in np.unique(inst_val[vth]) if (inst_val[vth] == i).sum() >= 30}
        s["rare_flag_rate_by_instrument"] = {i: round(100 * float(det[inst_rare == i].mean()), 1) for i in np.unique(inst_rare) if (inst_rare == i).sum() >= 10}
        rows[name] = s
        print(f"  {name:16s} objects {s['objects_pct']:5.1f}%  tier1 {s['tier1_pct']:5.1f}%  tier2 {s['tier2_pct']:5.1f}%  "
              f"val flag by instrument {s['val_flag_rate_by_instrument']}", flush=True)
    out = REPO / "results" / "oe"; out.mkdir(parents=True, exist_ok=True)
    res = {"tag": args.tag, "epochs": args.epochs, "pool": int(len(pool)), "with_labels": args.with_labels, "scores": M, "results": rows}
    (out / f"ssl_contrastive_{args.tag}{'_labels' if args.with_labels else ''}.json").write_text(json.dumps(res, indent=2, default=float))
    md = ["| score (SSL contrastive embedding) | objects | Tier 1 | Tier 2 |", "|---|---|---|---|"]
    md += [f"| {k} | {v['objects_pct']:.1f}% | {v['tier1_pct']:.1f}% | {v['tier2_pct']:.1f}% |" for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["objects_pct"])]
    (out / f"ssl_contrastive_{args.tag}{'_labels' if args.with_labels else ''}.md").write_text("\n".join(md) + "\n"); print("\n".join(md)); print(f"Saved -> {out}/ssl_contrastive_{args.tag}.json")


if __name__ == "__main__":
    main()
