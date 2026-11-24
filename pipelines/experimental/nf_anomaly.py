#!/usr/bin/env python3
"""Normalizing-flow anomaly head on the classifier embedding.

A RealNVP flow is trained on TRAIN-split embeddings (in-distribution only — no
anomaly labels anywhere), giving a proper likelihood; anomaly score = -log p(z).
Unlike the 16-score LightGBM Sentinel this is fully UNSUPERVISED w.r.t. anomalies,
so there is no rare-set leakage question at all.

Benchmarked with the same honest protocol as the Sentinel:
  * val split in half: half_cal sets the 1% FAR threshold, half_holdout checks FP
  * detection rate on the 289 rare (per-spectrum + per-object)
  * AUROC val-vs-rare
Also reports the LightGBM-16 baseline from the same cache, and LightGBM-17
(16 scores + NF as an extra feature).

Usage: python pipelines/nf_anomaly.py --spec 6model [--config configs/v7new.yaml]
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import torch
import torch.nn as nn

from _common import REPO, load_config, make_astra_config, load_cache
from astranet.model import build_hierarchy_mappings

FAR = 0.01


# ── RealNVP ───────────────────────────────────────────────────────────
class Coupling(nn.Module):
    def __init__(self, dim, hidden, flip):
        super().__init__()
        self.flip = flip
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * (dim - half)),
        )
        self.net[-1].weight.data.zero_(); self.net[-1].bias.data.zero_()

    def forward(self, x):
        half = x.shape[1] // 2
        a, b = (x[:, half:], x[:, :half]) if self.flip else (x[:, :half], x[:, half:])
        s, t = self.net(a).chunk(2, 1)
        s = torch.tanh(s) * 2.0
        b = b * torch.exp(s) + t
        out = torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)
        return out, s.sum(1)


class RealNVP(nn.Module):
    def __init__(self, dim, hidden=256, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [Coupling(dim, hidden, flip=i % 2 == 1) for i in range(n_layers)])
        self.dim = dim

    def log_prob(self, x):
        ldj = torch.zeros(len(x), device=x.device)
        for l in self.layers:
            x, s = l(x)
            ldj = ldj + s
        logp_z = -0.5 * (x ** 2 + np.log(2 * np.pi)).sum(1)
        return logp_z + ldj


def luna_dict(f):
    o = {"embeddings": f["emb"].astype(np.float32), "logits": f["fine_logits"].astype(np.float32),
         "coarse_logits": f["coarse_logits"].astype(np.float32), "projections": f["proj_fine"].astype(np.float32)}
    if f.get("labels") is not None:
        o["labels"] = np.asarray(f["labels"])
    return o


def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])


def honest_detect(s_val, s_rare, rng):
    """Split val scores in half; threshold on cal half at 1% FAR; report holdout FP + rare det."""
    vi = rng.permutation(len(s_val))
    cal, hold = vi[:len(vi) // 2], vi[len(vi) // 2:]
    thr = np.quantile(s_val[cal], 1 - FAR)
    return (s_rare >= thr), float((s_val[hold] >= thr).mean() * 100), thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v7fix6")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)

    tr = load_cache(cache, "train"); va = load_cache(cache, "val")
    ra = load_cache(cache, "rare_unknown_combined")
    robj = np.asarray([m["obj_id"] for m in ra["meta"]])
    rcls = np.asarray([m["class"] for m in ra["meta"]])

    # ---- standardize on train ----
    mu, sd = tr["emb"].mean(0), tr["emb"].std(0) + 1e-6
    Ztr = torch.from_numpy(((tr["emb"] - mu) / sd).astype(np.float32))
    Zva = torch.from_numpy(((va["emb"] - mu) / sd).astype(np.float32))
    Zra = torch.from_numpy(((ra["emb"] - mu) / sd).astype(np.float32))

    # ---- train flow on TRAIN only; early stop on val NLL ----
    flow = RealNVP(Ztr.shape[1]).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_nll, best_state, patience, bad = float("inf"), None, 30, 0
    Ztr_d = Ztr.to(device); Zva_d = Zva.to(device)
    for ep in range(1, args.epochs + 1):
        flow.train()
        perm = torch.randperm(len(Ztr_d), device=device)
        tot = 0.0
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            x = Ztr_d[idx] + 0.01 * torch.randn_like(Ztr_d[idx])   # tiny dequant noise
            nll = -flow.log_prob(x).mean()
            opt.zero_grad(); nll.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            tot += nll.item() * len(idx)
        sched.step()
        flow.eval()
        with torch.no_grad():
            vnll = -flow.log_prob(Zva_d).mean().item()
        if ep % 10 == 0:
            print(f"[{ep:3d}/{args.epochs}] train NLL={tot/len(Ztr_d):.3f}  val NLL={vnll:.3f}", flush=True)
        if vnll < best_nll - 1e-3:
            best_nll, bad = vnll, 0
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop @ {ep} (best val NLL={best_nll:.3f})", flush=True)
                break
    flow.load_state_dict(best_state); flow.eval()

    with torch.no_grad():
        s_val = (-flow.log_prob(Zva_d)).cpu().numpy()
        s_rare = (-flow.log_prob(Zra.to(device))).cpu().numpy()

    # ---- LightGBM baselines from the same cache ----
    from luna import LUNAPipeline
    import lightgbm as lgb
    pipe = LUNAPipeline(combiner="lightgbm", target_far=FAR, fine_to_coarse=f2c)
    pipe.fit(luna_dict(tr)); M = pipe.active_methods
    def X(g): s = pipe.score(luna_dict(g)); return np.column_stack([s[m] for m in M])
    Xv, Xr = X(va), X(ra)
    LGB = dict(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.05,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               class_weight="balanced", random_state=42, verbosity=-1)

    def lgb_scores(Xv_, Xr_):
        vi = rng.permutation(len(Xv_)); cal, hold = vi[:len(vi)//2], vi[len(vi)//2:]
        clf = lgb.LGBMClassifier(**LGB).fit(
            np.vstack([Xv_[cal], Xr_]), np.r_[np.zeros(len(cal)), np.ones(len(Xr_))])
        sv = clf.predict_proba(Xv_[hold])[:, 1]; sr = clf.predict_proba(Xr_)[:, 1]
        thr = np.quantile(sv, 1 - FAR)
        return sr >= thr, float((sv >= thr).mean() * 100), auroc(clf.predict_proba(Xv_)[:, 1], clf.predict_proba(Xr_)[:, 1])

    # NF-17: add flow score as extra LightGBM feature
    with torch.no_grad():
        s_val_all = (-flow.log_prob(Zva_d)).cpu().numpy()
    Xv17 = np.column_stack([Xv, s_val_all]); Xr17 = np.column_stack([Xr, s_rare])

    def per_object(det):
        ob = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(det)):
            ob[robj[i]][1] = rcls[i]
            if det[i]: ob[robj[i]][0] = True
        d = sum(1 for v in ob.values() if v[0])
        return d, len(ob), ob

    print("\n================ RESULTS (spec=%s) ================" % args.spec)
    res = {}
    det_nf, fp_nf, _ = honest_detect(s_val, s_rare, rng)
    a_nf = auroc(s_val, s_rare)
    d, n, ob_nf = per_object(det_nf)
    print(f"NF flow (UNSUPERVISED) : AUROC={a_nf:.4f}  det/spec={det_nf.mean()*100:.1f}%  "
          f"det/obj={d}/{n}={d/n*100:.1f}%  (holdout FP={fp_nf:.1f}%)")
    res["nf"] = {"auroc": a_nf, "det_spec": det_nf.mean(), "det_obj": d / n, "fp": fp_nf}

    det16, fp16, a16 = lgb_scores(Xv, Xr)
    d, n, _ = per_object(det16)
    print(f"LightGBM-16 (baseline) : AUROC={a16:.4f}  det/spec={det16.mean()*100:.1f}%  "
          f"det/obj={d}/{n}={d/n*100:.1f}%  (holdout FP={fp16:.1f}%)")
    res["lgb16"] = {"auroc": a16, "det_spec": det16.mean(), "det_obj": d / n, "fp": fp16}

    det17, fp17, a17 = lgb_scores(Xv17, Xr17)
    d, n, _ = per_object(det17)
    print(f"LightGBM-17 (16 + NF)  : AUROC={a17:.4f}  det/spec={det17.mean()*100:.1f}%  "
          f"det/obj={d}/{n}={d/n*100:.1f}%  (holdout FP={fp17:.1f}%)")
    res["lgb17"] = {"auroc": a17, "det_spec": det17.mean(), "det_obj": d / n, "fp": fp17}

    print("\nNF per-class detection (per object):")
    byc = collections.defaultdict(lambda: [0, 0])
    for det, c in ob_nf.values():
        byc[c][1] += 1; byc[c][0] += det
    for c, (dd, nn) in sorted(byc.items(), key=lambda x: x[1][0] / max(x[1][1], 1)):
        print(f"    {str(c):22s} {dd:>2d}/{nn:<2d} = {dd/nn*100:3.0f}%")

    out = REPO / "results" / args.spec / "nf_anomaly"; out.mkdir(parents=True, exist_ok=True)
    (out / "nf_results.json").write_text(json.dumps(res, indent=2))
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "val_nll": best_nll},
               out / "flow.pt")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
