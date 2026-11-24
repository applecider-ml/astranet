#!/usr/bin/env python3
"""Unsupervised anomaly detection via VIRTUAL outliers (no real anomaly ever used for training).

Virtual outliers are synthesised from the training embeddings only (Du et al. 2022, VOS): fit a
class-conditional Gaussian per trained class, then keep samples that fall in the low-likelihood
tail of every class (i.e. far from all clusters, but at the density scale of the data).  Detector
heads are then trained on validation-half (ID) vs virtual outliers, thresholded at 1% FAR on the
other validation half, and evaluated ONCE on the rare set — which is never touched before.

Three synthesis families are compared, all label-free w.r.t. anomalies:
  vos_tail     class-conditional Gaussian samples in the lowest-likelihood epsilon tail
  interpolate  convex mixes of embeddings from DIFFERENT classes (between-cluster space)
  shell        ID points pushed radially outward from their class mean (out-of-cluster shell)

Usage: python pipelines/virtual_outliers.py --spec v7fixmd6
Outputs -> results/<spec>/anomaly/virtual_outliers.{csv,md}
"""
from __future__ import annotations
import argparse, collections, csv
from pathlib import Path
import numpy as np
import torch
from scipy.stats import multivariate_normal

from _common import REPO, resolve_path, load_config, make_astra_config, load_cache, tiered_report
from astranet.model import build_hierarchy_mappings
from astranet.sentinel import luna_dict, detector_heads

FAR = 0.01


def fit_class_gaussians(E, y, shrink=0.1):
    out = {}
    for c in np.unique(y):
        X = E[y == c]; mu = X.mean(0); S = np.cov(X.T) + shrink * np.eye(E.shape[1]) * np.trace(np.cov(X.T)) / E.shape[1]
        out[c] = (mu, S)
    return out


def synth_vos(E, y, n, rng, eps=0.02, oversample=40):
    """Sample from each class Gaussian, keep the eps-tail (lowest log-likelihood under ITS class),
    then drop anything that is more likely under some other class than the ID data is."""
    G = fit_class_gaussians(E, y); keep = []
    per = n // len(G) + 1
    for c, (mu, S) in G.items():
        Z = rng.multivariate_normal(mu, S, size=per * oversample)
        ll = multivariate_normal(mu, S, allow_singular=True).logpdf(Z)
        keep.append(Z[np.argsort(ll)[: int(eps * len(Z))]][:per])
    return np.vstack(keep)[:n].astype(np.float32)


def synth_interpolate(E, y, n, rng):
    i = rng.integers(0, len(E), n); j = rng.integers(0, len(E), n)
    m = y[i] != y[j]; i, j = i[m], j[m]
    lam = rng.uniform(0.35, 0.65, size=len(i))[:, None]
    return (lam * E[i] + (1 - lam) * E[j]).astype(np.float32)


def synth_shell(E, y, n, rng, factor=(1.8, 3.0)):
    G = fit_class_gaussians(E, y); i = rng.integers(0, len(E), n)
    mu = np.stack([G[c][0] for c in y[i]]); d = E[i] - mu
    f = rng.uniform(*factor, size=(n, 1))
    return (mu + f * d).astype(np.float32)


SYNTH = {"vos_tail": synth_vos, "interpolate": synth_interpolate, "shell": synth_shell}


def per_object(det, objid, cls):
    ob = collections.defaultdict(lambda: [False, "?"])
    for k in range(len(cls)):
        ob[objid[k]][1] = cls[k]
        if det[k]: ob[objid[k]][0] = True
    return ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v7fixmd6"); ap.add_argument("--config", default=None)
    ap.add_argument("--n-virtual", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    tr, va, ra = (load_cache(cache, n) for n in ("train", "val", "rare_unknown_combined"))
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu", weights_only=False)
    cls = np.array(rd["class"]); objid = np.array(rd["obj_id"])
    rng = np.random.default_rng(args.seed)
    Etr, ytr = tr["emb"].astype(np.float32), np.asarray(tr["labels"])

    # 16 scores need a LUNA pipeline; virtual outliers only have embeddings, so score-based heads
    # get a "virtual" logits/projection via nearest training neighbour (embedding is what matters)
    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=FAR, fine_to_coarse=f2c); pipe.fit(luna_dict(tr)); M = pipe.active_methods
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(Etr)

    def as_feat(E):  # build a feature-cache-like dict for arbitrary embeddings
        j = nn.kneighbors(E, return_distance=False)[:, 0]
        return {"emb": E, "fine_logits": tr["fine_logits"][j], "coarse_logits": tr["coarse_logits"][j], "proj_fine": tr["proj_fine"][j], "labels": None}

    def S16(f): sc = pipe.score(luna_dict(f)); return np.column_stack([sc[m] for m in M])
    Xv16, Xr16 = S16(va), S16(ra); Ev, Er = va["emb"].astype(np.float32), ra["emb"].astype(np.float32)
    vi = np.random.default_rng(42).permutation(len(Ev)); vtr, vth = vi[:len(vi) // 2], vi[len(vi) // 2:]

    rows = []
    for sname, synth in SYNTH.items():
        V = synth(Etr, ytr, args.n_virtual, rng); V16 = S16(as_feat(V))
        for hname, (kind, make) in detector_heads().items():
            Xv, Xr, XV = {"emb": (Ev, Er, V), "scores": (Xv16, Xr16, V16),
                          "both": (np.hstack([Ev, Xv16]), np.hstack([Er, Xr16]), np.hstack([V, V16]))}[kind]
            clf = make().fit(np.vstack([Xv[vtr], XV]), np.r_[np.zeros(len(vtr)), np.ones(len(XV))])
            pv = clf.predict_proba(Xv[vth])[:, 1]; thr = np.quantile(pv, 1 - FAR)
            det = clf.predict_proba(Xr)[:, 1] >= thr
            ob = per_object(det, objid, cls); od = sum(v[0] for v in ob.values()); tiers = tiered_report((v[0], v[1]) for v in ob.values())
            rows.append({"synthesis": sname, "detector": hname, "spectra_pct": 100 * det.mean(), "objects_pct": 100 * od / len(ob),
                         "tier1_pct": tiers["tier1"]["pct"], "tier2_pct": tiers["tier2"]["pct"], "val_far_pct": 100 * float((pv >= thr).mean())})
            print(f"{sname:12s} {hname:20s} objects {rows[-1]['objects_pct']:5.1f}%  tier1 {rows[-1]['tier1_pct']:5.1f}%  tier2 {rows[-1]['tier2_pct']:5.1f}%", flush=True)
    rows.sort(key=lambda r: -r["objects_pct"])
    out = REPO / "results" / args.spec / "anomaly"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "virtual_outliers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| virtual-outlier synthesis | detector | spectra | objects | Tier 1 | Tier 2 |", "|---|---|---|---|---|---|"]
    md += [f"| {r['synthesis']} | {r['detector']} | {r['spectra_pct']:.1f}% | {r['objects_pct']:.1f}% | {r['tier1_pct']:.1f}% | {r['tier2_pct']:.1f}% |" for r in rows]
    (out / "virtual_outliers.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
