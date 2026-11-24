#!/usr/bin/env python3
"""Leave-class-out (discovery) with the classifier-free physics view added.
For each rare class C: detectors are trained on validation-half (ID) vs rare spectra of all OTHER
classes, thresholded at 1% FAR on the other validation half, and tested on C (object level).
  heads   embedding / physics descriptors / embedding+physics / embedding+16 scores+physics (LightGBM)
  union   the embedding head OR the zero-knowledge physics view, at a COMBINED 1% FAR
Usage: python pipelines/lco_multiview.py --spec v7fixmd6 --config configs/v7fixmd.yaml
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import numpy as np, torch, lightgbm as lgb
from _common import REPO, load_cache, load_config, make_astra_config, resolve_path, tiered_report
from astranet.model import build_hierarchy_mappings
from astranet.sentinel import luna_dict, LGB_PARAMS
from physics_view import descriptors


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="v7fixmd6"); ap.add_argument("--config", default=None); args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg); f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    dd = Path(resolve_path(cfg["data"]["dir"])); TR = torch.load(dd / "train.pt", weights_only=False); VA = torch.load(dd / "val.pt", weights_only=False)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), weights_only=False); cls = np.array(rd["class"]); obj = np.array(list(map(str, rd["obj_id"])))
    l2i = {c: i for i, c in enumerate(ac.class_names)}; ytr = np.array([l2i.get(str(c), -1) for c in TR["y"]]); ok = ytr >= 0
    Dtr, Dva, Dra = descriptors(TR["X"]), descriptors(VA["X"]), descriptors(rd["flux"])
    med, iqr = np.median(Dtr, 0), np.subtract(*np.percentile(Dtr, [75, 25], 0)) + 1e-6
    Z = lambda D: np.clip((D - med) / iqr, -20, 20).astype(np.float32)  # noqa: E731
    c = REPO / "cache" / args.spec; tr, va, ra = (load_cache(c, n) for n in ("train", "val", "rare_unknown_combined"))
    from luna import LUNAPipeline
    pA = LUNAPipeline(combiner="lightgbm", target_far=0.01, fine_to_coarse=f2c); pA.fit(luna_dict(tr)); MA = pA.active_methods
    sA_v, sA_r = pA.score(luna_dict(va)), pA.score(luna_dict(ra))
    S16 = {"val": np.column_stack([sA_v[m] for m in MA]), "rare": np.column_stack([sA_r[m] for m in MA])}
    pB = LUNAPipeline(combiner="lightgbm", target_far=0.01); pB.fit({"embeddings": Z(Dtr)[ok], "labels": ytr[ok]}); MB = pB.active_methods
    sB_v, sB_r = pB.score({"embeddings": Z(Dva)}), pB.score({"embeddings": Z(Dra)})
    E = {"val": va["emb"].astype(np.float32), "rare": ra["emb"].astype(np.float32)}; P = {"val": Z(Dva), "rare": Z(Dra)}
    feats = {"embedding head": E, "physics head": P,
             "embedding+physics head": {k: np.hstack([E[k], P[k]]) for k in E},
             "embedding+16 scores+physics head": {k: np.hstack([E[k], S16[k], P[k]]) for k in E}}
    vi = np.random.default_rng(42).permutation(len(E["val"])); vtr, vth = vi[:len(vi) // 2], vi[len(vi) // 2:]
    rk = lambda a, x: np.searchsorted(np.sort(a), x) / len(a)  # noqa: E731
    phys_v = np.mean([rk(sB_v[m][vtr], sB_v[m]) for m in MB], 0); phys_r = np.mean([rk(sB_v[m][vtr], sB_r[m]) for m in MB], 0)
    classes = sorted(set(cls))
    scores = {k: np.zeros(len(cls)) for k in feats}; vscores = {k: {} for k in feats}; thr = {k: {} for k in feats}
    for C in classes:
        te = np.where(cls == C)[0]; ex = np.where(cls != C)[0]
        for k, X in feats.items():
            clf = lgb.LGBMClassifier(**LGB_PARAMS).fit(np.vstack([X["val"][vtr], X["rare"][ex]]), np.r_[np.zeros(len(vtr)), np.ones(len(ex))])
            scores[k][te] = clf.predict_proba(X["rare"][te])[:, 1]; vscores[k][C] = clf.predict_proba(X["val"][vth])[:, 1]

    def objsum(det):
        o = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(cls)): o[obj[i]][1] = cls[i]; o[obj[i]][0] |= bool(det[i])
        t = tiered_report((v[0], v[1]) for v in o.values()); return 100 * sum(v[0] for v in o.values()) / len(o), t["tier1"]["pct"], t["tier2"]["pct"]
    out = {}
    for k in feats:   # per-class threshold at 1% FAR on the held-out validation half
        det = np.zeros(len(cls), bool)
        for C in classes:
            te = cls == C; det[te] = scores[k][te] >= np.quantile(vscores[k][C], 0.99)
        out[k] = objsum(det)
    for k in ("embedding head", "embedding+physics head"):   # union with the zero-knowledge physics view at combined 1% FAR
        det = np.zeros(len(cls), bool)
        for C in classes:
            te = cls == C; hv = vscores[k][C]; pv = phys_v[vth]
            best = None
            for q in np.linspace(0.990, 0.9998, 60):
                th, tp = np.quantile(hv, q), np.quantile(pv, q)
                if ((hv >= th) | (pv >= tp)).mean() <= 0.0105 and (best is None or q < best[0]): best = (q, th, tp)
            _, th, tp = best; det[te] = (scores[k][te] >= th) | (phys_r[te] >= tp)
        out[k + " OR physics view"] = objsum(det)
    print(f"{'leave-class-out detector (1% FAR)':48s} objects  Tier1  Tier2")
    for k, (o, t1, t2) in out.items(): print(f"  {k:46s} {o:5.1f}  {t1:5.1f}  {t2:5.1f}")
    (REPO / "results" / args.spec / "anomaly" / "lco_multiview.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
