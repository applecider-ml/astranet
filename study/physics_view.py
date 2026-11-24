#!/usr/bin/env python3
"""A second, classifier-free unsupervised view: physically motivated spectral descriptors, scored
with LUNA's density/distance scorers, and combined with the classifier-embedding view.

Descriptors (per spectrum, observer frame, robust-scaled flux; no redshift needed):
  continuum   flux ratios of 6 broad bands to the mean, and the slope of a smoothed continuum
  texture     residual power (flux minus a 60-pixel smoothed continuum) at 5 velocity scales
              (difference-of-Gaussians bands), i.e. how much narrow vs broad line structure there is
  asymmetry   emission-minus-absorption balance of the residual at 3 scales (skewness-like)
  windows     pseudo-equivalent widths in 8 observer-frame windows around strong transient lines
              (Ca II H&K, Hbeta/[OIII], He I 5876/Na D, Si II 6355, Halpha, O I 7774, Ca II NIR)
No noise-level feature is used, so that low signal-to-noise alone does not make a spectrum anomalous.

Both views are thresholded on the in-distribution validation split only (no rare data anywhere);
combinations (union, Fisher's method on per-view rank p-values) are thresholded so that the
COMBINED false-alarm rate on a held-out validation half is 1%.  Repeated over 10 random splits.

Usage: python pipelines/physics_view.py --spec tcn6 --config configs/spectranet.yaml
Outputs -> results/<spec>/anomaly/physics_view.{json,md}
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[0])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
import argparse, collections, json
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from _common import REPO, load_cache, load_config, make_astra_config, resolve_path, tiered_report
from astranet.model import build_hierarchy_mappings
from astranet.novelty import luna_dict

WL = np.linspace(3850, 9000, 4096)
WINDOWS = [(3850, 4050), (4800, 5100), (5750, 6050), (6050, 6400), (6450, 6900), (7600, 7900), (8300, 8800), (4300, 4700)]


def norm(X):
    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    p5, p95 = np.percentile(X, 5, 1, keepdims=True), np.percentile(X, 95, 1, keepdims=True)
    return np.clip((X - p5) / (p95 - p5 + 1e-6), -2, 3)


def descriptors(X):
    F = norm(X); cont = gaussian_filter1d(F, 60, axis=1); res = F - cont
    feats = []
    bands = np.array_split(np.arange(F.shape[1]), 6); m = cont.mean(1, keepdims=True) + 1e-3
    feats += [cont[:, b].mean(1) / m[:, 0] for b in bands]
    x = (WL - WL.mean()) / WL.std(); feats.append((cont * x).mean(1) / m[:, 0])
    prev = F
    for sig in (2, 5, 12, 30, 60):
        sm = gaussian_filter1d(F, sig, axis=1); band = prev - sm; prev = sm
        feats.append(np.log(band.std(1) + 1e-4))
    for sig in (3, 10, 30):
        r = res - gaussian_filter1d(res, sig * 4, axis=1); r = gaussian_filter1d(r, sig, axis=1)
        feats.append(r.clip(0).mean(1) - (-r).clip(0).mean(1))
    for lo, hi in WINDOWS:
        w = (WL >= lo) & (WL < hi); feats.append(res[:, w].mean(1) / m[:, 0])
    return np.column_stack(feats).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="astranet12"); ap.add_argument("--config", default=None)
    ap.add_argument("--splits", type=int, default=10); args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg); f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    dd = Path(resolve_path(cfg["data"]["dir"])); TR = torch.load(dd / "train.pt", weights_only=False); VA = torch.load(dd / "val.pt", weights_only=False)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), weights_only=False); cls = np.array(rd["class"]); obj = np.array(list(map(str, rd["obj_id"])))
    inst_val = np.array([m.get("instrument", "?") for m in VA["meta"]])
    l2i = {c: i for i, c in enumerate(ac.class_names)}; ytr = np.array([l2i.get(str(c), -1) for c in TR["y"]])
    Dtr, Dva, Dra = descriptors(TR["X"]), descriptors(VA["X"]), descriptors(rd["flux"])
    med, iqr = np.median(Dtr, 0), np.subtract(*np.percentile(Dtr, [75, 25], 0)) + 1e-6
    Z = lambda D: np.clip((D - med) / iqr, -20, 20)  # noqa: E731
    from luna import LUNAPipeline
    # view B: LUNA on physics descriptors (label-free + class-conditional scorers)
    ok = ytr >= 0
    pB = LUNAPipeline(combiner="lightgbm", target_far=0.01, fine_to_coarse=None); pB.fit({"embeddings": Z(Dtr)[ok], "labels": ytr[ok]}); MB = pB.active_methods
    svB, srB = pB.score({"embeddings": Z(Dva)}), pB.score({"embeddings": Z(Dra)})
    # view A: LUNA on the classifier embedding (the paper's zero-knowledge score)
    d = REPO / "cache" / args.spec; tr, va, ra = (load_cache(d, n) for n in ("train", "val", "rare_unknown_combined"))
    pA = LUNAPipeline(combiner="lightgbm", target_far=0.01, fine_to_coarse=f2c); pA.fit(luna_dict(tr)); MA = pA.active_methods
    svA, srA = pA.score(luna_dict(va)), pA.score(luna_dict(ra))

    def objsum(det):
        o = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(cls)): o[obj[i]][1] = cls[i]; o[obj[i]][0] |= bool(det[i])
        t = tiered_report((v[0], v[1]) for v in o.values())
        return 100 * sum(v[0] for v in o.values()) / len(o), t["tier1"]["pct"], t["tier2"]["pct"], {k for k, v in o.items() if v[0]}

    R = collections.defaultdict(list); overlap = []; corr = []; inst = []
    for rep in range(args.splits):
        vi = np.random.default_rng(rep).permutation(len(Dva)); ref, thr = vi[:len(vi) // 2], vi[len(vi) // 2:]
        rk = lambda a, x: np.searchsorted(np.sort(a), x) / len(a)  # noqa: E731
        A_v = np.mean([rk(svA[m][ref], svA[m]) for m in MA], 0); A_r = np.mean([rk(svA[m][ref], srA[m]) for m in MA], 0)
        B_v = np.mean([rk(svB[m][ref], svB[m]) for m in MB], 0); B_r = np.mean([rk(svB[m][ref], srB[m]) for m in MB], 0)
        # per-view p-values against the reference half (higher score = more anomalous)
        pv = lambda refv, x: (1 + (np.sort(refv)[None, :] >= x[:, None]).sum(1)) / (len(refv) + 1)  # noqa: E731
        pA_v, pA_r, pB_v, pB_r = pv(A_v[ref], A_v), pv(A_v[ref], A_r), pv(B_v[ref], B_v), pv(B_v[ref], B_r)
        comb = {"A: classifier embedding": (A_v, A_r), "B: physics descriptors": (B_v, B_r),
                "Fisher(A,B)": (-2 * (np.log(pA_v) + np.log(pB_v)), -2 * (np.log(pA_r) + np.log(pB_r)))}
        sets = {}
        for name, (v, r) in comb.items():
            t = np.quantile(v[thr], 0.99); det = r >= t; o, t1, t2, S = objsum(det); sets[name] = S
            R[name].append((o, t1, t2, 100 * (v[thr] >= t).mean()))
            if name.startswith("B") and rep == 0:
                fv = v >= t; inst.append({i: round(100 * fv[thr][inst_val[thr] == i].mean(), 1) for i in ("SEDM", "DBSP", "LRIS")})
        # union at matched combined FAR
        best = None
        for q in np.linspace(0.990, 0.9995, 40):
            tA, tB = np.quantile(A_v[ref], q), np.quantile(B_v[ref], q)
            far = ((A_v[thr] >= tA) | (B_v[thr] >= tB)).mean()
            if far <= 0.0105 and (best is None or q < best[0]): best = (q, (A_r >= tA) | (B_r >= tB), far)
        o, t1, t2, _ = objsum(best[1]); R["union(A,B) @ matched FAR"].append((o, t1, t2, 100 * best[2]))
        a, b = sets["A: classifier embedding"], sets["B: physics descriptors"]
        overlap.append((len(a & b), len(a - b), len(b - a), len(a & b) / max(len(a | b), 1))); corr.append(spearmanr(A_r, B_r)[0])
    out = {k: {"objects_mean": float(np.mean([x[0] for x in v])), "objects_sd": float(np.std([x[0] for x in v])),
               "tier1": float(np.mean([x[1] for x in v])), "tier2": float(np.mean([x[2] for x in v])), "held_out_far": float(np.mean([x[3] for x in v]))}
           for k, v in R.items()}
    for k in ("Fisher(A,B)", "union(A,B) @ matched FAR"):
        dA = np.array(R[k]) - np.array(R["A: classifier embedding"])
        out_key = f"_paired_gain_{k}"
        globals().setdefault("PAIRED", {})[out_key] = {"objects_gain_mean": float(dA[:, 0].mean()), "objects_gain_min": float(dA[:, 0].min()),
                                                     "objects_splits_improved": int((dA[:, 0] > 0).sum()), "objects_splits_worse": int((dA[:, 0] < 0).sum()),
                                                     "tier1_gain_mean": float(dA[:, 1].mean()), "tier1_splits_improved": int((dA[:, 1] > 0).sum()),
                                                     "tier1_splits_worse": int((dA[:, 1] < 0).sum())}
    ov = np.array(overlap)
    out["_overlap_A_B"] = {"both": float(ov[:, 0].mean()), "only_A": float(ov[:, 1].mean()), "only_B": float(ov[:, 2].mean()), "jaccard": float(ov[:, 3].mean()), "spearman": float(np.mean(corr))}
    out["_B_val_flag_rate_by_instrument"] = inst[0]
    out.update(PAIRED)
    res = REPO / "results" / args.spec / "anomaly"; (res / "physics_view.json").write_text(json.dumps(out, indent=2))
    print(f"{'detector (no rare data anywhere)':34s} {'objects':>13s} {'Tier1':>6s} {'Tier2':>6s} {'held-out FAR':>13s}   ({args.splits} splits)")
    for k, v in out.items():
        if k.startswith("_"): continue
        print(f"{k:34s} {v['objects_mean']:6.1f} ± {v['objects_sd']:3.1f} {v['tier1']:6.1f} {v['tier2']:6.1f} {v['held_out_far']:10.2f}%")
    print("overlap A vs B (objects):", out["_overlap_A_B"]); print("B val flag rate by instrument:", out["_B_val_flag_rate_by_instrument"])
    for k, v in PAIRED.items(): print(k, v)


if __name__ == "__main__":
    main()
