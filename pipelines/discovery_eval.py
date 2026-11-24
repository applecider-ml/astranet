#!/usr/bin/env python3
"""Discovery-mode evaluation of anomaly detectors: how well does each detector flag anomaly
classes it has NEVER seen?  Two protocols, one encoder state, one 1%-FAR threshold rule.

  unsupervised      no rare data at all: each LUNA score alone (fitted on train, thresholded at
                    1% FAR on a validation half), plus the unsupervised mean-rank of all 16 scores
  leave-class-out   for each rare class C: fit the detector on validation-half (ID) + rare spectra
                    of the OTHER classes, threshold at 1% FAR on the other validation half, count
                    detections on class C (object level: an object is detected if any spectrum is)

Every supervised head from astranet.sentinel.detector_heads() is evaluated under leave-class-out.
The k-fold protocol (pipelines/oe_kfold.py) measures in-distribution recovery of known rare
classes; THIS script measures generalisation to unseen classes, which is the discovery claim.

Usage: python pipelines/discovery_eval.py --spec v7fixmd6 [--cache-dir cache/<other>]
Outputs -> results/<spec>/anomaly/discovery_{unsupervised,leave_class_out}.{csv,md}
"""
from __future__ import annotations
import argparse, collections, csv
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata

from _common import REPO, resolve_path, load_config, make_astra_config, load_cache, tiered_report, RARE_TIER1, RARE_TIER2
from astranet.model import build_hierarchy_mappings
from astranet.sentinel import luna_dict, detector_heads

FAR = 0.01


def per_object(det, objid, cls):
    ob = collections.defaultdict(lambda: [False, "?"])
    for i in range(len(cls)):
        ob[objid[i]][1] = cls[i]
        if det[i]: ob[objid[i]][0] = True
    return ob


def summarize(name, det, objid, cls, val_far):
    ob = per_object(det, objid, cls)
    od = sum(v[0] for v in ob.values()); tiers = tiered_report((v[0], v[1]) for v in ob.values())
    byc = collections.defaultdict(lambda: [0, 0])
    for d, c in ob.values():
        byc[c][1] += 1; byc[c][0] += d
    return {"detector": name, "spectra_pct": 100 * det.mean(), "objects_pct": 100 * od / len(ob),
            "tier1_pct": tiers["tier1"]["pct"], "tier2_pct": tiers["tier2"]["pct"], "val_far_pct": 100 * val_far,
            **{f"obj_{c}": f"{d}/{n}" for c, (d, n) in sorted(byc.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v7fixmd6")
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache-dir", default=None, help="feature cache to use (default cache/<spec>)")
    ap.add_argument("--label", default=None, help="label for output files (default = spec)")
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = Path(args.cache_dir) if args.cache_dir else REPO / "cache" / args.spec
    rare_name = "rare_unknown_combined" if (cache / "rare_unknown_combined.npz").exists() else "rare"
    tr, va, ra = (load_cache(cache, n) for n in ("train", "val", rare_name))
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu", weights_only=False)
    cls = np.array(rd["class"]); objid = np.array(rd["obj_id"]); assert len(cls) == ra["emb"].shape[0]

    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=FAR, fine_to_coarse=f2c)
    pipe.fit(luna_dict(tr)); M = pipe.active_methods
    sv, sr = pipe.score(luna_dict(va)), pipe.score(luna_dict(ra))
    X16 = {"val": np.column_stack([sv[m] for m in M]), "rare": np.column_stack([sr[m] for m in M])}
    XE = {"val": va["emb"].astype(np.float32), "rare": ra["emb"].astype(np.float32)}
    XB = {k: np.hstack([XE[k], X16[k]]) for k in XE}
    rng = np.random.default_rng(42); vi = rng.permutation(len(XE["val"]))
    vtr, vth = vi[:len(vi) // 2], vi[len(vi) // 2:]
    out = REPO / "results" / args.spec / "anomaly"; out.mkdir(parents=True, exist_ok=True)
    label = args.label or args.spec

    # ---- 1. unsupervised: no rare data touched ------------------------------------------------
    rows = []
    for j, m in enumerate(M):
        thr = np.quantile(X16["val"][vth, j], 1 - FAR)          # higher = more anomalous (LUNA convention)
        rows.append(summarize(f"{m} (unsupervised)", X16["rare"][:, j] >= thr, objid, cls,
                              float((X16["val"][vth, j] >= thr).mean())))
    # mean-rank ensemble of the 16 scores, ranks computed against the ID validation distribution
    def mean_rank(X):  # rank each column against the val-half reference, average
        return np.mean([np.searchsorted(np.sort(X16["val"][vtr, j]), X[:, j]) / len(vtr) for j in range(len(M))], axis=0)
    r_val, r_rare = mean_rank(X16["val"][vth]), mean_rank(X16["rare"])
    thr = np.quantile(r_val, 1 - FAR)
    rows.append(summarize("mean-rank of 16 scores (unsupervised)", r_rare >= thr, objid, cls, float((r_val >= thr).mean())))
    rows.sort(key=lambda r: -r["objects_pct"])
    write(out / f"discovery_unsupervised_{label}", rows, "unsupervised (no rare data used anywhere)")

    # ---- 2. leave-class-out for every supervised head ------------------------------------------
    classes = sorted(set(cls)); rows = []
    for name, (kind, make) in detector_heads().items():
        X = {"scores": X16, "emb": XE, "both": XB}[kind]
        det = np.zeros(len(cls), bool); fars = []
        for C in classes:
            test_idx = np.where(cls == C)[0]; expo_idx = np.where(cls != C)[0]
            Xtr = np.vstack([X["val"][vtr], X["rare"][expo_idx]])
            ytr = np.r_[np.zeros(len(vtr)), np.ones(len(expo_idx))]
            clf = make().fit(Xtr, ytr)
            pv = clf.predict_proba(X["val"][vth])[:, 1]; thr = np.quantile(pv, 1 - FAR)
            det[test_idx] = clf.predict_proba(X["rare"][test_idx])[:, 1] >= thr; fars.append(float((pv >= thr).mean()))
        rows.append(summarize(f"{name} (leave-class-out)", det, objid, cls, float(np.mean(fars))))
    rows.sort(key=lambda r: -r["objects_pct"])
    write(out / f"discovery_leave_class_out_{label}", rows, "leave-class-out (detector never saw the tested class)")


def write(stem: Path, rows, title):
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = [f"### {title}", "", "| detector | spectra | objects | Tier 1 | Tier 2 | val FAR |", "|---|---|---|---|---|---|"]
    md += [f"| {r['detector']} | {r['spectra_pct']:.1f}% | {r['objects_pct']:.1f}% | {r['tier1_pct']:.1f}% | "
           f"{r['tier2_pct']:.1f}% | {r['val_far_pct']:.1f}% |" for r in rows]
    stem.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(f"saved -> {stem}.md\n")


if __name__ == "__main__":
    main()
