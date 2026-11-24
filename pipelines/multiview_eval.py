#!/usr/bin/env python3
"""Zero-knowledge novelty detection with luna.multiview on the ASTRAnet data, plus the voting cascade.

  views     A = classifier embedding (all LUNA scorers), B = classifier-free physics descriptors
  combos    union / Fisher / mean rank / vote k=2 across A and B, at a combined 1% FAR
  cascade   k-of-n voting over single scores of ONE representation (the classic cascade:
            typicality, kNN, GMM, within-class Mahalanobis, LOF, MSP on the embedding), at matched 1% FAR
No rare data is used anywhere; 10 random reference/calibration splits of the validation set.
Usage: python pipelines/multiview_eval.py --spec v7fixmd6 --config configs/v7fixmd.yaml
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import numpy as np, torch
from luna.multiview import MultiViewDetector, ViewScorer
from _common import REPO, load_cache, load_config, make_astra_config, resolve_path, tiered_report
from astranet.model import build_hierarchy_mappings
from astranet.sentinel import luna_dict
from physics_view import descriptors

CASCADE = ["typicality", "knn", "gmm", "mahal_within", "lof", "msp"]


def sub(d, idx):
    return {k: (None if v is None else v[idx]) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="v7fixmd6"); ap.add_argument("--config", default=None)
    ap.add_argument("--splits", type=int, default=10); args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg); f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    dd = Path(resolve_path(cfg["data"]["dir"])); TR = torch.load(dd / "train.pt", weights_only=False); VA = torch.load(dd / "val.pt", weights_only=False)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), weights_only=False); cls = np.array(rd["class"]); obj = np.array(list(map(str, rd["obj_id"])))
    l2i = {c: i for i, c in enumerate(ac.class_names)}; ytr = np.array([l2i.get(str(c), -1) for c in TR["y"]]); ok = ytr >= 0
    Dtr, Dva, Dra = descriptors(TR["X"]), descriptors(VA["X"]), descriptors(rd["flux"])
    med, iqr = np.median(Dtr, 0), np.subtract(*np.percentile(Dtr, [75, 25], 0)) + 1e-6
    Z = lambda D: np.clip((D - med) / iqr, -20, 20).astype(np.float32)  # noqa: E731
    c = REPO / "cache" / args.spec; tr, va, ra = (luna_dict(load_cache(c, n)) for n in ("train", "val", "rare_unknown_combined"))
    data = {"train": {"A": tr, "B": {"embeddings": Z(Dtr)[ok], "labels": ytr[ok]}},
            "val": {"A": va, "B": {"embeddings": Z(Dva)}}, "rare": {"A": ra, "B": {"embeddings": Z(Dra)}}}
    for s in CASCADE: data["train"][s], data["val"][s], data["rare"][s] = tr, va, ra

    def objsum(flags):
        o = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(cls)): o[obj[i]][1] = cls[i]; o[obj[i]][0] |= bool(flags[i])
        t = tiered_report((v[0], v[1]) for v in o.values()); return 100 * sum(v[0] for v in o.values()) / len(o), t["tier1"]["pct"], t["tier2"]["pct"]

    dets = {"A alone": MultiViewDetector({"A": ViewScorer(fine_to_coarse=f2c)}),
            "B alone": MultiViewDetector({"B": ViewScorer()})}
    for comb, k in (("union", 1), ("fisher", 1), ("mean_rank", 1), ("vote", 2)):
        dets[f"A,B {comb}" + (f" k={k}" if comb == "vote" else "")] = MultiViewDetector({"A": ViewScorer(fine_to_coarse=f2c), "B": ViewScorer()}, combination=comb, k=k)
    for k in (1, 2, 3, 4):
        dets[f"cascade 6 scores, >= {k} votes"] = MultiViewDetector({s: ViewScorer(scorers=[s], fine_to_coarse=f2c) for s in CASCADE}, combination="vote", k=k)
    for name, det in dets.items():   # fit scorers once on the training split
        for v, view in det.views.items(): view.pipeline.fit(data["train"][v])
    R = collections.defaultdict(list); overlap = []
    for rep in range(args.splits):
        vi = np.random.default_rng(rep).permutation(len(va["embeddings"])); ref, cal = vi[:len(vi) // 2], vi[len(vi) // 2:]
        for name, det in dets.items():
            det.calibrate({v: sub(data["val"][v], ref) for v in det.views}, {v: sub(data["val"][v], cal) for v in det.views})
            fl = det.predict({v: data["rare"][v] for v in det.views})["flags"]
            R[name].append(objsum(fl) + (100 * det.calibration_far,))
        overlap.append(dets["A,B union"].overlap({"A": ra, "B": data["rare"]["B"]})["A|B"])
    out = {k: {"objects": float(np.mean([x[0] for x in v])), "objects_sd": float(np.std([x[0] for x in v])), "tier1": float(np.mean([x[1] for x in v])),
               "tier2": float(np.mean([x[2] for x in v])), "cal_far": float(np.mean([x[3] for x in v]))} for k, v in R.items()}
    out["_overlap_A_B_spectra"] = {k: float(np.mean([o[k] for o in overlap])) for k in overlap[0]}
    (REPO / "results" / args.spec / "anomaly" / "multiview_eval.json").write_text(json.dumps(out, indent=2))
    print(f"{'detector (no rare data, combined 1% FAR, ' + str(args.splits) + ' splits)':50s} {'objects':>13s} {'Tier1':>6s} {'Tier2':>6s}")
    for k, v in out.items():
        if not k.startswith("_"): print(f"{k:50s} {v['objects']:6.1f} ± {v['objects_sd']:3.1f} {v['tier1']:6.1f} {v['tier2']:6.1f}")
    print("overlap A|B (spectrum level):", out["_overlap_A_B_spectra"])


if __name__ == "__main__":
    main()
