#!/usr/bin/env python3
"""ASTRANet-Sentinel: multi-seed aggregation — the paper's headline detection table.

Reads the per-seed k-fold score files written by `oe_kfold.py` and reports the
Sentinel operating regimes on the rare evaluation set:
  (i)   anchor      — seed-averaged score at the averaged target-FAR threshold
  (ii)  per-seed    — each seed at its own threshold (mean +/- std, transparency)
  (iii) seed-union  — flag if ANY seed fires (effective FAR ~ n_seeds x target)
with per-class and Tier-1/Tier-2 breakdowns (per spectrum AND per object).

Usage:
  python pipelines/sentinel_ensemble.py --tags v7fix_s42_f5 v7fix_s7_f5 v7fix_s123_f5
Outputs -> results/oe/sentinel_ensemble.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from _common import REPO, tiered_report
from astranet.sentinel import seed_average, seed_union


def per_object(det, obj, cls):
    ob = collections.defaultdict(lambda: [False, "?"])
    for i in range(len(det)):
        ob[obj[i]][1] = cls[i]
        if det[i]:
            ob[obj[i]][0] = True
    d = sum(1 for v in ob.values() if v[0])
    return d, len(ob), ob


def report(name, det, obj, cls):
    d, n, ob = per_object(det, obj, cls)
    print(f"\n[{name}] per-spectrum {det.sum()}/{len(det)} = {det.mean()*100:.1f}%   "
          f"per-object {d}/{n} = {100*d/n:.1f}%")
    byc = collections.defaultdict(lambda: [0, 0])
    for dd, c in ob.values():
        byc[c][1] += 1; byc[c][0] += dd
    for c, (dd, nn) in sorted(byc.items(), key=lambda x: x[1][0] / max(x[1][1], 1)):
        print(f"    {c:20s} {dd:>2d}/{nn:<2d} = {100*dd/nn:3.0f}%")
    tiers = tiered_report((v[0], v[1]) for v in ob.values())
    return {"per_spectrum": float(det.mean()), "per_object": d / n, "tiers": tiers,
            "per_class": {c: v for c, v in byc.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="oe_kfold score tags, e.g. v7fix_s42_f5 v7fix_s7_f5 v7fix_s123_f5")
    ap.add_argument("--out", default=None,
                    help="output json (default results/oe/sentinel_ensemble.json)")
    ap.add_argument("--head", default=None,
                    help="detector head to aggregate (e.g. emb_lgbm); default = the Sentinel combiner "
                         "(scores16_lgbm, stored under the legacy score_<which> keys)")
    ap.add_argument("--which", default="oe", choices=["oe", "base"],
                    help="aggregate OE-fine-tuned scores (default) or the no-OE baseline")
    args = ap.parse_args()
    Z = [np.load(REPO / "results" / "oe" / f"oe_kfold_scores_{t}.npz", allow_pickle=True)
         for t in args.tags]
    cls, obj = Z[0]["cls"], Z[0]["objid"]
    for z in Z:
        assert (z["cls"] == cls).all(), "score files must share the same rare set/order"
    key, tkey = (f"score_{args.which}", f"thr_{args.which}")
    if args.head:
        key, tkey = (f"score_{args.which}_{args.head}", f"thr_{args.which}_{args.head}")
    scores = [z[key] for z in Z]
    thrs = [float(z[tkey].mean()) for z in Z]

    res = {"tags": args.tags, "which": args.which, "head": args.head or "scores16_lgbm"}
    det, _, thr = seed_average(scores, thrs)
    res["anchor"] = report(f"anchor: {len(Z)}-seed score average (thr={thr:.3f})", det, obj, cls)

    per = []
    for t, z in zip(args.tags, Z):
        dz = z[f"det_{args.which}" + (f"_{args.head}" if args.head else "")]                    # per-fold thresholds (as run)
        d, n, _ = per_object(dz, obj, cls)
        per.append({"tag": t, "per_spectrum": float(dz.mean()), "per_object": d / n})
    po = [100 * p["per_object"] for p in per]
    print(f"\n[per-seed] per-object {np.mean(po):.1f}% +/- {np.std(po):.1f}  "
          f"({[round(x,1) for x in po]})")
    res["per_seed"] = per

    detU = seed_union(scores, thrs)
    res["union"] = report(f"seed-union (effective FAR ~{len(Z)}x target)", detU, obj, cls)

    out = Path(args.out) if args.out else REPO / "results" / "oe" / "sentinel_ensemble.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
