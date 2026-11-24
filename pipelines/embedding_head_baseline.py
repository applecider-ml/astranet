#!/usr/bin/env python3
"""Supervised anomaly-detector baselines on ONE consistent encoder state, under the SAME
object-grouped folds and 1%-FAR thresholding as the Sentinel k-fold protocol.

Answers: is the 16-score + gradient-boosting combiner better than (a) a supervised head placed
directly on the 192-d embeddings, (b) a linear combiner of the 16 scores?  All detectors below
see exactly the same data per fold: ID negatives = one half of the validation split, positives =
the exposure folds of the rare set; threshold = 1% FAR on the other validation half; detections
counted on the held-out rare fold only.  Encoder = the deployed ensemble features (no outlier
exposure), so differences are due to the detector alone.

Usage: python pipelines/embedding_head_baseline.py --spec v7fixmd6 [--folds 5]
Outputs -> results/<spec>/anomaly/detector_baselines.{csv,md}
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold

from _common import REPO, resolve_path, load_config, make_astra_config, load_cache, tiered_report
from astranet.model import build_hierarchy_mappings
from astranet.sentinel import LGB_PARAMS, luna_dict

FAR = 0.01


def detectors():
    return {
        "emb192 + logistic":      lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        "emb192 + MLP(64)":       lambda: make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=500, random_state=0, early_stopping=True)),
        "emb192 + LightGBM":      lambda: lgb.LGBMClassifier(**LGB_PARAMS),
        "16 scores + logistic":   lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        "16 scores + LightGBM (Sentinel combiner)": lambda: lgb.LGBMClassifier(**LGB_PARAMS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v7fixmd6")
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    tr, va, ra = (load_cache(cache, n) for n in ("train", "val", "rare_unknown_combined"))
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu", weights_only=False)
    cls = np.array(rd["class"]); objid = np.array(rd["obj_id"])
    assert len(cls) == ra["emb"].shape[0]

    # 16 LUNA scores fitted on train (identical to the Sentinel)
    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=FAR, fine_to_coarse=f2c)
    pipe.fit(luna_dict(tr)); M = pipe.active_methods
    S = lambda f: np.column_stack([pipe.score(luna_dict(f))[m] for m in M])   # noqa: E731
    X16 = {"val": S(va), "rare": S(ra)}
    XE = {"val": va["emb"].astype(np.float32), "rare": ra["emb"].astype(np.float32)}

    skf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=0)
    folds = list(skf.split(cls, cls, groups=objid))
    for e, t in folds:
        assert not (set(objid[e]) & set(objid[t]))
    rng = np.random.default_rng(42)                     # same val-half split as sentinel.detect_fold
    vi = rng.permutation(len(va["emb"])); vtr, vth = vi[:len(vi) // 2], vi[len(vi) // 2:]

    rows = []
    for name, make in detectors().items():
        X = XE if name.startswith("emb192") else X16
        det = np.zeros(len(cls), bool); fars = []
        for expo_idx, test_idx in folds:
            Xtr = np.vstack([X["val"][vtr], X["rare"][expo_idx]])
            ytr = np.r_[np.zeros(len(vtr)), np.ones(len(expo_idx))]
            clf = make().fit(Xtr, ytr)
            thr = np.quantile(clf.predict_proba(X["val"][vth])[:, 1], 1 - FAR)
            det[test_idx] = clf.predict_proba(X["rare"][test_idx])[:, 1] >= thr
            fars.append(float((clf.predict_proba(X["val"][vth])[:, 1] >= thr).mean()))
        ob = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(cls)):
            ob[objid[i]][1] = cls[i]
            if det[i]: ob[objid[i]][0] = True
        od = sum(v[0] for v in ob.values()); tiers = tiered_report((v[0], v[1]) for v in ob.values())
        rows.append({"detector": name, "spectra_pct": 100 * det.mean(), "objects_pct": 100 * od / len(ob),
                     "tier1_pct": tiers["tier1"]["pct"], "tier2_pct": tiers["tier2"]["pct"],
                     "val_far_pct": 100 * float(np.mean(fars))})
        print(f"{name:42s} spectra {rows[-1]['spectra_pct']:5.1f}%  objects {rows[-1]['objects_pct']:5.1f}%  "
              f"tier1 {rows[-1]['tier1_pct']:5.1f}%  tier2 {rows[-1]['tier2_pct']:5.1f}%", flush=True)

    out = REPO / "results" / args.spec / "anomaly"; out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / "detector_baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| detector (same encoder, same object-grouped folds, 1% FAR) | spectra | objects | Tier 1 | Tier 2 |",
          "|---|---|---|---|---|"] + [f"| {r['detector']} | {r['spectra_pct']:.1f}% | {r['objects_pct']:.1f}% | "
                                     f"{r['tier1_pct']:.1f}% | {r['tier2_pct']:.1f}% |" for r in rows]
    (out / "detector_baselines.md").write_text("\n".join(md) + "\n")
    print(f"saved -> {out / 'detector_baselines.md'}")


if __name__ == "__main__":
    main()
