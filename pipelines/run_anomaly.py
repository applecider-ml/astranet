#!/usr/bin/env python3
"""Anomaly detection on a consistent ensemble cache, using the LUNA library.

Fits LUNA's 16 scorers on train, trains the LightGBM combiner on val (ID) vs rare
(OOD) at 1% FAR, then writes the LUNA report (anomaly_report.csv, category_ablation,
detection/complementarity maps) plus extra plots (ROC/PR/UMAP/AUROC-bar) and a
metrics summary for the 6-vs-7 comparison.

Outputs -> results/<spec>/anomaly/

Usage: python pipelines/run_anomaly.py --spec 6model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from _common import REPO, load_config, make_astra_config, load_cache
from astranet.model import build_hierarchy_mappings

import luna.viz as lviz                       # extended below
from luna import LUNAPipeline
from luna.thresholding import calibrate_threshold


def _to_luna(d: dict) -> dict:
    """Map our cache dict -> LUNA input dict."""
    out = {"embeddings": d["emb"].astype(np.float32),
           "logits": d["fine_logits"].astype(np.float32),
           "coarse_logits": d["coarse_logits"].astype(np.float32),
           "projections": d["proj_fine"].astype(np.float32)}
    if d.get("labels") is not None:
        out["labels"] = np.asarray(d["labels"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ac = make_astra_config(cfg)
    class_names = ac.class_names
    fine_to_coarse, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)

    cache_dir = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "anomaly"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not args.force:
        print(f"[{args.spec}] anomaly already done -> {out} (use --force)")
        return

    train = load_cache(cache_dir, "train")
    val = load_cache(cache_dir, "val")
    rare = load_cache(cache_dir, "rare_unknown_combined")
    train_l, val_l, rare_l = _to_luna(train), _to_luna(val), _to_luna(rare)
    rare_meta = rare["meta"]

    pipe = LUNAPipeline(combiner=cfg.get("combiner", "lightgbm"),
                        target_far=cfg.get("target_far", 0.01),
                        fine_to_coarse=fine_to_coarse)
    print(f"[{args.spec}] fitting scorers ...")
    pipe.fit(train_l)
    print(f"  active scorers: {pipe.active_methods}")
    print(f"[{args.spec}] fitting combiner (val vs rare) @ FAR={pipe.target_far} ...")
    pipe.fit_combiner(val_l, rare_l, verbose=True)

    # Metrics
    val_pred = pipe.predict(val_l)
    rare_pred = pipe.predict(rare_l)
    y = np.r_[np.zeros(len(val_pred["ensemble_score"])), np.ones(len(rare_pred["ensemble_score"]))]
    s = np.r_[val_pred["ensemble_score"], rare_pred["ensemble_score"]]
    auroc = float(roc_auc_score(y, s))
    rec_far = float(rare_pred["flagged"].mean())          # recovery at target FAR
    fp_far = float(val_pred["flagged"].mean())

    # Union over per-method flags (notebook "Union-OR" analogue)
    id_scores = pipe.score(val_l)
    ood_scores = rare_pred["scores"]
    union = np.zeros(len(rare_meta), dtype=bool)
    for m in pipe.active_methods:
        thr = calibrate_threshold(id_scores[m], pipe.target_far)
        union |= (ood_scores[m] > thr)
    union_or = float(union.mean())

    # LUNA report (anomaly_report.csv, category_ablation, detection/complementarity maps)
    print(f"[{args.spec}] writing LUNA report + plots ...")
    pipe.report(id_data=val_l, ood_data=rare_l, out_dir=out,
                class_names=class_names, meta=rare_meta)

    # Cache the arrays needed for plotting so figures can be restyled cheaply
    # (make_figures.py) WITHOUT re-running the expensive scoring.
    fam = pipe.family_analysis(val_l, rare_l)
    rng = np.random.default_rng(42)
    tr_emb = train_l["embeddings"]
    sub = rng.choice(len(tr_emb), min(4000, len(tr_emb)), replace=False)
    np.savez(out / "_plotdata.npz",
             val_ens=val_pred["ensemble_score"], rare_ens=rare_pred["ensemble_score"],
             fam_names=np.array(list(fam)),
             fam_auroc=np.array([fam[f]["auroc"] for f in fam]),
             train_emb=tr_emb[sub], train_labels=np.asarray(train["labels"])[sub],
             rare_emb=rare_l["embeddings"], class_names=np.array(class_names))

    # Extra plots (ROC / PR / AUROC-bar / UMAP) — light + dark, paper fonts
    lviz.plot_roc_pr(val_pred["ensemble_score"], rare_pred["ensemble_score"], out)
    lviz.plot_family_auroc_bar(fam, out)
    lviz.plot_umap(tr_emb[sub], rare_l["embeddings"],
                   np.asarray(train["labels"])[sub], class_names, out)

    from _common import ENSEMBLE_SPECS
    n_models = sum(len(t) for t in ENSEMBLE_SPECS[args.spec].values())
    summary = {"spec": args.spec, "n_models": n_models,
               "ad_auroc": auroc, "recovery_at_far": rec_far, "fp_at_far": fp_far,
               "union_or": union_or, "target_far": pipe.target_far,
               "active_scorers": pipe.active_methods}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[{args.spec}] AD done. AUROC={auroc:.4f}  rec@{pipe.target_far:.0%}FAR={rec_far:.1%}  "
          f"Union-OR={union_or:.1%}  -> {out}")


if __name__ == "__main__":
    main()
