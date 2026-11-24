#!/usr/bin/env python3
"""Regenerate all figures (light + dark, paper fonts) from cached arrays.

Cheap: uses results/<spec>/anomaly/_plotdata.npz (saved by run_anomaly) and
results/<spec>/classification/metrics.json — no model loading or re-scoring.
Use this to restyle figures (colours/fonts/themes) without re-running the pipeline.

Usage: python pipelines/make_figures.py --spec 6model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import REPO
import luna.viz as lviz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    base = REPO / "results" / args.spec

    # Anomaly figures from _plotdata.npz
    pd_path = base / "anomaly" / "_plotdata.npz"
    if pd_path.exists():
        d = np.load(pd_path, allow_pickle=True)
        class_names = [str(x) for x in d["class_names"]]
        out = base / "anomaly"
        lviz.plot_roc_pr(d["val_ens"], d["rare_ens"], out)
        fam = {str(n): {"auroc": float(a)} for n, a in zip(d["fam_names"], d["fam_auroc"])}
        lviz.plot_family_auroc_bar(fam, out)
        lviz.plot_umap(d["train_emb"], d["rare_emb"], d["train_labels"], class_names, out)
        print(f"[{args.spec}] anomaly figures regenerated -> {out}")
    else:
        print(f"[{args.spec}] no _plotdata.npz (run run_anomaly first)")

    # Classification confusion matrix from metrics.json
    mp = base / "classification" / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        cm = np.array(m["confusion_matrix"])
        lviz.plot_confusion(cm, m["class_names"], f"ASTRAnet ({args.spec}, consistent)",
                            base / "classification", stem="confusion_test")
        print(f"[{args.spec}] confusion matrix regenerated")


if __name__ == "__main__":
    main()
