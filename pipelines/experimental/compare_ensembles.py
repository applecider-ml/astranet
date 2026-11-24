#!/usr/bin/env python3
"""Side-by-side comparison of the consistent 6-model vs 7-model ensembles.

Reads the per-spec AD summary, classification metrics, and conformal outputs, and
emits results/compare.csv + results/compare.png. Run after run_anomaly / run_conformal
/ evaluate for both specs.

Usage: python pipelines/compare_ensembles.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import REPO

SPECS = ["6model", "7model"]


def _load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def main():
    rows = []
    strat = {}
    for spec in SPECS:
        base = REPO / "results" / spec
        ad = _load_json(base / "anomaly" / "summary.json")
        cls = _load_json(base / "classification" / "metrics.json")
        uq = _load_json(base / "conformal" / "uq_summary.json")
        row = {"spec": spec,
               "n_models": ad.get("n_models"),
               "ad_auroc": ad.get("ad_auroc"),
               "recovery_at_far": ad.get("recovery_at_far"),
               "union_or": ad.get("union_or"),
               "oof_auroc": uq.get("lgb_auroc"),
               "test_acc": cls.get("test_acc"),
               "test_macro": cls.get("test_macro_recall")}
        rows.append(row)
        csv = base / "conformal" / "ad_strat_coverage.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            g = df.groupby("quintile")["coverage_ad_stratified"].mean()
            strat[spec] = g
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results" / "compare.csv", index=False)
    print(out.to_string(index=False))

    # Figure: bars for headline metrics + AD-stratified coverage lines
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = ["ad_auroc", "oof_auroc", "recovery_at_far", "union_or", "test_macro"]
    x = np.arange(len(metrics)); w = 0.38
    for i, spec in enumerate(SPECS):
        vals = [out.loc[out.spec == spec, m].values[0] or 0 for m in metrics]
        axes[0].bar(x + (i - 0.5) * w, vals, w, label=spec)
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.02); axes[0].set_title("Headline metrics"); axes[0].legend()
    for spec, g in strat.items():
        axes[1].plot(g.index.astype(str), g.values, "o-", label=spec)
    axes[1].axhline(0.95, ls="--", color="grey", label="nominal 0.95")
    axes[1].set_title("AD-stratified Mondrian coverage / quintile")
    axes[1].set_ylim(0.85, 1.0); axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPO / "results" / "compare.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved results/compare.csv and results/compare.png")


if __name__ == "__main__":
    main()
