#!/usr/bin/env python3
"""Run the full conformal / UQ chain on a consistent ensemble cache.

Chain (validated to reproduce the paper's §6):
  1. compute_uq_diagnostics : OOF LightGBM scores, id_cal/id_holdout, coverage, APS
  2. (glue)                  : build an OOF-consistent OOD report (rare set)
  3. conformal_novelty       : AD-conditional Mondrian coverage + p_wrong
  4. ad_stratified_mondrian  : AD-stratified Mondrian coverage (recovers 0.95/quintile)
  5. ad_conditioned_vap      : AD-conditioned VAP

Outputs -> results/<spec>/conformal/

Usage: python pipelines/run_conformal.py --spec 6model
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from _common import REPO, load_config, make_astra_config

PY = sys.executable
SCRIPTS = REPO / "astranet" / "uq" / "_scripts"


def _run(args, env_extra=None):
    print("  $", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"step failed: {args[1]}")
    return r.stdout


def _build_ood_report(cache_dir: Path, out_dir: Path, class_names):
    d = np.load(cache_dir / "rare_unknown_combined.npz", allow_pickle=True)
    meta = pickle.load(open(cache_dir / "rare_unknown_combined_meta.pkl", "rb"))
    fl = d["fine_logits"]
    rare_lgb = np.load(out_dir / "rare_lgb_scores.npy")
    assert len(rare_lgb) == len(fl), f"len mismatch {len(rare_lgb)} vs {len(fl)}"
    sm = softmax(fl, axis=1)
    df = pd.DataFrame({
        "obj_id": [m.get("obj_id", f"r{i}") for i, m in enumerate(meta)],
        "true_class": [m.get("class", "?") for m in meta],
        "pred_class": [class_names[p] for p in fl.argmax(1)],
        "confidence": sm.max(1),
        "ens_score": rare_lgb,
    })
    p = out_dir / "ood_report.csv"
    df.to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ac = make_astra_config(cfg)
    class_names = ac.class_names
    cache_dir = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "conformal"
    pmc = cache_dir / "per_model_softmax.npz"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "ad_strat_coverage.csv").exists() and not args.force:
        print(f"[{args.spec}] conformal already done -> {out} (use --force)")
        return

    alpha = str(cfg.get("alpha", 0.05)); nq = str(cfg.get("n_quintiles", 5))
    nf = str(cfg.get("n_folds", 5))

    print(f"[{args.spec}] 1/5 compute_uq_diagnostics")
    _run([PY, SCRIPTS / "compute_uq_diagnostics.py", "--cache-dir", cache_dir,
          "--per-model-cache", pmc, "--out-dir", out])

    print(f"[{args.spec}] 2/5 build OOD report")
    ood = _build_ood_report(cache_dir, out, class_names)

    print(f"[{args.spec}] 3/5 conformal_novelty")
    _run([PY, SCRIPTS / "conformal_novelty.py", "--cal", out / "id_cal.csv",
          "--holdout", out / "id_holdout.csv", "--ood", ood,
          "--alpha", alpha, "--n-quantiles", nq, "--outdir", out])

    print(f"[{args.spec}] 4/5 ad_stratified_mondrian")
    _run([PY, SCRIPTS / "ad_stratified_mondrian.py", "--per-model-cache", pmc,
          "--val-cache", cache_dir / "val.npz", "--val-ad", out / "val_lgb_scores.npy",
          "--anomaly-report", ood, "--outdir", out, "--alpha", alpha, "--n-folds", nf])

    print(f"[{args.spec}] 5/5 ad_conditioned_vap")
    _run([PY, SCRIPTS / "ad_conditioned_vap.py", "--per-model-cache", pmc,
          "--val-cache", cache_dir / "val.npz", "--val-ad", out / "val_lgb_scores.npy",
          "--anomaly-report", ood, "--outdir", out, "--n-folds", nf])

    print(f"\n[{args.spec}] conformal chain complete -> {out}")


if __name__ == "__main__":
    main()
