#!/usr/bin/env python3
"""Recovery-vs-FAR strategy analysis for the anomaly layer.

Evaluates several detection strategies and reports rare-set recovery and the
realised false-alarm rate (FAR) for each, so we can see what reaches a target
recovery (e.g. 80%) at a controlled FAR:

  - anchor        : the LightGBM ensemble at a single global FAR threshold
  - free-OR       : anchor  OR  per-scorer "free" detections (rare scoring above the
                    scorer's ID max -> zero added FP in-sample) — the notebook's
                    "anchor + 0-FP secondaries" union
  - voting        : flag if >= k of the 16 scorers fire at their own per-FAR threshold
  - class-cond    : per predicted-class ensemble threshold

FAR is reported honestly on a HELD-OUT half of val (thresholds set on the other half),
so "free" detections that only look free in-sample still show their true FAR.

Outputs -> results/<spec>/anomaly/recovery_strategies.csv

Usage: python pipelines/recovery_strategies.py --spec 6model --target-far 0.01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO, load_config, make_astra_config, load_cache
from astranet.model import build_hierarchy_mappings
from luna import LUNAPipeline


def _to_luna(d):
    o = {"embeddings": d["emb"].astype(np.float32),
         "logits": d["fine_logits"].astype(np.float32),
         "coarse_logits": d["coarse_logits"].astype(np.float32),
         "projections": d["proj_fine"].astype(np.float32)}
    if d.get("labels") is not None:
        o["labels"] = np.asarray(d["labels"])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    far = args.target_far
    cfg = load_config(args.config)
    ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "anomaly"
    out.mkdir(parents=True, exist_ok=True)

    train = load_cache(cache, "train"); val = load_cache(cache, "val")
    rare = load_cache(cache, "rare_unknown_combined")
    tl, vl, rl = _to_luna(train), _to_luna(val), _to_luna(rare)

    pipe = LUNAPipeline(combiner=cfg.get("combiner", "lightgbm"),
                        target_far=far, fine_to_coarse=f2c)
    print(f"[{args.spec}] fitting scorers + combiner ...", flush=True)
    pipe.fit(tl); pipe.fit_combiner(vl, rl, verbose=False)
    methods = pipe.active_methods

    # Per-method scores (higher = more anomalous, LUNA convention)
    v_s = pipe.score(vl); r_s = pipe.score(rl)
    v_ens = pipe.predict(vl)["ensemble_score"]; r_ens = pipe.predict(rl)["ensemble_score"]

    # Held-out split of val for honest FAR (calibrate on half A, measure FP on half B)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(v_ens)); A, B = idx[: len(idx)//2], idx[len(idx)//2:]
    nrare = len(r_ens)

    def far_recovery(rare_flag, val_flag_B):
        return rare_flag.mean()*100, val_flag_B.mean()*100

    rows = []

    # 1. anchor: ensemble at global FAR (threshold on A, FP on B)
    thr = np.quantile(v_ens[A], 1-far)
    rec, fp = far_recovery(r_ens >= thr, v_ens[B] >= thr)
    anchor_flag = r_ens >= thr
    rows.append(("anchor: ensemble @%.0f%%FAR" % (far*100), rec, fp))

    # 2. free-OR: anchor OR per-scorer free detections (rare > scorer ID-max on A)
    free = np.zeros(nrare, bool); free_val_B = np.zeros(len(B), bool)
    for m in methods:
        m_max = v_s[m][A].max()
        free |= r_s[m] > m_max
        free_val_B |= v_s[m][B] > m_max
    u = anchor_flag | free
    rec, fp = far_recovery(u, (v_ens[B] >= thr) | free_val_B)
    rows.append(("anchor @%.0f%% OR free(0-FP secondaries)" % (far*100), rec, fp))

    # 3. voting: >= k scorers fire at their own per-FAR threshold
    thr_m = {m: np.quantile(v_s[m][A], 1-far) for m in methods}
    rare_votes = np.sum([(r_s[m] >= thr_m[m]) for m in methods], axis=0)
    valB_votes = np.sum([(v_s[m][B] >= thr_m[m]) for m in methods], axis=0)
    for k in (2, 3, 4):
        rec, fp = far_recovery(rare_votes >= k, valB_votes >= k)
        rows.append((f"voting: >={k} of {len(methods)} scorers @{far*100:.0f}%FAR", rec, fp))

    # 4. union of all scorers each @ FAR (loose; higher FAR)
    rare_any = rare_votes >= 1; valB_any = valB_votes >= 1
    rec, fp = far_recovery(rare_any, valB_any)
    rows.append(("union: >=1 of %d scorers @%.0f%%FAR" % (len(methods), far*100), rec, fp))

    # 5. class-conditional ensemble threshold (per predicted class)
    vpred = val["fine_logits"].argmax(1); rpred = rare["fine_logits"].argmax(1)
    flagR = np.zeros(nrare, bool); flagB = np.zeros(len(B), bool)
    vpredA, vpredB = vpred[A], vpred[B]
    for c in range(ac.num_classes):
        sc = v_ens[A][vpredA == c]
        tc = np.quantile(sc, 1-far) if len(sc) >= 20 else thr
        flagR |= (rpred == c) & (r_ens >= tc)
        flagB |= (vpredB == c) & (v_ens[B] >= tc)
    rec, fp = far_recovery(flagR, flagB)
    rows.append(("class-conditional ensemble @%.0f%%/class" % (far*100), rec, fp))

    # 6. EACH individual scorer at the global FAR (threshold on A, FP on B)
    per_scorer = []
    for m in methods:
        tm = np.quantile(v_s[m][A], 1 - far)
        rec = (r_s[m] >= tm).mean() * 100
        fp = (v_s[m][B] >= tm).mean() * 100
        per_scorer.append((m, rec, fp))
    ps = pd.DataFrame(per_scorer, columns=["scorer", "recovery_%", "realised_FAR_%"]) \
        .sort_values("recovery_%", ascending=False)
    ps.to_csv(out / "per_scorer_recovery.csv", index=False)
    print(f"\n--- per-scorer recovery @{far*100:.0f}% FAR (single metric, held-out FAR) ---")
    print(ps.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))

    # 7. best single scorer + a greedy OR of top scorers at the SAME total FAR
    best_m = ps.iloc[0]["scorer"]
    rows.append((f"best single scorer ({best_m}) @{far*100:.0f}%FAR",
                 ps.iloc[0]["recovery_%"], ps.iloc[0]["realised_FAR_%"]))

    df = pd.DataFrame(rows, columns=["strategy", "recovery_%", "realised_FAR_%"])
    df.to_csv(out / "recovery_strategies.csv", index=False)
    print(f"\nrare N = {nrare}   target FAR = {far*100:.0f}%\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))
    best80 = df[df["recovery_%"] >= 80].sort_values("realised_FAR_%")
    print("\n>= 80% recovery:")
    print(best80.to_string(index=False, float_format=lambda x: f"{x:6.1f}") if len(best80) else "  none at this FAR")
    print(f"\nSaved -> {out/'recovery_strategies.csv'}")


if __name__ == "__main__":
    main()
