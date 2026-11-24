#!/usr/bin/env python3
"""Phase-dependence of anomaly detection for EXPLOSIVE transients (with a real peak).

Excludes the AGN family (QSO/Blazar/BL Lac) which have no meaningful 'days from peak'.
Produces publication figures (light + dark, watermelon palette):
  1. detection rate vs phase (binned, with binomial 68% CIs)
  2. phase distribution of detected vs missed spectra
  3. early-warning: per-object detection at the earliest spectrum vs any spectrum

Outputs -> results/phase/
Usage: python pipelines/phase_analysis.py --spec 6model
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import REPO, load_config
from luna.style import themed, theme_colors, save_fig, ACCENT, ACCENT_DEEP, WATERMELON_LIGHT

AGN_FAMILY = {"QSO", "Blazar", "BL Lac"}
PHASE_LO, PHASE_HI = -30, 60   # plotting window for explosive transients


def binom_ci(k, n, z=1.0):
    if n == 0:
        return 0, 0
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n)
    return max(0, p - half), min(1, p + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="6model")
    args = ap.parse_args()
    cfg = load_config()
    cache = REPO / "cache" / args.spec
    out = REPO / "results" / "phase"; out.mkdir(parents=True, exist_ok=True)

    meta = pickle.load(open(cache / "rare_unknown_combined_meta.pkl", "rb"))
    v = np.load(REPO / "results" / args.spec / "conformal" / "val_lgb_scores.npy")
    r = np.load(REPO / "results" / args.spec / "conformal" / "rare_lgb_scores.npy")
    thr = np.quantile(v, 0.99)
    det = r >= thr
    phase = np.array([m.get("phase", np.nan) for m in meta], float)
    cls = np.array([m.get("class", "?") for m in meta])
    obj = np.array([m.get("obj_id") for m in meta])

    expl = ~np.isin(cls, list(AGN_FAMILY)) & np.isfinite(phase)
    ph, dt = phase[expl], det[expl]
    inwin = (ph >= PHASE_LO) & (ph <= PHASE_HI)
    print(f"explosive spectra with phase: {expl.sum()}  (in [{PHASE_LO},{PHASE_HI}]: {inwin.sum()})")
    print(f"  overall explosive detection: {dt.mean()*100:.1f}%")

    # ---- Panel 1: detection rate vs phase ----
    bins = np.array([-30, -15, -7, 0, 7, 15, 30, 60])
    centers, rates, los, his, ns = [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (ph >= bins[i]) & (ph < bins[i + 1])
        if m.sum() == 0:
            continue
        k, n = dt[m].sum(), m.sum()
        centers.append((bins[i] + bins[i + 1]) / 2); rates.append(k / n)
        lo, hi = binom_ci(k, n); los.append(k / n - lo); his.append(hi - k / n); ns.append(n)
    centers, rates = np.array(centers), np.array(rates)

    for dark in (False, True):
        c = theme_colors(dark)
        with themed(dark):
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            ax.errorbar(centers, np.array(rates) * 100, yerr=[np.array(los) * 100, np.array(his) * 100],
                        fmt="o-", color=ACCENT, ecolor=c["fg_dim"], elinewidth=1.4,
                        capsize=4, markersize=9, lw=2.4)
            for x, y, n in zip(centers, rates, ns):
                ax.annotate(f"n={n}", (x, y * 100), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=11, color=c["fg_dim"])
            ax.axvline(0, ls="--", color=c["fg_dim"], lw=1.2)
            ax.text(0.5, 4, "peak", color=c["fg_dim"], fontsize=12)
            ax.set_xlabel("Phase (days from peak)"); ax.set_ylabel("Detection rate @ 1% FAR (%)")
            ax.set_title("Anomaly detection vs phase — explosive transients")
            ax.set_ylim(0, 105); ax.set_xlim(PHASE_LO, PHASE_HI)
            save_fig(fig, out / "detection_vs_phase", dark)

    # ---- Panel 2: phase distribution detected vs missed ----
    pw = ph[inwin]; dw = dt[inwin]
    for dark in (False, True):
        c = theme_colors(dark)
        with themed(dark):
            fig, ax = plt.subplots(figsize=(7.5, 5))
            b = np.linspace(PHASE_LO, PHASE_HI, 19)
            ax.hist(pw[dw], bins=b, alpha=0.8, color=ACCENT, label="detected", edgecolor=c["fg"])
            ax.hist(pw[~dw], bins=b, alpha=0.55, color="#64748b", label="missed", edgecolor=c["fg"])
            ax.axvline(0, ls="--", color=c["fg_dim"], lw=1.2)
            ax.set_xlabel("Phase (days from peak)"); ax.set_ylabel("# spectra")
            ax.set_title("Phase distribution: detected vs missed"); ax.legend()
            save_fig(fig, out / "phase_hist_detected_missed", dark)

    # ---- Panel 3: early-warning (per-object) ----
    early = both = tot = 0
    for o in set(obj[expl]):
        idx = np.where((obj == o) & expl)[0]
        if len(idx) < 1:
            continue
        tot += 1
        e = idx[np.argmin(phase[idx])]
        if det[idx].any(): both += 1
        if det[e]: early += 1
    print(f"\nExplosive objects: {tot}")
    print(f"  detected at earliest spectrum: {early}/{tot} = {early/tot*100:.0f}%")
    print(f"  detected at any spectrum     : {both}/{tot} = {both/tot*100:.0f}%")
    for dark in (False, True):
        c = theme_colors(dark)
        with themed(dark):
            fig, ax = plt.subplots(figsize=(6, 5))
            vals = [early / tot * 100, both / tot * 100]
            bars = ax.bar(["earliest\nspectrum", "any\nspectrum"], vals,
                          color=[ACCENT, ACCENT_DEEP], edgecolor=c["fg"], width=0.6)
            for b_, val in zip(bars, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, val + 1.5, f"{val:.0f}%",
                        ha="center", fontsize=15, color=c["fg"], fontweight="bold")
            ax.set_ylim(0, 105); ax.set_ylabel("Objects detected (%)")
            ax.set_title("Early-warning recall (explosive objects)")
            save_fig(fig, out / "early_warning", dark)

    print(f"\nSaved figures -> {out}/  (detection_vs_phase, phase_hist_detected_missed, early_warning) x light/dark")


if __name__ == "__main__":
    main()
