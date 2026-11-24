#!/usr/bin/env python3
"""Paper figure: per rare class, the share of spectra flagged by 0-4 score families (zero-knowledge,
1% FAR, no rare data), and how many of the spectra no family flags are recovered with outlier exposure
(three-seed anchor of the object-grouped k-fold).  Usage: python pipelines/make_family_agreement_figure.py --out fig.pdf"""
from __future__ import annotations
import argparse
import numpy as np, torch
from _common import REPO, load_cache, load_config, make_astra_config, resolve_path
from astranet.model import build_hierarchy_mappings
from astranet.sentinel import luna_dict

FAM = {"msp": "u", "entropy": "u", "energy": "u", "odin": "u", "mahal_global": "d", "mahal_class": "d", "mahal_within": "d", "knn": "d",
       "cosine": "d", "typicality": "d", "gmm": "n", "iforest": "n", "lof": "n", "pca_recon": "n", "hec": "h", "mrs": "h"}
TIER1 = ["FBOT", "ILRT", "afterglow", "Luminous Red Nova", "Ca-rich", "long GRB"]
TIER2 = ["Novae", "Classical Nova", "QSO", "Blazar", "BL Lac"]
NICE = {"Luminous Red Nova": "Luminous red nova", "afterglow": "GRB afterglow", "Novae": "Nova", "Classical Nova": "Classical nova", "long GRB": "Long GRB"}
COLORS = ["#d6d5d0", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # 0 families = neutral; 1-4 = validated ordinal blue ramp
INK, MUTED = "#1f1f1d", "#6b6a66"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="v7fixmd6"); ap.add_argument("--config", default="../configs/v7fixmd.yaml")
    ap.add_argument("--out", required=True); args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg); f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), weights_only=False); cls = np.array(rd["class"]); obj = np.array(list(map(str, rd["obj_id"])))
    d = REPO / "cache" / args.spec; tr, va, ra = (load_cache(d, n) for n in ("train", "val", "rare_unknown_combined"))
    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=0.01, fine_to_coarse=f2c); pipe.fit(luna_dict(tr)); M = pipe.active_methods
    sv, sr = pipe.score(luna_dict(va)), pipe.score(luna_dict(ra)); rk = lambda m, x: np.searchsorted(np.sort(sv[m]), x) / len(sv[m])  # noqa: E731
    nf = np.zeros(len(cls), int)
    for f in "udnh":
        ms = [m for m in M if FAM[m] == f]; v = np.mean([rk(m, sv[m]) for m in ms], 0); r = np.mean([rk(m, sr[m]) for m in ms], 0); nf += r >= np.quantile(v, 0.99)
    S, T = [], []
    for s in (42, 123, 7):
        z = np.load(REPO / "results" / "oe" / f"oe_kfold_scores_v7fixmd_s{s}_f5obj.npz", allow_pickle=True); S.append(z["score_oe"]); T.append(z["thr_oe"].mean())
    oe = np.mean(S, 0) >= np.mean(T)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
    rows = [("Tier 1: novel phenomena", None)] + [(c, c) for c in TIER1] + [("Tier 2: rare members of trained families", None)] + [(c, c) for c in TIER2]
    fig, ax = plt.subplots(figsize=(3.5, 3.1)); y = np.arange(len(rows))[::-1]; hdr = []
    for yi, (lab, c) in zip(y, rows):
        if c is None:
            hdr.append((yi, lab)); continue
        m = cls == c; n = int(m.sum()); share = np.array([(nf[m] == k).sum() for k in range(5)]) / n; left = 0.0
        for k in range(5):
            if share[k] > 0: ax.barh(yi, share[k], left=left, height=0.72, color=COLORS[k], edgecolor="white", linewidth=1.0)
            left += share[k]
        if share[0] >= 0.2: ax.text(share[0] / 2, yi, f"{100 * share[0]:.0f}%", ha="center", va="center", fontsize=7, color=INK)
        ax.text(-0.02, yi, f"{NICE.get(c, c)} ({n})", ha="right", va="center", fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
        z0 = m & (nf == 0); txt = f"{(z0 & oe).sum()}/{z0.sum()}" if z0.sum() else "–"
        ax.text(1.04, yi, txt, ha="left", va="center", fontsize=7.5, color=MUTED, transform=ax.get_yaxis_transform())
    ax.text(1.04, len(rows) - 0.25, "no family,\nrecovered\nwith OE", ha="left", va="bottom", fontsize=6.5, color=MUTED, transform=ax.get_yaxis_transform(), linespacing=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("share of the class's spectra", color=MUTED)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in range(5)]
    fig.legend(handles, ["none", "1", "2", "3", "4"], title="score families that flag the spectrum", ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7, title_fontsize=7, handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
    fig.subplots_adjust(left=0.40, right=0.83, top=0.86, bottom=0.12)
    for yi, lab in hdr:   # tier headings, left-aligned at the figure edge
        yf = fig.transFigure.inverted().transform(ax.transData.transform((0, yi)))[1]
        fig.text(0.015, yf, lab, ha="left", va="center", fontsize=7.5, fontweight="bold", color=INK)
    fig.savefig(args.out); fig.savefig(args.out.replace(".pdf", ".png"), dpi=220); print("saved", args.out)


if __name__ == "__main__":
    main()
