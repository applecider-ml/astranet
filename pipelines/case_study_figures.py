#!/usr/bin/env python3
"""Case-study figure: for each exemplar object, the observed spectrum (one epoch), its position in
the UMAP of the ensemble embedding (test split + rare set), and its four-family anomaly signature
(mean rank of the 16 LUNA scores vs the validation split, no outlier exposure).

Usage: python pipelines/case_study_figures.py --spec tcn6 --objects ZTF18acyvlkx ZTF21acddxti ZTF24abjjpbo
Outputs -> results/<spec>/figures/case_studies_light.{pdf,png}
"""
from __future__ import annotations
import argparse
import numpy as np, torch
from _common import REPO, resolve_path, load_cache, load_config, make_astra_config
from astranet.model import build_hierarchy_mappings
from astranet.novelty import luna_dict
FAMILY = {"msp": "uncertainty", "entropy": "uncertainty", "energy": "uncertainty", "odin": "uncertainty",
          "mahal_global": "distance", "mahal_class": "distance", "mahal_within": "distance", "knn": "distance",
          "cosine": "distance", "typicality": "distance", "gmm": "density", "iforest": "density", "lof": "density",
          "pca_recon": "density", "hec": "hybrid", "mrs": "hybrid"}
FAMS = ["uncertainty", "distance", "density", "hybrid"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="tcn6"); ap.add_argument("--config", default=None)
    ap.add_argument("--objects", nargs="+", required=True); args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg); f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    tr, va, te, ra = (load_cache(cache, n) for n in ("train", "val", "test", "rare_unknown_combined"))
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu", weights_only=False)
    obj = np.array([str(o) for o in rd["obj_id"]]); cls = np.array(rd["class"]); flux = rd["flux"].numpy()
    wl = np.linspace(rd["wl_range"][0], rd["wl_range"][1], rd["interp_len"])
    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=0.01, fine_to_coarse=f2c); pipe.fit(luna_dict(tr)); M = pipe.active_methods
    sv, sr = pipe.score(luna_dict(va)), pipe.score(luna_dict(ra))
    S16 = np.column_stack([np.searchsorted(np.sort(sv[m]), sr[m]) / len(sv[m]) for m in M])
    S4 = np.column_stack([S16[:, [i for i, m in enumerate(M) if FAMILY[m] == f]].mean(1) for f in FAMS])
    pred = ra["fine_logits"].argmax(1); probs = np.exp(ra["fine_logits"] - ra["fine_logits"].max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    import umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    E = np.vstack([te["emb"], ra["emb"]]); Z = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(E); n = len(te["emb"])
    tl = np.asarray(te["labels"])
    try:
        from astranet.plotstyle import class_colors; colors = class_colors(False)
    except Exception:
        colors = dict(zip(ac.class_names, plt.cm.tab10.colors))
    rows = [o for o in args.objects if (obj == o).any()]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.1, 1.75 * len(rows)), gridspec_kw={"width_ratios": [2.2, 1.3, 1.0]})
    axes = np.atleast_2d(axes)
    for r, o in enumerate(rows):
        idx = np.where(obj == o)[0]
        spk = np.array([np.abs(np.diff(flux[j])).max() / (np.percentile(np.abs(np.diff(flux[j])), 95) + 1e-9) for j in idx])
        i = idx[np.argmin(spk)]                                                   # cleanest epoch (least spiky)
        ax = axes[r, 0]; f = flux[i]; f = (f - np.nanpercentile(f, 5)) / (np.nanpercentile(f, 95) - np.nanpercentile(f, 5) + 1e-9)
        ax.plot(wl, f, lw=0.6, color="k"); ax.set_xlim(wl[0], wl[-1]); ax.set_yticks([])
        ax.set_title(f"{o}  ({cls[i]}; {rd['instrument'][i]}; epoch {list(idx).index(i)+1}/{len(idx)})", fontsize=8, loc="left")
        top = np.argsort(probs[i])[::-1][:2]
        ax.text(0.99, 0.95, "classifier: " + ", ".join(f"{ac.class_names[k]} {100*probs[i,k]:.0f}%" for k in top), ha="right", va="top", fontsize=7, transform=ax.transAxes)
        if r == len(rows) - 1: ax.set_xlabel("observed wavelength [Å]", fontsize=8)
        ax = axes[r, 1]
        for k, c in enumerate(ac.class_names):
            m = tl == k; ax.scatter(Z[:n][m, 0], Z[:n][m, 1], s=1.5, lw=0, color=colors.get(c, "0.5"), alpha=0.5, label=c if r == 0 else None)
        ax.scatter(Z[n + idx, 0], Z[n + idx, 1], s=14, marker="*", color="k", zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.legend(fontsize=5, frameon=False, markerscale=4, loc="lower left", ncol=1, bbox_to_anchor=(0.0, 0.0))
        ax = axes[r, 2]; ax.barh(FAMS, S4[i], color=["0.3", "0.5", "0.7", "0.85"]); ax.set_xlim(0, 1); ax.axvline(0.99, color="r", lw=0.8, ls="--")
        ax.tick_params(labelsize=7); ax.set_xlabel("rank vs validation", fontsize=7) if r == len(rows) - 1 else None
    fig.tight_layout(); out = REPO / "results" / args.spec / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "case_studies_light.pdf"); fig.savefig(out / "case_studies_light.png", dpi=200); print(f"saved -> {out}/case_studies_light.pdf", rows)


if __name__ == "__main__":
    main()
