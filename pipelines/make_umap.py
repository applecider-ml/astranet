#!/usr/bin/env python3
"""Publication UMAP of the ensemble embedding, coloured per class.

Two figures (each in light PDF/PNG + dark PNG, via luna.style):
  umap_classes  — test-split embeddings, one colour per ASTRA class, direct
                  labels at class centroids + legend (colour never alone).
  umap_classes_ood — same, with the 289 rare/OOD spectra overlaid as x markers.
Also writes an interactive plotly HTML (hover shows class).

Colours: luna.style.class_colors(dark) — fixed per-class assignment, validated
for CVD separation and per-surface lightness/contrast (light AND dark sets).

Usage: python pipelines/make_umap.py --spec v7fix6 --config configs/v7fix.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from _common import REPO, load_config, make_astra_config, load_cache
from luna.style import themed, theme_colors, save_fig
from astranet.plotstyle import class_colors, CLASS_ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="v7fix6")
    ap.add_argument("--config", default="configs/v7fix.yaml")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg)
    CN = ac.class_names
    cache = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "figures"; out.mkdir(parents=True, exist_ok=True)

    te = load_cache(cache, args.split)
    ra = load_cache(cache, "rare_unknown_combined")
    y = np.asarray(te["labels"])

    import umap
    print(f"UMAP on {len(te['emb'])} {args.split} + {len(ra['emb'])} rare embeddings ...", flush=True)
    # pca init + higher n_neighbors -> compact connected layout (spectral init can
    # throw small distinct clusters, e.g. CVs, arbitrarily far and squash the rest)
    Z = umap.UMAP(n_neighbors=60, min_dist=0.15, init="pca",
                  random_state=42).fit_transform(np.vstack([te["emb"], ra["emb"]]))
    Zt, Zr = Z[:len(te["emb"])], Z[len(te["emb"]):]
    # view limits from the central mass (robust to any remaining far-flung blob)
    lo, hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
    pad = 0.18 * (hi - lo)
    XL, YL = (lo[0]-pad[0], hi[0]+pad[0]), (lo[1]-pad[1], hi[1]+pad[1])

    def draw(dark, with_ood):
        c = theme_colors(dark); col = class_colors(dark)
        with themed(dark):
            fig, ax = plt.subplots(figsize=(9, 8))
            for name in CLASS_ORDER:
                m = y == CN.index(name)
                if not m.any():
                    continue
                ax.scatter(Zt[m, 0], Zt[m, 1], s=9, color=col[name], label=name,
                           alpha=0.6, edgecolors="none", rasterized=True)
            if with_ood:
                ax.scatter(Zr[:, 0], Zr[:, 1], s=42, color=c["fg"], marker="x",
                           linewidths=1.4, label="rare / OOD", alpha=0.9)
            # direct labels at class medians (relief for the contrast WARN),
            # nudged apart when two medians nearly coincide
            halo = [pe.withStroke(linewidth=3.5, foreground=c["bg"])]
            placed = []
            span = np.array([XL[1] - XL[0], YL[1] - YL[0]])
            for name in CLASS_ORDER:
                m = y == CN.index(name)
                if m.sum() < 5:
                    continue
                cx, cy = np.median(Zt[m, 0]), np.median(Zt[m, 1])
                for px, py in placed:                       # collision nudge
                    if abs(cx - px) < 0.07 * span[0] and abs(cy - py) < 0.05 * span[1]:
                        cy += 0.06 * span[1] * (1 if cy >= py else -1)
                placed.append((cx, cy))
                ax.text(cx, cy, name, color=c["fg"], fontsize=15, fontweight="bold",
                        ha="center", va="center", path_effects=halo, zorder=5)
            ax.legend(markerscale=2.2, loc="best", framealpha=0.85,
                      handletextpad=0.3, borderpad=0.5)
            ax.set_xlim(*XL); ax.set_ylim(*YL)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            stem = "umap_classes_ood" if with_ood else "umap_classes"
            save_fig(fig, out / stem, dark)

    for dark in (False, True):
        draw(dark, with_ood=False)
        draw(dark, with_ood=True)

    # interactive HTML
    try:
        import plotly.graph_objects as go
        col = class_colors(False)
        f = go.Figure()
        for name in CLASS_ORDER:
            m = y == CN.index(name)
            if m.any():
                f.add_trace(go.Scattergl(x=Zt[m, 0], y=Zt[m, 1], mode="markers", name=name,
                                         marker=dict(size=5, opacity=0.65, color=col[name]),
                                         hovertemplate=f"{name}<extra></extra>"))
        f.add_trace(go.Scattergl(x=Zr[:, 0], y=Zr[:, 1], mode="markers", name="rare / OOD",
                                 marker=dict(size=9, color="#111", symbol="x"),
                                 hovertemplate="rare/OOD<extra></extra>"))
        f.update_layout(template="plotly_white", font=dict(size=15),
                        title=f"ASTRAnet embedding — UMAP ({args.spec})",
                        xaxis=dict(visible=False), yaxis=dict(visible=False))
        f.write_html(str(out / "umap_classes.html"))
    except Exception as e:
        print(f"[html skipped] {e}")
    print(f"Saved -> {out}/umap_classes*", flush=True)


if __name__ == "__main__":
    main()
