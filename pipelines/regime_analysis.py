#!/usr/bin/env python3
"""Do the anomaly-score signatures of rare spectra form structured regimes?

For every rare spectrum the 16 LUNA scores are computed on the ensemble embedding (no outlier
exposure, no rare data used in fitting) and rank-normalised against the validation split, giving a
16-d signature in [0, 1]; the four family means give a 4-d signature.  We ask three questions:
  1. Is the signature class-structured?  Per-class mean family signature (heat map) and
     adjusted mutual information between an unsupervised clustering of the signatures and the class.
  2. How many regimes do the data support?  Silhouette over k for k-means on the 16-d signatures.
  3. Where do the clusters sit in the embedding?  UMAP of test + rare embeddings coloured by cluster.
Outputs -> results/<spec>/anomaly/regimes/{signature_by_class.csv, clusters.csv, silhouette.csv,
           regime_signatures_light.pdf/png, regime_umap_light.pdf/png}
Usage: python pipelines/regime_analysis.py --spec v7fixmd6
"""
from __future__ import annotations
import argparse, collections, csv
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score

from _common import REPO, resolve_path, load_config, make_astra_config, load_cache, RARE_TIER1
from astranet.model import build_hierarchy_mappings
from astranet.sentinel import luna_dict

FAMILY = {"msp": "uncertainty", "entropy": "uncertainty", "energy": "uncertainty", "odin": "uncertainty",
          "mahal_global": "distance", "mahal_class": "distance", "mahal_within": "distance", "knn": "distance",
          "cosine": "distance", "typicality": "distance", "gmm": "density", "iforest": "density", "lof": "density",
          "pca_recon": "density", "hec": "hybrid", "mrs": "hybrid"}
FAMS = ["uncertainty", "distance", "density", "hybrid"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="v7fixmd6"); ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    cache = REPO / "cache" / args.spec
    tr, va, te, ra = (load_cache(cache, n) for n in ("train", "val", "test", "rare_unknown_combined"))
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), map_location="cpu", weights_only=False)
    cls = np.array(rd["class"]); objid = np.array(rd["obj_id"])
    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=0.01, fine_to_coarse=f2c); pipe.fit(luna_dict(tr)); M = pipe.active_methods
    sv, sr = pipe.score(luna_dict(va)), pipe.score(luna_dict(ra))
    rank = lambda m, x: np.searchsorted(np.sort(sv[m]), x) / len(sv[m])          # noqa: E731
    S16 = np.column_stack([rank(m, sr[m]) for m in M])                              # (289, 16) in [0,1]
    S4 = np.column_stack([S16[:, [i for i, m in enumerate(M) if FAMILY[m] == f]].mean(1) for f in FAMS])
    out = REPO / "results" / args.spec / "anomaly" / "regimes"; out.mkdir(parents=True, exist_ok=True)

    # 1. class-structured signatures
    classes = sorted(set(cls), key=lambda c: (c not in RARE_TIER1, c))
    with open(out / "signature_by_class.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "tier", "n_spectra", "n_objects"] + FAMS + M)
        for c in classes:
            m = cls == c
            w.writerow([c, 1 if c in RARE_TIER1 else 2, int(m.sum()), len(set(objid[m]))] +
                       [f"{v:.3f}" for v in S4[m].mean(0)] + [f"{v:.3f}" for v in S16[m].mean(0)])
    # 2. how many regimes?  k-means on 16-d signatures, silhouette, AMI vs class / tier
    rows = []; best = None
    for k in range(2, 9):
        km = KMeans(k, n_init=20, random_state=0).fit(S16)
        sil = silhouette_score(S16, km.labels_); ami_c = adjusted_mutual_info_score(cls, km.labels_)
        ami_t = adjusted_mutual_info_score(np.isin(cls, list(RARE_TIER1)), km.labels_)
        rows.append({"k": k, "silhouette": round(sil, 3), "AMI_vs_class": round(ami_c, 3), "AMI_vs_tier": round(ami_t, 3)})
        if best is None or sil > best[1]: best = (k, sil, km.labels_)
        print(f"k={k}: silhouette {sil:.3f}  AMI(class) {ami_c:.3f}  AMI(tier) {ami_t:.3f}", flush=True)
    with open(out / "silhouette.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    k, sil, lab = best; print(f"\nbest k={k} (silhouette {sil:.3f})")
    # which family dominates each cluster, and which classes populate it
    with open(out / "clusters.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cluster", "n_spectra", "dominant_family", "mean_signature(" + ",".join(FAMS) + ")", "classes"])
        for c in range(k):
            m = lab == c; sig = S4[m].mean(0); comp = collections.Counter(cls[m]).most_common(4)
            w.writerow([c, int(m.sum()), FAMS[int(sig.argmax())], " ".join(f"{v:.2f}" for v in sig), "; ".join(f"{a}:{b}" for a, b in comp)])
            print(f"  cluster {c}: n={m.sum():3d}  dominant {FAMS[int(sig.argmax())]:11s}  sig={np.round(sig,2)}  {comp}")
    # 3. figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    H = np.array([S4[cls == c].mean(0) for c in classes])
    im = ax.imshow(H, vmin=0.5, vmax=1.0, cmap="magma", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(FAMS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([f"{c} ({(cls==c).sum()})" for c in classes], fontsize=8)
    ax.axhline(sum(c in RARE_TIER1 for c in classes) - 0.5, color="w", lw=1)
    plt.colorbar(im, ax=ax, label="mean rank vs validation"); fig.tight_layout()
    fig.savefig(out / "regime_signatures_light.pdf"); fig.savefig(out / "regime_signatures_light.png", dpi=200)
    try:
        import umap
        E = np.vstack([te["emb"], ra["emb"]]); Z = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(E)
        fig, ax = plt.subplots(figsize=(3.5, 3.2)); n = len(te["emb"])
        ax.scatter(Z[:n, 0], Z[:n, 1], s=2, c="0.8", lw=0, label="test (7 classes)")
        for c in range(k):
            m = lab == c; ax.scatter(Z[n:][m, 0], Z[n:][m, 1], s=9, lw=0, label=f"regime {c+1} ({FAMS[int(S4[m].mean(0).argmax())]})")
        ax.legend(fontsize=6, frameon=False); ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout()
        fig.savefig(out / "regime_umap_light.pdf"); fig.savefig(out / "regime_umap_light.png", dpi=200)
    except Exception as e:
        print("umap figure skipped:", e)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
