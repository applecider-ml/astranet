#!/usr/bin/env python3
"""Extract the ensemble features of every split and cache them.

The same members are used for train, val, test and the rare set, and the script asserts an
identical member count for every split.

Outputs (cache/<spec>/):
    train.npz, val.npz, test.npz, rare_unknown_combined.npz  (+ *_meta.pkl)
    per_model_softmax.npz   (val_softmax, rare_softmax)

Usage:
    python pipelines/build_ensemble.py --spec tcn6 --config configs/tcn.yaml
    python pipelines/build_ensemble.py --spec spectranet6 --config configs/spectranet.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from _common import (REPO, load_config, make_astra_config, device_for,
                     load_ensemble, extract, save_cache, ENSEMBLE_SPECS)
from astranet.model import build_hierarchy_mappings
from astranet.dataset import build_loaders, make_loader


def _rare_loader(ac, rare_pt: Path):
    """Build the rare/unknown loader from the combined .pt."""
    rd = torch.load(str(rare_pt), map_location="cpu")
    meta = [{
        "obj_id": rd["obj_id"][i], "file_path": rd["file_path"][i],
        "phase": rd["phase"][i], "z": rd["redshift"][i],
        "instrument": rd["instrument"][i], "class": rd["class"][i],
        "taxonomy_I": rd["taxonomy_I"][i], "taxonomy_II": rd["taxonomy_II"][i],
    } for i in range(len(rd["obj_id"]))]
    return make_loader(
        {"X": rd["flux"].unsqueeze(1), "meta": meta},
        ac.batch_size, shuffle=False, labeled=False, augment=False,
        num_workers=ac.num_workers, pin_memory=ac.pin_memory,
        compute_derivative=ac.compute_derivative, class_names=ac.class_names,
        savgol_window=ac.savgol_window, savgol_poly=ac.savgol_poly,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=list(ENSEMBLE_SPECS), required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ac = make_astra_config(cfg)
    device = device_for(cfg)
    _, _, superclass_names = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    n_coarse = len(superclass_names)

    cache_dir = REPO / "cache" / args.spec
    done_marker = cache_dir / "rare_unknown_combined.npz"
    if done_marker.exists() and not args.force:
        print(f"[{args.spec}] cache already exists at {cache_dir} — skipping (use --force)")
        return

    models_dir = REPO / "models" / cfg.get("weights_source", "tcn")
    members, labels = load_ensemble(args.spec, models_dir, ac, n_coarse, device)
    print(f"[{args.spec}] ensemble = {len(members)} models: {labels}")
    print(f"Device: {device}")

    # Loaders
    _, train_loader, val_loader, test_loader, _ = build_loaders(ac)
    rare_loader = _rare_loader(ac, Path(cfg["data"]["rare_pt"]))
    splits = [("train", train_loader, True), ("val", val_loader, True),
              ("test", test_loader, True), ("rare_unknown_combined", rare_loader, False)]

    n_models = len(members)
    per_model = {}
    for name, loader, labeled in splits:
        print(f"  extracting {name} ...", flush=True)
        data = extract(members, loader, device, labeled=labeled)
        # consistency assertion — every split MUST use the same model count
        assert data["per_model_softmax"].shape[0] == n_models, (
            f"model-count mismatch for {name}: {data['per_model_softmax'].shape[0]} != {n_models}")
        save_cache(data, cache_dir, name)
        if name in ("val", "rare_unknown_combined"):
            key = "val_softmax" if name == "val" else "rare_softmax"
            per_model[key] = data["per_model_softmax"]
        print(f"    {name}: N={data['emb'].shape[0]}, emb={data['emb'].shape[1]}, "
              f"models={data['per_model_softmax'].shape[0]}")
        torch.cuda.empty_cache()

    np.savez(cache_dir / "per_model_softmax.npz", **per_model)
    print(f"\n[{args.spec}] done -> {cache_dir}  (consistent {n_models}-model ensemble)")


if __name__ == "__main__":
    main()
