#!/usr/bin/env python3
"""Inference-time metadata ablation: the TRAINED ensemble is held fixed and evaluated on the
test split with redshift and/or phase withheld (set to NaN -> validity flag 0).

This answers "does the deployed model work without z/phase at inference?" without retraining.

Usage: python pipelines/metadata_masked_eval.py --spec v7fix6
Outputs -> results/<spec>/classification/metadata_masked.json (+ .md table)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from _common import (REPO, load_config, make_astra_config, device_for,
                     load_ensemble, ENSEMBLE_SPECS)
from astranet.model import build_hierarchy_mappings
from astranet.dataset import build_loaders

CONDITIONS = {
    "full":     (False, False),
    "no_z":     (True,  False),
    "no_phase": (False, True),
    "none":     (True,  True),
}

@torch.no_grad()
def run(models, loader, device, mask_z, mask_phase):
    per_model, ys = [], []
    for batch in loader:
        x = batch["x"].to(device); ph = batch["phase"].to(device); z = batch["z"].to(device)
        if mask_z:     z  = torch.full_like(z,  float("nan"))
        if mask_phase: ph = torch.full_like(ph, float("nan"))
        outs = [F.softmax(m(x, ph, z)["fine_logits"], -1) for m in models]
        per_model.append(torch.stack(outs).cpu()); ys.append(batch["y"])
    P = torch.cat(per_model, 1).numpy()      # (M, N, K)
    y = torch.cat(ys).numpy()
    return P, y

def metrics(prob, y, K):
    pred = prob.argmax(1)
    rec = [float((pred[y == c] == c).mean()) if (y == c).any() else float("nan") for c in range(K)]
    return {"acc": float((pred == y).mean()), "macro_recall": float(np.nanmean(rec)), "per_class_recall": rec}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=list(ENSEMBLE_SPECS), default="v7fix6")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="val = model-selection view (no test contact); test = final report")
    args = ap.parse_args()
    cfg = load_config(args.config); ac = make_astra_config(cfg); device = device_for(cfg)
    _, _, sup = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    models, labels = load_ensemble(args.spec, REPO / "models" / cfg.get("weights_source", "paper"), ac, len(sup), device)
    _, _, val_loader, test_loader, _ = build_loaders(ac)
    if args.split == "val": test_loader = val_loader
    K = len(ac.class_names)
    res = {"spec": args.spec, "split": args.split, "members": labels, "class_names": ac.class_names, "conditions": {}}
    for name, (mz, mp) in CONDITIONS.items():
        P, y = run(models, test_loader, device, mz, mp)
        ens = metrics(P.mean(0), y, K)
        ens["per_member"] = {lab: metrics(P[i], y, K)["acc"] for i, lab in enumerate(labels)}
        res["conditions"][name] = ens
        print(f"{name:9s} acc={ens['acc']:.4f} macro={ens['macro_recall']:.4f} "
              f"members={[round(v,3) for v in ens['per_member'].values()]}", flush=True)
    out = REPO / "results" / args.spec / "classification"; out.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.split == "test" else f"_{args.split}"
    (out / f"metadata_masked{sfx}.json").write_text(json.dumps(res, indent=2))
    lines = ["| condition | acc | macro recall | " + " | ".join(ac.class_names) + " |",
             "|---|---|---|" + "---|" * K]
    for name, r in res["conditions"].items():
        lines.append(f"| {name} | {r['acc']:.3f} | {r['macro_recall']:.3f} | " +
                     " | ".join(f"{v:.2f}" for v in r["per_class_recall"]) + " |")
    (out / f"metadata_masked{sfx}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
