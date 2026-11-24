#!/usr/bin/env python3
"""Train ASTRAnet across seeds from scratch -> models/trained/seed<seed>/.

Optional: the repo ships with `fetch_paper_weights.py` for the reference checkpoints.
Use this only to retrain. A seed with an existing test_results.json is skipped.
Set `weights_source: trained` in configs/default.yaml to use these instead of paper.

Usage: python pipelines/train_seeds.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from _common import REPO, load_config, make_astra_config
from astranet.train_loop import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="train only this seed (for one-job-per-seed condor); default = all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [42, 123, 7])
    out_root = REPO / "models" / cfg.get("train_out", "trained")
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        out_dir = out_root / f"seed{seed}"
        if (out_dir / "test_results.json").exists() and not args.force:
            print(f"seed {seed}: already done — skipping")
            continue
        print(f"\n{'='*60}\nTraining seed {seed}\n{'='*60}", flush=True)
        ac = make_astra_config(cfg)
        ac.augment_train = True            # restore training-time augmentation
        ac.use_balanced_sampling = True
        ac.out_dir = str(out_dir)
        ac.seed = seed
        ac.gpu_id = cfg.get("gpu_id", 0)
        if cfg.get("init_from"):
            ac.init_from = str(REPO / cfg["init_from"])   # e.g. SSL-pretrained trunk
        t0 = time.time()
        train(ac)
        print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)
        torch.cuda.empty_cache()

    print("\nAll seeds complete.")
    for seed in seeds:
        rp = out_root / f"seed{seed}" / "test_results.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            print(f"  seed={seed}: acc={r.get('val_acc',0):.4f} macro={r.get('val_macro_acc',0):.4f}")


if __name__ == "__main__":
    main()
