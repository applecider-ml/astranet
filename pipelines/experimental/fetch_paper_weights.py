#!/usr/bin/env python3
"""Fetch the paper-reconstructed reference checkpoints into models/paper/.

Provenance
----------
The published ASTRAnet model lives in ``ASTRAnet_final``. Its 7-model ensemble used
``seed42``'s peak-assembly checkpoint, which was later MOVED into a ``seed42/peak/``
subdirectory (this is what caused the original cache to silently drop to 6 models for
the rare split). We copy:

    seed{42,123,7}/checkpoint_best.pt
    seed{42,123,7}/checkpoint_swa.pt
    seed42/peak/checkpoint_peak_assembly.pt   ->  seed42/checkpoint_peak_assembly.pt

into ``models/paper/`` so the 6-model and 7-model specs are both fully available and
every split can be extracted consistently.

These files are git-ignored (large). Re-run this script on a fresh clone.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("ASTRANET_PAPER_RUNS", "runs_final"))  # original paper training runs (not distributed)
DST = REPO / "models" / "paper"

PLAN = {
    42: {"checkpoint_best.pt": "seed42/checkpoint_best.pt",
         "checkpoint_swa.pt": "seed42/checkpoint_swa.pt",
         "checkpoint_peak_assembly.pt": "seed42/peak/checkpoint_peak_assembly.pt"},
    123: {"checkpoint_best.pt": "seed123/checkpoint_best.pt",
          "checkpoint_swa.pt": "seed123/checkpoint_swa.pt"},
    7: {"checkpoint_best.pt": "seed7/checkpoint_best.pt",
        "checkpoint_swa.pt": "seed7/checkpoint_swa.pt"},
}


def main():
    force = "--force" in sys.argv
    if not SRC.exists():
        sys.exit(f"ERROR: source weights dir not found: {SRC}")
    n_copied = 0
    for seed, files in PLAN.items():
        out_seed = DST / f"seed{seed}"
        out_seed.mkdir(parents=True, exist_ok=True)
        for dst_name, src_rel in files.items():
            src = SRC / src_rel
            dst = out_seed / dst_name
            if not src.exists():
                print(f"  WARN missing source: {src}")
                continue
            if dst.exists() and not force:
                print(f"  skip (exists): {dst.relative_to(REPO)}")
                continue
            shutil.copy2(src, dst)
            n_copied += 1
            print(f"  copied: {src_rel}  ->  {dst.relative_to(REPO)}")
    print(f"\nDone. {n_copied} file(s) copied into {DST.relative_to(REPO)}/")
    print("Specs available: 6model (best+swa x3), 7model (+seed42 peak).")


if __name__ == "__main__":
    main()
