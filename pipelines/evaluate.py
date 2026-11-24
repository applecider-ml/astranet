#!/usr/bin/env python3
"""Classification evaluation for the consistent ensemble.

Uses the cached fine_logits (already the consistent ensemble-averaged logits) on the
test split to produce: overall acc / macro recall, per-class recall, and confusion
matrices (light + dark). For per-strategy comparison we report the single consistent
ensemble (the repo's canonical ensemble), avoiding the mixed-ensemble artifact.

Outputs -> results/<spec>/classification/

Usage: python pipelines/evaluate.py --spec 6model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from _common import REPO, load_config, make_astra_config, load_cache
from luna.viz import plot_confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ac = make_astra_config(cfg)
    class_names = ac.class_names
    cache_dir = REPO / "cache" / args.spec
    out = REPO / "results" / args.spec / "classification"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and not args.force:
        print(f"[{args.spec}] classification eval already done -> {out}")
        return

    test = load_cache(cache_dir, "test")
    logits, y = test["fine_logits"], np.asarray(test["labels"])
    preds = logits.argmax(1)
    acc = float((preds == y).mean())
    per_class = []
    N = len(class_names)
    cm = np.zeros((N, N), dtype=int)
    for t, p in zip(y, preds):
        cm[int(t), int(p)] += 1
    for c in range(N):
        m = y == c
        per_class.append(float((preds[m] == c).mean()) if m.any() else 0.0)
    macro = float(np.mean(per_class))

    plot_confusion(cm, class_names, f"ASTRAnet ({args.spec}, consistent)",
                   out, stem="confusion_test")   # emits light + dark

    metrics = {"spec": args.spec, "test_acc": acc, "test_macro_recall": macro,
               "per_class_recall": dict(zip(class_names, per_class)),
               "confusion_matrix": cm.tolist(), "class_names": class_names}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{args.spec}] classification: acc={acc:.4f}  macro={macro:.4f}  -> {out}")
    for c, r in zip(class_names, per_class):
        print(f"    {c:6s}: {r:.3f}")


if __name__ == "__main__":
    main()
