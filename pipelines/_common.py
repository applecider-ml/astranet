"""Shared utilities for the ASTRAnet pipelines.

- config loading (configs/default.yaml)
- ensemble specs (6model / 7model) — the SAME model set is used for every split
- model loading + consistent feature extraction (emb / logits / proj / per-model softmax)

The extraction logic mirrors the original notebook (anomaly_detection.ipynb) but is
parametrized so that *every* data split is extracted with the identical model set,
which structurally prevents the ID=7 / rare=6 cache mismatch.
"""
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import yaml
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# `luna` is the standalone LUNA package (pip install luna-ood / -e ../LUNA).

from astranet.config import ASTRAConfig, get_raw_to_group          # noqa: E402
from astranet.model import ASTRANet, build_hierarchy_mappings      # noqa: E402

# ── Ensemble specifications (consistent — used for ALL splits) ────────
ENSEMBLE_SPECS = {
    "6model": {42: ["best", "swa"], 123: ["best", "swa"], 7: ["best", "swa"]},
    "7model": {42: ["best", "swa", "peak_assembly"], 123: ["best", "swa"], 7: ["best", "swa"]},
    # Aliases of 6model for retrained weights: same composition, DISTINCT names so
    # cache/<spec> and results/<spec> never collide with the paper runs.
    # v7new6 -> configs/v7new.yaml (re-randomized split, fixed labels)
    # v7fix6 -> configs/v7fix.yaml (IDENTICAL v7 membership, only labels fixed —
    #           the apples-to-apples comparison against the paper weights)
    "v7new6": {42: ["best", "swa"], 123: ["best", "swa"], 7: ["best", "swa"]},
    "v7fix6": {42: ["best", "swa"], 123: ["best", "swa"], 7: ["best", "swa"]},
    # v7fixssl6 -> configs/v7fix_ssl.yaml (v7fix split, SSL-warm-started seeds)
    "v7fixssl6": {42: ["best", "swa"], 123: ["best", "swa"], 7: ["best", "swa"]},
    # Single-model specs (NO embedding averaging) — for anomaly detection, where
    # ensemble-averaging can smooth out the anomaly signal.
    "s42best": {42: ["best"]},
    "s123best": {123: ["best"]},
    "s7best": {7: ["best"]},
    "s42swa": {42: ["swa"]},
    # v7fixmd6 -> configs/v7fixmd.yaml (v7fix + metadata dropout at training; no-z inference)
    "v7fixmd6": {42: ["best", "swa"], 123: ["best", "swa"], 7: ["best", "swa"]},
    # metadata-dropout sweep, selected on the ROBUST validation criterion (never test/challenging)
    "v7fixmd26": {42: ["best", "swa"], 123: ["best", "swa"], 7: ["best", "swa"]},   # z 0.2 / phase 0.15
    "v7fixmd46": {42: ["best", "swa"], 123: ["best", "swa"], 7: ["best", "swa"]},   # z 0.4 / phase 0.3
    # v7fixsn6 -> configs/v7fixsn.yaml (v7fixmd recipe with the SpectraNet backbone of Xu+2026 in place of stem+TCN)
    "v7fixsn6": {42: ["best", "swa"], 123: ["best", "swa"], 7: ["best", "swa"]},
}
# Composite specs: the union of two trained ensembles with different backbones. The cache of a
# composite is built by pipelines/build_combo_cache.py from the caches of its parts (mean logits,
# concatenated embeddings); the paper's model is v7fixcombo12 = v7fixmd6 (TCN) + v7fixsn6 (SpectraNet).
COMPOSITE_SPECS = {
    "v7fixcombo12": [("v7fixmd6", "configs/v7fixmd.yaml"), ("v7fixsn6", "configs/v7fixsn.yaml")],
}
CKPT_FILES = {
    "best": "checkpoint_best.pt",
    "swa": "checkpoint_swa.pt",
    "peak_assembly": "checkpoint_peak_assembly.pt",
}


def load_config(path: str | None = None) -> dict:
    path = path or (REPO / "configs" / "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> str:
    """Resolve a config path: absolute stays as-is, relative is repo-rooted."""
    p = Path(p)
    return str(p if p.is_absolute() else REPO / p)


def make_astra_config(cfg: dict) -> ASTRAConfig:
    ac = ASTRAConfig(data_dir=resolve_path(cfg["data"]["dir"]))
    ac.augment_train = False
    ac.use_balanced_sampling = False
    md = cfg.get("meta_dropout") or {}
    ac.meta_dropout_z = float(md.get("z", 0.0))
    ac.meta_dropout_phase = float(md.get("phase", 0.0))
    if cfg.get("monitor"):  ac.monitor = str(cfg["monitor"])
    if cfg.get("patience"): ac.patience = int(cfg["patience"])
    if cfg.get("backbone"): ac.backbone = str(cfg["backbone"])
    if cfg.get("w_loco"): ac.w_loco = float(cfg["w_loco"])
    if cfg.get("z_shift"): ac.z_shift = True
    return ac


def device_for(cfg: dict) -> torch.device:
    gid = cfg.get("gpu_id", 0)
    return torch.device(f"cuda:{gid}" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────
def load_model(ckpt_path: str | Path, ac: ASTRAConfig, n_coarse: int, device) -> ASTRANet:
    model = ASTRANet(
        num_fine=ac.num_classes, num_coarse=n_coarse,
        in_channels=ac.in_channels, base_c=ac.base_c, emb_dim=ac.emb_dim,
        tcn_depth=ac.tcn_depth, tcn_kernel=ac.tcn_kernel, downsample=ac.downsample,
        dropout=ac.dropout, proj_dim=ac.proj_dim, drop_path_rate=ac.drop_path_rate,
        attn_heads=ac.attn_heads, backbone=getattr(ac, "backbone", "tcn"),
        z_shift=getattr(ac, "z_shift", False),
    )
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_ensemble(spec: str, models_dir: Path, ac: ASTRAConfig, n_coarse: int, device):
    """Load the model set for `spec` from models_dir/seed{N}/checkpoint_*.pt."""
    members, labels = [], []
    for seed, tags in ENSEMBLE_SPECS[spec].items():
        for tag in tags:
            p = models_dir / f"seed{seed}" / CKPT_FILES[tag]
            if not p.exists():
                raise FileNotFoundError(f"missing checkpoint for {spec}: {p}")
            members.append(load_model(p, ac, n_coarse, device))
            labels.append(f"seed{seed}/{tag}")
    return members, labels


@torch.no_grad()
def _ensemble_forward(models, x, phase, z):
    outs = [m(x, phase, z) for m in models]
    avg = {}
    for k in ["fine_logits", "coarse_logits", "embedding", "proj_fine", "proj_coarse"]:
        avg[k] = torch.stack([o[k] for o in outs]).mean(0)
    avg["proj_fine"] = F.normalize(avg["proj_fine"], dim=-1)
    avg["proj_coarse"] = F.normalize(avg["proj_coarse"], dim=-1)
    # per-model fine softmax (M, B, K)
    per_model = torch.stack([F.softmax(o["fine_logits"], dim=-1) for o in outs])
    return avg, per_model


@torch.no_grad()
def extract(models, loader, device, labeled=True):
    """Extract consistent ensemble features + per-model softmax from a loader."""
    R = {k: [] for k in ["embedding", "fine_logits", "coarse_logits", "proj_fine"]}
    per_model_chunks, labels, meta = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        ph = batch["phase"].to(device)
        z = batch["z"].to(device)
        avg, per_model = _ensemble_forward(models, x, ph, z)
        for k in R:
            R[k].append(avg[k].cpu())
        per_model_chunks.append(per_model.cpu())
        if labeled and "y" in batch:
            labels.append(batch["y"])
        meta.extend(batch.get("meta", [{}] * x.shape[0]))
    out = {k: torch.cat(v).numpy() for k, v in R.items()}
    out["emb"] = out.pop("embedding")
    out["per_model_softmax"] = torch.cat(per_model_chunks, dim=1).numpy()  # (M, N, K)
    out["labels"] = torch.cat(labels).numpy() if labels else None
    out["meta"] = meta
    return out


def save_cache(data: dict, cache_dir: Path, name: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items()
              if isinstance(v, np.ndarray) and k != "per_model_softmax"}
    np.savez(cache_dir / f"{name}.npz", **arrays)
    with open(cache_dir / f"{name}_meta.pkl", "wb") as f:
        pickle.dump(data["meta"], f)


def load_cache(cache_dir: Path, name: str) -> dict:
    npz = np.load(cache_dir / f"{name}.npz", allow_pickle=True)
    d = {k: npz[k] for k in npz.files}
    with open(cache_dir / f"{name}_meta.pkl", "rb") as f:
        d["meta"] = pickle.load(f)
    return d


# ── Rare-set (289) reporting tiers ────────────────────────────────────
# Tier 1: genuinely novel phenomena — no trained counterpart class at all.
# Tier 2: rare members of TRAINED families (novae ⊂ CV; QSO/Blazar/BL Lac ⊂ AGN)
#         — lower AD sensitivity here is partly CORRECT behaviour, report separately.
RARE_TIER1 = {"FBOT", "ILRT", "Luminous Red Nova", "Ca-rich", "afterglow", "long GRB"}
RARE_TIER2 = {"Novae", "Classical Nova", "QSO", "Blazar", "BL Lac"}


def rare_tier(raw_class: str) -> int:
    return 1 if raw_class in RARE_TIER1 else 2 if raw_class in RARE_TIER2 else 0


def tiered_report(obj_results):
    """obj_results: iterable of (detected: bool, raw_class: str) per OBJECT.
    Prints and returns tier detection summary."""
    t = {1: [0, 0], 2: [0, 0]}
    for det, c in obj_results:
        k = rare_tier(c)
        if k:
            t[k][1] += 1; t[k][0] += int(bool(det))
    out = {}
    for k, name in [(1, "Tier1 (true OOD)"), (2, "Tier2 (in-family rare)")]:
        d, n = t[k]
        out[f"tier{k}"] = {"det": d, "n": n, "pct": 100 * d / max(n, 1)}
        print(f"    {name:24s} {d:>3d}/{n:<3d} = {100*d/max(n,1):3.0f}%")
    return out
