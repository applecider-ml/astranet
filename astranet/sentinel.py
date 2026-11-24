"""ASTRANet-Sentinel — the anomaly-detection layer of the ASTRANet framework.

As described in the paper, the Sentinel non-linearly combines 16 embedding-space
anomaly scores spanning four families (uncertainty, distance, density, hybrid —
computed by the standalone LUNA package) with a LightGBM combiner, after
outlier-exposure (OE) fine-tuning of the encoder. Detection is always evaluated
under a leakage-free stratified k-fold protocol: no rare object is ever seen —
in fine-tuning, combiner training, or thresholding — by the detector that flags it.

Components (paper section in parentheses):
  oe_finetune()   encoder OE fine-tuning: CE + entropy-OE + embedding-margin (Methods)
  detect_fold()   Sentinel combiner on one fold: LightGBM on val-half + exposure
                  anomalies, threshold at target FAR on the held-out val half
  seed_average()  operating regime (i):  score-averaged anchor across seeds
  seed_union()    operating regime (iii): flag if any seed fires (higher recall,
                  effective FAR ~ n_seeds x target)

Drivers: pipelines/oe_kfold.py (per-seed k-fold run) and
pipelines/sentinel_ensemble.py (multi-seed aggregation + tiered report).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

LGB_PARAMS = dict(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.05,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  class_weight="balanced", random_state=42, verbosity=-1)


def luna_dict(f: dict) -> dict:
    """Adapt an ASTRAnet feature cache entry to LUNA's input format."""
    o = {"embeddings": f["emb"].astype(np.float32),
         "logits": f["fine_logits"].astype(np.float32),
         "coarse_logits": f["coarse_logits"].astype(np.float32),
         "projections": f["proj_fine"].astype(np.float32)}
    if f.get("labels") is not None:
        o["labels"] = np.asarray(f["labels"])
    return o


def oe_finetune(model, expo_loader, train_aug, device, *, epochs=20,
                lam=1.0, lam_emb=1.0, margin_mult=2.0, lr=1e-4, num_classes=7):
    """Outlier-exposure fine-tuning of an (already loaded) encoder.

    Loss = CE(train) + lam * entropy-OE(exposure) + lam_emb * embedding-margin:
    exposure anomalies are pushed toward a uniform softmax and at least a margin
    away from every class centroid. Returns the fine-tuned model in eval mode.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ei = iter(expo_loader)

    @torch.no_grad()
    def centroids(loader):
        model.eval(); E = []; Y = []
        for i, b in enumerate(loader):
            o = model(b["x"].to(device), b["phase"].to(device), b["z"].to(device))
            E.append(o["embedding"]); Y.append(b["y"].to(device))
            if i >= 8:
                break
        model.train()
        E = torch.cat(E); Y = torch.cat(Y)
        c = torch.stack([E[Y == k].mean(0) if (Y == k).any() else E.mean(0)
                         for k in range(num_classes)])
        return c, torch.cdist(E, c).min(1).values.median()

    for _ in range(epochs):
        cents = marg = None
        if lam_emb > 0:
            cents, med = centroids(train_aug)
            marg = margin_mult * med
        for b in train_aug:
            out = model(b["x"].to(device), b["phase"].to(device), b["z"].to(device))
            ce = F.cross_entropy(out["fine_logits"], b["y"].to(device))
            try:
                ob = next(ei)
            except StopIteration:
                ei = iter(expo_loader); ob = next(ei)
            oo = model(ob["x"].to(device), ob["phase"].to(device), ob["z"].to(device))
            oe = -(F.log_softmax(oo["fine_logits"], 1).mean(1)).mean()
            loss = ce + lam * oe
            if lam_emb > 0:
                d = torch.cdist(oo["embedding"], cents).min(1).values
                loss = loss + lam_emb * F.relu(marg - d).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def detect_fold(train_f, val_f, rare_f, fine_to_coarse, expo_idx, test_idx,
                *, target_far=0.01, seed=42):
    """One Sentinel fold: fit 16 scores on train, LightGBM on val-half + exposure
    anomalies, threshold at `target_far` on the held-out val half, score the
    held-out rare fold. Returns (flags, scores, threshold)."""
    import lightgbm as lgb
    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=target_far,
                        fine_to_coarse=fine_to_coarse)
    pipe.fit(luna_dict(train_f))
    M = pipe.active_methods

    def feats(f):
        sc = pipe.score(luna_dict(f))
        return np.column_stack([sc[m] for m in M])

    Xv, Xr = feats(val_f), feats(rare_f)
    rng = np.random.default_rng(seed)
    vi = rng.permutation(len(Xv))
    vtr, vth = vi[:len(Xv) // 2], vi[len(Xv) // 2:]
    X = np.vstack([Xv[vtr], Xr[expo_idx]])
    y = np.r_[np.zeros(len(vtr)), np.ones(len(expo_idx))]
    clf = lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y)
    thr = np.quantile(clf.predict_proba(Xv[vth])[:, 1], 1 - target_far)
    s = clf.predict_proba(Xr[test_idx])[:, 1]
    return s >= thr, s, thr


def detector_heads() -> dict:
    """Supervised anomaly detectors compared under identical folds/thresholds. Keys are the
    names used in results; values build an unfitted sklearn-style classifier."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))  # noqa: E731
    return {
        "scores16_lgbm":     ("scores", lambda: lgb.LGBMClassifier(**LGB_PARAMS)),   # the Sentinel combiner
        "scores16_logistic": ("scores", lr),
        "emb_logistic":      ("emb",    lr),
        "emb_mlp":           ("emb",    lambda: make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=500, random_state=0, early_stopping=True))),
        "emb_lgbm":          ("emb",    lambda: lgb.LGBMClassifier(**LGB_PARAMS)),
        "emb_scores16_lgbm": ("both",   lambda: lgb.LGBMClassifier(**LGB_PARAMS)),
    }


def detect_fold_heads(train_f, val_f, rare_f, fine_to_coarse, expo_idx, test_idx,
                      *, target_far=0.01, seed=42, heads=None):
    """Like detect_fold, but evaluates every detector in detector_heads() on the same fold:
    16 LUNA scores fitted on train; per head, fit on val-half (ID) + exposure (anomaly),
    threshold at target_far on the other val half, score the held-out rare fold.
    Returns {head: (flags, scores, threshold)}."""
    from luna import LUNAPipeline
    pipe = LUNAPipeline(combiner="lightgbm", target_far=target_far, fine_to_coarse=fine_to_coarse)
    pipe.fit(luna_dict(train_f))
    M = pipe.active_methods

    def scores16(f):
        sc = pipe.score(luna_dict(f))
        return np.column_stack([sc[m] for m in M])
    feats = {"scores": (scores16(val_f), scores16(rare_f)),
             "emb": (val_f["emb"].astype(np.float32), rare_f["emb"].astype(np.float32))}
    feats["both"] = tuple(np.hstack([feats["emb"][i], feats["scores"][i]]) for i in range(2))
    rng = np.random.default_rng(seed)
    vi = rng.permutation(len(feats["emb"][0]))
    vtr, vth = vi[:len(vi) // 2], vi[len(vi) // 2:]
    out = {}
    for name, (kind, make) in (heads or detector_heads()).items():
        Xv, Xr = feats[kind]
        X = np.vstack([Xv[vtr], Xr[expo_idx]])
        y = np.r_[np.zeros(len(vtr)), np.ones(len(expo_idx))]
        clf = make().fit(X, y)
        thr = float(np.quantile(clf.predict_proba(Xv[vth])[:, 1], 1 - target_far))
        sc = clf.predict_proba(Xr[test_idx])[:, 1]
        out[name] = (sc >= thr, sc, thr)
    return out


def seed_average(scores: list[np.ndarray], thresholds: list[float]):
    """Operating regime (i): average the seeds' Sentinel scores, threshold at the
    averaged 1%-FAR anchor. Returns (flags, avg_scores, avg_threshold)."""
    s = np.mean(scores, axis=0)
    thr = float(np.mean(thresholds))
    return s >= thr, s, thr


def seed_union(scores: list[np.ndarray], thresholds: list[float]):
    """Operating regime (iii): flag anything caught by ANY seed at its own anchor.
    Effective FAR ~ n_seeds x target. Returns boolean flags."""
    det = np.zeros(len(scores[0]), bool)
    for s, t in zip(scores, thresholds):
        det |= (s >= t)
    return det
