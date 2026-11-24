#!/usr/bin/env python3
"""Does the choice of combiner matter?  Same features (classifier embedding + classifier-free physics
descriptors), same folds, same 1%-FAR rule on a held-out validation half, ten validation splits.
  discovery  leave-class-out: trained on validation-half (ID) vs every OTHER rare class, tested on the held-out class
  recovery   object-grouped 5-fold on the base encoder (no outlier exposure): trained on the other folds' rare objects
Combiners: LightGBM, XGBoost, random forest, extra trees, logistic regression, MLP, RBF-SVM, and the rank
average of LightGBM + logistic + MLP.  Usage: python pipelines/combiner_comparison.py --spec v7fixmd6 --config configs/v7fixmd.yaml
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import numpy as np, torch
import lightgbm as lgb, xgboost as xgb
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from _common import REPO, load_cache, load_config, resolve_path, tiered_report
from astranet.sentinel import LGB_PARAMS
from physics_view import descriptors


def models(pos_weight):
    return {
        "LightGBM": lambda: lgb.LGBMClassifier(**LGB_PARAMS),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                             scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=8, verbosity=0),
        "random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=8, random_state=0),
        "extra trees": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=8, random_state=0),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=500, early_stopping=True, random_state=0)),
        "RBF-SVM": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", class_weight="balanced")),
    }


def score(clf, X):
    return clf.decision_function(X) if not hasattr(clf, "predict_proba") else clf.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="v7fixmd6"); ap.add_argument("--config", default=None)
    ap.add_argument("--splits", type=int, default=10); args = ap.parse_args()
    cfg = load_config(args.config)
    dd = Path(resolve_path(cfg["data"]["dir"])); TR = torch.load(dd / "train.pt", weights_only=False); VA = torch.load(dd / "val.pt", weights_only=False)
    rd = torch.load(resolve_path(cfg["data"]["rare_pt"]), weights_only=False); cls = np.array(rd["class"]); obj = np.array(list(map(str, rd["obj_id"])))
    Dtr, Dva, Dra = descriptors(TR["X"]), descriptors(VA["X"]), descriptors(rd["flux"])
    med, iqr = np.median(Dtr, 0), np.subtract(*np.percentile(Dtr, [75, 25], 0)) + 1e-6
    Z = lambda D: np.clip((D - med) / iqr, -20, 20).astype(np.float32)  # noqa: E731
    c = REPO / "cache" / args.spec; va, ra = (load_cache(c, n) for n in ("val", "rare_unknown_combined"))
    Xv = np.hstack([va["emb"].astype(np.float32), Z(Dva)]); Xr = np.hstack([ra["emb"].astype(np.float32), Z(Dra)])
    classes = sorted(set(cls))
    kfold = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(cls, cls, groups=obj))

    def objsum(det):
        o = collections.defaultdict(lambda: [False, "?"])
        for i in range(len(cls)): o[obj[i]][1] = cls[i]; o[obj[i]][0] |= bool(det[i])
        t = tiered_report((v[0], v[1]) for v in o.values()); return 100 * sum(v[0] for v in o.values()) / len(o), t["tier1"]["pct"]

    names = list(models(1.0)) + ["rank average (LightGBM, logistic, MLP)"]
    R = {p: collections.defaultdict(list) for p in ("discovery", "recovery")}
    for rep in range(args.splits):
        vi = np.random.default_rng(rep).permutation(len(Xv)); vtr, vth = vi[:len(vi) // 2], vi[len(vi) // 2:]
        for proto, folds in (("discovery", [(np.where(cls != C)[0], np.where(cls == C)[0]) for C in classes]), ("recovery", kfold)):
            det = {n: np.zeros(len(cls), bool) for n in names}
            for ex, te in folds:
                X = np.vstack([Xv[vtr], Xr[ex]]); y = np.r_[np.zeros(len(vtr)), np.ones(len(ex))]
                vs, rs = {}, {}
                for n, make in models(len(vtr) / len(ex)).items():
                    clf = make().fit(X, y); vs[n], rs[n] = score(clf, Xv[vth]), score(clf, Xr[te])
                    det[n][te] = rs[n] >= np.quantile(vs[n], 0.99)
                ens = ["LightGBM", "logistic", "MLP"]   # rank average, ranks taken against the held-out validation half
                ev = np.mean([rankdata(np.r_[vs[n], rs[n]])[:len(vth)] for n in ens], 0); er = np.mean([rankdata(np.r_[vs[n], rs[n]])[len(vth):] for n in ens], 0)
                det[names[-1]][te] = er >= np.quantile(ev, 0.99)
            for n in names: R[proto][n].append(objsum(det[n]))
        print(f"split {rep + 1}/{args.splits} done", flush=True)
    out = {p: {n: {"objects": float(np.mean([x[0] for x in v])), "objects_sd": float(np.std([x[0] for x in v])), "tier1": float(np.mean([x[1] for x in v]))}
               for n, v in R[p].items()} for p in R}
    (REPO / "results" / args.spec / "anomaly" / "combiner_comparison.json").write_text(json.dumps(out, indent=2))
    for p in out:
        print(f"\n== {p} ({'leave-class-out' if p == 'discovery' else 'object-grouped 5-fold, no OE'}), embedding + physics features, 1% FAR, {args.splits} splits")
        for n, v in sorted(out[p].items(), key=lambda kv: -kv[1]["objects"]):
            print(f"  {n:40s} objects {v['objects']:5.1f} ± {v['objects_sd']:3.1f}   Tier1 {v['tier1']:5.1f}")


if __name__ == "__main__":
    main()
