#!/usr/bin/env python3
"""Build the novelty bundle of the service (models/novelty_v1.joblib).

    python tools/build_novelty.py --service-dir <dir with svc_{val,test,rare}.npz> --out models/novelty_v1.joblib

Inputs: cache/v7fixcombo12 (training rows: reference of LUNA; meta of val, test, rare), the ensemble outputs in the
service condition (pipelines/extract_service.py: redshift as catalogued, phase withheld) and the flux tensors of data/.
Ordinary objects: the validation objects and the novae of the rare set. Exposed anomalies: the other rare classes.
The ExtraTrees head is fitted on all of them; its thresholds come from out-of-fold scores of the ordinary objects
(5 folds), one set per redshift stratum. The test objects enter nowhere and measure the realised alarm rates.
"""
import argparse, json, pickle, sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np, torch, joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(REPO / "pipelines"), str(REPO)]
from _common import load_cache, load_config, make_astra_config                      # noqa: E402
from astranet.model import build_hierarchy_mappings                                  # noqa: E402
from astranet.sentinel import luna_dict                                              # noqa: E402
from astranet.novelty import Novelty, descriptors, redshift_known, conformal_threshold, RATES  # noqa: E402
from luna import LUNAPipeline                                                        # noqa: E402
import luna                                                                          # noqa: E402

NOVA = ("Novae", "Classical Nova")


def head():
    return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=4)


def aligned_flux(raw, meta, rare):
    ids = np.asarray(raw["obj_id"]).astype(str) if rare else np.array([str(m["obj_id"]) for m in raw["meta"]])
    files = np.asarray(raw["file_path"]).astype(str) if rare else np.array([str(m["file_path"]) for m in raw["meta"]])
    index = defaultdict(deque)
    for i, k in enumerate(zip(ids, files)): index[k].append(i)
    order = np.array([index[(str(m["obj_id"]), str(m["file_path"]))].popleft() for m in meta])
    X = np.asarray(raw["flux"] if rare else raw["X"], dtype=np.float32).reshape(len(ids), -1)[order]; assert X.shape[1] == 4096
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "cache/v7fixcombo12")); ap.add_argument("--data", default=str(REPO / "data")); ap.add_argument("--service-dir", required=True)
    ap.add_argument("--out", default=str(REPO / "models/novelty_v1.joblib")); ap.add_argument("--report", default=str(REPO / "results/novelty_service.json")); ap.add_argument("--dump-features")
    a = ap.parse_args(); cache, data = Path(a.cache), Path(a.data)
    ac = make_astra_config(load_config(str(REPO / "configs/v7fixsn.yaml"))); f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    train = load_cache(cache, "train"); pipe = LUNAPipeline(combiner="rank", fine_to_coarse=f2c); pipe.fit(luna_dict(train)); methods = list(pipe.active_methods)
    Dtr = descriptors(aligned_flux(torch.load(data / "object-split-v7fix/train.pt", weights_only=False, map_location="cpu"), train["meta"], False))
    med = np.median(Dtr, 0); iqr = np.diff(np.percentile(Dtr, [25, 75], axis=0), axis=0)[0] + 1e-6
    nov = Novelty(dict(luna=pipe, methods=methods, phys_med=med, phys_iqr=iqr)); O = {}
    for s, cname, rawf in (("val", "val", "object-split-v7fix/val.pt"), ("test", "test", "object-split-v7fix/test.pt"), ("rare", "rare_unknown_combined", "rare_unknown.pt")):
        meta = pickle.load(open(cache / f"{cname}_meta.pkl", "rb")); f = np.load(Path(a.service_dir) / f"svc_{s}.npz"); raw = torch.load(data / rawf, weights_only=False, map_location="cpu")
        X = aligned_flux(raw, meta, s == "rare"); S = nov.luna_scores(f["emb"], f["proj_fine"], f["fine_logits"], f["coarse_logits"]); P = nov.scaled_descriptors(X)
        ids = np.array([str(m["obj_id"]) for m in meta]); z = np.array([m.get("z") if m.get("z") is not None else np.nan for m in meta], float); u, inv = np.unique(ids, return_inverse=True)
        O[s] = dict(ids=u, X=np.stack([Novelty.object_vector(f["emb"][inv == i], P[inv == i], S[inv == i]) for i in range(len(u))]), zk=np.array([any(redshift_known(v) for v in z[inv == i]) for i in range(len(u))]))
        if s == "rare":
            cl = np.array([str(m["class"]) for m in meta]); O[s]["classes"] = np.array([cl[inv == i][0] for i in range(len(u))])
        print(s, O[s]["X"].shape, "redshift known", int(O[s]["zk"].sum()), flush=True)
    if a.dump_features:
        np.savez_compressed(a.dump_features, **{f"{s}_{k}": v for s, d in O.items() for k, v in d.items()})
    isn = np.isin(O["rare"]["classes"], NOVA); Xid = np.vstack([O["val"]["X"], O["rare"]["X"][isn]]); zid = np.r_[O["val"]["zk"], O["rare"]["zk"][isn]]; Xex = O["rare"]["X"][~isn]
    X = np.vstack([Xid, Xex]); y = np.r_[np.zeros(len(Xid)), np.ones(len(Xex))]; oof = np.full(len(X), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y): oof[te] = head().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    cal = {"redshift": np.sort(oof[:len(Xid)][zid]), "no_redshift": np.sort(oof[:len(Xid)][~zid])}; thr = {k: {r: conformal_threshold(v, r) for r in RATES} for k, v in cal.items()}
    clf = head().fit(X, y); st = clf.predict_proba(O["test"]["X"])[:, 1]; zt = O["test"]["zk"]; tk = np.where(zt, "redshift", "no_redshift")
    rep = dict(ordinary_objects=int(len(Xid)), novae_among_ordinary=int(isn.sum()), exposed_rare_objects=int(len(Xex)), exposed_classes=sorted(set(O["rare"]["classes"][~isn])),
               calibration={k: int(len(v)) for k, v in cal.items()}, out_of_fold_recall_exposed={f"{int(r * 100)}%": float(np.mean(oof[len(Xid):] > np.array([thr["redshift" if k else "no_redshift"][r] for k in O["rare"]["zk"][~isn]]))) for r in RATES},
               test_objects=dict(n=int(len(st)), n_no_redshift=int((~zt).sum()), alarm_rate={f"{int(r * 100)}%": dict(all=float(np.mean(st > np.array([thr[k][r] for k in tk]))), redshift=float(np.mean(st[zt] > thr["redshift"][r])), no_redshift=float(np.mean(st[~zt] > thr["no_redshift"][r]))) for r in RATES}))
    bundle = dict(version="novelty_v1", luna=pipe, methods=methods, phys_med=med, phys_iqr=iqr, head=clf, calibration=cal, thresholds=thr, rates=RATES, luna_version=luna.__version__,
                  condition="redshift as catalogued, phase withheld", info=rep)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, a.out, compress=3); Path(a.report).parent.mkdir(parents=True, exist_ok=True); json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(rep, indent=1)); print("bundle", a.out, f"{Path(a.out).stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
