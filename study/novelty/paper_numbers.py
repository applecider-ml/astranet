"""Novelty numbers of the paper for the twelve-member model, with novae as an ordinary class (60 rare objects, nine
classes) and one threshold per redshift stratum. Ten calibration splits x two swaps of the nova halves (study_nvA/B).
  zero     no anomaly exposure: mean of the 16 reference percentiles (object = mean over epochs) and single scores
  lco      leave-class-out ExtraTrees, base and base + LUNA (heads caches)
  tier     whole tier withheld (Tier 1: 35 objects; Tier 2: QSO, blazar, BL Lac, 25 objects)
  known    object-grouped five-fold exposure to the same rare classes
Budgets 1, 5, 10, 15%. Paired object bootstrap (2000) for base + LUNA minus base. -> study/novelty/out/paper_numbers.json"""
import json, pickle, numpy as np
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
R = Path(__import__("os").environ.get("ASTRANET_REPO", Path(__file__).resolve().parents[2])); F = dict(np.load(R / "study/features.npz")); O = np.load(R / "study/objects/object_features.npz"); NOVA = ("Novae", "Classical Nova")
TIER1 = {"FBOT", "ILRT", "Luminous Red Nova", "Ca-rich", "afterglow", "long GRB"}; AL = (0.01, 0.05, 0.10, 0.15); M = list(F["emb_methods"]); ZK = {}
for s, cname in (("val", "val"), ("test", "test"), ("rare", "rare_unknown_combined")):
    m = pickle.load(open(R / f"cache/astranet12/{cname}_meta.pkl", "rb")); zz = np.array([x.get("z") if x.get("z") is not None else np.nan for x in m], float); has = np.isfinite(zz) & (zz > 0.001); ids = np.array([str(x["obj_id"]) for x in m]); assert (ids == F[s + "_objects"].astype(str)).all()
    for o in np.unique(ids): ZK[o] = bool(has[ids == o].any())
SP = {}                                                       # object -> rows of its spectra in the pooled score matrix
POOL = np.vstack([F[s + "_emb_scores"] for s in ("val", "test", "rare")]); off = 0
for s in ("val", "test", "rare"):
    ids = F[s + "_objects"].astype(str)
    for o in np.unique(ids): SP[o] = off + np.flatnonzero(ids == o)
    off += len(ids)
ecdf = lambda ref, x: np.searchsorted(np.sort(ref), x, side="right") / len(ref)
def thr(sc, zk, a):
    t = np.full(2, np.inf)
    for b in (0, 1):
        v = np.sort(sc[zk.astype(int) == b]); n = len(v)
        if n: t[b] = v[min(int(np.ceil((n + 1) * (1 - a))), n) - 1]
    return t
et = lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=8)
DET = {}; FAR = {}; NVF = {}                                  # DET[key][alpha] -> list over (sd, seed) of per-object flags (nan where not scored)
def put(key, a, flags, far, nvf): DET.setdefault(key, {}).setdefault(a, []).append(flags); FAR.setdefault(key, {}).setdefault(a, []).append(far); NVF.setdefault(key, {}).setdefault(a, []).append(nvf)
RID = None
for sd in ("study_nvA", "study_nvB"):
    d = dict(np.load(R / sd / "objects/object_features.npz")); lab = d["rare_classes"]; cls = np.unique(lab); rid = d["rare_ids"].astype(str); RID = rid if RID is None else RID; assert (rid == RID).all()
    w = d["val_embphys"].shape[1]; X = {s: {"base": d[s + "_embphys"], "luna": np.hstack([d[s + "_embphys"], d[s + "_summary"][:, 2 * w:]])} for s in ("val", "test", "rare")}
    vid, tid = d["val_ids"].astype(str), d["test_ids"].astype(str); zv, zt, zr = (np.array([ZK[o] for o in x]) for x in (vid, tid, rid)); tn = np.array([o in set(O["rare_ids"].astype(str)) for o in tid])
    for seed in range(10):
        f1 = np.load(R / sd / "heads/cache" / f"{seed:02d}_Tier1.npz"); fit, cal = f1["id_fit"], f1["id_cal"]
        # zero exposure
        ref = POOL[np.concatenate([SP[o] for o in vid[fit]])]; pct = lambda o: np.column_stack([ecdf(ref[:, j], POOL[SP[o], j]) for j in range(16)])
        cand = {"mean16": lambda o: pct(o).mean(1).mean()} | {k: (lambda o, j=M.index(k): POOL[SP[o], j].mean()) for k in ("typicality", "knn", "energy", "msp", "mahal_class", "mrs")}
        for k, fn in cand.items():
            sc, st, sr = (np.array([fn(o) for o in x]) for x in (vid[cal], tid, rid))
            for a in AL:
                t = thr(sc, zv[cal], a); al = st > t[zt.astype(int)]; put("zero/" + k, a, (sr > t[zr.astype(int)]).astype(float), al[~tn].mean(), al[tn].mean())
        # exposure heads from the heads caches
        for mode, keys in (("lco", [f"class{i}" for i in range(len(cls))]), ("Tier1", ["Tier1"]), ("Tier2", ["Tier2"])):
            for head in ("et", "et_plus_luna", "svm"):
                for a in AL:
                    det = np.full(len(lab), np.nan); fa, nv = [], []
                    for k in keys:
                        f = np.load(R / sd / "heads/cache" / f"{seed:02d}_{k}.npz"); j = list(f["names"]).index(head); held = f["held"]; t = thr(f["cal"][:, j], zv[f["id_cal"]], a)
                        al = f["test"][:, j] > t[zt.astype(int)]; det[held] = f["rare"][:, j] > t[zr[held].astype(int)]; fa.append(al[~tn].mean()); nv.append(al[tn].mean())
                    put(f"{mode}/{head}", a, det, np.mean(fa), np.mean(nv))
        # known rare classes: five folds over the rare objects
        for feat, name in (("base", "et"), ("luna", "et_plus_luna")):
            sr = np.full(len(lab), np.nan); sc = np.zeros(len(cal)); st = np.zeros(len(tid))
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(lab):
                c = et().fit(np.vstack([X["val"][feat][fit], X["rare"][feat][tr]]), np.r_[np.zeros(len(fit)), np.ones(len(tr))]); sr[te] = c.predict_proba(X["rare"][feat][te])[:, 1]
                sc += c.predict_proba(X["val"][feat][cal])[:, 1] / 5; st += c.predict_proba(X["test"][feat])[:, 1] / 5
            for a in AL:
                t = thr(sc, zv[cal], a); al = st > t[zt.astype(int)]; put(f"known/{name}", a, (sr > t[zr.astype(int)]).astype(float), al[~tn].mean(), al[tn].mean())
        print(sd, seed, flush=True)
lab = np.load(R / "study_nvA/objects/object_features.npz")["rare_classes"]; t1 = np.isin(lab, list(TIER1)); rng = np.random.default_rng(0); out = {"classes": {c: int((lab == c).sum()) for c in np.unique(lab)}, "n_tier1": int(t1.sum()), "n_tier2": int((~t1).sum())}
def split_means(key, a):                                      # 10 values: mean over the two swaps of the recall of each split
    A = np.array(DET[key][a]); n = len(A) // 2; return np.nanmean(np.stack([A[:n], A[n:]]), 0)
for key in DET:
    for a in AL:
        Dm = split_means(key, a); ok = np.isfinite(Dm[0]); rec = np.nanmean(Dm[:, ok], 1); per = np.nanmean(Dm, 0)
        out[f"{key}/{a}"] = dict(recall=float(rec.mean()), sd=float(rec.std()), n=int(ok.sum()), tier1=float(np.nanmean(Dm[:, ok & t1])) if (ok & t1).any() else None, tier2=float(np.nanmean(Dm[:, ok & ~t1])) if (ok & ~t1).any() else None,
                                 far=float(np.mean(FAR[key][a])), novae=float(np.nanmean(NVF[key][a])), per_class={c: float(np.nanmean(per[lab == c])) for c in np.unique(lab) if np.isfinite(per[lab == c]).any()})
for mode in ("lco", "Tier1", "Tier2", "known"):
    for a in AL:
        A, B = split_means(f"{mode}/et_plus_luna", a), split_means(f"{mode}/et", a); ok = np.isfinite(A[0]); dd = (A - B)[:, ok]; pm = dd.mean(0); bs = pm[rng.integers(0, len(pm), (2000, len(pm)))].mean(1)
        out[f"paired_luna/{mode}/{a}"] = dict(delta=float(100 * pm.mean()), lo=float(100 * np.percentile(bs, 2.5)), hi=float(100 * np.percentile(bs, 97.5)), splits_positive=int((dd.mean(1) > 0).sum()))
A, B = split_means("zero/mean16", 0.01), split_means("zero/msp", 0.01); pm = (A - B).mean(0); bs = pm[rng.integers(0, len(pm), (2000, len(pm)))].mean(1); out["paired_zero_mean16_vs_msp/0.01"] = dict(delta=float(100 * pm.mean()), lo=float(100 * np.percentile(bs, 2.5)), hi=float(100 * np.percentile(bs, 97.5)))
json.dump(out, open(R / "study/novelty/out/paper_numbers.json", "w"), indent=1)
for key in sorted(DET):
    print(f"{key:22s} " + " | ".join(f"{int(a * 100):2d}%: {100 * out[f'{key}/{a}']['recall']:5.1f}±{100 * out[f'{key}/{a}']['sd']:.1f} T1 {100 * (out[f'{key}/{a}']['tier1'] or 0):5.1f} T2 {100 * (out[f'{key}/{a}']['tier2'] or 0):5.1f} FAR {100 * out[f'{key}/{a}']['far']:.2f} nv {100 * out[f'{key}/{a}']['novae']:.0f}" for a in AL))
for k, v in out.items():
    if k.startswith("paired"): print(k, {a: (round(b, 1) if isinstance(b, float) else b) for a, b in v.items()})
print("per class lco/et 1%:", {c: round(100 * v) for c, v in out["lco/et/0.01"]["per_class"].items()}); print("PAPER NUMBERS V2 DONE")
