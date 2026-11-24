"""Examples table and counts for the paper: 60 rare objects (novae ordinary), one threshold per
redshift stratum, zero-exposure mean of 16 percentiles against maximum softmax, nominal 1%. Uses the conformal sets of
study/conformal_object/examples/rare_objects_conformal.csv. -> study/novelty/out/examples.{json,tex}"""
import json, csv, pickle, numpy as np
from pathlib import Path
R = Path(__import__("os").environ.get("ASTRANET_REPO", Path(__file__).resolve().parents[2])); F = dict(np.load(R / "study/features.npz")); O = np.load(R / "study/objects/object_features.npz"); M = list(F["emb_methods"]); ZK = {}
for cname in ("val", "test", "rare_unknown_combined"):
    m = pickle.load(open(R / f"cache/astranet12/{cname}_meta.pkl", "rb")); zz = np.array([x.get("z") if x.get("z") is not None else np.nan for x in m], float); has = np.isfinite(zz) & (zz > 0.001); ids = np.array([str(x["obj_id"]) for x in m])
    for o in np.unique(ids): ZK[o] = bool(has[ids == o].any())
SP = {}; POOL = np.vstack([F[s + "_emb_scores"] for s in ("val", "test", "rare")]); off = 0
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
D = {"mean16": [], "msp": []}; RID = None
for sd in ("study_nvA", "study_nvB"):
    d = dict(np.load(R / sd / "objects/object_features.npz")); rid = d["rare_ids"].astype(str); RID = rid; vid = d["val_ids"].astype(str); zv, zr = (np.array([ZK[o] for o in x]) for x in (vid, rid))
    for seed in range(10):
        f1 = np.load(R / sd / "heads/cache" / f"{seed:02d}_Tier1.npz"); fit, cal = f1["id_fit"], f1["id_cal"]; ref = POOL[np.concatenate([SP[o] for o in vid[fit]])]
        fn = {"mean16": lambda o: np.column_stack([ecdf(ref[:, j], POOL[SP[o], j]) for j in range(16)]).mean(1).mean(), "msp": lambda o: POOL[SP[o], M.index("msp")].mean()}
        for k in D:
            sc, sr = (np.array([fn[k](o) for o in x]) for x in (vid[cal], rid)); t = thr(sc, zv[cal], 0.01); D[k].append(sr > t[zr.astype(int)])
fl = {k: dict(zip(RID, np.mean(v, 0))) for k, v in D.items()}
C = {r["obj"]: r for r in csv.DictReader(open(R / "study/conformal_object/examples/rare_objects_conformal.csv"))}
T = np.load(R / "study/conformal_object/objects.npz"); tc = T["p"].max(1); corr = T["p"].argmax(1) == T["y"]
rows = [dict(C[o], luna=float(fl["mean16"][o]), msp=float(fl["msp"][o]), zknown=ZK[o], id_lower=float((tc < float(C[o]["p"])).mean())) for o in RID]
size = lambda s: 0 if s == "{}" else len(s.strip("{}").split(","))
lo = [r for r in rows if r["luna"] >= .5 and r["msp"] < .5]; mo = [r for r in rows if r["msp"] >= .5 and r["luna"] < .5]; bo = [r for r in rows if r["msp"] >= .5 and r["luna"] >= .5]
p = np.array([float(r["p"]) for r in lo]); out = dict(n_rare=len(rows), luna_only=len(lo), msp_only=len(mo), both=len(bo), p_range=[float(p.min()), float(p.max())], p_median=float(np.median(p)), id_median_conf=float(np.median(tc)),
    id_lower_range=[min(r["id_lower"] for r in lo), max(r["id_lower"] for r in lo)], global_empty=sum(size(r["set_global"]) == 0 for r in lo), global_single=sum(size(r["set_global"]) == 1 for r in lo),
    joint_sizes=[size(r["set_joint"]) for r in lo], joint_mean=float(np.mean([size(r["set_joint"]) for r in lo])), cells={f"L{r['luna_stratum']}C{r['conf_stratum']}": 0 for r in lo}, objects=rows)
for r in lo: out["cells"][f"L{r['luna_stratum']}C{r['conf_stratum']}"] += 1
sel = sorted([r for r in lo if r["luna"] >= .9 and r["msp"] == 0], key=lambda r: -float(r["p"]))[:7]; NAMES = ["Ia", "Ibc", "SLSN", "II", "AGN", "CV", "TDE"]; lab = {"afterglow": "GRB afterglow", "long GRB": "Long GRB", "Luminous Red Nova": "LRN"}; tex = []
for r in sel:
    j = r["set_joint"].strip("{}").split(","); g = r["set_global"].replace("{", "\\{").replace("}", "\\}") if size(r["set_global"]) else "$\\emptyset$"
    tex.append(f"\\texttt{{{r['obj']}}} & {lab.get(r['cls'], r['cls'])} & {r['epochs']} & {r['pred']} ${float(r['p']):.2f} \\pm {float(r['p_sd']):.2f}$ & ${100 * r['id_lower']:.1f}\\%$ & {round(20 * r['luna'])}/{round(20 * r['msp'])} & {g} & " + " & ".join("$+$" if k in j else "$-$" for k in NAMES) + " \\\\")
RES = R / "study/novelty/out"; json.dump(out, open(RES / "examples.json", "w"), indent=1); open(RES / "examples.tex", "w").write("\n".join(tex) + "\n")
print({k: v for k, v in out.items() if k != "objects"}); print("\n".join(tex))
for o in ("ZTF26aaglqak", "ZTF21abmkuji", "ZTF21aayokph"): print([ (k, r[k]) for r in rows if r["obj"] == o for k in ("obj", "cls", "pred", "p", "luna", "msp", "set_global", "set_joint", "zknown")])
print("EXAMPLES DONE")
