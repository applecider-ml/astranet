"""Confusion matrix and statistics of the novelty stage with novae as a known class and one threshold per redshift stratum.
Leave-class-out, ET+LUNA head, 10 splits x 2 nova half-swaps (study_nvA, study_nvB). Each rare class is scored by the head
that never saw it; each nova is scored when it sits in the held-out half; ordinary test objects are averaged over heads.
Output of the system for an object: "alarm" when the novelty score passes its threshold, else the class of the ensemble."""
import json, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R = Path(__import__("os").environ.get("ASTRANET_REPO", Path(__file__).resolve().parents[2])); RES = R / "study/novelty/out"; S = R / "study"; F = np.load(S / "features.npz"); O = np.load(S / "objects/object_features.npz"); NOVA = ("Novae", "Classical Nova")
CL = ["SN Ia", "SN Ib/c", "SLSN", "SN II", "AGN", "CV", "TDE"]; AL = (0.01, 0.05, 0.10, 0.15); HEAD = "et_plus_luna"; ZK, PRED, TRUE = {}, {}, {}
for s, cname in (("val", "val"), ("test", "test"), ("rare", "rare_unknown_combined")):
    m = pickle.load(open(R / f"cache/astranet12/{cname}_meta.pkl", "rb")); c = np.load(R / f"cache/astranet12/{cname}.npz"); zz = np.array([x.get("z") if x.get("z") is not None else np.nan for x in m], float); has = np.isfinite(zz) & (zz > 0.001)
    l = c["fine_logits"].astype(float); p = np.exp(l - l.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); u, inv = np.unique(F[s + "_objects"], return_inverse=True); assert (u == O[s + "_ids"]).all()
    for i, o in enumerate(u):
        ZK[str(o)] = bool(has[inv == i].any()); PRED[str(o)] = int(p[inv == i].mean(0).argmax())
        if s == "test": TRUE[str(o)] = CL[int(c["labels"][inv == i][0])]
for o, k in zip(O["rare_ids"], O["rare_classes"]): TRUE[str(o)] = "Nova" if k in NOVA else str(k)
A = {a: {} for a in AL}; PS = {a: [] for a in AL}
for sd in ("study_svA", "study_svB"):
    d = np.load(R / sd / "objects/object_features.npz"); lab = d["rare_classes"]; cls = np.unique(lab); zk = lambda ids: np.array([ZK[str(o)] for o in ids]); zv, zt, zr = zk(d["val_ids"]), zk(d["test_ids"]), zk(d["rare_ids"])
    for a in AL:
        sT = np.zeros(len(d["test_ids"])); nT = 0; sR = np.zeros(len(lab)); nR = np.zeros(len(lab)); per = []
        for seed in range(10):
            det = np.full(len(lab), np.nan)
            for i in range(len(cls)):
                f = np.load(R / sd / "heads/cache" / f"{seed:02d}_class{i}.npz"); j = list(f["names"]).index(HEAD); cal, held = f["id_cal"], f["held"]; t = np.full(2, np.inf)
                for b in (0, 1):
                    vc = np.sort(f["cal"][zv[cal].astype(int) == b, j]); n = len(vc)
                    if n: t[b] = vc[min(int(np.ceil((n + 1) * (1 - a))), n) - 1]
                sT += f["test"][:, j] > t[zt.astype(int)]; nT += 1; det[held] = f["rare"][:, j] > t[zr[held].astype(int)]
            sR += det; nR += 1; per.append(det.mean())
        PS[a].append(per)
        for o, v in zip(d["test_ids"], sT / nT): A[a].setdefault(str(o), []).append(v)
        for o, v in zip(d["rare_ids"], sR / nR): A[a].setdefault(str(o), []).append(v)
RARE = sorted({v for v in TRUE.values() if v not in CL and v != "Nova"}); ROWS = CL + ["Nova"] + RARE; COLS = CL + ["alarm"]; out = {}; M = {}
L = ["Service condition (redshift as catalogued, phase withheld). Novelty stage with novae as a known class, one threshold per redshift stratum. Leave-class-out, ET+LUNA, 10 splits x 2 nova half-swaps.", ""]
ST = ["| target rate | recall on rare objects [min, max over splits] | false alarms on ordinary test objects | novae that alarm | objects flagged: rare / ordinary / novae (of 60 / 1666 / 47) | precision in this test mix | balanced accuracy |", "|---|---|---|---|---|---|---|"]
for a in AL:
    al = {o: float(np.mean(v)) for o, v in A[a].items()}; mat = np.zeros((len(ROWS), len(COLS))); n = np.zeros(len(ROWS), int)
    for o, v in al.items():
        if o not in TRUE: continue
        r = ROWS.index(TRUE[o]); n[r] += 1; mat[r, -1] += v; mat[r, PRED[o]] += 1 - v
    M[a] = (mat / n[:, None], n); ir = np.array([ROWS.index(k) for k in RARE]); ii = np.arange(7); tp = mat[ir, -1].sum(); fp_id = mat[ii, -1].sum(); fp_nv = mat[7, -1]; per = np.mean(PS[a], 0)
    rec = tp / n[ir].sum(); far = fp_id / n[ii].sum(); spec = 1 - (fp_id + fp_nv) / (n[ii].sum() + n[7]); prec = tp / (tp + fp_id + fp_nv)
    ST.append(f"| {int(a * 100)}% | {rec:.2f} [{per.min():.2f}, {per.max():.2f}] | {far * 100:.1f}% | {fp_nv / n[7] * 100:.0f}% | {tp:.0f} / {fp_id:.0f} / {fp_nv:.0f} | {prec:.2f} | {(rec + spec) / 2:.2f} |")
    out[str(a)] = dict(recall=float(rec), recall_min=float(per.min()), recall_max=float(per.max()), far_ordinary=float(far), novae_flagged=float(fp_nv / n[7]), tp=float(tp), fp_ordinary=float(fp_id), fp_novae=float(fp_nv), precision=float(prec), balanced_accuracy=float((rec + spec) / 2),
                       alarm_rate_by_true_class={k: float(M[a][0][i, -1]) for i, k in enumerate(ROWS)}, n=dict(zip(ROWS, map(int, n))), matrix=M[a][0].tolist(), rows=ROWS, cols=COLS)
L += ST + ["", "Share of objects that alarm, by true class:", "", "| true class (objects) | " + " | ".join(f"{int(a * 100)}%" for a in AL) + " |", "|---|" + "---|" * len(AL)]
for i, k in enumerate(ROWS): L.append(f"| {k} ({M[AL[0]][1][i]}) | " + " | ".join(f"{M[a][0][i, -1]:.2f}" for a in AL) + " |")
for a in (0.05, 0.15):
    L += ["", f"Confusion matrix at {int(a * 100)}% (rows: true class, row-normalised; columns: output of the system):", "", "| true class (objects) | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
    for i, k in enumerate(ROWS): L.append(f"| {k} ({M[a][1][i]}) | " + " | ".join(f"{v:.2f}" for v in M[a][0][i]) + " |")
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True)
for x, a in zip(ax, (0.05, 0.15)):
    mat, n = M[a]; x.imshow(mat, cmap="magma", vmin=0, vmax=1, aspect="auto"); x.set_xticks(range(len(COLS))); x.set_xticklabels(COLS, rotation=40, ha="right"); x.set_yticks(range(len(ROWS))); x.set_yticklabels([f"{k} ({m})" for k, m in zip(ROWS, n)])
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if mat[i, j] >= 0.005: x.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black" if mat[i, j] > 0.6 else "white")
    x.axhline(6.5, color="white", lw=1.2); x.axhline(7.5, color="white", lw=1.2); x.axvline(6.5, color="white", lw=1.2); x.set_title(f"target alarm rate {int(a * 100)}%: recall {out[str(a)]['recall']:.2f}, ordinary flagged {out[str(a)]['far_ordinary'] * 100:.1f}%, novae flagged {out[str(a)]['novae_flagged'] * 100:.0f}%", fontsize=10); x.set_xlabel("output of the system")
ax[0].set_ylabel("true class (objects)"); fig.tight_layout(); fig.savefig(RES / "service_by_class.png", dpi=170)
print("\n".join(L)); open(RES / "service_by_class.md", "w").write("\n".join(L) + "\n"); json.dump(out, open(RES / "service_by_class.json", "w"), indent=1)
