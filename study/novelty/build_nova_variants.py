"""Novae as an ordinary class in the novelty heads. The 47 nova objects (Novae, Classical Nova) leave the rare set and
join the ordinary objects: one half goes to the validation objects (head fitting and threshold calibration), the other
half to the test objects. study_nvA and study_nvB swap the halves."""
import shutil, numpy as np
from pathlib import Path
R = Path(__import__("os").environ.get("ASTRANET_REPO", Path(__file__).resolve().parents[2])); NOVA = ("Novae", "Classical Nova"); KEYS = ("ids", "counts", "embedding", "embphys", "summary")
for src, tag in (("study", "nv"),):
    O = dict(np.load(R / src / "objects/object_features.npz")); lab = O["rare_classes"]; nv = np.flatnonzero(np.isin(lab, NOVA)); keep = np.flatnonzero(~np.isin(lab, NOVA))
    p = np.random.default_rng(0).permutation(nv); half = [p[:len(p) // 2], p[len(p) // 2:]]
    for name, tofit, totest in (("A", half[0], half[1]), ("B", half[1], half[0])):
        V = R / f"study_{tag}{name}"; (V / "objects").mkdir(parents=True, exist_ok=True); (V / "heads").mkdir(exist_ok=True); out = dict(O)
        for fn in ("evaluate.py", "_studypaths.py"): shutil.copy(R / "study" / fn, V / fn)
        for a in ("objects/experiment.py", "heads/experiment.py"): shutil.copy(R / "study" / a, V / a)
        for k in KEYS:
            out["rare_" + k] = O["rare_" + k][keep]
            if tofit is not None: out["val_" + k] = np.concatenate([O["val_" + k], O["rare_" + k][tofit]]); out["test_" + k] = np.concatenate([O["test_" + k], O["rare_" + k][totest]])
        out["rare_classes"] = lab[keep]; assert len(set(out["val_ids"]) & set(out["test_ids"])) == 0 and len(set(out["rare_ids"]) & (set(out["val_ids"]) | set(out["test_ids"]))) == 0
        np.savez_compressed(V / "objects/object_features.npz", **out); print(V.name, "val", len(out["val_ids"]), "test", len(out["test_ids"]), "rare", len(out["rare_ids"]), np.unique(out["rare_classes"]).tolist(), flush=True)
print("NOVA VARIANTS DONE", flush=True)
