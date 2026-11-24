"""Leave-class-out study in the service condition (redshift as catalogued, phase withheld): object features from
build_novelty.py --dump-features, novae as a known class, halves swapped (study_svA, study_svB), layout of heads.
Only the heads on [embedding + descriptors] and [.. + LUNA columns] are meaningful here (the dispersion block is zero)."""
import shutil, numpy as np
from pathlib import Path
R = Path(__import__("os").environ.get("ASTRANET_REPO", Path(__file__).resolve().parents[2])); O = np.load(R / "study/service/svc_object_features.npz", allow_pickle=True); B = np.load(R / "study/objects/object_features.npz"); NOVA = ("Novae", "Classical Nova"); W = 407
for s in ("val", "test", "rare"): assert (O[s + "_ids"] == B[s + "_ids"]).all()
assert (O["rare_classes"] == B["rare_classes"]).all(); lab = B["rare_classes"]; nv = np.flatnonzero(np.isin(lab, NOVA)); keep = np.flatnonzero(~np.isin(lab, NOVA)); p = np.random.default_rng(0).permutation(nv); half = [p[:len(p) // 2], p[len(p) // 2:]]
D = {s: dict(ids=O[s + "_ids"], counts=B[s + "_counts"], embedding=O[s + "_X"][:, :384], embphys=O[s + "_X"][:, :W], summary=np.hstack([np.zeros((len(O[s + "_X"]), 2 * W), "float32"), O[s + "_X"][:, W:]])) for s in ("val", "test", "rare")}
for name, tofit, totest in (("A", half[0], half[1]), ("B", half[1], half[0])):
    V = R / f"study_sv{name}"; (V / "objects").mkdir(parents=True, exist_ok=True); (V / "heads").mkdir(exist_ok=True); out = {k: B[k] for k in B.files if k.startswith("train_")}
    for fn in ("evaluate.py", "_studypaths.py"): shutil.copy(R / "study" / fn, V / fn)
    for a in ("objects/experiment.py", "heads/experiment.py"): shutil.copy(R / "study" / a, V / a)
    for k in ("ids", "counts", "embedding", "embphys", "summary"):
        out["rare_" + k] = D["rare"][k][keep]; out["val_" + k] = np.concatenate([D["val"][k], D["rare"][k][tofit]]); out["test_" + k] = np.concatenate([D["test"][k], D["rare"][k][totest]])
    out["rare_classes"] = lab[keep]; np.savez_compressed(V / "objects/object_features.npz", **out); print(V.name, out["val_embphys"].shape, out["val_summary"].shape, len(out["rare_ids"]), flush=True)
print("SVC VARIANTS DONE")
