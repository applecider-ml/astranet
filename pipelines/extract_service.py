"""Ensemble outputs in the service condition: redshift as catalogued, spectroscopic phase withheld (SkyPortal passes none).
For val, test and the rare set: emb (384), proj_fine (256, unit norm per backbone), fine and coarse logits of the 12 z-shift members, combined as in
build_combo_cache.py and aligned with cache/astranet12/<split>_meta.pkl. The run with the phase given must reproduce
the cache (parity check of this path). Output: $ASTRANET_SERVICE_DIR (default cache/service)/svc_<split>.npz, the input of tools/build_novelty.py."""
import sys, os, pickle, numpy as np, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(REPO / "pipelines"), str(REPO)]; os.chdir(REPO)
from _common import load_config, make_astra_config, load_ensemble
from astranet.dataset import build_loaders
from astranet.model import build_hierarchy_mappings
from build_ensemble import _rare_loader
OUT = Path(os.environ.get("ASTRANET_SERVICE_DIR", REPO / "cache/service")); OUT.mkdir(parents=True, exist_ok=True); DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
KEYS = ("embedding", "proj_fine", "fine_logits", "coarse_logits"); CN = {"val": "val", "test": "test", "rare": "rare_unknown_combined"}; res = {}
for cfgn, spec in (("tcn", "tcn6"), ("spectranet", "spectranet6")):
    cfg = load_config(str(REPO / f"configs/{cfgn}.yaml")); ac = make_astra_config(cfg); ac.num_workers = 0; _, _, sup = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    ms, _ = load_ensemble(spec, REPO / "models" / cfg.get("weights_source", "tcn"), ac, len(sup), DEV); _, _, val, test, _ = build_loaders(ac)
    for name, dl in (("val", val), ("test", test), ("rare", _rare_loader(ac, REPO / "data/rare_unknown.pt"))):
        A = {c: {k: [] for k in KEYS} for c in ("full", "svc")}
        with torch.no_grad():
            for b in dl:
                x = b["x"].to(DEV); z = b["z"].to(DEV); ph = b["phase"].to(DEV)
                for c, pp in (("full", ph), ("svc", torch.full_like(ph, float("nan")))):
                    outs = [m(x, pp, z) for m in ms]
                    for k in KEYS:
                        v = torch.stack([o[k] for o in outs]).mean(0); A[c][k].append((torch.nn.functional.normalize(v, dim=-1) if k == "proj_fine" else v).cpu())
        A = {c: {k: torch.cat(v).numpy() for k, v in d.items()} for c, d in A.items()}; assert all(np.isfinite(v).all() for d in A.values() for v in d.values())
        res[(spec, name)] = (A, pickle.load(open(REPO / f"cache/{spec}/{CN[name]}_meta.pkl", "rb"))); print(spec, name, A["svc"]["embedding"].shape, flush=True)
    del ms
key = lambda m: (str(m["obj_id"]), str(m["file_path"]))
for name in ("val", "test", "rare"):
    (Aa, ma), (Ab, mb) = res[("tcn6", name)], res[("spectranet6", name)]; pos = {key(m): i for i, m in enumerate(mb)}; o = np.array([pos[key(m)] for m in ma])
    mc = pickle.load(open(REPO / f"cache/astranet12/{CN[name]}_meta.pkl", "rb")); assert [key(m) for m in mc] == [key(m) for m in ma]; c = np.load(REPO / f"cache/astranet12/{CN[name]}.npz"); out = {}
    for cond in ("full", "svc"):
        a, b = Aa[cond], Ab[cond]; out[cond] = dict(emb=np.hstack([a["embedding"], b["embedding"][o]]), proj_fine=np.hstack([a["proj_fine"], b["proj_fine"][o]]), fine_logits=(a["fine_logits"] + b["fine_logits"][o]) / 2, coarse_logits=(a["coarse_logits"] + b["coarse_logits"][o]) / 2)
    d = {k: float(np.abs(out["full"][k] - c[k]).max()) for k in out["full"]}; print(name, "phase given vs cache, max abs diff", d, "| mean |svc - full| emb", float(np.abs(out["svc"]["emb"] - out["full"]["emb"]).mean()), flush=True); assert max(d.values()) < 2e-2
    np.savez(OUT / f"svc_{name}.npz", **{k: v.astype("float32") for k, v in out["svc"].items()})
print("SERVICE EXTRACT DONE", flush=True)
