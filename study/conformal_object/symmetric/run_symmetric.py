"""Symmetric stratification control. Reads ../objects.npz and the
ASTRAnet tcn6 validation cache; writes only into this directory."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
from pathlib import Path
import pickle, math, json, csv, hashlib
import numpy as np
from scipy.special import softmax
ROOT = Path(__file__).resolve().parent; PARENT = ROOT.parent
CACHE = (REPO/'cache/astranet12')
SEEDS = [20261011] + list(range(10)); ALPHAS = [.1, .05]
d = np.load(PARENT/'objects.npz'); P, y, ids = d['p'], d['y'], d['ids']
v = np.load(CACHE/'val.npz'); m = pickle.load(open(CACHE/'val_meta.pkl', 'rb'))
_, inv, n = np.unique([str(x['obj_id']) for x in m], return_inverse=True, return_counts=True)
pv = np.zeros((len(n), 7)); np.add.at(pv, inv, softmax(v['fine_logits'].astype(float), axis=1)); pv /= n[:, None]
conf_cut = float(np.quantile(1 - pv.max(1), .8))
luna = d['stratum'].astype(int); conf = (1 - P.max(1) > conf_cut).astype(int)
STRATA = {'global': np.zeros(len(y), int), 'luna': luna, 'confidence': conf, 'joint': 2*luna + conf}
SUBSETS = {'all': np.ones(len(y), bool), 'luna_high': luna == 1, 'confidence_high': conf == 1,
           'luna_high_confidence_low': (luna == 1) & (conf == 0), 'confidence_high_luna_low': (conf == 1) & (luna == 0),
           'misclassified': P.argmax(1) != y}
def q(s, alpha):
    k = math.ceil((len(s)+1)*(1-alpha)); return float(np.sort(s)[k-1]) if 0 < k <= len(s) else float('inf')
rows, cells, thr = [], [], []
for seed in SEEDS:
    ix = np.random.default_rng(seed).permutation(len(y)); cal, ev = ix[:len(ix)//2], ix[len(ix)//2:]
    # same assignment as ../run.py / ../assignments.csv
    for c in range(4):
        cells.append(dict(seed=seed, cell=['luna_low_conf_low', 'luna_low_conf_high', 'luna_high_conf_low', 'luna_high_conf_high'][c],
                          n_calibration=int((STRATA['joint'][cal] == c).sum()), n_evaluation=int((STRATA['joint'][ev] == c).sum())))
    score = 1 - P[cal, y[cal]]
    for alpha in ALPHAS:
        for meth, g in STRATA.items():
            t = np.array([q(score[g[cal] == k], alpha) for k in range(g.max()+1)])
            thr.append(dict(seed=seed, alpha=alpha, method=meth, thresholds=[None if np.isinf(x) else x for x in t.tolist()],
                            support=[int((g[cal] == k).sum()) for k in range(g.max()+1)]))
            sets = 1 - P[ev] <= t[g[ev], None]; hit = sets[np.arange(len(ev)), y[ev]]; size = sets.sum(1)
            for sub, mask in SUBSETS.items():
                mk = mask[ev]
                rows.append(dict(seed=seed, alpha=alpha, method=meth, subset=sub, n=int(mk.sum()),
                                 coverage=float(hit[mk].mean()), mean_size=float(size[mk].mean()), empty=float((size[mk] == 0).mean())))
summary = []
for alpha in ALPHAS:
    for meth in STRATA:
        for sub in SUBSETS:
            r = [x for x in rows if x['alpha'] == alpha and x['method'] == meth and x['subset'] == sub]
            c = np.array([x['coverage'] for x in r]); s = np.array([x['mean_size'] for x in r]); e = np.array([x['empty'] for x in r])
            summary.append(dict(alpha=alpha, method=meth, subset=sub, n_mean=float(np.mean([x['n'] for x in r])),
                                coverage_mean=c.mean(), coverage_sd=c.std(ddof=1), coverage_min=c.min(), coverage_max=c.max(),
                                size_mean=s.mean(), size_sd=s.std(ddof=1), empty_mean=e.mean()))
def write(name, data):
    with open(ROOT/name, 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
write('metrics_symmetric.csv', rows); write('summary_symmetric.csv', summary); write('cells.csv', cells)
(ROOT/'thresholds_symmetric.json').write_text(json.dumps(thr, indent=1))
audit = dict(confidence_cut=conf_cut, luna_high_fraction=float(luna.mean()), confidence_high_fraction=float(conf.mean()),
             jaccard_high=float((luna & conf).sum() / (luna | conf).sum()), misclassified_fraction=float((P.argmax(1) != y).mean()),
             assignments='identical permutations to ../assignments.csv (np.random.default_rng(seed).permutation(1668))',
             sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [PARENT/'objects.npz', CACHE/'val.npz', CACHE/'val_meta.pkl', Path(__file__)]})
(ROOT/'audit_symmetric.json').write_text(json.dumps(audit, indent=1))
print(json.dumps(audit, indent=1))
for x in summary:
    if x['alpha'] == .1: print(f"{x['method']:10} {x['subset']:26} n={x['n_mean']:5.0f} cov {100*x['coverage_mean']:5.1f}±{100*x['coverage_sd']:4.1f} [{100*x['coverage_min']:5.1f},{100*x['coverage_max']:5.1f}] size {x['size_mean']:.2f} empty {100*x['empty_mean']:4.1f}%")
