"""Rare objects that classifier confidence does not flag but zero-exposure LUNA does, with ensemble
probability +/- member spread and conformal prediction sets (primary split, symmetric-control thresholds)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
from pathlib import Path
import json, csv, pickle
import numpy as np
from scipy.special import softmax
ROOT = Path(__file__).resolve().parent; CO = ROOT.parent; STUDY = CO.parent
CACHE = (REPO/'cache/astranet12')
NAMES = ['Ia', 'Ibc', 'SLSN', 'II', 'AGN', 'CV', 'TDE']
F = np.load(STUDY/'features.npz'); R = np.load(CACHE/'rare_unknown_combined.npz'); M = np.load(CACHE/'per_model_softmax.npz')
meta = pickle.load(open(CACHE/'rare_unknown_combined_meta.pkl', 'rb'))
rid = F['rare_objects']; assert np.array_equal(R['emb'], F['rare_emb']) and np.array_equal(rid, np.array([str(x['obj_id']) for x in meta]))
def objmean(x, ids):
    o, inv, n = np.unique(ids, return_inverse=True, return_counts=True)
    a = np.zeros((len(o),) + x.shape[1:]); np.add.at(a, inv, x); return o, a / n.reshape((-1,) + (1,)*(x.ndim-1)), n
objs, P, nep = objmean(softmax(R['fine_logits'].astype(float), axis=1), rid)
_, Pm, _ = objmean(np.transpose(M['rare_softmax'], (1, 0, 2)), rid)          # (obj, member, class)
cls = np.array([str(meta[np.flatnonzero(rid == o)[0]]['class']) for o in objs])
# LUNA AD rank and strata exactly as ../run.py and ../symmetric/run_symmetric.py
_, vs, _ = objmean(F['val_emb_scores'], F['val_objects']); _, rs, _ = objmean(F['rare_emb_scores'], rid)
ad = np.mean([np.searchsorted(np.sort(vs[:, j]), rs[:, j], side='left') / (len(vs)+1) for j in range(16)], axis=0)
audit = json.load(open(CO/'audit.json')); sa = json.load(open(CO/'symmetric/audit_symmetric.json'))
luna = (ad > audit['ad_cut']).astype(int); conf = (1 - P.max(1) > sa['confidence_cut']).astype(int)
thr = {t['method']: t['thresholds'] for t in json.load(open(CO/'symmetric/thresholds_symmetric.json')) if t['seed'] == 20261011 and t['alpha'] == 0.1}
def cset(method, i):
    g = {'global': 0, 'luna': luna[i], 'confidence': conf[i], 'joint': 2*luna[i] + conf[i]}[method]
    q = thr[method][g]; q = np.inf if q is None else q
    return [NAMES[k] for k in range(7) if 1 - P[i, k] <= q]
# zero-exposure detections (fraction of the ten calibration splits), object level, nominal 1% FAR
det = {}
for r in csv.DictReader(open(STUDY/'zero_mean_objects.csv')):
    if r['method'] in ('object@0.01/embedding/LUNA16', 'object@0.01/single/msp'):
        det[(r['method'].split('/')[-1], r['obj_id'])] = float(r['detection_fraction_over_splits'])
rows = []
for i, o in enumerate(objs):
    k = P[i].argmax(); sd = Pm[i, :, k].std(ddof=1)
    rows.append(dict(obj=o, cls=cls[i], epochs=int(nep[i]), pred=NAMES[k], p=P[i, k], p_sd=sd, luna_rank=ad[i],
                     luna_flag=det[('LUNA16', o)], msp_flag=det[('msp', o)], luna_stratum=int(luna[i]), conf_stratum=int(conf[i]),
                     set_global='{' + ','.join(cset('global', i)) + '}', set_luna='{' + ','.join(cset('luna', i)) + '}',
                     set_conf='{' + ','.join(cset('confidence', i)) + '}', set_joint='{' + ','.join(cset('joint', i)) + '}'))
with open(ROOT/'rare_objects_conformal.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
sel = sorted([r for r in rows if r['luna_flag'] >= 0.9 and r['msp_flag'] == 0], key=lambda r: -r['p'])
print(len(rows), 'rare objects;', len(sel), 'flagged by LUNA (>=9/10 splits) and never by MSP')
for r in sel: print(f"{r['obj']:18} {r['cls']:16} n={r['epochs']:2} {r['pred']:4} {r['p']:.2f}±{r['p_sd']:.2f} LUNA {r['luna_flag']:.1f} rank {r['luna_rank']:.3f} strata L{r['luna_stratum']}C{r['conf_stratum']}  G{r['set_global']:12} L{r['set_luna']:12} C{r['set_conf']:12} J{r['set_joint']}")
print('\ncounts: LUNA-only', sum(r['luna_flag']>=.5 and r['msp_flag']<.5 for r in rows), ' MSP-only', sum(r['msp_flag']>=.5 and r['luna_flag']<.5 for r in rows), ' both', sum(r['msp_flag']>=.5 and r['luna_flag']>=.5 for r in rows))
