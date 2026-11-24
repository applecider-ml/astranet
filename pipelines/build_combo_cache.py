"""Combined backbone cache v7fixcombo12 = old TCN (v7fixmd6) + SpectraNet (v7fixsn6), 12 members.
emb / proj_fine = concatenation [old | sn] (384 / 256); fine/coarse logits = mean of the two 6-member mean logits
(= mean over all 12 member logits, the convention of both caches). Rows aligned by (obj_id, file_path)."""
import sys, pickle, shutil; from pathlib import Path; import numpy as np
REPO=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(REPO/'pipelines'),str(REPO)]
from _common import load_cache
A,B,C=REPO/'cache/v7fixmd6',REPO/'cache/v7fixsn6',REPO/'cache/v7fixcombo12'; C.mkdir(exist_ok=True)
key=lambda m:(str(m['obj_id']),str(m['file_path']))
for n in ['train','val','test','rare_unknown_combined']:
    a,b=load_cache(A,n),load_cache(B,n); pos={key(m):i for i,m in enumerate(b['meta'])}; o=np.array([pos[key(m)] for m in a['meta']])
    assert len(set(o))==len(o)==len(b['meta'])
    out={'emb':np.hstack([a['emb'],b['emb'][o]]),'proj_fine':np.hstack([a['proj_fine'],b['proj_fine'][o]]),
         'fine_logits':(a['fine_logits']+b['fine_logits'][o])/2,'coarse_logits':(a['coarse_logits']+b['coarse_logits'][o])/2}
    if 'labels' in a: assert np.array_equal(a['labels'],b['labels'][o]); out['labels']=a['labels']
    np.savez(C/f'{n}.npz',**{k:v.astype(a[k].dtype if k in a else v.dtype) for k,v in out.items()}); shutil.copy(A/f'{n}_meta.pkl',C/f'{n}_meta.pkl')
    print(n,{k:v.shape for k,v in out.items()})
pa,pb=np.load(A/'per_model_softmax.npz'),np.load(B/'per_model_softmax.npz')
np.savez(C/'per_model_softmax.npz',**{k:np.concatenate([pa[k],pb[k]]) for k in pa.files})
for s in ['val','test','rare']:
    ma,mb=np.load(A/f'members_{s}.npz'),np.load(B/f'members_{s}.npz'); assert np.array_equal(ma['obj_id'],mb['obj_id'])
    np.savez(C/f'members_{s}.npz',emb=np.concatenate([ma['emb'],mb['emb']]),fine_logits=np.concatenate([ma['fine_logits'],mb['fine_logits']]),
             obj_id=ma['obj_id'],labels=np.r_[['old/'+x for x in ma['labels']],['sn/'+x for x in mb['labels']]])
CH=REPO/'cache/challenging'; ca,cb=np.load(CH/'v7fixmd6_recheck.npz'),np.load(CH/'v7fixcombo12.npz')
np.savez(CH/'v7fixcombo12.npz',emb=np.hstack([ca['emb'],cb['emb']]),proj_fine=np.hstack([ca['proj_fine'],cb['proj_fine']]),
         fine_logits=(ca['fine_logits']+cb['fine_logits'])/2,coarse_logits=(ca['coarse_logits']+cb['coarse_logits'])/2)
sa,sb=np.load(CH/'v7fixmd6_recheck_per_model_softmax.npz'),np.load(CH/'v7fixcombo12_per_model_softmax.npz')
np.savez(CH/'v7fixcombo12_per_model_softmax.npz',softmax=np.concatenate([sa['softmax'],sb['softmax']]))
# sanity: classification on test
t=np.load(C/'test.npz'); print('test spectrum acc',float((t['fine_logits'].argmax(1)==t['labels']).mean()))
