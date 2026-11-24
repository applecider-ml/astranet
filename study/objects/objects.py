"""Construct object representations from identity-aligned, fixed cached spectra."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def aggregate(X, ids):
    obj,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    sums=np.zeros((len(obj),X.shape[1]),dtype=np.float64)
    sums2=np.zeros_like(sums)
    maxima=np.full_like(sums,-np.inf)
    np.add.at(sums,inv,X);np.add.at(sums2,inv,np.asarray(X,dtype=np.float64)**2)
    np.maximum.at(maxima,inv,X)
    mean=sums/count[:,None]
    std=np.sqrt(np.maximum(sums2/count[:,None]-mean**2,0))
    # A single observation has exactly zero dispersion, including floating roundoff.
    std[count==1]=0
    return obj,mean.astype('float32'),std.astype('float32'),maxima.astype('float32'),count

def build():
    d=dict(np.load(ROOT.parent/'features.npz',allow_pickle=False));out={}
    labels=np.load(REPO/'cache/astranet12/train.npz')['labels']
    for n in ['train','val','test','rare']:
        ids=d[f'{n}_objects'];obj,E,Es,_,count=aggregate(d[f'{n}_emb'],ids)
        _,P,Ps,_,_=aggregate(d[f'{n}_phys'],ids)
        out[f'{n}_ids']=obj;out[f'{n}_counts']=count
        out[f'{n}_embedding']=E
        out[f'{n}_embphys']=np.hstack([E,P])
        if n!='train':
            _,S,_,Sm,_=aggregate(d[f'{n}_emb_scores'],ids)
            out[f'{n}_summary']=np.hstack([E,Es,P,Ps,S,Sm])
        if n=='train':
            out['train_labels']=np.array([np.bincount(labels[ids==o].astype(int)).argmax() for o in obj])
            out['train_label_conflicts']=np.array(sum(len(np.unique(labels[ids==o]))>1 for o in obj))
        if n=='rare':
            assert all(len(set(d['rare_classes'][ids==o]))==1 for o in obj)
            out['rare_classes']=np.array([d['rare_classes'][np.flatnonzero(ids==o)[0]] for o in obj])
    for n in ['val','test','rare']:
        assert set(out['train_ids']).isdisjoint(out[f'{n}_ids'])
    np.savez_compressed(ROOT/'object_features.npz',**out)
    print({n:len(out[f'{n}_ids']) for n in ['train','val','test','rare']},flush=True)
    print('Training objects with mixed fine labels:',out['train_label_conflicts'],flush=True)

if __name__=='__main__':build()
