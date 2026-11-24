"""Frozen-head fusion, epoch dispersion and strictly nested class transfer."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
os.environ['MPLCONFIGDIR']='/tmp/astranet-ad-mpl'
os.environ['XDG_CACHE_HOME']='/tmp/astranet-ad-cache'
import sys, json, argparse, importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
OBJECTS=ROOT.parent/'objects'
spec=importlib.util.spec_from_file_location('objects_experiment',OBJECTS/'experiment.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from evaluate import split_objects, threshold, metric, rankmean, TIER1

HEADS=[('et','embphys','et'),('svm','embphys','svm10'),('luna_head','luna','et'),
       ('et_dispersion','dispersion','et'),('et_full_summary','summary','et'),
       ('et_plus_luna','embphys_luna','et')]
NAMES=[v[0] for v in HEADS]+['luna_rank', 'et25_svm75','et50_svm50','et75_svm25',
       'et50_luna50','et75_luna25','mean_three','max_et_svm','max_three']
ALL=NAMES+['nested_selected']

def load_data():
    d=dict(np.load(OBJECTS/'object_features.npz',allow_pickle=False))
    for split in ('val','test','rare'):
        width=d[split+'_embphys'].shape[1]
        d[split+'_luna']=d[split+'_summary'][:,2*width:]
        d[split+'_dispersion']=d[split+'_summary'][:,:2*width]
        d[split+'_embphys_luna']=np.hstack([d[split+'_embphys'],d[split+'_luna']])
        assert d[split+'_luna'].shape[1]==32
        assert np.isfinite(d[split+'_summary']).all()
    return d

def candidates(d,fit,expose,requests):
    """Return one score matrix per request; scale using fitting ID only."""
    output=[{} for _ in requests];scaled=[{} for _ in requests]
    for name,rep,kind in HEADS:
        values=base.fit_score(d,rep,kind,fit,expose,[('val',fit)]+requests)
        med=np.median(values[0]);spread=max(np.quantile(values[0],.9)-med,1e-6)
        for dst,z,raw in zip(output,scaled,values[1:]):
            dst[name]=raw;z[name]=(raw-med)/spread
    for dst,z,(split,ix) in zip(output,scaled,requests):
        dst['luna_rank']=rankmean(d['val_luna'][fit],d[split+'_luna'][ix])
        e,s,l=z['et'],z['svm'],z['luna_head']
        dst.update(et25_svm75=.25*e+.75*s,et50_svm50=.5*e+.5*s,
                   et75_svm25=.75*e+.25*s,et50_luna50=.5*e+.5*l,
                   et75_luna25=.75*e+.25*l,mean_three=(e+s+l)/3,
                   max_et_svm=np.maximum(e,s),max_three=np.maximum.reduce([e,s,l]))
    return [np.column_stack([dst[n] for n in NAMES]) for dst in output]

def run_outer(d,seed,heldclasses,key):
    (ROOT/'cache').mkdir(exist_ok=True); target=ROOT/'cache'/f'{seed:02d}_{key}.npz'
    note=target.with_suffix('.json')
    if target.exists() and note.exists():
        print('cached',seed,key,flush=True);return
    labels=d['rare_classes'];classes=np.unique(labels)
    fit,cal=split_objects(d['val_ids'],seed)
    held=np.flatnonzero(np.isin(labels,heldclasses));expose=np.flatnonzero(~np.isin(labels,heldclasses))
    other=np.random.default_rng(71).permutation(classes[~np.isin(classes,heldclasses)])
    classfolds=np.array_split(other,3)
    idfolds=np.array_split(np.random.default_rng(seed+101).permutation(fit),3)
    performance=[]
    for heldinner,idvalid in zip(classfolds,idfolds):
        rv=expose[np.isin(labels[expose],heldinner)];rt=expose[~np.isin(labels[expose],heldinner)]
        idtrain=np.setdiff1d(fit,idvalid)
        assert set(labels[rt]).isdisjoint(labels[rv])
        assert set(labels[np.r_[rt,rv]]).isdisjoint(heldclasses)
        assert set(np.r_[idtrain,idvalid]).isdisjoint(cal)
        iv,rs=candidates(d,idtrain,rt,[('val',idvalid),('rare',rv)])
        for c in heldinner:
            mask=labels[rv]==c
            y=np.r_[np.zeros(len(iv)),np.ones(mask.sum())]
            performance.append([roc_auc_score(y,np.r_[iv[:,j],rs[mask,j]],max_fpr=.02)
                                for j in range(len(NAMES))])
    objective=np.mean(performance,axis=0);chosen=int(np.argmax(objective))
    vc,ts,rs=candidates(d,fit,expose,[('val',cal),('test',np.arange(len(d['test_ids']))),('rare',held)])
    thresholds=np.array([threshold(vc[:,j]) for j in range(len(NAMES))])
    # Temp files are renamed only when complete. A partial cache is recomputed.
    tmp=target.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp,cal=vc,test=ts,rare=rs,thresholds=thresholds,
                        chosen=chosen,held=held,names=np.array(NAMES),
                        id_fit=fit,id_cal=cal,inner_objective=objective)
    tmp.replace(target)
    notetmp=note.with_suffix(f'.{os.getpid()}.tmp.json')
    notetmp.write_text(json.dumps({'seed':seed,'key':key,'target_classes':list(map(str,heldclasses)),
        'exposure_classes':other.tolist(),'chosen':NAMES[chosen],
        'inner_macro_pauc02':dict(zip(NAMES,map(float,objective)))},indent=2))
    notetmp.replace(note)
    print('completed',seed,key,'selected',NAMES[chosen],flush=True)

def collect(d,mode,seeds):
    labels=d['rare_classes'];classes=np.unique(labels)
    groups=[('class'+str(i),[c]) for i,c in enumerate(classes)] if mode=='lco' else [
        ('Tier1',[c for c in classes if c in TIER1]),('Tier2',[c for c in classes if c not in TIER1])]
    records=defaultdict(list);flags=defaultdict(list);idflags=defaultdict(list);choices=[]
    ref=np.load(OBJECTS/'nested_objects_flags.npz') if (OBJECTS/'nested_objects_flags.npz').exists() else None
    reftier=np.load(OBJECTS/'tier_transfer_flags.npz') if (OBJECTS/'tier_transfer_flags.npz').exists() else None
    for seed in range(seeds):
        if not all((ROOT/'cache'/f'{seed:02d}_{key}.json').exists() for key,_ in groups):break
        det={n:np.zeros(len(labels),bool) for n in ALL};tests={n:[] for n in ALL}
        for key,heldclasses in groups:
            with np.load(ROOT/'cache'/f'{seed:02d}_{key}.npz') as f:
                assert list(f['names'])==NAMES
                rf=f['rare']>f['thresholds'];tf=f['test']>f['thresholds'];held=f['held'];chosen=int(f['chosen'])
                for j,name in enumerate(ALL):
                    col=chosen if name=='nested_selected' else j
                    if mode=='lco':
                        det[name][held]=rf[:,col];tests[name].append(tf[:,col])
                    else:
                        full=np.full(len(labels),np.nan);full[held]=rf[:,col]
                        k=key+'/'+name
                        recall=float(rf[:,col].mean())
                        records[k].append({'seed':seed,'recall':recall,
                            'tier1':recall if key=='Tier1' else 0.,
                            'tier2':recall if key=='Tier2' else 0.,
                            'macro_class_recall':float(np.mean([rf[labels[held]==c,col].mean() for c in heldclasses])),
                            'test_object_far':float(tf[:,col].mean())})
                        flags[k].append(full);idflags[k].append(tf[:,col])
                        if reftier is not None and name in ('et','svm'):
                            refname='object_extra_trees' if name=='et' else 'object_rbf_C10'
                            np.testing.assert_array_equal(full,reftier['rare::'+key+'/'+refname][seed])
                            np.testing.assert_array_equal(tf[:,col],reftier['id::'+key+'/'+refname][seed])
                choices.append(json.loads((ROOT/'cache'/f'{seed:02d}_{key}.json').read_text()))
        if mode=='lco':
            for name in ALL:
                t=np.mean(tests[name],axis=0)
                records[name].append({'seed':seed,**metric(det[name],labels),'test_object_far':float(t.mean())})
                flags[name].append(det[name]);idflags[name].append(t)
                if ref is not None and name in ('et','svm'):
                    refname='object_extra_trees' if name=='et' else 'object_rbf_C10'
                    np.testing.assert_array_equal(det[name],ref['rare::'+refname][seed])
                    np.testing.assert_array_equal(t,ref['id::'+refname][seed])
    summary={name:{**{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='seed'},
                          'per_split':rows} for name,rows in records.items()}
    (ROOT/f'{mode}.json').write_text(json.dumps({'summary':summary,'choices':choices,
                                              'baselines_reproduced_exactly':True},indent=2))
    np.savez_compressed(ROOT/f'{mode}_flags.npz',**{'rare::'+k:np.asarray(v) for k,v in flags.items()},
                        **{'id::'+k:np.asarray(v) for k,v in idflags.items()})
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['lco','tier'],default='lco')
    ap.add_argument('--splits',type=int,default=10)
    ap.add_argument('--seed-start',type=int,default=0)
    ap.add_argument('--worker',action='store_true');args=ap.parse_args()
    d=load_data();classes=np.unique(d['rare_classes'])
    groups=[('class'+str(i),[c]) for i,c in enumerate(classes)] if args.mode=='lco' else [
        ('Tier1',[c for c in classes if c in TIER1]),('Tier2',[c for c in classes if c not in TIER1])]
    with threadpool_limits(limits=2):
        for seed in range(args.seed_start,args.splits):
            for key,heldclasses in groups:run_outer(d,seed,heldclasses,key)
            if args.worker:continue
            summary=collect(d,args.mode,seed+1)
            for n,r in summary.items():
                print(n,'recall',round(r['recall']*100,2),'FAR',round(r['test_object_far']*100,3),flush=True)
            print('COMPLETED SPLIT',args.mode,seed,flush=True)

if __name__=='__main__':main()
