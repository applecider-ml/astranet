"""Grouped, independent-FAR comparison. All output paths are study-local."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
from pathlib import Path
import argparse, json, csv
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
TIER1 = {'FBOT', 'ILRT', 'Luminous Red Nova', 'Ca-rich', 'afterglow', 'long GRB'}

def grouped_max(scores, ids):
    objects, inverse = np.unique(ids, return_inverse=True)
    out = np.full(len(objects), -np.inf)
    np.maximum.at(out, inverse, scores)
    return objects, out

def split_objects(ids, seed):
    objs = np.random.default_rng(seed).permutation(np.unique(ids))
    ref = np.isin(ids, objs[:len(objs)//2])
    assert not (set(ids[ref]) & set(ids[~ref]))
    return np.flatnonzero(ref), np.flatnonzero(~ref)

def threshold(scores, alpha=0.01):
    """Finite-sample upper quantile; use score > threshold (conservative ties)."""
    scores = np.sort(np.asarray(scores))
    k = int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(scores[k-1]) if k <= len(scores) else float('inf')

def ecdf(ref, x):
    # Strict-less count: larger values correspond to more anomalous scores.
    return np.searchsorted(np.sort(ref), x, side='left')/(len(ref)+1)

def rankmean(ref, X):
    return np.mean([ecdf(ref[:,j], X[:,j]) for j in range(X.shape[1])], axis=0)

def object_classes(d):
    ids = d['rare_objects']; classes = d['rare_classes']
    objs = np.unique(ids)
    assert all(len(set(classes[ids == o])) == 1 for o in objs)
    return objs, np.array([classes[np.flatnonzero(ids == o)[0]] for o in objs])

def make_candidates(d, ref, followup=False):
    cuts = [len(d['val_objects']), len(d['test_objects'])]
    nval = cuts[0]
    M = list(d['emb_methods'])
    matrices = {view:np.vstack([d[f'{n}_{view}_scores'] for n in ['val','test','rare']]) for view in ['emb','phys','shape']}
    emb = matrices['emb']; result = {f'single/{m}':emb[:,j] for j,m in enumerate(M)}
    result['embedding/LUNA16'] = rankmean(emb[ref], emb)
    trio = [M.index(m) for m in ['typicality','knn','mahal_class']]
    result['embedding/density_trio'] = rankmean(emb[ref][:,trio], emb[:,trio])
    result['physics/LUNA'] = rankmean(matrices['phys'][ref], matrices['phys'])
    result['shape/knn+residual'] = rankmean(matrices['shape'][ref], matrices['shape'])
    result['shape/knn'] = matrices['shape'][:,0]
    result['shape/residual'] = matrices['shape'][:,1]
    result['embedding/unit_knn'] = np.concatenate([d[f'{n}_unit_knn'] for n in ['val','test','rare']])
    pairs = {
        'embedding+physics':['embedding/LUNA16','physics/LUNA'],
        'density_trio+physics':['embedding/density_trio','physics/LUNA'],
        'embedding+shape':['embedding/LUNA16','shape/knn+residual'],
        'embedding+physics+shape':['embedding/LUNA16','physics/LUNA','shape/knn+residual'],
    }
    for label, keys in pairs.items():
        # p-values are only a transformation. The final combined statistic gets
        # its own independent calibration; no independence assumption between views.
        P = np.column_stack([1-ecdf(result[k][ref], result[k]) for k in keys])
        result[f'union/{label}'] = -np.log(P.min(axis=1))
        result[f'fisher/{label}'] = -2*np.log(P).sum(axis=1)
        if followup:
            # Follow-up: preserve score differences beyond the largest reference
            # sample, instead of collapsing them all to one minimum empirical p.
            # Location/scale are ID-reference-only; final FAR is calibrated again.
            tail=[]
            for k in keys:
                q95,q99=np.quantile(result[k][ref],[.95,.99])
                tail.append((result[k]-q95)/max(q99-q95,1e-12))
            result[f'tail_union/{label}']=np.max(tail,axis=0)
    return {k:np.split(v, [nval, sum(cuts)]) for k,v in result.items()}

def summarize(records, flags, idflags, d, filename, extra=None):
    objs, labels = object_classes(d)
    tier = np.isin(labels, list(TIER1))
    summary = {}
    for key, rows in records.items():
        summary[key] = {field:float(np.mean([r[field] for r in rows])) for field in rows[0] if field != 'seed'}
        summary[key]['recall_sd'] = float(np.std([r['recall'] for r in rows]))
        summary[key]['test_object_far_sd'] = float(np.std([r['test_object_far'] for r in rows]))
        summary[key]['per_split'] = rows
    # Paired object bootstrap of mean detection over calibration repetitions.
    # Conditional on the fitted encoder / scorers; not a fresh-dataset guarantee.
    rng = np.random.default_rng(314159)
    boot = rng.integers(0, len(objs), size=(4000,len(objs)))
    baseline = 'object@0.01/embedding/LUNA16'
    paired = {}
    if baseline in flags:
        base = np.mean(flags[baseline], axis=0)
        for key, values in flags.items():
            if not key.startswith('object@0.01/'): continue
            delta = np.mean(values, axis=0) - base
            paired[key] = {'gain_pp':float(delta.mean()*100), 'ci95_pp':(np.quantile(delta[boot].mean(1),[.025,.975])*100).tolist()}
    payload = {'summary':summary, 'paired_vs_LUNA16':paired, 'extra':extra or {}}
    (ROOT/f'{filename}.json').write_text(json.dumps(payload, indent=2))
    with open(ROOT/f'{filename}_objects.csv','w') as f:
        w=csv.writer(f); w.writerow(['method','obj_id','class','tier','detection_fraction_over_splits'])
        for key,v in flags.items():
            for obj,cls,t,p in zip(objs,labels,tier,np.mean(v,axis=0)):
                w.writerow([key,obj,cls,1 if t else 2,p])
    np.savez_compressed(ROOT/f'{filename}_flags.npz', **{f'rare::{k}':np.asarray(v) for k,v in flags.items()}, **{f'id::{k}':np.asarray(v) for k,v in idflags.items()})
    # Compact stdout, same nominal object FAR for every candidate.
    for key,r in sorted(summary.items(), key=lambda x:-x[1]['recall']):
        if key.startswith('object@0.01/'):
            print(f'{key:62} recall {100*r["recall"]:5.1f} T1 {100*r["tier1"]:5.1f} test FAR {100*r["test_object_far"]:4.2f}',flush=True)
    return payload

def metric(det, labels):
    tier=np.isin(labels,list(TIER1))
    return {'recall':float(det.mean()),'tier1':float(det[tier].mean()),'tier2':float(det[~tier].mean()),
            'macro_class_recall':float(np.mean([det[labels==c].mean() for c in np.unique(labels)]))}

def zero_exposure(d, nseeds, followup=False):
    objs, labels = object_classes(d)
    records, flags, idflags = defaultdict(list), defaultdict(list), defaultdict(list)
    instruments = defaultdict(list)
    for seed in range(nseeds):
        ref,cal=split_objects(d['val_objects'],seed)
        candidates=make_candidates(d,ref,followup=followup)
        for method,(v,t,r) in candidates.items():
            _,co=grouped_max(v[cal],d['val_objects'][cal])
            _,to=grouped_max(t,d['test_objects'])
            _,ro=grouped_max(r,d['rare_objects'])
            for unit,alpha in [('object',.01),('spectrum',.01),('object',.05)]:
                th=threshold(co if unit=='object' else v[cal],alpha)
                key=f'{unit}@{alpha}/{method}'
                record={'seed':seed,**metric(ro>th,labels),'test_object_far':float((to>th).mean()),
                        'test_spectrum_far':float((t>th).mean()),'rare_spectrum_recall':float((r>th).mean()),
                        'calibration_far':float(((co if unit=='object' else v[cal])>th).mean()),
                        'rare_object_tie_fraction':float((ro==th).mean())}
                records[key].append(record);flags[key].append(ro>th);idflags[key].append(to>th)
                if unit=='object' and alpha==.01:
                    perinst={}
                    for ins in np.unique(d['test_instruments']):
                        mask=d['test_instruments']==ins
                        if mask.sum()<20:continue
                        _,it=grouped_max(t[mask],d['test_objects'][mask])
                        perinst[ins]={'spectra':int(mask.sum()),'objects':len(it),'object_far':float((it>th).mean())}
                    instruments[key].append(perinst)
        print('zero exposure split',seed,'done',flush=True)
    return summarize(records,flags,idflags,d,'zero_exposure_followup' if followup else 'zero_exposure',{'instrument_far':instruments})

def weights(ids,y):
    w=np.empty(len(y))
    for cls in [0,1]:
        sel=y==cls; _,inv,counts=np.unique(ids[sel],return_inverse=True,return_counts=True)
        w[sel]=1/counts[inv]/len(counts)
    return w/w.mean()

def supervised(d,nseeds,protocol):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold
    objs,labels=object_classes(d); cls=d['rare_classes']; ids=d['rare_objects']
    # Each additional feature set tests incremental value over the same head.
    feature_sets={'LUNA16':['emb_scores'],'embedding':['emb'],'embedding+physics':['emb','phys'],
                  'embedding+physics+shape':['emb','phys','shape'],
                  'embedding+LUNA16+physics':['emb','emb_scores','phys']}
    if protocol=='family_lco':
        families={'uncertainty':['msp','entropy','energy','odin'],
                  'distance':['mahal_global','mahal_class','mahal_within','knn','cosine','typicality'],
                  'density':['gmm','iforest','lof','pca_recon'], 'hybrid':['hec','mrs']}
        feature_sets={'LUNA16':['emb_scores']}
        for family,methods in families.items():
            feature_sets[family]=['family_'+family]
            idx=[list(d['emb_methods']).index(m) for m in methods]
            for n in ['val','test','rare']:d[n+'_family_'+family]=d[n+'_emb_scores'][:,idx]
    X={k:{n:np.hstack([d[f'{n}_{f}'] for f in fs]) for n in ['val','test','rare']} for k,fs in feature_sets.items()}
    records,flags,idflags=defaultdict(list),defaultdict(list),defaultdict(list)
    if protocol!='recovery':
        folds=[(np.flatnonzero(cls!=c),np.flatnonzero(cls==c)) for c in np.unique(cls)]
    else:
        folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(cls,cls,groups=ids))
    for seed in range(nseeds):
        fit,cal=split_objects(d['val_objects'],seed)
        for method,features in X.items():
            detections={(unit,a):np.zeros(len(ids),bool) for unit,a in [('object',.01),('spectrum',.01),('object',.05)]}
            tflags=defaultdict(list);specfar=defaultdict(list)
            for expose,held in folds:
                assert not(set(ids[expose])&set(ids[held]))
                if protocol!='recovery':assert not(set(cls[expose])&set(cls[held]))
                trainX=np.vstack([features['val'][fit],features['rare'][expose]])
                y=np.r_[np.zeros(len(fit)),np.ones(len(expose))]
                trainids=np.r_[d['val_objects'][fit],ids[expose]]
                model=lgb.LGBMClassifier(n_estimators=100,num_leaves=15,max_depth=4,learning_rate=.05,
                      min_child_samples=20,subsample=.8,colsample_bytree=.8,random_state=42,verbosity=-1,n_jobs=2)
                model.fit(trainX,y,sample_weight=weights(trainids,y))
                v=model.predict_proba(features['val'][cal])[:,1]
                t=model.predict_proba(features['test'])[:,1]
                r=model.predict_proba(features['rare'][held])[:,1]
                _,co=grouped_max(v,d['val_objects'][cal]);_,to=grouped_max(t,d['test_objects'])
                for unit,a in detections:
                    th=threshold(co if unit=='object' else v,a)
                    detections[(unit,a)][held]=r>th
                    tflags[(unit,a)].append(to>th);specfar[(unit,a)].append((t>th).mean())
            for (unit,a),det in detections.items():
                _,objdet=grouped_max(det.astype(float),ids)
                key=f'{unit}@{a}/{method}'
                # LCO FAR is the mean over distinct held-class detectors, NOT their union.
                tf=np.mean(tflags[(unit,a)],axis=0)
                records[key].append({'seed':seed,**metric(objdet>0,labels),
                         'test_object_far':float(tf.mean()),'test_spectrum_far':float(np.mean(specfar[(unit,a)])),
                         'rare_spectrum_recall':float(det.mean())})
                flags[key].append(objdet>0);idflags[key].append(tf)
        print(protocol,'split',seed,'done',flush=True)
    return summarize(records,flags,idflags,d,protocol,{'test_far_definition':'mean FAR across held-class / held-fold detectors, not a union'})

def self_check():
    # Correct object aggregation, disjoint splits, strict-tie threshold and
    # reference-only scoring (new test samples cannot change an existing score).
    ids=np.array(['a','a','b','c','c']); _,m=grouped_max(np.array([1,4,2,3,0]),ids)
    assert np.array_equal(m,[4,2,3])
    a,b=split_objects(ids,0);assert set(ids[a]).isdisjoint(ids[b])
    assert threshold(np.zeros(200))==0 and not(np.zeros(3)>threshold(np.zeros(200))).any()
    assert ecdf(np.arange(20),np.array([3]))[0]==ecdf(np.arange(20),np.array([3,999]))[0]
    # For 999 calibration objects alpha=.01 permits at most 9 strict exceedances.
    assert (np.arange(999)>threshold(np.arange(999))).sum()==9

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['zero','leave_class_out','recovery','family_lco'],default='zero');ap.add_argument('--splits',type=int,default=10);ap.add_argument('--followup',action='store_true')
    args=ap.parse_args();self_check()
    with threadpool_limits(limits=2):
        d=dict(np.load(ROOT/'features.npz',allow_pickle=False))
        if args.mode=='zero':zero_exposure(d,args.splits,followup=args.followup)
        else:supervised(d,args.splits,args.mode)
