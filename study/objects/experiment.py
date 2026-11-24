"""Object-level anomaly detectors, with nested leave-class-out model choice."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['LOKY_MAX_CPU_COUNT']='2'
import sys,json,argparse,warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors,LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import lightgbm as lgb

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from evaluate import threshold,ecdf,rankmean,metric,grouped_max,split_objects

# Named, fixed candidates. No candidate is added after its outer result is inspected.
CANDIDATES=[('object_emb_lgb','embedding','lgb'),('object_embphys_lgb','embphys','lgb'),
            ('object_summary_lgb','summary','lgb'),('object_logistic_C0.1','embphys','lr0.1'),
            ('object_logistic_C1','embphys','lr1'),('object_rbf_C1','embphys','svm1'),
            ('object_rbf_C10','embphys','svm10'),('object_extra_trees','embphys','et')]

def model(kind):
    if kind=='lgb':
        return lgb.LGBMClassifier(n_estimators=100,num_leaves=15,max_depth=4,learning_rate=.05,
               min_child_samples=20,subsample=.8,colsample_bytree=.8,class_weight='balanced',
               random_state=42,verbosity=-1,n_jobs=2)
    if kind.startswith('lr'):
        return make_pipeline(StandardScaler(),LogisticRegression(C=float(kind[2:]),solver='liblinear',
                             max_iter=1000,class_weight='balanced',random_state=42))
    if kind.startswith('svm'):
        return make_pipeline(StandardScaler(),SVC(C=float(kind[3:]),gamma='scale',class_weight='balanced'))
    return ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features='sqrt',
                                class_weight='balanced',random_state=42,n_jobs=1)

def score(clf,X):
    return clf.decision_function(X) if hasattr(clf,'decision_function') else clf.predict_proba(X)[:,1]

def fit_score(d,representation,kind,idfit,rare_fit,requests):
    clf=model(kind)
    X=np.vstack([d['val_'+representation][idfit],d['rare_'+representation][rare_fit]])
    y=np.r_[np.zeros(len(idfit)),np.ones(len(rare_fit))]
    clf.fit(X,y)
    return [score(clf,d[split+'_'+representation][ix]) for split,ix in requests]

def dump(name,records,flags,idflags,d,extra):
    summary={}
    for key,rows in records.items():
        summary[key]={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='seed'}
        summary[key]['recall_sd']=float(np.std([r['recall'] for r in rows]))
        summary[key]['per_split']=rows
    out={'summary':summary,'extra':extra}
    (ROOT/f'{name}.json').write_text(json.dumps(out,indent=2))
    np.savez_compressed(ROOT/f'{name}_flags.npz',**{'rare::'+k:np.asarray(v) for k,v in flags.items()},
                        **{'id::'+k:np.asarray(v) for k,v in idflags.items()})
    for key,r in sorted(summary.items(),key=lambda kv:-kv[1]['recall']):
        print(f'{key:28} recall {r["recall"]*100:5.1f} Tier1 {r["tier1"]*100:5.1f} test FAR {r["test_object_far"]*100:4.2f}',flush=True)

def unsupervised(d,seeds):
    X=d['train_embedding'].astype(float);y=d['train_labels'];classes=np.unique(y)
    means=np.array([X[y==k].mean(0) for k in classes])
    classindex={c:i for i,c in enumerate(classes)}
    residual=X-np.array([means[classindex[c]] for c in y])
    pooled=LedoitWolf().fit(residual).precision_
    background=LedoitWolf().fit(X)
    radii=np.array([np.quantile(np.linalg.norm(X[y==k]-means[i],axis=1),.95) for i,k in enumerate(classes)])
    nn=NearestNeighbors(n_neighbors=10,n_jobs=2,algorithm='brute').fit(X)
    unit=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    cnn=NearestNeighbors(n_neighbors=10,n_jobs=2,algorithm='brute').fit(unit)
    lof=LocalOutlierFactor(n_neighbors=20,novelty=True,n_jobs=2,algorithm='brute').fit(X)
    names=['knn','cosine_knn','lof','shared_class_mahal','global_mahal','relative_mahal','nearest_class_radius']
    scores={}
    for n in ['val','test','rare']:
        E=d[n+'_embedding'].astype(float)
        MD=np.array([np.einsum('ij,jk,ik->i',E-m,pooled,E-m) for m in means]).min(0)
        globalMD=background.mahalanobis(E)
        radius=np.array([np.linalg.norm(E-m,axis=1)/max(r,1e-12) for m,r in zip(means,radii)]).min(0)
        scores[n]=np.column_stack([nn.kneighbors(E)[0].mean(1),
             cnn.kneighbors(E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-12))[0].mean(1),
             -lof.score_samples(E),MD,globalMD,MD-globalMD,radius])
    np.savez_compressed(ROOT/'object_density_scores.npz',methods=np.array(names),**scores)
    records,flags,idflags=defaultdict(list),defaultdict(list),defaultdict(list)
    for seed in range(seeds):
        ref,cal=split_objects(d['val_ids'],seed)
        for name in names+['object_density_mean_rank']:
            s={n:rankmean(scores['val'][ref],scores[n]) if name=='object_density_mean_rank' else scores[n][:,names.index(name)] for n in scores}
            th=threshold(s['val'][cal])
            rf=s['rare']>th;tf=s['test']>th
            records[name].append({'seed':seed,**metric(rf,d['rare_classes']),'test_object_far':float(tf.mean())})
            flags[name].append(rf);idflags[name].append(tf)
    dump('unsupervised_objects',records,flags,idflags,d,{'fit':'ID training object means only'})

def nested(d,seeds,resume=False):
    labels=d['rare_classes'];classes=np.unique(labels)
    nr=len(labels);nt=len(d['test_ids']);names=[c[0] for c in CANDIDATES]+['nested_selected']
    records,flags,idflags=defaultdict(list),defaultdict(list),defaultdict(list)
    choices=[]
    start=0
    if resume:
        saved=json.loads((ROOT/'nested_objects.json').read_text())
        savedflags=np.load(ROOT/'nested_objects_flags.npz')
        assert set(saved['summary'])==set(names)
        for name in names:
            records[name]=saved['summary'][name]['per_split']
            flags[name]=list(savedflags['rare::'+name]);idflags[name]=list(savedflags['id::'+name])
        start=len(records['nested_selected'])
        assert all([r['seed'] for r in records[n]]==list(range(start)) for n in names)
        choices=saved['extra']['choices']
        assert len(choices)==start*len(classes)
        print('Resuming after',start,'completed splits; serial ExtraTrees to avoid cluster thread limit',flush=True)
    for seed in range(start,seeds):
        fit,cal=split_objects(d['val_ids'],seed)
        det={n:np.zeros(nr,bool) for n in names};tf={n:[] for n in names}
        for outer_i,C in enumerate(classes):
            held=np.flatnonzero(labels==C);expose=np.flatnonzero(labels!=C)
            assert set(d['rare_ids'][held]).isdisjoint(d['rare_ids'][expose])
            other=np.random.default_rng(71).permutation(classes[classes!=C])
            # Balance class counts among inner folds; never split a rare class across them.
            class_folds=np.array_split(other,3)
            id_folds=np.array_split(np.random.default_rng(seed+101).permutation(fit),3)
            performance={n:[] for n,_,_ in CANDIDATES}
            for heldclasses,id_valid in zip(class_folds,id_folds):
                rare_valid=expose[np.isin(labels[expose],heldclasses)]
                rare_train=expose[~np.isin(labels[expose],heldclasses)]
                id_train=np.setdiff1d(fit,id_valid)
                assert not(set(labels[rare_train])&set(labels[rare_valid]))
                assert C not in set(labels[np.r_[rare_train,rare_valid]])
                assert not(set(id_valid)&set(cal))
                for name,rep,kind in CANDIDATES:
                    iv,rv=fit_score(d,rep,kind,id_train,rare_train,[('val',id_valid),('rare',rare_valid)])
                    for c in heldclasses:
                        v=rv[labels[rare_valid]==c]
                        value=roc_auc_score(np.r_[np.zeros(len(iv)),np.ones(len(v))],np.r_[iv,v],max_fpr=.05)
                        performance[name].append(float(value))
            selection={n:float(np.mean(v)) for n,v in performance.items()}
            chosen=max(CANDIDATES,key=lambda c:selection[c[0]])[0]
            choices.append({'seed':seed,'held_class':str(C),'chosen':chosen,'inner_partial_auc':selection,
                            'selection_classes':other.tolist(), 'calibration_objects':d['val_ids'][cal].tolist()})
            for name,rep,kind in CANDIDATES:
                vc,ts,rs=fit_score(d,rep,kind,fit,expose,[('val',cal),('test',np.arange(nt)),('rare',held)])
                th=threshold(vc);rf=rs>th;tflag=ts>th
                det[name][held]=rf;tf[name].append(tflag)
                if name==chosen:
                    det['nested_selected'][held]=rf;tf['nested_selected'].append(tflag)
            print('nested',seed,str(C),'selected',chosen,flush=True)
        for name in names:
            mean_test=np.mean(tf[name],axis=0)
            records[name].append({'seed':seed,**metric(det[name],labels),'test_object_far':float(mean_test.mean())})
            flags[name].append(det[name]);idflags[name].append(mean_test)
        # Checkpoints preserve every completed split if interrupted.
        dump('nested_objects',records,flags,idflags,d,{'choices':choices,'ID_FAR':'mean across outer-class models; not a union'})
        print('completed nested split',seed,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['unsupervised','nested'],required=True);ap.add_argument('--splits',type=int,default=10);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names')
    with threadpool_limits(limits=2):
        d=dict(np.load(ROOT/'object_features.npz',allow_pickle=False))
        if args.mode=='unsupervised':unsupervised(d,args.splits)
        else:nested(d,args.splits,resume=args.resume)
