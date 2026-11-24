import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
from pathlib import Path
import numpy as np
import pickle,json,csv,hashlib,math
from scipy.special import softmax
ROOT=Path(__file__).resolve().parent
REPO=REPO
CACHE=REPO/'cache/astranet12'
NAMES=['Ia','Ibc','SLSN','II','AGN','CV','TDE']
METHODS=['global','class','ad','class_ad']
D=np.load(ROOT.parent/'features.npz',allow_pickle=False)
def agg(x,inv,n):
 a=np.zeros((len(n),x.shape[1]),dtype=float);np.add.at(a,inv,x);return a/n[:,None]
def load(split):
 d=np.load(CACHE/f'{split}.npz');m=pickle.load(open(CACHE/f'{split}_meta.pkl','rb'))
 ids=np.array([str(v['obj_id']) for v in m]);obj,inv,n=np.unique(ids,return_inverse=True,return_counts=True)
 y=np.array([d['labels'][inv==i][0] for i in range(len(obj))],dtype=int)
 assert all(np.all(d['labels'][inv==i]==y[i]) for i in range(len(obj)))
 out={'ids':obj,'y':y,'p':agg(softmax(d['fine_logits'].astype(float),axis=1),inv,n),'epochs':n}
 if split!='train':
  assert np.array_equal(ids,D[f'{split}_objects'])
  assert np.array_equal(d['emb'],D[f'{split}_emb'])
  out['scores']=agg(D[f'{split}_emb_scores'],inv,n)
 assert np.allclose(out['p'].sum(1),1) and np.isfinite(out['p']).all()
 return out
train,val,test=[load(x) for x in ['train','val','test']]
for a,b in [(train,val),(train,test),(val,test)]:assert not set(a['ids'])&set(b['ids'])
def ad(x):return np.mean([np.searchsorted(np.sort(val['scores'][:,j]),x[:,j],side='left')/(len(val['ids'])+1) for j in range(16)],axis=0)
cut=float(np.quantile(ad(val['scores']),.8))
a=ad(test['scores']);g=(a>cut).astype(int)
np.savez_compressed(ROOT/'objects.npz',**test,ad=a,stratum=g,class_names=np.array(NAMES))
def q(s,alpha):
 k=math.ceil((len(s)+1)*(1-alpha));return float(np.sort(s)[k-1]) if k<=len(s) else float('inf')
assert q([], .1)==float('inf') and q(np.arange(9),.1)==8 and q(np.arange(8),.1)==float('inf')
def wilson(hits,n):
 if not n:return [None,None]
 z=1.95996398454;p=hits/n;den=1+z*z/n
 mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [float(mid-half),float(mid+half)]
rows=[];thresholds=[];assignments=[];saved={};paired=[]
def measure(seed,alpha,method,subset,mask,cover,size):
 n=int(mask.sum());c=cover[mask];s=size[mask]
 lo,hi=wilson(int(c.sum()),n)
 rows.append(dict(seed=seed,alpha=alpha,method=method,subset=subset,n=n,coverage=float(c.mean()) if n else None,coverage_lo=lo,coverage_hi=hi,mean_size=float(s.mean()) if n else None,singleton=float(np.mean(s==1)) if n else None,empty=float(np.mean(s==0)) if n else None,full=float(np.mean(s==7)) if n else None))
for seed in [20261011]+list(range(10)):
 ix=np.random.default_rng(seed).permutation(len(test['ids']));cal=ix[:len(ix)//2];ev=ix[len(ix)//2:]
 assignments.extend({'seed':seed,'object':str(test['ids'][i]),'role':'calibration' if i in set(cal) else 'evaluation'} for i in range(len(ix)))
 truth=test['y'];score=1-test['p'][cal,truth[cal]]
 for alpha in [.1,.05]:
  arm={}
  for method in METHODS:
   t=np.zeros((2,7));support=np.zeros((2,7),int)
   for gg in range(2):
    for yy in range(7):
     mask=np.ones(len(cal),bool)
     if 'class' in method:mask &= truth[cal]==yy
     if 'ad' in method:mask &= g[cal]==gg
     t[gg,yy]=q(score[mask],alpha);support[gg,yy]=mask.sum()
   sets=(1-test['p'][ev])<=t[g[ev]]
   cover=sets[np.arange(len(ev)),truth[ev]];size=sets.sum(1);arm[method]=(cover,size)
   thresholds.append(dict(seed=seed,alpha=alpha,method=method,thresholds=[[None if np.isinf(v) else float(v) for v in row] for row in t],cell_support=support.tolist(),null_means='positive infinity'))
   for name,mask in [('all',np.ones(len(ev),bool)),('low_ad',g[ev]==0),('high_ad',g[ev]==1)]+[('class_'+NAMES[y],truth[ev]==y) for y in range(7)]:measure(seed,alpha,method,name,mask,cover,size)
   if seed==20261011:saved[f'{alpha}_{method}']=sets
  if seed==20261011:
   saved['evaluation_index']=ev;saved['calibration_index']=cal
   for left,right in [('ad','global'),('class_ad','class')]:
    for label,mask in [('all',np.ones(len(ev),bool)),('high_ad',g[ev]==1)]:
     dc=arm[left][0][mask].astype(float)-arm[right][0][mask].astype(float);ds=arm[left][1][mask]-arm[right][1][mask]
     rng=np.random.default_rng(92831);boot=rng.integers(0,len(dc),size=(2000,len(dc)))
     paired.append(dict(alpha=alpha,comparison=left+' - '+right,subset=label,n=len(dc),coverage_gain_pp=float(dc.mean()*100),coverage_ci_pp=np.quantile(dc[boot].mean(1)*100,[.025,.975]).tolist(),size_change=float(ds.mean()),size_ci=np.quantile(ds[boot].mean(1),[.025,.975]).tolist()))
def csvwrite(name,data):
 with (ROOT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
csvwrite('metrics.csv',rows);csvwrite('assignments.csv',assignments)
(ROOT/'thresholds.json').write_text(json.dumps(thresholds,indent=2));(ROOT/'paired.json').write_text(json.dumps(paired,indent=2));np.savez_compressed(ROOT/'primary_sets.npz',**saved)
meta={'counts':{s:len(d['ids']) for s,d in [('train',train),('validation_design',val),('test_partitioned',test)]},'ad_cut':cut,'ad_high_test_objects':int(g.sum()),'probability':'mean over epochs of softmax of ensemble-mean logits','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.txt').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT.parent/'features.npz',CACHE/'val.npz',CACHE/'test.npz',CACHE/'val_meta.pkl',CACHE/'test_meta.pkl']}}
(ROOT/'audit.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
for r in rows:
 if r['seed']==20261011 and r['subset'] in ['all','high_ad']:print(r)
print('PAIRED',json.dumps(paired,indent=2))
