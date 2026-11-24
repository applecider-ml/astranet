import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
from pathlib import Path
import pickle,json,csv,math
import numpy as np
from scipy.special import softmax
R=Path(__file__).resolve().parent;C=(REPO/'cache/astranet12')
d=np.load(R/'objects.npz');v=np.load(C/'val.npz');m=pickle.load(open(C/'val_meta.pkl','rb'))
ids,inv,n=np.unique([str(x['obj_id']) for x in m],return_inverse=True,return_counts=True)
p=np.zeros((len(n),7));np.add.at(p,inv,softmax(v['fine_logits'].astype(float),axis=1));p/=n[:,None]
cut=float(np.quantile(1-p.max(1),.8));g=(1-d['p'].max(1)>cut).astype(int)
rows=[];save={};out=[]
for seed in [20261011]+list(range(10)):
 ix=np.random.default_rng(seed).permutation(len(g));cal=ix[:len(ix)//2];ev=ix[len(ix)//2:];score=1-d['p'][cal,d['y'][cal]]
 for alpha in [.1,.05]:
  q=[]
  for gg in [0,1]:
   a=score[g[cal]==gg];k=math.ceil((len(a)+1)*(1-alpha));q.append(float(np.sort(a)[k-1]) if k<=len(a) else float('inf'))
  sets=1-d['p'][ev]<=np.array(q)[g[ev],None];hit=sets[np.arange(len(ev)),d['y'][ev]];size=sets.sum(1)
  for name,mask in [('all',np.ones(len(ev),bool)),('high_ad',d['stratum'][ev]==1)]:rows.append(dict(seed=seed,alpha=alpha,method='confidence',subset=name,n=int(mask.sum()),coverage=float(hit[mask].mean()),mean_size=float(size[mask].mean())))
  if seed==20261011:
   save[str(alpha)]=sets
   orig=np.load(R/'primary_sets.npz')[f'{alpha}_ad'];oh=orig[np.arange(len(ev)),d['y'][ev]];os=orig.sum(1)
   for name,mask in [('all',np.ones(len(ev),bool)),('high_ad',d['stratum'][ev]==1)]:
    dc=oh[mask].astype(float)-hit[mask];ds=os[mask]-size[mask];boot=np.random.default_rng(119).integers(0,len(dc),(2000,len(dc)))
    out.append(dict(alpha=alpha,subset=name,comparison='AD minus confidence',coverage_gain_pp=float(dc.mean()*100),coverage_ci_pp=np.quantile(dc[boot].mean(1)*100,[.025,.975]).tolist(),size_change=float(ds.mean())))
with (R/'confidence_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(R/'confidence_sets.npz',**save)
(R/'confidence_audit.json').write_text(json.dumps({'validation_cut':cut,'paired':out},indent=2))
for x in rows:
 if x['seed']==20261011:print(x)
print(json.dumps(out,indent=2))
