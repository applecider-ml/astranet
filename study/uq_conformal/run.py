import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
from pathlib import Path
import numpy as np,json,csv,pickle,math,hashlib
from scipy.special import softmax
from scipy.optimize import minimize_scalar
R=Path(__file__).resolve().parent;C=(REPO/'cache/astranet12')
d=np.load(R.parent/'conformal_object/objects.npz');y=d['y'];ids=d['ids'];P=d['p'];luna=d['stratum'].astype(int)
v=np.load(C/'val.npz');m=pickle.load(open(C/'val_meta.pkl','rb'));vi,inv,n=np.unique([str(x['obj_id']) for x in m],return_inverse=True,return_counts=True)
vp=np.zeros((len(n),7));np.add.at(vp,inv,softmax(v['fine_logits'].astype(float),axis=1));vp/=n[:,None]
regimes={'full':(P,vp)}
if (R/'masked_probabilities.npz').exists():
 md=np.load(R/'masked_probabilities.npz');assert np.array_equal(md['test_ids'],ids) and np.array_equal(md['val_ids'],vi) and np.array_equal(md['test_y'],y)
 regimes.update({k:(md['test_'+k],md['val_'+k]) for k in ['no_z','spectrum_only']})
def temp(p,t):return softmax(np.log(p)/t,axis=1)
def q(s,a):
 k=math.ceil((len(s)+1)*(1-a));return np.sort(s)[k-1] if 0<k<=len(s) else np.inf
def probmetric(p,y):
 c=p.max(1);ok=p.argmax(1)==y;b=np.minimum((10*c).astype(int),9)
 return dict(accuracy=float(ok.mean()),nll=float(-np.log(p[np.arange(len(y)),y]).mean()),brier=float(((p-np.eye(7)[y])**2).sum(1).mean()),ece10=float(sum((b==i).mean()*abs(c[b==i].mean()-ok[b==i].mean()) for i in range(10) if (b==i).any())))
rows=[];pr=[];temps=[];thresholds=[];paired=[];assign=[];arrays={};cuts={}
for seed in [20261011]+list(range(10)):
 ix=np.random.default_rng(seed).permutation(len(ids));fit=ix[:417];cal=ix[417:834];ev=ix[834:];ref=ix[:834]
 assert not(set(fit)&set(cal) or set(fit)&set(ev) or set(cal)&set(ev))
 assign.extend(dict(seed=seed,object=ids[i],role=role) for role,ii in [('temperature',fit),('conformal',cal),('evaluation',ev)] for i in ii)
 if seed==20261011:arrays.update(ids=ids,y=y,temperature_index=fit,calibration_index=cal,evaluation_index=ev)
 for regime,(p,vp0) in regimes.items():
  cut=float(np.quantile(1-vp0.max(1),.8));cuts[regime]=cut;conf=(1-p.max(1)>cut).astype(int)
  strata={'global':np.zeros(len(ids),int),'confidence':conf}
  subsets={'all':np.ones(len(ids),bool),'low_confidence':conf==1,'high_confidence':conf==0,'misclassified':p.argmax(1)!=y}
  if regime=='full':
   strata.update(luna=luna,joint=2*luna+conf)
   subsets.update(high_LUNA=luna==1,low_LUNA=luna==0,high_LUNA_only=(luna==1)&(conf==0),low_confidence_only=(luna==0)&(conf==1),both=(luna==1)&(conf==1),neither=(luna==0)&(conf==0))
  fitres=minimize_scalar(lambda lt:-np.log(temp(p[fit],np.exp(lt))[np.arange(len(fit)),y[fit]]).mean(),bounds=(np.log(.05),np.log(20)),method='bounded');assert fitres.success
  T=float(np.exp(fitres.x));pt=temp(p,T);assert np.array_equal(p.argmax(1),pt.argmax(1))
  temps.append(dict(seed=seed,regime=regime,temperature=T))
  for arm,a in [('raw',p),('temperature',pt)]:pr.append(dict(seed=seed,regime=regime,arm=arm,**probmetric(a[ev],y[ev])))
  if seed==20261011:
   arrays[regime+'_raw']=p;arrays[regime+'_temperature']=pt
   for name,mask in subsets.items():arrays[regime+'_subset_'+name]=mask
  for alpha in [.1,.05]:
   for method,g in strata.items():
    outcomes={}
    for arm,a,cc in [('raw',p,cal),('temperature',pt,cal),('raw_reference834',p,ref)]:
     thresholds0=np.array([q(1-a[cc[g[cc]==k],y[cc[g[cc]==k]]],alpha) for k in range(int(g.max())+1)])
     support=[int((g[cc]==k).sum()) for k in range(len(thresholds0))]
     thresholds.append(dict(seed=seed,regime=regime,alpha=alpha,method=method,arm=arm,support=support,threshold=[float(z) if np.isfinite(z) else None for z in thresholds0]))
     sets=1-a[ev]<=thresholds0[g[ev],None];hit=sets[np.arange(len(ev)),y[ev]];size=sets.sum(1);outcomes[arm]=(hit,size)
     if seed==20261011:arrays[f'{regime}_{alpha}_{method}_{arm}_sets']=sets
     for subset,mask in subsets.items():
      mm=mask[ev];assert mm.any()
      rows.append(dict(seed=seed,regime=regime,alpha=alpha,method=method,arm=arm,subset=subset,n=int(mm.sum()),coverage=float(hit[mm].mean()),mean_size=float(size[mm].mean()),singleton=float((size[mm]==1).mean()),empty=float((size[mm]==0).mean()),full_set=float((size[mm]==7).mean())))
    if seed==20261011:
     for subset,mask in subsets.items():
      mm=mask[ev];rng=np.random.default_rng(31415);boot=rng.integers(0,mm.sum(),(2000,mm.sum()))
      for metric,j in [('coverage',0),('mean_size',1)]:
       diff=outcomes['temperature'][j][mm].astype(float)-outcomes['raw'][j][mm].astype(float)
       lo,hi=np.quantile(diff[boot].mean(1),[.025,.975]);paired.append(dict(regime=regime,alpha=alpha,method=method,subset=subset,metric=metric,difference=float(diff.mean()),lo=float(lo),hi=float(hi)))
def write(name,rr):
 with (R/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
for name,a in [('metrics.csv',rows),('probability_metrics.csv',pr),('temperatures.csv',temps),('assignments.csv',assign),('paired.csv',paired)]:write(name,a)
summary=[]
keys=['regime','alpha','method','arm','subset']
for key in dict.fromkeys(tuple(r[k] for k in keys) for r in rows):
 rr=[r for r in rows if tuple(r[k] for k in keys)==key and r['seed']!=20261011]
 summary.append(dict(zip(keys,key))|{f'{metric}_{stat}':float(fn([r[metric] for r in rr])) for metric in ['coverage','mean_size','singleton','empty','full_set'] for stat,fn in [('mean',np.mean),('sd',lambda x:np.std(x,ddof=1))]})
write('sensitivity.csv',summary);np.savez_compressed(R/'primary.npz',**arrays)
(R/'thresholds.json').write_text(json.dumps(thresholds,indent=1))
source=[Path(__file__),R.parent/'conformal_object/objects.npz',C/'val.npz',C/'val_meta.pkl']
if (R/'masked_probabilities.npz').exists():source.append(R/'masked_probabilities.npz')
(R/'audit.json').write_text(json.dumps({'regimes':list(regimes),'confidence_cuts':cuts,'null_threshold_means':'positive infinity','sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}},indent=2))
print('PRIMARY full joint at90:')
for r in rows:
 if r['seed']==20261011 and r['regime']=='full' and r['alpha']==.1 and r['method']=='joint' and r['subset'] in ['all','high_LUNA','low_confidence']:print(r)
print('SENSITIVITY full joint at90:')
for r in summary:
 if r['regime']=='full' and r['alpha']==.1 and r['method']=='joint' and r['subset']=='all':print(r)
print('FINISHED',list(regimes))
