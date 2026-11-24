import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parent; REPO=REPO
sys.path[:0]=[str(REPO),str(REPO/'pipelines')]
from _common import load_config,make_astra_config,load_ensemble
from astranet.model import build_hierarchy_mappings
from astranet.dataset import make_loader
torch.set_num_threads(2)
cfg=load_config(REPO/'configs/spectranet.yaml');ac=make_astra_config(cfg);device=torch.device('cuda:0')
_,_,coarse=build_hierarchy_mappings(ac.hierarchy,ac.class_names)
# combined ensemble: 6 TCN members (tcn6, configs/tcn.yaml) + 6 SpectraNet members (spectranet6)
acm=make_astra_config(load_config(REPO/'configs/tcn.yaml'))
m1,n1=load_ensemble('tcn6',REPO/'models/tcn',acm,len(coarse),device)
m2,n2=load_ensemble('spectranet6',REPO/'models/spectranet',ac,len(coarse),device)
models,names=m1+m2,['tcn/'+x for x in n1]+['spectranet/'+x for x in n2]
out={};audit={'device':str(device),'members':names,'alignment':{}}
with torch.inference_mode():
 for split in ['val','test']:
  data=torch.load(Path(ac.data_dir)/f'{split}.pt',map_location='cpu',weights_only=False)
  meta=pickle.load(open(REPO/f'cache/astranet12/{split}_meta.pkl','rb'))
  ids=np.array([str(m['obj_id']) for m in data['meta']]);assert np.array_equal(ids,[str(m['obj_id']) for m in meta])
  cached=np.load(REPO/f'cache/astranet12/{split}.npz')
  loader=make_loader(data,32,shuffle=False,labeled=True,augment=False,num_workers=0,pin_memory=True,compute_derivative=ac.compute_derivative,class_names=ac.class_names,savgol_window=ac.savgol_window,savgol_poly=ac.savgol_poly)
  assert np.array_equal(np.asarray(loader.dataset.y),cached['labels'])
  probs={k:[] for k in ['no_z','spectrum_only']}
  for bi,b in enumerate(loader):
   x=b['x'].to(device);ph=b['phase'].to(device);z=b['z'].to(device)
   if bi==0:
    logits=torch.stack([m(x,ph,z)['fine_logits'] for m in models]).mean(0).cpu().numpy()
    delta=float(abs(logits-cached['fine_logits'][:len(x)]).max());assert delta<1e-3,delta;audit['alignment'][split]=delta
   for regime in probs:
    zz=torch.full_like(z,float('nan'));pp=ph if regime=='no_z' else torch.full_like(ph,float('nan'))
    logits=torch.stack([m(x,pp,zz)['fine_logits'] for m in models]).mean(0)
    probs[regime].append(logits.softmax(-1).cpu().numpy())
   if bi%20==0:print(split,bi,len(loader),flush=True)
  obj,inv,n=np.unique(ids,return_inverse=True,return_counts=True)
  out[split+'_ids']=obj
  out[split+'_y']=np.array([cached['labels'][inv==i][0] for i in range(len(obj))])
  for regime,parts in probs.items():
   p=np.concatenate(parts);a=np.zeros((len(obj),7));np.add.at(a,inv,p);a/=n[:,None];out[split+'_'+regime]=a
np.savez_compressed(R/'masked_probabilities.npz',**out)
audit['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(R/'extraction_audit.json').write_text(json.dumps(audit,indent=2));print('DONE',flush=True)
