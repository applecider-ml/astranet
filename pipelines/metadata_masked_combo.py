"""Metadata-masked classification for the combined ensemble astranet12 (tcn6 + spectranet6).
Runs each 6-member ensemble under the four masking conditions, stores per-member softmax, and reports the
12-member ensemble with both prob-mean and logit-mean (cache convention).
Outputs -> results/astranet12/classification/metadata_masked.{json,md}"""
import sys, json; from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
REPO=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(REPO/'pipelines'),str(REPO)]
from _common import load_config, make_astra_config, load_ensemble
from astranet.model import build_hierarchy_mappings
from astranet.dataset import build_loaders
from metadata_masked_eval import CONDITIONS, metrics
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=REPO/'results/astranet12/classification'; out.mkdir(parents=True,exist_ok=True)
L={}; Y=None
for spec,cfgf in [('tcn6','configs/tcn.yaml'),('spectranet6','configs/spectranet.yaml')]:
    cfg=load_config(str(REPO/cfgf)); ac=make_astra_config(cfg); _,_,sup=build_hierarchy_mappings(ac.hierarchy,ac.class_names)
    models,labels=load_ensemble(spec,REPO/'models'/cfg['weights_source'],ac,len(sup),dev); _,_,_,test_loader,_=build_loaders(ac)
    for name,(mz,mp) in CONDITIONS.items():
        lg,ys=[],[]
        with torch.no_grad():
            for b in test_loader:
                x=b['x'].to(dev); ph=b['phase'].to(dev); z=b['z'].to(dev)
                if mz: z=torch.full_like(z,float('nan'))
                if mp: ph=torch.full_like(ph,float('nan'))
                lg.append(torch.stack([m(x,ph,z)['fine_logits'] for m in models]).cpu()); ys.append(b['y'])
        L.setdefault(name,[]).append(torch.cat(lg,1).numpy()); y=torch.cat(ys).numpy()
        assert Y is None or np.array_equal(Y,y); Y=y
        print(spec,name,'acc',float((L[name][-1].mean(0).argmax(1)==y).mean()),flush=True)
    del models; torch.cuda.empty_cache()
K=len(ac.class_names); res={'spec':'astranet12','class_names':ac.class_names,'conditions':{}}
for name in CONDITIONS:
    lg=np.concatenate(L[name]); P=torch.softmax(torch.from_numpy(lg),-1).numpy()
    r=metrics(P.mean(0),Y,K); r['logit_mean']=metrics(lg.mean(0),Y,K); r['old6']=metrics(P[:6].mean(0),Y,K)['acc']; r['sn6']=metrics(P[6:].mean(0),Y,K)['acc']
    res['conditions'][name]=r
np.savez_compressed(out/'masked_member_logits.npz',y=Y,**{n:np.concatenate(v) for n,v in L.items()})
(out/'metadata_masked.json').write_text(json.dumps(res,indent=2))
lines=["| condition | acc | macro recall | "+" | ".join(ac.class_names)+" | acc (logit mean) |","|---|---|---|"+"---|"*K+"---|"]
for n,r in res['conditions'].items(): lines.append(f"| {n} | {r['acc']:.3f} | {r['macro_recall']:.3f} | "+" | ".join(f"{v:.2f}" for v in r['per_class_recall'])+f" | {r['logit_mean']['acc']:.3f} |")
(out/'metadata_masked.md').write_text("\n".join(lines)+"\n"); print("\n".join(lines))
