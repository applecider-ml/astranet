"""Per-member embeddings / logits for val, test and rare (for LUNA-on-ensemble ablations).
Usage: python pipelines/extract_members.py --spec v7fixsn6 --config configs/v7fixsn.yaml
Writes cache/<spec>/members_{val,test,rare}.npz with emb (M,N,192), fine_logits (M,N,7) and obj ids."""
import argparse, pickle, numpy as np, torch
from pathlib import Path
from _common import REPO, load_config, make_astra_config, device_for, load_ensemble, ENSEMBLE_SPECS
from astranet.model import build_hierarchy_mappings
from astranet.dataset import build_loaders
from build_ensemble import _rare_loader
ap=argparse.ArgumentParser(); ap.add_argument('--spec',required=True); ap.add_argument('--config',default=None); a=ap.parse_args()
cfg=load_config(a.config); ac=make_astra_config(cfg); dev=device_for(cfg)
_,_,sup=build_hierarchy_mappings(ac.hierarchy,ac.class_names); members,labels=load_ensemble(a.spec,REPO/'models'/cfg['weights_source'],ac,len(sup),dev)
_,_,val_loader,test_loader,_=build_loaders(ac); rare_loader=_rare_loader(ac,Path(cfg['data']['rare_pt']))
out=REPO/'cache'/a.spec
with torch.no_grad():
    for name,loader in [('val',val_loader),('test',test_loader),('rare',rare_loader)]:
        E,L,ids=[[] for _ in members],[[] for _ in members],[]
        for b in loader:
            x,ph,z=b['x'].to(dev),b['phase'].to(dev),b['z'].to(dev)
            for i,m in enumerate(members):
                o=m(x,ph,z); E[i].append(o['embedding'].cpu()); L[i].append(o['fine_logits'].cpu())
            ids.extend([str(mm['obj_id']) for mm in b['meta']])
        np.savez(out/f'members_{name}.npz',emb=torch.stack([torch.cat(e) for e in E]).numpy(),fine_logits=torch.stack([torch.cat(l) for l in L]).numpy(),obj_id=np.array(ids),labels=np.array(labels))
        print(name,len(ids),flush=True)
