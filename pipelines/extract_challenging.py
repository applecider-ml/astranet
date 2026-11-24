"""Ensemble features for data/challenging.pt (shuffle=False) -> cache/challenging/<spec>.npz"""
import argparse, numpy as np, torch
from pathlib import Path
from _common import REPO, load_config, make_astra_config, device_for, load_ensemble, extract
from astranet.model import build_hierarchy_mappings
from build_ensemble import _rare_loader
ap=argparse.ArgumentParser(); ap.add_argument('--spec',required=True); ap.add_argument('--config',default=None); ap.add_argument('--tag',default=None); a=ap.parse_args()
cfg=load_config(a.config); ac=make_astra_config(cfg); dev=device_for(cfg); _,_,sup=build_hierarchy_mappings(ac.hierarchy,ac.class_names)
members,_=load_ensemble(a.spec,REPO/'models'/cfg['weights_source'],ac,len(sup),dev)
loader=_rare_loader(ac,REPO/'data/challenging.pt'); d=extract(members,loader,dev,labeled=False)
OUT=REPO/'cache/challenging'/(f'{a.spec}.npz' if not getattr(a,'tag',None) else f'{a.spec}_{a.tag}.npz'); np.savez(OUT,emb=d['emb'],fine_logits=d['fine_logits'],coarse_logits=d['coarse_logits'],proj_fine=d['proj_fine'])
np.savez(str(OUT).replace('.npz','_per_model_softmax.npz'),softmax=d['per_model_softmax']); print('done',d['emb'].shape)
