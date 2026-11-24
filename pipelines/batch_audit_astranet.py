"""Batch-size / precision numerical audit of the ASTRANet ensemble (v7fixmd6), adapted from
the user's batch_audit.py. Fixed weights and fixed held-out inputs (the ZTF test split,
3,396 spectra with their redshift/phase); only the batch size and the arithmetic change.

Configurations:
  fp32  : TF32 disabled for matmul and cuDNN (strict FP32)
  tf32  : PyTorch defaults (cuDNN convolutions may use TF32)
  bf16  : torch.autocast(bfloat16) on top of the strict-FP32 settings
Score = energy of the ensemble-mean fine logits, -logsumexp (higher = more anomalous),
as for the LUNA energy score. Nights: 1000 random draws of NIGHT_SIZE test spectra; we
compare the top-15 membership against the FP32, batch-8 reference.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / 'pipelines')); sys.path.insert(0, str(REPO))
from _common import load_config, make_astra_config, load_ensemble, build_hierarchy_mappings  # noqa: E402
from astranet.dataset import build_loaders  # noqa: E402

BATCH_SIZES = [8, 16, 32]; REF = ('fp32', 8)
N_NIGHTS, TOP, SEED = 1000, 15, 0

def set_mode(mode):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = (mode == 'tf32')
    torch.backends.cudnn.allow_tf32 = (mode == 'tf32')

@torch.no_grad()
def ensemble_logits(models, X, PH, Z, bs, mode):
    set_mode(mode); out = []
    for i in range(0, len(X), bs):
        x, ph, z = X[i:i+bs].cuda(), PH[i:i+bs].cuda(), Z[i:i+bs].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(mode == 'bf16')):
            l = torch.stack([m(x, ph, z)['fine_logits'].float() for m in models]).mean(0)
        out.append(l.cpu())
    return torch.cat(out).double()

def energy(l): return -torch.logsumexp(l, dim=1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--night-size', type=int, default=300); ap.add_argument('--out', default='results.json')
    args = ap.parse_args()
    torch.manual_seed(SEED)
    cfg = load_config(str(REPO / 'configs/v7fixsn.yaml')); ac = make_astra_config(cfg)
    _, _, sup = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    models, labels = load_ensemble('v7fixsn6', REPO / 'models' / cfg['weights_source'], ac, len(sup), torch.device('cuda'))
    _, _, _, test_loader, _ = build_loaders(ac)
    X, PH, Z, Y = [], [], [], []
    for b in test_loader:
        X.append(b['x']); PH.append(b['phase']); Z.append(b['z']); Y.append(b['y'])
    X, PH, Z, Y = torch.cat(X), torch.cat(PH), torch.cat(Z), torch.cat(Y)
    env = dict(N=len(X), gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), members=labels, night_size=args.night_size)
    print(json.dumps(env))
    # sanity: reproduce the cached test logits (batch 128, default settings)
    cache = np.load(REPO / 'cache/v7fixsn6/test.npz')['fine_logits']
    L = {}
    for mode in ['fp32', 'tf32', 'bf16']:
        for bs in BATCH_SIZES:
            L[(mode, bs)] = ensemble_logits(models, X, PH, Z, bs, mode); print('done', mode, bs, flush=True)
    sanity = float(np.abs(L[('tf32', 32)].numpy() - cache).max())
    rng = np.random.default_rng(SEED)
    nights = [rng.choice(len(X), args.night_size, replace=False) for _ in range(N_NIGHTS)]
    eref = energy(L[REF]).numpy()
    margins = [np.sort(eref[i])[::-1][TOP-1] - np.sort(eref[i])[::-1][TOP] for i in nights]
    acc = {k: float((v.argmax(1) == Y).double().mean()) for k, v in L.items()}
    rows = []
    for (mode, bs), l in L.items():
        if (mode, bs) == REF: continue
        d = (l - L[REF]).abs(); e = energy(l).numpy()
        changed, agree = 0, []
        for i in nights:
            a = set(i[np.argsort(-eref[i], kind='stable')[:TOP]]); b = set(i[np.argsort(-e[i], kind='stable')[:TOP]])
            agree.append(len(a & b) / TOP); changed += a != b
        rows.append(dict(mode=mode, bs=bs, bitwise_identical=bool(torch.equal(l, L[REF])), max_abs_dlogit=float(d.max()),
                         median_abs_dlogit=float(d.median()), max_abs_denergy=float(np.abs(e - eref).max()),
                         argmax_flips=int((l.argmax(1) != L[REF].argmax(1)).sum()), nights_top15_changed=int(changed),
                         mean_agreement=float(np.mean(agree)), min_agreement=float(np.min(agree)), accuracy=acc[(mode, bs)]))
    # within-precision batch sensitivity (each mode vs its own batch-8 run)
    within = []
    for mode in ['fp32', 'tf32', 'bf16']:
        for bs in [16, 32]:
            l0, l = L[(mode, 8)], L[(mode, bs)]; e0, e = energy(l0).numpy(), energy(l).numpy()
            ch = sum(set(i[np.argsort(-e0[i], kind='stable')[:TOP]]) != set(i[np.argsort(-e[i], kind='stable')[:TOP]]) for i in nights)
            within.append(dict(mode=mode, bs=bs, bitwise_identical=bool(torch.equal(l0, l)), max_abs_dlogit=float((l - l0).abs().max()),
                               argmax_flips=int((l.argmax(1) != l0.argmax(1)).sum()), nights_top15_changed=int(ch)))
    res = dict(env=env, reference='fp32 (TF32 off), batch 8', sanity_max_abs_vs_cache=sanity, accuracy_ref=acc[REF],
               median_top15_margin=float(np.median(margins)), min_top15_margin=float(np.min(margins)),
               vs_reference=rows, within_precision_vs_batch8=within)
    Path(args.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
