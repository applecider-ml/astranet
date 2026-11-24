"""Per-spectrum LUNA scores and spectral descriptors of every split, aligned to the cache by identity."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[0])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
import sys, json, pickle, hashlib, itertools, time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO = REPO
sys.path[:0] = [str(REPO / 'pipelines'), str(REPO)]
from _common import load_cache, load_config, make_astra_config
from astranet.model import build_hierarchy_mappings
from astranet.novelty import luna_dict
from physics_view import descriptors, norm
from luna import LUNAPipeline

def main():
    torch.set_num_threads(2)
    cache = REPO / 'cache/astranet12'
    names = ['train', 'val', 'test', 'rare']
    data = {n: load_cache(cache, 'rare_unknown_combined' if n == 'rare' else n) for n in names}
    raw = {n: torch.load(REPO / 'data' / ('rare_unknown.pt' if n == 'rare' else f'object-split/{n}.pt'), weights_only=False, map_location='cpu') for n in names}
    objects = {n: np.array([str(m['obj_id']) for m in data[n]['meta']]) for n in names}
    arrays, audit = {}, {'counts': {}, 'object_overlap': {}, 'exact_spectrum_overlap': {}}
    hashes = {}
    for n in names:
        rawids = np.asarray(raw[n]['obj_id']).astype(str) if n == 'rare' else np.array([str(m['obj_id']) for m in raw[n]['meta']])
        rawfiles = np.asarray(raw[n]['file_path']).astype(str) if n == 'rare' else np.array([str(m['file_path']) for m in raw[n]['meta']])
        # The training feature extractor deliberately shuffles its loader.
        # Match physical features to cached embeddings by identity, never row number.
        index=defaultdict(deque)
        for i,key in enumerate(zip(rawids,rawfiles)):index[key].append(i)
        order=[]
        for m in data[n]['meta']:
            key=(str(m['obj_id']),str(m['file_path']))
            assert index[key], f'cache sample missing or duplicated in raw data: {n} {key}'
            order.append(index[key].popleft())
        assert not any(index.values()), f'raw/cache sample coverage: {n}'
        order=np.asarray(order)
        assert np.array_equal(rawids[order], objects[n]), f'raw/cache alignment: {n}'
        X = np.asarray(raw[n]['flux'] if n == 'rare' else raw[n]['X'], dtype=np.float32).reshape(len(rawids), -1)
        X=X[order]
        assert X.shape[1] == 4096
        hashes[n] = {hashlib.sha256(row.tobytes()).hexdigest() for row in X}
        arrays[n] = X
        audit['counts'][n] = {'spectra': len(X), 'objects': len(set(rawids)), 'unique_flux': len(hashes[n]), 'rows_reordered':int((order!=np.arange(len(order))).sum())}
    for a, b in itertools.combinations(names, 2):
        audit['object_overlap'][f'{a}/{b}'] = sorted(set(objects[a]) & set(objects[b]))
        audit['exact_spectrum_overlap'][f'{a}/{b}'] = len(hashes[a] & hashes[b])
    (ROOT / 'audit.json').write_text(json.dumps(audit, indent=2))
    print('AUDIT', json.dumps(audit), flush=True)
    assert not any(audit['object_overlap'].values()), 'Cross-split object overlap'
    assert not any(audit['exact_spectrum_overlap'].values()), 'Cross-split exact flux overlap'

    out = {}
    for n in names:
        out[f'{n}_objects'] = objects[n]
        out[f'{n}_classes'] = np.array([str(m['class']) for m in data[n]['meta']])
        out[f'{n}_instruments'] = np.array([str(m.get('instrument', '?')) for m in data[n]['meta']])
        out[f'{n}_emb'] = data[n]['emb'].astype('float32')
    cfg = load_config(str(REPO / 'configs/spectranet.yaml'))
    ac = make_astra_config(cfg)
    f2c, _, _ = build_hierarchy_mappings(ac.hierarchy, ac.class_names)
    pipe = LUNAPipeline(combiner='rank', fine_to_coarse=f2c)
    pipe.fit(luna_dict(data['train']))
    out['emb_methods'] = np.array(pipe.active_methods)
    for n in ['val', 'test', 'rare']:
        scores = pipe.score(luna_dict(data[n]))
        out[f'{n}_emb_scores'] = np.column_stack([scores[m] for m in pipe.active_methods])
        print('Embedding scores', n, flush=True)

    D = {n: descriptors(arrays[n]) for n in names}
    med = np.median(D['train'], axis=0)
    iqr = np.diff(np.percentile(D['train'], [25, 75], axis=0), axis=0)[0] + 1e-6
    D = {n: np.clip((d-med)/iqr, -20, 20).astype('float32') for n, d in D.items()}
    pphys = LUNAPipeline(combiner='rank')
    pphys.fit({'embeddings': D['train'], 'labels': data['train']['labels']})
    out['phys_methods'] = np.array(pphys.active_methods)
    for n in names:
        out[f'{n}_phys'] = D[n]
        if n != 'train':
            scores = pphys.score({'embeddings': D[n]})
            out[f'{n}_phys_scores'] = np.column_stack([scores[m] for m in pphys.active_methods])

    # Continuum-subtracted, low-resolution shape, dimension fixed without rare data.
    shape = {}
    for n in names:
        smooth = gaussian_filter1d(norm(arrays[n]), 4, axis=1).reshape(len(arrays[n]), 256, 16).mean(2)
        residual = smooth - gaussian_filter1d(smooth, 8, axis=1)
        shape[n] = residual / np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-8)
    pca = PCA(n_components=32, svd_solver='randomized', random_state=0).fit(shape['train'])
    Z = {n: pca.transform(shape[n]).astype('float32') for n in names}
    knn = NearestNeighbors(n_neighbors=10, algorithm='brute', n_jobs=2).fit(Z['train'])
    unit = {n: data[n]['emb'] / np.maximum(np.linalg.norm(data[n]['emb'], axis=1, keepdims=True), 1e-8) for n in names}
    knnunit = NearestNeighbors(n_neighbors=10, algorithm='brute', n_jobs=2).fit(unit['train'])
    for n in names:
        out[f'{n}_shape'] = Z[n]
        if n != 'train':
            out[f'{n}_shape_scores'] = np.column_stack([knn.kneighbors(Z[n])[0].mean(1), np.linalg.norm(shape[n]-pca.inverse_transform(Z[n]), axis=1)])
            out[f'{n}_unit_knn'] = knnunit.kneighbors(unit[n])[0].mean(1)
    out['shape_methods'] = np.array(['shape_knn', 'shape_residual'])
    out['rare_classes'] = np.asarray(raw['rare']['class']).astype(str)
    assert all(np.isfinite(v).all() for v in out.values() if np.issubdtype(v.dtype, np.number))
    np.savez_compressed(ROOT/'features.npz', **out)
    print('Saved', ROOT/'features.npz', flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
