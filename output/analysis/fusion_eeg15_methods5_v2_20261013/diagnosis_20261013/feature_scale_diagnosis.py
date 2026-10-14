"""Read-only cached-feature scale audit; no model imports, forwards, or fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FEATURES = ROOT / 'tridep_ccml/experiments/Fusion-EEG15-Methods5-v2-37-20261013/features'
DEST = Path(__file__).with_suffix('.json')
EEG = [f'E{i:02d}' for i in range(14)] + ['E15']
THRESHOLDS = [3, 5, 10, 100, 1000, 10000]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def describe(values):
    a = np.asarray(values).ravel()
    assert np.isfinite(a).all()
    return {'count': int(a.size), 'min': float(a.min()), 'max': float(a.max()),
            'mean': float(a.mean()), 'quantiles': {str(q): float(np.quantile(a, q))
            for q in [0.5, 0.9, 0.95, 0.99, 0.999]}}

def describe_abs(values):
    a = np.abs(np.asarray(values).ravel())
    return {**describe(a), 'counts_strictly_above': {str(k): int((a > k).sum()) for k in THRESHOLDS}}

def main():
    assert not DEST.exists(), 'Refuse to overwrite a prior diagnostic'
    pools = {name: {'std': [], 'train': [], 'val': [], 'test': []} for name in ['S', 'T', *EEG]}
    records, sources, extreme = [], [], []
    for group in ['ST', *EEG]:
        for seed in [11, 22, 33]:
            for fold in range(1, 6):
                path = FEATURES / group / f'seed_{seed}' / f'fold_{fold}.npz'
                meta = json.loads(path.with_suffix('.json').read_text('utf-8'))
                digest = sha(path)
                assert digest == meta['cache']['sha256']
                sources.append({'path': str(path.relative_to(ROOT)), 'sha256': digest,
                                'metadata_sha256': sha(path.with_suffix('.json'))})
                with np.load(path, allow_pickle=False) as z:
                    x, ids, splits = [z[k].copy() for k in ['features', 'subject_ids', 'splits']]
                if group != 'ST':
                    x = x[:, None, :]
                    # All EEG and ST sources use exactly the same ordered people and roles.
                    stpath = FEATURES / 'ST' / f'seed_{seed}' / f'fold_{fold}.json'
                    roles = json.loads(stpath.read_text('utf-8'))['roles']
                else:
                    roles = meta['roles']
                lookup = {str(s): i for i, s in enumerate(ids)}
                ix = {role: np.asarray([lookup[str(s)] for s in members], dtype=np.int64)
                      for role, members in roles.items()}
                assert all(all(splits[j] == role for j in js) for role, js in ix.items())
                mean = x[ix['train']].astype(np.float64).mean(0)
                rawstd = x[ix['train']].astype(np.float64).std(0)
                effective = np.maximum(rawstd, 1e-6)
                normalized = ((x.astype(np.float64) - mean) / effective).astype(np.float32)
                names = ['S', 'T'] if group == 'ST' else [group]
                for v, name in enumerate(names):
                    row = {'modality': name, 'seed': seed, 'fold': fold,
                           'raw_train_std': describe(rawstd[v]),
                           'raw_std_below_floor': int((rawstd[v] < 1e-6).sum()),
                           'raw_std_equal_zero': int((rawstd[v] == 0).sum()),
                           'roles': {role: describe_abs(normalized[js, v]) for role, js in ix.items()}}
                    pools[name]['std'].append(rawstd[v])
                    for role, js in ix.items():
                        pools[name][role].append(normalized[js, v].ravel())
                        # Keep the 5 largest coordinates per modality/fold/role for traceable examples.
                        a = np.abs(normalized[js, v]); order = np.argsort(a.ravel())[-5:][::-1]
                        for flat in order:
                            subject, dim = np.unravel_index(flat, a.shape)
                            j = int(js[subject])
                            extreme.append({'modality': name, 'seed': seed, 'fold': fold, 'role': role,
                                            'subject_id': str(ids[j]), 'dimension_zero_based': int(dim),
                                            'raw_value': float(x[j, v, dim]), 'train_mean': float(mean[v, dim]),
                                            'raw_train_std': float(rawstd[v, dim]),
                                            'effective_train_std': float(effective[v, dim]),
                                            'normalized_value': float(normalized[j, v, dim])})
                    records.append(row)
    groups = {}
    for name, pool in pools.items():
        std = np.concatenate(pool['std'])
        groups[name] = {'bundles': len(pool['std']), 'std_coordinate_count': int(std.size),
                        'raw_train_std': describe(std), 'raw_std_below_floor': int((std < 1e-6).sum()),
                        'raw_std_equal_zero': int((std == 0).sum()),
                        'roles': {role: describe_abs(np.concatenate(pool[role])) for role in ['train', 'val', 'test']}}
    all_pools = {'all_unique_modality_bundles': list(pools), 'ST_unique': ['S', 'T'], 'EEG_unique': EEG}
    global_summary = {}
    for group, names in all_pools.items():
        std = np.concatenate([np.concatenate(pools[n]['std']) for n in names])
        global_summary[group] = {
            'modality_bundles': sum(len(pools[n]['std']) for n in names),
            'raw_std_below_floor': int((std < 1e-6).sum()), 'raw_std_equal_zero': int((std == 0).sum()),
            'raw_train_std': describe(std),
            'roles': {role: describe_abs(np.concatenate([np.concatenate(pools[n][role]) for n in names]))
                      for role in ['train', 'val', 'test']}}
    out = {'schema': 'feature_scale_diagnosis_v1', 'mode': 'read_only_cached_features_no_model_forward',
           'definitions': {
               'standardization': 'Exact training.py arithmetic: train-only float64 population mean/std, floor 1e-6; output float32.',
               'counting': '240 source NPZ files = 15 ST caches + 225 EEG caches. ST is split into S/T: 255 unique modality-fold-seed bundles; never duplicated over fusion methods or over EEG reuse of S/T.',
               'coordinates': 'One count per subject feature coordinate within a cached seed/fold/role; each vector has 128 dimensions. The same person reappears across seeds/folds; counts are not independent people.',
               'quantiles': 'NumPy default linear pooled-coordinate quantile, applied to abs(standardized input). Raw train std quantiles are across modality/fold/seed/dimension.',
               'threshold_counts': 'Strict > comparison after float32 normalization.',
               'scope': 'Post hoc feature distribution diagnostic only. No test threshold, epoch, model, optimization, or source changes. Association is not causal attribution.'},
           'source_count': len(sources), 'global': global_summary, 'by_modality': groups,
           'top_examples': {role: sorted([x for x in extreme if x['role'] == role],
                                        key=lambda x: abs(x['normalized_value']), reverse=True)[:30]
                            for role in ['train', 'val', 'test']},
           'per_modality_fit': records, 'sources': sources,
           'script_sha256': sha(Path(__file__)),
           'training_source_sha256': sha(ROOT / 'tridep_ccml/fusion_eeg_grid_v2/training.py')}
    DEST.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'path': str(DEST), 'source_count': len(sources), 'global': global_summary,
                      'modality_outliers': {k: {'std_floor_count': v['raw_std_below_floor'],
                                                'test_abs_max': v['roles']['test']['max'],
                                                'test_abs_p99': v['roles']['test']['quantiles']['0.99']}
                                           for k, v in groups.items()},
                      'top_test': out['top_examples']['test'][:5]}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
