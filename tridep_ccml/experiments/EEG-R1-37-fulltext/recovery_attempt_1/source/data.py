"""Read-only historical inputs; native raw/token exports and train-only scaling."""
from __future__ import annotations

import argparse
import sys
import time
from .common import *
from scipy.io import loadmat
from tridep_ccml.v2.common import label_indices
from tridep_ccml.v2.extract_eeg import patches


def check_splits(context):
    tests = []
    for fold in FOLDS:
        parts = context.folds[f'fold_{fold}']
        flattened = sum([parts[k] for k in ['train', 'val', 'test']], [])
        require(len(flattened) == len(set(flattened)) == 37 and set(flattened) == set(context.ids),
                'Subject leakage/incomplete fold')
        require(len(parts['train']) in [26, 27] and len(parts['val']) == 3, 'Fold sizes changed')
        tests += parts['test']
    require(sorted(tests) == sorted(context.ids), 'OOF coverage mismatch')


def load_backbone():
    from lib.models.cbramod.cbramod import CBraMod
    path = WORK/'S00/upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth'
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                   seq_len=30, n_layer=12, nhead=8)
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = {k.replace('backbone.', ''): v for k, v in state.get('state_dict', state).items()}
    model.proj_out = torch.nn.Identity()
    model.load_state_dict({k: state[k] for k in model.state_dict()}, strict=True)
    return model.requires_grad_(False).cuda().eval(), descriptor(path)


def raw_and_tokens(context):
    """No jitter, synthetic observations, augmentation cache or label-dependent preprocessing."""
    set_seed(11)
    require(torch.cuda.is_available(), 'R1 requires registered single CUDA GPU')
    revision = read_json(REVISION)
    names = [r['target_name'] for r in revision['selection']]
    required = [r['required_source_label'] for r in revision['selection']]
    expected = [r['expected_zero_based_index'] for r in revision['selection']]
    staged = {r['subject_id']: r for r in read_json(WORK/'S00/manifests/staged_dataset.json')}
    subjects = {r['subject_id']: r for r in read_json(WORK/'S00/manifests/subjects.json')}
    model, weights = load_backbone()
    before = state_digest(model.state_dict())
    rows = []
    for sid in context.ids:
        start = time.perf_counter()
        old = read_json(TC/f'experiments/CCML-v2-E-37-fulltext/eeg/{sid}.json')
        p = old['provenance']
        require(p['fit_subject_ids'] == [] and p['backbone'] == weights and p['native_scale'] and
                not p['normalization_added'] and p['cohort_sha256'] == context.cohort_sha256 and
                p['folds_sha256'] == context.folds_sha256, 'Revised EEG ancestry mismatch')
        for key in ['source_mat', 'source_set', 'revision', 'producer', 'common_source']:
            verify_descriptor(p[key])
        for name, sha in p['backbone_code_hashes'].items():
            require(digest(WORK/name) == sha, 'Frozen backbone source code changed')
        mat_path = WORK/'S00'/staged[sid]['eeg_mat']
        set_path = WORK/subjects[sid]['eeg_set']
        require(digest(mat_path) == staged[sid]['mat_sha256'] == p['source_mat']['sha256'], 'MAT source mismatch')
        require(descriptor(set_path) == p['source_set'], 'SET source mismatch')
        obj = loadmat(set_path, simplify_cells=True)
        labels = [str(ch['labels']) for ch in obj.get('EEG', obj)['chanlocs']]
        indices = label_indices(labels, required)
        require(indices == expected == p['resolved_indices'], 'Electrode order mismatch')
        mean_cache = np.load(verify_descriptor(old['feature']), allow_pickle=False)
        raw_path, token_path = OUT/f'cache/raw/{sid}.npy', OUT/f'cache/tokens/{sid}.npy'
        meta_path = OUT/f'cache/provenance/{sid}.json'
        if meta_path.exists():
            row = read_json(meta_path)
            for key in ['raw', 'tokens', 'reference']:
                verify_descriptor(row[key])
            require(row['source_mat'] == p['source_mat'] and row['source_set'] == p['source_set'] and
                    row['backbone'] == weights and row['labels'] == required, 'Stale R1 source')
            rows.append(row)
            print(f'CACHE reuse {sid}', flush=True)
            continue
        x = patches(loadmat(mat_path)['eeg_mat'], indices, names)
        raw = x.astype(np.float32).reshape(60, 19, 1000)
        require(np.array_equal(raw.reshape(60, 19, 5, 200), x.astype(np.float32)), 'Flatten changes axes')
        with torch.inference_mode():
            tokens = torch.cat([model(torch.tensor(x[i:i+4], dtype=torch.float32, device='cuda')).cpu()
                                for i in range(0, 60, 4)]).numpy()
        require(tokens.shape == (60, 19, 5, 200) and np.isfinite(tokens).all(), 'Invalid CBraMod tokens')
        # Repeat original CUDA mean reduction, so export batch boundaries are identical.
        reduced = torch.from_numpy(tokens).cuda().mean((1, 2)).cpu().numpy()
        require(np.allclose(reduced, mean_cache, rtol=1e-4, atol=1e-4), 'Token means fail revised-cache replay')
        save_array(raw_path, raw)
        save_array(token_path, tokens)
        row = {'subject_id': sid, 'label': context.labels[sid], 'source_mat': p['source_mat'],
            'source_set': p['source_set'], 'backbone': weights, 'revision': descriptor(REVISION),
            'labels': required, 'target_names': names, 'resolved_indices': indices,
            'raw': descriptor(raw_path), 'tokens': descriptor(token_path), 'reference': old['feature'],
            'raw_shape': list(raw.shape), 'token_shape': list(tokens.shape), 'dtype': 'float32',
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'augmentation': False, 'native_scale': True, 'fit_subject_ids': [], 'flatten_reconstruction_exact': True,
            'token_mean_rtol': 1e-4, 'token_mean_atol': 1e-4,
            'token_mean_max_abs_error': float(np.abs(reduced-mean_cache).max()),
            'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
            'raw_rms': float(np.sqrt(np.mean(raw.astype(float)**2))),
            'raw_abs_p99': float(np.quantile(np.abs(raw), .99)), 'seconds': time.perf_counter()-start}
        write_new(meta_path, row)
        rows.append(row)
        print(f'CACHE {sid} raw/tokens verified seconds={row["seconds"]:.1f}', flush=True)
    require(state_digest(model.state_dict()) == before and not model.training and
            not any(p.requires_grad for p in model.parameters()), 'Frozen backbone changed')
    del model
    torch.cuda.empty_cache()
    return rows


def pooled_inputs(context, rows):
    root = TC/'experiments/CCML-v2-S-37-fulltext'
    manifest = read_json(root/'input_manifest.json')
    require(manifest['subject_ids'] == context.ids and manifest['fit_subject_ids'] == [] and
            not manifest['labels_used'] and manifest['cohort_sha256'] == context.cohort_sha256,
            'S-control input identity changed')
    for source in manifest['sources']:
        for key in ['eeg', 'eeg_manifest', 'text']:
            verify_descriptor(source[key])
    text_manifest = read_json(verify_descriptor(manifest['text_manifest']))
    require(all(r['fit_subject_ids'] == [] and not r.get('labels_used', False) and
                r['folds_sha256'] == context.folds_sha256 for r in text_manifest), 'Text not label-free')
    from tridep_ccml.v2.short_audio import load_pool
    speech, speech_source = load_pool(context)
    with np.load(verify_descriptor(manifest['cache']), allow_pickle=False) as z:
        require(z['subject_ids'].tolist() == context.ids and np.array_equal(z['speech'], speech), 'AS means changed')
        arrays = {k: z[k].copy() for k in MODALITIES}
    source_by_id = {r['subject_id']: r for r in manifest['sources']}
    seq = []
    for i, row in enumerate(rows):
        x = np.load(verify_descriptor(row['reference']), allow_pickle=False)
        require(np.array_equal(x.mean(0, dtype=np.float32), arrays['eeg'][i]), 'C0 EEG mean differs from S-control')
        txt = np.load(verify_descriptor(source_by_id[row['subject_id']]['text']), allow_pickle=False)
        require(txt.shape == (29, 768) and np.array_equal(txt.mean(0, dtype=np.float32), arrays['text'][i]), 'Text pooling differs')
        seq.append(x)
    arrays['sequence'] = np.stack(seq)
    for key, x in arrays.items():
        save_array(OUT/f'cache/{key}.npy', x)
    return {'S_control_manifest': descriptor(root/'input_manifest.json'), 'speech_source': speech_source,
            'arrays': {key: descriptor(OUT/f'cache/{key}.npy') for key in arrays}}


def fit_standardizer(raw, indices):
    require(raw.ndim == 4 and raw.shape[1:] == (60, 19, 1000), 'Raw axes must be subject/window/channel/time')
    fit = raw[indices].astype(np.float64)
    mean = fit.mean(axis=(0, 1, 3))
    std = fit.std(axis=(0, 1, 3), ddof=0)
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 1e-12).all(), 'Near-zero/nonfinite EEG channel variance')
    return mean, std


def prepare():
    context = RecordContext()
    check_splits(context)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = raw_and_tokens(context)
    pool = pooled_inputs(context, rows)
    raw = np.stack([np.load(verify_descriptor(r['raw']), allow_pickle=False) for r in rows])
    stats = {}
    for fold in FOLDS:
        ids = context.folds[f'fold_{fold}']['train']
        mean, std = fit_standardizer(raw, [context.ids.index(s) for s in ids])
        path = OUT/f'cache/standardizers/fold_{fold}.json'
        write_new(path, {'fold': fold, 'fit_subject_ids': ids, 'mean': mean.tolist(), 'std': std.tolist(),
            'ddof': 0, 'method': 'float64 population moments over all equal train windows; apply then cast float32',
            'minimum_std': 1e-12, 'train_sources': [rows[context.ids.index(s)]['raw'] for s in ids],
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256})
        stats[str(fold)] = descriptor(path)
    manifest = {'subject_ids': context.ids, 'labels': [context.labels[s] for s in context.ids],
        'cohort': descriptor(context.cohort_path), 'folds': descriptor(context.folds_path),
        'subjects': descriptor(WORK/'S00/manifests/subjects.json'), 'augmentation': False,
        'raw_and_tokens': rows, 'pooled_inputs': pool, 'standardizers': stats,
        'frozen_backbone_unchanged': True, 'data_preprocessing': descriptor(TC/'v2/extract_eeg.py')}
    write_new(OUT/'data_manifest.json', manifest)
    print('PREPARE complete: 37 original subjects; raw + token replay; five train-only standardizers.', flush=True)


class FoldData:
    def __init__(self, fold, device='cuda'):
        self.context = RecordContext()
        self.manifest = read_json(OUT/'data_manifest.json')
        require(self.manifest['subject_ids'] == self.context.ids and not self.manifest['augmentation'], 'R1 input identity mismatch')
        for key in ['cohort', 'folds', 'subjects', 'data_preprocessing']:
            verify_descriptor(self.manifest[key])
        self.fold, self.device = fold, device
        self.splits = self.context.folds[f'fold_{fold}']
        self.arrays = {k: torch.from_numpy(np.load(verify_descriptor(v), allow_pickle=False)).to(device)
                       for k, v in self.manifest['pooled_inputs']['arrays'].items()}
        rows = self.manifest['raw_and_tokens']
        raw = np.stack([np.load(verify_descriptor(r['raw']), allow_pickle=False) for r in rows])
        stats = read_json(verify_descriptor(self.manifest['standardizers'][str(fold)]))
        require(stats['fit_subject_ids'] == self.splits['train'], 'Standardizer leaks outside train')
        self.stats = stats
        normalized = ((raw.astype(np.float64)-np.array(stats['mean'])[None,None,:,None]) /
                      np.array(stats['std'])[None,None,:,None]).astype(np.float32)
        self.arrays['raw'] = torch.from_numpy(normalized).to(device)
        tokens = np.stack([np.load(verify_descriptor(r['tokens']), allow_pickle=False) for r in rows])
        self.arrays['tokens'] = torch.from_numpy(tokens).to(device)
        self.labels = torch.tensor([self.context.labels[s] for s in self.context.ids], device=device)
        self.index = {s: i for i, s in enumerate(self.context.ids)}

    def batch(self, ids, condition):
        idx = [self.index[s] for s in ids]
        key = {'U0':'sequence', 'C0':'eeg', 'U1':'raw', 'C1':'raw', 'U2':'raw', 'C2':'raw', 'C3':'tokens', 'C4':'tokens'}[condition]
        inputs = {'eeg': self.arrays[key][idx]}
        if condition.startswith('C'):
            inputs.update({k: self.arrays[k][idx] for k in ['speech', 'text']})
        return inputs, self.labels[idx]


if __name__ == '__main__':
    prepare()
