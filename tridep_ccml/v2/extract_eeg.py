"""Label-resolved EEG export into an independent, resumable v2 namespace."""
from __future__ import annotations

import argparse
import sys
import time

from .common import (WORK, OUT, REVISION, setup, descriptor, verify_descriptor,
                     read_json, require, set_seed, label_indices, save_array, write_new, digest)
import mne
import numpy as np
import torch
from scipy.io import loadmat

sys.path.insert(0, str(WORK/'S00/upstream-dev'))
from lib.models.cbramod.cbramod import CBraMod


def patches(mat, indices, names):
    raw = mne.io.RawArray(mat[indices], mne.create_info(names, 250, 'eeg'), verbose='ERROR')
    raw.resample(200, verbose='ERROR')
    raw.filter(.3, 75, verbose='ERROR')
    raw.notch_filter(50, verbose='ERROR')
    require(raw.n_times >= 60000, 'EEG shorter than 300 seconds')
    x = raw.get_data(start=0, stop=60000)
    require(np.isfinite(x).all(), 'Nonfinite EEG after filtering')
    return x.reshape(19, 60, 5, 200).transpose(1, 0, 2, 3).copy()


def embed(model, x, device):
    with torch.inference_mode():
        y = [model(torch.tensor(x[i:i+4], dtype=torch.float32, device=device))
             .mean(dim=(1, 2)).cpu().numpy() for i in range(0, 60, 4)]
    return np.concatenate(y).astype(np.float32)


def run(subject=None):
    context, cfg = setup()
    set_seed(11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    revision = read_json(REVISION)
    required = [e['required_source_label'] for e in revision['selection']]
    names = [e['target_name'] for e in revision['selection']]
    staged = {s['subject_id']: s for s in read_json(WORK/'S00/manifests/staged_dataset.json')}
    subjects = {s['subject_id']: s for s in read_json(WORK/'S00/manifests/subjects.json')}
    old_manifest = {r['subject_id']: r for r in read_json(WORK/'S00/features/B0-37/features_manifest.json') if r['modality'] == 'eeg'}
    weights = WORK/'S00/upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth'
    weight_desc = descriptor(weights)
    require(all(r['details']['checkpoint_sha256'] == weight_desc['sha256'] for r in old_manifest.values()), 'Backbone differs from original EEG export')
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    ckpt = ckpt.get('state_dict', ckpt)
    ckpt = {k.replace('backbone.', ''): v for k,v in ckpt.items()}
    model.proj_out = torch.nn.Identity()
    model.load_state_dict({k: ckpt[k] for k in model.state_dict()}, strict=True)
    model.requires_grad_(False).to(device).eval()
    producer = descriptor(__file__)
    common_source = descriptor(WORK/'tridep_ccml/v2/common.py')
    vendor = {p.relative_to(WORK).as_posix(): digest(p) for p in sorted((WORK/'S00/upstream-dev/lib/models/cbramod').rglob('*.py'))}
    ids = [subject] if subject else context.ids
    require(set(ids) <= set(context.ids), 'Unknown subject')
    for sid in ids:
        start = time.perf_counter()
        src = WORK/'S00'/staged[sid]['eeg_mat']
        require(digest(src) == staged[sid]['mat_sha256'], 'Staged EEG hash changed')
        set_path = WORK/subjects[sid]['eeg_set']
        set_data = loadmat(set_path, simplify_cells=True)
        set_data = set_data.get('EEG', set_data)
        labels = [str(ch['labels']) for ch in set_data['chanlocs']]
        indices = label_indices(labels, required)
        require(indices == [e['expected_zero_based_index'] for e in revision['selection']], 'Unexpected raw ordering; investigate rather than silently adapting')
        provenance = {'subject_id': sid, 'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
                      'source_mat': descriptor(src), 'source_set': descriptor(set_path),
                      'revision': cfg['revision'], 'producer': producer, 'common_source': common_source,
                      'backbone': weight_desc, 'backbone_code_hashes': vendor,
                      'selection_by_label': required, 'resolved_indices': indices,
                      'native_scale': True, 'fit_subject_ids': [], 'shape': [60, 200],
                      'device': device, 'torch': torch.__version__, 'mne': mne.__version__,
                      'normalization_added': False, 'reference_changed': False}
        target = OUT/'eeg'/f'{sid}.npy'
        sidecar = target.with_suffix('.json')
        if sidecar.exists():
            old = read_json(sidecar)
            require(old['provenance'] == provenance, 'Stale EEG cache provenance')
            verify_descriptor(old['feature'])
            require(np.load(target).shape == (60,200) and np.isfinite(np.load(target)).all(), 'Invalid EEG cache')
            print(f'EEG reuse {sid}', flush=True)
            continue
        mat = loadmat(src)['eeg_mat']
        # Reproduce the original export once with its original indices before changing selection.
        smoke_path = OUT/'extraction_original_path_check.json'
        if not smoke_path.exists():
            original_indices = [e['old_zero_based_index'] for e in revision['selection']]
            reproduced = embed(model, patches(mat, original_indices, names), device)
            original_path = WORK/'S00/features/B0-37'/old_manifest[sid]['path']
            require(digest(original_path) == old_manifest[sid]['sha256'], 'Original EEG cache hash changed')
            original = np.load(original_path, allow_pickle=False)
            require(np.allclose(reproduced, original, rtol=1e-4, atol=1e-4), 'Original EEG path cannot be reproduced')
            write_new(smoke_path, {'subject_id': sid, 'max_abs_error': float(np.max(np.abs(reproduced-original))),
                                  'rtol': 1e-4, 'atol': 1e-4, 'original_feature': descriptor(original_path),
                                  'scope': 'one real subject; same backbone and preprocessing, original indices'})
        x = patches(mat, indices, names)
        new = embed(model, x, device)
        require(new.shape == (60,200) and np.isfinite(new).all(), 'Invalid revised EEG embeddings')
        save_array(target, new)
        write_new(sidecar, {'provenance': provenance, 'feature': descriptor(target),
                            'filtered_patch_rms': float(np.sqrt(np.mean(x*x))),
                            'filtered_patch_abs_p99': float(np.quantile(np.abs(x), .99)),
                            'embedding_rms': float(np.sqrt(np.mean(new.astype(float)**2)))})
        print(f'EEG corrected {sid} [60,200] seconds={time.perf_counter()-start:.1f}', flush=True)
    if all((OUT/'eeg'/f'{sid}.json').exists() for sid in context.ids):
        rows = [read_json(OUT/'eeg'/f'{sid}.json') for sid in context.ids]
        write_new(OUT/'eeg_manifest.json', rows)
        print('All 37 corrected EEG exports complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject')
    run(parser.parse_args().subject)
