"""Export fixed, label-free EEG time-jitter views through the frozen CBraMod."""
from __future__ import annotations

import argparse
import sys
import time
from hashlib import sha256
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.io import loadmat

from tridep_ccml.v2.common import (RecordContext, REVISION, descriptor,
                                   digest, label_indices, read_json, require,
                                   set_seed, verify_descriptor, write_new, WORK)
from tridep_ccml.v2.extract_eeg import embed
from .common import OUT, BASE_EEG, VIEWS

sys.path.insert(0, str(WORK / 'S00/upstream-dev'))
from lib.models.cbramod.cbramod import CBraMod

PLAN = WORK / 'output/plans/EEG数据增强与三终点重跑固定计划.md'
WEIGHTS = WORK / 'S00/upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth'
COHORT_N = 37
N_CHANNELS = 19
N_SECONDS = 300
SAMPLE_RATE = 200
WINDOW_SAMPLES = 1000
MAX_SHIFT_SAMPLES = 100


def configuration(context):
    return {'experiment': OUT.name, 'date': '2026-09-25',
            'plan': descriptor(PLAN), 'cohort_sha256': context.cohort_sha256,
            'folds_sha256': context.folds_sha256, 'subject_ids': context.ids,
            'views': list(VIEWS), 'augmentation': {
                'name': 'time_window_start_jitter', 'number_augmented_views': 2,
                'window_count': 60, 'window_seconds': 5, 'filtered_seconds': N_SECONDS,
                'source_rate_hz': 250, 'resample_rate_hz': SAMPLE_RATE,
                'offset_samples_inclusive': [-MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES],
                'clamp_start_samples': [0, N_SECONDS * SAMPLE_RATE - WINDOW_SAMPLES],
                'hash_algorithm': 'SHA256',
                'hash_key': 'eeg-aug-jitter-window-v1|subject_id|view_name|window_index',
                'sampling_formula': 'int.from_bytes(digest[:8],little)%201-100',
                'channel_policy': 'same shift on every channel',
                'preprocessing': 'native -> MNE resample 200 -> 0.3..75 Hz -> notch 50 -> first 300 s',
                'augmentation_position': 'after full-continuous filtering, before frozen CBraMod',
                'clean_evaluation_only': True, 'no_label_fitting': True},
            'revision': descriptor(REVISION), 'extractor': descriptor(__file__),
            'original_exporter': descriptor(WORK / 'tridep_ccml/v2/extract_eeg.py'),
            'backbone': descriptor(WEIGHTS), 'baseline_manifest': descriptor(BASE_EEG / 'eeg_manifest.json')}


def window_shift(sid: str, view_name: str, window_index: int) -> int:
    require(view_name in VIEWS[1:] and 0 <= window_index < 60, 'Bad jitter request')
    key = f'eeg-aug-jitter-window-v1|{sid}|{view_name}|{window_index}'.encode()
    value = int.from_bytes(sha256(key).digest()[:8], 'little') % (2 * MAX_SHIFT_SAMPLES + 1)
    return value - MAX_SHIFT_SAMPLES


def starts_for_view(sid: str, view_name: str) -> list[int]:
    last = N_SECONDS * SAMPLE_RATE - WINDOW_SAMPLES
    return [min(last, max(0, i * WINDOW_SAMPLES + window_shift(sid, view_name, i)))
            for i in range(60)]


def preprocess(mat, indices, names):
    """Match v2-E extraction exactly until its 60-window reshape."""
    raw = mne.io.RawArray(mat[indices], mne.create_info(names, 250, 'eeg'), verbose='ERROR')
    raw.resample(SAMPLE_RATE, verbose='ERROR')
    raw.filter(.3, 75, verbose='ERROR')
    raw.notch_filter(50, verbose='ERROR')
    require(raw.n_times >= N_SECONDS * SAMPLE_RATE, 'EEG shorter than 300 seconds')
    x = raw.get_data(start=0, stop=N_SECONDS * SAMPLE_RATE)
    require(x.shape == (N_CHANNELS, 60000) and np.isfinite(x).all(), 'Invalid EEG after filtering')
    return x


def windows(signal, starts):
    require(len(starts) == 60 and all(0 <= s <= 59000 for s in starts), 'Bad window starts')
    result = np.stack([signal[:, s:s + WINDOW_SAMPLES].reshape(N_CHANNELS, 5, 200)
                       for s in starts])
    require(result.shape == (60, 19, 5, 200) and np.isfinite(result).all(), 'Bad EEG view')
    return result


def backbone(device):
    set_seed(11)
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                    seq_len=30, n_layer=12, nhead=8)
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    checkpoint = checkpoint.get('state_dict', checkpoint)
    checkpoint = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
    model.proj_out = torch.nn.Identity()
    model.load_state_dict({k: checkpoint[k] for k in model.state_dict()}, strict=True)
    return model.requires_grad_(False).to(device).eval()


def save_npy_new(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)


def prepare(subject=None):
    context = RecordContext()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = configuration(context)
    write_new(OUT / 'configuration.json', cfg)
    revision = read_json(REVISION)
    required = [e['required_source_label'] for e in revision['selection']]
    names = [e['target_name'] for e in revision['selection']]
    expected_indices = [e['expected_zero_based_index'] for e in revision['selection']]
    staged = {r['subject_id']: r for r in read_json(WORK / 'S00/manifests/staged_dataset.json')}
    subjects = {r['subject_id']: r for r in read_json(WORK / 'S00/manifests/subjects.json')}
    historical = {r['provenance']['subject_id']: r for r in read_json(BASE_EEG / 'eeg_manifest.json')}
    require(set(staged) == set(subjects) == set(historical) == set(context.ids), 'Cohort mismatch')
    ids = [subject] if subject else context.ids
    require(set(ids) <= set(context.ids), 'Unknown subject')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = backbone(device)
    for sid in ids:
        started = time.perf_counter()
        base = historical[sid]
        p = base['provenance']
        require(p['fit_subject_ids'] == [] and p['resolved_indices'] == expected_indices
                and p['selection_by_label'] == required and p['cohort_sha256'] == context.cohort_sha256,
                'Historical corrected EEG mismatch')
        old_path = verify_descriptor(base['feature'])
        old = np.load(old_path, allow_pickle=False)
        require(old.shape == (60, 200) and old.dtype == np.float32, 'Historical EEG shape mismatch')
        mat_path = WORK / 'S00' / staged[sid]['eeg_mat']
        require(digest(mat_path) == staged[sid]['mat_sha256'], 'Staged EEG source changed')
        set_path = WORK / subjects[sid]['eeg_set']
        set_data = loadmat(set_path, simplify_cells=True)
        set_data = set_data.get('EEG', set_data)
        labels = [str(ch['labels']) for ch in set_data['chanlocs']]
        indices = label_indices(labels, required)
        require(indices == expected_indices, 'Corrected electrode resolution mismatch')
        sidecar = OUT / 'features' / sid / 'views.json'
        if sidecar.exists():
            row = read_json(sidecar)
            require(row['configuration'] == descriptor(OUT / 'configuration.json')
                    and row['clean_source'] == descriptor(old_path), 'Stale augmentation sidecar')
            for item in VIEWS[1:]:
                verify_descriptor(row['features'][item])
            print(f'EEG augmented views reuse {sid}', flush=True)
            continue
        mat = loadmat(mat_path)['eeg_mat']
        signal = preprocess(mat, indices, names)
        clean_starts = [i * WINDOW_SAMPLES for i in range(60)]
        clean_reproduced = embed(model, windows(signal, clean_starts), device)
        max_abs = float(np.max(np.abs(clean_reproduced - old)))
        require(np.allclose(clean_reproduced, old, rtol=1e-4, atol=1e-4),
                f'Historical clean EEG not reproduced for {sid}: max_abs={max_abs}')
        features = {'clean': descriptor(old_path)}
        start_info = {}
        for view_name in VIEWS[1:]:
            view_starts = starts_for_view(sid, view_name)
            view = embed(model, windows(signal, view_starts), device)
            require(view.shape == (60, 200) and view.dtype == np.float32 and np.isfinite(view).all(),
                    'Invalid augmented embedding')
            dest = OUT / 'features' / sid / f'{view_name}.npy'
            if dest.exists():
                require(np.array_equal(np.load(dest, allow_pickle=False), view),
                        'Partial prior feature is incompatible')
            else:
                save_npy_new(dest, view)
            features[view_name] = descriptor(dest)
            start_info[view_name] = {'min': min(view_starts), 'max': max(view_starts),
                                     'shifted_windows': sum(a != b for a, b in zip(view_starts, clean_starts)),
                                     'starts_sha256': sha256(np.asarray(view_starts, dtype='<i4').tobytes()).hexdigest()}
        row = {'subject_id': sid, 'views': list(VIEWS), 'features': features,
               'clean_source': descriptor(old_path), 'configuration': descriptor(OUT / 'configuration.json'),
               'source_mat': descriptor(mat_path), 'source_set': descriptor(set_path),
               'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
               'fit_subject_ids': [], 'labels_used': False, 'backbone': descriptor(WEIGHTS),
               'resolved_labels': required, 'resolved_indices': indices,
               'clean_reproduction': {'max_abs': max_abs, 'rtol': 1e-4, 'atol': 1e-4},
               'window_starts': start_info,
               'augmentation_only_for_training': True}
        write_new(sidecar, row)
        print(f'EEG augmented views {sid} clean_max_abs={max_abs:.6g} seconds={time.perf_counter()-started:.1f}', flush=True)
    if all((OUT / 'features' / sid / 'views.json').exists() for sid in context.ids):
        rows = [read_json(OUT / 'features' / sid / 'views.json') for sid in context.ids]
        require(all(row['subject_id'] == sid for row, sid in zip(rows, context.ids)), 'Manifest ID mismatch')
        write_new(OUT / 'feature_manifest.json', rows)
        print('All 37 EEG augmentation views ready.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject')
    prepare(parser.parse_args().subject)
