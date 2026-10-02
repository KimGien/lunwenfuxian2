"""Read-only CCML-v2 input audit; writes only a new audit output directory."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import mne
import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tridep_ccml/reports/ccml_v2_audit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all()
    return {'shape': list(x.shape), 'mean': float(x.mean()), 'std': float(x.std()),
            'rms': float(np.sqrt(np.mean(x*x))), 'abs_quantiles_50_95_99_max':
            np.quantile(np.abs(x), [.5, .95, .99, 1]).tolist()}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / 'audit.json'
    if result_path.exists():
        raise FileExistsError('Preserve completed audit; use its recorded results.')
    inputs = [ROOT / p for p in [
        'S00/configs/b0.json', 'S00/configs/protocol_fulltext.json',
        'S00/manifests/subjects.json', 'S00/manifests/staged_dataset.json',
        'S00/scripts/extract_b0.py',
        'S00/upstream-dev/lib/models/encoders/cnn_bigru.py',
        'S00/upstream-dev/lib/feature_extraction/eeg/extract_cbramod.py',
        'S00/upstream-dev/README.md',
        'S00/upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth',
        'tridep_ccml/train.py', 'tridep_ccml/ccml.py', 'tridep_ccml/records.py',
        'tridep_ccml/reports/experiment_status.json',
        'tridep_ccml/reports/tc01_tc04_analysis/diagnostics.json',
        'output/plans/CCML_v2_性能优化与后续计划.md']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    cfg = read(ROOT / 'S00/configs/b0.json')['preprocessing']['eeg']
    indices = cfg['python_indices']
    staged = {s['subject_id']: s for s in read(ROOT / 'S00/manifests/staged_dataset.json')}
    subjects = read(ROOT / 'S00/manifests/subjects.json')
    assert len(subjects) == 37
    eeg = []
    for s in subjects:
        sid = s['subject_id']
        set_path = ROOT / s['eeg_set']
        d = loadmat(set_path, simplify_cells=True)
        d = d.get('EEG', d)
        labels = [str(ch['labels']) for ch in d['chanlocs']]
        assert labels[:128] == [f'E{i}' for i in range(1, 129)]
        native = np.memmap(ROOT / s['eeg_fdt'], dtype='<f4', mode='r',
                           shape=(int(d['pnts']), int(d['nbchan']))).T
        mat_path = ROOT / 'S00' / staged[sid]['eeg_mat']
        assert digest(mat_path) == staged[sid]['mat_sha256']
        mat = loadmat(mat_path)['eeg_mat']
        assert np.array_equal(mat, native)
        raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose='ERROR')
        volts = raw.get_data(start=0, stop=1000)
        assert np.allclose(volts, native[:, :1000] * 1e-6, rtol=1e-6, atol=1e-12)
        mapping = []
        for target, index in zip(cfg['channels'], indices):
            ch = d['chanlocs'][index]
            alt = d['chanlocs'][index-1]
            mapping.append({'target_label': target, 'python_index': index,
                            'actual_label': labels[index],
                            'actual_xyz': [float(ch[k]) for k in ['X', 'Y', 'Z']],
                            'minus_one_candidate_label': labels[index-1],
                            'minus_one_candidate_xyz': [float(alt[k]) for k in ['X', 'Y', 'Z']],
                            'candidate_is_approved_correction': False})
        chaninfo = d.get('chaninfo', {})
        eeg.append({'subject_id': sid, 'set_sha256': digest(set_path),
                    'staged_sha256': staged[sid]['mat_sha256'],
                    'native_equals_staged': True, 'reader_microvolt_to_volt_check': True,
                    'nbchan': int(d['nbchan']), 'sampling_rate': float(d['srate']),
                    'reference_metadata': str(d['ref']),
                    'coordinate_file': str(chaninfo.get('filename', '')),
                    'current_selected_native': stats(native[indices]), 'mapping': mapping})
        del mat, native, raw, volts
        print(f'EEG audited {sid}', flush=True)

    encoder_path = ROOT / 'S00/upstream-dev/lib/models/encoders/cnn_bigru.py'
    spec = importlib.util.spec_from_file_location('audit_cnn_bigru', encoder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    speech = []
    cache_root = ROOT / 'S00/features/B0-37'
    sample = next(p for p in sorted((cache_root/'hubert_segments').glob('*/*.npy'))
                  if np.load(p, mmap_mode='r').shape[0] > 1)
    vectors = np.load(sample)
    for seed in [11, 22, 33]:
        p = cache_root / f'encoders/seed_{seed}.pt'
        checkpoint = torch.load(p, map_location='cpu', weights_only=True)
        torch.manual_seed(seed)
        model = module.BiGRUAttentionEncoder(input_dim=768).eval()
        assert checkpoint['fit_subject_ids'] == []
        assert all(torch.equal(v, checkpoint['state_dict'][k]) for k, v in model.state_dict().items())
        observed = []
        hook = model.attention.register_forward_hook(lambda mod, args, out: observed.append(out.detach()))
        with torch.no_grad():
            x = torch.from_numpy(vectors)
            z = model(x)
            reverse = model(x.flip(0)).mean(0)
        hook.remove()
        weights = observed[0].softmax(dim=1)
        assert observed[0].shape[1] == 1 and torch.equal(weights, torch.ones_like(weights))
        saved = np.load(cache_root / f'speech/seed_{seed}/{sample.parent.name}.npy')[int(sample.stem)-1]
        error = float(np.max(np.abs(z.mean(0).numpy()-saved)))
        assert np.allclose(z.mean(0).numpy(), saved, rtol=1e-4, atol=1e-5)
        speech.append({'seed': seed, 'checkpoint_sha256': digest(p), 'matches_fresh_random_initialization': True,
                       'fit_subject_ids': [], 'sample_cache': str(sample.relative_to(ROOT)),
                       'sample_shape': list(vectors.shape), 'attention_shape': list(weights.shape),
                       'attention_all_one': True, 'saved_question_reproduction_max_abs_error': error,
                       'reversed_segment_mean_max_abs_error': float((z.mean(0)-reverse).abs().max())})
    caches = sorted((cache_root / 'hubert_segments').glob('*/*.npy'))
    assert len(caches) == 37*29
    zero_caches = []
    for p in caches:
        x = np.load(p)
        assert x.ndim == 2 and x.shape[1] == 768 and np.isfinite(x).all()
        assert digest(p) == read(p.with_suffix('.json'))['sha256']
        if not len(x):
            zero_caches.append(str(p.relative_to(ROOT)))

    features = []
    for p in sorted((ROOT/'tridep_ccml/features').glob('seed_*/fold_*.npz')):
        with np.load(p, allow_pickle=False) as data:
            x, splits = data['features'], data['splits']
            assert x.shape == (37, 3, 128) and np.isfinite(x).all()
            row = {'path': str(p.relative_to(ROOT)), 'sha256': digest(p), 'splits': {}}
            for split in np.unique(splits):
                row['splits'][str(split)] = {}
                for v, name in enumerate(['eeg', 'speech', 'text']):
                    a = x[splits == split, v]
                    normalized = torch.nn.functional.layer_norm(torch.from_numpy(a), (128,), eps=1e-5).numpy()
                    row['splits'][str(split)][name] = {'raw': stats(a), 'per_subject_l2_mean': float(np.linalg.norm(a, axis=1).mean()),
                                                     'per_subject_std_min': float(a.std(axis=1).min()), 'layernorm_diagnostic_only': stats(normalized)}
            features.append(row)
    assert len(features) == 15
    checkpoint = torch.load(inputs[8], map_location='cpu', weights_only=True)
    diagnostics = read(ROOT/'tridep_ccml/reports/tc01_tc04_analysis/diagnostics.json')
    result = {'audit_id': 'CCML-v2-Audit', 'date': '2026-09-21', 'training_runs': 0,
              'scope': 'read-only audit; normalization is a numerical diagnostic, not fitted or scored',
              'input_hashes': hashes, 'script_sha256': digest(Path(__file__)), 'independent_subjects': 37,
              'eeg': eeg, 'speech_encoder_checks': speech, 'hubert_cache_count': len(caches),
              'zero_segment_caches': zero_caches, 'feature_distributions': features,
              'checkpoint_top_level_count': len(checkpoint),
              'checkpoint_non_tensor_top_level_keys': [k for k,v in checkpoint.items() if not isinstance(v,torch.Tensor)],
              'historical_tc04_evidence': diagnostics['evidence']['TC-04'],
              'limitations': ['EEGLAB reader conversion is not acquisition calibration.',
                              'Mumtaz checkpoint expected numerical input scale remains unestablished.',
                              'Raw coordinate metadata cannot by itself establish physical acquisition montage.',
                              'Minus-one candidates must not be treated as verified anatomical corrections.']}
    assert all(digest(ROOT/p) == h for p,h in hashes.items())
    result['tracked_existing_inputs_unchanged'] = True
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'eeg_subjects': len(eeg), 'encoder_seeds': len(speech),
                      'hubert_caches': len(caches), 'feature_folds': len(features), 'existing_inputs_unchanged': True}))


if __name__ == '__main__':
    main()
