"""Audited E2v2 inputs and frozen CBraMod exports for the registered stage 1.

No extraction runs at import. ``prepare_inputs`` creates one frozen backbone and
never reads predictions to select preprocessing, coordinates, or graph edges.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

WORK = Path(__file__).resolve().parents[2]
S00 = WORK / 'S00'
E2V2 = WORK / 'tridep_ccml/experiments/EEG-E2v2-37-fulltext'
STAGE0 = WORK / 'output/analysis/tridep_stage0_20261004'
CHANNEL_NAMES = ['FP2', 'FP1', 'F7', 'F3', 'F4', 'F8', 'FCz', 'C3', 'C4',
                 'T3', 'CPz', 'T4', 'P3', 'Pz', 'P4', 'T5', 'T6', 'O1', 'O2']
MONTAGE_NAMES = ['Fp2', 'Fp1', 'F7', 'F3', 'F4', 'F8', 'FCz', 'C3', 'C4',
                 'T7', 'CPz', 'T8', 'P3', 'Pz', 'P4', 'P7', 'P8', 'O1', 'O2']
RAW_SHAPE = (60, 19, 1000)
TOKEN_SHAPE = (60, 19, 5, 200)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SourceLedger:
    """Snapshot every original source before inference and check it afterwards."""
    def __init__(self):
        self.items = {}

    def add(self, path, role, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(WORK), 'Source outside workspace: ' + str(path))
        key = path.relative_to(WORK).as_posix()
        if key not in self.items:
            self.items[key] = {**descriptor(path), 'bytes': path.stat().st_size, 'roles': []}
        item = self.items[key]
        require(expected is None or expected == item['sha256'], 'Source hash differs: ' + key)
        if role not in item['roles']:
            item['roles'].append(role)
        return {'path': key, 'sha256': item['sha256']}

    def bind(self, value, role):
        return self.add(WORK / value['path'], role, value['sha256'])

    def rows(self):
        return [self.items[k] for k in sorted(self.items)]

    def verify(self):
        for item in self.rows():
            require(digest(WORK / item['path']) == item['sha256'],
                    'Source changed during extraction: ' + item['path'])
        return len(self.items)


def _validate_partitions(folds, ids, labels):
    require(len(ids) == len(set(ids)) == 37 and set(labels) == set(ids), 'Wrong 37-person cohort')
    require(Counter(labels.values()) == Counter({0: 21, 1: 16}), 'Wrong diagnosis counts')
    require(set(folds) == {f'fold_{i}' for i in range(1, 6)}, 'Wrong outer folds')
    test_ids = []
    for name, fold in folds.items():
        require(set(fold) == {'train', 'val', 'test'}, 'Wrong split names: ' + name)
        flat = sum((fold[p] for p in ('train', 'val', 'test')), [])
        require(len(flat) == len(set(flat)) == 37 and set(flat) == set(ids), 'Person overlap: ' + name)
        require(len(fold['train']) in (26, 27) and len(fold['val']) == 3, 'Split size changed')
        for part in fold.values():
            require({labels[s] for s in part} == {0, 1}, 'Split lacks a diagnosis class')
        test_ids.extend(fold['test'])
    require(Counter(test_ids) == Counter({s: 1 for s in ids}), 'Wrong test-person coverage')


def validate_raw(array):
    require(array.shape == RAW_SHAPE and array.dtype == np.float32,
            'Expected float32 E2v2 raw [60,19,1000]')
    require(np.isfinite(array).all(), 'Nonfinite EEG input')
    return array.reshape(TOKEN_SHAPE)


def validate_inputs():
    """Read and hash original sources; return context without creating files/GPU work."""
    from scipy.io import loadmat
    ledger = SourceLedger()
    ledger.add(__file__, 'stage1_input_exporter')
    lock = read(S00 / 'configs/cohort.json')
    for name in ('cohort', 'b0_fulltext', 'protocol_fulltext'):
        ledger.add(S00 / f'configs/{name}.json', 'S00_configuration')
    for name in ('subjects', 'staged_dataset', 'modma_raw_files', 'modma_labels', 'source_sha256'):
        ledger.add(S00 / f'manifests/{name}.json', 'S00_source_manifest')
    ledger.add(S00 / lock['fold_file'], 'active_folds', lock['folds_sha256'])
    ledger.add(S00 / lock['official_fold_file'], 'original_38_folds', lock['official_folds_sha256'])
    folds = read(S00 / lock['fold_file'])
    subjects = read(S00 / 'manifests/subjects.json')
    require(len(subjects) == 37, 'Subject manifest cardinality changed')
    labels = {s['subject_id']: s['label'] for s in subjects}
    ids = lock['subject_ids']
    _validate_partitions(folds, ids, labels)
    original = read(S00 / lock['official_fold_file'])
    require(folds == {n: {p: [s for s in ss if s in ids] for p, ss in f.items()}
                      for n, f in original.items()}, 'Active assignments differ from original removal policy')
    for table in read(S00 / 'manifests/modma_labels.json').values():
        require(all(table['labels'][s]['label'] == labels[s] for s in ids), 'Diagnosis metadata differs')
    stage0_status = read(STAGE0 / 'status.json')
    require(stage0_status['passed'] and stage0_status['status'] == 'complete' and
            stage0_status['phase'] == 0 and stage0_status['folds_sha256'] == lock['folds_sha256'],
            'Stage 0 must be complete for these folds')
    ledger.add(STAGE0 / 'status.json', 'fixed_stage0_status')
    ledger.add(STAGE0 / 'source_manifest.json', 'fixed_stage0_sources', stage0_status['source_manifest_sha256'])
    ledger.add(STAGE0 / 'output_manifest.json', 'fixed_stage0_outputs', stage0_status['output_manifest_sha256'])
    # Retain the earlier audit's identities, rather than accepting newly edited
    # S00 source metadata merely because it is internally consistent today.
    prior_sources = {r['path']: r for r in read(STAGE0 / 'source_manifest.json')}
    for item in ledger.rows():
        if item['path'] in prior_sources:
            ledger.add(WORK / item['path'], 'stage0_pre_existing_source', prior_sources[item['path']]['sha256'])
    stage0_outputs = {r['path']: r for r in read(STAGE0 / 'output_manifest.json')}
    for name in ('branch_probabilities.jsonl', 'control_predictions.jsonl', 'checkpoint_provenance.json',
                 'feature_bundles.json', 'metrics.json', 'validation_weight_selections.json'):
        path = STAGE0 / name
        ledger.add(path, 'fixed_stage0_binding', stage0_outputs[path.relative_to(WORK).as_posix()]['sha256'])
    manifest_path = E2V2 / 'input_manifest.json'
    manifest = read(manifest_path)
    gate = read(E2V2 / 'real_gate.json')
    require(gate['passed'] and gate['subjects'] == 37, 'E2v2 real-input gate failed')
    ledger.add(E2V2 / 'real_gate.json', 'accepted_E2v2_gate')
    ledger.add(manifest_path, 'accepted_E2v2_inputs', gate['new_input_manifest']['sha256'])
    ledger.bind(manifest['revision'], 'verified_HydroCel_mapping')
    ledger.bind(manifest['engineering_gate'], 'E2v2_synthetic_gate')
    synthetic = read(WORK / manifest['engineering_gate']['path'])
    require(synthetic['passed'] and synthetic['accepted_method'] == manifest['method'], 'Wrong accepted boundary method')
    require(manifest['method'] == 'harmonic_guarded' and not manifest['labels_used'] and not manifest['augmentation'],
            'Changed or label-dependent input preprocessing')
    registration_path = E2V2 / 'engineering_registration.json'
    registration = read(registration_path)
    ledger.add(registration_path, 'E2v2_pre_existing_registration')
    for path, expected in registration['protocol']['sources'].items():
        ledger.add(WORK / path, 'E2v2_registered_code_and_configuration', expected)
    for name in ('real_inputs.py', 'preprocessing.py', 'common.py', 'probe.py'):
        ledger.add(WORK / 'tridep_ccml/eeg_e2v2' / name, 'E2v2_input_producer')
    real_registration_path = E2V2 / 'real_registration.json'
    real_registration = read(real_registration_path)
    ledger.add(real_registration_path, 'E2v2_real_input_registration')
    for key in ('source', 'appendix', 'synthetic_gate'):
        ledger.bind(real_registration[key], 'E2v2_real_input_registered_source')
    revision = read(WORK / manifest['revision']['path'])
    require(revision['do_not_use_raw_GSN129_coordinates_for_spatial_operations'], 'Unsafe raw coordinate policy')
    names = [r['target_name'] for r in revision['selection']]
    native_names = [r['required_source_label'] for r in revision['selection']]
    indices = [r['expected_zero_based_index'] for r in revision['selection']]
    require(names == CHANNEL_NAMES, 'Target channel order changed')
    records = manifest['records']
    require(len(records) == 37 and [r['subject_id'] for r in records] == ids, 'E2v2 subject order differs')
    staged = {r['subject_id']: r for r in read(S00 / 'manifests/staged_dataset.json')}
    for row in records:
        sid = row['subject_id']
        require(row['indices'] == indices and row['required_labels'] == native_names and
                row['method'] == manifest['method'] and row['shape'] == list(RAW_SHAPE) and
                row['recorded_samples_unchanged'] and row['removed_windows'] == 0 and
                not row['labels_used'] and not row['augmentation'], 'E2v2 record mismatch: ' + sid)
        require(row['source']['subject_id'] == sid and
                row['source']['mat']['sha256'] == staged[sid]['mat_sha256'], 'Wrong raw ancestry')
        for key in ('set', 'fdt', 'mat', 'cache'):
            ledger.bind(row['source'][key], 'E2v2_original_' + key)
        ledger.bind(row['new_cache'], 'selected_E2v2_raw')
        ledger.add(E2V2 / 'subjects' / f'{sid}.json', 'E2v2_per_subject_provenance')
        require(read(E2V2 / 'subjects' / f'{sid}.json') == row, 'E2v2 sidecar differs from input manifest')
        obj = loadmat(WORK / row['source']['set']['path'], simplify_cells=True)
        obj = obj.get('EEG', obj)
        actual_labels = [str(c['labels']) for c in obj['chanlocs']]
        require(actual_labels == [f'E{i}' for i in range(1, 129)] + ['Cz'], 'Unexpected raw channel order')
        require([actual_labels.index(n) for n in native_names] == indices, 'Source electrode mapping mismatch')
        validate_raw(np.load(WORK / row['new_cache']['path'], allow_pickle=False))
    weight_path = S00 / 'upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth'
    source_hashes = read(S00 / 'manifests/source_sha256.json')
    weight = ledger.add(weight_path, 'original_frozen_Mumtaz_CBraMod',
                        source_hashes[str(weight_path.relative_to(WORK))]['sha256'])
    for path in (S00 / 'upstream-dev/lib/models/cbramod').glob('*.py'):
        expected = prior_sources[path.relative_to(WORK).as_posix()]['sha256']
        ledger.add(path, 'CBraMod_backbone_code', expected)
    return {'ledger': ledger, 'records': records, 'subject_ids': ids, 'labels': labels,
            'folds': folds, 'cohort': lock, 'channel_names': names, 'native_channel_names': native_names,
            'weight': weight, 'input_manifest': descriptor(manifest_path),
            'unit_policy': 'native_numeric_as_EEGLAB_microvolt_reader_convention',
            'physical_calibration_confirmed': False}


def build_adjacency(coordinates):
    """Three nearest neighbors, undirected union, self loops, row normalization."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    require(coordinates.shape == (19, 3) and np.isfinite(coordinates).all(), 'Wrong standard coordinates')
    distances = np.sum((coordinates[:, None] - coordinates[None, :]) ** 2, axis=-1)
    np.fill_diagonal(distances, np.inf)
    edges = np.zeros((19, 19), dtype=bool)
    for i in range(19):
        # Stable sorting fixes exact-distance ties by registered channel order.
        edges[i, np.argsort(distances[i], kind='stable')[:3]] = True
    edges |= edges.T
    np.fill_diagonal(edges, True)
    return (edges.astype(np.float64) / edges.sum(1, keepdims=True)).astype(np.float32)


def standard_graph(ledger):
    import mne
    montage = mne.channels.make_standard_montage('standard_1020')
    positions = montage.get_positions()['ch_pos']
    xyz = np.stack([positions[n] for n in MONTAGE_NAMES])
    data_path = Path(mne.__file__).resolve().parent / 'channels/data/montages/standard_1020.elc'
    coordinate_asset = ledger.add(data_path, 'MNE_standard_1020_coordinate_asset')
    for filename in ('montage.py', '_standard_montage_utils.py'):
        path = Path(mne.__file__).resolve().parent / 'channels' / filename
        if path.exists():
            ledger.add(path, 'MNE_coordinate_loader_code')
    info = {'montage': 'MNE standard_1020', 'mne_version': mne.__version__, 'coordinate_asset': coordinate_asset,
            'channel_names': CHANNEL_NAMES, 'montage_names': MONTAGE_NAMES, 'coordinates_meters': xyz.tolist(),
            'coordinate_frame': montage.get_positions()['coord_frame'], 'k': 3,
            'edges': 'euclidean_3_nearest_union_transpose_plus_self_loop', 'normalization': 'row_sum',
            'distance_ties': 'stable_registered_channel_order', 'raw_GSN129_coordinates_used': False,
            'electrode_equivalence': 'approximate_template_mapping_not_individual_digitization', 'labels_used': False}
    return build_adjacency(xyz), info


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, 'Refusing changed existing export: ' + str(path))
        return
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def _write_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(np.array_equal(np.load(path, allow_pickle=False), array), 'Existing export differs: ' + str(path))
        return
    tmp = path.with_suffix(path.suffix + '.part')
    with tmp.open('wb') as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def _state_digest(model):
    state_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        state_hash.update(name.encode('utf-8'))
        state_hash.update(str(array.dtype).encode('ascii'))
        state_hash.update(str(array.shape).encode('ascii'))
        state_hash.update(array.tobytes())
    return state_hash.hexdigest()


def prepare_inputs(export_root: Path, *, device='cuda'):
    """Export E2v2 CBraMod tokens/means once; root invokes this sequentially."""
    import torch
    require(str(device).startswith('cuda') and torch.cuda.is_available(), 'Registered extraction requires CUDA')
    require(torch.are_deterministic_algorithms_enabled() and
            not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32,
            'Root must set registered deterministic float32/TF32 runtime before extraction')
    runtime = {'torch_version': torch.__version__, 'numpy_version': np.__version__,
               'device': str(device), 'cuda_version': torch.version.cuda,
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
               'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
    context = validate_inputs()
    ledger = context['ledger']
    adjacency, coordinate_info = standard_graph(ledger)
    # Every original input, producer, weight, and coordinate asset is now hashed.
    initial_sources = ledger.rows()
    export_root = Path(export_root).resolve()
    require(export_root.is_relative_to(WORK), 'Export root outside workspace')
    adjacency_path = export_root / 'cache/adjacency.npy'
    coordinate_path = export_root / 'cache/coordinate_source.json'
    source_manifest_path = export_root / 'cache/input_source_manifest.json'
    _write_array(adjacency_path, adjacency)
    _write_json(coordinate_path, coordinate_info)
    _write_json(source_manifest_path, initial_sources)
    sys.path.insert(0, str(S00 / 'upstream-dev'))
    from lib.models.cbramod.cbramod import CBraMod
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                    seq_len=30, n_layer=12, nhead=8)
    saved = torch.load(WORK / context['weight']['path'], map_location='cpu', weights_only=True)
    saved = {k.replace('backbone.', ''): v for k, v in saved.get('state_dict', saved).items()}
    model.proj_out = torch.nn.Identity()
    own = model.state_dict()
    require(all(k in saved and saved[k].shape == v.shape for k, v in own.items()), 'Missing pretrained CBraMod state')
    model.load_state_dict({k: saved[k] for k in own}, strict=True)
    model.requires_grad_(False).to(device=device, dtype=torch.float32).eval()
    require(not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters()), 'Backbone not frozen')
    before = _state_digest(model)
    rows = []
    repeated_checks = 0
    for source in context['records']:
        sid = source['subject_id']
        patches = validate_raw(np.load(WORK / source['new_cache']['path'], allow_pickle=False))
        tokens, means = [], []
        with torch.inference_mode():
            for start in range(0, 60, 4):
                tensor = torch.from_numpy(patches[start:start + 4]).to(device=device, dtype=torch.float32)
                y = model(tensor)
                require(y.shape == (len(tensor), 19, 5, 200) and bool(torch.isfinite(y).all()), 'Invalid CBraMod tokens')
                # Keep the original GPU reduction and original four-window batches.
                means.append(y.mean((1, 2)).cpu().numpy())
                tokens.append(y.cpu().numpy())
                if start == 0:
                    require(torch.equal(model(tensor), y), 'Repeated frozen input changed tokens')
                    repeated_checks += 1
        tokens = np.concatenate(tokens).astype(np.float32, copy=False)
        means = np.concatenate(means).astype(np.float32, copy=False)
        require(tokens.shape == TOKEN_SHAPE and means.shape == (60, 200), 'Invalid EEG export shape')
        cpu_mean = tokens.mean((1, 2))
        require(np.allclose(cpu_mean, means, rtol=1e-4, atol=1e-4), 'Token/mean interface mismatch')
        token_path = export_root / 'cache/cbramod' / f'{sid}.npy'
        mean_path = export_root / 'cache/cbramod_mean' / f'{sid}.npy'
        sidecar_path = token_path.with_suffix('.json')
        _write_array(token_path, tokens)
        _write_array(mean_path, means)
        row = {'subject_id': sid, 'label': context['labels'][sid], 'raw': source['new_cache'],
               'tokens': descriptor(token_path), 'mean': descriptor(mean_path)}
        sidecar = {**row, 'source_manifest': context['input_manifest'], 'source_record': source,
                   'channel_names': CHANNEL_NAMES, 'native_channel_names': context['native_channel_names'],
                   'cohort_id': context['cohort']['cohort_id'], 'folds_sha256': context['cohort']['folds_sha256'],
                   'backbone': context['weight'], 'exporter': descriptor(__file__),
                   'token_shape': list(TOKEN_SHAPE), 'mean_shape': [60, 200], 'dtype': 'float32',
                   'batch_windows': 4, 'mean_operation': 'same_forward_GPU_y.mean((1,2))',
                   'cpu_mean_rtol': 1e-4, 'cpu_mean_atol': 1e-4,
                   'cpu_mean_max_abs_error': float(np.max(np.abs(cpu_mean - means))),
                   'frozen_eval': True, 'fit_subject_ids': [], 'normalization_added': False,
                   'runtime': runtime,
                   'R1_token_cache_used': False, 'unit_policy': context['unit_policy'],
                   'physical_calibration_confirmed': False}
        _write_json(sidecar_path, sidecar)
        row['sidecar'] = descriptor(sidecar_path)
        rows.append(row)
        print(f'CBraMod E2v2 export {len(rows)}/37: {sid}', flush=True)
    after = _state_digest(model)
    require(before == after and not any(p.requires_grad for p in model.parameters()) and
            not any(m.training for m in model.modules()), 'Frozen parameters/buffers changed')
    require(len(rows) == repeated_checks == 37, 'Incomplete CBraMod exports')
    source_count = ledger.verify()
    require(ledger.rows() == initial_sources, 'Original source set changed after extraction began')
    result = {'records': rows, 'subject_ids': context['subject_ids'], 'labels': context['labels'],
              'folds': context['folds'], 'channel_names': CHANNEL_NAMES, 'montage_names': MONTAGE_NAMES,
              'adjacency': descriptor(adjacency_path), 'coordinate_source': descriptor(coordinate_path),
              'sources': ledger.rows(), 'input_manifest': context['input_manifest'],
              'input_source_manifest': descriptor(source_manifest_path), 'runtime': runtime,
              'physical_calibration_confirmed': False, 'unit_policy': context['unit_policy'],
              'backbone_state_before_sha256': before, 'backbone_state_after_sha256': after,
              'backbone_loaded_once': True, 'frozen_state_unchanged': True,
              'repeated_input_checks': repeated_checks, 'original_sources_unchanged': source_count}
    _write_json(export_root / 'cache/input_exports.json', result)
    del model
    torch.cuda.empty_cache()
    return result
