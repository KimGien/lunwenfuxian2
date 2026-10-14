"""Independent matched4s EEGPT provenance, math, fit and output verification.

Importing this file does not run a model. ``verify`` uses saved evidence and
NumPy/sklearn arithmetic only. ``replay`` is a separate, preregistered entry
available after all30 fits/evaluation; it makes1110 frozen person forwards.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import pickle
import pickletools
import types
import zipfile
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, recall_score, roc_auc_score

WORK = Path(__file__).resolve().parents[2]
OUT = WORK / 'tridep_ccml/experiments/TriDep-Stage4-EEGPT-Matched4s-37-fulltext-20261006'
REPORT = WORK / 'output/analysis/tridep_eegpt_20261006'
STAGE0 = WORK / 'output/analysis/tridep_stage0_20261004'
CONDITIONS = ('LaBraM-native4s-originalCNN', 'EEGPT-mean-projection4s-originalCNN')
SEEDS = (11, 22, 33)
VIEWS = {'eeg': 'p_eeg', 'fixed_EST': 'p_fixed_EST'}
KEYS = ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')
TRAINING = {'epochs': 60, 'batch_size': 16, 'optimizer': 'Adamax', 'learning_rate': .005,
            'weight_decay': 0., 'gradient_clip': None, 'parameters': 498434,
            'updates_per_model': 120, 'checkpoint_selection': 'fixed_final_epoch'}
FEATURE_BUDGET = {model + '_' + key: value for model in ('eegpt', 'labram')
                  for key, value in (('first_windows', 2220), ('repeat_windows', 296), ('total_windows', 2516), ('batches', 333))}
AUTHOR_MD5 = 'a165305f329f51b57691c8a3131f9b79'
AUTHOR_SHA = '9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0'
AUTHOR_BYTES = 1020602524
AUTHOR_SOURCE_SHA = 'cfe3d786e6c351e9678f9f7183cd17dab0dfbb131590ef88e3e4a9efae728a95'
PROJECTION_SEED = 20261005
CHANNELS = ('FP2', 'FP1', 'F7', 'F3', 'F4', 'F8', 'FCZ', 'C3', 'C4', 'T7', 'CPZ', 'T8', 'P3', 'PZ', 'P4', 'P7', 'P8', 'O1', 'O2')
CHANNEL_IDS = (2, 0, 7, 9, 13, 15, 20, 27, 31, 25, 38, 33, 45, 47, 49, 43, 51, 59, 61)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file(), 'EEGPT verification evidence missing: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def desc(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.is_file(), 'Missing/nonworkspace EEGPT evidence')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest(path)}


def checked(source):
    path = (WORK / source['path']).resolve()
    require(path.is_relative_to(WORK) and path.is_file() and digest(path) == source['sha256'], 'EEGPT evidence SHA differs: ' + source['path'])
    return path


def save(path, value, *, mutable=False):
    path = Path(path)
    require(path.is_relative_to(OUT) or path.is_relative_to(REPORT), 'Independent EEGPT writes must stay in the new tree')
    if path.exists() and not mutable:
        require(read(path) == value, 'Independent EEGPT evidence differs; previous result preserved')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def nested_descriptors(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield {'path': value['path'], 'sha256': value['sha256']}
        for item in value.values():
            yield from nested_descriptors(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_descriptors(item)


def equal(actual, expected, name, tolerance=1e-12):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + ': dictionary schema differs')
        for key in expected:
            equal(actual[key], expected[key], name + '/' + str(key), tolerance)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), name + ': sequence differs')
        for i, value in enumerate(expected):
            equal(actual[i], value, name + '/' + str(i), tolerance)
    elif expected is None or isinstance(expected, (bool, str, int)):
        require(type(actual) is type(expected) and actual == expected, name + ': exact value differs')
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) and
                abs(actual - expected) <= tolerance, name + ': numeric value differs')


def input_math(raw):
    require(raw.shape == (60, 19, 1000) and raw.dtype == np.float32 and np.isfinite(raw).all(), 'Expected corrected E2v2 raw[60,19,1000]')
    eegpt = (resample_poly(raw.astype(np.float64), 32, 25, axis=-1, window=('kaiser', 5.0), padtype='constant')[..., :1024] / 1000.).astype(np.float32)
    labram = (raw[..., :800] / 100.).astype(np.float32)
    return eegpt, labram


def fixed_projection():
    generator = np.random.Generator(np.random.PCG64(PROJECTION_SEED))
    normal = generator.standard_normal((512, 200), dtype=np.float64)
    q, r = np.linalg.qr(normal, mode='reduced')
    q *= np.where(np.diag(r) < 0., -1., 1.)[None, :]
    return np.ascontiguousarray(q, dtype=np.float32), np.ascontiguousarray(q, dtype=np.float64)


def pool_and_project(tokens, projection):
    require(tokens.dtype == np.float32 and tokens.shape == (60, 16, 4, 512) and np.isfinite(tokens).all(), 'Author summary token endpoint differs')
    native, projected = [], []
    for start in range(0, 60, 8):
        block = tokens[start:start + 8].mean(axis=(1, 2), dtype=np.float32)
        native.append(block)
        projected.append(np.matmul(block.astype(np.float64), projection.astype(np.float64)).astype(np.float32))
    return np.concatenate(native), np.concatenate(projected)


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        require(bool(value.isfinite().all()), 'Nonfinite head/encoder checkpoint tensor')
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def expected_initial_state(seed):
    """Independent tensor-only reconstruction of original PyTorch CNN init."""
    import torch
    generator = torch.Generator(device='cpu').manual_seed(seed)
    state = {}
    gain = math.sqrt(2. / (1. + math.sqrt(5.) ** 2))
    for prefix, shape in (('conv.0', (256, 200, 3)), ('conv.3', (128, 256, 3)),
                          ('classifier.2', (128, 1920)), ('classifier.5', (2, 128))):
        fan = math.prod(shape[1:]); bound = math.sqrt(3.) * gain / math.sqrt(fan)
        state[prefix + '.weight'] = torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator)
        state[prefix + '.bias'] = torch.empty((shape[0],), dtype=torch.float32).uniform_(-1 / math.sqrt(fan), 1 / math.sqrt(fan), generator=generator)
    require(sum(value.numel() for value in state.values()) == 498434, 'Matched CNN parameter count differs')
    return state


def metric(group, key):
    labels = np.asarray([r['label'] for r in group], dtype=np.int64)
    p = np.asarray([r[key] for r in group], dtype=np.float64)
    require(set(labels.tolist()) == {0, 1} and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), 'Invalid independent metric inputs')
    predictions = (p > .5).astype(np.int64)
    true_class = np.where(labels == 1, p, 1 - p).clip(1e-15, 1 - 1e-15)
    return {'macro_f1': float(f1_score(labels, predictions, average='macro', labels=[0, 1], zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'mdd_recall': float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        'auc': float(roc_auc_score(labels, p)), 'nll': float(-np.log(true_class).mean()),
        'brier': float(brier_score_loss(labels, p)), 'n_subjects': len(group),
        'confusion_hc_mdd': confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


def gate(candidate, control, candidate_seeds, control_seeds):
    delta = {key: candidate[key] - control[key] for key in KEYS}
    improving = sum(candidate_seeds[str(seed)]['test']['macro_f1'] > control_seeds[str(seed)]['test']['macro_f1'] for seed in SEEDS)
    checks = {'F1_delta_at_least_0_01': delta['macro_f1'] >= .01, 'at_least2_improving_seeds': improving >= 2,
              'BA_not_lower': delta['balanced_accuracy'] >= 0., 'MDD_recall_drop_at_most0_05': delta['mdd_recall'] >= -.05}
    return {'passed': all(checks.values()), 'checks': checks, 'improving_seeds': improving, 'delta': delta,
            'probability_improvement': delta['nll'] < 0 and delta['brier'] < 0}


def audit_registration():
    reg = read(OUT / 'registration.json')
    require(read(OUT / 'registration_anchor.json')['registration'] == desc(OUT / 'registration.json'), 'Registration anchor differs')
    require(reg['stage'] == 4 and reg['followup'] == 'previously_deferred_EEGPT_matched4s' and reg['conditions'] == list(CONDITIONS) and
            reg['seeds'] == list(SEEDS) and reg['training'] == {c: TRAINING for c in CONDITIONS} and
            reg['formal_fits'] == 30 and reg['formal_updates'] == reg['training_gradient_passes'] == 3600 and
            reg['budget']['features'] == FEATURE_BUDGET and reg['model_execution_allowed'] is True and
            reg['new_independent_people'] == 0 and reg['stage6_disclosed_14_people_used_for_fitting_or_selection'] is False and
            reg['physical_calibration_confirmed'] is False, 'Registered matched4s design or development-role budget differs')
    for key in ('inner_fits', 'calibrator_fits', 'threshold_searches', 'fusion_selections'):
        require(reg[key] == 0, 'Unregistered additional fitting/selection')
    require(reg['early_stopping'] is False and reg['head_selection'] == 'fixed_final_epoch60', 'Final-epoch head selection differs')
    for key, value in {'formal_fits': 30, 'formal_updates': 3600, 'training_gradient_passes': 3600,
          'producer_head_person_forward_calls': 2220, 'independent_replay_head_person_forward_calls': 1110,
          'total_head_person_forward_calls': 3330, 'prediction_records': 1110, 'metric_groups': 180,
          'new_S_T_forwards': 0, 'new_S_T_fits': 0, 'new_EEG_pretraining_or_finetuning': 0,
          'new_ASR_calls': 0, 'new_fusion_selection_calls': 0, 'new_calibration_or_threshold_searches': 0}.items():
        require(reg['budget'][key] == value, 'Registered work budget differs: ' + key)
    require(reg['sources'].get(desc(__file__)['path']) == digest(__file__), 'Independent verifier not source-frozen')
    for name, value in reg['sources'].items():
        checked({'path': name, 'sha256': value})
    old_out = WORK / 'tridep_ccml/experiments/TriDep-Stage6-ST-Generalization-Resume-20261006'
    old_report = WORK / 'output/analysis/tridep_stage6_20261006'
    inventory = read(checked(reg['protected_stage6_inventory']))
    existing = {p.relative_to(WORK).as_posix() for root in (old_out, old_report) for p in root.rglob('*')
                if p.is_file() and p != old_report / 'deliverable_hashes.json'}
    require(len(inventory) == 1347 and set(inventory) == existing, 'Protected prior stage6 full inventory differs')
    for name, value in {**read(old_out / 'registration.json')['sources'], **inventory}.items():
        require(reg['sources'].get(name) == value, 'Protected previous source/output omitted or changed')
    cohort = read(WORK / 'S00/configs/cohort.json')
    folds = read(WORK / 'S00' / cohort['fold_file'])
    ids = cohort['subject_ids']
    labels = {r['subject_id']: r['label'] for r in read(WORK / 'S00/manifests/subjects.json')}
    require(ids == reg['subject_ids'] and len(ids) == len(set(ids)) == 37 and set(labels) == set(ids) and
            reg['folds_sha256'] == cohort['folds_sha256'] == digest(WORK / 'S00' / cohort['fold_file']), 'Original cohort/fold source differs')
    for split in folds.values():
        require(set(split) == {'train', 'val', 'test'} and len(split['train']) in (26, 27) and len(split['val']) == 3 and
                len(split['test']) in (7, 8) and set(sum(split.values(), [])) == set(ids) and len(sum(split.values(), [])) == 37,
                'Original train/val/test personnel isolation differs')
    require(Counter(sid for split in folds.values() for sid in split['test']) == Counter(ids), 'Each seed must test each person exactly once')
    return reg, ids, folds, labels, {'registered_sources_verified': len(reg['sources']), 'protected_stage6_deliverables': 1347,
        'new_independent_people': 0, 'old37_remain_development': True, 'stage6_14_not_used_for_fitting_or_selection': True}


def author_checkpoint_tensors(path):
    require(path.stat().st_size == AUTHOR_BYTES and digest(path) == AUTHOR_SHA and digest(path, 'md5') == AUTHOR_MD5,
            'Full checkpoint does not match authorMD5/mirrorSHA/exactbytes')
    import torch
    allowed = {('__builtin__', 'dict'): dict, ('__builtin__', 'getattr'): getattr,
        ('collections', 'OrderedDict'): OrderedDict, ('collections', 'defaultdict'): defaultdict,
        ('torch', 'FloatStorage'): torch.FloatStorage, ('torch._utils', '_rebuild_parameter'): torch._utils._rebuild_parameter,
        ('torch._utils', '_rebuild_tensor_v2'): torch._utils._rebuild_tensor_v2,
        ('torch.optim.adamw', 'AdamW'): torch.optim.AdamW, ('torch.optim.lr_scheduler', 'OneCycleLR'): torch.optim.lr_scheduler.OneCycleLR}
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        require(len(names) == 1, 'Formal checkpoint pickle archive layout differs')
        ops = list(pickletools.genops(archive.read(names[0])))
    require(ops[0][0].name == 'PROTO' and ops[0][1] == 2 and
            {tuple(value.split(' ', 1)) for op, value, _ in ops if op.name == 'GLOBAL'} == set(allowed) and
            not any(op.name in ('STACK_GLOBAL', 'EXT1', 'EXT2', 'EXT4') for op, _, _ in ops), 'Formal checkpoint has unregistered pickle globals')
    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            require((module, name) in allowed, 'Unapproved checkpoint metadata global')
            return allowed[module, name]
    payload = torch.load(path, map_location='cpu', weights_only=False,
                         pickle_module=types.SimpleNamespace(__name__='independent_author_restricted_pickle', Unpickler=Restricted))
    tensors = {key[len('target_encoder.'):]: value for key, value in payload['state_dict'].items() if key.startswith('target_encoder.')}
    require(tensors and all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and torch.isfinite(value).all()
                            for value in tensors.values()), 'Formal target encoder tensors missing/nonfinite/notfloat32')
    return tensors, {'formal_checkpoint_SHA256': AUTHOR_SHA, 'author_file_MD5': AUTHOR_MD5, 'formal_checkpoint_bytes': AUTHOR_BYTES,
        'restricted_globals_verified': 9, 'unknown_globals_allowed': False, 'encoder_state_sha256': state_digest(tensors),
        'encoder_tensor_keys': len(tensors), 'new_encoder_forward_calls': 0}


def audit_assets(reg):
    qualification = read(checked(reg['asset_qualification']))
    require(qualification['passed'] and qualification['new_real_encoder_forward_calls'] == qualification['new_formal_fits'] ==
            qualification['new_updates'] == 0 and qualification['physical_calibration_confirmed'] is False, 'Asset qualification claimed model fitting/physical calibration')
    loading = qualification['eegpt_loading']
    tensors, audit = author_checkpoint_tensors(checked(loading['checkpoint']))
    require(loading['passed'] and loading['encoder_state_sha256'] == audit['encoder_state_sha256'] and
            loading['encoder_keys'] == loading['loaded_encoder_keys'] == len(tensors) and
            loading['strict_encoder_coverage'] == 1. and loading['missing_encoder_keys'] == loading['unexpected_encoder_keys'] == [] and
            loading['key_mapping'] == {name: 'target_encoder.' + name for name in tensors} and loading['trainable_parameters'] == 0 and
            loading['eval'] and loading['forward_calls_during_loading'] == 0 and loading['no_random_encoder_residual'] is True,
            'Qualified encoder does not cover every exact formal author tensor')
    source_path = WORK / 'tridep_ccml/tridep_eegpt/vendor/EEGPT_mcae.py'
    require(digest(source_path) == AUTHOR_SOURCE_SHA, 'Pinned author model source modified')
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    mapping = next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'CHANNEL_DICT' for t in n.targets))
    channel_names = ast.literal_eval(mapping.generators[0].iter.args[0])
    dictionary = {name.upper(): i for i, name in enumerate(channel_names)}
    require(len(dictionary) == 62 and tuple(dictionary[name] for name in CHANNELS) == CHANNEL_IDS and
            loading['channel_names'] == list(CHANNELS) and loading['channel_ids'] == list(CHANNEL_IDS), 'Original19 channel alias/positional mapping differs')
    projected, q64 = fixed_projection()
    actual = np.load(checked(qualification['projection']), allow_pickle=False)
    require(np.array_equal(actual, projected) and actual.dtype == np.float32, 'Fixed seed/sign/precision QR projection differs')
    projection_audit = qualification['projection_audit']
    require(projection_audit['seed'] == PROJECTION_SEED and projection_audit['shape'] == [512, 200] and
            projection_audit['label_or_data_inputs'] == [] and projection_audit['trainable_parameters'] == 0 and
            projection_audit['extra_scale'] == 1. and projection_audit['float64_array_sha256'] == hashlib.sha256(q64.tobytes()).hexdigest() and
            projection_audit['float32_array_sha256'] == hashlib.sha256(projected.tobytes()).hexdigest() and
            np.max(np.abs(q64.T @ q64 - np.eye(200))) < 1e-12, 'Fixed projection has data dependence or different orthogonal recipe')
    del tensors
    return qualification, projected, {**audit, 'author_source_SHA256': AUTHOR_SOURCE_SHA, 'all_author_encoder_tensors_matched': True,
        'projection_recipe_independently_verified': True, 'physical_calibration_confirmed': False}


def audit_features(reg, ids, qualification, projection):
    manifest_path = OUT / 'features_manifest.json'
    manifest = read(manifest_path)
    require(manifest['passed'] and manifest['registration'] == desc(OUT / 'registration.json') and manifest['subject_ids'] == ids and
            manifest['counts'] == FEATURE_BUDGET and manifest['raw_manifest'] == reg['features']['raw_manifest'] and
            manifest['projection'] == reg['features']['projection'] and manifest['frozen_states_exact'] is True and
            manifest['frozen_states_before'] == manifest['frozen_states_after'] and
            manifest['formal_fits'] == manifest['formal_updates'] == 0, 'Frozen feature export scope/state differs')
    raw_manifest = read(checked(manifest['raw_manifest']))
    raw_rows = {r['subject_id']: r for r in raw_manifest['records']}
    records = {r['subject_id']: r for r in manifest['records']}
    require(len(records) == len(manifest['records']) == 37 and set(records) == set(ids), 'Feature identities repeated/omitted')
    shapes = {'eegpt': [60, 200], 'labram': [60, 200], 'eegpt_input': [60, 19, 1024], 'labram_input': [60, 19, 800],
        'eegpt_tokens': [60, 16, 4, 512], 'labram_tokens': [60, 76, 200], 'eegpt_native512': [60, 512],
        'eegpt_repeat_tokens': [8, 16, 4, 512], 'labram_repeat_tokens': [8, 76, 200],
        'eegpt_repeat_native512': [8, 512], 'eegpt_repeat': [8, 200], 'labram_repeat': [8, 200]}
    arrays = {key: [] for key in ('labram', 'eegpt')}
    for sid in ids:
        record = records[sid]; sidecar = read(checked(record['sidecar']))
        require(record['features'] == record['bundle'] == sidecar['features'] and record['shapes'] == sidecar['shapes'] == shapes and
                sidecar['subject_id'] == sid and sidecar['registration'] == manifest['registration'] and
                sidecar['raw'] == raw_rows[sid]['raw'] and sidecar['raw_manifest'] == manifest['raw_manifest'] and
                sidecar['projection'] == manifest['projection'] and sidecar['fit_subject_ids'] == [] and
                sidecar['projection_fitted'] is False and sidecar['frozen_eval'] and sidecar['batch_size'] == 8 and
                sidecar['first_windows'] == 60 and sidecar['repeat_windows'] == 8 and
                sidecar['eegpt_canonical_channels'] == list(CHANNELS) and sidecar['eegpt_channel_ids'] == list(CHANNEL_IDS) and
                sidecar['repeat_exact'] == {'eegpt_tokens': True, 'labram_tokens': True, 'eegpt_features': True, 'labram_features': True},
                'Feature sidecar role/input/interface/repeat scope differs')
        raw = np.load(checked(raw_rows[sid]['raw']), allow_pickle=False)
        eegpt_input, labram_input = input_math(raw)
        with np.load(checked(record['features']), allow_pickle=False) as bundle:
            require(set(bundle.files) == set(shapes) and all(bundle[key].dtype == np.float32 and list(bundle[key].shape) == shape and
                    np.isfinite(bundle[key]).all() for key, shape in shapes.items()), 'Frozen audit array shapes/dtypes/keys differ')
            require(np.array_equal(bundle['eegpt_input'], eegpt_input) and np.array_equal(bundle['labram_input'], labram_input),
                    'Independent 200→256Hz/full-resample-first4s/scaling math differs')
            native, projected = pool_and_project(bundle['eegpt_tokens'], projection)
            labram_pooled = np.concatenate([bundle['labram_tokens'][start:start + 8].mean(axis=1, dtype=np.float32) for start in range(0, 60, 8)])
            require(np.array_equal(bundle['eegpt_native512'], native) and np.array_equal(bundle['eegpt'], projected) and
                    np.array_equal(bundle['labram'], labram_pooled), 'Independent native pooling/QR interface differs')
            for repeated, initial in [('eegpt_repeat_tokens', 'eegpt_tokens'), ('labram_repeat_tokens', 'labram_tokens'),
                ('eegpt_repeat_native512', 'eegpt_native512'), ('eegpt_repeat', 'eegpt'), ('labram_repeat', 'labram')]:
                require(np.array_equal(bundle[repeated], bundle[initial][:8]), 'Same-batch8 frozen feature repeat is not exact')
            for key in arrays:
                arrays[key].append(bundle[key].copy())
    ledger = read(checked(manifest['ledger']))
    require(ledger['registration'] == manifest['registration'] and ledger['status'] == 'complete' and ledger['counts'] == ledger['budget'] == FEATURE_BUDGET and
            ledger['started_batches'] == ledger['returned_batches'] == {'eegpt': 333, 'labram': 333} and
            ledger['states_before'] == ledger['states_after'] == manifest['frozen_states_before'] and
            ledger['formal_fits'] == ledger['formal_updates'] == ledger['encoder_gradient_passes'] == 0,
            'Encoder forward ledger/state/fitting scope differs')
    expected = [(sid, encoder, phase, list(range(start, end))) for sid in ids for encoder in ('eegpt', 'labram')
                for phase, start, end in [('first', start, min(start + 8, 60)) for start in range(0, 60, 8)] + [('repeat', 0, 8)]]
    require(len(ledger['attempts']) == len(expected) == 666, 'Encoder call start/return record count differs')
    for row, (sid, encoder, phase, indices) in zip(ledger['attempts'], expected):
        require(row['subject_id'] == sid and row['encoder'] == encoder and row['phase'] == phase and
                row['window_indices'] == indices and row['batch_size'] == len(indices) and
                datetime.fromisoformat(row['returned_at']) >= datetime.fromisoformat(row['started_at']), 'Encoder batch8/repeat/call accounting differs')
    require(ledger['load_audits']['eegpt']['encoder_state_sha256'] == qualification['eegpt_loading']['encoder_state_sha256'] ==
            ledger['states_before']['eegpt'], 'Extracted EEGPT is not the strict qualified author encoder')
    for model, qualification_key in (('eegpt', 'eegpt_loading'), ('labram', 'labram_loading')):
        require({k: v for k, v in ledger['load_audits'][model].items() if k != 'device'} ==
                {k: v for k, v in qualification[qualification_key].items() if k != 'device'},
                'Extracted encoder loading differs from source-qualified model: ' + model)
    return {key: np.stack(values) for key, values in arrays.items()}, {'features_manifest': desc(manifest_path),
        'subject_features_verified': 37, 'independent_input_math_checks': 74, 'same_batch8_repeats_exact': 74,
        'encoder_batch_records_verified': 666, 'feature_counts': FEATURE_BUDGET, 'fixed_projection_independently_replayed': True,
        'new_encoder_forward_calls_in_verifier': 0}


def audit_fits(reg, ids, folds):
    import torch
    ledger = read(OUT / 'training_ledger.json')
    require(ledger['fits_completed'] == ledger['planned_fits'] == 30 and ledger['updates_completed'] == ledger['planned_updates'] ==
            ledger['gradient_passes_completed'] == 3600 and ledger['registration'] == desc(OUT / 'registration.json'), 'Formal30fit/3600update ledger incomplete')
    indexed = {(r['condition'], r['seed'], r['fold']): r for r in ledger['fits']}
    require(len(indexed) == len(ledger['fits']) == 30 and set(indexed) == {(c, s, f) for c in CONDITIONS for s in SEEDS for f in range(1, 6)},
            'Formal fit identifiers duplicate/missing')
    states, checks = {}, []
    for seed in SEEDS:
        expected_initial = expected_initial_state(seed); expected_hash = state_digest(expected_initial)
        for fold in range(1, 6):
            paired = []
            for condition, feature_key in zip(CONDITIONS, ('labram', 'eegpt')):
                folder = OUT / f'runs/{condition}/seed_{seed}/fold_{fold}'
                complete = read(folder / 'complete.json')
                require(complete['passed'] and complete['registration'] == desc(OUT / 'registration.json') and complete['updates'] == 120 and
                    {Path(r['path']).name for r in complete['artifacts']} == {'started.json', 'initial.pt', 'checkpoint.pt', 'metadata.json',
                     'train_history.json', 'batch_order.npy', 'progress.json'} and len(complete['artifacts']) == 7, 'Completed fit artifact inventory differs')
                for source in complete['artifacts']:
                    require(checked(source).parent == folder, 'Fit artifact points to another model/fold')
                meta = read(folder / 'metadata.json'); split = folds[f'fold_{fold}']
                expected = {'condition': condition, 'seed': seed, 'fold': fold, 'train_ids': split['train'], 'val_ids': split['val'],
                    'test_ids': split['test'], 'training': TRAINING, 'registration': desc(OUT / 'registration.json'), 'updates': 120,
                    'gradient_passes': 120, 'epochs': 60, 'parameters': 498434, 'input_key': feature_key,
                    'cohort_id': reg['cohort_id'], 'folds_sha256': reg['folds_sha256'], 'no_val_test_forward_during_training': True,
                    'checkpoint_selection': 'fixed_final_epoch', 'feature_manifest': desc(OUT / 'features_manifest.json')}
                require(meta == indexed[condition, seed, fold] and all(meta.get(k) == v for k, v in expected.items()), 'Fit metadata/source/train roles differ')
                checked(meta['input_independent_verification'])
                initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
                saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
                require(initial['registration'] == expected['registration'] and initial['condition'] == condition and initial['seed'] == seed and
                        initial['fold'] == fold and set(initial['state_dict']) == set(expected_initial) and
                        all(torch.equal(initial['state_dict'][k], expected_initial[k]) for k in expected_initial) and
                        meta['initial_state_sha256'] == expected_hash and saved['metadata'] == meta and
                        state_digest(saved['state_dict']) == meta['final_state_sha256'], 'Independent original CNN init/final checkpoint tensor identity differs')
                optimizer = saved['optimizer_state_dict']
                require(len(optimizer['state']) == 8 and all(float(v['step']) == 120 for v in optimizer['state'].values()) and
                        optimizer['param_groups'][0]['lr'] == .005 and optimizer['param_groups'][0]['weight_decay'] == 0., 'Adamax state/120 steps/config differs')
                order = np.load(folder / 'batch_order.npy', allow_pickle=False)
                indices = [ids.index(s) for s in split['train']]
                require(order.dtype == np.int64 and order.shape == (60, len(indices)) and
                        all(Counter(row.tolist()) == Counter(indices) for row in order) and
                        hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == meta['batch_order_sha256'], 'Saved batching includes nontrain/repeats/omitted people')
                history = read(folder / 'train_history.json')
                require(len(history) == 60, 'Training history epochs missing')
                for epoch, row in enumerate(history, 1):
                    batches = row['batches']
                    require(row['epoch'] == epoch and row['updates_completed'] == row['gradient_passes_completed'] == epoch * 2 and
                            [r['batch_size'] for r in batches] == [16, len(indices) - 16] and
                            all(math.isfinite(r['train_ce']) and r['train_ce'] >= 0 for r in batches) and
                            abs(row['train_ce'] - sum(r['batch_size'] * r['train_ce'] for r in batches) / len(indices)) <= 1e-12,
                            'Epoch fulltrain minibatch loss/update/gradient accounting differs')
                progress = read(folder / 'progress.json')
                require(progress['epoch'] == 60 and progress['updates_completed'] == progress['updates_started'] ==
                        progress['gradient_passes_completed'] == 120, 'Training action not fully closed')
                states[condition, seed, fold] = (meta, saved['state_dict']); paired.append((expected_hash, meta['batch_order_sha256']))
            require(paired[0] == paired[1], 'Matched conditions do not share initialization and batch order')
            checks.append({'seed': seed, 'fold': fold, 'initial_state': paired[0][0], 'batch_order': paired[0][1]})
    return states, {'fits_verified': 30, 'updates_verified': 3600, 'training_gradient_passes_verified': 3600,
        'paired_initial_and_batch_orders_verified': 15, 'initial_tensor_checks_from_independent_seed_recipe': 30,
        'only_original_train_people_reach_updates': True, 'paired_checks': checks}


def audit_outputs(reg, ids, folds, labels, states):
    predictions = rows(REPORT / 'prediction_rows.jsonl')
    indexed = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']): r for r in predictions}
    require(len(predictions) == len(indexed) == 1110 and set(indexed) == {(c, s, f, sid) for c in CONDITIONS for s in SEEDS
            for f in range(1, 6) for sid in ids}, 'Prediction identities duplicate/omitted')
    branches = {(r['seed'], r['outer_fold'], r['subject_id']): r for r in rows(STAGE0 / 'branch_probabilities.jsonl')}
    controls = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']): r for r in rows(STAGE0 / 'control_predictions.jsonl')}
    require(len(branches) == 555, 'Original S/T probability source coverage differs')
    reload = read(REPORT / 'reload_validation.json')
    checks = {(r['condition'], r['seed'], r['fold']): r for r in reload['checks']}
    require(reload['passed'] and reload['registration'] == desc(OUT / 'registration.json') and
            reload['producer_head_person_forward_calls'] == 2220 and len(checks) == len(reload['checks']) == 30,
            'Producer two-reload/frozen-state proof incomplete')
    replay = read(REPORT / 'independent_replay.json')
    repeated = {(r['condition'], r['seed'], r['fold']): r for r in replay['records']}
    require(replay['passed'] and replay['registration'] == desc(OUT / 'registration.json') and replay['source'] == desc(__file__) and
            replay['logical_head_person_forward_calls'] == 1110 and replay['frozen_states_unchanged'] == 30 and
            len(repeated) == len(replay['records']) == 30, 'Independent frozen1110-head replay incomplete')
    marker = read(OUT / 'actions/replay.json')
    require(marker['status'] == 'complete' and marker['registration'] == replay['registration'] and marker['source'] == desc(__file__),
            'Independent pre-forward replay action not fully closed')
    for (condition, seed, fold), (meta, _) in states.items():
        bundle_path = OUT / f'evaluation/{condition}/seed_{seed}/fold_{fold}.npz'
        sidecar = read(bundle_path.with_suffix('.json'))
        cp = desc(OUT / f'runs/{condition}/seed_{seed}/fold_{fold}/checkpoint.pt')
        require(sidecar['bundle'] == desc(bundle_path) and sidecar['checkpoint'] == cp and sidecar['registration'] == replay['registration'] and
                sidecar['feature_manifest'] == desc(OUT / 'features_manifest.json') and sidecar['producer_head_person_forward_calls'] == 74 and
                sidecar['state_before'] == sidecar['state_after'] == meta['final_state_sha256'], 'Evaluation bundle/checkpoint/state binding differs')
        check = checks[condition, seed, fold]
        require(check['checkpoint'] == cp and check['state_sha256'] == meta['final_state_sha256'] and
                all(check[key] is True for key in ('exact_logits', 'exact_probabilities', 'exact_representations', 'frozen_eval_state_unchanged')),
                'Producer reload probability/logit/128h equality failed')
        independent = repeated[condition, seed, fold]
        require(independent['checkpoint'] == cp and independent['state_before'] == independent['state_after'] == meta['final_state_sha256'],
                'Independent replay checkpoint/state differs')
        with np.load(checked(independent['bundle']), allow_pickle=False) as expected, np.load(bundle_path, allow_pickle=False) as actual:
            require(set(actual.files) == {'logits', 'probabilities', 'features', 'p_eeg', 'subject_ids', 'labels', 'splits'} and
                    set(actual.files) == set(expected.files) and all(np.array_equal(actual[k], expected[k]) for k in actual.files),
                    'Independent frozen head replay differs from producer outputs bitwise')
            require(actual['subject_ids'].tolist() == ids and actual['labels'].tolist() == [labels[s] for s in ids] and
                    actual['logits'].shape == (37, 2) and actual['features'].shape == (37, 128) and
                    actual['probabilities'].shape == (37, 2) and actual['p_eeg'].shape == (37,) and
                    actual['p_eeg'].dtype == np.float64 and all(actual[k].dtype == np.float32 and np.isfinite(actual[k]).all()
                        for k in ('logits', 'features', 'probabilities')), 'Saved output endpoint/dtype/finite contract differs')
            exponent = np.exp(actual['logits'].astype(np.float64) - actual['logits'].max(axis=1, keepdims=True))
            require(np.max(np.abs(exponent / exponent.sum(axis=1, keepdims=True) - actual['probabilities'])) < 1e-7,
                    'Saved frozen probabilities inconsistent with logits')
            for i, sid in enumerate(ids):
                row = indexed[condition, seed, fold, sid]; old = branches[seed, fold, sid]
                est = controls['EST-original-val-weighted', seed, fold, sid]; st = controls['ST-val-weighted', seed, fold, sid]
                split = next(k for k in ('train', 'val', 'test') if sid in folds[f'fold_{fold}'][k])
                require(row['label'] == old['label'] == labels[sid] and row['split'] == old['split'] == actual['splits'][i] == split and
                        row['p_eeg'] == float(actual['p_eeg'][i]) == float(actual['probabilities'][i, 1]) and
                        row['p_speech'] == old['p_speech'] and row['p_text'] == old['p_text'] and
                        row['eeg_checkpoint'] == cp and row['fixed_ST_checkpoints'] == {m: old['parent_checkpoints'][m] for m in ('speech', 'text')} and
                        row['registration'] == replay['registration'], 'Row output/label/split/frozenS/T source differs')
                fixed = sum(w * p for w, p in zip(est['fusion_weights'], (row['p_eeg'], old['p_speech'], old['p_text'])))
                require(row['fusion_weights'] == est['fusion_weights'] and row['p_fixed_EST'] == fixed and row['p_fixed_ST'] == st['p_final'] and
                        row['p_original_EST'] == est['p_final'] and row['prediction_eeg'] == int(row['p_eeg'] > .5) and
                        row['prediction_fixed_EST'] == int(fixed > .5) and (est['fusion_weights'][0] != 0 or fixed == est['p_final']),
                        'Fixed original fusion was reselected/changed or zeroEEG-weight invariance failed')
    return predictions, controls, {'producer_head_person_forwards': 2220, 'independent_head_person_forwards': 1110,
        'total_head_person_forwards': 3330, 'prediction_rows_verified': 1110, 'bitwise_independent_logits_probs_128h_models': 30,
        'original_S_T_rows_exact': 1110, 'new_fusion_selections': 0}


def audit_metrics_and_gates(predictions, controls, folds):
    stored = read(REPORT / 'metrics.json'); calculated = {}; groups = 0
    def compare(actual, expected, label):
        for key in KEYS:
            equal(actual[key], expected[key], label + '/' + key)
        for key in ('n_subjects', 'confusion_hc_mdd'):
            if key in actual:
                equal(actual[key], expected[key], label + '/' + key)
    for condition in CONDITIONS:
        calculated[condition] = {}
        for view, key in VIEWS.items():
            reference = stored[condition][view]
            fm = {(r['seed'], r['fold'], r['split']): r for r in reference['fold_metrics']}
            require(len(fm) == len(reference['fold_metrics']) == 45, 'Per-view fold metrics count differs')
            seed_means, fold_values = {}, []
            for seed in SEEDS:
                for fold in range(1, 6):
                    for split in ('train', 'val', 'test'):
                        group = [r for r in predictions if (r['condition'], r['seed'], r['outer_fold'], r['split']) == (condition, seed, fold, split)]
                        require({r['subject_id'] for r in group} == set(folds[f'fold_{fold}'][split]) and len(group) == len(folds[f'fold_{fold}'][split]),
                                'Independent metric personnel coverage differs')
                        value = {'seed': seed, 'fold': fold, 'split': split, **metric(group, key)}
                        compare(fm[seed, fold, split], value, 'Fold metric'); fold_values.append(value); groups += 1
                seed_means[str(seed)] = {split: {k: float(np.mean([r[k] for r in fold_values if r['seed'] == seed and r['split'] == split]))
                    for k in KEYS} for split in ('train', 'val', 'test')}
                for split in ('train', 'val', 'test'):
                    compare(reference['seed_means'][str(seed)][split], seed_means[str(seed)][split], 'Seed fivefold mean')
                oof = [r for r in predictions if (r['condition'], r['seed'], r['split']) == (condition, seed, 'test')]
                require(len(oof) == len({r['subject_id'] for r in oof}) == 37, 'PooledOOF reinterpreted repeated people')
                compare(reference['pooled_oof'][str(seed)], metric(oof, key), 'Pooled OOF')
            main = {k: float(np.mean([seed_means[str(s)]['test'][k] for s in SEEDS])) for k in KEYS}
            compare(reference['main_test'], main, 'Primary5fold-then3seed metric')
            calculated[condition][view] = {'main_test': main, 'seed_means': seed_means}
    require(groups == 180, 'Expected180 independent condition/view/seed/fold/role metric groups')
    legacy = read(STAGE0 / 'metrics.json')
    control_metrics = {}
    for name in ('ST-val-weighted', 'EST-original-val-weighted'):
        seed_means = {}
        for seed in SEEDS:
            seed_means[str(seed)] = {}
            for split in ('train', 'val', 'test'):
                values = [metric([r for (c, s, f, _), r in controls.items() if (c, s, f, r['split']) == (name, seed, fold, split)], 'p_final') for fold in range(1, 6)]
                seed_means[str(seed)][split] = {k: float(np.mean([v[k] for v in values])) for k in KEYS}
                compare(legacy[name]['train_val_test_seed_means'][str(seed)][split], seed_means[str(seed)][split], 'Historical control seedmean')
        main = {k: float(np.mean([seed_means[str(s)]['test'][k] for s in SEEDS])) for k in KEYS}
        compare(legacy[name]['main_test'], main, 'Historical control main')
        control_metrics[name] = {'main_test': main, 'seed_means': seed_means}
    acceptance = read(REPORT / 'acceptance.json'); expected_acceptance = {}
    matched = CONDITIONS[0]; bindings = read(REPORT / 'fusion_weight_bindings.json')
    require(bindings['selection_count'] == 0 and bindings['source'] == desc(STAGE0 / 'control_predictions.jsonl') and len(bindings['records']) == 30,
            'Fusion binding/zero-new-selection ledger differs')
    expected_bindings = [{'condition': condition, 'seed': seed, 'fold': fold,
        'weights': controls['EST-original-val-weighted', seed, fold, folds[f'fold_{fold}']['train'][0]]['fusion_weights'],
        'w_E_zero': controls['EST-original-val-weighted', seed, fold, folds[f'fold_{fold}']['train'][0]]['fusion_weights'][0] == 0,
        'selection_count': 0, 'original_val_ids': folds[f'fold_{fold}']['val']}
        for seed in SEEDS for fold in range(1, 6) for condition in CONDITIONS]
    equal(bindings['records'], expected_bindings, 'Exact historical EST weight/val provenance bindings')
    for condition in CONDITIONS:
        # Classification boundaries use the published float representation,
        # after every scalar above has been independently recomputed. This
        # preserves strict >=/</> gates at a possible rounding equality.
        eeg, est = stored[condition]['eeg'], stored[condition]['fixed_EST']
        direct_eeg, direct_est = stored[matched]['eeg'], stored[matched]['fixed_EST']
        result = {'is_candidate': condition != matched, 'direct_control': matched,
            'eeg_vs_matched_control': gate(eeg['main_test'], direct_eeg['main_test'], eeg['seed_means'], direct_eeg['seed_means']),
            'fixed_EST_vs_matched_control': gate(est['main_test'], direct_est['main_test'], est['seed_means'], direct_est['seed_means']),
            'fixed_EST_vs_original_EST': gate(est['main_test'], legacy['EST-original-val-weighted']['main_test'], est['seed_means'], legacy['EST-original-val-weighted']['train_val_test_seed_means']),
            'fixed_EST_vs_fixed_ST': gate(est['main_test'], legacy['ST-val-weighted']['main_test'], est['seed_means'], legacy['ST-val-weighted']['train_val_test_seed_means']),
            'w_E_zero_model_groups': sum(r['condition'] == condition and r['w_E_zero'] for r in bindings['records']),
            'fusion_selections': 0, 'new_calibration_or_threshold_selection': False}
        result['eeg_keep'] = condition != matched and result['eeg_vs_matched_control']['passed']
        result['fusion_keep'] = condition != matched and all(result[k]['passed'] for k in ('fixed_EST_vs_matched_control', 'fixed_EST_vs_original_EST', 'fixed_EST_vs_fixed_ST'))
        expected_acceptance[condition] = result
    equal(acceptance, expected_acceptance, 'Independent classification/probability acceptance')
    lookup = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']): r for r in predictions}
    expected_paired = []
    for row in predictions:
        if row['split'] != 'test':
            continue
        for view, key in VIEWS.items():
            references = {'matched-control': lookup[matched, row['seed'], row['outer_fold'], row['subject_id']][key]}
            if view == 'fixed_EST':
                references.update({'fixed-ST': row['p_fixed_ST'], 'original-EST': row['p_original_EST']})
            for reference, probability in references.items():
                pc, pr = int(row[key] > .5), int(probability > .5); cc, cr = pc == row['label'], pr == row['label']
                change = 'add_correct' if cc and not cr else 'add_error' if cr and not cc else 'unchanged_correct' if cc else 'unchanged_error'
                expected_paired.append({k: row[k] for k in ('condition', 'seed', 'outer_fold', 'subject_id', 'label')} |
                    {'view': view, 'reference': reference, 'p_candidate': row[key], 'p_reference': probability,
                     'prediction_candidate': pc, 'prediction_reference': pr, 'change': change})
    equal(rows(REPORT / 'paired_changes.jsonl'), expected_paired, 'Paired subject classification changes')
    summary = {c: {v: {r: dict(Counter(p['change'] for p in expected_paired if p['condition'] == c and p['view'] == v and p['reference'] == r))
            for r in (('matched-control',) if v == 'eeg' else ('matched-control', 'fixed-ST', 'original-EST'))} for v in VIEWS} for c in CONDITIONS}
    equal(read(REPORT / 'paired_summary.json'), summary, 'Paired count summary')
    fixed = read(REPORT / 'fixed_controls.json')
    require(fixed['stage0'] == desc(STAGE0 / 'metrics.json') and fixed['ST'] == legacy['ST-val-weighted'] and
            fixed['EST'] == legacy['EST-original-val-weighted'] and fixed['LaBraM5s_diagnostic_only'] is True and
            fixed['different_window_length_is_not_direct_backbone_control'] is True, 'Historical controls changed/5s substituted as matched control')
    return {'metric_groups_verified': groups, 'additional_original_control_metric_groups': 90,
        'paired_change_rows_verified': len(expected_paired), 'main_test': {c: {v: calculated[c][v]['main_test'] for v in VIEWS} for c in CONDITIONS},
        'eeg_keep': [c for c in CONDITIONS if expected_acceptance[c]['eeg_keep']],
        'fusion_keep': [c for c in CONDITIONS if expected_acceptance[c]['fusion_keep']], 'aggregation': '5fold means then3seed mean; OOF separately',
        'probability_improvement_does_not_replace_classification_gate': True}


def replay():
    """One independently loaded single-person forward per30×37 frozen heads."""
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    reg, ids, folds, labels, _ = audit_registration()
    require(read(OUT / 'actions/evaluate.json')['status'] == 'complete' and
            read(OUT / 'evaluation/complete.json')['models'] == 30 and read(REPORT / 'reload_validation.json')['passed'],
            'All30 formal fits and producer uniform two-reload evaluation must complete before independent replay')
    output_path = REPORT / 'independent_replay.json'
    action_path = OUT / 'actions/replay.json'
    if output_path.exists():
        old = read(output_path)
        require(old['source'] == desc(__file__) and old['registration'] == desc(OUT / 'registration.json') and
                old['passed'] and read(action_path)['status'] == 'complete', 'Existing replay evidence not fully closed')
        for record in old['records']:
            checked(record['bundle'])
        return old
    require(not action_path.exists() and not (OUT / 'independent_replay').exists(), 'Replay already started/failed; no automatic additional forwards')
    action = {'action': 'replay', 'status': 'started', 'started_at': datetime.now(timezone.utc).isoformat(),
        'registration': desc(OUT / 'registration.json'), 'source': desc(__file__), 'planned_head_person_forward_calls': 1110}
    save(action_path, action)
    try:
        import torch
        states, _ = audit_fits(reg, ids, folds)
        qualification, projection, _ = audit_assets(reg)
        arrays, _ = audit_features(reg, ids, qualification, projection)
        sys.path.insert(0, str(WORK / 'S00/upstream-dev'))
        from lib.models.models import EEGConvPoolClassifier
        require(torch.cuda.is_available(), 'Registered independent replay requires CUDA float32')
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True; torch.use_deterministic_algorithms(True)
        records, calls = [], 0
        for condition, input_key in zip(CONDITIONS, ('labram', 'eegpt')):
            x = torch.as_tensor(arrays[input_key], dtype=torch.float32, device='cuda')
            for seed in SEEDS:
                for fold in range(1, 6):
                    meta, state = states[condition, seed, fold]
                    model = EEGConvPoolClassifier(input_dim=200, seq_len=60).to('cuda')
                    model.load_state_dict(state, strict=True); model.requires_grad_(False).eval()
                    before = state_digest(model.state_dict())
                    require(before == meta['final_state_sha256'], 'Independent strictly loaded head checkpoint state differs')
                    captured = []
                    hook = model.classifier[-1].register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
                    logits, probabilities, hidden = [], [], []
                    with torch.inference_mode():
                        for index in range(37):
                            value = model(x[index:index + 1]); h = captured.pop()
                            require(value.shape == (1, 2) and h.shape == (1, 128) and torch.isfinite(value).all() and torch.isfinite(h).all(),
                                    'Independent frozen head output shape/nonfinite differs')
                            logits.append(value.cpu().numpy()[0]); hidden.append(h.cpu().numpy()[0])
                            probabilities.append(torch.softmax(value, dim=-1).cpu().numpy()[0]); calls += 1
                    hook.remove()
                    require(before == state_digest(model.state_dict()) and not any(m.training for m in model.modules()) and
                            not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'Frozen independent head state/gradient changed')
                    path = OUT / f'independent_replay/{condition}/seed_{seed}/fold_{fold}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    require(not path.exists(), 'Refusing to overwrite independent replay bundle')
                    p = np.stack(probabilities)
                    np.savez_compressed(path, logits=np.stack(logits), probabilities=p, features=np.stack(hidden),
                        p_eeg=p[:, 1].astype(np.float64), subject_ids=np.asarray(ids), labels=np.asarray([labels[s] for s in ids]),
                        splits=np.asarray([next(k for k in ('train', 'val', 'test') if sid in folds[f'fold_{fold}'][k]) for sid in ids]))
                    records.append({'condition': condition, 'seed': seed, 'fold': fold, 'bundle': desc(path),
                        'checkpoint': desc(OUT / f'runs/{condition}/seed_{seed}/fold_{fold}/checkpoint.pt'),
                        'state_before': before, 'state_after': before, 'logical_head_person_forward_calls': 37,
                        'strict_checkpoint_load': True, 'metadata_exact': True})
                    del model
        require(calls == 1110 and len(records) == 30, 'Independent replay budget differs')
        result = {'passed': True, 'registration': desc(OUT / 'registration.json'), 'source': desc(__file__), 'records': records,
            'logical_head_person_forward_calls': calls, 'frozen_states_unchanged': 30, 'formal_fits': 0, 'formal_updates': 0,
            'new_encoder_forwards': 0, 'new_S_T_forwards': 0, 'feature_manifest': desc(OUT / 'features_manifest.json')}
        save(output_path, result)
        action.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(), actual_head_person_forward_calls=calls)
        save(action_path, action, mutable=True)
        print(json.dumps({'passed': True, 'independent_head_person_forward_calls': calls, 'models': 30, 'formal_updates': 0}))
        return result
    except BaseException as error:
        action.update(status='failed', failed_at=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__, error=str(error))
        save(action_path, action, mutable=True)
        raise


def audit_closeout(reg):
    status = read(REPORT / 'status.json')
    require(status['status'] == 'complete' and status['passed'] and status['formal_fits'] == 30 and
            status['formal_updates'] == status['training_gradient_passes'] == 3600 and status['budget'] == reg['budget'] and
            status['registration'] == desc(OUT / 'registration.json') and status['independent_verification'] == desc(REPORT / 'independent_verification.json') and
            status['new_independent_people'] == 0 and status['stage6_14_people_reused_for_selection'] is False and
            status['original_stage4_outputs_modified'] is False and status['next_phase_started'] is False, 'Matched4s final status/role/work budget differs')
    checked(status['report']); checked(status['acceptance'])
    inventory_path = REPORT / 'deliverable_hashes.json'; inventory = read(inventory_path)
    actual = {p.relative_to(WORK).as_posix() for root in (OUT, REPORT) for p in root.rglob('*') if p.is_file() and p != inventory_path}
    require(set(inventory) == actual, 'Complete inventory omits/adds a real deliverable')
    for name, value in inventory.items():
        checked({'path': name, 'sha256': value})
    return {'passed': True, 'deliverable_hashes_verified': len(inventory), 'protected_stage6_deliverables': 1347, 'read_only': True}


def verify(inputs_only=False, post_closeout=False, save_result=True):
    require(not (inputs_only and post_closeout), 'Feature-input acceptance is not a completed experiment')
    if post_closeout:
        save_result = False
    reg, ids, folds, labels, source_audit = audit_registration()
    qualification, projection, assets_audit = audit_assets(reg)
    arrays, feature_audit = audit_features(reg, ids, qualification, projection)
    result = {'passed': True, 'registration': desc(OUT / 'registration.json'), 'source': desc(__file__),
        'mode': 'inputs_only' if inputs_only else 'full', 'source_audit': source_audit, 'asset_audit': assets_audit,
        'feature_audit': feature_audit, 'new_model_forwards_in_verifier': 0, 'new_fits_updates_in_verifier': 0}
    if inputs_only:
        require(not (OUT / 'runs').exists() and not (OUT / 'training_ledger.json').exists() and not (REPORT / 'metrics.json').exists(),
                'Initial input acceptance must precede formal fits and score release')
        target = REPORT / 'input_independent_verification.json'
    else:
        input_audit = read(REPORT / 'input_independent_verification.json')
        require(input_audit['passed'] and input_audit['registration'] == result['registration'] and input_audit['source'] == desc(__file__),
                'Fitting lacked the registered independent input gate')
        states, result['fit_audit'] = audit_fits(reg, ids, folds)
        predictions, controls, result['output_audit'] = audit_outputs(reg, ids, folds, labels, states)
        result['metric_and_acceptance_audit'] = audit_metrics_and_gates(predictions, controls, folds)
        source_path = REPORT / 'independent_verifier_source.json'
        source = {'source': desc(__file__), 'registration': result['registration'], 'independence': 'No producer computation imported; NumPy/sklearn audit plus separately budgeted1110 frozen-head replay'}
        if source_path.exists():
            require(read(source_path) == source, 'Independent source descriptor differs')
        elif save_result:
            save(source_path, source)
        elif post_closeout:
            require(False, 'Post-closeout requires previously saved independent source descriptor')
        if source_path.exists():
            result['source_descriptor'] = desc(source_path)
        target = REPORT / 'independent_verification.json'
    if target.exists():
        require(read(target) == result, 'Read-only reverification differs from saved independent evidence')
    elif post_closeout:
        require(False, 'Missing saved independent result for post-closeout')
    elif save_result:
        save(target, result)
    if post_closeout:
        result = {**result, 'post_closeout': audit_closeout(reg)}
    print(json.dumps({'passed': True, 'mode': result['mode'], 'registered_sources_verified': source_audit['registered_sources_verified'],
        'protected_stage6_deliverables': 1347, 'new_model_forwards_in_verifier': 0, 'saved': save_result,
        'deliverable_hashes_verified': result.get('post_closeout', {}).get('deliverable_hashes_verified')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('verify', 'replay'), default='verify')
    parser.add_argument('--inputs-only', action='store_true')
    parser.add_argument('--post-closeout', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    if args.action == 'replay':
        require(not (args.inputs_only or args.post_closeout or args.no_save), 'Replay cannot masquerade as read-only/input verification')
        replay()
    else:
        verify(args.inputs_only, args.post_closeout, not args.no_save)
