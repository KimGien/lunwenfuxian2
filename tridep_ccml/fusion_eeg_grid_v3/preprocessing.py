"""Train-fitted N0/N1 transforms and read-only sealed v2 feature loading."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
import numpy as np
import torch

from tridep_ccml.fusion_eeg_grid_v2 import common as V2
from . import common as C

THRESHOLD = 1e-6


def _verified_bytes(descriptor):
    path = (C.ROOT / descriptor['path']).resolve()
    C.require(path.is_relative_to(C.ROOT), 'Source outside workspace')
    data = path.read_bytes()
    C.require(hashlib.sha256(data).hexdigest() == descriptor['sha256'],
              'Registered feature source changed: ' + str(path))
    return data


@lru_cache(maxsize=1)
def _registered_feature_sources():
    """Load the accepted source graph once; never rebind to mutable metadata."""
    path = C.OUT / 'source_gate.json'
    data = path.read_bytes()
    gate = json.loads(data.decode('utf-8-sig'))
    C.require(gate.get('schema') == 'fusion_v3_source_gate_v1' and gate.get('passed') is True
              and gate.get('bundles') == 240, 'Accepted v3 source gate is required')
    expected = {(eeg, seed, fold) for eeg in ('ST', *C.EEG) for seed in C.SEEDS for fold in range(1, 6)}
    registered = {}
    for item in gate['caches']:
        key = item['eeg'], item['seed'], item['fold']
        C.require(key in expected and key not in registered, 'Duplicate/unregistered feature source identity')
        cache_path = V2.OUT / 'features' / key[0] / f'seed_{key[1]}' / f'fold_{key[2]}.npz'
        for field, expected_path in (('cache', cache_path), ('metadata', cache_path.with_suffix('.json'))):
            C.require((C.ROOT / item[field]['path']).resolve() == expected_path.resolve(),
                      'Registered feature path/identity differs')
        registered[key] = {field: dict(item[field]) for field in ('cache', 'metadata')}
    C.require(set(registered) == expected, 'Source gate is missing fixed feature identities')
    gate_descriptor = {'path': path.relative_to(C.ROOT).as_posix(),
                       'sha256': hashlib.sha256(data).hexdigest()}
    return registered, gate_descriptor


def _load_registered_bundle(eeg, seed, fold):
    registered, gate_descriptor = _registered_feature_sources()
    item = registered[(eeg, seed, fold)]
    # Verify and consume the same immutable byte buffer: paired cache/metadata
    # changes cannot redefine the accepted source, even between hash and load.
    metadata = json.loads(_verified_bytes(item['metadata']).decode('utf-8-sig'))
    C.require(metadata['cache'] == item['cache'], 'Metadata does not bind the registered cache')
    with np.load(io.BytesIO(_verified_bytes(item['cache'])), allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in ('features', 'subject_ids', 'labels', 'splits')}
    return arrays, metadata, [dict(item['cache']), dict(item['metadata'])], dict(gate_descriptor)


def fit_preprocessor(train, variant='N1', fit_people=None):
    train = np.asarray(train)
    if variant not in ('N0', 'N1'):
        raise ValueError('Unknown normalization variant')
    if train.ndim != 3 or train.shape[1] not in (2, 3) or train.shape[2] != 128:
        raise ValueError('Expected [people, ST/EST modalities, 128]')
    if len(train) == 0 or not np.isfinite(train).all():
        raise ValueError('Training representations must be nonempty and finite')
    if fit_people is not None and (len(fit_people) != len(train) or len(set(fit_people)) != len(train)):
        raise ValueError('Training people do not bind one-to-one to train rows')
    mean = train.astype(np.float64).mean(0)
    raw_std = train.astype(np.float64).std(0, ddof=0)
    near_constant = raw_std <= THRESHOLD
    mask = near_constant if variant == 'N1' else np.zeros_like(near_constant)
    scale = np.where(mask, 1., raw_std) if variant == 'N1' else np.maximum(raw_std, THRESHOLD)
    return {'schema': 'fusion_v3_normalization_1', 'variant': variant,
            'mean': mean.tolist(), 'std': scale.tolist(), 'raw_std': raw_std.tolist(),
            'mask': mask.tolist(), 'near_constant_mask': near_constant.tolist(),
            'threshold': THRESHOLD, 'comparison': 'raw_std <= threshold',
            'population_ddof': 0, 'fit_count': len(train),
            'fit_people': list(fit_people) if fit_people is not None else None,
            'shape': list(train.shape[1:]),
            'arithmetic': 'float64 population mean/std; normalized output float32',
            'masked_output': 0. if variant == 'N1' else None}


def apply_preprocessor(x, state):
    x = np.asarray(x)
    if state.get('schema') != 'fusion_v3_normalization_1' or state.get('variant') not in ('N0', 'N1'):
        raise ValueError('Invalid preprocessing state')
    if x.ndim != 3 or list(x.shape[1:]) != state['shape'] or not np.isfinite(x).all():
        raise ValueError('Input representation shape or finiteness differs')
    mean, scale = np.asarray(state['mean'], dtype=np.float64), np.asarray(state['std'], dtype=np.float64)
    mask = np.asarray(state['mask'], dtype=bool)
    raw = np.asarray(state['raw_std'], dtype=np.float64)
    if any(a.shape != x.shape[1:] for a in (mean, scale, mask, raw)) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid normalization statistics')
    expected_mask = raw <= THRESHOLD if state['variant'] == 'N1' else np.zeros_like(mask)
    expected_scale = np.where(expected_mask, 1., raw) if state['variant'] == 'N1' else np.maximum(raw, THRESHOLD)
    if state['threshold'] != THRESHOLD or not np.array_equal(mask, expected_mask) or not np.array_equal(scale, expected_scale):
        raise ValueError('Normalization policy differs from registration')
    result = (x.astype(np.float64) - mean) / scale
    result[:, mask] = 0.
    result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite standardized features')
    return result


def distribution_summary(x, indexes, modalities):
    """Input-only diagnostics: no model predictions or candidate test metrics."""
    summaries = {}
    for role, rows in indexes.items():
        summaries[role] = {}
        for v, name in enumerate(modalities):
            a = np.asarray(x)[rows, v].astype(np.float64)
            summaries[role][name] = {
                'shape': list(a.shape), 'min': float(a.min()), 'max': float(a.max()),
                'mean': float(a.mean()), 'std': float(a.std(ddof=0)),
                'abs_max': float(np.abs(a).max()),
                'abs_quantiles_50_90_99_100': np.quantile(np.abs(a), [.5, .9, .99, 1.]).tolist(),
                'finite': bool(np.isfinite(a).all())}
    return summaries


def load_inputs(identity, device, variant='N1'):
    """Same return contract as v2 training.load_inputs, with explicit variant.

    All feature files and their metadata are read-only v2 sources.  No backbone
    runs and no feature files are re-exported here.
    """
    seed, fold, eeg = identity['seed'], identity['fold'], identity['eeg']
    arrays, stmeta, sources, gate_descriptor = _load_registered_bundle('ST', seed, fold)
    x, ids, labels, splits = (arrays[k] for k in ('features', 'subject_ids', 'labels', 'splits'))
    modalities = ('S', 'T')
    if eeg != 'ST':
        z, emeta, eeg_sources, _ = _load_registered_bundle(eeg, seed, fold)
        V2.require(emeta['roles'] == stmeta['roles'], 'E/ST metadata role ordering differs')
        V2.require(np.array_equal(ids, z['subject_ids']) and np.array_equal(labels, z['labels']) and
                   np.array_equal(splits, z['splits']), 'E/ST people, labels or roles differ')
        x = np.concatenate([z['features'][:, None], x], axis=1)
        sources.extend(eeg_sources)
        modalities = ('E', 'S', 'T')
    roles = stmeta['roles']
    V2.require(set(roles) == {'train', 'val', 'test'}, 'Unknown input roles')
    flat = [sid for role in roles.values() for sid in role]
    V2.require(len(flat) == len(set(flat)) == len(ids) == 37 and set(flat) == set(ids.tolist()),
               'Roles must partition the 37 unique people')
    V2.require(set(labels.tolist()) == {0, 1} and int(labels.sum()) == 16, 'Cohort labels differ')
    lookup = {sid: i for i, sid in enumerate(ids.tolist())}
    indexes = {role: np.asarray([lookup[sid] for sid in people], dtype=np.int64)
               for role, people in roles.items()}
    V2.require(all(all(splits[j] == role for j in ix) for role, ix in indexes.items()), 'Role index differs')
    state = fit_preprocessor(x[indexes['train']], variant, roles['train'])
    normalized = apply_preprocessor(x, state)
    binding = {'source_gate': gate_descriptor, 'sources': sources, 'subject_ids': ids.tolist(), 'labels': labels.tolist(),
               'splits': splits.tolist(), 'roles': roles, 'modalities': list(modalities),
               'normalization': state,
               'input_distributions': {'raw': distribution_summary(x, indexes, modalities),
                                       'normalized': distribution_summary(normalized, indexes, modalities)}}
    return torch.as_tensor(normalized, device=device), torch.as_tensor(labels, device=device), indexes, binding
