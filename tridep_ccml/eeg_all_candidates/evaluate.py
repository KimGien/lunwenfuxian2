"""Release fixed-checkpoint outer tests only after all360 formal fits verify.

Each durable probability batch is bound to the global gate, exact checkpoint,
training source lock, role/person order and input hashes. No fitting, threshold
selection, encoder update, or EMA update is performed here.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean
import sys
import time
import uuid

from . import protocol as P, registry as R, storage as S

CONDITIONS = tuple(c.condition_id for c in P.CONDITIONS if c.objective == 'supervised')
ROOT = S.OUT / 'evaluation'
SOURCE = Path(__file__).resolve()
ROLES = ('train', 'val', 'test')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def predicted_class(probability):
    require(math.isfinite(probability) and 0 <= probability <= 1, 'Invalid MDD probability')
    return int(probability > .5)


def metrics(rows):
    """Fixed two-class MacroF1; AUC is pairwise ranking with half-credit ties."""
    require(rows and all(r['label'] in (0, 1) for r in rows), 'Binary labels and nonempty predictions required')
    confusion = [[0, 0], [0, 0]]
    for row in rows:
        confusion[row['label']][predicted_class(row['p_mdd'])] += 1
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    f1_hc = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    f1_mdd = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    positive = [r['p_mdd'] for r in rows if r['label'] == 1]
    negative = [r['p_mdd'] for r in rows if r['label'] == 0]
    recall = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    auc = (sum(float(p > n) + .5*float(p == n) for p in positive for n in negative)/len(positive)/len(negative)
           if positive and negative else None)
    nll = mean(-math.log(max(min(r['p_mdd'] if r['label'] else 1-r['p_mdd'], 1.), 1e-15)) for r in rows)
    return {'macro_f1': (f1_hc+f1_mdd)/2, 'balanced_accuracy': (recall+specificity)/2 if recall is not None and specificity is not None else None,
            'mdd_recall': recall, 'auc': auc, 'nll': nll,
            'brier': mean((r['p_mdd']-r['label'])**2 for r in rows), 'confusion_matrix': confusion, 'subjects': len(rows)}


def _mean_metrics(values):
    return {key: mean(v[key] for v in values) if all(v[key] is not None for v in values) else None
            for key in ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')}


def _gate():
    # The first operational action is the full registry gate. In particular,
    # no constructor, data loader, checkpoint load or forward precedes this.
    checked = R.assert_test_unlocked()
    require(checked['passed'] and checked['formal_fits_verified'] == 360, 'All360 source-bound completions required')
    stable = {k: checked[k] for k in ('passed', 'registration', 'formal_fits_verified', 'statuses')}
    path = ROOT / ('test_gate_' + canonical(stable) + '.json')
    S.save(path, stable, immutable=True)
    return S.desc(path)


def _check_sources(records):
    for record in records:
        S.checked(record)


def _sources_from_lock(condition):
    lock = R._condition_lock(condition)
    contract = S.read(S.checked(lock))['contract']
    return lock, contract['implementation_sources'] + contract['weight_sources']


def _input_plan(condition, metadata, registration):
    policy = P.CONDITION_BY_ID[condition].policy_key
    if policy == 'frozen':
        descriptor = metadata['feature_manifest']
        manifest = S.read(S.checked(descriptor))
        require(manifest['passed'] and manifest['condition'] == condition and manifest['encoder_fit_subject_ids'] == [], 'Fixed feature manifest differs')
        records = {r['subject_id']: r for r in manifest['records']}
        sources = [descriptor] + manifest.get('sources', []) + [r['features'] for r in manifest['records']]
        return {'kind': 'frozen', 'manifest': manifest, 'records': records, 'sources': sources}
    if policy == 'diagnostic':
        from tridep_ccml.tridep_eegpt import common as OLD
        descriptor = metadata['frozen_features']
        manifest = S.read(S.checked(descriptor))
        require(S.checked(descriptor) == OLD.OUT / 'features_manifest.json', 'Diagnostic feature source differs')
        records = {r['subject_id']: r for r in manifest['records']}
        key = 'eegpt' if condition.startswith('EEGPT') else 'labram'
        return {'kind': 'diagnostic', 'archive_key': key, 'records': records,
                'sources': [descriptor] + [r['features'] for r in manifest['records']]}
    records = {r['subject_id']: {'raw': r['raw']} for r in registration['raw_sources']}
    return {'kind': 'raw', 'records': records,
            'sources': [registration['sources']['accepted_raw_manifest']] + [r['raw'] for r in registration['raw_sources']]}


def _load_array(plan, person):
    import numpy as np
    record = plan['records'][person]
    key = plan.get('archive_key', record.get('archive_key'))
    descriptor = record['raw'] if plan['kind'] == 'raw' else record['features']
    path = S.checked(descriptor)
    if key:
        with np.load(path, allow_pickle=False) as bundle:
            value = bundle[key].copy()
    else:
        value = np.load(path, allow_pickle=False)
    expected = (60, 19, 1000) if plan['kind'] == 'raw' else (60, plan['manifest']['dimension'] if plan['kind'] == 'frozen' else 200)
    require(value.shape == expected and value.dtype == np.float32 and bool(np.isfinite(value).all()), 'Bound evaluation input shape/dtype changed')
    return value


def _materialize_predictor(condition, seed, fold, split, metadata, checkpoint, plan, snapshot):
    import torch
    from tridep_ccml.tridep_stage1 import common as C
    policy = P.CONDITION_BY_ID[condition].policy_key
    if condition == 'EEGMamba-lastblock-finetune' and sys.platform != 'linux':
        raise RuntimeError('Missing qualified Linux EEGMamba predictions; run Linux evaluate --conditions EEGMamba-lastblock-finetune --no-report first')
    if condition == 'EEGMamba-lastblock-finetune':
        from .linux_entry import _require_runtime
        _require_runtime()
    C.runtime(seed)
    saved = torch.load(S.checked(checkpoint), map_location='cpu', weights_only=False)
    require(isinstance(saved.get('state_dict'), dict) and all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Incomplete/nonfinite fixed checkpoint')
    if policy == 'diagnostic':
        from tridep_ccml.tridep_eegpt import train_eval as T
        require(saved['registry'] == R.registration_descriptor() and saved['epoch'] == snapshot and snapshot in P.SNAPSHOT_EPOCHS, 'Fixed diagnostic snapshot identity differs')
        model = T.original_model().to('cuda')
    elif policy == 'frozen':
        from .frozen_heads import SubjectHead
        require(saved['registry'] == R.registration_descriptor() and saved['epoch'] == metadata['best_epoch'], 'Frozen head checkpoint registry/selected epoch differs')
        model = SubjectHead(plan['manifest']['dimension']).to('cuda')
    elif policy == 'finetune':
        from . import finetune as F
        require(all(saved[k] == v for k, v in {'condition': condition, 'seed': seed, 'fold': fold, 'registry': R.registration_descriptor(),
                                              'condition_source_lock': metadata['condition_source_lock']}.items()), 'Fine-tuned checkpoint identity differs')
        require(saved['best_epoch'] == metadata['best_epoch'], 'Fine-tuned selected epoch differs')
        model, audit = F.build_model(condition, seed=seed, device='cuda', window_chunk=metadata['window_chunk'])
        require(audit['selected_encoder_parameter_names'] == metadata['adapter_audit']['selected_encoder_parameter_names'], 'Fine-tuned adapter parameter identity differs')
        if condition == 'EEGMamba-lastblock-finetune':
            for key in ('torch', 'mamba_ssm', 'causal_conv1d', 'triton'):
                require(audit['runtime'][key] == metadata['adapter_audit']['runtime'][key], 'EEGMamba runtime differs from completed training: ' + key)
    else:
        from . import supervised as U
        require(saved['identity'] == {k: metadata[k] for k in saved['identity']}, 'Supervised checkpoint identity differs')
        require(saved['best_epoch'] == metadata['best_epoch'], 'Supervised selected epoch differs')
        statistics = saved['statistics']
        if condition == 'EEGConformer-scratch':
            bound = S.read(S.checked(metadata['statistics']))
            require(bound['statistics'] == statistics and statistics['fit_subject_ids'] == split['train'], 'Fold-fitted statistics identity differs')
            require(bound['validation_or_test_fit_people'] == [], 'Normalizer used nontraining people')
        else:
            require(statistics is None, 'EEG2Rep must not introduce new fitted preprocessing')
        model, audit = U.build_model(condition, seed=seed, fold=fold, split=split, raw={}, device='cuda',
                                    window_chunk=metadata['window_chunk'], statistics=statistics)
        require(audit['ssl_binding'] == metadata['ssl_binding'], 'EEG2Rep same-fold SSL source binding differs')
        if condition == 'EEG2Rep-supervised':
            ssl_meta = S.read(S.checked(metadata['ssl_binding']['metadata']))
            ssl_saved = torch.load(S.checked(metadata['ssl_binding']['checkpoint']), map_location='cpu', weights_only=False)
            require(ssl_meta['online_backbone_update_verified'] is True and ssl_meta['online_changed_parameter_count'] > 0
                    and ssl_saved['online_state_sha256'] == ssl_meta['selected_online_state_sha256']
                    and ssl_meta['selected_online_state_sha256'] != ssl_meta['initial_online_state_sha256'], 'SSL online backbone update evidence differs')
    model.load_state_dict(saved['state_dict'], strict=True)
    if policy != 'diagnostic':
        require(C.state_digest(model.state_dict()) == metadata['selected_state_sha256'], 'Selected checkpoint fingerprint differs')
    elif saved['epoch'] == 300:
        require(C.state_digest(model.state_dict()) == metadata['final_state_sha256'], 'Fixed300 checkpoint fingerprint differs')
    model.requires_grad_(False).eval()
    return model


def _validate_rows(rows, people, labels, role):
    require([r['subject_id'] for r in rows] == people, 'Prediction person order/coverage differs')
    for row in rows:
        require(row['role'] == role and row['label'] == labels[row['subject_id']], 'Prediction role/label identity differs')
        require(len(row['logits']) == 2 and all(math.isfinite(x) for x in row['logits']), 'Invalid saved logits')
        gap = row['logits'][1]-row['logits'][0]
        expected = 1/(1+math.exp(-gap)) if gap >= 0 else math.exp(gap)/(1+math.exp(gap))
        require(abs(row['p_mdd']-expected) < 1e-12 and abs(row['p_hc']+row['p_mdd']-1.) < 1e-12
                and row['prediction'] == predicted_class(row['p_mdd']), 'Saved probability/threshold arithmetic differs')


def _validate_batch_value(value, binding):
    require(value['binding'] == binding, 'Completed probability batch belongs to another source/checkpoint/gate')
    fingerprints = [value.get('model_state_before'), value.get('model_state_after'), binding.get('selected_model_state_sha256')]
    require(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v) for v in fingerprints),
            'Inference batch/checkpoint state fingerprints must be64 lowercase hex characters')
    require(value.get('frozen_state_verified') is True and value.get('no_gradients_verified') is True
            and value['model_state_before'] == value['model_state_after'] == binding['selected_model_state_sha256'],
            'Returned inference batch lacks unchanged-state/checkpoint evidence')
    if binding['condition'] == 'EEGMamba-lastblock-finetune':
        runtime = value['runtime']
        require(runtime['platform'] == 'linux' and runtime['torch'] == '2.5.1+cu124'
                and runtime['mamba_ssm'] == '2.2.4' and runtime['causal_conv1d'] == '1.5.0.post8'
                and runtime['triton'] == '3.1.0', 'Mamba prediction bundle was not made in the qualified official Linux runtime')


def _batch_cached(path, binding, people, labels, role):
    if not path.exists():
        # Recover complete serialized atomic-write remnants without repeating
        # a returned test forward. Conflicting remnants are refused.
        candidates = []
        for pending in path.parent.glob(path.name + '.*.part'):
            try:
                value = S.read(pending)
                if value.get('binding') == binding:
                    _validate_batch_value(value, binding)
                    _validate_rows(value['rows'], people, labels, role)
                    candidates.append(value)
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
        if candidates:
            require(all(canonical(v) == canonical(candidates[0]) for v in candidates), 'Conflicting saved probability remnants require inspection')
            S.save(path, candidates[0], immutable=True)
    if not path.exists():
        return None
    value = S.read(path)
    _validate_batch_value(value, binding)
    _validate_rows(value['rows'], people, labels, role)
    return value


def _fit_context(condition, seed, fold, snapshot, gate, cohort, split, registration):
    folder = R.fit_directory(condition, seed, fold)
    status = R.fit_status(condition, seed, fold)
    evidence = status['evidence']
    metadata = S.read(S.checked(evidence['metadata']))
    require(metadata['condition'] == condition and metadata['seed'] == seed and metadata['fold'] == fold
            and metadata['train_ids'] == split['train'] and metadata['val_ids'] == split['val'], 'Training role identity differs')
    lock, source_records = _sources_from_lock(condition)
    checkpoint = metadata['snapshots'][str(snapshot)] if snapshot != 'best' else evidence['checkpoint']
    S.checked(checkpoint)
    if snapshot != 'best':
        # Header validation is CPU-only and also protects cached snapshot reuse.
        import torch
        from tridep_ccml.tridep_stage1 import common as C
        saved = torch.load(S.checked(checkpoint), map_location='cpu', weights_only=False)
        require(snapshot in P.SNAPSHOT_EPOCHS and saved['epoch'] == snapshot
                and saved['registry'] == R.registration_descriptor(), 'Requested diagnostic snapshot/header differs')
        selected_signature = C.state_digest(saved['state_dict'])
    else:
        selected_signature = metadata['selected_state_sha256']
    plan = _input_plan(condition, metadata, registration)
    require(set(plan['records']) == set(cohort['subject_ids']), 'Evaluation input cohort differs')
    _check_sources(plan['sources'] + source_records)
    ordered = {role: split[role] for role in ROLES}
    binding = {'condition': condition, 'seed': seed, 'fold': fold, 'snapshot': snapshot, 'test_gate': gate,
               'registration': R.registration_descriptor(), 'checkpoint': checkpoint, 'metadata': evidence['metadata'],
               'completion': S.desc(folder / 'complete.json'), 'condition_source_lock': lock,
               'evaluator_source': S.desc(SOURCE), 'roles': ordered, 'ordered_cohort': cohort['subject_ids'],
               'selected_model_state_sha256': selected_signature,
               'input_sources': plan['sources'], 'training_sources': source_records,
               'checkpoint_selection': 'fixed_diagnostic_epoch' if snapshot != 'best' else 'training_validation_selected_best',
               'test_selection_or_preprocessing_fit': False}
    destination = ROOT / condition / f'seed_{seed}' / f'fold_{fold}' / ('epoch_' + str(snapshot) if snapshot != 'best' else 'best')
    return metadata, plan, binding, destination


def _verified_completed_fit(complete_path, binding, split, labels):
    existing = S.read(complete_path)
    require(existing['passed'] is True and existing['binding'] == binding, 'Evaluation completion source binding changed')
    _check_sources(existing['attempts'])
    all_rows = []
    for record in existing['batches']:
        value = S.read(S.checked(record))
        _validate_batch_value(value, binding)
        all_rows.extend(value['rows'])
    for role in ROLES:
        _validate_rows([r for r in all_rows if r['role'] == role], split[role], labels, role)
    recomputed = {role: metrics([r for r in all_rows if r['role'] == role]) for role in ROLES}
    require(recomputed == existing['metrics'], 'Saved evaluation metrics differ from bound probabilities')
    require(existing['all60_windows_per_subject'] is True and existing['optimizer_updates'] == existing['ema_updates'] == 0,
            'Prediction package violated no-update/all60-window contract')
    if binding['condition'] == 'EEGMamba-lastblock-finetune':
        terminal = [S.read(S.checked(a)) for a in existing['attempts']]
        require(any(a.get('status') == 'complete' and a.get('runtime', {}).get('platform') == 'linux'
                    and a.get('terminal_state_verified') is True for a in terminal), 'Mamba bundle lacks a successful qualified Linux terminal ledger')
    return existing, all_rows


def _evaluate_fit(condition, seed, fold, snapshot, gate, cohort, split, labels, registration):
    metadata, plan, binding, destination = _fit_context(condition, seed, fold, snapshot, gate, cohort, split, registration)
    checkpoint = binding['checkpoint']
    ordered = binding['roles']
    complete_path = destination / 'complete.json'
    if complete_path.exists():
        return _verified_completed_fit(complete_path, binding, split, labels)
    if condition == 'EEGMamba-lastblock-finetune' and sys.platform != 'linux':
        raise RuntimeError('A complete bound Linux EEGMamba prediction package is required before Windows aggregation')
    batch_size = 1 if plan['kind'] == 'raw' else (16 if plan['kind'] == 'diagnostic' else 8)
    batches = [(role, index, people[start:start+batch_size]) for role in ROLES for index, start in enumerate(range(0, len(ordered[role]), batch_size))
               for people in [ordered[role]]]
    cached = { (role, index): _batch_cached(destination / f'{role}_batch_{index}.json', binding, people, labels, role)
              for role, index, people in batches }
    missing = any(value is None for value in cached.values())
    model = None
    attempt_path = destination / 'attempts' / (str(time.time_ns()) + '_' + uuid.uuid4().hex + '.json')
    ledger = {'binding': binding, 'started_at': S.now(), 'status': 'running', 'outer_forward_started': 0, 'outer_forward_returned': 0,
              'encoder_forward_started': 0, 'encoder_forward_returned': 0, 'encoder_windows_started': 0, 'encoder_windows_returned': 0,
              'prediction_people_started': 0, 'prediction_people_returned': 0, 'formal_fits': 0, 'optimizer_updates': 0, 'ema_updates': 0}
    started = time.monotonic()
    adapter_original = None
    try:
        if missing:
            S.save(attempt_path, ledger, immutable=True)
            model = _materialize_predictor(condition, seed, fold, split, metadata, checkpoint, plan, snapshot)
            import torch
            from tridep_ccml.tridep_stage1 import common as C
            state_before = C.state_digest(model.state_dict())
            require(state_before == binding['selected_model_state_sha256'], 'Fixed predictor state differs from bound selected checkpoint')
            runtime = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'platform': sys.platform}
            if condition == 'EEGMamba-lastblock-finetune':
                import importlib.metadata
                runtime.update({name: importlib.metadata.version(name) for name in ('mamba_ssm', 'causal_conv1d', 'triton')})
            if plan['kind'] == 'raw':
                adapter_original = model.adapter.encode_windows
                def observe_encoder(encoder, tensor):
                    ledger['encoder_forward_started'] += 1
                    ledger['encoder_windows_started'] += len(tensor)
                    S.save(attempt_path, ledger)
                    out = adapter_original(encoder, tensor)
                    ledger['encoder_forward_returned'] += 1
                    ledger['encoder_windows_returned'] += len(tensor)
                    S.save(attempt_path, ledger)
                    return out
                model.adapter.encode_windows = observe_encoder
            require(not any(module.training for module in model.modules()) and not any(p.requires_grad for p in model.parameters()), 'Evaluation model is not frozen/eval')
        all_rows, descriptors = [], []
        for role, index, people in batches:
            path = destination / f'{role}_batch_{index}.json'
            cached_value = cached[(role, index)]
            rows = cached_value['rows'] if cached_value else None
            if cached_value is None:
                import numpy as np
                import torch
                values = np.stack([_load_array(plan, person) for person in people])
                ledger['outer_forward_started'] += 1
                ledger['prediction_people_started'] += len(people)
                S.save(attempt_path, ledger)
                with torch.inference_mode():
                    logits = model(values if plan['kind'] == 'raw' else torch.as_tensor(values, device='cuda'))
                require(tuple(logits.shape) == (len(people), 2) and bool(torch.isfinite(logits).all()), 'Invalid binary subject logits')
                logits = logits.detach().double().cpu().tolist()
                rows = []
                for person, pair in zip(people, logits):
                    gap = pair[1]-pair[0]
                    probability = 1/(1+math.exp(-gap)) if gap >= 0 else math.exp(gap)/(1+math.exp(gap))
                    rows.append({'subject_id': person, 'role': role, 'label': labels[person], 'logits': pair,
                                 'p_mdd': probability, 'p_hc': 1-probability, 'prediction': predicted_class(probability)})
                ledger['outer_forward_returned'] += 1
                ledger['prediction_people_returned'] += len(people)
                S.save(attempt_path, ledger)
                after = C.state_digest(model.state_dict())
                require(after == state_before and all(p.grad is None for p in model.parameters()), 'Inference batch mutated state or created gradients')
                S.save(path, {'binding': binding, 'rows': rows, 'runtime': runtime,
                              'model_state_before': state_before, 'model_state_after': after,
                              'frozen_state_verified': True, 'no_gradients_verified': True}, immutable=True)
            _validate_rows(rows, people, labels, role)
            all_rows.extend(rows)
            descriptors.append(S.desc(path))
        if model is not None:
            require(C.state_digest(model.state_dict()) == state_before and all(p.grad is None for p in model.parameters()), 'Evaluation mutated state or created gradients')
            ledger.update(status='complete', finished_at=S.now(), seconds=time.monotonic()-started,
                          runtime=runtime, terminal_state_verified=True,
                          terminal_state_sha256=state_before)
            S.save(attempt_path, ledger)
        elif condition == 'EEGMamba-lastblock-finetune':
            from .linux_entry import _require_runtime
            _require_runtime()
            runtime = next(iter(cached.values()))['runtime']
            ledger.update(status='complete', finished_at=S.now(), seconds=time.monotonic()-started,
                          runtime=runtime, terminal_state_verified=True, terminal_state_sha256=binding['selected_model_state_sha256'],
                          cache_finalize_only=True, terminal_verification_basis='all_verified_cached_batches')
            S.save(attempt_path, ledger, immutable=True)
        result = {'passed': True, 'binding': binding, 'batches': descriptors,
                  'metrics': {role: metrics([r for r in all_rows if r['role'] == role]) for role in ROLES},
                  'attempts': [S.desc(p) for p in sorted((destination / 'attempts').glob('*.json'))],
                  'all60_windows_per_subject': True, 'optimizer_updates': 0, 'ema_updates': 0}
        S.save(complete_path, result, immutable=True)
        # Apply the same postcondition used by Linux/Windows reuse to the actual
        # durable completion, including recovered terminal ledgers.
        return _verified_completed_fit(complete_path, binding, split, labels)
    except BaseException as exc:
        if missing:
            ledger.update(status='failed', failure_type=type(exc).__name__, failure=str(exc), finished_at=S.now(), seconds=time.monotonic()-started)
            S.save(attempt_path, ledger)
        raise
    finally:
        if adapter_original is not None and model is not None:
            model.adapter.encode_windows = adapter_original
        if model is not None:
            del model
            gc.collect()
            import torch
            torch.cuda.empty_cache()


def condition_summary(entries, cohort_ids):
    """Primary = five-fold arithmetic mean, then three-seed arithmetic mean."""
    by_seed = {}
    for seed in P.SEEDS:
        chosen = [e for e in entries if e['seed'] == seed]
        require(sorted(e['fold'] for e in chosen) == list(P.FOLDS), 'Five folds required for eachseed')
        oof = [row for entry in chosen for row in entry['rows'] if row['role'] == 'test']
        require(len(oof) == len(cohort_ids) and {r['subject_id'] for r in oof} == set(cohort_ids), 'One OOF prediction per37 people required perseed')
        fold_metrics = [entry['result']['metrics']['test'] for entry in sorted(chosen, key=lambda e: e['fold'])]
        by_seed[str(seed)] = {'fold_mean': _mean_metrics(fold_metrics), 'oof_metrics': metrics(oof), 'fold_metrics': fold_metrics,
                             'fold_role_metrics': [{'fold': e['fold'], 'metrics': e['result']['metrics']} for e in sorted(chosen, key=lambda e: e['fold'])]}
    return {'primary': _mean_metrics([v['fold_mean'] for v in by_seed.values()]), 'seeds': by_seed,
            'oof_predictions': 111, 'distinct_subjects': 37, 'repeated_seed_predictions_are_independent_people': False,
            'roles': {role: _mean_metrics([_mean_metrics([e['result']['metrics'][role] for e in entries if e['seed'] == seed]) for seed in P.SEEDS]) for role in ROLES}}


def matched_control(condition):
    spec = P.CONDITION_BY_ID[condition]
    if spec.control:
        return None
    if spec.policy_key == 'diagnostic':
        return P.DIAGNOSTIC_CONDITIONS[1]
    suffix = 'lastblock-finetune' if spec.policy_key == 'finetune' else 'frozen-smallhead'
    return f'LaBraM-native{spec.input_seconds}s-{suffix}'


def retention(candidate, control):
    c, b = candidate['primary'], control['primary']
    improvements = sum(candidate['seeds'][str(s)]['fold_mean']['macro_f1'] > control['seeds'][str(s)]['fold_mean']['macro_f1'] for s in P.SEEDS)
    delta = {key: c[key]-b[key] if c[key] is not None and b[key] is not None else None for key in c}
    passed = (delta['macro_f1'] >= .01 and improvements >= 2 and delta['balanced_accuracy'] >= 0 and delta['mdd_recall'] >= -.05)
    return {'passed': passed, 'strictly_improved_seeds': improvements, 'deltas': delta,
            'probability_benefit': delta['nll'] < 0 and delta['brier'] < 0, 'not_a_significance_test': True}


def _write_csv(path, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    S.write(path, lambda stream: stream.write(buffer.getvalue().encode('utf-8-sig')))


def _comparison_contract(condition, entries):
    policy = P.CONDITION_BY_ID[condition].policy_key
    selected = [e for e in entries if e['condition'] == condition and e['snapshot'] == (300 if policy == 'diagnostic' else 'best')]
    metadata = [S.read(S.checked(e['result']['binding']['metadata'])) for e in selected]
    first = metadata[0]
    audit = first.get('adapter_audit', {})
    contract = audit.get('input_contract', {})
    if policy == 'frozen':
        manifest = S.read(S.checked(first['feature_manifest']))
        dimension, head = first['dimension'], first['parameters']
        contract = manifest.get('input_contract', manifest)
    elif policy == 'diagnostic':
        dimension, head = 200, first['parameters']
        contract = {'reference': 'unchanged accepted E2v2 common reference',
                    'amplitude': 'identical historical4s cache; EEGPT fixed512to200 projection' if condition.startswith('EEGPT') else 'identical historical4s cache; LaBraM nativeD200 feature'}
    else:
        dimension = audit.get('native_dimension', audit.get('feature_dim'))
        head = 130*dimension+386  # Exact registered LayerNorm/Linear128/classifier2 parameter count.
    return {'native_dimension': dimension, 'head_parameters': head,
            'reference': contract.get('reference', audit.get('reference', 'see condition input source lock')),
            'amplitude': contract.get('amplitude', contract.get('scale', audit.get('amplitude', 'see condition input source lock'))),
            'normalization': contract.get('normalization', contract.get('new_statistical_normalization', 'see condition input source lock')),
            'train_windows_per_subject': 8 if policy in ('scratch', 'ssl_supervised', 'finetune') else 60,
            'eval_windows_per_subject': 60, 'epochs_min': min(m['epochs'] for m in metadata), 'epochs_max': max(m['epochs'] for m in metadata),
            'optimizer_updates_total': sum(m.get('optimizer_updates', m.get('updates', 0)) for m in metadata),
            'input_contract_source': selected[0]['result']['binding']['condition_source_lock']['path'],
            'reproduction_scope': 'author-public-implementation adaptation; full paper unavailable' if P.CONDITION_BY_ID[condition].method == 'EEGMamba' else 'registered local implementation adaptation'}


def report(all_entries, gate, cohort_ids):
    summaries, comparison, oof_rows = {}, [], []
    for condition in CONDITIONS:
        snapshots = P.SNAPSHOT_EPOCHS if P.CONDITION_BY_ID[condition].policy_key == 'diagnostic' else ('best',)
        variants = {}
        for snapshot in snapshots:
            entries = [e for e in all_entries if e['condition'] == condition and e['snapshot'] == snapshot]
            variants[str(snapshot)] = condition_summary(entries, cohort_ids)
            for entry in entries:
                oof_rows.extend({'condition': condition, 'snapshot': snapshot, 'seed': entry['seed'], 'fold': entry['fold'], **r}
                                for r in entry['rows'] if r['role'] == 'test')
        primary_key = '300' if '300' in variants else 'best'
        summaries[condition] = {'primary_snapshot': primary_key, 'variants': variants}
    for condition, summary in summaries.items():
        control_id = matched_control(condition)
        primary = summary['variants'][summary['primary_snapshot']]
        control = summaries[control_id]['variants'][summaries[control_id]['primary_snapshot']] if control_id else None
        criterion = retention(primary, control) if control else None
        summary.update(matched_control=control_id, engineering_retention=criterion)
        if summary['primary_snapshot'] == '300':
            summary['predefined_duration_comparison'] = {'primary': 'epoch300 minus epoch60', 'deltas': {k: primary['primary'][k]-summary['variants']['60']['primary'][k] for k in primary['primary']},
                                                         'secondary_snapshots': [120, 200], 'test_best_snapshot_selected': False}
        spec = P.CONDITION_BY_ID[condition]
        comparison.append({'condition': condition, 'method': spec.method, 'route': spec.route, 'seconds_per_window': spec.input_seconds,
                           'primary_snapshot': summary['primary_snapshot'], 'matched_control': control_id or '',
                           **_comparison_contract(condition, all_entries), **primary['primary'],
                           'macro_f1_delta': criterion['deltas']['macro_f1'] if criterion else '',
                           'retention_passed': criterion['passed'] if criterion else '', 'probability_benefit': criterion['probability_benefit'] if criterion else ''})
    papers = {}
    for method in P.METHODS:
        routes = [c.condition_id for c in P.CONDITIONS if c.method == method and c.objective == 'supervised' and c.policy_key != 'diagnostic']
        passes = [c for c in routes if summaries[c]['engineering_retention']['passed']]
        papers[method] = {'status': '改善' if passes else '未改善', 'all_prespecified_routes_completed': True,
                          'routes': routes, 'passing_routes': passes, 'negative_results_count_as_completed': True,
                          'scope': '作者公开实现适配实验，未取得全文，不称完整论文复现' if method == 'EEGMamba' else '本项目预先登记的输入与训练适配'}
    require(len(all_entries) == 435, '345 supervised fits and all4 diagnostic snapshots require435 evaluated checkpoints')
    attempt_records = {d['path']: d for e in all_entries for d in e['result']['attempts']}
    _check_sources(list(attempt_records.values()))
    attempts = [S.read(S.checked(d)) for d in attempt_records.values()]
    costs = {key: sum(a.get(key, 0) for a in attempts) for key in ('outer_forward_started', 'outer_forward_returned',
             'encoder_forward_started', 'encoder_forward_returned', 'encoder_windows_started', 'encoder_windows_returned',
             'prediction_people_started', 'prediction_people_returned')}
    costs.update(attempts=len(attempts), seconds=sum(a.get('seconds', 0.) for a in attempts), formal_fits=0, optimizer_updates=0, ema_updates=0)
    value = {'passed': True, 'test_gate': gate, 'formal_fits_verified': 360, 'supervised_conditions': 23,
             'supervised_fits_evaluated': 345, 'evaluated_checkpoints': 435, 'candidate_papers_count': 9,
             'distinct_people': 37, 'seed_repeats_not_new_people': True, 'threshold': .5, 'tie': 'HC',
             'nll_probability_floor': 1e-15, 'primary_aggregation': 'mean5folds then mean3seeds',
             'conditions': summaries, 'nine_papers': papers, 'test_guided_model_or_threshold_selection': False,
             'probability_batch_artifacts': [S.desc(e['path']) for e in all_entries], 'execution_attempts': list(attempt_records.values()), 'execution_costs': costs}
    S.save(ROOT / 'evaluation_report.json', value, immutable=True)
    _write_csv(S.REPORT / 'unified_comparison.csv', comparison)
    _write_csv(S.REPORT / 'oof_probabilities.csv', [{**r, 'logits': json.dumps(r['logits'])} for r in oof_rows])
    lines = ['# 九篇 EEG 单模态统一完成报告', '', '360 次正式训练全部通过来源、权重和完成证据核验后，才进行了本轮固定外层测试。', '',
             '主要分数为每个种子先平均五折，再平均三个种子；另保留每种子 37 人 OOF 指标。每条件 111 条种子预测只来自 37 人。阈值固定为 MDD 概率 > 0.5，相等为 HC。', '',
             '| 方法 | 结论 | 全部预定路线 |', '|---|---|---|']
    lines.extend(f"| {method} | {result['status']} | 完成 |" for method, result in papers.items())
    lines += ['', 'EEGMamba 是作者公开实现适配实验；公开全文未取得，不称为完整论文复现。', '',
              '“改善”仅表示至少一条预先登记路线达到匹配控制的工程保留门槛，不是统计显著性结论。阴性结果全部保留。', '',
              'EEGPT 与匹配 4 秒 LaBraM 的纯轮次主要比较固定第 300 轮与第 60 轮；120/200 轮仅作次要观察，没有选择测试分数最高的快照。', '',
              '逐条件比较、train/val/test 指标和混淆矩阵在 evaluation/evaluation_report.json；OOF 概率及固定控制差值见本目录 CSV。参考、幅值、窗口与训练范式差异仍以各条件输入合同及训练 metadata 为准。']
    S.write(S.REPORT / '九候选统一完成报告.md', lambda stream: stream.write(('\n'.join(lines)+'\n').encode('utf-8')))
    S.save(ROOT / 'complete.json', {'passed': True, 'test_gate': gate, 'formal_fits_verified': 360,
           'supervised_conditions': 23, 'supervised_fits_evaluated': 345, 'evaluated_checkpoints': 435,
           'candidate_papers_count': 9, 'evaluation_report': S.desc(ROOT / 'evaluation_report.json'),
           'comparison_table': S.desc(S.REPORT / 'unified_comparison.csv'), 'oof_probabilities': S.desc(S.REPORT / 'oof_probabilities.csv'),
           'nine_paper_report': S.desc(S.REPORT / '九候选统一完成报告.md'), 'evaluator_source': S.desc(SOURCE)}, immutable=True)
    return value


def _completed_condition_verified(condition, gate, registration):
    require(condition in CONDITIONS, 'Registered supervised condition required')
    cohort = {'subject_ids': registration['subject_ids']}
    entries, artifacts = [], []
    snapshots = P.SNAPSHOT_EPOCHS if P.CONDITION_BY_ID[condition].policy_key == 'diagnostic' else ('best',)
    for seed in P.SEEDS:
        for fold in P.FOLDS:
            split = registration['folds'][f'fold_{fold}']
            for snapshot in snapshots:
                _, _, binding, destination = _fit_context(condition, seed, fold, snapshot, gate, cohort, split, registration)
                path = destination / 'complete.json'
                require(path.exists(), 'Evaluation bundle incomplete: ' + str(path))
                result, rows = _verified_completed_fit(path, binding, split, registration['labels'])
                entries.append({'seed': seed, 'fold': fold, 'snapshot': snapshot, 'rows': rows, 'result': result})
                artifacts.append(S.desc(path))
    primary = [e for e in entries if e['snapshot'] == (300 if P.CONDITION_BY_ID[condition].policy_key == 'diagnostic' else 'best')]
    condition_summary(primary, cohort['subject_ids'])
    return {'passed': True, 'condition': condition, 'test_gate': gate, 'supervised_fits_evaluated': 15,
            'primary_oof_predictions': 111, 'distinct_subjects': 37, 'evaluated_checkpoints': len(entries),
            'evaluation_fit_artifacts': artifacts, 'model_constructions': 0, 'gpu_forward_calls': 0}


def completed_condition_verified(condition):
    """Verify15fold bundles with no model construction or GPU.

    Full360 gate is first. Diagnostic headers/fingerprints are deserialized on
    CPU only. Windows verifies Linux Mamba packages without its encoder.
    """
    gate = _gate()
    return _completed_condition_verified(condition, gate, R.load_registration())


def completed_report_verified():
    """Queue postcondition: validate all23 bundles and all final report hashes."""
    gate = _gate()
    path = ROOT / 'complete.json'
    value = S.read(path)
    require(value['passed'] is True and value['test_gate'] == gate and value['formal_fits_verified'] == 360
            and value['supervised_conditions'] == 23 and value['supervised_fits_evaluated'] == 345
            and value['evaluated_checkpoints'] == 435 and value['candidate_papers_count'] == 9, 'Global evaluation completion arithmetic/source differs')
    _check_sources([value[k] for k in ('evaluation_report', 'comparison_table', 'oof_probabilities', 'nine_paper_report', 'evaluator_source')])
    data = S.read(S.checked(value['evaluation_report']))
    require(data['passed'] is True and set(data['conditions']) == set(CONDITIONS) and len(data['probability_batch_artifacts']) == 435,
            'All23 conditions and435 evaluated checkpoint artifacts required')
    _check_sources(data['probability_batch_artifacts'] + data['execution_attempts'])
    registration = R.load_registration()
    conditions = [_completed_condition_verified(c, gate, registration) for c in CONDITIONS]
    return {'passed': True, 'complete_manifest': S.desc(path), 'supervised_conditions': len(conditions),
            'supervised_fits_evaluated': 345, 'formal_fits_verified': 360, 'gpu_forward_calls': 0, 'model_constructions': 0}


def run(conditions=CONDITIONS, *, no_report=False):
    gate = _gate()
    require(conditions and len(set(conditions)) == len(conditions) and all(c in CONDITIONS for c in conditions), 'Registered unique supervised conditions required')
    if not no_report:
        require(set(conditions) == set(CONDITIONS), 'Global report requires all23 supervised conditions; use --no-report for staged Linux export')
    from tridep_ccml.tridep_stage1 import common as C
    cohort, folds, labels = C.context()
    registration = R.load_registration()
    entries = []
    for condition in conditions:
        snapshots = P.SNAPSHOT_EPOCHS if P.CONDITION_BY_ID[condition].policy_key == 'diagnostic' else ('best',)
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                for snapshot in snapshots:
                    result, rows = _evaluate_fit(condition, seed, fold, snapshot, gate, cohort, folds[f'fold_{fold}'], labels, registration)
                    name = 'epoch_' + str(snapshot) if snapshot != 'best' else 'best'
                    path = ROOT / condition / f'seed_{seed}' / f'fold_{fold}' / name / 'complete.json'
                    entries.append({'condition': condition, 'seed': seed, 'fold': fold, 'snapshot': snapshot, 'result': result, 'rows': rows, 'path': path})
                    print(f'EVALUATE {condition} seed={seed} fold={fold} snapshot={snapshot} probabilities_saved', flush=True)
    if no_report:
        manifests = []
        for condition in conditions:
            chosen = [e for e in entries if e['condition'] == condition]
            primary = [e for e in chosen if e['snapshot'] == ('best' if P.CONDITION_BY_ID[condition].policy_key != 'diagnostic' else 300)]
            condition_summary(primary, cohort['subject_ids'])  # Verify15folds/111OOF/37people without releasing global comparisons.
            path = ROOT / condition / 'predictions_manifest.json'
            S.save(path, {'passed': True, 'condition': condition, 'test_gate': gate, 'registration': R.registration_descriptor(),
                          'supervised_fits_evaluated': 15, 'primary_oof_predictions': 111, 'distinct_subjects': 37,
                          'evaluation_fit_artifacts': [S.desc(e['path']) for e in chosen], 'evaluator_source': S.desc(SOURCE),
                          'global_report_generated': False}, immutable=True)
            manifests.append(S.desc(path))
        return {'passed': True, 'staged_conditions': list(conditions), 'test_gate': gate, 'predictions_manifests': manifests}
    return report(entries, gate, cohort['subject_ids'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--no-report', action='store_true', help='Save bound predictions for staged conditions; do not produce global comparisons')
    args = parser.parse_args()
    run(args.conditions, no_report=args.no_report)


if __name__ == '__main__':
    main()
