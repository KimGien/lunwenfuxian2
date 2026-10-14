"""Review all15 CSBrain FT completions using JSON/source and an existing CPU proof.

No torch/numpy/project imports, tensor/weight/raw/feature reads or evaluation.
Incomplete identities return pending before the supplied CPU proof is opened.
Run without arguments while pending. After all15 complete, supply --cpu-audit
with the genuine source-bound audit_completed_finetune_checkpoints_v3 report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time

WORK = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
CONDITION = 'CSBrain-lastblock-finetune'
SEEDS = (11, 22, 33)
REGISTRY_SHA = '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'
AUDITOR_SHA = '307d6feabb9b4964db335fd9a0390e3026015cd57f6205117de125c79508c6d7'
AUTHOR_SHA = 'cfcdf9dcb069905e5e02e92fe3d59bd3574e779bf554cc02cb701994a79e99cc'
COUNTERS = ('forward', 'encoder_forward', 'encoder_windows', 'encode', 'backward', 'optimizer_step')
EXPECTED_IDS = {f'{CONDITION}/seed_{seed}/fold_{fold}' for seed in SEEDS for fold in range(1, 6)}


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class Reader:
    """Every hash is of the same bytes decoded; binary artifact paths forbidden."""
    def __init__(self):
        self.inputs = {}

    def snapshot(self, path, *, bind=True):
        path = Path(path).resolve()
        require(path.is_relative_to(WORK) and path.suffix in ('.json', '.py'),
                'Only workspace JSON and Python source may be read')
        for attempt in range(4):
            try:
                raw = path.read_bytes()
                value = json.loads(raw.decode('utf-8-sig')) if path.suffix == '.json' else raw.decode('utf-8-sig')
                break
            except (PermissionError, FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
                if attempt == 3:
                    raise
                time.sleep((.02, .04, .08)[attempt])
        descriptor = {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}
        if bind:
            previous = self.inputs.get(descriptor['path'])
            require(previous is None or previous == descriptor, 'Bound input changed during review: ' + descriptor['path'])
            self.inputs[descriptor['path']] = descriptor
        return value, descriptor

    def checked(self, descriptor):
        require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256'}, 'Expected a JSON/source path+SHA descriptor')
        value, actual = self.snapshot(WORK / descriptor['path'])
        require(actual == descriptor, 'Bound JSON/source hash differs: ' + descriptor['path'])
        return value

    def recheck(self):
        for descriptor in list(self.inputs.values()):
            self.checked(descriptor)


def pending(reason, registry, statuses, details=None):
    return {'status': 'pending', 'passed': False, 'reason': reason, 'registration': registry,
            'CSBrain_registered': 15, 'CSBrain_completed_observed': sum(x['status'] == 'training_complete' for x in statuses),
            'identity_status_observations': statuses, 'details': details,
            'actual_cpu_report_read': False, 'full_135_ft_completion_claimed': False,
            'full_360_plan_completion_claimed': False}


def gate(reader):
    registration, binding = reader.snapshot(OUT / 'registration.json')
    require(binding['sha256'] == REGISTRY_SHA and registration['formal_fits'] == 360
            and len(registration['fits']) == 360, 'Fixed full360 registration differs')
    fits = [f for f in registration['fits'] if f['condition'] == CONDITION]
    require(len(fits) == 15 and {f['fit_id'] for f in fits} == EXPECTED_IDS, 'Expected exactly CSBrain3seeds x5fold identities')
    statuses, values = [], {}
    for fit in fits:
        path = OUT / 'runs' / fit['fit_id'] / 'status.json'
        if path.exists():
            value, descriptor = reader.snapshot(path)
            require((value['condition'], value['seed'], value['fold'], value['fit_id']) ==
                    (CONDITION, fit['seed'], fit['fold'], fit['fit_id']), 'Actual status identity differs')
            require(value['registration'] == binding, 'Actual status registration differs')
            values[fit['fit_id']] = (value, descriptor)
            state = value['status']
        else:
            state, descriptor = 'engineering_pending', None
        statuses.append({'fit_id': fit['fit_id'], 'status': state, 'artifact': descriptor})
    return registration, binding, fits, values, statuses


def decision(policy, losses, extended):
    best, best_loss = 1, losses[0]
    for epoch, loss in enumerate(losses[1:], 2):
        if best_loss - loss >= policy['minimum_improvement']:
            best, best_loss = epoch, loss
    epoch = len(losses)
    cap = policy['final_epochs'] if extended else policy['initial_epochs']
    require(epoch <= cap, 'Curve exceeds registered cap')
    result = {'best_epoch': best, 'stop': False, 'extend': False, 'reason': 'continue'}
    if epoch >= policy['minimum_stop_epoch'] and epoch - best >= policy['patience']:
        result.update(stop=True, reason='validation_patience')
    elif epoch == cap:
        previous, recent = mean(losses[-40:-20]), mean(losses[-20:])
        improvement = (previous - recent) / previous if previous > 0 else 0.
        eligible = (not extended and policy['final_epochs'] > policy['initial_epochs']
                    and best > .8 * policy['initial_epochs'] and improvement >= .01)
        result.update(stop=not eligible, extend=eligible, reason='registered_extension' if eligible else 'epoch_cap')
    return result


def lr_multiplier(policy, epoch, backbone=False):
    start = policy['warmup_epochs'] if backbone else 0
    warmup = policy['backbone_warmup_epochs'] if backbone else policy['warmup_epochs']
    if epoch <= start:
        return 0.
    if epoch <= start + warmup:
        return (epoch - start) / warmup
    if epoch >= policy['initial_epochs']:
        return .1
    fraction = (epoch - start - warmup) / (policy['initial_epochs'] - start - warmup)
    return .1 + .9 * .5 * (1. + math.cos(math.pi * fraction))


def verify_curve(fit, metadata, history, validation):
    epochs, policy = metadata['epochs'], fit['policy']
    require(integer(epochs) and epochs > 10 and len(history) == len(validation) == epochs, 'Incomplete actual curves')
    require(policy['warmup_epochs'] == 10 and policy['minimum_stop_epoch'] == 40 and policy['patience'] == 30
            and policy['initial_epochs'] == 120 and policy['final_epochs'] == 200, 'Registered CSBrain FT policy differs')
    ntrain = len(fit['train_ids'])
    losses, extended, extension_epochs = [], False, []
    for epoch, (row, val) in enumerate(zip(history, validation), 1):
        require(row['epoch'] == val['epoch'] == epoch and all(finite(row[k]) for k in ('train_bag_ce', 'train_eval_ce', 'val_eval_ce'))
                and val['val_ce'] == row['val_eval_ce'], 'Actual validation curve differs')
        require(row['optimizer_updates'] == math.ceil(ntrain / 8) * epoch and row['backward_calls'] == ntrain * epoch
                and row['unfrozen'] is (epoch > 10) and row['training_module_subtrees'] == (['backbone.encoder.layers.11'] if epoch > 10 else []), 'Epoch counters/phase differ')
        require(row['window_indices'] == [(8 * ((epoch - 1) % 8) + j) % 60 for j in range(8)]
                and row['cycle_epoch'] == (epoch - 1) % 8 + 1 and row['cycle'] == (epoch - 1) // 8 + 1
                and row['repeated_padding_windows_this_epoch'] == ([0, 1, 2, 3] if epoch % 8 == 0 else []), 'Registered window cycle differs')
        for group, base in (('head', policy['head_lr']), ('backbone', policy['backbone_lr'])):
            require(math.isclose(row[group + '_lr'], base * lr_multiplier(policy, epoch, group == 'backbone'), rel_tol=1e-12, abs_tol=1e-16), 'Registered LR schedule differs')
        losses.append(row['val_eval_ce'])
        last = decision(policy, losses, extended)
        require(not last['stop'] or epoch == epochs, 'Continued after registered stop')
        if last['extend']:
            require(not extended and epoch == 120 and not extension_epochs, 'Extension must occur once at registered120')
            extension_epochs.append(epoch)
            extended = True
    require(last['stop'] is True and metadata['best_epoch'] == last['best_epoch']
            and metadata['stop_reason'] == last['reason'] and metadata['extended'] is extended, 'Actual selection/stop/extension differs')
    return {**last, 'extension_epochs': extension_epochs, 'extended': extended,
            'selection_inclusive_warmup_last_epoch': 10}


def inventory(reader):
    formal, engineering = defaultdict(list), []
    for path in sorted((OUT / 'finetune_v2_attempts' / CONDITION).glob('*/attempt.json')):
        data, descriptor = reader.snapshot(path)
        require(data.get('condition') == CONDITION and data.get('kind') in ('formal', 'engineering'), 'Unknown F2 attempt cannot be omitted')
        require(data.get('status') in ('complete', 'failed'), 'CSBrain F2 trace is still mutable or unknown')
        if data['kind'] == 'engineering':
            require(data['seed'] is None and data['fold'] is None, 'Engineering trace identity differs')
            engineering.append((data, descriptor))
        else:
            require(type(data['seed']) is int and data['seed'] in SEEDS and type(data['fold']) is int and data['fold'] in range(1, 6), 'Unknown formal attempt identity')
            formal[f'{CONDITION}/seed_{data["seed"]}/fold_{data["fold"]}'].append((data, descriptor))
    require(set(formal) == EXPECTED_IDS, 'Some completed CSBrain fit lacks F2 traces')
    return formal, engineering


def verify_traces(fit, metadata, boundary, traces, implementation):
    seen, unique, aliases = {}, [], []
    totals = Counter()
    for data, descriptor in traces:
        require(data['implementation'] == implementation and finite(data.get('seconds')) and data.get('finished_at'), 'Attempt source/elapsed observation differs')
        for key in COUNTERS:
            started, returned = data[key + '_started'], data[key + '_returned']
            require(integer(started) and integer(returned) and returned <= started, 'Invalid physical trace counter')
            require(key == 'encoder_windows' or started - returned <= 1, 'Serial trace has multiple unreturned calls')
            if data['status'] == 'complete':
                require(started == returned, 'Returned complete attempt has an unfinished call')
        require(data['encode_started'] == data['encoder_forward_started'] and data['encode_returned'] == data['encoder_forward_returned'], 'Encoder aliases differ')
        if descriptor['sha256'] in seen:
            aliases.append({'artifact': descriptor, 'alias_of': seen[descriptor['sha256']], 'cost_counted_again': False})
            continue
        seen[descriptor['sha256']] = descriptor
        unique.append((data, descriptor))
        totals.update({key + suffix: data[key + suffix] for key in COUNTERS for suffix in ('_started', '_returned')})
    require([v for _, v in unique] == [r['artifact'] for r in boundary['attempts']]
            and aliases == boundary['duplicate_content_aliases'] and dict(totals) == boundary['recorded_physical_call_subtotals'], 'CPU proof trace inventory/subtotals differ')
    require(any(d['status'] == 'complete' for d, _ in unique) and totals['optimizer_step_returned'] >= metadata['optimizer_updates']
            and totals['backward_returned'] >= metadata['backward_calls'], 'Observed attempts do not cover durable training')
    require(boundary['durable_optimizer_updates_counted_once'] == metadata['optimizer_updates']
            and boundary['durable_backward_calls_counted_once'] == metadata['backward_calls'], 'Durable boundary differs')
    for key in ('physical_and_durable_counts_added_together', 'nested_metadata_seconds_added_to_attempt_seconds', 'legacy_failure_counters_added_to_f2'):
        require(boundary[key] is False, 'Known overlapping scopes were added')
    require(boundary['physical_optimizer_calls_exact'] is None and boundary['physical_backward_calls_exact'] is None
            and boundary['all_historical_wall_seconds_exact'] is None, 'Unknown history cannot become an exact total')
    # A pristine one-attempt trajectory permits stronger independent call arithmetic.
    exact_formula_checked = len(unique) == 1 and unique[0][0]['status'] == 'complete'
    if exact_formula_checked:
        n, v, epoch = len(fit['train_ids']), len(fit['val_ids']), metadata['epochs']
        expected = {'forward': epoch * (2 * n + v) + 2, 'encoder_forward': epoch * (n + (n + v) * 8) + 16,
                    'encode': epoch * (n + (n + v) * 8) + 16, 'encoder_windows': epoch * (n * 8 + (n + v) * 60) + 120,
                    'backward': epoch * n, 'optimizer_step': epoch * math.ceil(n / 8)}
        require(all(totals[key + '_returned'] == value for key, value in expected.items()), 'One-attempt physical call arithmetic differs')
    return {'attempts': [{'artifact': item, 'status': data['status'], 'seconds': data['seconds']} for data, item in unique],
            'duplicate_aliases_not_added': aliases, 'recorded_physical_call_subtotals': dict(totals),
            'recorded_attempt_seconds_known_sum': sum(data['seconds'] for data, _ in unique),
            'single_complete_attempt_math_checked': exact_formula_checked}


def review_fit(reader, fit, row, registration, registry, actual_status, traces):
    metadata = reader.checked(row['metadata'])
    complete = reader.checked(row['completion'])
    status = reader.checked(row['status_artifact'])
    lock = reader.checked(row['condition_source_lock'])
    contract = lock['contract']
    require(status == actual_status and status['evidence'] == complete and status['status'] == 'training_complete', 'CPU proof/current actual completion differs')
    require(row['passed'] is True and row['training_json_verified'] is True and row['fit_id'] == fit['fit_id'], 'CPU proof fit differs')
    require((row['condition'], row['seed'], row['fold']) == (CONDITION, fit['seed'], fit['fold']), 'CPU proof explicit identity differs')
    require(complete['registry'] == status['registration'] == lock['registration'] == registry and complete['metadata'] == row['metadata'], 'Completion source binding differs')
    require(metadata['condition_source_lock'] == status['condition_source_lock'] == row['condition_source_lock'], 'Actual engineering source lock differs')
    require(reader.checked(status['event_record']) == {k: v for k, v in status.items() if k != 'event_record'}, 'Actual status event differs')
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(lock['configuration_sha256'] == canonical, 'Engineering contract checksum differs')
    for value in contract['implementation_sources']:
        reader.checked(value)
    require((metadata['condition'], metadata['seed'], metadata['fold']) == (CONDITION, fit['seed'], fit['fold']), 'Metadata identity differs')
    for role in ('train', 'val', 'test'):
        require(metadata[role + '_ids'] == fit[role + '_ids'] == registration['folds']['fold_' + str(fit['fold'])][role], 'Original role/order differs')
    raw = {x['subject_id']: x['raw'] for x in registration['raw_sources']}
    require(set(metadata['raw_sources']) == set(fit['train_ids'] + fit['val_ids'])
            and not set(metadata['raw_sources']) & set(fit['test_ids'])
            and all(value == raw[sid] for sid, value in metadata['raw_sources'].items()), 'Nontraining/test raw descriptor mismatch')
    require(metadata['test_forward_count'] == 0 and metadata['reload_probe_subject_ids'] == fit['train_ids'][:1], 'Test/reload probe roles differ')
    require(metadata['fresh_official_pretrained'] is True and metadata['fresh_head'] is True and metadata['frozen_best_head_inherited'] is False, 'Fresh initializer claim differs')
    start, _ = reader.snapshot(OUT / 'runs' / fit['fit_id'] / 'started.json')
    require(start['fresh_formal_pretrained'] is True and start['fresh_head_seed'] == fit['seed'] and start['frozen_best_head_inherited'] is False, 'Actual epoch-zero fresh start differs')
    require((start['condition'], start['seed'], start['fold']) == (CONDITION, fit['seed'], fit['fold'])
            and start['train_ids'] == fit['train_ids'] and start['val_ids'] == fit['val_ids']
            and start['condition_source_lock'] == row['condition_source_lock'] and start['test_forward_count'] == 0
            and start['raw_sources'] == metadata['raw_sources'], 'Started identity/source/role boundary differs')
    adapter, official = metadata['adapter_audit'], contract['input_contract']['adapter']
    require(adapter['source'] == official['source'] and adapter['source']['weight']['sha256'] == AUTHOR_SHA
            and adapter['formal_pretrained_strict_loaded'] is official['formal_pretrained_strict_loaded'] is True
            and adapter['loaded_backbone_keys'] == official['loaded_backbone_keys'] == 295
            and adapter['source']['weight'] in contract['weight_sources']
            and adapter['input_contract'] == official['input_contract'], 'Official CSBrain adapter binding differs')
    for value in adapter['implementation_sources']:
        reader.checked(value)
    if fit['seed'] == 11:
        require(metadata['initial_state_sha256'] == contract['initial_state_sha256'], 'Seed11 initialization differs from sealed engineering initializer')
    implementation = metadata['versioned_finetune_implementation']
    require(implementation == contract['versioned_finetune_implementation'], 'Actual sealed F2 source provenance differs')
    for key in ('wrapper_source', 'delegated_runner_source', 'luna_adapter_source'):
        reader.checked(implementation[key])
    history, validation = reader.checked(complete['train_history']), reader.checked(complete['validation_history'])
    policy_decision = verify_curve(fit, metadata, history, validation)
    epochs, updates, backwards = metadata['epochs'], metadata['optimizer_updates'], metadata['backward_calls']
    require(metadata['micro_subjects'] == 1 and metadata['window_chunk'] == metadata['effective_subject_batch'] == metadata['window_bag_size'] == 8
            and metadata['eval_window_count'] == 60 and metadata['precision'] == 'float32', 'Actual registered input/batch/precision differs')
    require(updates == metadata['head_optimizer_updates'] == math.ceil(len(fit['train_ids']) / 8) * epochs
            and backwards == len(fit['train_ids']) * epochs and metadata['backbone_optimizer_updates'] == math.ceil(len(fit['train_ids']) / 8) * (epochs - 10), 'Actual group/durable counters differ')
    require(row['epochs'] == complete['epochs'] == epochs and row['updates'] == complete['optimizer_updates'] == updates
            and row['best_epoch'] == metadata['best_epoch'] and row['stop'] == complete['stop_reason'] == metadata['stop_reason'], 'CPU row/completion counters differ')
    active = contract['trainable_parameter_names']
    enc, heads = [n for n in active if n.startswith('encoder.')], [n for n in active if n.startswith('head.')]
    require(len(enc) == 20 and len(heads) == 6 and len(contract['frozen_parameter_names']) == 273
            and all(n.startswith('encoder.backbone.encoder.layers.11.') for n in enc) and set(metadata['parameter_changed_after_training']) == set(active)
            and all(metadata['parameter_changed_after_training'].values()), 'Terminal intended parameter updates differ')
    phase = 'head_only_warmup' if metadata['best_epoch'] <= 10 else 'lastblock_finetune'
    require(metadata['selected_checkpoint_phase'] == phase and metadata['terminal_encoder_changed_parameter_names'] == enc
            and metadata['terminal_head_changed_parameter_names'] == heads and metadata['terminal_encoder_changed_parameter_count'] == 20
            and metadata['terminal_head_changed_parameter_count'] == 6, 'Selected/terminal phase disclosure differs')
    require(metadata['selected_encoder_changed_parameter_count'] == len(metadata['selected_encoder_changed_parameter_names'])
            and set(metadata['selected_encoder_changed_parameter_names']).issubset(enc)
            and metadata['selected_head_changed_parameter_count'] == len(metadata['selected_head_changed_parameter_names'])
            and set(metadata['selected_head_changed_parameter_names']).issubset(heads), 'Selected change identity/count differs')
    if phase == 'head_only_warmup':
        require(metadata['selected_encoder_changed_parameter_count'] == 0
                and metadata['selected_encoder_initial_state_sha256'] == metadata['selected_encoder_state_sha256'], 'Warmup selected encoder must remain initial')
    require(metadata['selected_encoder_parameters_changed'] is bool(metadata['selected_encoder_changed_parameter_count']),
            'Selected encoder change boolean differs from recorded changes')
    require(metadata['frozen_initial_sha256'] == metadata['frozen_final_sha256']
            and metadata['terminal_updates_are_not_claimed_as_selected_encoder_updates'] is True, 'Frozen/selected update boundary differs')
    proof = row['cpu_checkpoint_audit']
    require(proof['cpu_tensors_verified'] is True and proof['checkpoint_loads'] == 3 and proof['only_map_location_cpu'] is True
            and proof['model_constructions'] == proof['gpu_operations'] == 0, 'Genuine CPU tensor proof differs')
    checkpoints = {Path(v['path']).name: v for v in proof['checkpoints']}
    require(set(checkpoints) == {'initial.pt', 'latest.pt', 'best.pt'} and checkpoints['best.pt'] == complete['checkpoint'], 'CPU checkpoint descriptor set differs')
    bounds = {x['path']: x for x in proof['durable_counter_boundaries']}
    require((bounds['initial.pt']['epoch'], bounds['initial.pt']['optimizer_updates'], bounds['initial.pt']['backward_calls']) == (0, 0, 0)
            and (bounds['latest.pt']['epoch'], bounds['latest.pt']['optimizer_updates'], bounds['latest.pt']['backward_calls']) == (epochs, updates, backwards)
            and bounds['best.pt']['epoch'] == metadata['best_epoch'] and bounds['best.pt']['optimizer_updates'] is None
            and bounds['best.pt']['backward_calls'] is None, 'Existing CPU scalar boundary differs')
    require(bounds['latest.pt']['optimizer_groups']['group_names'] == ['head', 'backbone']
            and bounds['latest.pt']['optimizer_groups']['moment_shapes_verified'] is True
            and bounds['latest.pt']['optimizer_groups']['serialized_id_parameter_names'] == heads + enc
            and bounds['latest.pt']['optimizer_groups']['head_parameters'] == 6
            and bounds['latest.pt']['optimizer_groups']['backbone_parameters'] == 20, 'Actual optimizer group proof differs')
    physical = verify_traces(fit, metadata, row['f2_physical_durable_boundary'], traces, implementation)
    failures = []
    for path in sorted((OUT / 'runs' / fit['fit_id']).glob('failure_*.json')):
        data, binding = reader.snapshot(path)
        failures.append({'artifact': binding, 'supplemental_only_not_added_to_F2': True, 'observed_fields': data})
    cpu_failures = row['legacy_failure_records_not_added']
    require({x['artifact']['path'] for x in failures} == {x['artifact']['path'] for x in cpu_failures}, 'Per-fit failure inventory differs from CPU proof')
    for failure in cpu_failures:
        reader.checked(failure['artifact'])
    require(finite(metadata.get('seconds_this_attempt')) and metadata['seconds_this_attempt'] > 0
            and integer(metadata.get('peak_allocated_bytes')), 'Completed metadata resource observations invalid')
    if physical['single_complete_attempt_math_checked']:
        require(physical['recorded_attempt_seconds_known_sum'] >= metadata['seconds_this_attempt'], 'Nested duration exceeds enclosing single formal attempt')
    return {'fit_id': fit['fit_id'], 'seed': fit['seed'], 'fold': fit['fold'], 'metadata': row['metadata'], 'completion': row['completion'],
            'epochs': epochs, 'best_epoch': metadata['best_epoch'], 'stop': metadata['stop_reason'], 'extended': metadata['extended'],
            'policy_decision': policy_decision, 'frozen_parameter_names': 273,
            'selected_phase': phase, 'selected_encoder_changed': metadata['selected_encoder_changed_parameter_count'],
            'selected_head_changed': metadata['selected_head_changed_parameter_count'], 'terminal_encoder_changed': 20, 'terminal_head_changed': 6,
            'durable_optimizer_updates': updates, 'head_group_updates': updates, 'backbone_group_updates': metadata['backbone_optimizer_updates'],
            'durable_backward_calls': backwards, 'initial_full_state_sha256': metadata['initial_state_sha256'],
            'initial_encoder_state_sha256': metadata['selected_encoder_initial_state_sha256'],
            'frozen_initial_state_sha256': metadata['frozen_initial_sha256'], 'official_checkpoint_descriptor_not_read': adapter['source']['weight'],
            'CPU_verified_checkpoint_descriptors_not_reopened': proof['checkpoints'], 'physical_attempts': physical,
            'legacy_failure_supplementary_records': failures, 'last_success_metadata_seconds_not_added_to_attempt_seconds': metadata.get('seconds_this_attempt'),
            'peak_allocated_bytes': metadata.get('peak_allocated_bytes')}


def engineering_summary(reader, traces, contract, implementation, original):
    """Keep engineering attempts separate and content aliases uncounted twice."""
    require(contract['training_only_engineering_checked'] is True
            and contract['probe_subject_ids'] == original['fold_1']['train'][:1]
            and contract['engineering_optimizer_updates'] == 0 and contract['engineering_backward_calls'] == 2,
            'Sealed training-only engineering contract differs')
    required_path = (WORK / contract['engineering_attempt_path']).resolve().relative_to(WORK).as_posix()
    selected = [(data, descriptor) for data, descriptor in traces if descriptor['path'] == required_path]
    require(len(selected) == 1, 'Sealed successful engineering attempt is missing')
    data, _ = selected[0]
    require(data['status'] == 'complete' and data['optimizer_step_started'] == data['optimizer_step_returned'] == 0
            and data['backward_started'] == data['backward_returned'] == 2, 'Sealed engineering successful counters differ')
    seen, records, aliases = {}, [], []
    for data, descriptor in traces:
        require(data['implementation'] == implementation and finite(data.get('seconds')) and data.get('finished_at'),
                'Engineering source/duration differs')
        for key in COUNTERS:
            started, returned = data[key + '_started'], data[key + '_returned']
            require(integer(started) and integer(returned) and returned <= started, 'Invalid engineering physical counter')
            if data['status'] == 'complete':
                require(started == returned, 'Completed engineering attempt has unreturned call')
        if descriptor['sha256'] in seen:
            aliases.append({'artifact': descriptor, 'alias_of': seen[descriptor['sha256']], 'cost_counted_again': False})
            continue
        seen[descriptor['sha256']] = descriptor
        records.append({'artifact': descriptor, 'status': data['status'], 'seconds': data['seconds'],
                        'counters': {base + suffix: data[base + suffix] for base in COUNTERS for suffix in ('_started', '_returned')}})
    return {'unique_attempts': records, 'duplicate_content_aliases': aliases,
            'known_seconds_separate_from_formal': sum(x['seconds'] for x in records)}


def review(cpu_path=None, check_only=False):
    reader = Reader()
    registration, registry, fits, actual, statuses = gate(reader)
    if any(x['status'] != 'training_complete' for x in statuses):
        return pending('All15 actual CSBrain training completions are required before reading CPU proof or detailed fits', registry, statuses), reader
    if check_only or cpu_path is None:
        return pending('Actual all15 status gate reached; supply the real reviewed v3 CPU audit for formal stage verification', registry, statuses), reader
    audit, audit_binding = reader.snapshot(cpu_path)
    require(audit['schema_version'] == 3 and audit['auditor_source']['sha256'] == AUDITOR_SHA
            and Path(audit['auditor_source']['path']).name == 'audit_completed_finetune_checkpoints_v3.py', 'Only the reviewed actual v3 CPU audit is accepted')
    reader.checked(audit['auditor_source'])
    for value in audit['delegated_sources'].values():
        reader.checked(value)
    require(audit['registration'] == registry and audit['status'] in ('partial_verified', 'all_finetune_verified')
            and not audit['errors'] and not audit['deferred_completed_fits']
            and audit['registered_finetune_fits'] == 135 and audit['registered_total_fits'] == 360, 'CPU audit scope/errors differ')
    require(audit['model_constructions'] == audit['gpu_operations'] == 0 and audit['raw_arrays_loaded'] is False
            and audit['outer_prediction_artifacts_read'] is False and audit['test_gate_called'] is False
            and audit['evaluation_driver_called'] is False, 'Supplied CPU audit has the wrong operation scope')
    selected = [r for r in audit['completed_rows'] if r['condition'] == CONDITION]
    require(len(selected) == 15 and {r['fit_id'] for r in selected} == EXPECTED_IDS, 'CPU proof does not verify all15 distinct CSBrain fits')
    require(audit['verified_completed_finetune_fits'] == len(audit['completed_rows']) == 30
            and audit['checkpoint_loads'] == 90 and len(audit['pending_fits']) == 105
            and Counter(r['condition'] for r in audit['completed_rows']) == {CONDITION: 15, 'EEGPT-lastblock-finetune': 15},
            'This stage requires the real30-fit snapshot: EEGPT15 plus CSBrain15,90 CPU loads')
    cohort = reader.checked(registration['sources']['cohort'])
    original = reader.checked(registration['sources']['folds'])
    copy = reader.checked(registration['sources']['folds_copy'])
    official38 = reader.checked(registration['sources']['official38_folds'])
    labels = reader.checked(registration['sources']['labels'])
    require(registration['subject_ids'] == cohort['subject_ids'] and len(set(cohort['subject_ids'])) == 37
            and registration['folds'] == original == copy, 'Original37 cohort/fold order differs')
    require(original == {f: {role: [sid for sid in ids if sid != '02010004'] for role, ids in split.items()} for f, split in official38.items()}, 'Official38 exclusion changed other split roles')
    require(registration['labels'] == {r['subject_id']: r['label'] for r in labels if r['subject_id'] in cohort['subject_ids']}
            and Counter(registration['labels'].values()) == {0: 21, 1: 16}, '37-person class labels differ')
    require(Counter(sid for split in original.values() for sid in split['test']) == Counter(cohort['subject_ids']), 'Outer-test folds do not partition37 once')
    for split in original.values():
        require(sum(len(ids) for ids in split.values()) == len(set().union(*(set(ids) for ids in split.values()))) == 37, 'Fold roles overlap or omit people')
    for fit in registration['fits']:
        require(all(fit[role + '_ids'] == original['fold_' + str(fit['fold'])][role] for role in ('train', 'val', 'test')),
                'A registered condition/seed changed original role identities')
    formal, engineering = inventory(reader)
    by_id = {r['fit_id']: r for r in selected}
    rows = [review_fit(reader, fit, by_id[fit['fit_id']], registration, registry, actual[fit['fit_id']][0], formal[fit['fit_id']]) for fit in fits]
    initializers = {}
    for seed in SEEDS:
        values = {x['initial_full_state_sha256'] for x in rows if x['seed'] == seed}
        require(len(values) == 1, 'Same-seed fresh initializer differs across folds')
        initializers[str(seed)] = next(iter(values))
    require(len(set(initializers.values())) == 3 and len({x['initial_encoder_state_sha256'] for x in rows}) == 1
            and len({x['frozen_initial_state_sha256'] for x in rows}) == 1,
            'Three fresh head seeds/common official encoder initialization differs')
    lock_bindings = [by_id[fit['fit_id']]['condition_source_lock'] for fit in fits]
    require(all(value == lock_bindings[0] for value in lock_bindings), 'CSBrain shared condition lock differs')
    contract = reader.checked(lock_bindings[0])['contract']
    engineering_costs = engineering_summary(reader, engineering, contract, contract['versioned_finetune_implementation'], original)
    physical_totals = Counter()
    for row in rows:
        physical_totals.update(row['physical_attempts']['recorded_physical_call_subtotals'])
    summaries = []
    for seed in SEEDS:
        group = [x for x in rows if x['seed'] == seed]
        summaries.append({'seed': seed, 'fits': 5, 'epochs': [x['epochs'] for x in group], 'best_epochs': [x['best_epoch'] for x in group],
                          'selected_phases': dict(Counter(x['selected_phase'] for x in group)),
                          **{key: sum(x[key] for x in group) for key in ('durable_optimizer_updates', 'head_group_updates', 'backbone_group_updates', 'durable_backward_calls')}})
    # Match the complete local condition inventory again, excluding other running conditions.
    formal_after, engineering_after = inventory(reader)
    require({k: [d for _, d in v] for k, v in formal_after.items()} == {k: [d for _, d in v] for k, v in formal.items()}
            and [d for _, d in engineering_after] == [d for _, d in engineering], 'CSBrain attempt inventory changed')
    reader.recheck()
    return {'status': 'verified_CSBrain_15_stage_only', 'passed': True, 'registration': registry, 'actual_cpu_audit': audit_binding,
            'actual_cpu_auditor_source': audit['auditor_source'], 'CSBrain_completed': 15, 'CSBrain_registered': 15,
            'stage_FT_fits_reviewed': 15, 'all_FT_registered': 135, 'full_135_ft_completion_claimed': False, 'full_360_plan_completion_claimed': False,
            'actual_CPU_snapshot_completed_FT': 30, 'actual_CPU_snapshot_scope': '30/135, EEGPT15 plus CSBrain15',
            'actual_CPU_snapshot_loads': 90, 'CSBrain_existing_CPU_loads': 45,
            'scope': 'Only CSBrain3seeds x5fold, backed by existing real CPU tensor proof; this helper reads JSON/source only.',
            'cohort': {'people': 37, 'HC': 21, 'MDD': 16, 'original_folds_roles_order_preserved': True, 'raw_descriptors_compared_only': True},
            'fresh_initializers': initializers, 'official_encoder_shared_across_seeds': True,
            'initial_tensor_evidence_basis': 'Referenced genuine v3 CPU audit; no new checkpoint reads. Same-seed full initializer digests equal, three head-seed full digests differ, official encoder digests shared.',
            'seeds': summaries, 'fits': rows, 'selected_phase_counts': dict(Counter(x['selected_phase'] for x in rows)),
            'selection_disclosure': 'Best epoch<=10 is inclusive head-only warmup with initial encoder; best epoch>10 is last-block phase. Report selected parameter changes separately from all20 terminal encoder and6 head changes. No post-warmup restriction is imposed on validation selection.',
            'extension_disclosure': 'Per fit, validation-only registered policy allows at most one extension at120 and a final cap200; actual extension decisions and stop are recomputed from complete curves.',
            'terminal_encoder_changed_parameter_counts': dict(Counter(x['terminal_encoder_changed'] for x in rows)),
            'durable_optimizer_updates': sum(x['durable_optimizer_updates'] for x in rows),
            'durable_backward_calls': sum(x['durable_backward_calls'] for x in rows),
            'formal_recorded_physical_call_subtotals': dict(physical_totals),
            'formal_attempt_seconds_known_sum': sum(x['physical_attempts']['recorded_attempt_seconds_known_sum'] for x in rows),
            'last_success_metadata_seconds_known_sum_not_added': sum(x['last_success_metadata_seconds_not_added_to_attempt_seconds'] for x in rows),
            'peak_allocated_bytes_max': max(x['peak_allocated_bytes'] for x in rows),
            'engineering_attempts_separate': engineering_costs,
            'cost_limits': {'group_steps_overlap_one_optimizer_call': True, 'durable_plus_physical_not_added': True,
                'metadata_plus_attempt_seconds_not_added': True, 'engineering_not_added_to_formal': True, 'peak_bytes_are_max_not_sum': True,
                'historical_exact_physical_total': None, 'historical_exact_wall_seconds': None,
                'scope': 'All matching recorded CSBrain attempts/aliases and failure side evidence preserved; no all-plan zero-failure or exact unrecorded-history claim.'},
            'verified_JSON_and_source_descriptors': list(reader.inputs.values()), 'actual_cpu_report_read': True}, reader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-audit', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Only inspect the15 identity statuses; never open CPU report')
    parser.add_argument('--require-complete', action='store_true', help='Exit2 if the stage is pending')
    args = parser.parse_args(argv)
    value, reader = review(args.cpu_audit, args.check_only)
    _, source = reader.snapshot(Path(__file__))
    value.update(kind='CSBrain_full_finetune_15_readonly_stage_review', created_at=datetime.now(timezone.utc).isoformat(),
                 review_source=source, operations={'checkpoint_binary_reads': 0, 'tensor_loads': 0, 'torch_imports': 0,
                 'model_constructions': 0, 'GPU': 0, 'raw_features': 0, 'test_probabilities': 0, 'evaluation': 0,
                 'training_process_actions': 0, 'sealed_or_historical_files_modified': False})
    if not value['passed']:
        value['verified_JSON_and_source_descriptors'] = list(reader.inputs.values())
    prefix = 'review_csbrain_full_finetune_15_passed_' if value['passed'] else 'review_csbrain_full_finetune_15_pending_'
    path = REPORT / (prefix + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    _, binding = reader.snapshot(path)
    print(json.dumps({'status': value['status'], 'passed': value['passed'], 'artifact': binding,
                      'CSBrain_completed': value.get('CSBrain_completed', value.get('CSBrain_completed_observed'))}, ensure_ascii=True))
    return 0 if value['passed'] or not args.require_complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
