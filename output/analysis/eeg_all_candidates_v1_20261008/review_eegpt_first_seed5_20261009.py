"""Independent first seed/five-fold review. Stdlib JSON/source reads only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

WORK = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
CONDITION = 'EEGPT-lastblock-finetune'
AUDIT = REPORT / 'completed_finetune_checkpoint_audit_v2_1791513474513739600.json'
AUDIT_SHA = 'fe03b9f185756f040aea4bc419bd4efd6e3afb1b305254af6209c717e2310fa8'
INPUTS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def desc(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix in ('.json', '.py', '.md'),
            'This review may read only workspace JSON and source text')
    value = {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    INPUTS[value['path']] = value
    return value


def read(path):
    path = Path(path)
    desc(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def checked(value):
    path = WORK / value['path']
    require(desc(path) == {k: value[k] for k in ('path', 'sha256')}, 'Bound JSON/source changed: ' + value['path'])
    return path


def main():
    require(desc(AUDIT)['sha256'] == AUDIT_SHA, 'Unexpected CPU audit bytes')
    audit = read(AUDIT)
    require(audit['status'] == 'partial_verified' and audit['verified_completed_finetune_fits'] == 5
            and audit['registered_finetune_fits'] == 135 and audit['checkpoint_loads'] == 15
            and audit['errors'] == [] and audit['full_360_plan_completion_claimed'] is False,
            'Actual five-fit CPU proof scope differs')
    for value in [audit['auditor_source'], *audit['delegated_sources'].values()]:
        checked(value)
    registration = read(checked(audit['registration']))
    require(registration['formal_fits'] == 360 and len(registration['fits']) == 360, 'Full identity registration differs')
    cohort = read(checked(registration['sources']['cohort']))
    original = read(checked(registration['sources']['folds']))
    copied = read(checked(registration['sources']['folds_copy']))
    official38 = read(checked(registration['sources']['official38_folds']))
    labels = read(checked(registration['sources']['labels']))
    people = registration['subject_ids']
    require(len(people) == len(set(people)) == 37 and people == cohort['subject_ids'], 'Original 37-person cohort differs')
    require(registration['folds'] == original == copied, 'Exact original fold identities/order differ')
    expected37 = {f: {role: [sid for sid in ids if sid != '02010004'] for role, ids in split.items()} for f, split in official38.items()}
    require(expected37 == original, 'Official38 removal changed other identities/order')
    require(registration['labels'] == {r['subject_id']: r['label'] for r in labels if r['subject_id'] in people}, 'Labels differ')
    require(Counter(registration['labels'].values()) == {0: 21, 1: 16}, 'Class counts differ')
    require(Counter(sid for split in original.values() for sid in split['test']) == Counter(people), 'Five test folds do not partition the 37 people')
    for fit in registration['fits']:
        require(all(fit[role + '_ids'] == original['fold_' + str(fit['fold'])][role] for role in ('train', 'val', 'test')),
                'Registered condition/seed uses different folds')
    lock_path = OUT / 'engineering' / (CONDITION + '.json')
    lock = read(lock_path)
    contract = lock['contract']
    canonical = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(lock['configuration_sha256'] == canonical and lock['registration'] == audit['registration'], 'Engineering contract/source binding differs')
    for value in contract['implementation_sources']:
        checked(value)
    require(len(contract['trainable_parameter_names']) == 18 and contract['training_only_engineering_checked'] is True, 'Exact head/last-block contract differs')
    bound_author = contract['input_contract']['adapter']
    author_weight = {k: bound_author['checkpoint'][k] for k in ('path', 'sha256')}
    require(author_weight in contract['weight_sources'] and author_weight['sha256'] == '9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0', 'Official weight identity differs')
    rows, traces, failures = [], [], []
    raw_descriptors = {row['subject_id']: row['raw'] for row in registration['raw_sources']}
    require(len(raw_descriptors) == 37, 'Registered raw descriptor identities differ')
    cpu_rows = {r['fold']: r for r in audit['completed_rows']}
    require(set(cpu_rows) == set(range(1, 6)), 'CPU proof is not exactly first seed five folds')
    for fold in range(1, 6):
        fit = next(r for r in registration['fits'] if r['condition'] == CONDITION and r['seed'] == 11 and r['fold'] == fold)
        folder = OUT / 'runs' / fit['fit_id']
        cpu = cpu_rows[fold]
        require(cpu['fit_id'] == fit['fit_id'] and cpu['seed'] == 11 and cpu['passed'] is True, 'CPU proof fit identity differs')
        metadata = read(checked(cpu['metadata']))
        status = read(checked(cpu['status_artifact']))
        complete = read(checked(cpu['completion']))
        require(status['status'] == 'training_complete' and status['test_evaluation_status'] == 'frozen' and status['evidence'] == complete, 'Actual completed identity differs')
        event = read(checked(status['event_record']))
        require(event == {k: v for k, v in status.items() if k != 'event_record'}, 'Status event binding differs')
        require(complete['metadata'] == cpu['metadata'] and status['registration'] == audit['registration']
                and metadata['condition_source_lock'] == desc(lock_path) == cpu['condition_source_lock'], 'Completion/source lock differs')
        history = read(checked(complete['train_history']))
        validation = read(checked(complete['validation_history']))
        started = read(folder / 'started.json')
        split = original['fold_' + str(fold)]
        require(all(metadata[role + '_ids'] == fit[role + '_ids'] == split[role] for role in ('train', 'val', 'test')), 'Actual fit role/order differs')
        require(sum(len(ids) for ids in split.values()) == len(set().union(*(set(ids) for ids in split.values()))) == 37, 'Roles overlap or omit a person')
        require(set(metadata['raw_sources']) == set(split['train'] + split['val']) and not set(metadata['raw_sources']) & set(split['test']), 'Raw descriptors include test people')
        require(all(value == raw_descriptors[sid] for sid, value in metadata['raw_sources'].items()), 'Raw descriptor differs from registration')
        require(metadata['test_forward_count'] == started['test_forward_count'] == 0 and metadata['reload_probe_subject_ids'] == split['train'][:1], 'Nontraining reload or test forward reported')
        require(started['fresh_formal_pretrained'] is True and started['fresh_head_seed'] == 11 and started['frozen_best_head_inherited'] is False,
                'Actual epoch-zero start was not fresh')
        require(metadata['fresh_official_pretrained'] is True and metadata['fresh_head'] is True and metadata['frozen_best_head_inherited'] is False, 'Fresh initializer differs')
        adapter = metadata['adapter_audit']
        require(adapter['checkpoint'] == bound_author['checkpoint'] and adapter['encoder_state_sha256'] == bound_author['encoder_state_sha256']
                and adapter['encoder_keys'] == adapter['loaded_encoder_keys'] == 102 and adapter['missing_encoder_keys'] == adapter['unexpected_encoder_keys'] == []
                and adapter['fresh_head_seed'] == 11 and adapter['initialized_from_frozen_best_head'] is False, 'Actual adapter/official initialization binding differs')
        require(metadata['versioned_finetune_implementation'] == contract['versioned_finetune_implementation'], 'Actual F2 implementation differs')
        epochs, best_epoch, ntrain = metadata['epochs'], metadata['best_epoch'], len(split['train'])
        require(len(history) == len(validation) == epochs and metadata['micro_subjects'] == 1, 'Actual curve/microbatch differs')
        best, loss = 1, history[0]['val_eval_ce']
        for epoch, (h, v) in enumerate(zip(history, validation), 1):
            if loss - h['val_eval_ce'] >= fit['policy']['minimum_improvement']:
                best, loss = epoch, h['val_eval_ce']
            should_stop = epoch >= 40 and epoch - best >= 30
            require((not should_stop) or epoch == epochs, 'Actual curve continued past registered stop')
            windows = [(8 * ((epoch - 1) % 8) + j) % 60 for j in range(8)]
            require(h['epoch'] == v['epoch'] == epoch and h['val_eval_ce'] == v['val_ce']
                    and h['window_indices'] == windows and h['optimizer_updates'] == math.ceil(ntrain / 8) * epoch
                    and h['backward_calls'] == ntrain * epoch and h['unfrozen'] is (epoch > 10)
                    and h['training_module_subtrees'] == (['blocks.7'] if epoch > 10 else []), 'Actual epoch/counter/window/phase differs')
        require(should_stop and best == best_epoch and metadata['stop_reason'] == 'validation_patience' and metadata['extended'] is False, 'Actual stop/selection differs')
        require(metadata['optimizer_updates'] == metadata['head_optimizer_updates'] == epochs * 4
                and metadata['backward_calls'] == epochs * ntrain and metadata['backbone_optimizer_updates'] == (epochs - 10) * 4, 'Terminal update counters differ')
        phase = 'head_only_warmup' if best_epoch <= 10 else 'lastblock_finetune'
        require(metadata['selected_checkpoint_phase'] == phase and metadata['selected_encoder_changed_parameter_count'] == (0 if best_epoch <= 10 else 12)
                and metadata['selected_head_changed_parameter_count'] == 6 and metadata['terminal_encoder_changed_parameter_count'] == 12
                and metadata['terminal_head_changed_parameter_count'] == 6 and metadata['frozen_initial_sha256'] == metadata['frozen_final_sha256'], 'Selected/terminal change disclosure differs')
        require((metadata['selected_encoder_state_sha256'] == metadata['selected_encoder_initial_state_sha256']) is (best_epoch <= 10), 'Selected encoder signature differs')
        require(cpu['cpu_checkpoint_audit']['cpu_tensors_verified'] is True and cpu['cpu_checkpoint_audit']['checkpoint_loads'] == 3
                and cpu['cpu_checkpoint_audit']['model_constructions'] == cpu['cpu_checkpoint_audit']['gpu_operations'] == 0, 'Existing CPU tensor proof invalid')
        cpus = {Path(v['path']).name: v for v in cpu['cpu_checkpoint_audit']['checkpoints']}
        require(cpus['best.pt'] == complete['checkpoint'], 'Selected descriptor is not the CPU-verified checkpoint')
        boundaries = {v['path']: v for v in cpu['cpu_checkpoint_audit']['durable_counter_boundaries']}
        require(boundaries['initial.pt']['epoch'] == boundaries['initial.pt']['optimizer_updates'] == 0
                and boundaries['latest.pt']['epoch'] == epochs and boundaries['latest.pt']['optimizer_updates'] == metadata['optimizer_updates']
                and boundaries['latest.pt']['backward_calls'] == metadata['backward_calls'] and boundaries['best.pt']['epoch'] == best_epoch,
                'Existing CPU scalar boundaries differ')
        require(boundaries['best.pt']['optimizer_updates'] is None and boundaries['best.pt']['backward_calls'] is None, 'Do not invent selected checkpoint counters')
        observed = cpu['f2_physical_durable_boundary']
        require(len(observed['attempts']) == 1 and observed['legacy_failure_counters_added_to_f2'] is False
                and observed['physical_and_durable_counts_added_together'] is False, 'Actual F2 boundary scope differs')
        trace = read(checked(observed['attempts'][0]['artifact']))
        require(trace['kind'] == 'formal' and trace['seed'] == 11 and trace['fold'] == fold and trace['status'] == 'complete'
                and trace['implementation'] == metadata['versioned_finetune_implementation'], 'Formal attempt identity/source differs')
        expected = {'forward': epochs * (2 * ntrain + len(split['val'])) + 2,
                    'encoder_forward': epochs * (ntrain + (ntrain + len(split['val'])) * 8) + 16,
                    'encoder_windows': epochs * (ntrain * 8 + (ntrain + len(split['val'])) * 60) + 120,
                    'encode': epochs * (ntrain + (ntrain + len(split['val'])) * 8) + 16,
                    'backward': epochs * ntrain, 'optimizer_step': epochs * 4}
        for key, value in expected.items():
            require(trace[key + '_started'] == trace[key + '_returned'] == value == observed['recorded_physical_call_subtotals'][key + '_returned'], 'Actual full attempt call arithmetic differs')
        actual_failure_paths = sorted(folder.glob('failure_*.json'))
        require(actual_failure_paths == [] and cpu['legacy_failure_records_not_added'] == [], 'New failure supplementary records require review')
        failures.extend(desc(p) for p in actual_failure_paths)
        traces.append(trace)
        rows.append({'fit_id': fit['fit_id'], 'fold': fold, 'metadata': cpu['metadata'], 'completion': cpu['completion'],
                     'train_val_test_counts': [len(split[r]) for r in ('train', 'val', 'test')],
                     'epochs': epochs, 'best_epoch': best_epoch, 'selected_checkpoint_phase': phase,
                     'selected_encoder_changed_parameters': metadata['selected_encoder_changed_parameter_count'],
                     'selected_head_changed_parameters': 6, 'terminal_encoder_changed_parameters': 12, 'terminal_head_changed_parameters': 6,
                     'durable_optimizer_updates': metadata['optimizer_updates'], 'durable_backward_calls': metadata['backward_calls'],
                     'terminal_backbone_optimizer_updates': metadata['backbone_optimizer_updates'],
                     'initial_state_sha256': metadata['initial_state_sha256'], 'bound_cpu_checkpoint_proof': cpu['cpu_checkpoint_audit'],
                     'f2_attempt': observed['attempts'][0]['artifact'], 'independently_recomputed_calls': expected,
                     'actual_attempt_seconds': trace['seconds'], 'nested_metadata_seconds_not_added': metadata['seconds_this_attempt'],
                     'peak_allocated_bytes': metadata['peak_allocated_bytes'], 'test_forward_count': 0})
    all_seed11 = []
    for path in sorted((OUT / 'finetune_v2_attempts' / CONDITION).glob('*/attempt.json')):
        candidate = json.loads(path.read_text(encoding='utf-8-sig'))
        if candidate['kind'] == 'formal' and candidate['seed'] == 11:
            all_seed11.append(desc(path))
    require(all_seed11 == sorted((r['f2_attempt'] for r in rows), key=lambda v: v['path']), 'An unaccounted first-seed formal attempt exists')
    engineering_path = WORK / contract['engineering_attempt_path']
    engineering = read(engineering_path)
    require(engineering['kind'] == 'engineering' and engineering['status'] == 'complete'
            and engineering['optimizer_step_started'] == engineering['optimizer_step_returned'] == 0
            and engineering['backward_started'] == engineering['backward_returned'] == 2
            and contract['probe_subject_ids'] == original['fold_1']['train'][:1], 'Separate training-only engineering record differs')
    previous = []
    for path in sorted(REPORT.glob('completed_finetune_checkpoint_audit_v2_*.json')):
        data = read(path)
        if path != AUDIT:
            previous.append({'artifact': desc(path), 'verified_completed_finetune_fits': data['verified_completed_finetune_fits'], 'not_overwritten': True})
    require({0, 1, 2, 3}.issubset({r['verified_completed_finetune_fits'] for r in previous}), 'Prior partial proofs missing')
    totals = {key: sum(trace[key] for trace in traces) for key in traces[0] if key.endswith(('_started', '_returned'))}
    require(sum(r['durable_optimizer_updates'] for r in rows) == audit['durable_optimizer_updates_verified_subtotal'] == 1140
            and sum(r['durable_backward_calls'] for r in rows) == audit['durable_backward_calls_verified_subtotal'] == 7557, 'Five-fit durable subtotal differs')
    require(len({r['initial_state_sha256'] for r in rows}) == 1, 'Same seed fresh initializer changed between folds')
    value = {'kind': 'independent_eegpt_first_seed5_stage_review', 'reviewer': '/root/acceptance_review',
             'passed': True, 'created_at': datetime.now(timezone.utc).isoformat(), 'review_source': desc(Path(__file__)),
             'actual_cpu_audit': desc(AUDIT), 'registration': audit['registration'],
             'scope': 'Exactly EEGPT last-block finetune seed11/folds1-5; 5 of135 planned FT fits and 5 of15 EEGPT fits, not an all135/all360 completion certificate.',
             'full_135_ft_completion_claimed': False, 'full_360_plan_completion_claimed': False,
             'cohort_and_roles': {'people': 37, 'HC': 21, 'MDD': 16, 'exact_original_folds_and_order': True,
                                  'all360_registered_role_sets_match_original': True, 'test_folds_partition37_once': True,
                                  'raw_descriptors_include_train_val_only': True, 'signal_or_features_read': False},
             'official_fresh_initialization': {'author_weight_descriptor_compared_not_reopened': author_weight,
                 'official_encoder_keys_loaded': 102, 'same_seed_initial_state_sha256': rows[0]['initial_state_sha256'],
                 'fresh_head_seed': 11, 'frozen_head_inherited': False, 'actual_initial_tensor_basis': 'Existing bound CPU audit, no new checkpoint deserialization',
                 'source_locations': ['finetune.py:124 build_model reloads official adapter then fresh seeded head', 'finetune.py:405 fit creates per-fit model/optimizer and zero-epoch state', 'adapters/eegpt.py:23 author loader and exact last-block list']},
             'selected_phase_counts': dict(Counter(r['selected_checkpoint_phase'] for r in rows)),
             'selected_phase_disclosure': 'Folds1-2 select epochs37/68 after head warmup and have12 changed encoder parameters. Folds3-5 select epochs4/3/6 during head-only warmup and have0 changed encoder parameters. All five terminal fits changed12 encoder and6 head parameters. Selected checkpoints were not replaced or restricted to post-warmup. Terminal update counts do not describe selected encoder changes.',
             'folds': rows, 'durable_optimizer_updates': 1140, 'durable_backward_calls': 7557,
             'formal_recorded_physical_call_subtotals': totals,
             'actual_five_formal_attempt_seconds_known_sum': sum(t['seconds'] for t in traces),
             'cost_scope': {'formal_attempts': 5, 'failed_formal_attempts_observed_for_these_five': 0, 'legacy_failure_records': failures,
                 'engineering_separate': {'artifact': desc(engineering_path), 'optimizer_updates': 0, 'backward_calls': 2, 'seconds': engineering['seconds']},
                 'durable_and_attempt_counters_added_together': False, 'metadata_and_attempt_seconds_added_together': False,
                 'historical_physical_optimizer_calls_exact': None, 'historical_wall_seconds_exact': None,
                 'limitation': 'No failed F2 or legacy failure records observed for this exact five-fit scope. This does not establish zero failure cost for other conditions, dispatcher history or the entire plan. Per-attempt resume baselines remain unrecorded in prior auditor; recorded physical subtotals are not promoted to exact all-history totals. Active seed22 excluded.'},
             'preserved_previous_partial_cpu_proofs': previous,
             'this_review_operations': {'json_and_source_text_only': True, 'checkpoint_loads': 0, 'checkpoint_binary_reads': 0,
                 'model_constructions': 0, 'gpu_calls': 0, 'raw_or_feature_reads': 0, 'test_probability_reads': 0,
                 'test_gate_calls': 0, 'evaluation_calls': 0, 'training_process_actions': 0, 'sealed_or_historical_files_modified': False},
             'final_semantic_companion': 'Internal stage evidence only; does not replace the final all360 source-bound semantic companion or wide acceptance.'}
    value['verified_json_and_source_descriptors'] = list(INPUTS.values())
    for item in list(INPUTS.values()):
        checked(item)
    path = REPORT / ('independent_eegpt_first_seed5_review_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'artifact': desc(path), 'fits': 5, 'updates': 1140, 'backwards': 7557,
                      'selected_phase_counts': value['selected_phase_counts']}))


if __name__ == '__main__':
    main()
