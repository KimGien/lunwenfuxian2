"""CSBrain first seed/five folds: bound JSON/source and existing CPU proof only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPORT = Path(__file__).resolve().parent
WORK = REPORT.parents[2]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
CONDITION = 'CSBrain-lastblock-finetune'
INPUTS = {}
AUDIT = {'path': 'output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v3_1791530442765370800.json',
         'sha256': 'f5c0a3be878b25a81a421445003dc55876a216f7fd2e831d1173091c01c1a91e'}


def capture(path, record=True):
    path = Path(path).resolve()
    relative = path.relative_to(WORK).as_posix()
    assert path.suffix in ('.json', '.py'), 'Only JSON/source reads permitted'
    data = path.read_bytes()
    descriptor = {'path': relative, 'sha256': hashlib.sha256(data).hexdigest()}
    if record:
        if relative in INPUTS:
            assert INPUTS[relative] == descriptor, 'Input changed: ' + relative
        INPUTS[relative] = descriptor
    return data, descriptor


def read(value):
    path = WORK / value['path'] if isinstance(value, dict) else value
    data, descriptor = capture(path)
    if isinstance(value, dict):
        assert descriptor['sha256'] == value['sha256'], value['path']
    return json.loads(data)


def check(binding):
    assert capture(WORK / binding['path'])[1]['sha256'] == binding['sha256']


def main():
    audit = read(AUDIT)
    assert audit['status'] == 'partial_verified' and audit['verified_completed_finetune_fits'] == 20
    assert audit['checkpoint_loads'] == 60 and audit['errors'] == audit['deferred_completed_fits'] == []
    assert len(audit['pending_fits']) == 115 and audit['registered_finetune_fits'] == 135
    assert audit['full_360_plan_completion_claimed'] is False
    assert audit['auditor_source']['sha256'] == '307d6feabb9b4964db335fd9a0390e3026015cd57f6205117de125c79508c6d7'
    for binding in [audit['auditor_source'], *audit['delegated_sources'].values()]:
        check(binding)
    sys.path.insert(0, str(WORK))
    from tridep_ccml.eeg_all_candidates import protocol as P
    registration = read(audit['registration'])
    cohort = read(registration['sources']['cohort'])
    original = read(registration['sources']['folds'])
    copied = read(registration['sources']['folds_copy'])
    official38 = read(registration['sources']['official38_folds'])
    labels = read(registration['sources']['labels'])
    people = registration['subject_ids']
    assert len(people) == len(set(people)) == 37 and people == cohort['subject_ids']
    assert registration['formal_fits'] == len(registration['fits']) == 360
    assert registration['folds'] == original == copied
    assert original == {f: {r: [s for s in ids if s != '02010004'] for r, ids in split.items()} for f, split in official38.items()}
    assert registration['labels'] == {r['subject_id']: r['label'] for r in labels if r['subject_id'] in people}
    assert Counter(registration['labels'].values()) == {0: 21, 1: 16}
    assert Counter(s for split in original.values() for s in split['test']) == Counter(people)
    for fit in registration['fits']:
        assert all(fit[r + '_ids'] == original['fold_' + str(fit['fold'])][r] for r in ('train', 'val', 'test'))
    cpu_rows = [r for r in audit['completed_rows'] if r['condition'] == CONDITION]
    assert len(cpu_rows) == 5 and {(r['seed'], r['fold']) for r in cpu_rows} == {(11, f) for f in range(1, 6)}
    assert Counter(r['condition'] for r in audit['completed_rows']) == {CONDITION: 5, 'EEGPT-lastblock-finetune': 15}
    lock = read(cpu_rows[0]['condition_source_lock'])
    contract = lock['contract']
    assert lock['registration'] == audit['registration']
    assert lock['configuration_sha256'] == hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for binding in contract['implementation_sources']:
        check(binding)
    heads = [n for n in contract['trainable_parameter_names'] if n.startswith('head.')]
    encoder = [n for n in contract['trainable_parameter_names'] if n.startswith('encoder.')]
    assert (len(heads), len(encoder), len(contract['frozen_parameter_names'])) == (6, 20, 273)
    assert all(n.startswith('encoder.backbone.encoder.layers.11.') for n in encoder)
    official = contract['input_contract']['adapter']
    assert official['formal_pretrained_strict_loaded'] is True and official['loaded_backbone_keys'] == 295
    assert official['source']['weight']['sha256'] == 'cfcdf9dcb069905e5e02e92fe3d59bd3574e779bf554cc02cb701994a79e99cc'
    assert official['source']['weight'] in contract['weight_sources']
    raw = {r['subject_id']: r['raw'] for r in registration['raw_sources']}
    results, traces, failure_records, aliases = [], [], [], []
    epochs_expected, best_expected = [147, 105, 40, 40, 40], [117, 75, 8, 3, 10]
    for row in sorted(cpu_rows, key=lambda x: x['fold']):
        fold = row['fold']
        fit = next(f for f in registration['fits'] if f['fit_id'] == row['fit_id'])
        assert fit['condition'] == CONDITION and fit['seed'] == 11 and row['passed'] is True
        folder = OUT / 'runs' / row['fit_id']
        metadata, complete, status = (read(row[k]) for k in ('metadata', 'completion', 'status_artifact'))
        event, started = read(status['event_record']), read(folder / 'started.json')
        assert event == {k: v for k, v in status.items() if k != 'event_record'}
        assert status['status'] == 'training_complete' and status['test_evaluation_status'] == 'frozen'
        assert status['evidence'] == complete and complete['metadata'] == row['metadata']
        assert complete['registry'] == status['registration'] == audit['registration']
        assert metadata['condition_source_lock'] == status['condition_source_lock'] == row['condition_source_lock'] == cpu_rows[0]['condition_source_lock']
        assert (metadata['condition'], metadata['seed'], metadata['fold']) == (CONDITION, 11, fold)
        split = original['fold_' + str(fold)]
        for role in ('train', 'val', 'test'):
            assert metadata[role + '_ids'] == fit[role + '_ids'] == split[role]
        assert sum(map(len, split.values())) == len(set().union(*map(set, split.values()))) == 37
        assert set(metadata['raw_sources']) == set(split['train'] + split['val'])
        assert all(d == raw[s] for s, d in metadata['raw_sources'].items())
        assert started['train_ids'] == split['train'] and started['val_ids'] == split['val']
        assert metadata['test_forward_count'] == started['test_forward_count'] == 0
        assert metadata['reload_probe_subject_ids'] == split['train'][:1]
        assert started['fresh_formal_pretrained'] is True and started['fresh_head_seed'] == 11
        assert started['frozen_best_head_inherited'] is metadata['frozen_best_head_inherited'] is False
        assert metadata['fresh_official_pretrained'] is metadata['fresh_head'] is True
        adapter = metadata['adapter_audit']
        assert adapter['source'] == official['source'] and adapter['formal_pretrained_strict_loaded'] is True
        assert adapter['loaded_backbone_keys'] == 295 and adapter['input_contract'] == official['input_contract']
        for binding in adapter['implementation_sources']:
            check(binding)
        assert metadata['initial_state_sha256'] == contract['initial_state_sha256']
        assert metadata['versioned_finetune_implementation'] == contract['versioned_finetune_implementation']
        assert metadata['micro_subjects'] == 1 and metadata['effective_subject_batch'] == metadata['window_bag_size'] == metadata['window_chunk'] == 8
        assert metadata['eval_window_count'] == metadata['full_cycle_unique_windows'] == 60
        epochs, best, ntrain = metadata['epochs'], metadata['best_epoch'], len(split['train'])
        assert (epochs, best) == (epochs_expected[fold - 1], best_expected[fold - 1])
        assert (row['epochs'], row['best_epoch'], row['updates']) == (epochs, best, epochs * 4)
        history, validation = (read(complete[k]) for k in ('train_history', 'validation_history'))
        assert len(history) == len(validation) == epochs
        losses, extended, extension_epochs = [], False, []
        extension_decision = None
        for epoch, (h, v) in enumerate(zip(history, validation), 1):
            assert h['epoch'] == v['epoch'] == epoch and h['val_eval_ce'] == v['val_ce']
            assert h['optimizer_updates'] == epoch * 4 and h['backward_calls'] == epoch * ntrain
            assert h['unfrozen'] is (epoch > 10)
            assert h['training_module_subtrees'] == (['backbone.encoder.layers.11'] if epoch > 10 else [])
            assert h['window_indices'] == [(((epoch - 1) % 8) * 8 + i) % 60 for i in range(8)]
            assert h['head_lr'] == 3e-4 * P.learning_rate_multiplier('finetune', epoch)
            assert h['backbone_lr'] == 1e-5 * P.learning_rate_multiplier('finetune', epoch, parameter_group='backbone')
            losses.append(v['val_ce'])
            decision = P.training_decision('finetune', losses, extended=extended)
            assert decision['stop'] is (epoch == epochs)
            if decision['extend']:
                extension_epochs.append(epoch)
                assert fold == 1 and epoch == 120
                extension_decision = decision
                extended = True
        assert decision['best_epoch'] == best and decision['reason'] == metadata['stop_reason'] == 'validation_patience'
        assert extended is metadata['extended'] and extension_epochs == ([120] if fold == 1 else [])
        assert metadata['optimizer_updates'] == metadata['head_optimizer_updates'] == 4 * epochs
        assert metadata['backbone_optimizer_updates'] == 4 * (epochs - 10) and metadata['backward_calls'] == epochs * ntrain
        phase = 'head_only_warmup' if best <= 10 else 'lastblock_finetune'
        assert metadata['selected_checkpoint_phase'] == phase
        assert metadata['selected_encoder_changed_parameter_names'] == ([] if best <= 10 else encoder)
        assert metadata['selected_encoder_changed_parameter_count'] == (0 if best <= 10 else 20)
        assert metadata['selected_encoder_parameters_changed'] is (best > 10)
        assert (metadata['selected_encoder_state_sha256'] == metadata['selected_encoder_initial_state_sha256']) is (best <= 10)
        assert metadata['terminal_encoder_changed_parameter_names'] == encoder and metadata['terminal_encoder_changed_parameter_count'] == 20
        for stage in ('selected', 'terminal'):
            assert metadata[stage + '_head_changed_parameter_names'] == heads and metadata[stage + '_head_changed_parameter_count'] == 6
        assert set(metadata['parameter_changed_after_training']) == set(heads + encoder) and all(metadata['parameter_changed_after_training'].values())
        assert metadata['frozen_initial_sha256'] == metadata['frozen_final_sha256']
        proof = row['cpu_checkpoint_audit']
        assert proof['cpu_tensors_verified'] is proof['only_map_location_cpu'] is True
        assert proof['checkpoint_loads'] == 3 and proof['gpu_operations'] == proof['model_constructions'] == 0
        cp = {Path(d['path']).name: d for d in proof['checkpoints']}
        assert cp['best.pt'] == complete['checkpoint']
        b = {d['path']: d for d in proof['durable_counter_boundaries']}
        assert b['initial.pt']['epoch'] == b['initial.pt']['optimizer_updates'] == b['initial.pt']['backward_calls'] == 0
        assert (b['latest.pt']['epoch'], b['latest.pt']['optimizer_updates'], b['latest.pt']['backward_calls']) == (epochs, epochs * 4, epochs * ntrain)
        assert b['best.pt']['epoch'] == best and b['best.pt']['optimizer_updates'] is b['best.pt']['backward_calls'] is None
        groups = b['latest.pt']['optimizer_groups']
        assert groups['group_names'] == ['head', 'backbone'] and groups['serialized_id_parameter_names'] == heads + encoder and groups['moment_shapes_verified'] is True
        boundary = row['f2_physical_durable_boundary']
        assert len(boundary['attempts']) == 1 and not boundary['duplicate_content_aliases']
        assert boundary['legacy_failure_counters_added_to_f2'] is boundary['physical_and_durable_counts_added_together'] is boundary['nested_metadata_seconds_added_to_attempt_seconds'] is False
        attempt = boundary['attempts'][0]
        trace = read(attempt['artifact'])
        assert (trace['kind'], trace['status'], trace['condition'], trace['seed'], trace['fold']) == ('formal', 'complete', CONDITION, 11, fold)
        assert trace['implementation'] == metadata['versioned_finetune_implementation']
        expected = {'forward': epochs * (2 * ntrain + len(split['val'])) + 2,
                    'encoder_forward': epochs * (ntrain + (ntrain + len(split['val'])) * 8) + 16,
                    'encode': epochs * (ntrain + (ntrain + len(split['val'])) * 8) + 16,
                    'encoder_windows': epochs * (ntrain * 8 + (ntrain + len(split['val'])) * 60) + 120,
                    'backward': epochs * ntrain, 'optimizer_step': epochs * 4}
        for name, count in expected.items():
            for suffix in ('_started', '_returned'):
                assert trace[name + suffix] == boundary['recorded_physical_call_subtotals'][name + suffix] == count
        assert sorted(folder.glob('failure_*.json')) == [] and row['legacy_failure_records_not_added'] == []
        assert math.isfinite(trace['seconds']) and trace['seconds'] > metadata['seconds_this_attempt'] > 0
        assert trace['seconds'] == attempt['seconds_this_attempt'] and metadata['peak_allocated_bytes'] == 223574528
        traces.append((attempt['artifact'], trace))
        results.append({'fit_id': row['fit_id'], 'metadata': row['metadata'], 'completion': row['completion'],
                        'roles': {k: len(v) for k, v in split.items()}, 'epochs': epochs, 'best_epoch': best,
                        'selected_phase': phase, 'selected_encoder_changes': 0 if best <= 10 else 20, 'selected_head_changes': 6,
                        'terminal_encoder_changes': 20, 'terminal_head_changes': 6, 'frozen_parameter_names': 273,
                        'extended': extended, 'extension_epochs': extension_epochs, 'extension_decision': extension_decision,
                        'stop_reason': decision['reason'], 'head_steps_each': epochs * 4, 'backbone_steps_each': (epochs - 10) * 4,
                        'durable_optimizer_updates': epochs * 4, 'durable_backward_calls': epochs * ntrain,
                        'selected_epoch_history_counts_only': {'epoch': best, 'head_steps': best * 4, 'backbone_steps': max(0, best - 10) * 4, 'backwards': best * ntrain},
                        'selected_checkpoint_serialized_optimizer_counts': None,
                        'bound_cpu_checkpoint_descriptors': proof['checkpoints'], 'f2_attempt': attempt['artifact'],
                        'recorded_physical_calls': expected, 'attempt_seconds': trace['seconds'],
                        'nested_metadata_seconds_not_added': metadata['seconds_this_attempt'], 'peak_allocated_bytes': metadata['peak_allocated_bytes']})
    seen = []
    for path in sorted((OUT / 'finetune_v2_attempts' / CONDITION).glob('*/attempt.json')):
        # Later seeds may be active: inspect only JSON identity without binding their mutable bytes.
        for retry in range(4):
            try:
                data, descriptor = capture(path, record=False)
                candidate = json.loads(data)
                break
            except (PermissionError, json.JSONDecodeError):
                if retry == 3:
                    raise
                time.sleep(0.025)
        if candidate['kind'] == 'formal' and candidate['seed'] == 11:
            check(descriptor)
            seen.append(descriptor)
    assert seen == sorted([b for b, _ in traces], key=lambda d: d['path'])
    assert len({b['sha256'] for b, _ in traces}) == 5
    engineering_path = WORK / contract['engineering_attempt_path']
    engineering = read(engineering_path)
    assert engineering['kind'] == 'engineering' and engineering['status'] == 'complete'
    assert engineering['optimizer_step_started'] == engineering['optimizer_step_returned'] == 0
    assert engineering['backward_started'] == engineering['backward_returned'] == 2
    assert contract['probe_subject_ids'] == original['fold_1']['train'][:1]
    prior = []
    for name, sha in [('independent_csbrain_first_extended_completion_review_1791525172742025200.json', '79b4ff41f4b68453f9af5ed695cf94079ce7af958bd02347690eda5dcbbd6294'),
                      ('independent_csbrain_head_warmup_completion_review_1791528721401651600.json', '09bd80a311d0f81bf4f7c4d45d08e38d74060f1a6a49c51cfd7dddaf053525d3')]:
        binding = {'path': (REPORT / name).relative_to(WORK).as_posix(), 'sha256': sha}
        assert read(binding)['passed'] is True
        prior.append(binding)
    updates = sum(r['durable_optimizer_updates'] for r in results)
    backwards = sum(r['durable_backward_calls'] for r in results)
    assert (updates, backwards) == (1488, 9899)
    assert sum(r['updates'] for r in audit['completed_rows']) == audit['durable_optimizer_updates_verified_subtotal'] == 5140
    assert audit['durable_backward_calls_verified_subtotal'] == 34121 == backwards + 24222
    assert Counter(r['selected_phase'] for r in results) == {'lastblock_finetune': 2, 'head_only_warmup': 3}
    for binding in list(INPUTS.values()):
        check(binding)
    result = {'kind': 'independent_csbrain_first_seed5_stage_review', 'passed': True,
              'created_at': datetime.now(timezone.utc).isoformat(), 'reviewer': '/root/acceptance_review',
              'review_source': capture(Path(__file__), record=False)[1], 'actual_cpu_audit': AUDIT,
              'scope': {'csbrain': '5/15', 'all_finetune_actual_cpu_snapshot': '20/135', 'existing_cpu_loads': 60,
                        'this_five_fit_existing_cpu_loads': 15, 'all_registered_target': 360,
                        'full_plan_completion_claimed': False, 'all_csbrain15_completion_claimed': False},
              'cohort_and_sources': {'people': 37, 'HC': 21, 'MDD': 16, 'same_original_five_folds': True,
                                     'all360_registered_roles_checked': True, 'official_weight_descriptor_only': official['source'],
                                     'same_seed_fresh_initial_state': contract['initial_state_sha256'], 'loaded_backbone_keys': 295,
                                     'test_forward_count': 0, 'actual_role_descriptors_train_val_only': True},
              'folds': results, 'selected_phase_counts': dict(Counter(r['selected_phase'] for r in results)),
              'selection_disclosure': 'Folds1/2 best117/75 select changed last block; folds3/4/5 best8/3/10 select unchanged encoder in inclusive first10 head-only warmup. Fold5 best10 is a legitimate validation-selected warmup boundary. All five terminal states changed20 encoder and6 head parameter tensors while273 frozen tensors remain unchanged; selected and terminal states are not conflated.',
              'extension_disclosure': 'Fold1 alone receives the preregistered one-time extension at120 and first stops147 with best117 and patience30. No new training budget or post-warmup-only selection rule was introduced.',
              'durable_optimizer_updates': updates, 'durable_backward_calls': backwards,
              'recorded_physical_call_subtotals': {name: sum(t[name] for _, t in traces) for name in traces[0][1] if name.endswith(('_started', '_returned'))},
              'formal_attempt_seconds_known_sum': sum(t['seconds'] for _, t in traces),
              'nested_metadata_seconds_not_added_sum': sum(r['nested_metadata_seconds_not_added'] for r in results),
              'cost_scope': {'unique_formal_attempts': 5, 'duplicate_content_aliases': aliases, 'legacy_failure_records': failure_records,
                             'failed_formal_attempts_observed_in_this_scope': 0, 'head_and_backbone_group_steps_added': False,
                             'physical_and_durable_counts_added': False, 'nested_and_attempt_times_added': False,
                             'historical_physical_optimizer_calls_exact': None, 'historical_wall_seconds_exact': None,
                             'engineering_separate': {'artifact': capture(engineering_path)[1], 'backwards': 2, 'optimizer_steps': 0, 'seconds': engineering['seconds']},
                             'limitation': 'Counts concern exactly these five completed fits. Mutable later-seed trace identities were only filtered out. No zero-failure-cost claim for other seeds/conditions/dispatcher history. Null historical/resume quantities stay null.'},
              'prior_case_evidence_preserved': prior,
              'verified_json_and_source_descriptors': list(INPUTS.values()),
              'review_operations': {'checkpoint_binary_reads': 0, 'checkpoint_deserializations': 0, 'models_constructed': 0,
                                   'gpu_calls': 0, 'raw_or_feature_reads': 0, 'test_probability_reads': 0, 'evaluation_calls': 0,
                                   'training_actions': 0, 'sealed_sources_or_historical_artifacts_modified': False}}
    path = REPORT / ('independent_csbrain_first_seed5_review_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': capture(path, record=False)[1], 'passed': True, 'updates': updates, 'backwards': backwards,
                      'formal_seconds': result['formal_attempt_seconds_known_sum'], 'scope': result['scope']}))


if __name__ == '__main__':
    main()
