"""Focused independent source/JSON review; no project imports or binary reads."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import time

SOURCE = Path(__file__).resolve()
REPORT = SOURCE.parent
WORK = SOURCE.parents[3]
INPUTS = {}


def read(path, sha=None):
    path = (WORK / path).resolve()
    assert path.is_relative_to(WORK) and path.suffix in ('.py', '.json')
    raw = path.read_bytes()
    desc = {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert sha is None or desc['sha256'] == sha
    assert desc['path'] not in INPUTS or INPUTS[desc['path']] == desc
    INPUTS[desc['path']] = desc
    return (json.loads(raw.decode('utf-8-sig')) if path.suffix == '.json' else raw.decode('utf-8-sig')), desc


def checked(desc):
    return read(desc['path'], desc['sha256'])[0]


def emit(prefix, value):
    for binding in list(INPUTS.values()):
        checked(binding)
    path = REPORT / (prefix + '_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    raw = path.read_bytes()
    print(json.dumps({'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}))


def main():
    _, own = read(SOURCE)
    stage, stage_desc = read(REPORT / 'CSBrain_full15_JSON_stage_review_v3_1791601874868214000.json', 'fb18ae903abd0f520a1a4b30923c248eda7b6592d7f5d3297589ed4f50cdfe8a')
    helper = checked(stage['source'])
    delegate = checked(stage['delegated_semantic_source'])
    assert stage['source']['sha256'] == '9f7eaec7fb4a00787512b2cde064051d7e75d03f5947e6728c03040da6ccbec3'
    assert stage['delegated_semantic_source']['sha256'] == 'af4fae12a5cb77ef0e9a3dffc6b3f48bc9e707a5162710a8d51259ae60f9f79c'
    # Parse sources only: never import/execute either reviewer or live X/L.
    delegate_tree, helper_tree = ast.parse(delegate), ast.parse(helper)
    reader_node = next(n for n in delegate_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Reader')
    assert "path.suffix in ('.json', '.py')" in ast.unparse(reader_node)
    assert 'X.verify_observer_record(' not in helper and 'L.verify_lost_f3_attempt(' not in helper
    assert "if k not in ('queue_owner', 'queue_attempt')" in helper
    assert "control['queue_owner_snapshot']" in helper and "control['queue_attempt_snapshot']" in helper
    cpu = checked(stage['actual_cpu_audit'])
    checked(stage['actual_cpu_auditor_source'])
    assert stage['actual_cpu_audit']['sha256'] == 'a4617947d048d4f58359aaffb8fb2fec13034025baf3ac6a9222feeac3bd05db'
    assert cpu['schema_version'] == 5 and cpu['auditor_source'] == stage['actual_cpu_auditor_source']
    assert cpu['verified_completed_finetune_fits'] == 30 and cpu['checkpoint_loads'] == 90
    assert not cpu['errors'] and not cpu['deferred_completed_fits']
    assert stage['CSBrain_completed'] == stage['stage_FT_fits_reviewed'] == 15
    assert stage['full_135_ft_completion_claimed'] is stage['full_360_plan_completion_claimed'] is False
    registry = checked(stage['registration'])
    registered = {r['fit_id']: r for r in registry['fits'] if r['condition'] == 'CSBrain-lastblock-finetune'}
    rows = {r['fit_id']: r for r in stage['fits']}
    cpu_rows = {r['fit_id']: r for r in cpu['completed_rows'] if r['condition'] == 'CSBrain-lastblock-finetune'}
    assert len(rows) == len(stage['fits']) == 15 and set(rows) == set(registered) == set(cpu_rows)
    totals, phases, unknown = Counter(), Counter(), []
    metadata_by_id, complete_by_id = {}, {}
    for fit_id, row in rows.items():
        cr, fit = cpu_rows[fit_id], registered[fit_id]
        assert cr['metadata'] == row['metadata'] and cr['completion'] == row['completion']
        metadata = metadata_by_id[fit_id] = checked(row['metadata'])
        complete_by_id[fit_id] = checked(row['completion'])
        assert metadata['epochs'] == row['epochs'] and metadata['best_epoch'] == row['best_epoch']
        assert row['durable_optimizer_updates'] == row['head_group_updates'] == math.ceil(len(fit['train_ids']) / 8) * row['epochs']
        assert row['backbone_group_updates'] == math.ceil(len(fit['train_ids']) / 8) * (row['epochs'] - 10)
        assert row['durable_backward_calls'] == len(fit['train_ids']) * row['epochs']
        assert row['terminal_encoder_changed'] == 20 and row['terminal_head_changed'] == 6 and row['frozen_parameter_names'] == 273
        assert row['selected_phase'] == ('head_only_warmup' if row['best_epoch'] <= 10 else 'lastblock_finetune')
        if row['selected_phase'] == 'head_only_warmup':
            assert row['selected_encoder_changed'] == 0
            assert metadata['selected_encoder_state_sha256'] == metadata['selected_encoder_initial_state_sha256']
        assert metadata['frozen_initial_sha256'] == metadata['frozen_final_sha256']
        physical = row['physical_attempts']
        boundary = cr['f2_physical_durable_boundary']
        assert physical['recorded_physical_call_subtotals'] == boundary['recorded_physical_call_subtotals']
        assert cr['cpu_checkpoint_audit']['checkpoint_loads'] == 3
        assert cr['cpu_checkpoint_audit']['checkpoints'] == row['CPU_verified_checkpoint_descriptors_not_reopened']
        totals.update(physical['recorded_physical_call_subtotals']); phases[row['selected_phase']] += 1
        unknown += physical['unknown_elapsed_attempts']
    assert dict(phases) == stage['selected_phase_counts'] == {'lastblock_finetune': 9, 'head_only_warmup': 6}
    assert dict(totals) == stage['formal_recorded_physical_call_subtotals']
    assert sum(r['durable_optimizer_updates'] for r in rows.values()) == stage['durable_optimizer_updates'] == 5948
    assert sum(r['durable_backward_calls'] for r in rows.values()) == stage['durable_backward_calls'] == 39660
    assert totals['optimizer_step_returned'] == 5952 and totals['backward_returned'] == 39689
    assert totals['optimizer_step_returned'] - stage['durable_optimizer_updates'] == 4
    assert totals['backward_returned'] - stage['durable_backward_calls'] == 29
    assert unknown == stage['unknown_elapsed_attempts'] and len(unknown) == 2
    assert {d['sha256'] for d in unknown} == {'b51b7f3c5212d01660b5789ae2e248a864b40e818c0eb70074198e739332cd17', 'e190ee531b30275322398edeeab6a93c4db4a28dc239510e28a59ea4939fd52f'}
    # Strong formula applies only to the actual pristine final fit.
    single = [r for r in rows.values() if len(r['physical_attempts']['attempts']) == 1 and r['physical_attempts']['attempts'][0]['status'] == 'complete']
    assert len(single) == 1 and single[0]['seed'] == 33 and single[0]['fold'] == 5
    row = single[0]; fit = registered[row['fit_id']]; metadata = metadata_by_id[row['fit_id']]
    attempt = checked(row['physical_attempts']['attempts'][0]['artifact'])
    n, v, epoch = len(fit['train_ids']), len(fit['val_ids']), row['epochs']
    assert (n, v, epoch) == (27, 3, 41)
    expected = {'forward': epoch * (2 * n + v) + 2, 'encoder_forward': epoch * (n + (n + v) * 8) + 16,
                'encode': epoch * (n + (n + v) * 8) + 16, 'encoder_windows': epoch * (n * 8 + (n + v) * 60) + 120,
                'backward': epoch * n, 'optimizer_step': epoch * math.ceil(n / 8)}
    for key, value in expected.items():
        assert type(attempt[key + '_returned']) is int and attempt[key + '_started'] == attempt[key + '_returned'] == value
        assert row['physical_attempts']['recorded_physical_call_subtotals'][key + '_returned'] == value
    assert attempt['seconds'] >= metadata['seconds_this_attempt'] > 0
    terminal = checked(stage['guardian_terminal']['terminal'])
    control = checked(stage['guardian_terminal']['control_registration'])
    for key in ('queue_owner', 'queue_attempt'):
        assert control[key]['sha256'] == control[key + '_snapshot']['sha256']
        checked(control[key + '_snapshot'])
    assert terminal['control_registration'] == stage['guardian_terminal']['control_registration']
    assert terminal['same_dispatcher_resumed'] is True and terminal['dispatcher_resume_calls_returned'] == 1
    assert terminal['outcomes'] == terminal['probe_intervals'] == []
    assert terminal['actual_sync_optimization_applied'] is terminal['all_qualifications_passed'] is False
    bindings = {d['path']: d for d in stage['verified_JSON_and_source_descriptors']}
    expected_artifacts = []
    for row in stage['fits']:
        complete = complete_by_id[row['fit_id']]
        expected_artifacts.extend(complete[key] for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'))
        folder = 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/runs/' + row['fit_id']
        expected_artifacts.extend(bindings[folder + '/' + name] for name in ('complete.json', 'status.json'))
    assert len(expected_artifacts) == 90 and terminal['completed_CSBrain_artifacts'] == expected_artifacts
    assert sum(Path(d['path']).suffix == '.pt' for d in expected_artifacts) == 15
    observations = [
        {'source': stage['delegated_semantic_source'], 'line': 45, 'finding': 'Reader accepts workspace .json/.py only; no tensor or official binary read in this helper.'},
        {'source': stage['source'], 'line': 186, 'finding': 'Only review CPU schema/source routing is recompiled; original per-fit selection, role, policy, optimizer and tensor-evidence comparisons remain delegated.'},
        {'source': stage['source'], 'line': 131, 'finding': 'Historical running attempts require existing CPU-backed closure evidence, keep seconds/tails unknown and extras already counted in observed returns.'},
        {'source': stage['source'], 'line': 203, 'finding': 'Historical owner/attempt descriptors are validated through same-SHA immutable snapshots; live owner not used for historical-byte verification.'},
        {'source': stage['source'], 'line': 254, 'finding': 'Native dual exit0 basis is explicitly the sealed guardian path, not invented per-child fields in deferred terminal.'},
    ]
    emit('independent_CSBrain_full15_JSON_stage_review', {
        'schema': 'independent_CSBrain_full15_JSON_stage_review_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': own, 'passed': True, 'reviewed_stage_report': stage_desc,
        'scope': 'Independent static I/O/adapter review and source-bound JSON aggregate/CPU-proof cross-check; no execution of reviewed helper, X/L, full CPU auditor, current queue inspection, or full-plan audit.',
        'actual_CPU_proof': stage['actual_cpu_audit'], 'CSBrain_scope': '15/15', 'global_FT_snapshot': '30/135; 90 prior CPU loads; this review 0',
        'selected_phase_counts': dict(phases), 'durable_optimizer_updates': 5948, 'durable_backward_calls': 39660,
        'known_physical_optimizer_returns_lower_bound': 5952, 'known_physical_backward_returns_lower_bound': 39689,
        'extra_optimizer_returns_already_in_physical_total': 4, 'extra_backward_returns_already_in_physical_total': 29,
        'unknown_elapsed_attempts': unknown, 'historical_exact_elapsed_or_unflushed_tail': None,
        'additional_single_complete_attempt_formula': {'fit_id': single[0]['fit_id'], 'n_train': n, 'n_val': v, 'epochs': epoch,
            'attempt': single[0]['physical_attempts']['attempts'][0]['artifact'], 'expected_started_and_returned': expected,
            'passed': True, 'previous_stage_single_complete_attempt_math_checked': False, 'other_multi_attempt_fits_not_forced_into_formula': 14,
            'nested_metadata_seconds_not_added': True},
        'guardian_terminal': stage['guardian_terminal'], 'guardian_90_descriptors_match_stage_completions': True,
        'checkpoint_descriptors_carried_not_reopened': 15, 'observations': observations, 'definite_blockers': [],
        'actual_cpu_checkpoint_loads': 0, 'official_binary_weight_reads': 0, 'raw_feature_test_reads': 0,
        'model_GPU_process_operations': 0, 'project_verifier_or_reviewed_helper_executions': 0,
        'old_sources_reports_modified': False, 'full_135_or_360_completion_claimed': False,
        'actual_verified_JSON_source_inputs': list(INPUTS.values())})
    # Correct our own historical scope claim without reopening any binary or rerunning L.
    old, old_desc = read(REPORT / 'independent_actual_lost_f3_closure_json_review_1791597281658863000.json', '0bf21e4de14408f636aafcdbcb0f4d2a98a337094ac5ebce8ebb5aad662a056e')
    precedent, precedent_desc = read(REPORT / 'historical_binary_hash_scope_erratum_1791601956592773300.json', 'c7542365ed904799da58242605a3d46dabfd164ec00cdb18a409887405279038')
    for call in precedent['source_proven_transitive_call_chain']:
        text = checked(call['source'])
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == call['function'])
        section = ast.get_source_segment(text, node)
        assert all(fragment in section for fragment in call['required_source_fragments'])
    assert old['closure_proof']['passed'] is True and 'JSON-only' in old['scope']
    assert old['verifier_source']['sha256'] == '97dd30a04a0f028890c124001a4f06e4f2b22cf86677a3d6f0731fd53b790a57'
    emit('independent_closure_review_binary_hash_scope_erratum', {
        'schema': 'independent_closure_review_binary_hash_scope_erratum_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': own, 'affected_review': old_desc, 'related_peer_erratum': precedent_desc,
        'correction': 'The historical actual L verification transitively ran X.verify_observer_record -> registry._condition_lock -> checked/digest over official weight_sources. The JSON-only clause in that review scope was inaccurate: official binary weight bytes were read for SHA verification. No saved training tensor was deserialized and no model/GPU work occurred.',
        'source_proven_transitive_call_chain': precedent['source_proven_transitive_call_chain'],
        'historical_official_binary_hash_read_occurred': True, 'historical_binary_hash_call_count': None, 'historical_binary_hash_seconds': None,
        'historical_tensor_loads': 0, 'historical_model_GPU_process_operations': 0, 'historical_raw_feature_test_reads': 0,
        'effect_on_findings': 'I/O scope correction only; original closure lineage, 4/27 already-counted lower-bound excess, unknown tail/elapsed and CPU proof counts remain unchanged.',
        'old_source_and_report_bytes_preserved': True, 'this_erratum_binary_reads': 0, 'this_erratum_project_verifier_executions': 0,
        'full_plan_completion_claimed': False, 'actual_verified_JSON_source_inputs': list(INPUTS.values())})


if __name__ == '__main__':
    main()
