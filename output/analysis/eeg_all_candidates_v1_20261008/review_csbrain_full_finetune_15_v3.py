"""JSON/source-only CSBrain15 stage review of actual CPU5 and mixed F2/F3.

The reviewed V2 semantic functions are delegated unchanged. Only the CPU proof
schema and physical-trace adapter change. No project verifier is executed:
some such verifiers transitively hash official binary weight sources.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve()
REPORT = SOURCE.parent
WORK = SOURCE.parents[3]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
DELEGATE = REPORT / 'review_csbrain_full_finetune_15_v2.py'
DELEGATE_SHA = 'af4fae12a5cb77ef0e9a3dffc6b3f48bc9e707a5162710a8d51259ae60f9f79c'
CPU_SHA = '58b101d301f92da4283890e0a4fa80bef40cdb38eeace1001d6fa18e583a3eba'
F3_SHA = '13601bd5f78b5c543bbc9d6c4dcb856fd9e72a6d09c55d18a5a38febeb670e59'
GUARD_SHA = '012cc88f26e6693c5e986a43e92ba72a990acd0d3c0c662e4de6c5ee431efc10'
CONDITION = 'CSBrain-lastblock-finetune'


def require(value, message):
    if not value: raise ValueError(message)


require(hashlib.sha256(DELEGATE.read_bytes()).hexdigest() == DELEGATE_SHA, 'Reviewed semantic delegate changed')
spec = importlib.util.spec_from_file_location('_private_csbrain_semantics_v2', DELEGATE)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
ACTIVE = {}


def inventory(reader):
    ACTIVE['reader'] = reader
    formal, engineering = defaultdict(list), []
    for path in sorted((OUT / 'finetune_v2_attempts' / CONDITION).glob('*/attempt.json')):
        record, binding = reader.snapshot(path)
        require(record.get('condition') == CONDITION and record.get('kind') in ('formal', 'engineering'), 'Unknown condition trace')
        require(record.get('status') in ('running', 'complete', 'failed'), 'Unknown attempt status')
        if record['kind'] == 'engineering':
            require(record['seed'] is record['fold'] is None and record['status'] in ('complete', 'failed'), 'Invalid engineering identity/state')
            engineering.append((record, binding))
        else:
            require(type(record['seed']) is int and record['seed'] in H.SEEDS and type(record['fold']) is int and record['fold'] in range(1, 6), 'Invalid formal identity')
            formal[f'{CONDITION}/seed_{record["seed"]}/fold_{record["fold"]}'].append((record, binding))
    require(set(formal) == H.EXPECTED_IDS, 'Missing completed condition trace identity')
    return formal, engineering


def descriptor_tree(reader, value):
    """Bind JSON/source descendants; explicitly carry binary descriptors only."""
    if type(value) is dict:
        if set(value) == {'path', 'sha256'}:
            if Path(value['path']).suffix in ('.py', '.json'):
                reader.checked(value)
            return
        for child in value.values(): descriptor_tree(reader, child)
    elif type(value) is list:
        for child in value: descriptor_tree(reader, child)


def verify_traces(fit, metadata, boundary, traces, implementation):
    reader = ACTIVE['reader']; seen, unique, aliases = {}, [], []
    totals = Counter(); known_seconds = 0.; unknown_seconds = []; observations = []
    by_path = {row['artifact']['path']: row for row in boundary['attempts']}
    require(len(by_path) == len(boundary['attempts']), 'Duplicate CPU physical path')
    for record, binding in traces:
        require((record['condition'], record['seed'], record['fold'], record['kind']) ==
                (fit['condition'], fit['seed'], fit['fold'], 'formal'), 'Trace identity differs')
        version = record['implementation']['version']
        require(type(version) is int and version in (2, 3), 'Unknown observer version')
        if binding['sha256'] in seen:
            aliases.append({'artifact': binding, 'alias_of': seen[binding['sha256']], 'cost_counted_again': False})
            continue
        seen[binding['sha256']] = binding
        require(binding['path'] in by_path, 'Trace omitted by supplied actual CPU proof')
        cpu = by_path[binding['path']]
        require(cpu['artifact'] == binding, 'CPU trace binding differs')
        observed = cpu['actual_execution_observation']
        require(observed['artifact'] == binding and observed['actual_implementation'] == record['implementation']
                and observed['observer_version'] == version, 'Actual observer provenance differs')
        if version == 2:
            require(record['implementation'] == implementation and 'execution_acceleration' not in record, 'F2 actual source differs')
        else:
            actual = record['implementation']; execution = record['execution_acceleration']
            require(actual['wrapper_source']['sha256'] == F3_SHA and actual['delegated_finetune_implementation'] == implementation
                    and actual['delegated_wrapper_source'] == implementation['wrapper_source']
                    and actual['delegated_runner_source'] == implementation['delegated_runner_source'], 'F3 delegation is not sealed F2/F1')
            reader.checked(actual['wrapper_source'])
            require(observed['passed'] is True and observed['execution_acceleration'] == execution, 'CPU real F3 proof differs')
            sidecar = reader.checked(execution['registration']); descriptor_tree(reader, sidecar)
            entry = sidecar['conditions'][CONDITION]
            require(entry['diagnostic_window_chunk'] == 8 and entry['qualification'] is None
                    and execution['diagnostics']['window_chunk'] == 8 and execution['diagnostics']['qualification'] is None
                    and execution['delegated_finetune_implementation'] == implementation
                    and execution['condition_source_lock'] == metadata['condition_source_lock']
                    and (execution['condition'], execution['seed'], execution['fold']) == (CONDITION, fit['seed'], fit['fold']), 'Actual unchanged-chunk sidecar differs')
            any_work = any(record[k + suffix] for k in H.COUNTERS for suffix in ('_started', '_returned'))
            if any_work and (record['backward_started'] or record['optimizer_step_started'] or 'execution_acceleration' in metadata):
                require(metadata.get('execution_acceleration') == execution, 'F3 work lacks matching completed metadata execution')
            require(observed['zero_call_existing_completion_recheck'] is (not any_work), 'Zero-work recheck disclosure differs')
            require(record['durable_counter_snapshot'] == {k + suffix: record[k + suffix] for k in H.COUNTERS for suffix in ('_started', '_returned')}, 'Persisted F3 counter snapshot differs')
            require(sum(record['flush_counts'].values()) == record['flush_sequence'] and record['flush_counts']['attempt_created'] == 1,
                    'Actual buffered flush sequence differs')
            terminal = record['graceful_terminal_flush']
            require(type(terminal) is bool and record['counter_snapshot_exact_for_process'] is terminal
                    and record['unflushed_tail_unknown'] is not terminal
                    and observed['unflushed_tail_unknown'] is not terminal, 'F3 unknown-tail disclosure differs')
            if terminal:
                require(record['unflushed_tail_counters'] == {k + suffix: 0 for k in H.COUNTERS for suffix in ('_started', '_returned')}, 'Graceful F3 nonzero tail')
            else:
                require(record['unflushed_tail_counters'] is None, 'Lost F3 tail cannot be zero-filled')
            durable = record.get('latest_durable_checkpoint')
            if record['status'] == 'complete' and record['optimizer_step_returned']:
                require(durable is not None and (durable['epoch'], durable['optimizer_updates'], durable['backward_calls']) ==
                        (metadata['epochs'], metadata['optimizer_updates'], metadata['backward_calls']), 'Terminal F3 durable commit differs')
        for key in H.COUNTERS:
            start, end = record[key + '_started'], record[key + '_returned']
            require(H.integer(start) and H.integer(end) and end <= start and (key == 'encoder_windows' or start-end <= 1), 'Physical counter malformed')
            if record['status'] == 'complete': require(start == end, 'Completed physical call did not return')
        require(record['encode_started'] == record['encoder_forward_started'] and record['encode_returned'] == record['encoder_forward_returned'], 'Encoder alias mismatch')
        if record['status'] == 'running':
            require(record.get('seconds') is None and record.get('finished_at') is None and cpu['seconds_this_attempt'] is None, 'Historical elapsed endpoint invented')
            stopped = observed.get('proven_stopped_handoff'); lost = observed.get('proven_process_loss')
            require((stopped is None) != (lost is None), 'Running historical attempt lacks one unique completed closure')
            proof = stopped or lost
            require(proof['passed'] is True, 'CPU closure proof not passed')
            descriptor_tree(reader, proof)
            if lost:
                closure = reader.checked(lost['closure'])
                require(binding['sha256'] == 'e190ee531b30275322398edeeab6a93c4db4a28dc239510e28a59ea4939fd52f'
                        and closure['original_attempt'] == binding and reader.checked(closure['attempt_snapshot']) == record
                        and closure['durable'] == {'epoch': 8, 'optimizer_updates': 32, 'backward_calls': 216}
                        and closure['known_extra_lower_bounds'] == {'optimizer_updates': 4, 'backward_calls': 27}
                        and closure['unflushed_tail_counters'] is None and closure['final_elapsed_seconds'] is None,
                        'Oct10 process-loss closure differs')
                descriptor_tree(reader, closure)
                require(cpu['status'] == 'process_loss_observed' and lost['extra_work_not_added_again'] is True,
                        'Loss closure was promoted or extra work added twice')
            else:
                safe = reader.checked(stopped['proof'])
                require(safe['f2_attempt'] == binding and reader.checked(safe['f2_attempt_snapshot']) == record
                        and cpu['status'] == 'stopped_by_verified_handoff', 'Historical safe-stop proof differs')
                descriptor_tree(reader, safe)
                require(stopped['extra_work_not_added_again_to_observed_trace'] is True, 'Historical stop delta added again')
            require(record['optimizer_step_returned'] - proof['durable_optimizer_updates'] == proof['known_extra_optimizer_returns_outside_checkpoint']
                    and record['backward_returned'] - proof['durable_backward_calls'] == proof['known_extra_backward_returns_outside_checkpoint'], 'Closure lower-bound arithmetic differs')
            unknown_seconds.append(binding)
        else:
            require(H.finite(record.get('seconds')) and record.get('finished_at') and cpu['seconds_this_attempt'] == record['seconds'], 'Terminal observed time differs')
            known_seconds += record['seconds']
        counters = {k + suffix: record[k + suffix] for k in H.COUNTERS for suffix in ('_started', '_returned')}
        require(cpu['observed_counters'] == counters, 'CPU counter row differs')
        totals.update(counters)
        unique.append({'artifact': binding, 'status': record['status'], 'observer_version': version,
                       'seconds': record.get('seconds'), 'normalized_status_for_cpu_boundary': cpu['status'],
                       'zero_model_or_training_call_recheck': not any(counters.values())})
        observations.append(observed)
    require([x['artifact'] for x in unique] == [x['artifact'] for x in boundary['attempts']]
            and aliases == boundary['duplicate_content_aliases'] and dict(totals) == boundary['recorded_physical_call_subtotals'], 'Independent physical inventory/subtotals differ')
    require(boundary['durable_optimizer_updates_counted_once'] == metadata['optimizer_updates']
            and boundary['durable_backward_calls_counted_once'] == metadata['backward_calls'], 'Durable trajectory differs')
    require(totals['optimizer_step_returned'] - metadata['optimizer_updates'] == boundary['optimizer_returns_above_durable_known_lower_bound']
            and totals['backward_returned'] - metadata['backward_calls'] == boundary['backward_returns_above_durable_known_lower_bound'], 'Observed extra work differs')
    for key in ('physical_and_durable_counts_added_together', 'nested_metadata_seconds_added_to_attempt_seconds',
                'legacy_failure_counters_added_to_f2', 'historical_or_unflushed_physical_work_zero_filled'):
        require(boundary[key] is False, 'Overlapping/unknown work incorrectly combined')
    require(all(boundary[k] is None for k in ('physical_optimizer_calls_exact', 'physical_backward_calls_exact', 'all_historical_wall_seconds_exact')), 'Unknown history promoted to exact')
    require(any(x['status'] == 'complete' for x in unique), 'No terminal successful invocation')
    return {'attempts': unique, 'duplicate_aliases_not_added': aliases, 'recorded_physical_call_subtotals': dict(totals),
            'recorded_attempt_seconds_known_sum': known_seconds, 'unknown_elapsed_attempts': unknown_seconds,
            'single_complete_attempt_math_checked': False, 'actual_execution_observations': observations,
            'known_extra_optimizer_returns_above_durable_not_added_again': totals['optimizer_step_returned'] - metadata['optimizer_updates'],
            'known_extra_backward_returns_above_durable_not_added_again': totals['backward_returned'] - metadata['backward_calls']}


def adapted_review():
    """Recompile only the CPU-schema routing in the original semantic review."""
    tree = ast.parse(DELEGATE.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'review')
    changed = 0
    for item in ast.walk(node):
        if isinstance(item, ast.Compare) and ast.unparse(item.left) == "audit['schema_version']":
            require(len(item.comparators) == 1 and item.comparators[0].value == 3, 'Original schema anchor changed')
            item.comparators[0].value = 5; changed += 1
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            item.value = item.value.replace('audit_completed_finetune_checkpoints_v3.py', 'audit_completed_finetune_checkpoints_v5.py').replace('genuine v3 CPU audit', 'genuine v5 CPU audit').replace('reviewed v3 CPU', 'reviewed v5 CPU').replace('actual v3 CPU', 'actual v5 CPU')
    require(changed == 1, 'Expected one CPU schema change')
    H.AUDITOR_SHA = CPU_SHA; H.inventory = inventory; H.verify_traces = verify_traces
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), 'exec'), H.__dict__)
    return H.review


def guardian_terminal(reader, result):
    root = OUT / 'execution_acceleration_v1/sync_boundary_v3'
    paths = list(root.glob('same_dispatcher_resume_complete_*.json'))
    require(len(paths) == 1, 'Expected one real same-parent terminal')
    terminal, binding = reader.snapshot(paths[0]); control = reader.checked(terminal['control_registration'])
    require(terminal['source']['sha256'] == GUARD_SHA and terminal['source'] == control['source'], 'Guardian source differs')
    # These two descriptors describe historical bytes now retained by snapshots.
    # The dispatcher legitimately updates its live owner and terminal job record.
    descriptor_tree(reader, {k: v for k, v in control.items() if k not in ('queue_owner', 'queue_attempt')})
    require(control['queue_owner']['sha256'] == control['queue_owner_snapshot']['sha256']
            and control['queue_attempt']['sha256'] == control['queue_attempt_snapshot']['sha256'],
            'Historical queue snapshots do not preserve their original bytes')
    require(len(control['protected_sources']) == 80 and terminal['held_process_identities'] == control['roles'], 'Actual recovery80/identities differ')
    require(terminal['passed'] is True and terminal['phase'] == 'qualification_finished_same_Q5_resumed'
            and terminal['dispatcher_suspended'] is False and terminal['same_dispatcher_resumed'] is True
            and terminal['boundary_workflow_completed'] is True and terminal['dispatcher_resume_calls_returned'] == 1,
            'Actual same-parent resume not confirmed')
    for key in ('GPU_training_child_suspend_or_terminate_calls', 'formal_process_terminate_calls', 'formal_queue_spawn_calls'):
        require(type(terminal[key]) is int and terminal[key] == 0, 'Unexpected formal process control')
    require(terminal['outcomes'] == terminal['probe_intervals'] == [] and terminal['all_qualifications_passed'] is False
            and terminal['actual_sync_optimization_applied'] is False, 'Deferred GPU probes cannot be called passed/applied')
    deferred = reader.checked(terminal['deferred_qualification'])
    require(deferred['source'] == terminal['source'] and deferred['control_registration'] == terminal['control_registration']
            and deferred['status'] == 'deferred_external_GPU_contexts' and deferred['GPU_speedup_claimed'] is False,
            'Actual zero-probe deferral differs')
    require(all(type(deferred[k]) is int and deferred[k] == 0 for k in ('model_constructions', 'GPU_forward_calls', 'backward_calls', 'optimizer_updates', 'formal_fits')), 'Deferred probe invented calls')
    contexts = deferred['observed_GPU_contexts']['contexts']
    require(len(contexts) == 1 and contexts[0]['pid'] == 15304 and contexts[0]['type'] == 'C+G'
            and contexts[0]['process_name'].replace('\\', '/').casefold() == 'c:/program files/tencent/qqnt/qq.exe', 'Observed QQ deferral differs')
    expected_artifacts = []
    for row in result['fits']:
        folder = OUT / 'runs' / row['fit_id']
        complete = reader.checked(row['completion'])
        expected_artifacts.extend(complete[key] for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'))
        expected_artifacts.extend(reader.snapshot(folder / name)[1] for name in ('complete.json', 'status.json'))
    require(len(terminal['completed_CSBrain_artifacts']) == 90 and terminal['completed_CSBrain_artifacts'] == expected_artifacts,
            'Guardian complete90 must exactly match CPU-backed original15')
    owner_snapshot = reader.checked(control['queue_owner_snapshot'])
    require(owner_snapshot['pid'] == terminal['held_process_identities']['dispatcher']['pid'] == 45980,
            'Guardian did not resume the same actual Q5 parent')
    queue_attempt = reader.checked(control['queue_attempt_snapshot'])
    actual_attempt, actual_binding = reader.snapshot(WORK / control['queue_attempt']['path'])
    require(queue_attempt['status'] == 'running' and actual_attempt['status'] == 'complete' and actual_attempt['returncode'] == 0
            and actual_attempt['child_pid'] == queue_attempt['child_pid'] == terminal['held_process_identities']['launcher']['pid'], 'Actual child launcher did not finish exit0')
    require(H.finite(terminal['hold_seconds']) and H.finite(terminal['children_wait_seconds'])
            and terminal['hold_seconds'] >= terminal['children_wait_seconds']
            and math.isclose(terminal['hold_seconds'], terminal['hold_ended_monotonic'] - terminal['hold_started_monotonic']), 'Observed guardian time differs')
    return {'terminal': binding, 'control_registration': terminal['control_registration'],
            'actual_terminal_queue_attempt': actual_binding, 'completed_artifacts_verified': 90,
            'same_dispatcher_pid': 45980, 'resume_calls_returned': 1, 'actual_probes_launched': 0,
            'actual_GPU_probe_qualification_passed': False, 'sync_optimization_applied': False,
            'deferred_qualification': terminal['deferred_qualification'], 'observed_external_contexts': contexts,
            'hold_seconds_overlapping_queue_wait': terminal['hold_seconds'],
            'children_wait_seconds_overlapping_training': terminal['children_wait_seconds'],
            'timings_added_to_formal_or_probe_seconds': False,
            'both_exit0_evidence_basis': 'Sealed guardian success path checks both held native child handles before artifacts/defer; queue terminal separately records launcher returncode0. Individual native exit-code fields were not emitted in this deferred terminal.',
            'stale_current_GPU_training_continues_field_not_used_as_terminal_evidence': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--cpu-audit', type=Path, required=True)
    args = parser.parse_args()
    result, reader = adapted_review()(args.cpu_audit)
    require(result['passed'] is True, 'Actual full15 CPU-backed semantic result required')
    result['guardian_terminal'] = guardian_terminal(reader, result)
    reader.snapshot(DELEGATE); reader.snapshot(SOURCE)
    require('torch' not in sys.modules, 'No tensor library may be imported')
    result.update(kind='CSBrain_full_finetune_15_readonly_stage_review_v3', source=reader.snapshot(SOURCE)[1],
                  delegated_semantic_source=reader.snapshot(DELEGATE)[1], created_at=datetime.now(timezone.utc).isoformat(),
                  actual_cpu_checkpoint_loads_this_review=0, official_binary_weight_reads_this_review=0,
                  raw_or_feature_or_test_reads_this_review=0, model_or_GPU_or_process_operations=0,
                  prior_live_X_L_verifier_not_executed_because_it_hashes_official_weight_sources=True)
    result['known_extra_optimizer_returns_above_durable'] = sum(r['physical_attempts']['known_extra_optimizer_returns_above_durable_not_added_again'] for r in result['fits'])
    result['known_extra_backward_returns_above_durable'] = sum(r['physical_attempts']['known_extra_backward_returns_above_durable_not_added_again'] for r in result['fits'])
    result['unknown_elapsed_attempts'] = [x for row in result['fits'] for x in row['physical_attempts']['unknown_elapsed_attempts']]
    result['verified_JSON_and_source_descriptors'] = list(reader.inputs.values())
    reader.recheck()
    target = REPORT / ('CSBrain_full15_JSON_stage_review_v3_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'report': target.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'passed': True, 'CSBrain': '15/15', 'FT_CPU_snapshot': result['actual_CPU_snapshot_completed_FT'],
                      'selected_phases': result['selected_phase_counts'], 'durable_updates': result['durable_optimizer_updates'],
                      'durable_backward': result['durable_backward_calls'], 'physical_updates_lower_bound': result['formal_recorded_physical_call_subtotals']['optimizer_step_returned'],
                      'physical_backward_lower_bound': result['formal_recorded_physical_call_subtotals']['backward_returned']}))


if __name__ == '__main__': main()
