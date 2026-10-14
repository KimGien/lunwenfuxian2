"""Cost18 preparation: retain all Cost9 losses, add true observer12/repair ledgers.

Default does not collect. Actual F12 counts are primary once; qualifier worker
attempts are primary once. New driver/controller/status records are nested
supplemental only. Unknown tails, elapsed endpoints and active training time
remain unknown. No test probabilities, tensor loads, model or GPU are used.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v4 as PREVIOUS_Q
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v5 as PREVIOUS_SUCCESS_Q
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v6 as PREVIOUS_IDENTITY_Q
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v7 as PREVIOUS_TREE_Q
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v8 as PREVIOUS_CORE_Q
from tridep_ccml.eeg_all_candidates import storage as S
from tridep_ccml.eeg_all_candidates import post_training_cpu_recovery_v1 as RECOVERY
from tridep_ccml.eeg_all_candidates import repair_observation_v9 as V
from tridep_ccml.eeg_all_candidates import repair_observation_v8 as OLD_V
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v9 as PREVIOUS_REPEAT_Q
from tridep_ccml.eeg_all_candidates import diagnose_mamba_repeat_v1 as DIAG
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v3 as L
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v10 as Q
DELEGATE = SOURCE.with_name('cost_aggregation_v9.py')
DELEGATE_SHA = '4d0863abf0a4dbfe30e8e2bd853ca7d7bb8242212fce65683be36d9cd70811e0'
COUNTER_CATEGORY = 'deterministic_backward_repair_qualification'
NESTED_CATEGORY = 'repair_dispatcher_and_worker_elapsed'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def delegate():
    require(S.digest(DELEGATE) == DELEGATE_SHA, 'Original Cost9 source changed')
    spec = importlib.util.spec_from_file_location('_Cost18_private_original_Cost9', DELEGATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def buffered_observation(old, data, artifact):
    version = data.get('implementation', {}).get('version')
    if version not in (11, 12):
        return old(data, artifact)
    provider = OLD_V if version == 11 else V
    require(data['condition'] == ('CodeBrain-lastblock-finetune' if version == 11 else 'EEGMamba-lastblock-finetune'), 'Actual repair observer/condition mismatch')
    fit = {key: data[key] for key in ('condition', 'seed', 'fold')}
    fit['fit_id'] = fit['condition'] + '/seed_' + str(fit['seed']) + '/fold_' + str(fit['fold'])
    proof = provider.verify_observer(data, artifact, fit)
    return {**proof, 'native_backward_supplementary': data.get('native_backward_supplementary'), 'native_counts_not_added_to_model_backward_or_seconds': True, 'execution_acceleration': proof['actual_execution'], 'unflushed_tail_counters': data.get('unflushed_tail_counters'), 'flush_sequence': data['flush_sequence'], 'last_flush_reason': data['last_flush_reason'], 'counter_semantics': data['counter_semantics'], 'durable_counter_snapshot': data['durable_counter_snapshot'], 'latest_durable_checkpoint_commit_observation': data.get('latest_durable_checkpoint'), 'phase_counters_are_not_inferred_from_total_calls': True, 'training_window_chunk': 8, 'diagnostic_window_chunk': 8, 'unobserved_optimizer_side_effects_of_unreturned_calls_are_unknown': True}

def category_totals(rows):
    primary = [x for x in rows if x['aggregation_role'] == 'primary_observed_invocation']
    names = {name for x in primary for name in x['observed_counters']}
    counters = {}
    for name in sorted(names):
        known = [x['observed_counters'].get(name) for x in primary if x['observed_counters'].get(name) is not None]
        counters[name] = {'known_observed_subtotal': sum(known) if known else None, 'records_with_value': len(known), 'records_unknown': len(primary) - len(known), 'exact_category_total': None}
    seconds = [x['seconds_observed'] for x in primary if x['seconds_observed'] is not None]
    return {'primary_invocations': len(primary), 'supplemental_records': len(rows) - len(primary), 'counter_totals': counters, 'known_recorded_scope_seconds_subtotal': sum(seconds) if seconds else None, 'seconds_unknown_records': len(primary) - len(seconds), 'full_category_wall_or_gpu_busy_time': None, 'nested_driver_times_added': False}

def repair_worker_record(artifact, value, normalizer):
    expected = PREVIOUS_Q if '/deterministic_repair_qualification_v4/' in artifact['path'] else PREVIOUS_SUCCESS_Q if '/deterministic_repair_qualification_v5/' in artifact['path'] else PREVIOUS_IDENTITY_Q if '/deterministic_repair_qualification_v6/' in artifact['path'] else PREVIOUS_TREE_Q if '/deterministic_repair_qualification_v7/' in artifact['path'] else PREVIOUS_CORE_Q if '/deterministic_repair_qualification_v8/' in artifact['path'] else PREVIOUS_REPEAT_Q if '/deterministic_repair_qualification_v9/' in artifact['path'] else DIAG if '/mamba_repeat_diagnosis_v1/' in artifact['path'] else Q
    require(any(('/deterministic_repair_qualification_v' + str(v) + '/' in artifact['path'] for v in (4, 5, 6, 7, 8, 9, 10))) or '/mamba_repeat_diagnosis_v1/' in artifact['path'], 'Unknown qualification ledger generation')
    require(value.get('source') == S.desc(expected.SOURCE) and value.get('kind') == 'deterministic_repair_engineering' and (value.get('condition') in Q.CONDITIONS) and (value.get('status') in ('running', 'failed', 'engineering_verified')) and (value.get('formal_fits') == 0), 'Unknown or falsely formal repair-engineering ledger')
    for kind in ('forward', 'backward', 'encoder_forward', 'encoder_windows', 'optimizer_step'):
        a, b = (value.get(kind + '_started'), value.get(kind + '_returned'))
        require(type(a) is int and type(b) is int and (0 <= b <= a), 'Malformed actual repair count')
    require(value['optimizer_step_started'] == value['optimizer_step_returned'] == 0, 'Repair qualification cannot optimize')
    seconds = value.get('primary_engineering_seconds')
    require(seconds is None or (type(seconds) in (int, float) and seconds >= 0), 'Invalid actual repair elapsed')
    temporary = copy.deepcopy(value)
    if seconds is not None:
        temporary['seconds'] = seconds
    normalized = normalizer(temporary)
    normalized.update(seconds_observed=seconds, seconds_source_field='primary_engineering_seconds' if seconds is not None else None, seconds_scope='Actual qualifier worker scope once; unknown on nonterminal crash; no active-time inference')
    return {'artifact': artifact, 'category': COUNTER_CATEGORY, 'aggregation_role': 'primary_observed_invocation', 'condition': value['condition'], 'status': value['status'], 'attempt_id': artifact['path'], 'started_at': value.get('started_at'), 'finished_at': value.get('finished_at'), 'counter_scope': 'Actual started/returned qualifier calls; unreturned internal work unknown', 'unreturned_call_internal_work_unknown': value.get('unreturned_call_internal_work_unknown'), 'formal_fits': 0, 'new_forward_back_counts_not_formal_training': True, 'accumulation_order_changed': value.get('accumulation_order_changed'), 'original_input_status_preserved': True, **normalized}

def add_repair_ledgers(result, normalizer):
    existing = {x['artifact']['path'] for x in result['cost_records']}
    added = []
    errors = []
    patterns = ('deterministic_repair_qualification_v8/*/*/attempt.json', 'deterministic_repair_qualification_v8/*/*/controller.json', 'deterministic_repair_qualification_v7/*/*/attempt.json', 'deterministic_repair_qualification_v7/*/*/controller.json', 'deterministic_repair_qualification_v6/*/*/attempt.json', 'deterministic_repair_qualification_v6/*/*/controller.json', 'deterministic_repair_qualification_v5/*/*/attempt.json', 'deterministic_repair_qualification_v5/*/*/controller.json', 'deterministic_repair_qualification_v4/*/*/attempt.json', 'deterministic_repair_qualification_v4/*/*/controller.json', 'deterministic_repair_qualification_v9/*/*/attempt.json', 'deterministic_repair_qualification_v9/*/*/controller.json', 'continuation_queue_v14/attempts/*/status.json', 'continuation_queue_v14/attempts/*/worker.json')
    patterns += ('deterministic_repair_qualification_v10/*/*/attempt.json', 'deterministic_repair_qualification_v10/*/*/controller.json', 'mamba_repeat_diagnosis_v1/*/*/attempt.json', 'mamba_repeat_diagnosis_v1/*/*/controller.json', 'continuation_queue_v13/attempts/*/status.json', 'continuation_queue_v13/attempts/*/worker.json')
    require(len(set(patterns)) == len(patterns), 'Duplicate engineering ledger patterns')
    for pattern in patterns:
        for path in sorted(S.OUT.glob(pattern)):
            try:
                artifact = S.desc(path)
                value = S.read(S.checked(artifact))
                if artifact['path'] in existing:
                    matches = [x for x in result['cost_records'] if x['artifact']['path'] == artifact['path']]
                    require(len(matches) == 1 and matches[0]['artifact'] == artifact and (path.name == 'status.json') and any(prefix in artifact['path'] for prefix in ('continuation_queue_v13/attempts/', 'continuation_queue_v14/attempts/')) and (matches[0]['aggregation_role'] == 'supplemental_nested_or_unmeasured'), 'Existing repair ledger mismatched/duplicated/primary')
                    row = matches[0]
                    row.update(repair_parent_scope_verified=True, parent_seconds_added_to_formal_or_primary_work=False, seconds_observed=value.get('inclusive_parent_seconds'), seconds_source_field='inclusive_parent_seconds' if 'inclusive_parent_seconds' in value else None, seconds_scope='Existing dispatcher record once; nested inclusive Q14 scope not added')
                    continue
                if path.name == 'attempt.json':
                    row = repair_worker_record(artifact, value, normalizer)
                else:
                    normalized = normalizer(value, invocation_aliases=False)
                    field = next((x for x in ('inclusive_controller_seconds', 'inclusive_parent_seconds', 'inclusive_worker_driver_seconds') if x in value), None)
                    normalized.update(seconds_observed=value.get(field) if field else None, seconds_source_field=field, seconds_scope='Nested parent/worker scope only; never added to F12 or qualifier primary')
                    row = {'artifact': artifact, 'category': NESTED_CATEGORY, 'aggregation_role': 'supplemental_nested_or_unmeasured', 'condition': value.get('condition'), 'status': value.get('status'), 'attempt_id': artifact['path'], 'started_at': value.get('started_at'), 'finished_at': value.get('finished_at'), 'parent_seconds_added_to_formal_or_primary_work': False, **normalized}
                if value.get('native_backward_function_counts') is not None or value.get('native_backward_supplementary') is not None:
                    row['native_backward_supplementary'] = value.get('native_backward_function_counts', value.get('native_backward_supplementary'))
                    row['native_calls_not_added_to_model_backward_or_seconds'] = True
                added.append(row)
                existing.add(artifact['path'])
                result['inputs'].append(artifact)
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append({'path': path.relative_to(WORK).as_posix(), 'error_type': type(error).__name__, 'error': str(error)})
    result['cost_records'].extend(added)
    result['input_read_errors'].extend(errors)
    for category in (COUNTER_CATEGORY, NESTED_CATEGORY):
        result['category_observations'][category] = category_totals([x for x in added if x['category'] == category])
    result['repair_ledger_coverage'] = {'patterns': list(patterns), 'actual_records_observed': len(added), 'errors': errors, 'failed_attempts_retained': True, 'qualifier_report_duplicates_not_primary': True, 'parent_worker_elapsed_not_added': True}
    peaks = [result.get('max_observed_peak_allocated_bytes'), *[x.get('peak_allocated_bytes') for x in added]]
    result['max_observed_peak_allocated_bytes'] = max([x for x in peaks if type(x) is int], default=None)
    result['unknown_or_opaque_records'].extend((x['artifact']['path'] for x in added if x['unknown_counter_fields'] or x['aggregation_role'] != 'primary_observed_invocation'))
    return result


FAILURE_CLOSURE = {'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/deterministic_repair_qualification_v5/failure_closures/pre_model_linux_permission_1791778566044163800/closure.json', 'sha256': '98bd04eb5256ef3d895eb750cb1a7ef9c4ac3f69ccce97b5dd37414739a21e37'}

def add_preserved_failure_closure(result):
    """Bind a zero-model failure without converting unknown old timing to zero."""
    value = S.read(S.checked(FAILURE_CLOSURE))
    require(value.get('schema') == 'actual_pre_model_qualification5_failure_closure_v1'
            and value.get('source_qualification') == S.desc(PREVIOUS_SUCCESS_Q.SOURCE), 'Exact original Qual5 failure closure required')
    require(all(type(value.get(k)) is int and value[k] == 0 for k in
                ('formal_fits', 'model_GPU_calls', 'forward_calls', 'backward_calls', 'optimizer_updates', 'checkpoint_tensor_loads'))
            and value.get('model_construction_started') is False and value.get('cleanup_verified') is True,
            'Failure closure must preserve actual pre-model zero-call scope')
    require(value.get('original_controller_status_preserved') == 'worker_starting'
            and value.get('original_controller_exit_code') is None and value.get('original_controller_elapsed') is None
            and value.get('actual_exit_time_unknown') is True, 'Original failure endpoint uncertainty changed')
    controller = S.read(S.checked(value['original_controller']))
    require(controller.get('status') == 'worker_starting'
            and controller.get('returncode') is None and controller.get('inclusive_controller_seconds') is None,
            'Original unreturned controller bytes/status must remain unchanged')
    artifacts = [FAILURE_CLOSURE] + [value[k] for k in
                 ('source_qualification', 'archived_owner', 'original_controller', 'worker_claim',
                  'failure_log', 'native_absence', 'root_linux_absence', 'permission_diagnosis')]
    existing = {x['path']: x for x in result['inputs']}
    for artifact in artifacts:
        S.checked(artifact)
        if artifact['path'] in existing:
            require(existing[artifact['path']] == artifact, 'Failure closure input byte binding differs')
        else:
            result['inputs'].append(artifact)
            existing[artifact['path']] = artifact
    result['repair_qualification_failure_closures'] = [{
        'artifact': FAILURE_CLOSURE, 'original_controller': value['original_controller'],
        'original_status_preserved': 'worker_starting', 'original_elapsed': None,
        'actual_exit_time_unknown': True, 'extra_primary_counters_added': False,
        'extra_primary_seconds_added': False, 'zero_model_scope': True}]
    return result


def add_post_training_recovery(result, normalizer):
    gate_binding=S.desc(RECOVERY.ROOT/'complete.json')
    gate=RECOVERY.verify_gate(gate_binding)
    categories=set()
    inputs={x['path']:x for x in result['inputs']}
    def bind(artifact):
        S.checked(artifact)
        if artifact['path'] in inputs:require(inputs[artifact['path']]==artifact,'Post-training input byte binding changed')
        else:result['inputs'].append(artifact);inputs[artifact['path']]=artifact
    proof={'gate':gate_binding,'failed_CPU16':RECOVERY.FAILED_CPU16,'successful_CPU17':gate['CPU_audit'],'platform_captures':[RECOVERY.LINUX_REFERENCE,RECOVERY.WINDOWS_REFERENCE],'failed_CPU16_loads':360,'successful_CPU17_loads':405,'CPU_audit_elapsed_seconds':None,'formal_fits_or_updates_added':0,'nested_times_added':False}
    for artifact,kind in [(RECOVERY.FAILED_CPU16,'CPU16_failed'),(gate['CPU_audit'],'CPU17_passed'),(RECOVERY.LINUX_REFERENCE,'platform_capture_linux'),(RECOVERY.WINDOWS_REFERENCE,'platform_capture_windows')]:
        data=S.read(S.checked(artifact));bind(artifact)
        existing=[x for x in result['cost_records'] if x['artifact']['path']==artifact['path']]
        require(len(existing)<=1 and all(x['artifact']==artifact for x in existing),'Duplicate/mutated post-training cost artifact')
        row=existing[0] if existing else {'artifact':artifact}
        prior_category=row.get('category')
        if prior_category:categories.add(prior_category)
        normalized=normalizer(data,invocation_aliases=False)
        if kind.startswith('CPU'):
            expected=360 if kind=='CPU16_failed' else 405
            operations=data['observed_auditor_operations_including_failed_checks']
            require(operations['cpu_checkpoint_loads_started']==operations['cpu_checkpoint_loads_returned']==expected,'Actual CPU load accounting differs')
            category='post_training_CPU_checkpoint_audit'
            seconds=None;field=None
            counters={'cpu_checkpoint_loads_started':expected,'cpu_checkpoint_loads_returned':expected}
        else:
            require(data['schema']=='actual_platform_learning_rate_reference_v1','Actual platform capture required')
            require(data['Tensor_loads']==data['model_calls']==data['GPU_calls']==data['outer_test_reads']==0,'Platform capture cannot contain model/test work')
            category='original_platform_LR_stdlib_capture';field='primary_stdlib_capture_seconds';seconds=data[field]
            require(type(seconds) in (float,int) and seconds>0,'Real capture elapsed required')
            counters={'cpu_checkpoint_loads_started':0,'cpu_checkpoint_loads_returned':0}
        normalized['observed_counters'].update(counters)
        normalized.update(seconds_observed=seconds,seconds_source_field=field,seconds_scope='Independent actual audit read count; elapsed unrecorded' if kind.startswith('CPU') else 'One actual independent stdlib platform capture; not training or nested parent time')
        row.update(artifact=artifact,category=category,aggregation_role='primary_observed_invocation',status=data.get('status'),condition=None,attempt_id=artifact['path'],post_training_scope=kind,formal_fits=0,optimizer_updates=0,new_scope_not_added_to_formal_training=True,prior_inventory_category=prior_category,**normalized)
        if not existing:result['cost_records'].append(row)
        categories.add(category)
    for artifact in [gate_binding,gate['source'],gate['original_Q14_failed_terminal'],gate['original_Q14_released_owner'],gate['original_child_complete'],gate['post_training_source_review'],*gate['post_training_sources']]:bind(artifact)
    for category in categories:result['category_observations'][category]=category_totals([x for x in result['cost_records'] if x['category']==category])
    for artifact in [RECOVERY.FAILED_CPU16,gate['CPU_audit'],RECOVERY.LINUX_REFERENCE,RECOVERY.WINDOWS_REFERENCE]:
        require(sum(x['artifact']==artifact and x['aggregation_role']=='primary_observed_invocation' for x in result['cost_records'])==1,'Independent actual post-training cost must occur exactly once')
    result['post_training_CPU_recovery_accounting']=proof
    result['rules'].append('CPU16 failed360 and CPU17 passed405 real reads are separate once; absent elapsed stays null. Two platform stdlib captures each retain their own primary elapsed, never training or nested time.')
    return result


def collect():
    previous = delegate()
    old = previous.load_v5()
    original = old.buffered_observation
    old.buffered_observation = lambda data, artifact: buffered_observation(original, data, artifact)
    result = old.collect()
    result = previous.augment(result, verifier=L.verify_lost_f3_attempt, reader=L.read_json, verifier_source=S.desc(Path(L.__file__)))
    result = add_repair_ledgers(result, old.load_v4().normalized)
    result = add_preserved_failure_closure(result)
    result = add_post_training_recovery(result, old.load_v4().normalized)
    result.update(schema_version=18, source=S.desc(SOURCE), previous_Cost9_source=S.desc(DELEGATE), actual_repair_observer_verifier_source=S.desc(V.SOURCE), actual2_3_loss_closures_unchanged=True, actual11_12_versions_not_relabelled=True, prior_CodeBrain_observer_source=S.desc(OLD_V.SOURCE), all_plan_wall_seconds_exact=None, all_plan_physical_optimizer_returns_exact=None, no_new_formal_fit_or_primary_scope_duplicates=True)
    result['rules'].append('Actual observer12 and zero-step repair engineering are version/source-bound; nested Q14/worker timers remain supplemental.')
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        require(args.output is None, 'Actual output requires --run')
        print(json.dumps({'status': 'preparation_only', 'actual_cost_collection_started': False, 'model_calls': 0, 'GPU_calls': 0, 'checkpoint_tensor_loads': 0, 'new_output_writes': 0}))
        return
    result = collect()
    target = (args.output or SOURCE.parent / ('actual_cost_v18_' + str(time.time_ns()) + '.json')).resolve()
    require(target.parent == SOURCE.parent, 'Standalone new analysis artifact required')
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'artifact': S.desc(target), 'read_errors': result['input_read_errors'], 'draft': True}))
if __name__ == '__main__':
    main()
