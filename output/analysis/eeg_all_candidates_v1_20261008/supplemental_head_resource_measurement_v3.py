"""Q4/E3 gate for unchanged, independent zero-update head-resource probes.

Default/check-only pending branch reads registration/status JSON only. All12
original measurements are delegated to the byte-pinned v2 implementation.
No historical resource fields, formal checkpoints or evaluation are rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import types

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
OUT_REL = 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT_REL = 'output/analysis/eeg_all_candidates_v1_20261008'
V2_SOURCE = SOURCE.with_name('supplemental_head_resource_measurement_v2.py')
V2_SHA = '351f67b6941b12cfb8139372783d92158c5040d7ad20abf4a8b2d435e5a9a54b'
Q3_SHA = 'e135585ca475b733bbc6bfbaa1616fe71f9b8fb1af8f2c0b413d2a9c8565b3a0'


def isolated_delegate():
    data = V2_SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != V2_SHA:
        raise ValueError('Reviewed v2 resource measurement source changed')
    module = types.ModuleType('_supplemental_head_resource_v3_delegate')
    module.__file__ = str(V2_SOURCE)
    exec(compile(data, str(V2_SOURCE), 'exec'), module.__dict__)
    module.SOURCE = SOURCE
    return module


D = isolated_delegate()
require, read, descriptor, checked = D.require, D.read, D.descriptor, D.checked
write_new, replace_own_trace, now = D.write_new, D.replace_own_trace, D.now
choose_records, probe_plan = D.choose_records, D.probe_plan
exclusive_measurement_lease = D.exclusive_measurement_lease
existing_test_gate = D.existing_test_gate


def delegated_source():
    binding = descriptor(V2_SOURCE)
    require(binding['sha256'] == V2_SHA, 'Original v2 source changed')
    return binding


def screening(registration, statuses, queue, owner_exists):
    value = D.screening(registration, statuses, queue, owner_exists)
    value['reasons'] = [reason.replace('Q3', 'Q4') for reason in value['reasons']]
    return value


def json_checked(binding, work=WORK):
    require(Path(binding['path']).suffix == '.json', 'This gate only checks JSON, never tensor/raw bytes')
    return checked(binding, work)


def verify_cpu_shape(value, registration, registration_binding):
    """Pure evidence shape; actual source/artifact bindings are checked separately."""
    expected = {r['fit_id'] for r in registration['fits'] if r['stage'] == '3'}
    require(len(expected) == 135 and value.get('schema_version') == 4
            and value.get('registration') == registration_binding, 'Actual v4 FT135 CPU proof required')
    for key, expected_count in [('registered_total_fits', 360), ('registered_finetune_fits', 135),
                                ('training_complete_finetune_observed', 135), ('verified_completed_finetune_fits', 135),
                                ('logical_completed_fits_counted', 135), ('checkpoint_loads', 405)]:
        require(type(value.get(key)) is int and value[key] == expected_count, 'Incomplete CPU audit count: ' + key)
    require(value.get('status') == 'all_finetune_verified' and value.get('all_135_finetune_checkpoints_verified') is True
            and value.get('partial_snapshot') is False and value.get('pending_fits') == []
            and value.get('deferred_completed_fits') == [] and value.get('errors') == [], 'Partial/deferred/error CPU evidence is inadmissible')
    rows = value.get('completed_rows', [])
    require(len(rows) == 135 and len({r['fit_id'] for r in rows}) == 135
            and {r['fit_id'] for r in rows} == expected, 'Exact unique FT135 identities required')
    for row in rows:
        deep = row.get('cpu_checkpoint_audit', {})
        require(row.get('passed') is True and deep.get('cpu_tensors_verified') is True
                and type(deep.get('checkpoint_loads')) is int and deep['checkpoint_loads'] == 3
                and deep.get('only_map_location_cpu') is True
                and type(deep.get('gpu_operations')) is int and deep['gpu_operations'] == 0
                and type(deep.get('model_constructions')) is int and deep['model_constructions'] == 0,
                'A fit lacks actual three-checkpoint CPU verification')
    operations = value.get('observed_auditor_operations_including_failed_checks', {})
    require(all(type(operations.get(k)) is int and operations[k] == n for k, n in
                [('cpu_checkpoint_loads_started', 405), ('cpu_checkpoint_loads_returned', 405),
                 ('model_construction_attempts_blocked', 0), ('gpu_initialization_attempts_blocked', 0)]),
            'Actual CPU audit operation observations differ')
    return rows


def verify_cpu_gate(out, registration, registration_binding, q4, *, work=WORK):
    path = out / 'continuation_queue_v4/pre_test_finetune_cpu_gate.json'
    gate = read(path)
    require(gate.get('passed') is True and type(gate.get('checkpoint_loads')) is int and gate['checkpoint_loads'] == 405
            and type(gate.get('test_forward_calls_this_check')) is int and gate['test_forward_calls_this_check'] == 0,
            'Q4 pre-test CPU gate must prove 405 actual loads with no inference')
    audit_path = json_checked(gate['audit'], work)
    require(audit_path.parent == out / 'continuation_queue_v4'
            and audit_path.name.startswith('pre_test_finetune_cpu_audit_'), 'CPU evidence is not the actual Q4 pre-test audit')
    audit = read(audit_path)
    auditor = descriptor(Path(work) / REPORT_REL / 'audit_completed_finetune_checkpoints_v4.py', work)
    require(audit.get('auditor_source') == auditor and auditor in q4['sources'], 'CPU auditor must be source-bound by Q4')
    rows = verify_cpu_shape(audit, registration, registration_binding)
    for row in rows:
        folder = out / 'runs' / row['fit_id']
        for key, name in (('status_artifact', 'status.json'), ('metadata', 'metadata.json'), ('completion', 'complete.json')):
            require(json_checked(row[key], work) == folder / name, 'CPU audit JSON binding belongs to another fit')
        require({r['path'] for r in row['cpu_checkpoint_audit']['checkpoints']} == {
            (folder / name).relative_to(work).as_posix() for name in ('initial.pt', 'latest.pt', 'best.pt')},
            'CPU audit checkpoint descriptors belong to another fit')
    return {'gate': descriptor(path, work), 'audit': gate['audit'], 'auditor_source': auditor,
            'actual_cpu_checkpoint_loads': 405, 'actual_verified_finetune_identities': 135}


def inspect_gate(*, work=WORK, verifier=None):
    work = Path(work).resolve(); out = work / OUT_REL
    registration = read(out / 'registration.json')
    statuses = []
    for row in registration['fits']:
        path = out / 'runs' / row['condition'] / ('seed_' + str(row['seed'])) / ('fold_' + str(row['fold'])) / 'status.json'
        statuses.append(read(path)['status'] if path.exists() else 'engineering_pending')
    queue_path = out / 'continuation_queue_v4/status.json'
    queue = read(queue_path) if queue_path.exists() else {}
    queue_binding = descriptor(queue_path, work) if queue_path.exists() else None
    value = screening(registration, statuses, queue, (out / 'continuation_queue_v4/owner.json').exists())
    if (out / 'continuation_queue_v3/owner.json').exists():
        value['reasons'].append('An original Q3 owner is still present')
        value['status'] = 'pending'
    value.update(checked_at=now(), registration=descriptor(out / 'registration.json', work), model_constructions=0, gpu_calls=0,
                 actual_completed_report_verifier_called=False, delegated_resource_measurement_source=delegated_source())
    if value['status'] == 'pending':
        return value
    q3_path = out / 'continuation_queue_v3/registration.json'
    q3_binding = descriptor(q3_path, work); q3 = read(q3_path)
    require(q3_binding['sha256'] == Q3_SHA and len(q3['sources']) == 47 and q3['registry'] == value['registration'], 'Original Q3 registration differs')
    q4_path = out / 'continuation_queue_v4/registration.json'; q4 = read(q4_path)
    require(type(q4.get('version')) is int and q4['version'] == 4 and q4['registry'] == value['registration']
            and q4['previous_execution_registration'] == q3_binding, 'Q4 registration lineage differs')
    require(len({r['path'] for r in q4['sources']}) == len(q4['sources'])
            and all(record in q4['sources'] for record in q3['sources'] + [q3_binding]), 'Q4 did not preserve every original sealed source')
    for record in q4['sources']:
        checked(record, work)
    expected_jobs = {row['id'] for row in q4['jobs']}
    require(len(expected_jobs) == len(q4['jobs']) and expected_jobs
            and set(queue.get('outcomes', {})) == expected_jobs
            and all(v == 'complete' for v in queue['outcomes'].values()), 'Not every fixed Q4 job completed successfully')
    evaluator_source = descriptor(work / 'tridep_ccml/eeg_all_candidates/evaluate_v3.py', work)
    require(evaluator_source in q4['sources'], 'Q4 does not bind actual E3 evaluator source')
    cpu_gate = verify_cpu_gate(out, registration, value['registration'], q4, work=work)
    if verifier is None:
        require(work == WORK, 'Production verifier may only use this actual workspace')
        sys.path.insert(0, str(WORK))
        from tridep_ccml.eeg_all_candidates import registry as R, evaluate_v3 as E
        training = R.assert_test_unlocked()
        require(training['formal_fits_verified'] == 360, 'Full360 byte-bound gate failed')
        # E3 delegates the immutable E1/E2 gate. It must already exist so a
        # nominally read-only completed-report verification cannot create it.
        gate_binding = existing_test_gate(out, training)
        proof = E.completed_report_verified()
        checked(gate_binding)
    else:
        proof = verifier()
    require(proof.get('passed') is True and proof.get('formal_fits_verified') == 360
            and proof.get('supervised_fits_evaluated') == 345 and proof.get('supervised_conditions') == 23
            and all(type(proof.get(k)) is int and proof[k] == 0 for k in ('model_constructions', 'gpu_forward_calls'))
            and proof.get('evaluator_execution', {}).get('version') == 3
            and proof['evaluator_execution']['wrapper_source'] == evaluator_source,
            'Read-only E3 full report verification failed')
    for folder in ('continuation_queue_v3', 'continuation_queue_v4'):
        require(not (out / folder / 'owner.json').exists(), 'A Q3/Q4 owner appeared during verification')
    require(descriptor(queue_path, work) == queue_binding, 'Queue terminal status changed during verification')
    value.update(status='ready_after_full_e3_verification', evaluation_verification=proof,
                 q3_registration=q3_binding, q4_registration=descriptor(q4_path, work), q4_terminal_status=queue_binding,
                 pre_test_finetune_cpu_verification=cpu_gate, test_or_feature_reads_performed=True,
                 actual_completed_report_verifier_called=True,
                 read_scope='Only existing E3 evaluation artifacts after360/Q4/actual135CPU405 gates; no feature arrays/models/GPU')
    return value


def assert_terminal_still_bound(gate, out):
    for folder in ('continuation_queue_v3', 'continuation_queue_v4'):
        require(not (Path(out) / folder / 'owner.json').exists(), 'A Q3/Q4 owner appeared; no supplementary model operation allowed')
    for key in ('registration', 'q3_registration', 'q4_registration', 'q4_terminal_status'):
        json_checked(gate[key])
    for key in ('gate', 'audit'):
        json_checked(gate['pre_test_finetune_cpu_verification'][key])
    json_checked(gate['evaluation_verification']['complete_manifest'])
    delegated_source()


def successful_existing(path, *, plan, gate, source):
    record = read(path)
    return (record.get('delegated_resource_measurement_source') == delegated_source()
            and record.get('gate', {}).get('q4_registration') == gate['q4_registration']
            and record.get('gate', {}).get('q4_terminal_status') == gate['q4_terminal_status']
            and record.get('gate', {}).get('pre_test_finetune_cpu_verification') == gate['pre_test_finetune_cpu_verification']
            and D.successful_existing(path, plan=plan, gate=gate, source=source))


def write_measurement(path, value):
    if value.get('subtype') == 'supplemental_head_resource_measurement' and 'measurement_scope' in value:
        value['delegated_resource_measurement_source'] = delegated_source()
        value['resource_measurement_version'] = 3
        value['measurement_scope']['seconds'] = value['measurement_scope']['seconds'].replace('E2 prerequisite', 'E3 prerequisite')
    write_new(path, value)


D.assert_terminal_still_bound = assert_terminal_still_bound
D.write_new = write_measurement
measure = D.measure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run', action='store_true', help='Execute all12 only after360/Q4-owner-release/E3/actual135CPU405 gate')
    group.add_argument('--check-only', action='store_true', help='Default: inspect gate without supplementary model work')
    args = parser.parse_args(argv)
    gate = inspect_gate()
    gate_path = WORK / REPORT_REL / ('supplemental_head_resource_v3_gate_' + str(time.time_ns()) + '.json')
    write_new(gate_path, {'source': descriptor(SOURCE), **gate, 'supplemental_gpu_execution_requested': args.run})
    if not args.run or gate['status'] != 'ready_after_full_e3_verification':
        print(json.dumps({'gate': descriptor(gate_path), 'status': gate['status'], 'reasons': gate.get('reasons', []), 'supplemental_model_work': 0}, ensure_ascii=False))
        return 2 if args.run else 0
    out = WORK / OUT_REL; registration = read(checked(gate['registration']))
    conditions = [r['condition_id'] for r in registration['configuration']['conditions'] if r['policy_key'] in ('diagnostic', 'frozen')]
    require(len(conditions) == 12 and len(set(conditions)) == 12, 'Exactly12 diagnostic/frozen head conditions required')
    completed = []
    with exclusive_measurement_lease(out):
        for condition in conditions:
            assert_terminal_still_bound(gate, out)
            plan = probe_plan(condition, registration)
            existing = [path for path in (out / 'supervised_engineering' / condition).glob('measurement_*.json')
                        if successful_existing(path, plan=plan, gate=gate, source=descriptor(SOURCE))]
            require(len(existing) <= 1, 'Multiple completed supplemental attempts need explicit cost review')
            if existing:
                checked(read(existing[0])['resume_probe']); completed.append(descriptor(existing[0])); continue
            completed.append(measure(plan, gate))
    summary = {'subtype': 'supplemental_head_resource_measurement_completion', 'passed': True,
               'completed_at': now(), 'source': descriptor(SOURCE), 'delegated_resource_measurement_source': delegated_source(),
               'gate': descriptor(gate_path), 'conditions': conditions, 'measurements': completed,
               'optimizer_updates': 0, 'historical_missing_measurements_backfilled': False}
    path = WORK / REPORT_REL / ('supplemental_head_resource_v3_complete_' + str(time.time_ns()) + '.json')
    write_new(path, summary)
    print(json.dumps({'complete': descriptor(path), 'conditions': len(completed), 'optimizer_updates': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
