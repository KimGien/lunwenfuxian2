"""Read-only actual Q10/CPU12/E8 postprocessing gate; default no operations.

The pending path reads fit/owner JSON only. No Tensor/model/GPU/raw/test reads.
Byte binding can hash official/checkpoint sources in the verified path.
This is not a resource runner, final cost, or final semantic acceptance.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R
from tridep_ccml.eeg_all_candidates import evaluate_v8 as E
from tridep_ccml.eeg_all_candidates import continuation_queue_v10 as Q6
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v7 as Q
PINS = {S.WORK / 'tridep_ccml/eeg_all_candidates/continuation_queue_v10.py': '7f954af132e1dd1a8f0311d407119fce8da53fbf9d5a285fc956f1f7e0bc196e', S.WORK / 'tridep_ccml/eeg_all_candidates/evaluate_v8.py': 'bbc0fca1494f6b9c634a70c841cfc78b8db95293038e5bcca3c5ca982e290d33', S.WORK / 'output/analysis/eeg_all_candidates_v1_20261008/audit_completed_finetune_checkpoints_v12.py': '1e711b503bb42180a42f7ab4ea2190d8b1359be8e8ebe9f24ecd3181ed8fed09', S.WORK / 'output/analysis/eeg_all_candidates_v1_20261008/cost_aggregation_v13.py': 'ab7c48540fa59cbff0a45ad4d4db3ed69c072610f700f2c590644e411a714c28'}
OWNERS = ('continuation_queue_v3/owner.json', 'continuation_queue_v4/owner.json', 'continuation_queue_v5/owner.json', 'continuation_queue_v10/owner.json', 'evaluation_v8_control/owner.json', 'deterministic_repair_qualification_v7/owner.json')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def screening(registration, statuses, owners, control_complete_present):
    fits = registration.get('fits', [])
    require(registration.get('formal_fits') == 360 and len(fits) == 360 and (len({x['fit_id'] for x in fits}) == 360), 'Exact original360 identities required')
    require(len(statuses) == 360, 'Every original fit needs a real status')
    counts = dict(Counter(statuses))
    reasons = []
    if counts.get('training_complete') != 360:
        reasons.append('Actual360 formal fits remain incomplete')
    if owners:
        reasons.append('Actual training/engineering/evaluation owners remain: ' + ','.join(owners))
    if not control_complete_present:
        reasons.append('Actual E8 controller completion remains absent')
    return {'status': 'pending' if reasons else 'metadata_ready', 'reasons': reasons, 'actual_fit_statuses': counts, 'model_calls': 0, 'GPU_queries': 0, 'outer_probability_reads': 0}

def inspect():
    Q.check_pause()
    registration = S.read(S.OUT / 'registration.json')
    statuses = []
    for fit in registration['fits']:
        path = S.OUT / 'runs' / fit['fit_id'] / 'status.json'
        value = S.read(path) if path.exists() else {'status': 'missing'}
        require(not path.exists() or value['fit_id'] == fit['fit_id'], 'Status belongs to another original fit')
        statuses.append(value['status'])
    owners = [name for name in OWNERS if (S.OUT / name).exists()]
    return screening(registration, statuses, owners, (E.CONTROL / 'complete.json').exists())

def validate_CPU_shape(value, registration, registration_binding, cpu_source):
    expected = {x['fit_id'] for x in registration['fits'] if x['stage'] == '3'}
    require(len(expected) == 135 and value.get('schema_version') == 12 and (value.get('registration') == registration_binding) and (value.get('auditor_source') == cpu_source), 'Original FT135 and actual source-bound CPU12 required')
    for key, total in [('registered_total_fits', 360), ('registered_finetune_fits', 135), ('training_complete_finetune_observed', 135), ('verified_completed_finetune_fits', 135), ('logical_completed_fits_counted', 135), ('checkpoint_loads', 405)]:
        require(type(value.get(key)) is int and value[key] == total, 'Incomplete exact CPU12 count: ' + key)
    require(value.get('status') == 'all_finetune_verified' and value.get('all_135_finetune_checkpoints_verified') is True and (value.get('partial_snapshot') is False) and (value.get('errors') == []) and (value.get('deferred_completed_fits') == []) and (value.get('pending_fits') == []), 'Partial/deferred/error CPU12 report rejected')
    rows = value.get('completed_rows', [])
    require(len(rows) == 135 and {x['fit_id'] for x in rows} == expected and (len({x['fit_id'] for x in rows}) == 135), 'CPU12 rows must match original135 unique identities')
    for row in rows:
        deep = row.get('cpu_checkpoint_audit', {})
        require(row.get('passed') is True and deep.get('cpu_tensors_verified') is True and (type(deep.get('checkpoint_loads')) is int) and (deep['checkpoint_loads'] == 3) and (deep.get('only_map_location_cpu') is True) and (type(deep.get('gpu_operations')) is int) and (deep['gpu_operations'] == 0) and (type(deep.get('model_constructions')) is int) and (deep['model_constructions'] == 0), 'Actual three-checkpoint CPU12 row evidence missing')
    operations = value.get('observed_auditor_operations_including_failed_checks', {})
    require(all((type(operations.get(key)) is int and operations[key] == count for key, count in [('cpu_checkpoint_loads_started', 405), ('cpu_checkpoint_loads_returned', 405), ('model_construction_attempts_blocked', 0), ('gpu_initialization_attempts_blocked', 0)])), 'Actual CPU12 operation counts differ')
    return rows

def verify_existing():
    cheap = inspect()
    require(cheap['status'] == 'metadata_ready', 'Actual360/owners/E8 completion prerequisites pending')
    for path, digest in PINS.items():
        require(S.desc(path)['sha256'] == digest, 'Reviewed postprocessing prerequisite source changed')
    registration = S.read(S.OUT / 'registration.json')
    reg_binding = S.desc(S.OUT / 'registration.json')
    control_path = E.CONTROL / 'complete.json'
    control = S.read(control_path)
    require(control.get('source') == S.desc(E.SOURCE) and control.get('actual_evaluation_complete') is True and (control.get('all_plan_complete') is False), 'Actual E8 completion/controller source required')
    gate_binding = control['CPU_gate']
    gate = E.verify_gate(gate_binding)
    queue = S.read(S.checked(gate['registration']))
    terminal = S.read(Q6.ROOT / 'status.json')
    require(terminal.get('status') == 'all_training_CPU_gate_complete_evaluation_pending' and terminal.get('source') == S.desc(Q6.SOURCE) and (terminal.get('registration') == gate['registration']) and (terminal.get('CPU_gate') == gate_binding), 'Exact actual Q10 terminal/CPU gate required')
    source_review = S.read(S.checked(queue['source_review']))
    require(source_review.get('status') == 'new_repair_chain_sources_reviewed' and source_review.get('passed') is True and (source_review.get('reviewed_sources') == queue['sources']), 'Actual full new execution-source review required')
    S.checked(source_review['reviewer_source'])
    cpu_source = S.desc(SOURCE.with_name('audit_completed_finetune_checkpoints_v12.py'))
    cpu = S.read(S.checked(gate['CPU_audit']))
    rows = validate_CPU_shape(cpu, registration, reg_binding, cpu_source)
    for row in rows:
        folder = S.OUT / 'runs' / row['fit_id']
        for key, name in [('status_artifact', 'status.json'), ('metadata', 'metadata.json'), ('completion', 'complete.json')]:
            require(S.checked(row[key]) == (folder / name).resolve(), 'CPU12 JSON binding belongs to another fit')
        require({x['path'] for x in row['cpu_checkpoint_audit']['checkpoints']} == {(folder / name).relative_to(S.WORK).as_posix() for name in ('initial.pt', 'latest.pt', 'best.pt')}, 'CPU12 checkpoint descriptors belong to another fit')
    training = R.assert_test_unlocked()
    require(training['formal_fits_verified'] == 360, 'Full original360 byte-bound completion gate failed')
    runtime_path = S.checked(control['status'])
    require(runtime_path.name == 'status.json' and runtime_path.parent.parent == (E.CONTROL / 'runs').resolve(), 'Canonical actual E8 controller run status required')
    runtime = S.read(runtime_path)
    require(runtime.get('source') == S.desc(E.SOURCE) and runtime.get('status') == 'all_evaluation_complete' and (runtime.get('CPU_gate') == gate_binding) and (set(runtime['stages']) == {'mamba_linux', 'all_windows'}), 'Both actual owned E8 stages required')
    for mode, stage in runtime['stages'].items():
        require(type(stage.get('returncode')) is int and stage['returncode'] == 0, 'E8 child did not return exit0')
        worker = S.read(S.checked(stage['worker_report']))
        require(worker['status'] == 'complete' and worker['mode'] == mode and (worker['source'] == S.desc(E.SOURCE)) and (worker['CPU_gate'] == gate_binding), 'Exact complete E8 worker/source/CPU proof required')
    released = S.checked({'path': str(Path(control['status']['path']).parent / 'released_owner.json').replace('\\', '/'), 'sha256': S.desc(S.checked(control['status']).parent / 'released_owner.json')['sha256']})
    lease = S.read(released)
    require(lease['source'] == S.desc(E.SOURCE) and lease['CPU_gate'] == gate_binding and all((lease.get(key) is None for key in ('child_pid', 'child_token', 'child_identity'))), 'Actual E8 released-owner terminal missing')
    for mode, stage in runtime['stages'].items():
        worker_path = S.checked(stage['worker_report'])
        admission = S.read(worker_path.parent / 'worker_admission.json')
        actual_owner = S.read(worker_path.parent / 'worker_owner.json')
        require(admission['queue_owner_id'] == lease['id'] and admission['source'] == S.desc(E.SOURCE) and (admission['CPU_gate'] == gate_binding) and (admission['mode'] == mode), 'Actual E8 worker admission belongs to another released controller')
        require(actual_owner['source'] == S.desc(E.SOURCE) and actual_owner['CPU_gate'] == gate_binding and all((role in actual_owner['native_roles'] for role in lease['native_roles'])), 'Actual E8 worker/controller born identities differ')
    manifest = S.read(E.ROOT / 'complete.json')
    require(manifest.get('evaluator_execution') == E.execution_provenance(), 'Actual E8 completed manifest source differs')
    proof = E.completed_report_verified()
    require(proof.get('passed') is True and proof.get('formal_fits_verified') == 360 and (proof.get('supervised_fits_evaluated') == 345) and (proof.get('supervised_conditions') == 23) and (proof.get('model_constructions') == proof.get('gpu_forward_calls') == 0) and (proof.get('evaluator_execution') == E.execution_provenance()), 'Read-only full existing E8 verification required')
    require(control['evaluation_proof'] == proof and runtime['evaluation_proof'] == proof, 'Actual controller completion/proof mismatch')
    require(inspect()['status'] == 'metadata_ready', 'An owner or training status changed during postprocessing')
    return {'status': 'ready_after_full_E8_verification', 'source': S.desc(SOURCE), 'registration': reg_binding, 'q9_registration': gate['registration'], 'q9_terminal': S.desc(Q6.ROOT / 'status.json'), 'q5_terminal': S.desc(Q.QUEUE / 'status.json'), 'recovery3_registration': S.desc(Q.QUEUE / 'recovery_registration_v3.json'), 'execution_sources': queue['sources'], 'source_review': queue['source_review'], 'CPU_gate': gate_binding, 'CPU_audit': gate['CPU_audit'], 'CPU_auditor': cpu_source, 'evaluation_control_completion': S.desc(control_path), 'evaluation_released_owner': S.desc(released), 'evaluation_verification': proof, 'formal_fits_verified': 360, 'FT_verified': 135, 'CPU_loads_verified': 405, 'model_calls': 0, 'GPU_queries': 0, 'all_plan_complete': False, 'scope': 'Read-only existing completion verification; not a live GPU admission or final cost/acceptance.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    if args.verify_existing:
        print(json.dumps(verify_existing()))
        return
    if args.check_only:
        print(json.dumps(inspect()))
        return
    print(json.dumps({'status': 'preparation_only', 'model_calls': 0, 'GPU_queries': 0, 'outer_probability_reads': 0, 'writes': 0}))
if __name__ == '__main__':
    main()
