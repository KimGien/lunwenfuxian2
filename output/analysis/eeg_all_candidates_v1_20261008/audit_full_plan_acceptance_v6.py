"""Recovery-aware Q5 lifecycle checks around the sealed wide V4 mathematics.

The delegate retains every training, CPU-checkpoint, evaluation, cost and
semantic requirement. New checks read only JSON/source lineage and the existing
Q5 pre-test CPU proof. They never create a test gate or load another checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import types

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
REPORT = SOURCE.parent
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
QUEUE = OUT / 'continuation_queue_v5'
DELEGATE = REPORT / 'audit_full_plan_acceptance_v4.py'
DELEGATE_SHA = '6e659d2ca7a9d3a364f04aaef8d17385608d7db6a25e9e8b5e941f657b440f84'
CPU_SHAPE_SOURCE = REPORT / 'supplemental_head_resource_measurement_v3.py'
CPU_SHAPE_SHA = '3d668df402b8bbe5d269f590ade377bd1ca05120630792b92965fa4e338a2e76'
Q4_SHA = 'c831bca3afad5574a7bc084fa4a0ebcaec7bea629679d7c1b426171092f8b089'
Q3_SHA = 'e135585ca475b733bbc6bfbaa1616fe71f9b8fb1af8f2c0b413d2a9c8565b3a0'


def require(value, message):
    if not value:
        raise ValueError(message)


def load_pinned(path, digest, name):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, 'Sealed delegate changed: ' + path.name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


D = load_pinned(DELEGATE, DELEGATE_SHA, '_wide_acceptance_v5_math_delegate')
D.SOURCE = SOURCE
S, R = D.S, D.R
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v1 as L

PREVIOUS_V5 = REPORT / "audit_full_plan_acceptance_v5.py"
require(S.desc(PREVIOUS_V5)["sha256"] == "bd2d88b30a0e2712896c8e9d939d69f20e18a3b08e67355eec48e41708db458b", "Sealed wideV5 changed")


def bind_recovery_consumers():
    """Change only source/version constants in private original functions."""
    tree = ast.parse(DELEGATE.read_text(encoding="utf8"))
    for name, replacements in (("check_fit", {"audit_completed_finetune_checkpoints_v4.py": "audit_completed_finetune_checkpoints_v5.py"}),
                               ("final_cost_audit", {"cost_aggregation_v5.py": "cost_aggregation_v6.py", 5: 6})):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        counts = {key: 0 for key in replacements}
        for item in ast.walk(node):
            if isinstance(item, ast.Constant) and type(item.value) in (str, int) and item.value in replacements:
                counts[item.value] += 1; item.value = replacements[item.value]
        require(all(count == 1 for count in counts.values()), "Unexpected private source/version replacement count")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), D.__dict__)


bind_recovery_consumers()


def __getattr__(name):
    """Expose unchanged mathematical helpers and V5-bound semantic binding."""
    return getattr(D, name)


def json_bound(binding, *, expected=None):
    require(Path(binding['path']).suffix == '.json', 'Lifecycle checks read only JSON artifacts')
    path = S.checked(binding)
    if expected is not None:
        require(path == Path(expected).resolve(), 'JSON binding belongs to another lifecycle or fit')
    return S.read(path)


def verify_job_identity(previous, current):
    require(len(previous) == len(current) == 36, 'Original 36 queue jobs required')
    old = {row['id']: row for row in previous}
    new = {row['id']: row for row in current}
    require(len(old) == len(new) == 36 and set(old) == set(new), 'Queue job identities differ')
    routes = {'acceleration_prepare_v1': 'acceleration_prepare_v2', 'linux_entry_v3': 'linux_entry_v4'}
    for ident, row in new.items():
        expected = {**old[ident], 'module': routes.get(old[ident]['module'], old[ident]['module'])}
        require(row == expected, 'Only reviewed preparation/Linux execution routing may change: ' + ident)
    require(sum(row.get('fits', 0) for row in current) == 195, 'Original 195 reused formal identities required')


def queue_lineage(registration):
    registry = R.registration_descriptor()
    q3_path, q4_path = OUT / 'continuation_queue_v3/registration.json', OUT / 'continuation_queue_v4/registration.json'
    q3_binding, q4_binding = S.desc(q3_path), S.desc(q4_path)
    require(q3_binding['sha256'] == Q3_SHA and q4_binding['sha256'] == Q4_SHA, 'Sealed Q3/Q4 registration bytes changed')
    q3, q4 = json_bound(q3_binding), json_bound(q4_binding)
    require(len(q3['sources']) == 47 and len(q4['sources']) == 59
            and q3['registry'] == q4['registry'] == registry
            and q4['previous_execution_registration'] == q3_binding,
            'Original source lineage differs')
    binding = S.desc(QUEUE / 'registration.json')
    q5 = json_bound(binding)
    require(type(q5.get('version')) is int and q5['version'] == 5
            and q5['registry'] == registry and q5['previous_execution_registration'] == q4_binding,
            'Q5 registration must retain the exact original registry and Q4 lineage')
    require(type(q5.get('formal_fits_in_reused_matrix')) is int and q5['formal_fits_in_reused_matrix'] == 195
            and q5.get('one_gpu_sequential') is True
            and q5.get('training_protocol_matrix_or_hyperparameters_changed') is False
            and type(q5.get('training_micro_subjects')) is int and q5['training_micro_subjects'] == 1
            and type(q5.get('training_window_chunk')) is int and q5['training_window_chunk'] == 8
            and q5.get('test_gate_unchanged') is True, 'Q5 changed formal training or evaluation contract')
    verify_job_identity(q4['jobs'], q5['jobs'])
    sources = q5['sources']
    require(len({r['path'] for r in sources}) == len(sources)
            and all(r in sources for r in q4['sources'] + [q4_binding]),
            'Q5 must preserve every Q4 source and original registration descriptor')
    for row in sources:
        require(Path(row['path']).suffix in ('.py', '.json'), 'Queue lineage is source/JSON only')
        S.checked(row)
    required = ['continuation_queue_v5.py', 'acceleration_prepare_v2.py', 'linux_entry_v4.py',
                'finetune_v3.py', 'evaluate_v3.py', 'execution_acceleration_v1.py']
    for name in required:
        require(S.desc(WORK / 'tridep_ccml/eeg_all_candidates' / name) in sources,
                'Actual Q5 execution source is unbound: ' + name)
    cpu_source = S.desc(REPORT / 'audit_completed_finetune_checkpoints_v4.py')
    require(cpu_source in sources, 'Actual FT CPU auditor must remain registered')
    require(len(registration['fits']) == 360 and registration['formal_fits'] == 360, 'Original complete matrix required')
    return q5, {'passed': True, 'q3_registration': q3_binding, 'q4_registration': q4_binding,
                'q5_registration': binding, 'q5_source': S.desc(WORK / 'tridep_ccml/eeg_all_candidates/continuation_queue_v5.py'),
                'preserved_Q4_sources': 59, 'queue_jobs': 36, 'formal_fits_in_reused_matrix': 195,
                'training_math_unchanged': True, 'source_descriptors_verified': len(sources)}


def pre_test_cpu_gate(registration, q5):
    path = QUEUE / 'pre_test_finetune_cpu_gate.json'
    gate_binding = S.desc(path)
    gate = json_bound(gate_binding)
    require(gate.get('passed') is True and type(gate.get('checkpoint_loads')) is int and gate['checkpoint_loads'] == 405
            and type(gate.get('test_forward_calls_this_check')) is int and gate['test_forward_calls_this_check'] == 0,
            'Q5 pre-test proof must record 405 actual CPU loads and no test inference')
    audit_path = (WORK / gate['audit']['path']).resolve()
    require(audit_path.parent == QUEUE.resolve() and audit_path.name.startswith('pre_test_finetune_cpu_audit_'),
            'Actual CPU report must belong to Q5 pre-test lifecycle')
    value = json_bound(gate['audit'])
    recovery = L.verify_recovery_registration()
    cpu_source = S.desc(REPORT / 'audit_completed_finetune_checkpoints_v5.py')
    require(value.get('auditor_source') == cpu_source and cpu_source in recovery['value']['sources']
            and type(value.get('schema_version')) is int and value['schema_version'] == 5
            and gate.get('recovery_registration') == recovery['registration'], 'Actual CPU5/recovery binding differs')
    require(value.get('delegated_sources', {}).get('lost_f3_closure_verifier') == S.desc(Path(L.__file__))
            and value['delegated_sources'].get('previous_cpu_auditor_v4') == S.desc(REPORT / 'audit_completed_finetune_checkpoints_v4.py'),
            'CPU5 closure/math delegates differ')
    shape = load_pinned(CPU_SHAPE_SOURCE, CPU_SHAPE_SHA, '_wide_v5_cpu_evidence_shape')
    rows = shape.verify_cpu_shape({**value, "schema_version": 4}, registration, R.registration_descriptor())
    for row in rows:
        folder = OUT / 'runs' / row['fit_id']
        for key, name in (('status_artifact', 'status.json'), ('metadata', 'metadata.json'), ('completion', 'complete.json')):
            json_bound(row[key], expected=folder / name)
        require({r['path'] for r in row['cpu_checkpoint_audit']['checkpoints']} == {
            (folder / name).relative_to(WORK).as_posix() for name in ('initial.pt', 'latest.pt', 'best.pt')},
            'CPU report checkpoint descriptors belong to another fit')
    return {'passed': True, 'gate': gate_binding, 'audit': gate['audit'], 'auditor_source': cpu_source,
            'actual_verified_finetune_identities': 135, 'actual_cpu_checkpoint_loads': 405,
            'new_checkpoint_loads_by_this_lifecycle_check': 0}


def queue_terminal(registration):
    q5, lineage = queue_lineage(registration)
    path = QUEUE / 'status.json'
    binding = S.desc(path)
    state = json_bound(binding)
    require(state.get('source') == lineage['q5_source']
            and state.get('execution_registration') == lineage['q5_registration']
            and type(state.get('execution_version')) is int and state['execution_version'] == 5,
            'Terminal queue evidence is not actual source-bound Q5 execution')
    require(state.get('status') == 'all_training_complete' and state.get('test_evaluation_completed') is True
            and state.get('test_evaluation_frozen') is False and state.get('test_evaluation_status') == 'complete',
            'Q5 training/evaluation lifecycle remains incomplete')
    outcomes = state.get('outcomes', {})
    require(set(outcomes) == {job['id'] for job in q5['jobs']} and all(value == 'complete' for value in outcomes.values()),
            'Every registered Q5 job must genuinely complete')
    for name in ('continuation_queue_v3', 'continuation_queue_v4', 'continuation_queue_v5'):
        require(not (OUT / name / 'owner.json').exists(), 'An old/current execution owner remains: ' + name)
    recovery_proof = verify_recovery_terminal(state)
    cpu = pre_test_cpu_gate(registration, q5)
    require(binding == S.desc(path) and lineage['q5_registration'] == S.desc(QUEUE / 'registration.json'),
            'Terminal lifecycle bytes changed during verification')
    return {**lineage, 'terminal_status': binding, 'all_registered_jobs_completed': True,
            'owners_released': True, 'pre_test_finetune_cpu_verification': cpu,
            'recovery_execution_proof': recovery_proof}


def verify_recovery_terminal(state):
    lineage = L.verify_recovery_registration()
    expected = {'recovery_registration': lineage['registration'],
                'recovery_execution_source': lineage['source'],
                'recovery_closure': state.get('recovery_closure')}
    require(all(state.get(key) == value for key, value in expected.items())
            and isinstance(expected['recovery_closure'], dict), 'Actual terminal recovery execution fields differ')
    closure = L.read_json(expected['recovery_closure'])
    original = L.read_json(closure['original_attempt'])
    fit = {key: closure[key] for key in ('condition', 'seed', 'fold')}
    proof = L.verify_lost_f3_attempt(closure['original_attempt'], original, fit)
    require(proof is not None and proof.get('passed') is True
            and proof['closure'] == expected['recovery_closure']
            and proof['recovery_registration'] == lineage['registration'], 'Actual terminal lost-process proof differs')
    return expected


_delegate_evaluation_audit = D.evaluation_audit
_delegate_final_cost_audit = D.final_cost_audit


def evaluation_audit(registration):
    # This hook applies only to the private module; sealed V4 remains untouched.
    queue_terminal(registration)
    return _delegate_evaluation_audit(registration)


D.evaluation_audit = evaluation_audit


def final_cost_audit(path, companion, training_rows):
    result = _delegate_final_cost_audit(path, companion, training_rows)
    lifecycle = queue_terminal(R.load_registration(verify_sources=False))
    expected = {'registration': lifecycle['q5_registration'], 'terminal_status': lifecycle['terminal_status'],
                'source': lifecycle['q5_source'], 'execution_version': 5,
                'previous_q4_registration': lifecycle['q4_registration']}
    require(S.read(path).get('completion_gate', {}).get('q5_execution_proof') == expected,
            'Final cost snapshot must bind the same actual Q5 completion lifecycle')
    data = S.read(path)
    recovery = lifecycle['recovery_execution_proof']
    require(data.get('completion_gate', {}).get('recovery_execution_proof') == recovery,
            'Final cost must bind the actual recovery execution')
    closure_verification = data.get('lost_f3_closure_verification', {})
    require(closure_verification.get('passed_for_observed_closures') is True
            and closure_verification.get('errors') == []
            and any(row.get('closure') == recovery['recovery_closure']
                    and row.get('aggregation_role') == 'primary_observed_invocation'
                    for row in data.get('source_bound_lost_f3_attempts', [])), 'Verified original lost-F3 primary evidence absent')
    return {**result, 'q5_execution_proof': expected, 'recovery_execution_proof': recovery}


D.final_cost_audit = final_cost_audit


def audit(*, cost_artifact=None, cost_report=None, semantic_review=None):
    value = D.audit(cost_artifact=cost_artifact, cost_report=cost_report, semantic_review=semantic_review)
    lifecycle = {'passed': False, 'status': 'pending'}
    try:
        lifecycle = queue_terminal(R.load_registration(verify_sources=False))
    except Exception as error:
        lifecycle.update(error_type=type(error).__name__, reason=str(error))
    value.update(schema_version=6, source=S.desc(SOURCE), delegated_v4_auditor_source=S.desc(DELEGATE),
                 cpu_proof_shape_source=S.desc(CPU_SHAPE_SOURCE), q5_execution_lifecycle=lifecycle,
                 previous_v5_auditor_source=S.desc(PREVIOUS_V5), recovery_verifier_source=S.desc(Path(L.__file__)),
                 actual_cpu_execution_version=5, actual_cost_schema_version=6,
                 original_training_evaluation_and_cost_math_unchanged=True)
    if lifecycle.get('passed') is not True:
        value['objective_satisfied'] = value['full_plan_completed'] = False
        for row in value['requirements']:
            if row['id'] in ('R01', 'R29', 'R30'):
                row.update(status='pending', reason='Actual Q5 source/terminal/owner/pre-test CPU proof remains pending: ' + lifecycle['reason'])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cost-artifact', type=Path)
    parser.add_argument('--cost-report', type=Path)
    parser.add_argument('--semantic-review', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    value = audit(cost_artifact=args.cost_artifact, cost_report=args.cost_report, semantic_review=args.semantic_review)
    path = (args.output or REPORT / ('full_plan_acceptance_audit_v6_' + str(time.time_ns()) + '.json')).resolve()
    require(path.is_relative_to(REPORT), 'Save a new audit only in the analysis directory')
    S.save(path, value, immutable=True)
    print(json.dumps({'artifact': S.desc(path), 'full_plan_completed': value['full_plan_completed'],
                      'q5_lifecycle_verified': value['q5_execution_lifecycle']['passed'],
                      'training_verified': value['completed_training_json_verified'],
                      'test_predictions_read': value['outer_test_artifacts_read']}))
    return 1 if args.require_complete and not value['full_plan_completed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
