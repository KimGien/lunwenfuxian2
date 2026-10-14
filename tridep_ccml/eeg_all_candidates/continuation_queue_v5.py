"""Continue every mandatory fit while optional timing probes can be deferred."""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import time

from . import continuation_queue as Q1, continuation_queue_v4 as Q4, registry as R, storage as S

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'continuation_queue_v5'
NEW_MODULE_SOURCES = ('continuation_queue_v5.py', 'linux_entry_v4.py', 'acceleration_prepare_v2.py', 'test_continuation_queue_v5.py',
                      'sync_qualification_gate_v2.py', 'sync_qualification_gate_v1.py', 'sync_reduction_v1.py',
                      'sync_subject_update_v1.py', 'sync_state_digest_v1.py')
NEW_ANALYSIS_SOURCES = ('qualify_acceleration_v2.py', 'sync_qualification_boundary_guard_v2.py', 'sync_qualification_boundary_guard_v1.py',
                        'qualify_sync_reduction_v2.py', 'qualify_training_sync_v2.py', 'qualify_sync_reduction_v1.py', 'qualify_training_sync_v1.py')


def jobs():
    mapping = {'acceleration_prepare_v1': 'acceleration_prepare_v2', 'linux_entry_v3': 'linux_entry_v4'}
    values = [{**job, 'module': mapping.get(job['module'], job['module']),
               'args': list(job['args']), 'dependencies': list(job['dependencies'])} for job in Q4.jobs()]
    for before, after in zip(Q4.jobs(), values):
        restored = {**after, 'module': {'acceleration_prepare_v2': 'acceleration_prepare_v1',
                                       'linux_entry_v4': 'linux_entry_v3'}.get(after['module'], after['module'])}
        if restored != before:
            raise ValueError('Only optional qualification control and Linux wrapper routing may change')
    if len(values) != 36 or sum(j.get('fits', 0) for j in values) != 195:
        raise ValueError('Same original36 jobs/195 formal fits required')
    return values


def sources():
    previous = Q4.register()
    if len(previous['sources']) != 59:
        raise ValueError('All original59 Q4 sources must remain unchanged')
    added = ([S.desc(SOURCE.parent / name) for name in NEW_MODULE_SOURCES]
             + [S.desc(S.REPORT / name) for name in NEW_ANALYSIS_SOURCES]
             + [S.desc(Q4.ROOT / 'registration.json')])
    return list({r['path']: r for r in previous['sources'] + added}.values())


def register():
    R.load_registration()
    path = ROOT / 'registration.json'
    if not path.exists():
        S.save(path, {'version': 5, 'registry': R.registration_descriptor(), 'registered_at': S.now(),
                      'source': S.desc(SOURCE), 'previous_execution_registration': S.desc(Q4.ROOT / 'registration.json'),
                      'jobs': jobs(), 'sources': sources(), 'one_gpu_sequential': True,
                      'formal_fits_in_reused_matrix': 195, 'training_protocol_matrix_or_hyperparameters_changed': False,
                      'formal_runner': S.desc(SOURCE.parent / 'finetune_v3.py'),
                      'training_micro_subjects': 1, 'training_window_chunk': 8,
                      'diagnostic_window_chunk': 8, 'actual_expanded_batch_activation': False,
                      'changes': ['defer optional performance probes when GPU is shared',
                                  'verify and reuse completed engineering instead of rerunning it'],
                      'requires_safe_handoff': True, 'test_gate_unchanged': True}, immutable=True)
    value = S.read(path)
    if (value['version'] != 5 or value['source'] != S.desc(SOURCE) or value['registry'] != R.registration_descriptor()
            or value['previous_execution_registration'] != S.desc(Q4.ROOT / 'registration.json') or value['jobs'] != jobs()):
        raise ValueError('Frozen Q5 execution registration differs')
    for record in value['sources']: S.checked(record)
    return value


class ActualExecutionStorage:
    def __getattr__(self, name):
        return getattr(S, name)

    def save(self, path, value, *, immutable=False):
        target = Path(path).resolve()
        if target.is_relative_to(ROOT.resolve()) and target.name in ('status.json', 'owner.json'):
            value = {**value, 'source': S.desc(SOURCE), 'execution_registration': S.desc(ROOT / 'registration.json'),
                     'execution_version': 5}
        return S.save(path, value, immutable=immutable)


def corrected_run_child(driver):
    tree = ast.parse(Path(Q1.__file__).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_child')
    changed = 0
    for item in ast.walk(node):
        if (isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == 'linux_owner'):
            if ast.dump(item.value, include_attributes=False) != ast.dump(ast.parse("folder / 'linux_owner.json'", mode='eval').body, include_attributes=False):
                raise ValueError('Original owner assignment differs')
            item.value = ast.parse("(S.OUT / 'continuation_queue/attempts' / ('v5_' + folder.name) / 'linux_owner.json') if job.get('linux') else (folder / 'linux_owner.json')", mode='eval').body
            changed += 1
    if changed != 1: raise ValueError('Exactly one owner assignment must change')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), 'exec'), driver.__dict__)


def previous_completed_engineering(job, driver):
    if job.get('fits') or job['id'].endswith('-acceleration-prepare') or job['id'].startswith('evaluate-'):
        return None
    old = next((j for j in Q4.jobs() if j['id'] == job['id']), None)
    if old is None: return None
    candidates = []
    for path in sorted((Q4.ROOT / 'attempts').glob('*-' + job['id'] + '/status.json')):
        value = S.read(path)
        if (value.get('job') == old and value.get('status') == 'complete' and value.get('returncode') == 0
                and value.get('registration') == S.desc(Q4.ROOT / 'registration.json')):
            S.checked(value['log'])
            candidates.append(S.desc(path))
    if not candidates: return None
    if driver.postcondition(job) is not True:
        raise ValueError('Previously completed engineering no longer verifies: ' + job['id'])
    return candidates


def isolated_driver():
    driver = Q4.isolated_driver()
    driver.ROOT = ROOT; driver.register = register; driver.jobs = jobs; driver.sources = sources
    driver.S = ActualExecutionStorage()
    corrected_run_child(driver)
    actual_run_child = driver.run_child

    def run_child(job, owner, *, resume=False):
        fits_complete = bool(job.get('fits') or job['id'].endswith('-acceleration-prepare')) and job.get('conditions') and Q1.all_complete(job['conditions'])
        reused_engineering = None if fits_complete else previous_completed_engineering(job, driver)
        if fits_complete or reused_engineering:
            S.save(ROOT / 'reused_jobs' / (job['id'] + '.json'),
                   {'job': job, 'status': 'verified_existing_work_reused', 'checked_at': S.now(),
                    'source': S.desc(SOURCE), 'execution_registration': S.desc(ROOT / 'registration.json'),
                    'execution_version': 5, 'previous_engineering_attempts': reused_engineering,
                    'all_formal_fits_already_complete': bool(fits_complete), 'new_model_calls': 0,
                    'new_optimizer_updates': 0, 'old_qualification_and_report_bytes_preserved': True})
            return 'complete'
        return actual_run_child(job, owner, resume=resume)

    driver.run_child = run_child

    def command(job, *, resume=False):
        mapping = {'linux_entry': 'linux_entry_v4', 'evaluate': 'evaluate_v3'}
        routed = {**job, 'module': mapping.get(job['module'], job['module']), 'args': list(job['args'])}
        if job.get('fits') and job['conditions'][0] in Q1.FINETUNE:
            binding = S.desc(Q4.acceleration_registration(job['conditions'][0]))
            routed['args'].extend(['--acceleration-registration', binding['path'], '--acceleration-registration-sha256', binding['sha256']])
        return Q1.command(routed, resume=resume)

    driver.command = command
    # The inherited postcondition reads E3, F3 source sidecars, and the same
    # original condition locks. AP2 never claims an unexecuted GPU pass.
    tree = ast.parse(Path(Q1.__file__).read_text(encoding='utf-8'))
    original_tail = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_tail')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original_tail], type_ignores=[])), str(SOURCE), 'exec'), driver.__dict__)
    evaluation_tail = driver.evaluate_tail

    def cpu_verified_evaluation_tail(owner):
        if Q1.all_complete(tuple(c.condition_id for c in Q1.P.CONDITIONS)):
            path = S.REPORT / 'audit_completed_finetune_checkpoints_v4.py'
            spec = importlib.util.spec_from_file_location('q5_pre_test_ft_cpu_audit', path)
            auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
            result = auditor.audit()
            target = ROOT / ('pre_test_finetune_cpu_audit_' + str(time.time_ns()) + '.json')
            S.save(target, result, immutable=True)
            if not (result['all_135_finetune_checkpoints_verified'] is True and not result['errors']
                    and not result['deferred_completed_fits'] and result['checkpoint_loads'] == 405):
                raise ValueError('Real full135 FT CPU405 verification required before outer inference')
            S.save(ROOT / 'pre_test_finetune_cpu_gate.json',
                   {'passed': True, 'audit': S.desc(target), 'checkpoint_loads': 405,
                    'test_forward_calls_this_check': 0, 'source': S.desc(SOURCE),
                    'execution_registration': S.desc(ROOT / 'registration.json'), 'execution_version': 5})
        return evaluation_tail(owner)

    driver.evaluate_tail = cpu_verified_evaluation_tail
    return driver


def require_handoff():
    proof = S.read(ROOT / 'handoff_complete.json')
    expected = S.read(Q4.ROOT / 'registration.json')['sources'] + [S.desc(Q4.ROOT / 'registration.json')]
    if not (proof.get('passed') is True and proof.get('version') == 5
            and proof.get('q5_registration') == S.desc(ROOT / 'registration.json')
            and proof.get('old_q4_registration') == S.desc(Q4.ROOT / 'registration.json')
            and proof.get('old_dispatcher_terminated_after_completed_children') is True
            and proof.get('launcher_exit_code') == proof.get('gpu_child_exit_code') == 0
            and proof.get('completed_CSBrain_fits_verified') == 15
            and proof.get('GPU_training_child_suspend_or_terminate_calls') == 0
            and proof.get('protected_sources') == expected and proof.get('completed_CSBrain_artifacts')
            and proof.get('source') == S.desc(S.REPORT / 'sync_qualification_boundary_guard_v2.py')):
        raise ValueError('Actual source-bound natural Q4-to-Q5 handoff is required')
    for record in proof['protected_sources'] + proof['completed_CSBrain_artifacts']: S.checked(record)
    owner = S.read(S.checked(proof['old_owner_snapshot']))
    old_attempt = S.read(S.checked(proof['old_attempt_snapshot']))
    if not (owner.get('id') == '3266366ec77046d09bbd58372bd9bee5' and owner.get('pid') == 40840
            and owner.get('child_pid') == 18224 and owner.get('child_job') == 'CSBrain-lastblock-finetune-fit'
            and old_attempt.get('status') == 'running' and old_attempt.get('child_pid') == 18224
            and old_attempt.get('registration') == S.desc(Q4.ROOT / 'registration.json')
            and old_attempt.get('job') == next(j for j in Q4.jobs() if j['id'] == 'CSBrain-lastblock-finetune-fit')
            and type(proof.get('old_dispatcher_exit_code')) is int and proof['old_dispatcher_exit_code'] == 0xEED80008):
        raise ValueError('Exact archived Q4 owner/attempt and held parent exit proof required')
    roles = proof.get('held_process_identities', {})
    expected_roles = {'dispatcher': (40840, 15428, '134360214625265856'),
                      'launcher': (18224, 40840, '134360215362961641'),
                      'gpu_child': (32880, 18224, '134360215363064824')}
    if set(roles) != set(expected_roles) or any((roles[name].get('pid'), roles[name].get('parent_pid'), roles[name].get('creation_100ns')) != value for name, value in expected_roles.items()):
        raise ValueError('Exact held process object identities differ')
    actual_artifacts = []
    for seed in Q1.P.SEEDS:
        for fold in Q1.P.FOLDS:
            directory = R.fit_directory('CSBrain-lastblock-finetune', seed, fold)
            status = R.fit_status('CSBrain-lastblock-finetune', seed, fold)
            R._validate_complete(status['evidence'], directory)
            actual_artifacts.extend(status['evidence'][key] for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'))
            actual_artifacts.extend(S.desc(directory / name) for name in ('complete.json', 'status.json'))
    if len(actual_artifacts) != 90 or proof['completed_CSBrain_artifacts'] != actual_artifacts:
        raise ValueError('Actual fifteen CSBrain completed artifact coverage differs')
    if (Q4.ROOT / 'owner.json').exists() or not Q1.all_complete(('CSBrain-lastblock-finetune',)):
        raise ValueError('Old owner has not been archived or full CSBrain evidence changed')
    return proof


def run(*, resume=False):
    register(); require_handoff(); isolated_driver().run(resume=resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    register() if args.register_only else run(resume=args.resume)
