"""Execution-only v3: correct Linux owner routing and qualified FT-v2 modules."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

from . import continuation_queue as Q1, continuation_queue_v2 as Q2, registry as R, storage as S

ROOT = S.OUT / 'continuation_queue_v3'
SOURCE = Path(__file__).resolve()
NEW_SOURCES = ('continuation_queue_v3.py', 'linux_entry_v2.py', 'handoff_supervisor_v3.py',
               'test_continuation_queue_v3.py', 'finetune_v2.py', 'adapters/luna_v2.py', 'test_finetune_v2.py',
               'evaluate_v2.py', 'test_evaluate_v2.py')


def jobs():
    mapping = {'finetune': 'finetune_v2', 'linux_entry': 'linux_entry_v2'}
    values = [{**job, 'module': mapping.get(job['module'], job['module'])} for job in Q2.jobs()]
    for original, changed in zip(Q2.jobs(), values):
        restore = dict(changed)
        restore['module'] = {'finetune_v2': 'finetune', 'linux_entry_v2': 'linux_entry'}.get(restore['module'], restore['module'])
        if restore != original:
            raise ValueError('V3 may only change qualified execution modules')
    if sum(job.get('fits', 0) for job in values) != 195:
        raise ValueError('Original195-fit job identities must remain unchanged')
    return values


def sources():
    path = Q2.ROOT / 'registration.json'
    if not path.exists():
        raise ValueError('Existing frozen v2 registration required')
    previous = S.read(path)
    for source in previous['sources']:
        S.checked(source)
    added = [S.desc(SOURCE.parent / name) for name in NEW_SOURCES] + [S.desc(path)]
    return list({record['path']: record for record in previous['sources'] + added}.values())


def register():
    R.load_registration()
    path = ROOT / 'registration.json'
    if not path.exists():
        S.save(path, {'version': 3, 'registry': R.registration_descriptor(), 'registered_at': S.now(),
                     'jobs': jobs(), 'sources': sources(), 'previous_execution_registration': S.desc(Q2.ROOT / 'registration.json'),
                     'formal_fits_in_reused_matrix': 195, 'new_remaining_fits_after_ssl_handoff': 165,
                     'completed_conformer_and_ssl_reused': 30, 'one_gpu_sequential': True,
                     'linux_owner_record_root': 'continuation_queue/attempts/v3_<unique>',
                     'attempt_status_and_log_root': 'continuation_queue_v3/attempts',
                     'training_protocol_matrix_or_hyperparameters_changed': False,
                     'requires_ssl_batch_complete_handoff': True}, immutable=True)
    value = S.read(path)
    if value['jobs'] != jobs() or value['registry'] != R.registration_descriptor():
        raise ValueError('Frozen v3 execution registration changed')
    for source in value['sources']:
        S.checked(source)
    return value


def corrected_run_child(driver):
    """Recompile exactly one owner-path expression, preserving all other AST."""
    tree = ast.parse(Path(Q1.__file__).read_text(encoding='utf-8'))
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_child')
    changed = 0
    for node in ast.walk(original):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'linux_owner':
            expected = ast.parse("folder / 'linux_owner.json'", mode='eval').body
            if ast.dump(node.value, include_attributes=False) != ast.dump(expected, include_attributes=False):
                raise ValueError('Original Linux owner expression changed')
            node.value = ast.parse("(S.OUT / 'continuation_queue/attempts' / ('v3_' + folder.name) / 'linux_owner.json') if job.get('linux') else (folder / 'linux_owner.json')", mode='eval').body
            changed += 1
    if changed != 1:
        raise ValueError('V3 owner-path rewrite must affect exactly one assignment')
    program = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
    exec(compile(program, str(SOURCE) + ':one-owner-path-rewrite', 'exec'), driver.__dict__)
    return driver.run_child


def isolated_driver():
    driver = Q2.isolated_driver()
    driver.ROOT = ROOT
    driver.register = register; driver.jobs = jobs; driver.sources = sources
    corrected_run_child(driver)
    # evaluate_tail constructs two commands directly rather than through jobs.
    original_command = driver.command
    def command(job, *, resume=False):
        mapping = {'linux_entry': 'linux_entry_v2', 'evaluate': 'evaluate_v2'}
        routed = {**job, 'module': mapping.get(job['module'], job['module'])}
        return original_command(routed, resume=resume)
    driver.command = command
    original_postcondition = driver.postcondition
    def postcondition(job):
        if job['id'] in ('evaluate-mamba-finetune', 'evaluate-all'):
            from . import evaluate_v2 as E
            if job['id'] == 'evaluate-mamba-finetune':
                value = E.completed_condition_verified('EEGMamba-lastblock-finetune')
                return value.get('passed') is True and value.get('supervised_fits_evaluated') == 15 and value.get('primary_oof_predictions') == 111
            value = E.completed_report_verified()
            return value.get('passed') is True and value.get('formal_fits_verified') == 360 and value.get('supervised_conditions') == 23 and value.get('supervised_fits_evaluated') == 345
        return original_postcondition(job)
    driver.postcondition = postcondition
    return driver


def require_handoff():
    proof = S.read(ROOT / 'handoff_complete.json')
    if not (proof['passed'] is True and proof['old_q2_dispatcher_terminated_after_ssl_children'] is True
            and proof['launcher_exit_code'] == proof['gpu_child_exit_code'] == 0
            and proof['ssl_fits_verified'] == 15 and proof['v3_registration'] == S.desc(ROOT / 'registration.json')):
        raise ValueError('Reviewed full SSL batch handoff proof required')
    for record in proof['protected_sources'] + proof['ssl_completed_artifacts']:
        S.checked(record)
    if not Q1.all_complete(Q1.FIRST) or not Q1.all_complete(('EEGConformer-scratch', 'EEG2Rep-fold-local-ssl')):
        raise ValueError('FIRST165, Conformer15 and SSL15 must all verify')
    return proof


def run(*, resume=True):
    register(); require_handoff(); isolated_driver().run(resume=resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    register() if args.register_only else run(resume=args.resume)
