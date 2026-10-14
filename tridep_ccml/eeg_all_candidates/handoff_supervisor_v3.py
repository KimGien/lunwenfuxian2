"""Scoped SSL-batch handoff: only Q2 dispatcher41664 is controlled, never its GPU tree."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import types
import uuid

from . import continuation_queue as Q1, continuation_queue_v2 as Q2, continuation_queue_v3 as Q3
from . import handoff_supervisor_v2 as H2, registry as R, storage as S

ROOT = Q3.ROOT / 'handoff'
ROLE_PIDS = {'dispatcher': 41664, 'launcher': 48392, 'gpu_child': 40680}
ROLE_PARENTS = {'dispatcher': 21576, 'launcher': 41664, 'gpu_child': 48392}
CREATED_LOCAL = {'dispatcher': '2026-10-08T15:52:22.708496+08:00',
                 'launcher': '2026-10-08T15:52:42.101097+08:00',
                 'gpu_child': '2026-10-08T15:52:42.111535+08:00'}
OWNER_ID = '44fba8ab96f847cfaacca5fbee167c06'
SSL = 'EEG2Rep-fold-local-ssl'
TERMINATION_CODE = 0xEED30003


def require(value, message):
    if not value:
        raise ValueError(message)


def reviewed_ssl_v2():
    settings = S.read(S.OUT / 'ssl/engineering_settings.json')
    repair = settings.get('ssl_precision_repair', {})
    require(settings['actual_logical_batch'] == 64 and settings['precision'] == 'bfloat16'
            and settings['loss_precision'] == 'float32' and settings.get('loss_autocast_enabled') is False
            and repair.get('version') == 2, 'Only qualified real64/BF16/FP32-loss SSLv2 may be handed off')
    R._condition_lock(SSL)
    S.checked(repair['wrapper_source'])
    require(repair['wrapper_source'] == S.desc(Path(__file__).parent / 'ssl_v2.py'), 'SSLv2 actual source differs')


def boundary():
    owner = S.read(Q2.ROOT / 'owner.json')
    status = S.read(Q2.ROOT / 'status.json')
    require(owner['id'] == OWNER_ID and owner['pid'] == ROLE_PIDS['dispatcher']
            and owner.get('child_pid') == ROLE_PIDS['launcher'] and owner.get('child_job') == 'ssl-fit',
            'Exact task-owned Q2 SSL dispatcher boundary changed')
    require(status['status'] == 'running' and status.get('job') == 'ssl-fit', 'Q2 advanced beyond SSL batch')
    locator = owner.get('linux_owner')
    require(isinstance(locator, str), 'Original SSL attempt locator missing')
    path = (S.WORK / locator).resolve().with_name('status.json')
    require(path.is_relative_to((Q2.ROOT / 'attempts').resolve()), 'SSL attempt leaves its registered Q2 root')
    attempt = S.read(path)
    require(attempt['status'] == 'running' and attempt.get('child_pid') == ROLE_PIDS['launcher']
            and attempt['job']['id'] == 'ssl-fit' and attempt['job']['module'] == 'ssl_v2',
            'Only this experiment ssl_v2 full15-fit job may be controlled')
    reviewed_ssl_v2()
    return owner, path, attempt


def native_runtime():
    """Isolated copy of reviewed Win32 handle backend with only scoped constants replaced."""
    module = types.ModuleType(__package__ + '._handoff_v3_native')
    module.__package__ = __package__; module.__file__ = H2.__file__
    exec(compile(Path(H2.__file__).read_text(encoding='utf-8'), H2.__file__, 'exec'), module.__dict__)
    module.ROLE_PIDS = ROLE_PIDS; module.ROLE_PARENTS = ROLE_PARENTS
    module.CREATED_LOCAL = CREATED_LOCAL; module.DISPATCHER_TERMINATION_CODE = TERMINATION_CODE
    return module


def windows_backend():
    return native_runtime().WindowsBackend()


def identities(refs):
    return {ref['role']: H2.public_identity(ref) for ref in refs}


def inspect(backend):
    refs = []
    try:
        owner, _, _ = boundary()
        for role in ROLE_PIDS:
            refs.append(backend.capture(role, arm=False))
        require(all(not backend.exited(ref) for ref in refs), 'SSL process exited before reviewed inspection')
        require(boundary()[0] == owner, 'SSL boundary changed during inspection')
        return {'roles': identities(refs), 'owner_id': OWNER_ID, 'owner': S.desc(Q2.ROOT / 'owner.json'),
                'q2_registration': S.desc(Q2.ROOT / 'registration.json'), 'inspected_at': S.now(), 'process_actions': 0}
    finally:
        for ref in refs:
            backend.close(ref)


def ssl_artifacts():
    require(Q1.all_complete(Q1.FIRST) and Q1.all_complete(('EEGConformer-scratch', SSL)), 'FIRST165/Conformer15/SSL15 incomplete')
    records = []
    for seed in Q1.P.SEEDS:
        for fold in Q1.P.FOLDS:
            folder = R.fit_directory(SSL, seed, fold)
            status = R.fit_status(SSL, seed, fold); R._validate_complete(status['evidence'], folder)
            metadata = S.read(S.checked(status['evidence']['metadata']))
            require(metadata.get('ssl_precision_repair', {}).get('version') == 2
                    and metadata.get('loss_autocast_enabled') is False and metadata['online_backbone_update_verified'] is True,
                    'SSL completed fit lacks actual FP32-loss/online-update proof')
            records.extend(status['evidence'][key] for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'))
            records.append(S.desc(folder / 'status.json'))
    return records


def spawn_v3():
    with (ROOT / 'v3_dispatcher.log').open('ab') as log:
        return subprocess.Popen([str(S.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', '-m',
                    'tridep_ccml.eeg_all_candidates.continuation_queue_v3', '--resume'], cwd=S.WORK,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000)


def arm(expected, backend, *, launch=True, spawn=spawn_v3):
    Q3.register()
    require(expected['owner_id'] == OWNER_ID and set(expected['roles']) == set(ROLE_PIDS)
            and expected['q2_registration'] == S.desc(Q2.ROOT / 'registration.json'), 'Wrong reviewed Q2 SSL scope')
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'supervisor_owner.json').open('x', encoding='utf-8') as stream:
        json.dump({'id': uuid.uuid4().hex, 'pid': os.getpid(), 'started_at': S.now()}, stream)
    refs = []; paused = False
    state = {'phase': 'capturing_exact_ssl_handles', 'passed': False, 'expected': expected,
             'no_child_suspend_or_terminate': True, 'auto_resume_q2': False, 'v3_started': False,
             'h2_guardian_or_ancestors_operated': False}
    try:
        owner, attempt_path, attempt = boundary()
        for role in ROLE_PIDS:
            ref = backend.capture(role, arm=True); refs.append(ref)
            require(H2.public_identity(ref) == expected['roles'][role], 'Exact SSL process identity changed: ' + role)
        require(all(not backend.exited(ref) for ref in refs), 'SSL child exit raced with handoff capture')
        parent, launcher, gpu = refs
        prior = S.read(Q2.ROOT / 'registration.json')
        protected = prior['sources'] + [S.desc(Q2.ROOT / 'registration.json'), S.desc(Q2.ROOT / 'owner.json'), S.desc(attempt_path)]
        backend.suspend_dispatcher(parent); paused = True
        require(boundary()[0] == owner, 'Q2 boundary changed while suspending only its dispatcher')
        require(backend.descendants().issubset({ROLE_PIDS['launcher'], ROLE_PIDS['gpu_child']}), 'Unexpected Q2 child; keep parent suspended')
        state.update(phase='q2_suspended_ssl15_running', held_identities=identities(refs))
        S.save(ROOT / 'state.json', state)
        while not (backend.exited(launcher) and backend.exited(gpu)):
            require(boundary()[0] == owner, 'Suspended Q2 owner changed')
            require(backend.descendants().issubset({ROLE_PIDS['launcher'], ROLE_PIDS['gpu_child']}), 'Unexpected child while preserving SSL batch')
            backend.wait_tick()
        require(backend.exit_code(launcher) == backend.exit_code(gpu) == 0, 'SSL child failed; do not terminate parent or start Q3')
        artifacts = ssl_artifacts()
        for record in protected + artifacts:
            S.checked(record)
        state.update(phase='ssl15_complete_exit0_before_q2_termination')
        S.save(ROOT / 'state.json', state)
        backend.terminate_dispatcher(parent)
        require(backend.exited(parent), 'Exact Q2 dispatcher termination not confirmed')
        proof = {'passed': True, 'version': 3, 'finished_at': S.now(),
                 'v3_registration': S.desc(Q3.ROOT / 'registration.json'), 'held_identities': identities(refs),
                 'old_q2_dispatcher_terminated_after_ssl_children': True, 'old_dispatcher_exit_code': backend.exit_code(parent),
                 'launcher_exit_code': 0, 'gpu_child_exit_code': 0, 'ssl_fits_verified': 15,
                 'first165_and_conformer15_verified': True, 'old_owner_snapshot': owner, 'old_attempt_snapshot': attempt,
                 'old_attempt_observed_terminal_result': {'ssl15_complete': True, 'both_children_exit0': True},
                 'protected_sources': protected, 'ssl_completed_artifacts': artifacts,
                 'old_sources_and_registrations_preserved': True, 'h2_guardian_or_ancestors_operated': False,
                 'gpu_child_suspend_or_terminate_calls': 0, 'model_rng_optimizer_operations': 0}
        S.save(Q3.ROOT / 'handoff_complete.json', proof, immutable=True)
        if launch:
            process = spawn(); state.update(v3_started=True, v3_launcher_pid=process.pid)
        state.update(phase='handoff_complete', passed=True)
        S.save(ROOT / 'state.json', state)
        if launch:
            code = backend.wait_v2(process)
            state.update(phase='v3_dispatcher_finished', v3_exit_code=code, v3_finished_at=S.now())
            S.save(ROOT / 'state.json', state)
        return proof
    except BaseException as error:
        state.update(phase='failed_q2_kept_suspended' if paused else 'refused_before_suspend',
                     error_type=type(error).__name__, error=str(error), auto_resume_q2=False, safe_to_start_v3=False)
        S.save(ROOT / 'state.json', state)
        raise
    finally:
        for ref in refs:
            backend.close(ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true'); parser.add_argument('--arm', action='store_true')
    parser.add_argument('--expected'); parser.add_argument('--no-launch', action='store_true')
    args = parser.parse_args()
    require(args.inspect != args.arm, 'Choose one explicit readonly inspect or reviewed arm')
    backend = windows_backend()
    if args.inspect:
        print(json.dumps(inspect(backend), ensure_ascii=False, indent=2))
    else:
        require(args.expected is not None, 'Exact reviewed identities required')
        arm(S.read(Path(args.expected)), backend, launch=not args.no_launch)
