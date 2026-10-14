"""Sequentially dispatch the remaining195 fits after the first165-fit queue.

This dispatcher never computes a score or unlocks a test. Engineering failure
is preserved; independent conditions continue and dependent fits stay blocked.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from . import protocol as P, registry as R, storage as S

ROOT = S.OUT / 'continuation_queue'
FIRST = tuple(P.DIAGNOSTIC_CONDITIONS) + tuple(
    method + '-frozen-smallhead' for method in ('EEGPT', 'CSBrain', 'REVE', 'LUNA', 'CodeBrain', 'BIOT')) + (
    'CBraMod-frozen-smallhead', 'LaBraM-native5s-frozen-smallhead', 'LaBraM-native4s-frozen-smallhead')
REMAINING = tuple(c.condition_id for c in P.CONDITIONS if c.condition_id not in FIRST)
FINETUNE = tuple(c.condition_id for c in P.CONDITIONS if c.policy_key == 'finetune')


def jobs():
    """Each command has one GPU owner; fitting commands retain all15 folds."""
    result = [
        {'id': 'conformer-prepare', 'module': 'supervised', 'args': ['--conditions', 'EEGConformer-scratch', '--prepare'], 'conditions': ['EEGConformer-scratch'], 'dependencies': []},
        {'id': 'conformer-fit', 'module': 'supervised', 'args': ['--conditions', 'EEGConformer-scratch'], 'conditions': ['EEGConformer-scratch'], 'dependencies': ['conformer-prepare'], 'fits': 15},
        {'id': 'ssl-prepare', 'module': 'ssl', 'args': ['--prepare', '--logical-batch', '64', '--precision', 'bfloat16'], 'conditions': ['EEG2Rep-fold-local-ssl'], 'dependencies': []},
        {'id': 'ssl-fit', 'module': 'ssl', 'args': [], 'conditions': ['EEG2Rep-fold-local-ssl'], 'dependencies': ['ssl-prepare'], 'fits': 15},
        {'id': 'eeg2rep-supervised-prepare', 'module': 'supervised', 'args': ['--conditions', 'EEG2Rep-supervised', '--prepare'], 'conditions': ['EEG2Rep-supervised'], 'dependencies': ['ssl-fit']},
        {'id': 'eeg2rep-supervised-fit', 'module': 'supervised', 'args': ['--conditions', 'EEG2Rep-supervised'], 'conditions': ['EEG2Rep-supervised'], 'dependencies': ['eeg2rep-supervised-prepare'], 'fits': 15},
        {'id': 'mamba-check', 'module': 'linux_entry', 'args': ['--action', 'mamba-check'], 'conditions': ['EEGMamba-frozen-smallhead', 'EEGMamba-lastblock-finetune'], 'dependencies': [], 'linux': True},
        {'id': 'mamba-export', 'module': 'linux_entry', 'args': ['--action', 'mamba-export'], 'conditions': ['EEGMamba-frozen-smallhead'], 'dependencies': ['mamba-check'], 'linux': True},
        {'id': 'mamba-frozen-fit', 'module': 'frozen_heads', 'args': ['--conditions', 'EEGMamba-frozen-smallhead'], 'conditions': ['EEGMamba-frozen-smallhead'], 'dependencies': ['mamba-export'], 'fits': 15},
    ]
    for condition in FINETUNE:
        method = P.CONDITION_BY_ID[condition].method
        prepare_id = condition + '-prepare'
        if method == 'EEGMamba':
            result.extend([
                {'id': prepare_id, 'module': 'linux_entry', 'args': ['--action', 'mamba-finetune-prepare'], 'conditions': [condition], 'dependencies': ['mamba-check'], 'linux': True},
                {'id': condition + '-fit', 'module': 'linux_entry', 'args': ['--action', 'mamba-finetune'], 'conditions': [condition], 'dependencies': [prepare_id], 'fits': 15, 'linux': True},
            ])
        else:
            result.extend([
                {'id': prepare_id, 'module': 'finetune', 'args': ['--conditions', condition, '--prepare'], 'conditions': [condition], 'dependencies': []},
                {'id': condition + '-fit', 'module': 'finetune', 'args': ['--conditions', condition], 'conditions': [condition], 'dependencies': [prepare_id], 'fits': 15},
            ])
    ids = set()
    fitted = []
    for item in result:
        if item['id'] in ids or not set(item['dependencies']).issubset(ids):
            raise ValueError('Duplicate job or forward dependency')
        ids.add(item['id'])
        if item.get('fits'):
            fitted.extend(item['conditions'])
    if len(FIRST) != 11 or len(REMAINING) != 13 or sum(j.get('fits', 0) for j in result) != 195 or sorted(fitted) != sorted(REMAINING):
        raise ValueError('Continuation does not cover the exact remaining195 registered fits')
    return result


def linux_root():
    path = P.WORK.as_posix()
    if len(path) < 3 or path[1:3] != ':/':
        raise ValueError('Dispatcher expects its Windows workspace')
    return '/mnt/' + path[0].lower() + '/' + path[3:]


def command(job, *, resume=False):
    arguments = list(job['args'])
    if resume and job.get('fits'):
        arguments.append('--resume')
    module = 'tridep_ccml.eeg_all_candidates.' + job['module']
    if job.get('linux'):
        root = linux_root()
        python = root + '/tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/bin/python'
        return ['wsl.exe', '-d', 'TRI-DEP-Ubuntu-24.04', '--cd', root, '--exec', python, '-B', '-u', '-m', module, *arguments]
    return [str(P.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', '-m', module, *arguments]


def sources():
    module = Path(__file__).resolve().parent
    names = ('continuation_queue.py', 'linux_entry.py', 'ready_recovery.py', 'evaluate.py', 'supervised.py', 'ssl.py', 'finetune.py',
             'storage.py', 'protocol.py', 'registry.py', 'frozen_heads.py', 'export_features.py', 'check_eegmamba_gpu.py')
    paths = [module / name for name in names]
    paths.extend(sorted((module / 'adapters').glob('*.py')))
    paths.extend(module / name for name in ('test_finetune.py', 'test_supervised.py', 'test_ssl.py', 'test_continuation_queue.py', 'test_evaluate.py'))
    paths.append(S.OUT / 'ready_queue/initial_owner.json')
    return [S.desc(path) for path in paths]


def register():
    R.load_registration()
    path = ROOT / 'registration.json'
    if not path.exists():
        S.save(path, {'registry': R.registration_descriptor(), 'registered_at': S.now(), 'jobs': jobs(),
                     'sources': sources(), 'preceding_conditions': list(FIRST), 'remaining_conditions': list(REMAINING),
                     'formal_fits_in_this_batch': 195, 'one_gpu_sequential': True, 'test_forward_allowed': False,
                     'engineering_uses_train_only': True, 'fixed_ssl_logical_batch': 64,
                     'no_smaller_batch_or_replacement_backbone_on_failure': True}, immutable=True)
    value = S.read(path)
    if value['jobs'] != jobs() or value['registry'] != R.registration_descriptor():
        raise ValueError('Continuation registration changed')
    for source in value['sources']:
        S.checked(source)
    return value


def all_complete(conditions):
    statuses = [(c, seed, fold, R.fit_status(c, seed, fold))
                for c in conditions for seed in P.SEEDS for fold in P.FOLDS]
    if any(value['status'] != 'training_complete' for _, _, _, value in statuses):
        return False
    locks = {condition: R._condition_lock(condition) for condition in conditions}
    for condition, seed, fold, value in statuses:
        R._validate_complete(value['evidence'], R.fit_directory(condition, seed, fold))
        if value['condition_source_lock'] != locks[condition]:
            raise ValueError('Completed fit source lock changed')
    return True


def pid_running(pid):
    """Read-only Windows process query; never use os.kill(pid,0) on Windows."""
    if sys.platform != 'win32':
        raise RuntimeError('PID inspection requires the Windows dispatcher host')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return False
        raise OSError(error, 'Cannot prove training process has exited')
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError(ctypes.get_last_error(), 'Cannot inspect training process exit code')
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def acquire_owner():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'owner.json'
    value = {'id': uuid.uuid4().hex, 'pid': os.getpid(), 'started_at': S.now(), 'child_pid': None}
    # A crashed owner's record stays until its actual child has been inspected.
    # A second dispatcher cannot overwrite the active owner's status or lease.
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return value


def release_owner(owner):
    path = ROOT / 'owner.json'
    if path.exists() and S.read(path)['id'] == owner['id'] and not owner.get('child_cleanup_unverified'):
        path.unlink()


def terminate_child(process, job, linux_owner):
    """Reap both the Windows launcher and the actual Linux CUDA process group."""
    if job.get('linux'):
        if not linux_owner.exists():
            if process.poll() is None:
                raise RuntimeError('Linux owner record absent; retain exclusive lease for inspection')
        else:
            record = S.read(linux_owner)
            group = record.get('pgid')
            if not isinstance(group, int) or group <= 1 or group != record.get('pid'):
                raise ValueError('Unsafe Linux process group identity')
            for signal in ('TERM', 'KILL'):
                subprocess.run(['wsl.exe', '-d', 'TRI-DEP-Ubuntu-24.04', '--exec', '/bin/kill', '-' + signal, '--', '-' + str(group)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
                probe = subprocess.run(['wsl.exe', '-d', 'TRI-DEP-Ubuntu-24.04', '--exec', '/bin/kill', '-0', '--', '-' + str(group)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
                if probe.returncode != 0:
                    break
                time.sleep(.2)
            else:
                raise RuntimeError('Linux CUDA process group still alive; retain lease')
    if process.poll() is None:
        # Windows venv python.exe is a launcher with a separate real Python
        # child. Terminate the job's tree before reaping the launcher itself.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def postcondition(job):
    if job.get('fits'):
        return all_complete(job['conditions'])
    if job['id'] == 'evaluate-mamba-finetune':
        from .evaluate import completed_condition_verified
        value = completed_condition_verified('EEGMamba-lastblock-finetune')
        return value.get('passed') is True and value.get('supervised_fits_evaluated') == 15 and value.get('primary_oof_predictions') == 111
    if job['id'] == 'evaluate-all':
        from .evaluate import completed_report_verified
        value = completed_report_verified()
        return value.get('passed') is True and value.get('formal_fits_verified') == 360 and value.get('supervised_conditions') == 23 and value.get('supervised_fits_evaluated') == 345
    if job['id'] == 'mamba-check':
        value = S.read(S.REPORT / 'eegmamba_official_gpu_check.json')
        return value.get('passed') is True and value.get('official_mamba2_block_count') == 12 and value.get('frozen_parameters_unchanged_after_step') is True
    if job['id'] in ('mamba-export', 'first-codebrain-repair'):
        condition = job['conditions'][0]
        value = S.read(S.OUT / 'frozen_features' / condition / 'manifest.json')
        if value.get('passed') is not True or value.get('encoder_fit_subject_ids') != [] or value.get('frozen_state_exact') is not True:
            return False
        for source in value['sources']:
            S.checked(source)
        for row in value['records']:
            S.checked(row['features'])
        return True
    for condition in job['conditions']:
        R._condition_lock(condition)
    if job['id'] == 'ssl-prepare':
        value = S.read(S.OUT / 'ssl/engineering_settings.json')
        return value['actual_logical_batch'] == 64 and value['precision'] == 'bfloat16'
    return True


def evaluate_tail(owner):
    """The full360 gate is checked before either test-inference child starts."""
    try:
        for source in S.read(ROOT / 'registration.json')['sources']:
            S.checked(source)
        gate = R.assert_test_unlocked()
    except Exception as error:
        failure = ROOT / 'evaluation_gate_failures' / (str(time.time_ns()) + '.json')
        S.save(failure, {'passed': False, 'error_type': type(error).__name__, 'error': str(error),
                         'test_forward_calls_this_attempt': 0, 'checked_at': S.now()}, immutable=True)
        S.save(ROOT / 'evaluation_gate_status.json', {'passed': False, 'failure': S.desc(failure)})
        return False
    path = ROOT / 'evaluation_gate.json'
    if path.exists() and S.read(path).get('passed') is True:
        previous = S.read(path)
        if any(previous.get(key) != gate.get(key) for key in ('registration', 'formal_fits_verified', 'statuses')):
            raise ValueError('Previously opened evaluation gate refers to different completions')
    else:
        S.save(path, gate, immutable=True)
    S.save(ROOT / 'evaluation_gate_status.json', {'passed': True, 'gate': S.desc(path)})
    linux = {'id': 'evaluate-mamba-finetune', 'module': 'linux_entry', 'args': ['--action', 'mamba-evaluate'],
             'conditions': ['EEGMamba-lastblock-finetune'], 'dependencies': [], 'linux': True}
    aggregate = {'id': 'evaluate-all', 'module': 'evaluate', 'args': [], 'conditions': [], 'dependencies': []}
    if run_child(linux, owner) != 'complete':
        return False
    return run_child(aggregate, owner) == 'complete'


def run_child(job, owner, *, resume=False):
    folder = ROOT / 'attempts' / (str(time.time_ns()) + '-' + job['id'])
    arguments = command(job, resume=resume)
    linux_owner = folder / 'linux_owner.json'
    if job.get('linux'):
        arguments.extend(['--owner-record', linux_owner.relative_to(S.WORK).as_posix()])
    attempt = {'job': job, 'command': arguments, 'started_at': S.now(), 'status': 'running',
               'registration': S.desc(ROOT / 'registration.json')}
    S.save(folder / 'status.json', attempt)
    S.save(ROOT / 'status.json', {'status': 'running', 'job': job['id'], 'updated_at': S.now(), 'test_evaluation_frozen': True})
    print('QUEUE_START ' + job['id'], flush=True)
    started = time.monotonic()
    process = None
    code = None
    try:
        with (folder / 'process.log').open('ab') as log:
            process = subprocess.Popen(arguments, cwd=P.WORK, stdout=log, stderr=subprocess.STDOUT)
            owner.update(child_pid=process.pid, child_job=job['id'], linux_owner=linux_owner.relative_to(S.WORK).as_posix())
            S.save(ROOT / 'owner.json', owner)
            attempt['child_pid'] = process.pid
            S.save(folder / 'status.json', attempt)
            code = process.wait()
        try:
            good = code == 0 and postcondition(job)
        except Exception as error:
            good = False
            attempt['postcondition_error'] = {'type': type(error).__name__, 'message': str(error)}
        attempt.update(status='complete' if good else 'failed', returncode=code, seconds=time.monotonic()-started,
                       finished_at=S.now(), log=S.desc(folder / 'process.log'))
        S.save(folder / 'status.json', attempt)
        print('QUEUE_END ' + job['id'] + ' ' + attempt['status'], flush=True)
        return attempt['status']
    finally:
        if process is not None:
            try:
                if process.poll() is None or (job.get('linux') and code != 0):
                    terminate_child(process, job, linux_owner)
                owner.update(child_pid=None, child_job=None, linux_owner=None)
                S.save(ROOT / 'owner.json', owner)
            except BaseException:
                owner['child_cleanup_unverified'] = True
                S.save(ROOT / 'owner.json', owner)
                raise


def block_dependent(job, reason):
    for condition in job['conditions']:
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                if R.fit_status(condition, seed, fold)['status'] != 'training_complete':
                    R.record_fit_status(condition, seed, fold, 'blocked', reason=reason)


def wait_for_first(owner=None, *, resume=False):
    S.save(ROOT / 'status.json', {'status': 'waiting_for_preceding165', 'updated_at': S.now(),
                                'formal_fits_in_this_batch': 195, 'test_evaluation_frozen': True})
    path = S.OUT / 'ready_queue/status.json'
    while True:
        value = S.read(path) if path.exists() else {}
        if value.get('status') == 'failed':
            if value.get('condition') != 'CodeBrain-frozen-smallhead':
                raise RuntimeError('Preceding frozen queue failed outside the registered CodeBrain repair')
            initial = S.read(S.OUT / 'ready_queue/initial_owner.json')
            while any(pid_running(pid) for pid in initial['process_ids']):
                time.sleep(1)
            repair = {'id': 'first-codebrain-repair', 'module': 'ready_recovery', 'args': [],
                      'conditions': ['CodeBrain-frozen-smallhead'], 'dependencies': []}
            if run_child(repair, owner) != 'complete':
                raise RuntimeError('Registered CodeBrain repair failed; preserved for inspection')
            restart = {'id': 'resume-first-frozen135', 'module': 'ready_queue', 'args': ['--resume'],
                       'conditions': list(FIRST[2:]), 'dependencies': [], 'fits': 135}
            if run_child(restart, owner) != 'complete':
                raise RuntimeError('Preceding frozen queue failed after repair; preserved for inspection')
            continue
        if value.get('status') == 'registered_ready_batch_complete' and all_complete(FIRST):
            return
        time.sleep(10)


def run(*, resume=False):
    registration = register()
    owner = acquire_owner()
    try:
        wait_for_first(owner, resume=resume)
        outcomes = {}
        for job in registration['jobs']:
            for source in registration['sources']:
                S.checked(source)
            unsuccessful = [name for name in job['dependencies'] if outcomes.get(name) != 'complete']
            if unsuccessful:
                reason = 'Actual prerequisite jobs failed or were blocked: ' + ', '.join(unsuccessful)
                block_dependent(job, reason)
                outcomes[job['id']] = 'blocked'
                S.save(ROOT / 'status.json', {'status': 'running', 'last_job': job['id'], 'outcomes': outcomes,
                                            'updated_at': S.now(), 'test_evaluation_frozen': True})
                continue
            outcomes[job['id']] = run_child(job, owner, resume=resume)
            if outcomes[job['id']] != 'complete':
                block_dependent(job, 'Actual queue command failed: ' + job['id'] + '; see continuation_queue/attempts logs')
            S.save(ROOT / 'status.json', {'status': 'running', 'last_job': job['id'], 'outcomes': outcomes,
                                        'updated_at': S.now(), 'test_evaluation_frozen': True})
        summary = R.status_summary()
        evaluated = evaluate_tail(owner)
        gate_status = ROOT / 'evaluation_gate_status.json'
        opened = gate_status.exists() and S.read(gate_status).get('passed') is True
        S.save(ROOT / 'status.json', {'status': 'all_training_complete' if all_complete(REMAINING) else 'required_work_remaining',
                                    'outcomes': outcomes, 'registry_status': summary, 'updated_at': S.now(),
                                    'test_evaluation_completed': evaluated,
                                    'test_evaluation_frozen': not opened,
                                    'test_evaluation_status': 'complete' if evaluated else 'pending_or_blocked'})
        print('Continuation dispatch finished; evaluation completed=' + str(evaluated), flush=True)
    except BaseException as error:
        S.save(ROOT / 'failure_{}.json'.format(time.time_ns()), {'error_type': type(error).__name__, 'error': str(error), 'time': S.now()})
        S.save(ROOT / 'status.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                                    'updated_at': S.now(), 'test_evaluation_frozen': True})
        raise
    finally:
        release_owner(owner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    if args.register_only:
        register()
    else:
        run(resume=args.resume)
