"""Scoped handoff of this task's dispatcher; GPU children are never suspended/killed.

Default inspection is read-only. Only explicit --arm may suspend the exact
dispatcher handle and later terminate that dispatcher after both Conformer
children exit0 and all15 completed artifacts verify. Failures never resume v1.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from . import continuation_queue as Q1, continuation_queue_v2 as Q2, registry as R, storage as S

ROOT = Q2.ROOT / 'handoff'
ROLE_PIDS = {'dispatcher': 39128, 'launcher': 2768, 'gpu_child': 33744}
ROLE_PARENTS = {'dispatcher': 36004, 'launcher': 39128, 'gpu_child': 2768}
CREATED_LOCAL = {'dispatcher': '2026-10-08T13:29:03.431918+08:00',
                 'launcher': '2026-10-08T13:56:01.862181+08:00',
                 'gpu_child': '2026-10-08T13:56:01.872091+08:00'}
OLD_OWNER_ID = 'b99db00fc1d04908b35c4d7508877bcd'
CONDITION = 'EEGConformer-scratch'
DISPATCHER_TERMINATION_CODE = 0xEED20002


def require(value, message):
    if not value:
        raise ValueError(message)


def no_old_ssl_started():
    condition = 'EEG2Rep-fold-local-ssl'
    require(not (S.OUT / 'ssl/engineering_settings.json').exists()
            and not (S.OUT / 'engineering' / (condition + '.json')).exists(), 'Old SSL engineering already started/completed')
    require(not list((S.OUT / 'runs' / condition).glob('seed_*/fold_*/started.json')), 'Old SSL fit already started')
    require(not list((S.OUT / 'ssl/engineering_attempts').glob('*.json')), 'Old SSL has an engineering execution attempt')


def check_old_boundary():
    owner = S.read(Q1.ROOT / 'owner.json')
    status = S.read(Q1.ROOT / 'status.json')
    require(owner['id'] == OLD_OWNER_ID and owner['pid'] == ROLE_PIDS['dispatcher']
            and owner.get('child_pid') == ROLE_PIDS['launcher'] and owner.get('child_job') == 'conformer-fit',
            'Exact old dispatcher/Conformer owner boundary changed')
    require(status['status'] == 'running' and status.get('job') == 'conformer-fit', 'Old dispatcher advanced beyond Conformer')
    no_old_ssl_started()
    return owner


def old_attempt(owner):
    require(isinstance(owner.get('linux_owner'), str), 'Original attempt locator missing')
    path = (S.WORK / owner['linux_owner']).resolve().with_name('status.json')
    require(path.is_relative_to((Q1.ROOT / 'attempts').resolve()), 'Original attempt locator leaves its queue')
    value = S.read(path)
    require(value['status'] == 'running' and value.get('child_pid') == ROLE_PIDS['launcher']
            and value['job']['id'] == 'conformer-fit' and value['job']['module'] == 'supervised', 'Original running attempt identity differs')
    return path, value


class WindowsBackend:
    """Native process-object handles, opened lazily only by explicit CLI use."""
    def __init__(self):
        require(sys.platform == 'win32', 'This scoped handoff requires Windows')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.ntdll = ctypes.WinDLL('ntdll', use_last_error=True)
        k = self.kernel
        k.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong); k.OpenProcess.restype = ctypes.c_void_p
        k.GetProcessTimes.argtypes = (ctypes.c_void_p,) + (ctypes.c_void_p,) * 4
        k.QueryFullProcessImageNameW.argtypes = (ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong))
        k.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong); k.WaitForSingleObject.restype = ctypes.c_ulong
        k.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        k.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint); k.TerminateProcess.restype = ctypes.c_int
        k.CloseHandle.argtypes = (ctypes.c_void_p,)
        k.CreateToolhelp32Snapshot.argtypes = (ctypes.c_ulong, ctypes.c_ulong); k.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        self.ntdll.NtSuspendProcess.argtypes = (ctypes.c_void_p,); self.ntdll.NtSuspendProcess.restype = ctypes.c_long

    def parents(self):
        class ProcessEntry(ctypes.Structure):
            _fields_ = [('size', ctypes.c_ulong), ('usage', ctypes.c_ulong), ('pid', ctypes.c_ulong),
                        ('heap', ctypes.c_size_t), ('module', ctypes.c_ulong), ('threads', ctypes.c_ulong),
                        ('parent', ctypes.c_ulong), ('priority', ctypes.c_long), ('flags', ctypes.c_ulong), ('name', ctypes.c_wchar * 260)]
        k = self.kernel
        k.Process32FirstW.argtypes = (ctypes.c_void_p, ctypes.POINTER(ProcessEntry))
        k.Process32NextW.argtypes = (ctypes.c_void_p, ctypes.POINTER(ProcessEntry))
        snapshot = k.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        result = {}
        try:
            entry = ProcessEntry(); entry.size = ctypes.sizeof(entry)
            present = k.Process32FirstW(snapshot, ctypes.byref(entry))
            while present:
                result[int(entry.pid)] = int(entry.parent)
                present = k.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            k.CloseHandle(snapshot)
        return result

    def capture(self, role, *, arm=False):
        pid = ROLE_PIDS[role]
        rights = 0x1000 | 0x00100000
        if role == 'dispatcher' and arm:
            rights |= 0x0800 | 0x0001  # suspend/resume and terminate only this parent
        handle = self.kernel.OpenProcess(rights, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            times = [ctypes.c_ulonglong() for _ in range(4)]
            if not self.kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            creation = times[0].value
            created = (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=creation // 10)).astimezone(timezone(timedelta(hours=8))).isoformat()
            size = ctypes.c_ulong(32768); image = ctypes.create_unicode_buffer(size.value)
            if not self.kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            parent = self.parents().get(pid)
            require(parent == ROLE_PARENTS[role] and created == CREATED_LOCAL[role], 'Creation time or parent identity changed: ' + role)
            return {'role': role, 'pid': pid, 'parent_pid': parent, 'creation_100ns': str(creation),
                    'created_local': created, 'image': image.value, 'handle': handle}
        except BaseException:
            self.kernel.CloseHandle(handle)
            raise

    def descendants(self):
        parents = self.parents(); found = set(); pending = [ROLE_PIDS['dispatcher']]
        while pending:
            parent = pending.pop()
            for pid, actual_parent in parents.items():
                if actual_parent == parent and pid not in found:
                    found.add(pid); pending.append(pid)
        return found

    def exited(self, ref):
        result = self.kernel.WaitForSingleObject(ref['handle'], 0)
        if result == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 0

    def exit_code(self, ref):
        require(self.exited(ref), 'Process object is not yet signaled')
        value = ctypes.c_ulong()
        if not self.kernel.GetExitCodeProcess(ref['handle'], ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.value

    def suspend_dispatcher(self, ref):
        require(ref['role'] == 'dispatcher' and ref['pid'] == ROLE_PIDS['dispatcher'], 'Only exact dispatcher may be suspended')
        require(self.ntdll.NtSuspendProcess(ref['handle']) == 0, 'Native dispatcher suspension failed; do not resume v1')

    def terminate_dispatcher(self, ref):
        require(ref['role'] == 'dispatcher' and ref['pid'] == ROLE_PIDS['dispatcher'], 'Only exact dispatcher may be terminated')
        if not self.kernel.TerminateProcess(ref['handle'], DISPATCHER_TERMINATION_CODE):
            raise ctypes.WinError(ctypes.get_last_error())
        require(self.kernel.WaitForSingleObject(ref['handle'], 15000) == 0, 'Dispatcher termination not confirmed')

    def close(self, ref):
        self.kernel.CloseHandle(ref['handle'])

    def wait_tick(self):
        time.sleep(2)

    def spawn_v2(self):
        path = ROOT / 'v2_dispatcher.log'
        with path.open('ab') as log:
            process = subprocess.Popen([str(S.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', '-m',
                        'tridep_ccml.eeg_all_candidates.continuation_queue_v2', '--resume'], cwd=S.WORK,
                        stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000)
        return process

    def wait_v2(self, process):
        # Keep this guardian and its tool job alive for the entire new driver.
        # Neither stdout nor stderr uses a parent-drained PIPE.
        return process.wait()


def public_identity(ref):
    return {key: value for key, value in ref.items() if key != 'handle'}


def capture_expected(backend):
    refs = []
    try:
        check_old_boundary()
        for role in ROLE_PIDS:
            refs.append(backend.capture(role, arm=False))
        require(all(not backend.exited(ref) for ref in refs), 'Expected process already exited')
        check_old_boundary()
        return {'roles': {ref['role']: public_identity(ref) for ref in refs}, 'old_owner_id': OLD_OWNER_ID,
                'old_owner': S.desc(Q1.ROOT / 'owner.json'), 'inspected_at': S.now(), 'process_actions': 0}
    finally:
        for ref in refs:
            backend.close(ref)


def conformer_artifacts():
    require(Q1.all_complete((CONDITION,)) and Q1.all_complete(Q1.FIRST), 'Conformer15 or preceding165 incomplete')
    records = []
    for seed in Q1.P.SEEDS:
        for fold in Q1.P.FOLDS:
            folder = R.fit_directory(CONDITION, seed, fold)
            status = R.fit_status(CONDITION, seed, fold)
            R._validate_complete(status['evidence'], folder)
            records.extend(status['evidence'][key] for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'))
            records.append(S.desc(folder / 'status.json'))
    return records


def arm(expected, backend, *, launch=True):
    """Root must explicitly run this after reviewing immutable exact identities."""
    registration = Q2.register()
    require(expected['old_owner_id'] == OLD_OWNER_ID and set(expected['roles']) == set(ROLE_PIDS), 'Wrong scoped handoff request')
    ROOT.mkdir(parents=True, exist_ok=True)
    lease = ROOT / 'supervisor_owner.json'
    with lease.open('x', encoding='utf-8') as stream:
        json.dump({'id': uuid.uuid4().hex, 'pid': os.getpid(), 'started_at': S.now()}, stream)
    refs = []; paused = False
    state = {'phase': 'capturing_exact_handles', 'passed': False, 'expected': expected,
             'no_child_suspend_or_terminate': True, 'auto_resume_v1': False, 'v2_started': False}
    try:
        owner = check_old_boundary()
        attempt_path, attempt_before = old_attempt(owner)
        for role in ROLE_PIDS:
            ref = backend.capture(role, arm=True)
            refs.append(ref)
            require(public_identity(ref) == expected['roles'][role], 'Held process identity does not match inspection: ' + role)
        require(all(not backend.exited(ref) for ref in refs), 'Capture crossed the child exit boundary')
        protected = list(S.read(Q1.ROOT / 'registration.json')['sources']) + [S.desc(Q1.ROOT / 'registration.json')]
        before_owner = S.desc(Q1.ROOT / 'owner.json')
        before_attempt = S.desc(attempt_path)
        protected.extend([before_owner, before_attempt])
        parent, launcher, gpu = refs
        backend.suspend_dispatcher(parent); paused = True
        require(check_old_boundary() == owner, 'Owner changed while dispatcher was being suspended')
        require(backend.descendants().issubset({ROLE_PIDS['launcher'], ROLE_PIDS['gpu_child']}),
                'Unexpected child appeared; old SSL may have started; retain suspended dispatcher')
        state.update(phase='dispatcher_suspended_children_running', held_identities=[public_identity(ref) for ref in refs],
                     old_owner_snapshot=owner, old_owner_bytes=before_owner)
        S.save(ROOT / 'state.json', state)
        while not (backend.exited(launcher) and backend.exited(gpu)):
            require(check_old_boundary() == owner, 'Suspended dispatcher owner changed; refuse handoff')
            require(backend.descendants().issubset({ROLE_PIDS['launcher'], ROLE_PIDS['gpu_child']}), 'Unexpected child during drain')
            backend.wait_tick()
        launcher_code, gpu_code = backend.exit_code(launcher), backend.exit_code(gpu)
        require(launcher_code == gpu_code == 0, 'Conformer child exited abnormally; keep v1 suspended')
        no_old_ssl_started()
        artifacts = conformer_artifacts()
        for record in protected + [before_owner] + artifacts:
            S.checked(record)
        state.update(phase='children_exit0_conformer15_verified_before_parent_termination', conformer_fits_verified=15,
                     launcher_exit_code=launcher_code, gpu_child_exit_code=gpu_code)
        S.save(ROOT / 'state.json', state)
        backend.terminate_dispatcher(parent)
        require(backend.exited(parent), 'Exact old dispatcher still active')
        proof = {'passed': True, 'version': 2, 'finished_at': S.now(),
                 'v2_registration': S.desc(Q2.ROOT / 'registration.json'),
                 'held_identities': [public_identity(ref) for ref in refs], 'old_owner_snapshot': owner,
                 'old_attempt_original_bytes': before_attempt, 'old_attempt_before_handoff': attempt_before,
                 'old_attempt_observed_terminal_result': {'child_exit_code': 0, 'conformer_all15_completed': True,
                                                         'parent_bookkeeping_replaced_by_versioned_handoff': True},
                 'old_owner_original_bytes': before_owner, 'old_dispatcher_terminated_after_children': True,
                 'old_dispatcher_exit_code': backend.exit_code(parent), 'launcher_exit_code': 0, 'gpu_child_exit_code': 0,
                 'conformer_fits_verified': 15, 'first165_verified': True, 'ssl_v1_never_started': True,
                 'conformer_completed_artifacts': artifacts, 'protected_sources': protected,
                 'model_or_optimizer_or_rng_operations': 0, 'gpu_child_suspend_or_terminate_calls': 0,
                 'all_old_source_and_registration_bytes_preserved': True}
        S.save(Q2.ROOT / 'handoff_complete.json', proof, immutable=True)
        if launch:
            v2_process = backend.spawn_v2()
            state['v2_launcher_pid'] = v2_process.pid; state['v2_started'] = True
        state.update(phase='handoff_complete', passed=True)
        S.save(ROOT / 'state.json', state)
        if launch:
            code = backend.wait_v2(v2_process)
            state.update(phase='v2_dispatcher_finished', v2_exit_code=code, v2_finished_at=S.now(),
                         v2_dispatcher_normal_exit=code == 0)
            S.save(ROOT / 'state.json', state)
        return proof
    except BaseException as error:
        state.update(phase='failed_parent_kept_suspended' if paused else 'refused_before_suspend',
                     error_type=type(error).__name__, error=str(error), old_dispatcher_was_suspended=paused,
                     auto_resume_v1=False, safe_to_start_v2=False)
        S.save(ROOT / 'state.json', state)
        raise
    finally:
        for ref in refs:
            backend.close(ref)
        # The lease remains as a durable handoff/failure record; never steal it.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--expected', help='Saved exact100ns handle identity inspection JSON')
    parser.add_argument('--no-launch', action='store_true')
    args = parser.parse_args()
    require(args.inspect != args.arm, 'Choose exactly one explicit --inspect or --arm action')
    backend = WindowsBackend()
    if args.inspect:
        print(json.dumps(capture_expected(backend), ensure_ascii=False, indent=2))
    else:
        require(args.expected is not None, 'Reviewed exact identity JSON required for --arm')
        arm(S.read(Path(args.expected)), backend, launch=not args.no_launch)
