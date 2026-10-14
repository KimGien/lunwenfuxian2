"""Explicit user pause: suspend only identity-verified current Windows queue/fit.

Does not terminate or restart any process, or execute a model/optimizer. The
immutable record contains exact process birth times for a future manual resume.
"""
import ctypes
from ctypes import wintypes as W
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import registry as R, storage as S

ROOT = S.OUT / 'continuation_queue_v3'
EXPECTED = [
    {'role': 'dispatcher', 'pid': 39760, 'parent_pid': 39768,
     'created': '2026-10-08T19:26:11.932906+08:00', 'image': r'F:\develop\python.exe',
     'module_tail': 'tridep_ccml.eeg_all_candidates.continuation_queue_v3 --resume'},
    {'role': 'fit_launcher', 'pid': 46676, 'parent_pid': 39760,
     'created': '2026-10-08T19:26:36.218837+08:00',
     'image': str(WORK / 'S00/.venv/Scripts/python.exe'),
     'module_tail': 'tridep_ccml.eeg_all_candidates.supervised --conditions EEG2Rep-supervised --resume'},
    {'role': 'gpu_fit', 'pid': 21104, 'parent_pid': 46676,
     'created': '2026-10-08T19:26:36.228414+08:00', 'image': r'F:\develop\python.exe',
     'module_tail': 'tridep_ccml.eeg_all_candidates.supervised --conditions EEG2Rep-supervised --resume'},
]


def require(value, message):
    if not value:
        raise ValueError(message)


def expected_birth(value):
    delta = datetime.fromisoformat(value).astimezone(timezone.utc) - datetime(1601, 1, 1, tzinfo=timezone.utc)
    return ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 10


def main():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    native = ctypes.WinDLL('ntdll', use_last_error=True)
    kernel.OpenProcess.argtypes = (W.DWORD, W.BOOL, W.DWORD)
    kernel.OpenProcess.restype = W.HANDLE
    kernel.CloseHandle.argtypes = (W.HANDLE,)
    kernel.GetProcessTimes.argtypes = (W.HANDLE, ctypes.POINTER(W.FILETIME), ctypes.POINTER(W.FILETIME),
                                       ctypes.POINTER(W.FILETIME), ctypes.POINTER(W.FILETIME))
    kernel.GetExitCodeProcess.argtypes = (W.HANDLE, ctypes.POINTER(W.DWORD))
    kernel.QueryFullProcessImageNameW.argtypes = (W.HANDLE, W.DWORD, W.LPWSTR, ctypes.POINTER(W.DWORD))
    native.NtSuspendProcess.argtypes = (W.HANDLE,)
    native.NtSuspendProcess.restype = ctypes.c_long
    owner = S.read(ROOT / 'owner.json')
    require(owner['id'] == '91f618c8cd274e16aa74df5650420d11' and owner['pid'] == 39760
            and owner['child_pid'] == 46676 and owner['child_job'] == 'eeg2rep-supervised-fit',
            'Current queue owner/job changed; refuse pause of a guessed process')
    command = ('Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @(39760,46676,21104) } '
               '| Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Depth 3')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True,
                            text=True, encoding='utf-8', check=True)
    observed = {row['ProcessId']: row for row in json.loads(result.stdout)}
    folder = S.OUT / 'pause_events' / str(time.time_ns())
    folder.mkdir(parents=True, exist_ok=False)
    handles, identities = [], []
    state = {'status': 'pause_preflight', 'requested_by': 'explicit user pause', 'started_at': S.now(),
             'source': S.desc(Path(__file__)), 'owner': S.desc(ROOT / 'owner.json'),
             'owner_value': owner, 'actions': [], 'test_evaluation_frozen': True}
    S.save(folder / 'state.json', state)
    try:
        for item in EXPECTED:
            actual = observed.get(item['pid'])
            require(actual and actual['ParentProcessId'] == item['parent_pid']
                    and actual['ExecutablePath'].casefold() == item['image'].casefold()
                    and actual['CommandLine'].endswith(item['module_tail']), 'Process tree/image/command identity differs')
            handle = kernel.OpenProcess(0x1000 | 0x0800 | 0x100000, False, item['pid'])
            require(handle, 'Cannot open process for exact reversible suspension')
            handles.append(handle)
            created, exited, cpu_kernel, cpu_user = W.FILETIME(), W.FILETIME(), W.FILETIME(), W.FILETIME()
            require(kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                           ctypes.byref(cpu_kernel), ctypes.byref(cpu_user)), 'Cannot read process birth')
            birth = (created.dwHighDateTime << 32) | created.dwLowDateTime
            require(birth // 10 == expected_birth(item['created']) // 10, 'Exact process birth differs')
            code = W.DWORD()
            require(kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259,
                    'Process has already exited; refuse suspended replacement')
            image = ctypes.create_unicode_buffer(32768)
            size = W.DWORD(len(image))
            require(kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size))
                    and image.value.casefold() == item['image'].casefold(), 'Opened process image differs')
            identities.append({**item, 'creation_100ns': str(birth), 'command_line': actual['CommandLine']})
        state['identities'] = identities
        S.save(folder / 'state.json', state)
        require(S.read(ROOT / 'owner.json') == owner, 'Queue owner changed before suspension')
        for handle, identity in zip(handles, identities):
            code = native.NtSuspendProcess(handle)
            state['actions'].append({'role': identity['role'], 'pid': identity['pid'],
                                     'native_status': int(code), 'observed_at': S.now()})
            S.save(folder / 'state.json', state)
            require(code == 0, 'Native suspension failed; preserve exact partial pause record')
        progress = []
        for path in sorted((S.OUT / 'runs/EEG2Rep-supervised').glob('seed_*/fold_*/progress.json')):
            if S.read(path.parent / 'status.json')['status'] == 'training':
                progress.append({'progress': S.desc(path), 'value': S.read(path),
                                 'latest_checkpoint': S.desc(path.parent / 'latest.pt')})
        state.update(status='paused_by_user', finished_at=S.now(), registry_status=R.status_summary(),
                     suspended_processes=len(handles), active_fit_progress=progress,
                     processes_terminated=0, model_or_optimizer_operations=0,
                     runtime_memory_preserved=True, automatic_resume_authorized=False,
                     resume_note='Resume only after a new user instruction. Recheck all exact births/images and owner, '
                                 'then NtResumeProcess gpu_fit, fit_launcher, dispatcher once in reverse order. '
                                 'Do not start duplicate queues or change the old source-bound owner/status files.')
        S.save(folder / 'state.json', state)
        S.save(folder / 'pause_complete.json', state, immutable=True)
        print(json.dumps({'paused': True, 'record': S.desc(folder / 'pause_complete.json'),
                          'registry_status': state['registry_status'], 'suspended_processes': len(handles)}))
    except BaseException as error:
        state.update(status='pause_incomplete_preserve_actions', error=str(error), finished_at=S.now())
        S.save(folder / 'state.json', state)
        raise
    finally:
        for handle in handles:
            kernel.CloseHandle(handle)


if __name__ == '__main__':
    main()
