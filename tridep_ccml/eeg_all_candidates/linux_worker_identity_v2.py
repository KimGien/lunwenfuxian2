"""Stdlib-only Linux admission identities, with no model or GPU operations.

The Windows controller independently captures one exact module/nonce worker.
That worker must see the same boot, pid, parent, start ticks, argv, executable
and parent identity before model work and at every subsequent owned boundary.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[2]
MODULE_PATTERN = 'eeg_all_candidates|audit_completed_finetune|qualify_.*sync|sync_qualification_boundary|supplemental_head_resource'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def process_identity(pid, proc_root=Path('/proc')):
    folder = proc_root / str(pid)
    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    argv = [x.decode('utf-8', errors='strict') for x in (folder / 'cmdline').read_bytes().split(b'\x00') if x]
    return {'pid': int(pid), 'parent': int(fields[1]), 'start_ticks': fields[19], 'argv': argv, 'executable': str((folder / 'exe').resolve(strict=True))}

def snapshot_local(*, exclude_pid=None, proc_root=Path('/proc'), observer_caller_pid=None):
    require(os.getuid() == 0, 'Complete /proc observer requires root; model workers remain unprivileged')
    require(sys.platform == 'linux', 'Local process identity requires Linux')
    rows = []
    errors = []
    for folder in proc_root.iterdir():
        if not folder.name.isdigit() or int(folder.name) == exclude_pid:
            continue
        try:
            command = (folder / 'cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', errors='strict')
            if not re.search(MODULE_PATTERN, command):
                continue
            row = process_identity(int(folder.name), proc_root)
            row['parent_identity'] = process_identity(row['parent'], proc_root)
            rows.append(row)
        except FileNotFoundError:
            continue
        except (OSError, ValueError, UnicodeError) as error:
            errors.append({'pid': int(folder.name), 'error_type': type(error).__name__})
    excluded_bridges = []
    if observer_caller_pid is not None:
        require(type(observer_caller_pid) is int and observer_caller_pid > 0, 'Exact observer caller required')
        own_argv = process_identity(os.getpid(), proc_root)['argv']
        require(own_argv.count('-c') == 1, 'Isolated observer command absent')
        own_program = own_argv[own_argv.index('-c') + 1]
        expected = ['/init', '/mnt/c/Windows/System32/wsl.exe', '/mnt/c/Windows/System32/wsl.exe', '-d', 'TRI-DEP-Ubuntu-24.04', '-u', 'root', '--exec', linux_path(WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/bin/python'), '-I', '-S', '-B', '-c', own_program]
        callers = [r for r in rows if r['pid'] == observer_caller_pid]
        require(len(callers) == 1, 'Exact unprivileged observer caller absent')
        keep = []
        for row in rows:
            if row['executable'] == '/init' and row['argv'] == expected and (row['parent'] == observer_caller_pid) and (row['parent_identity'] == {k: v for k, v in callers[0].items() if k != 'parent_identity'}):
                excluded_bridges.append(row)
            else:
                keep.append(row)
        require(len(excluded_bridges) <= 1, 'Ambiguous readonly interop relay')
        rows = keep
    return {'schema': 'linux_worker_identity_snapshot_v2', 'query_succeeded': True, 'boot_id': (proc_root / 'sys/kernel/random/boot_id').read_text().strip(), 'rows': sorted(rows, key=lambda x: x['pid']), 'errors': errors, 'observer_uid': os.getuid(), 'observer_scope': 'read_only_proc_no_models_or_gpu', 'excluded_exact_readonly_interop_relays': excluded_bridges}

def linux_path(path):
    path = Path(os.path.abspath(path))
    if sys.platform == 'linux':
        require(path.is_absolute() and path.is_relative_to(WORK), 'Pinned workspace Linux path required')
        return path.as_posix()
    require(path.drive and len(path.drive) == 2, 'Windows drive required')
    return '/mnt/' + path.drive[0].lower() + '/' + path.as_posix()[3:]

def snapshot():
    require(sys.platform in ('linux', 'win32'), 'Pinned Windows/WSL bridge only')
    source_path = linux_path(SOURCE)
    import hashlib
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    caller = os.getpid() if sys.platform == 'linux' else None
    program = 'import hashlib,importlib.util,json,os; from pathlib import Path; p=Path(' + repr(source_path) + '); assert hashlib.sha256(p.read_bytes()).hexdigest()==' + repr(digest) + "; s=importlib.util.spec_from_file_location('_read_only_proc_observer',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.snapshot_local(exclude_pid=os.getpid(), observer_caller_pid=CALLER_VALUE)))"
    program = program.replace('CALLER_VALUE', repr(caller))
    executable = 'wsl.exe' if sys.platform == 'win32' else '/mnt/c/Windows/System32/wsl.exe'
    command = [executable, '-d', 'TRI-DEP-Ubuntu-24.04', '-u', 'root', '--exec', linux_path(WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/bin/python'), '-I', '-S', '-B', '-c', program]
    options = {'capture_output': True, 'text': True, 'check': True, 'timeout': 30}
    if sys.platform == 'win32':
        options['creationflags'] = 134217728
    result = subprocess.run(command, **options)
    value = json.loads(result.stdout.strip().splitlines()[-1])
    require(value.get('observer_uid') == 0 and value.get('observer_scope') == 'read_only_proc_no_models_or_gpu', 'Wrong or unverified privileged observation scope')
    return value

def select_unique(value, *, module, flags):
    require(value.get('observer_uid') == 0 and value.get('observer_scope') == 'read_only_proc_no_models_or_gpu', 'Complete read-only /proc observer required')
    require(value.get('query_succeeded') is True and value.get('errors') == [] and (type(value.get('boot_id')) is str) and bool(value['boot_id']), 'Linux inventory unavailable or incomplete')
    rows = value['rows']
    require(type(rows) is list and len(rows) == 1, 'Exactly one experimental Linux worker required')
    row = rows[0]
    argv = row['argv']
    require(type(row['pid']) is int and row['pid'] > 0 and (type(row['parent']) is int) and (row['parent'] > 0) and (type(row['start_ticks']) is str) and row['start_ticks'].isdigit(), 'Exact Linux pid/parent/start ticks required')
    require(argv.count('-m') == 1 and argv[argv.index('-m') + 1] == module, 'Linux worker module differs from controller launch')
    for flag, expected in flags.items():
        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == str(expected), 'Linux worker nonce/admission/role differs: ' + flag)
    parent = row.get('parent_identity')
    require(type(parent) is dict and parent['pid'] == row['parent'] and parent['start_ticks'].isdigit() and parent['argv'] and parent['executable'], 'Complete Linux parent identity required')
    require(row['executable'] and argv[0], 'Linux executable/argv absent')
    return {'boot_id': value['boot_id'], 'identity': row}

def validate_bound(value, binding, *, module, flags, worker_pid):
    actual = select_unique(value, module=module, flags=flags)
    require(type(worker_pid) is int and actual['identity']['pid'] == worker_pid, 'Actual Linux caller is not controller-captured worker')
    require(actual == binding, 'Linux boot/pid/parent/birth/argv/executable identity changed')
    return actual

def wait_capture(*, module, flags, is_alive, timeout=20):
    deadline = time.monotonic() + timeout
    while True:
        require(is_alive() and time.monotonic() < deadline, 'Linux child exited or identity handshake timed out')
        value = snapshot()
        require(value.get('query_succeeded') is True and value.get('errors') == [], 'Linux inventory query failed')
        if value['rows']:
            return select_unique(value, module=module, flags=flags)
        time.sleep(0.1)
