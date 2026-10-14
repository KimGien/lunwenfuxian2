"""One explicit disk recovery after the third observed process loss after the Oct11 pause.

This is a new one-shot control transaction, not a rerun of the first recovery.
No original attempt is rewritten and no exit code, elapsed time or flushed tail
is inferred from process absence. Default CLI performs no runtime inspection.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[2]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT = WORK / 'output/analysis/eeg_all_candidates_v1_20261008'
ROOT = OUT / 'continuation_queue_v5'
MANIFEST = ROOT / 'recovery_registration_v3.json'
INTENT = ROOT / 'recovery_intent_v3.json'
PREVIOUS_MANIFEST = ROOT / 'recovery_registration_v2.json'
PREVIOUS_MANIFEST_SHA = '525f6ed6e838fe267956c54d90d4e6d13d7da3bc710863aca90c35af466782cd'
PREVIOUS_CLOSURE = OUT / 'process_loss_closures/lost_f3_v2_1791706969935734100.json'
PREVIOUS_CLOSURE_SHA = '14286a281be2d896653fb717ea2faefb86b84b3d80e572e36795154b746bfc8e'
BASE_SHA = 'a9db8e215f1d32cd215d550319c0fa1646985c54d2167783a7547b3a6ea5312f'
Q5_SHA = 'e9a281d44b2e5361215ad178bc6eff96e46327c56cf3b9c2c7050081c9356971'
ENTRY_MODULE = 'tridep_ccml.eeg_all_candidates.recover_q5_after_process_loss_v3'
WORKER_PATTERN = (r'tridep_ccml\.eeg_all_candidates\.[A-Za-z0-9_]+'
                  r'|(?:audit_[A-Za-z0-9_]+|qualify_[A-Za-z0-9_]+'
                  r'|sync_qualification_boundary_guard_v\d+|recover_q5_[A-Za-z0-9_]+|repair_owned_worker_v\d+)\.py')
TARGET_CONDITION = 'LaBraM-native4s-lastblock-finetune'
TARGET_JOB = TARGET_CONDITION + '-fit'
BLOCKED_CONDITIONS = ('CodeBrain-lastblock-finetune', 'EEGMamba-lastblock-finetune')
PAUSE_PATHS = (OUT / 'user_pause_q5_owner.json', OUT / 'user_pause_q5_owner_20261011.json', ROOT / 'user_pause.json',
               *(OUT / 'execution_acceleration_v1' / p / 'user_pause.json'
                 for p in ('sync_boundary', 'sync_boundary_v2', 'sync_boundary_v3')))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def desc(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK), 'Descriptor outside workspace')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest}


def checked(binding):
    require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Exact descriptor required')
    path = (WORK / binding['path']).resolve()
    require(desc(path) == binding, 'Bound bytes changed: ' + binding['path'])
    return path


def read(binding):
    path = checked(binding)
    require(path.suffix == '.json', 'Expected JSON evidence')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_new(path, value):
    """True exclusive create, including permanent transaction intent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return desc(path)


def archive(binding, target):
    original = checked(binding)
    raw = original.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Archive source changed during read')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    result = desc(target)
    require(result['sha256'] == binding['sha256'] and desc(original) == binding, 'Exact archive differs')
    return result


def assert_not_paused():
    # Any marker, including malformed content, blocks recovery. This controller
    # does not delete or interpret a future user pause as expired authorization.
    require(not any(p.exists() for p in PAUSE_PATHS), 'User pause control exists; do not start recovery')


def assert_no_existing_closure(artifact):
    for path in (OUT / 'process_loss_closures').glob('lost_f3_*.json'):
        value = read(desc(path))
        require(value.get('original_attempt', {}).get('path') != artifact['path'],
                'This physical attempt already has a closure; do not create an ambiguous second closure')


def native_birth(pid, *, allow_absent=False):
    """Read native creation time from a single handle, without process control."""
    require(os.name == 'nt', 'Windows native identity required')
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    handle = k.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if allow_absent and error == 87:
            return None
        raise ctypes.WinError(error)
    try:
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        code = wintypes.DWORD()
        require(k.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user))
                and k.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259,
                'Native process object is not verifiably active')
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        k.CloseHandle(handle)


def validate_source_review(binding, sources):
    value = read(binding)
    require(value.get('schema') == 'q5_recovery_v3_independent_source_review' and value.get('passed') is True
            and value.get('reviewed_entry') == desc(SOURCE) and value.get('sources') == sources
            and type(value.get('reviewer')) is str and bool(value['reviewer'].strip())
            and value.get('training_protocol_changed') is False and value.get('test_evaluation_permanently_disabled') is True
            and all(type(value.get(k)) is int and value[k] == 0 for k in ('process_actions', 'model_operations'))
            and type(value.get('tests')) is list and bool(value['tests']), 'Actual independent88-source review required')
    for binding in value['tests']:
        checked(binding)


def validate_graphics_context(context, evidence, row, birth):
    expected = evidence.get('process', {})
    image = 'c:/program files/tencent/qqnt/qq.exe'
    norm = lambda value: str(value).replace('\\', '/').casefold()
    require(evidence.get('schema') == 'q5_recovery_v3_external_graphics_observation'
            and evidence.get('nvidia_observed_type') == 'C+G'
            and type(evidence.get('model_or_process_control_operations')) is int
            and evidence['model_or_process_control_operations'] == 0
            and type(expected.get('pid')) is int and type(expected.get('parent_pid')) is int
            and type(expected.get('creation_100ns')) is str and expected['creation_100ns'].isdigit()
            and context.get('type') == 'C+G' and context.get('pid') == expected['pid']
            and norm(context.get('process_name')) == norm(expected.get('image')) == image
            and row.get('pid') == expected['pid'] and row.get('parent_pid') == expected['parent_pid']
            and row.get('creation_time_100ns') == birth == int(expected['creation_100ns'])
            and norm(row.get('image_path')) == image and row.get('command_line') == expected.get('command_line')
            and '--type=gpu-process' in row.get('command_line', '').split(),
            'Unknown compute context or changed QQ native identity blocks launch')


def gpu_and_wsl_observation(graphics_binding):
    """Read-only current admission; no model and no stopped-distro start request."""
    xml = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True,
                         check=True, timeout=30, creationflags=0x08000000).stdout
    root = ET.fromstring(xml)
    devices = root.findall('gpu')
    require(devices and len(devices) == int(root.findtext('attached_gpus', '-1')), 'Complete GPU XML required')
    contexts, exceptions = [], []
    for device in devices:
        processes = device.find('processes')
        require(processes is not None, 'GPU process inventory unavailable')
        for item in processes.findall('process_info'):
            pid = item.findtext('pid', '')
            require(pid.isdigit() and int(pid) > 0 and item.findtext('type') in ('C', 'G', 'C+G'), 'Unknown GPU process row')
            contexts.append({'pid': int(pid), 'type': item.findtext('type'), 'process_name': item.findtext('process_name', '')})
    for context in contexts:
        if context['type'] == 'G':
            continue
        require(graphics_binding is not None, 'Unbound compute context blocks launch')
        evidence = read(graphics_binding)
        require(context['type'] == 'C+G' and context['pid'] == evidence.get('process', {}).get('pid'), 'Unknown CUDA context')
        command = ("$ErrorActionPreference='Stop';$r=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(context['pid']) + "';"
                   "$p=Get-Process -Id " + str(context['pid']) + " -ErrorAction Stop;"
                   "[pscustomobject]@{pid=[int]$r.ProcessId;parent_pid=[int]$r.ParentProcessId;"
                   "creation_time_100ns=$p.StartTime.ToUniversalTime().ToFileTimeUtc();image_path=$r.ExecutablePath;"
                   "command_line=$r.CommandLine}|ConvertTo-Json -Depth 4 -Compress")
        row = json.loads(subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                         capture_output=True, text=True, check=True, timeout=30, creationflags=0x08000000).stdout)
        birth = native_birth(context['pid'])
        validate_graphics_context(context, evidence, row, birth)
        exceptions.append({'evidence': graphics_binding, 'actual_process': row, 'GetProcessTimes_creation_100ns': birth})
    result = subprocess.run(['wsl.exe', '--list', '--running', '--quiet'], capture_output=True,
                            check=True, timeout=30, creationflags=0x08000000)
    names = result.stdout.decode('utf-16-le' if b'\x00' in result.stdout else 'utf-8-sig').strip('\ufeff \r\n')
    distros = [name.strip() for name in names.splitlines() if name.strip()]
    require(len(distros) == len(set(distros)) and all(re.fullmatch(r'[A-Za-z0-9_.-]+', d) for d in distros), 'Invalid running WSL inventory')
    inventories = []
    for distro in distros:
        listing = subprocess.run(['wsl.exe', '-d', distro, '--exec', '/bin/ps', '-eo', 'pid,ppid,args'],
                                 capture_output=True, text=True, check=True, timeout=30, creationflags=0x08000000).stdout
        require(not re.search(r'(?i)(?:python[0-9.]*|torchrun|tridep_ccml|qualify_|finetune|repair_owned_worker)', listing),
                'Another WSL Python/training worker is live')
        inventories.append({'distribution': distro, 'process_inventory': listing})
    return {'observed_at': now(), 'gpu_xml': xml, 'contexts': contexts, 'verified_external_graphics': exceptions,
            'running_wsl_distributions': distros, 'wsl_process_inventories': inventories,
            'model_operations': 0, 'process_controls': 0, 'stopped_distro_start_requests': 0,
            'scope': 'point-in-time training admission; no performance-idle claim'}


def validate_absence_rows(payload, expected, *, current_pid, parent_pid, loss_binding):
    """Pure validation of a fresh broad CIM plus exact GetProcess observation."""
    pids = [x['pid'] for x in expected]
    require(len(pids) == 4 and len(set(pids)) == 4 and all(type(p) is int and p > 0 for p in pids), 'Four exact old identities required')
    require(payload.get('query_succeeded') is True and payload.get('queried_pids') == pids
            and payload.get('exact_getprocess_alive_pids') == [], 'Old native process remains or exact query failed')
    rows = payload.get('rows')
    require(type(rows) is list and all(type(r) is dict for r in rows), 'Broad process rows missing')
    require(not any(r.get('pid') in pids for r in rows), 'An old PID is present; fail closed even if reused')
    by_pid = {r.get('pid'): r for r in rows}
    require(len(by_pid) == len(rows), 'Duplicate process rows')
    own = by_pid.get(current_pid)
    require(own is not None and own.get('parent_pid') == parent_pid, 'Actual entry identity missing')
    allowed = {current_pid}
    for pid in (current_pid, parent_pid):
        row = by_pid.get(pid)
        if row is None:
            continue
        command = row.get('command_line') or ''
        require(type(row.get('creation_time_100ns')) is int and row['creation_time_100ns'] > 0
                and Path(row.get('image_path') or '').name.lower() in ('python.exe', 'python'), 'Entry native identity incomplete')
        require(ENTRY_MODULE in command and re.search(r'(^|\s)--run(?=\s|$)', command)
                and loss_binding['sha256'] in command, 'Only this exact V3 recovery command may be live')
        allowed.add(pid)
    require(set(by_pid) <= allowed, 'Another EEG worker or qualification/control process is live')
    rows = sorted(rows, key=lambda row: row['pid'])
    return {'schema': 'q5_process_absence_observation_v3', 'source': desc(SOURCE),
            'observed_at': now(), 'query_succeeded': True, 'queried_pids': pids, 'absent_pids': pids,
            'all_expected_processes_absent': True, 'expected_identities': expected,
            'broad_worker_pattern': WORKER_PATTERN, 'workspace_substring_filter_used': False,
            'allowed_recovery_processes': rows, 'other_training_workers': [],
            'workspace_training_workers': [], 'observed_rows': [],
            'observed_rows_scope': 'other workers after exact current recovery entry exclusion; raw rows retained',
            'exact_getprocess_alive_pids': [], 'raw_observation': payload,
            'actual_exit_code': None, 'actual_exit_time': None, 'cause': None, 'exit_codes': None, 'reason': None}


def inspect_absence(expected, loss_binding, graphics_binding=None):
    pids = [x['pid'] for x in expected]
    require(all(type(p) is int and p > 0 for p in pids), 'Native PIDs must be integers')
    # There is deliberately no workspace-name/path filter. Full module matching
    # also detects another checkout using the same experiment worker modules.
    command = ("$ErrorActionPreference='Stop'; $ids=@(" + ','.join(map(str, pids)) + ");"
               "$all=@(Get-CimInstance Win32_Process);$rows=@($all | Where-Object {"
               " $_.ProcessId -in $ids -or $_.ProcessId -in @(" + str(os.getpid()) + ',' + str(os.getppid()) + ") -or "
               " $_.CommandLine -match '" + WORKER_PATTERN + "' } | ForEach-Object {"
               " $p=Get-Process -Id $_.ProcessId -ErrorAction Stop;"
               " [pscustomobject]@{pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;"
               "creation_time_100ns=$p.StartTime.ToUniversalTime().ToFileTimeUtc();"
               "image_path=$_.ExecutablePath;command_line=$_.CommandLine}});"
               "$alive=@(foreach($id in $ids){$p=Get-Process -Id $id -ErrorAction SilentlyContinue;if($null -ne $p){[int]$id}});"
               "[pscustomobject]@{query_succeeded=$true;queried_pids=$ids;exact_getprocess_alive_pids=$alive;rows=$rows} | ConvertTo-Json -Depth 6 -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, check=True, timeout=30, creationflags=0x08000000)
    payload = json.loads(result.stdout)
    proof = validate_absence_rows(payload, expected, current_pid=os.getpid(), parent_pid=os.getppid(), loss_binding=loss_binding)
    require(all(native_birth(pid, allow_absent=True) is None for pid in pids), 'An old PID still has a native object')
    for row in proof['allowed_recovery_processes']:
        require(native_birth(row['pid']) == row['creation_time_100ns'], 'Recovery native birth changed')
    proof['native_birth_api'] = 'GetProcessTimes'
    proof['gpu_and_wsl_admission'] = gpu_and_wsl_observation(graphics_binding)
    proof['query'] = command
    return proof


def provenance(manifest, closure):
    return {'recovery_registration': manifest, 'recovery_execution_source': desc(SOURCE),
            'recovery_closure': closure, 'previous_recovery_registration': desc(PREVIOUS_MANIFEST),
            'previous_recovery_closure': desc(PREVIOUS_CLOSURE), 'recovery_execution_version': 3,
            'earlier_recovery_registration': read(desc(PREVIOUS_MANIFEST))['previous_recovery_registration']}


class Runtime:
    """Real filesystem/runtime backend; imported only after explicit --run."""
    def __init__(self, source_review, graphics_evidence=None):
        from . import lost_f3_closure_v3 as L
        self.L = L
        self.folder = ROOT / 'recoveries' / ('process_loss_v3_' + str(time.time_ns()))
        self.first = None
        self.source_review, self.graphics_evidence = source_review, graphics_evidence

    def verify(self, loss):
        context = self.L.verify_recovery_inputs(loss)
        require(context.get('passed') is True and context.get('loss_evidence') == loss, 'Loss evidence did not verify')
        require(desc(PREVIOUS_MANIFEST)['sha256'] == PREVIOUS_MANIFEST_SHA
                and desc(PREVIOUS_CLOSURE)['sha256'] == PREVIOUS_CLOSURE_SHA
                and desc(ROOT / 'registration.json')['sha256'] == BASE_SHA
                and desc(SOURCE.with_name('continuation_queue_v5.py'))['sha256'] == Q5_SHA, 'Historical lineage changed')
        require(context['old_owner_original']['path'] == (ROOT / 'owner.json').relative_to(WORK).as_posix(), 'Only exact stale Q5 owner may be removed')
        checked(context['resume_checkpoint']); checked(context['preserved_checkpoint'])
        assert_no_existing_closure(context['original_attempt'])
        validate_source_review(self.source_review, self.verify_sources())
        return context

    def verify_sources(self):
        previous = read(desc(PREVIOUS_MANIFEST))
        sources = previous['sources']
        require(len(sources) == 84 and len({r['path'] for r in sources}) == 84, 'All old84 sources required')
        extra = [desc(p) for p in (SOURCE, SOURCE.with_name('lost_f3_closure_v3.py'),
                                  REPORT / 'audit_completed_finetune_checkpoints_v8.py', REPORT / 'cost_aggregation_v9.py')]
        require(len({r['path'] for r in sources + extra}) == 88, 'Exactly four new runtime sources required')
        for row in sources + extra:
            require(Path(row['path']).suffix in ('.py', '.json'), 'Unexpected runtime source type')
            checked(row)
        return sources + extra

    def no_pause(self):
        assert_not_paused()

    def intent(self, loss):
        require(not MANIFEST.exists(), 'Existing V3 registration requires new reviewed recovery, not retry')
        write_new(INTENT, {'schema': 'q5_disk_recovery_exclusive_intent_v3', 'source': desc(SOURCE),
                          'pid': os.getpid(), 'created_at': now(), 'loss_evidence': loss})

    def absence(self, context, loss, index):
        value = inspect_absence(context['expected_native_identities'], loss, self.graphics_evidence)
        if self.first is not None:
            require(value['allowed_recovery_processes'] == self.first['allowed_recovery_processes'], 'Recovery process identity changed between observations')
        else:
            self.first = value
        return write_new(self.folder / ('fresh_absence_' + str(index) + '.json'), value)

    def register(self, context, loss, absence):
        sources = self.verify_sources()
        validate_source_review(self.source_review, sources)
        return write_new(MANIFEST, {'schema': 'q5_checkpoint_recovery_execution_v3', 'version': 3,
            'source': desc(SOURCE), 'registered_at': now(), 'base_queue_registration': desc(ROOT / 'registration.json'),
            'previous_recovery_registration': desc(PREVIOUS_MANIFEST), 'registry': desc(OUT / 'registration.json'),
            'previous_recovery_closure': desc(PREVIOUS_CLOSURE),
            'earlier_recovery_registration': read(desc(PREVIOUS_MANIFEST))['previous_recovery_registration'],
            'loss_evidence': loss, 'process_absence_evidence': context['process_absence_evidence'],
            'runtime_process_absence_evidence': absence, 'checkpoint_cpu_resume_proof': context['checkpoint_cpu_resume_proof'],
            'sources': sources, 'same_formal_F3_and_E3_core': True, 'training_protocol_or_hyperparameters_changed': False,
            'independent_source_review': self.source_review, 'external_graphics_evidence': self.graphics_evidence,
            'actual_allowed_job_ids': [TARGET_JOB], 'evaluation_permanently_disabled_in_this_entry': True,
            'engineering_and_optional_qualification_launches_allowed': False,
            'checkpoint_recovery_is_not_natural_handoff': True, 'user_authorization': 'Explicit human continue on 2026-10-12',
            'cpu_auditor': desc(REPORT / 'audit_completed_finetune_checkpoints_v8.py'),
            'cost_collector': desc(REPORT / 'cost_aggregation_v9.py')})

    def closure(self, context, loss, manifest, absence):
        owner = archive(context['old_owner_original'], self.folder / 'old_q5_owner.json')
        attempt = archive(context['old_queue_attempt_original'], self.folder / 'old_q5_attempt_status.json')
        value = {**context, 'schema': 'lost_f3_process_closure_v3', 'status': 'process_absence_observed', 'passed': True,
            'closure_source': desc(SOURCE), 'recovery_registration': manifest, 'loss_evidence': loss,
            'runtime_process_absence_evidence': absence, 'old_owner_snapshot': owner, 'old_queue_attempt_snapshot': attempt,
            'actual_exit_code': None, 'actual_exit_time': None, 'cause': None, 'unflushed_tail_counters': None,
            'final_elapsed_seconds': None, 'queue_attempt_final_elapsed_seconds': None, 'graceful_terminal_flush': False,
            'old_attempt_bytes_unchanged': True, 'known_extra_not_added_again_to_original_returned_counters': True,
            'observed_at': now()}
        result = write_new(OUT / 'process_loss_closures' / ('lost_f3_v3_' + str(time.time_ns()) + '.json'), value)
        self.L.verify_recovery_registration(manifest)
        verified = self.L.verify_lost_f3_attempt(context['original_attempt'], read(context['original_attempt']),
                {k: context[k] for k in ('condition', 'seed', 'fold')})
        require(verified is not None and verified.get('passed') is True and verified.get('closure') == result,
                'Exactly one new closure must independently verify')
        return result

    def recheck(self, context, manifest, closure):
        validate_source_review(self.source_review, self.verify_sources())
        self.L.verify_recovery_registration(manifest)
        for key in ('original_attempt', 'attempt_snapshot', 'old_owner_original', 'old_queue_attempt_original',
                    'train_history', 'history_snapshot', 'resume_checkpoint', 'preserved_checkpoint'):
            checked(context[key])
        value = read(closure)
        require(value['old_owner_snapshot']['sha256'] == context['old_owner_original']['sha256'], 'Stale owner archive differs')
        checked(value['old_owner_snapshot'])

    def remove_owner(self, context):
        # No native process action is performed. Only this verified stale file.
        checked(context['old_owner_original']).unlink()

    def ready(self, context, manifest, closure, second_absence):
        return write_new(self.folder / 'recovery_ready.json', {'schema': 'q5_disk_recovery_ready_v3', 'passed': True,
            'source': desc(SOURCE), **provenance(manifest, closure), 'second_runtime_process_absence': second_absence,
            'checkpoint_epoch': context['durable']['epoch'], 'prepared_at': now(), 'natural_handoff_claimed': False})


def prepare_recovery(loss, backend):
    """Ordered transaction tested with fake backends; never retries a mutation."""
    b = backend
    b.no_pause()
    context = b.verify(loss)
    b.intent(loss)
    b.no_pause()
    first = b.absence(context, loss, 1)
    manifest = b.register(context, loss, first)
    closure = b.closure(context, loss, manifest, first)
    b.recheck(context, manifest, closure)
    b.no_pause()
    second = b.absence(context, loss, 2)
    b.no_pause()
    b.recheck(context, manifest, closure)
    b.remove_owner(context)
    # A newly arrived pause leaves the owner archived and transaction retained;
    # it never starts the queue or silently removes a new control marker.
    b.no_pause()
    ready = b.ready(context, manifest, closure, second)
    return manifest, closure, ready


class RecoveryStorage:
    def __init__(self, previous, manifest, closure):
        self.previous, self.fields = previous, provenance(manifest, closure)
    def __getattr__(self, name):
        return getattr(self.previous, name)
    def save(self, path, value, *, immutable=False):
        target = Path(path).resolve()
        if target.is_relative_to(ROOT.resolve()) and target.name in ('status.json', 'owner.json'):
            value = {**value, **self.fields}
        return self.previous.save(path, value, immutable=immutable)


def validate_cpu_gate(value, auditor):
    require(type(value.get('schema_version')) is int and value['schema_version'] == 8 and value.get('auditor_source') == auditor
            and value.get('all_135_finetune_checkpoints_verified') is True
            and value.get('errors') == [] and value.get('deferred_completed_fits') == []
            and value.get('pending_fits') == [] and value.get('status') == 'all_finetune_verified'
            and all(type(value.get(k)) is int and value[k] == 135 for k in
                    ('registered_finetune_fits', 'training_complete_finetune_observed', 'verified_completed_finetune_fits'))
            and type(value.get('checkpoint_loads')) is int and value['checkpoint_loads'] == 405,
            'Actual CPU8 all135/405 must pass before any test inference')


def validate_training_scope(rows, *, before_launch):
    """Only five remaining original LaBraM4s fits may execute."""
    require(type(rows) is list and len(rows) == 360, 'Original360 status rows required')
    keys = [(r.get('condition'), r.get('seed'), r.get('fold')) for r in rows]
    require(len(set(keys)) == 360 and all(type(s) is int and type(f) is int for _, s, f in keys),
            'Original fit identities must remain unique strict integers')
    target = [r for r in rows if r['condition'] == TARGET_CONDITION]
    blocked = [r for r in rows if r['condition'] in BLOCKED_CONDITIONS]
    other = [r for r in rows if r['condition'] not in (*BLOCKED_CONDITIONS, TARGET_CONDITION)]
    require(len(target) == 15 and len(blocked) == 30 and len(other) == 315
            and all(r['status'] == 'blocked' for r in blocked)
            and all(r['status'] == 'training_complete' for r in other),
            'Only LaBraM4s may resume; engineering-failed groups stay blocked')
    require({(r['seed'], r['fold']) for r in target} == {(s, f) for s in (11, 22, 33) for f in range(1, 6)},
            'Exact original LaBraM4s15 matrix required')
    require(all(r['status'] == 'training_complete' for r in target if r['seed'] in (11, 22)),
            'Completed LaBraM4s seed11/22 must remain complete')
    if before_launch:
        require(all(r['status'] == ('training' if r['fold'] == 1 else 'ready')
                    for r in target if r['seed'] == 33), 'Only seed33 fold1 loss and fold2..5 ready may start')
    return {'training_complete': sum(r['status'] == 'training_complete' for r in rows),
            'remaining_registered_repair_fits': len(blocked),
            'target_completed': sum(r['status'] == 'training_complete' for r in target)}


def registry_rows(driver):
    return [{k: fit[k] for k in ('condition', 'seed', 'fold')} |
            {'status': driver.R.fit_status(fit['condition'], fit['seed'], fit['fold'])['status']}
            for fit in driver.R.load_registration()['fits']]


def disable_evaluation(owner):
    raise RuntimeError('Outer evaluation is permanently disabled in recovery3; reviewed repair chain required')


def recovery_driver(manifest, closure):
    from . import continuation_queue_v5 as Q5
    require(desc(Q5.SOURCE)['sha256'] == Q5_SHA, 'Actual Q5 core changed')
    driver = Q5.isolated_driver()
    driver.S = RecoveryStorage(driver.S, manifest, closure)
    driver.evaluate_tail = disable_evaluation
    original_run_child = driver.run_child

    def scoped_run_child(job, owner, *, resume=False):
        assert_not_paused()
        registration = read(desc(ROOT / 'registration.json'))
        registered = [j for j in registration['jobs'] if j['id'] == TARGET_JOB]
        require(len(registered) == 1 and job == registered[0] and resume is True,
                'Only exact original registered LaBraM4s F3 fit command may execute')
        require(job['conditions'] == [TARGET_CONDITION] and job.get('fits') == 15
                and job.get('linux') is not True, 'Original native F3 matrix required')
        for row in read(manifest)['sources']:
            checked(row)
        return original_run_child(job, owner, resume=True)

    driver.run_child = scoped_run_child

    def run_target_only(*, resume=False):
        assert_not_paused()
        require(resume is True, 'Only checkpoint resume is allowed')
        registration = driver.register()
        require(len(registration['jobs']) == 36 and sum(j.get('fits', 0) for j in registration['jobs']) == 195,
                'Original36 jobs/195 formal fit registration must stay unchanged')
        validate_training_scope(registry_rows(driver), before_launch=True)
        job, = [j for j in registration['jobs'] if j['id'] == TARGET_JOB]
        owner = driver.acquire_owner()
        try:
            driver.S.save(ROOT / 'owner.json', owner)
            assert_not_paused()
            outcome = driver.run_child(job, owner, resume=True)
            after = validate_training_scope(registry_rows(driver), before_launch=False)
            driver.S.save(ROOT / 'status.json', {
                'status': 'required_work_remaining', 'outcomes': {job['id']: outcome},
                'registry_status': driver.R.status_summary(), 'updated_at': now(),
                'test_evaluation_completed': False, 'test_evaluation_frozen': True,
                'test_evaluation_status': 'permanently_disabled_in_recovery3',
                'actual_resumed_job_ids': [job['id']], 'other_jobs_executed_this_recovery': 0,
                'remaining_engineering_blocked_conditions': list(BLOCKED_CONDITIONS),
                'verified_scope_status': after, 'all_plan_complete': False})
            print('Recovered Q5 target dispatch ended; thirty registered repair fits remain; evaluation disabled', flush=True)
        except BaseException as error:
            driver.S.save(ROOT / ('failure_' + str(time.time_ns()) + '.json'),
                          {'error_type': type(error).__name__, 'error': str(error), 'time': now()})
            driver.S.save(ROOT / 'status.json', {'status': 'failed', 'error_type': type(error).__name__,
                'error': str(error), 'updated_at': now(), 'test_evaluation_frozen': True,
                'test_evaluation_completed': False, 'all_plan_complete': False})
            raise
        finally:
            driver.release_owner(owner)

    driver.run = run_target_only
    return driver


def run(loss, source_review, graphics_evidence=None):
    backend = Runtime(source_review, graphics_evidence)
    manifest, closure, ready = prepare_recovery(loss, backend)
    assert_not_paused()
    print(json.dumps({'recovery_registration': manifest, 'recovery_closure': closure, 'ready': ready}), flush=True)
    recovery_driver(manifest, closure).run(resume=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--loss-evidence')
    parser.add_argument('--loss-evidence-sha256')
    parser.add_argument('--source-review')
    parser.add_argument('--source-review-sha256')
    parser.add_argument('--graphics-evidence')
    parser.add_argument('--graphics-evidence-sha256')
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'status': 'pending_explicit_run_and_verified_loss_evidence', 'source': desc(SOURCE),
                          'actual_process_queries': 0, 'process_controls': 0, 'model_operations': 0}))
        return 2
    require(args.loss_evidence and re.fullmatch('[0-9a-f]{64}', args.loss_evidence_sha256 or ''), 'Exact loss evidence path and SHA required')
    require(args.source_review and re.fullmatch('[0-9a-f]{64}', args.source_review_sha256 or ''),
            'Exact independent source review path and SHA required')
    require(bool(args.graphics_evidence) == bool(args.graphics_evidence_sha256)
            and (not args.graphics_evidence or re.fullmatch('[0-9a-f]{64}', args.graphics_evidence_sha256)),
            'Graphics evidence path and SHA must be supplied together')
    def argument_binding(path, sha):
        return {'path': Path(path).resolve().relative_to(WORK).as_posix(), 'sha256': sha}
    run(argument_binding(args.loss_evidence, args.loss_evidence_sha256),
        argument_binding(args.source_review, args.source_review_sha256),
        argument_binding(args.graphics_evidence, args.graphics_evidence_sha256) if args.graphics_evidence else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
