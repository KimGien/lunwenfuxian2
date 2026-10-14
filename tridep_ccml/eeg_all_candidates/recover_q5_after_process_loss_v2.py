"""One explicit disk recovery after the second observed process loss.

This is a new one-shot control transaction, not a rerun of the first recovery.
No original attempt is rewritten and no exit code, elapsed time or flushed tail
is inferred from process absence. Default CLI performs no runtime inspection.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
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
MANIFEST = ROOT / 'recovery_registration_v2.json'
INTENT = ROOT / 'recovery_intent_v2.json'
PREVIOUS_MANIFEST = ROOT / 'recovery_registration.json'
PREVIOUS_MANIFEST_SHA = 'ec1500ce2671600fca1588e41f14fc559e12300824aaa5c4a2625671f8d2ec9f'
PREVIOUS_CLOSURE = OUT / 'process_loss_closures/lost_f3_1791597116209422300.json'
PREVIOUS_CLOSURE_SHA = 'e56bbb201c067bf0bab31c0d405520fb2860deb12796f088072bb30bdd73de65'
BASE_SHA = 'a9db8e215f1d32cd215d550319c0fa1646985c54d2167783a7547b3a6ea5312f'
Q5_SHA = 'e9a281d44b2e5361215ad178bc6eff96e46327c56cf3b9c2c7050081c9356971'
ENTRY_MODULE = 'tridep_ccml.eeg_all_candidates.recover_q5_after_process_loss_v2'
WORKER_PATTERN = (r'tridep_ccml\.eeg_all_candidates\.(?:continuation_queue|finetune|linux_entry|'
                  r'supervised|ssl|frozen_heads|evaluate|recover_q5|acceleration_prepare)'
                  r'|(?:qualify_(?:acceleration|sync_reduction|training_sync)|sync_qualification_boundary_guard)_v\d+\.py')
PAUSE_PATHS = (OUT / 'user_pause_q5_owner.json', ROOT / 'user_pause.json',
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
                and loss_binding['sha256'] in command, 'Only this exact V2 recovery command may be live')
        allowed.add(pid)
    require(set(by_pid) <= allowed, 'Another EEG worker or qualification/control process is live')
    rows = sorted(rows, key=lambda row: row['pid'])
    return {'schema': 'q5_process_absence_observation_v2', 'source': desc(SOURCE),
            'observed_at': now(), 'query_succeeded': True, 'queried_pids': pids, 'absent_pids': pids,
            'all_expected_processes_absent': True, 'expected_identities': expected,
            'broad_worker_pattern': WORKER_PATTERN, 'workspace_substring_filter_used': False,
            'allowed_recovery_processes': rows, 'other_training_workers': [],
            'workspace_training_workers': [], 'observed_rows': [],
            'observed_rows_scope': 'other workers after exact current recovery entry exclusion; raw rows retained',
            'exact_getprocess_alive_pids': [], 'raw_observation': payload,
            'actual_exit_code': None, 'actual_exit_time': None, 'cause': None, 'exit_codes': None, 'reason': None}


def inspect_absence(expected, loss_binding):
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
    proof['query'] = command
    return proof


def provenance(manifest, closure):
    return {'recovery_registration': manifest, 'recovery_execution_source': desc(SOURCE),
            'recovery_closure': closure, 'previous_recovery_registration': desc(PREVIOUS_MANIFEST),
            'previous_recovery_closure': desc(PREVIOUS_CLOSURE), 'recovery_execution_version': 2}


class Runtime:
    """Real filesystem/runtime backend; imported only after explicit --run."""
    def __init__(self):
        from . import lost_f3_closure_v2 as L
        self.L = L
        self.folder = ROOT / 'recoveries' / ('process_loss_v2_' + str(time.time_ns()))
        self.first = None

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
        self.verify_sources()
        return context

    def verify_sources(self):
        previous = read(desc(PREVIOUS_MANIFEST))
        sources = previous['sources']
        require(len(sources) == 80 and len({r['path'] for r in sources}) == 80, 'All old80 sources required')
        extra = [desc(p) for p in (SOURCE, SOURCE.with_name('lost_f3_closure_v2.py'),
                                  REPORT / 'audit_completed_finetune_checkpoints_v6.py', REPORT / 'cost_aggregation_v7.py')]
        require(len({r['path'] for r in sources + extra}) == 84, 'Exactly four new runtime sources required')
        for row in sources + extra:
            require(Path(row['path']).suffix in ('.py', '.json'), 'Unexpected runtime source type')
            checked(row)
        return sources + extra

    def no_pause(self):
        assert_not_paused()

    def intent(self, loss):
        require(not MANIFEST.exists(), 'Existing V2 registration requires new reviewed recovery, not retry')
        write_new(INTENT, {'schema': 'q5_disk_recovery_exclusive_intent_v2', 'source': desc(SOURCE),
                          'pid': os.getpid(), 'created_at': now(), 'loss_evidence': loss})

    def absence(self, context, loss, index):
        value = inspect_absence(context['expected_native_identities'], loss)
        if self.first is not None:
            require(value['allowed_recovery_processes'] == self.first['allowed_recovery_processes'], 'Recovery process identity changed between observations')
        else:
            self.first = value
        return write_new(self.folder / ('fresh_absence_' + str(index) + '.json'), value)

    def register(self, context, loss, absence):
        sources = self.verify_sources()
        return write_new(MANIFEST, {'schema': 'q5_checkpoint_recovery_execution_v2', 'version': 2,
            'source': desc(SOURCE), 'registered_at': now(), 'base_queue_registration': desc(ROOT / 'registration.json'),
            'previous_recovery_registration': desc(PREVIOUS_MANIFEST), 'registry': desc(OUT / 'registration.json'),
            'loss_evidence': loss, 'process_absence_evidence': context['process_absence_evidence'],
            'runtime_process_absence_evidence': absence, 'checkpoint_cpu_resume_proof': context['checkpoint_cpu_resume_proof'],
            'sources': sources, 'same_formal_F3_and_E3_core': True, 'training_protocol_or_hyperparameters_changed': False,
            'checkpoint_recovery_is_not_natural_handoff': True, 'user_authorization': 'Explicit human continue on 2026-10-11',
            'cpu_auditor': desc(REPORT / 'audit_completed_finetune_checkpoints_v6.py'),
            'cost_collector': desc(REPORT / 'cost_aggregation_v7.py')})

    def closure(self, context, loss, manifest, absence):
        owner = archive(context['old_owner_original'], self.folder / 'old_q5_owner.json')
        attempt = archive(context['old_queue_attempt_original'], self.folder / 'old_q5_attempt_status.json')
        value = {**context, 'schema': 'lost_f3_process_closure_v2', 'status': 'process_absence_observed', 'passed': True,
            'closure_source': desc(SOURCE), 'recovery_registration': manifest, 'loss_evidence': loss,
            'runtime_process_absence_evidence': absence, 'old_owner_snapshot': owner, 'old_queue_attempt_snapshot': attempt,
            'actual_exit_code': None, 'actual_exit_time': None, 'cause': None, 'unflushed_tail_counters': None,
            'final_elapsed_seconds': None, 'queue_attempt_final_elapsed_seconds': None, 'graceful_terminal_flush': False,
            'old_attempt_bytes_unchanged': True, 'known_extra_not_added_again_to_original_returned_counters': True,
            'observed_at': now()}
        result = write_new(OUT / 'process_loss_closures' / ('lost_f3_v2_' + str(time.time_ns()) + '.json'), value)
        self.L.verify_recovery_registration(manifest)
        verified = self.L.verify_lost_f3_attempt(context['original_attempt'], read(context['original_attempt']),
                {k: context[k] for k in ('condition', 'seed', 'fold')})
        require(verified is not None and verified.get('passed') is True and verified.get('closure') == result,
                'Exactly one new closure must independently verify')
        return result

    def recheck(self, context, manifest, closure):
        self.verify_sources()
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
        return write_new(self.folder / 'recovery_ready.json', {'schema': 'q5_disk_recovery_ready_v2', 'passed': True,
            'source': desc(SOURCE), **provenance(manifest, closure), 'second_runtime_process_absence': second_absence,
            'checkpoint_epoch': context['durable']['epoch'], 'prepared_at': now(), 'natural_handoff_claimed': False})


def prepare_recovery(loss, backend=None):
    """Ordered transaction tested with fake backends; never retries a mutation."""
    b = backend or Runtime()
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
    require(type(value.get('schema_version')) is int and value['schema_version'] == 6 and value.get('auditor_source') == auditor
            and value.get('all_135_finetune_checkpoints_verified') is True
            and value.get('errors') == [] and value.get('deferred_completed_fits') == []
            and value.get('pending_fits') == [] and value.get('status') == 'all_finetune_verified'
            and all(type(value.get(k)) is int and value[k] == 135 for k in
                    ('registered_finetune_fits', 'training_complete_finetune_observed', 'verified_completed_finetune_fits'))
            and type(value.get('checkpoint_loads')) is int and value['checkpoint_loads'] == 405,
            'Actual CPU6 all135/405 must pass before any test inference')


def recovery_driver(manifest, closure):
    from . import continuation_queue_v5 as Q5, storage as S
    require(desc(Q5.SOURCE)['sha256'] == Q5_SHA, 'Actual Q5 core changed')
    driver = Q5.isolated_driver()
    driver.S = RecoveryStorage(driver.S, manifest, closure)
    tree = ast.parse(Path(Q5.Q1.__file__).read_text(encoding='utf-8'))
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_tail']
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), 'exec'), driver.__dict__)
    original = driver.evaluate_tail
    def recovered_cpu_gate(owner):
        assert_not_paused()
        if Q5.Q1.all_complete(tuple(c.condition_id for c in Q5.Q1.P.CONDITIONS)):
            for row in read(manifest)['sources']:
                checked(row)
            path = REPORT / 'audit_completed_finetune_checkpoints_v6.py'
            spec = importlib.util.spec_from_file_location('recovered_q5_v2_cpu_verification', path)
            auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
            value = auditor.audit()
            report = ROOT / ('pre_test_finetune_cpu_audit_v6_' + str(time.time_ns()) + '.json')
            report_binding = write_new(report, value)
            validate_cpu_gate(value, desc(path))
            S.save(ROOT / 'pre_test_finetune_cpu_gate.json', {'passed': True, 'audit': report_binding,
                'checkpoint_loads': 405, 'test_forward_calls_this_check': 0, 'source': desc(Q5.SOURCE),
                'execution_registration': desc(ROOT / 'registration.json'), 'execution_version': 5,
                **provenance(manifest, closure), 'actual_cpu_auditor': desc(path)})
        assert_not_paused()
        return original(owner)
    driver.evaluate_tail = recovered_cpu_gate
    return driver


def run(loss):
    manifest, closure, ready = prepare_recovery(loss)
    assert_not_paused()
    print(json.dumps({'recovery_registration': manifest, 'recovery_closure': closure, 'ready': ready}), flush=True)
    recovery_driver(manifest, closure).run(resume=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--loss-evidence')
    parser.add_argument('--loss-evidence-sha256')
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'status': 'pending_explicit_run_and_verified_loss_evidence', 'source': desc(SOURCE),
                          'actual_process_queries': 0, 'process_controls': 0, 'model_operations': 0}))
        return 2
    require(args.loss_evidence and re.fullmatch('[0-9a-f]{64}', args.loss_evidence_sha256 or ''), 'Exact loss evidence path and SHA required')
    path = Path(args.loss_evidence).resolve()
    run({'path': path.relative_to(WORK).as_posix(), 'sha256': args.loss_evidence_sha256})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
