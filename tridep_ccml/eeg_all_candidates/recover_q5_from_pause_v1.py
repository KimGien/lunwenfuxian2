"""Explicit verified disk recovery into the unchanged actual Q5/F3 core.

This entry records process loss rather than inventing a natural child exit0.
Original mutable F3 observations remain unchanged. A separately registered
CPU auditor interprets their source-bound closure and unknown buffered tail.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from . import continuation_queue_v4 as Q4, continuation_queue_v5 as Q5, registry as R, storage as S

SOURCE = Path(__file__).resolve()
ROOT = Q5.ROOT
MANIFEST = ROOT / 'recovery_registration.json'
INTENT = ROOT / 'recovery_intent.json'
PAUSE = S.OUT / 'pause_events/q4_user_pause_1791558499460173100/pause_complete.json'
PAUSE_SHA = '247d0b0dc3329a248ef7ca384b4acb2b2956e403d6b512c6b493306ed697d910'
CPU_PROOF = S.REPORT / 'recovery_cpu_preflight_csb33f4_1791596275747982600.json'
CPU_PROOF_SHA = '6e8985f94502b9d36de56fd4a9c3b6d901346e3b5c28f9b62db5d45663632308'
NEW_SOURCES = ((SOURCE,), (SOURCE.with_name('lost_f3_closure_v1.py'),),
               (S.REPORT / 'audit_completed_finetune_checkpoints_v5.py',), (S.REPORT / 'cost_aggregation_v6.py',))


def require(value, message):
    if not value: raise ValueError(message)


def inspect_absence():
    """Exact old identities and workspace train dispatchers, not arbitrary apps."""
    pause = S.read(PAUSE)
    require(S.digest(PAUSE) == PAUSE_SHA, 'Original user pause evidence changed')
    command = ("$rows=@(Get-CimInstance Win32_Process | Where-Object {"
               " $_.ProcessId -in @(40840,18224,32880,46716) -or "
               "($_.CommandLine -like '*lunwen2*' -and $_.CommandLine -match 'tridep_ccml\\.eeg_all_candidates\\.(continuation_queue|finetune|linux_entry|supervised|ssl|frozen_heads|evaluate)')"
               " } | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine);"
               " ConvertTo-Json -InputObject $rows -Depth 4 -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, check=True, timeout=30, creationflags=0x08000000)
    rows = json.loads(result.stdout)
    require(type(rows) is list and not rows, 'An old identity or another workspace training instance is live; do not duplicate')
    return {'schema': 'q4_process_absence_observation_v1', 'source': S.desc(SOURCE), 'observed_at': S.now(),
            'expected_identities': pause['identities'], 'queried_pids': [40840,18224,32880,46716],
            'absent_pids': [40840,18224,32880,46716], 'query_succeeded': True,
            'all_expected_processes_absent': True, 'workspace_training_workers': [],
            'exit_codes': None, 'actual_exit_time': None, 'reason': None,
            'observed_rows': rows, 'query': command}


def runtime_sources():
    paths = [group[0] for group in NEW_SOURCES]
    return [S.desc(p) for p in paths]


def register(absence_binding):
    base = Q5.register()
    require(S.digest(CPU_PROOF) == CPU_PROOF_SHA, 'Actual CPU resume verification changed')
    if not MANIFEST.exists():
        records = base['sources'] + runtime_sources()
        sources = list({row['path']: row for row in records}.values())
        S.save(MANIFEST, {'schema': 'q5_checkpoint_recovery_execution_v1', 'version': 1, 'source': S.desc(SOURCE),
                          'registered_at': S.now(), 'base_queue_registration': S.desc(ROOT / 'registration.json'),
                          'registry': R.registration_descriptor(), 'pause_proof': S.desc(PAUSE),
                          'process_absence_evidence': absence_binding, 'checkpoint_cpu_resume_proof': S.desc(CPU_PROOF),
                          'sources': sources, 'same_formal_F3_and_E3_core': True,
                          'training_protocol_or_hyperparameters_changed': False,
                          'checkpoint_recovery_is_not_natural_CSBrain15_handoff': True,
                          'user_authorization': 'Explicit human continue on 2026-10-10',
                          'cpu_auditor': S.desc(S.REPORT / 'audit_completed_finetune_checkpoints_v5.py'),
                          'cost_collector': S.desc(S.REPORT / 'cost_aggregation_v6.py')}, immutable=True)
    manifest = S.read(MANIFEST)
    require(manifest['source'] == S.desc(SOURCE) and manifest['base_queue_registration'] == S.desc(ROOT / 'registration.json')
            and manifest['registry'] == R.registration_descriptor(), 'Recovery execution manifest changed')
    for row in manifest['sources']: S.checked(row)
    return S.desc(MANIFEST)


def prepare_recovery():
    if (ROOT / 'owner.json').exists(): raise ValueError('Q5 owner already exists; do not duplicate')
    # Retained exclusive intent prevents concurrent or repeated partial recovery
    # from creating an ambiguous second closure. A failed transaction requires
    # explicit inspection and a new recovery version, never a blind retry.
    ROOT.mkdir(parents=True, exist_ok=True)
    with INTENT.open('x', encoding='utf-8') as stream:
        json.dump({'schema': 'q5_disk_recovery_exclusive_intent_v1', 'pid': os.getpid(),
                   'created_at': S.now(), 'source': S.desc(SOURCE)}, stream)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    require(not MANIFEST.exists(), 'Existing recovery manifest requires inspection; do not recreate a closure')
    require(not any((S.OUT / 'process_loss_closures').glob('lost_f3_*.json')),
            'Existing process-loss closure requires inspection; do not duplicate it')
    for name in ('sync_boundary', 'sync_boundary_v2'):
        require(S.read(S.OUT / 'execution_acceleration_v1' / name / 'user_pause.json') == {'paused': True},
                'Strict paused control must verify before archival or owner removal')
    absence = inspect_absence()
    folder = ROOT / 'recoveries' / ('resume_' + str(time.time_ns()))
    folder.mkdir(parents=True)
    absence_path = folder / 'process_absence.json'; S.save(absence_path, absence, immutable=True)
    manifest_binding = register(S.desc(absence_path))
    pause = S.read(PAUSE)
    for row in pause['retained']: S.checked(row)
    fit = R.fit_directory('CSBrain-lastblock-finetune', 33, 4)
    latest = S.desc(fit / 'latest.pt')
    preserved, = [d for d in pause['retained'] if Path(d['path']).name == 'latest.pt']
    require(latest['sha256'] == preserved['sha256'] == '8844db8f6443ecb7777730fd06aa33655fa6be8b1aa0fee46b4b447983b43782', 'Paused checkpoint bytes changed')
    attempt, = pause['attempt_snapshots']
    require(attempt['original']['sha256'] == attempt['snapshot']['sha256'] == 'e190ee531b30275322398edeeab6a93c4db4a28dc239510e28a59ea4939fd52f', 'Paused physical observation differs')
    S.checked(attempt['original']); S.checked(attempt['snapshot'])
    old_owner = Q4.ROOT / 'owner.json'
    require(S.read(old_owner) == pause['owner_snapshot'], 'Stale old Q4 owner changed')
    old_attempt_path = (S.WORK / pause['owner_snapshot']['linux_owner']).with_name('status.json')
    for path, name in ((old_owner, 'old_q4_owner.json'), (old_attempt_path, 'old_q4_attempt_status.json')):
        shutil.copy2(path, folder / name)
        require(S.digest(path) == S.digest(folder / name), 'Old source/owner bytes failed archival')
    closure = {'schema': 'lost_f3_process_closure_v1', 'status': 'process_absence_observed', 'passed': True,
               'closure_source': S.desc(SOURCE), 'recovery_registration': manifest_binding,
               'condition': 'CSBrain-lastblock-finetune', 'seed': 33, 'fold': 4,
               'original_attempt': attempt['original'], 'attempt_snapshot': attempt['snapshot'],
               'pause_proof': S.desc(PAUSE), 'checkpoint_cpu_resume_proof': S.desc(CPU_PROOF),
               'resume_checkpoint': latest, 'preserved_checkpoint': preserved,
               'old_owner_snapshot': S.desc(folder / 'old_q4_owner.json'),
               'old_queue_attempt_snapshot': S.desc(folder / 'old_q4_attempt_status.json'),
               'process_absence_evidence': S.desc(absence_path),
               'durable': {'epoch': 8, 'optimizer_updates': 32, 'backward_calls': 216},
               'observed_progress': {'epoch': 9, 'optimizer_updates': 36, 'backward_calls': 243},
               'known_extra_lower_bounds': {'optimizer_updates': 4, 'backward_calls': 27},
               'unflushed_tail_counters': None, 'final_elapsed_seconds': None, 'graceful_terminal_flush': False,
               'actual_exit_time': None, 'exit_code': None, 'reason': None, 'observed_at': S.now(),
               'overnight_pause_excluded_from_training': True, 'old_attempt_bytes_unchanged': True,
               'known_extra_not_added_again_to_original_returned_counters': True}
    closure_path = S.OUT / 'process_loss_closures' / ('lost_f3_' + str(time.time_ns()) + '.json')
    S.save(closure_path, closure, immutable=True)
    closure_binding = S.desc(closure_path)
    from . import lost_f3_closure_v1 as L
    L.verify_recovery_registration(manifest_binding)
    require(L.verify_lost_f3_attempt(attempt['original'], S.read(S.checked(attempt['original'])),
            {'condition':'CSBrain-lastblock-finetune','seed':33,'fold':4}) is not None,
            'Lost process closure must independently verify before recovery')
    # Requery the old process objects immediately before removing only their
    # proven-stale exclusive owner. Old status/trace/source files remain intact.
    inspect_absence()
    require(S.digest(old_owner) == S.digest(folder / 'old_q4_owner.json'), 'Stale owner raced with archival')
    old_owner.unlink()
    for name in ('sync_boundary', 'sync_boundary_v2'):
        marker = S.OUT / 'execution_acceleration_v1' / name / 'user_pause.json'
        require(S.read(marker) == {'paused': True}, 'Explicit paused control marker differs')
        archived = folder / (name + '_pause_marker.json'); shutil.copy2(marker, archived)
        require(S.digest(marker) == S.digest(archived), 'Pause control archive differs')
        marker.unlink()  # User explicitly revoked pause by saying continue.
    S.save(folder / 'recovery_ready.json', {'passed': True, 'source': S.desc(SOURCE), 'closure': closure_binding,
                                          'recovery_registration': manifest_binding, 'prepared_at': S.now()}, immutable=True)
    return manifest_binding, closure_binding


class RecoveryStorage:
    def __init__(self, previous, manifest, closure):
        self.previous, self.manifest, self.closure = previous, manifest, closure
    def __getattr__(self, name): return getattr(self.previous, name)
    def save(self, path, value, *, immutable=False):
        target = Path(path).resolve()
        if target.is_relative_to(ROOT.resolve()) and target.name in ('status.json','owner.json'):
            value = {**value, 'recovery_registration': self.manifest, 'recovery_execution_source': S.desc(SOURCE),
                     'recovery_closure': self.closure}
        return self.previous.save(path, value, immutable=immutable)


def recovery_driver(manifest, closure):
    driver = Q5.isolated_driver()
    driver.S = RecoveryStorage(driver.S, manifest, closure)
    tree = __import__('ast').parse(Path(Q5.Q1.__file__).read_text(encoding='utf-8'))
    ast = __import__('ast')
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_tail')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),driver.__dict__)
    original = driver.evaluate_tail
    def recovered_cpu_gate(owner):
        if Q5.Q1.all_complete(tuple(c.condition_id for c in Q5.Q1.P.CONDITIONS)):
            for row in S.read(S.checked(manifest))['sources']: S.checked(row)
            path = S.REPORT / 'audit_completed_finetune_checkpoints_v5.py'
            spec = importlib.util.spec_from_file_location('recovered_q5_cpu_verification',path)
            auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
            value = auditor.audit()
            report = ROOT / ('pre_test_finetune_cpu_audit_v5_' + str(time.time_ns()) + '.json')
            S.save(report,value,immutable=True)
            require(value['all_135_finetune_checkpoints_verified'] is True and not value['errors']
                    and not value['deferred_completed_fits'] and value['checkpoint_loads']==405,
                    'Recovered actual all135 FT CPU405 gate must pass before any test inference')
            S.save(ROOT/'pre_test_finetune_cpu_gate.json',{'passed':True,'audit':S.desc(report),'checkpoint_loads':405,
                   'test_forward_calls_this_check':0,'source':S.desc(Q5.SOURCE),'execution_registration':S.desc(ROOT/'registration.json'),
                   'execution_version':5,'recovery_registration':manifest,'recovery_execution_source':S.desc(SOURCE),
                   'recovery_closure':closure,'actual_cpu_auditor':S.desc(path)})
        return original(owner)
    driver.evaluate_tail=recovered_cpu_gate
    return driver


def run():
    manifest, closure = prepare_recovery()
    print(json.dumps({'recovery_registration':manifest,'process_loss_closure':closure,'checkpoint_epoch':8}),flush=True)
    recovery_driver(manifest,closure).run(resume=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run: run()
    else: print(json.dumps(inspect_absence()))
