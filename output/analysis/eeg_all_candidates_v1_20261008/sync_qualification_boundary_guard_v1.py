"""Drain the current full CSBrain job, test sequentially, then resume Q4.

Inspection is read-only. Only --arm suspends the exactly identified dispatcher.
The current GPU children are never suspended or terminated. Qualification starts
only after both children exit0 and all fifteen real fits verify. No optimizer or
model code is imported while waiting. All old training sources remain unchanged.
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
import types

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R
from tridep_ccml.eeg_all_candidates import handoff_supervisor_v2 as H2
from tridep_ccml.eeg_all_candidates import continuation_queue as Q1, continuation_queue_v4 as Q4

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'execution_acceleration_v1/sync_boundary'
OWNER_ID = '3266366ec77046d09bbd58372bd9bee5'
PIDS = {'dispatcher': 40840, 'launcher': 18224, 'gpu_child': 32880}
PARENTS = {'dispatcher': 15428, 'launcher': 40840, 'gpu_child': 18224}
BIRTHS = {'dispatcher': '2026-10-09T20:11:02.526585+08:00',
          'launcher': '2026-10-09T20:12:16.296164+08:00',
          'gpu_child': '2026-10-09T20:12:16.306482+08:00'}
CONDITION = 'CSBrain-lastblock-finetune'
QUALIFIERS = ('qualify_sync_reduction_v1.py', 'qualify_training_sync_v1.py')


def require(value, message):
    if not value:
        raise ValueError(message)


class UserPauseRequested(RuntimeError):
    """A durable user control marker revokes this guard's automatic resume."""


def require_no_user_pause():
    marker = ROOT / 'user_pause.json'
    if marker.exists():
        value = S.read(marker)
        require(type(value) is dict and set(value) == {'paused'} and value['paused'] is True,
                'Malformed user-pause marker; explicit operator resolution required')
        raise UserPauseRequested('User pause remains active; no probe start or automatic Q4 resume')


def backend():
    module = types.ModuleType('natural_sync_boundary_native')
    module.__package__ = H2.__package__
    exec(compile(Path(H2.__file__).read_text(encoding='utf-8'), H2.__file__, 'exec'), module.__dict__)
    module.ROLE_PIDS, module.ROLE_PARENTS, module.CREATED_LOCAL = PIDS, PARENTS, BIRTHS
    return module.WindowsBackend()


def boundary():
    path = Q4.ROOT / 'owner.json'
    owner = S.read(path)
    require(owner['id'] == OWNER_ID and owner['pid'] == PIDS['dispatcher']
            and owner['child_pid'] == PIDS['launcher'] and owner['child_job'] == CONDITION + '-fit',
            'Current Q4/CSBrain process identity changed')
    attempt_path = (S.WORK / owner['linux_owner']).resolve().with_name('status.json')
    require(attempt_path.is_relative_to((Q4.ROOT / 'attempts').resolve()), 'Attempt escaped Q4')
    attempt = S.read(attempt_path)
    require(attempt['status'] == 'running' and attempt['child_pid'] == PIDS['launcher']
            and attempt['job']['module'] == 'finetune_v3', 'Current F3 child route changed')
    return owner, attempt_path


def self_identity(native):
    handle = native.kernel.OpenProcess(0x1000, False, os.getpid())
    require(handle, 'Cannot inspect guardian process identity')
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        require(native.kernel.GetProcessTimes(handle, *(ctypes.byref(x) for x in times)), 'Guardian process times unavailable')
        return {'pid': os.getpid(), 'creation_100ns': str(times[0].value),
                'parent_pid': native.parents().get(os.getpid()), 'source': S.desc(SOURCE)}
    finally:
        native.kernel.CloseHandle(handle)


def inspect():
    native = backend(); refs = []
    try:
        owner, _ = boundary()
        for role in PIDS:
            refs.append(native.capture(role))
        require(all(not native.exited(ref) for ref in refs), 'A current CSBrain process has exited')
        require(native.descendants() == {PIDS['launcher'], PIDS['gpu_child']}, 'Unexpected current child')
        require(boundary()[0] == owner, 'Boundary raced during inspection')
        path = ROOT / ('inspection_' + str(time.time_ns()) + '.json')
        S.save(path, {'source': S.desc(SOURCE), 'owner': S.desc(Q4.ROOT / 'owner.json'),
                      'registration': S.desc(Q4.ROOT / 'registration.json'),
                      'roles': {ref['role']: H2.public_identity(ref) for ref in refs},
                      'checked_at': S.now(), 'process_actions': 0}, immutable=True)
        return S.desc(path)
    finally:
        for ref in refs: native.close(ref)


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=20)
    require(not result.stdout.strip(), 'Another GPU compute process remains')
    return {'passed': True, 'stdout': result.stdout, 'checked_at': S.now()}


def current_process_identity(native, expected):
    handle = native.kernel.OpenProcess(0x1000 | 0x100000, False, expected['pid'])
    require(handle, 'Guard process is no longer present')
    try:
        require(native.kernel.WaitForSingleObject(handle, 0) == 0x102, 'Guard process has exited')
        times = [ctypes.c_ulonglong() for _ in range(4)]
        require(native.kernel.GetProcessTimes(handle, *(ctypes.byref(x) for x in times)), 'Cannot inspect guard creation time')
        require(str(times[0].value) == expected['creation_100ns']
                and native.parents().get(expected['pid']) == expected['parent_pid'], 'Guardian PID object changed')
    finally:
        native.kernel.CloseHandle(handle)


def verify_live_boundary(proof_descriptor):
    """Read-only verifier called by a qualifier before importing model code.

    The source-bound guardian records successful native suspension and holds
    the actual dispatcher handle until both qualifier processes have exited.
    A live guardian, exact dispatcher object and its published lease are all
    independently rechecked here. This function performs no process control.
    """
    require_no_user_pause()
    proof = S.read(S.checked(proof_descriptor))
    require(proof.get('schema') == 'sync_qualification_safe_boundary_v1'
            and proof.get('source') == S.desc(SOURCE) and proof.get('passed') is True,
            'Reviewed natural boundary proof required')
    require(proof['queue_owner'] == S.desc(Q4.ROOT / 'owner.json')
            and proof['registration'] == S.desc(Q4.ROOT / 'registration.json'), 'Held queue bytes changed')
    state = S.read(S.WORK / proof['guardian_state_path'])
    lease = S.read(S.checked(proof['guardian_owner']))
    require(state.get('phase') == 'qualification_running' and state.get('dispatcher_suspended') is True
            and state.get('boundary_proof') == proof_descriptor
            and state.get('source') == proof['source']
            and state.get('guardian_identity') == proof['guardian_identity']
            and state.get('held_process_identities') == proof['held_process_identities']
            and proof['guardian_identity'].get('source') == proof['source']
            and lease['identity'] == proof['guardian_identity'], 'Guardian suspension lease is not live')
    native = backend(); ref = None
    try:
        current_process_identity(native, proof['guardian_identity'])
        ref = native.capture('dispatcher')
        require(not native.exited(ref) and H2.public_identity(ref) == proof['held_process_identities']['dispatcher'],
                'Exact held dispatcher process changed')
        require(not native.descendants(), 'A training child remains or a new child started')
        require(proof.get('gpu_children_exit_verified') is True
                and proof.get('launcher_exit_code') == proof.get('gpu_child_exit_code') == 0,
                'Both held GPU child exits0 are required')
        require_no_user_pause()
        return {'passed': True, 'gpu_children_exit_verified': True,
                'held_dispatcher_running_suspended': True,
                'suspension_basis': 'successful native suspension; source-bound live guardian retains handle and publishes lease',
                'process_control_calls': 0}
    finally:
        if ref is not None: native.close(ref)


def resume_dispatcher(native, ref):
    require_no_user_pause()
    native.ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
    native.ntdll.NtResumeProcess.restype = ctypes.c_long
    require(not native.exited(ref), 'Old dispatcher exited while held')
    require(native.ntdll.NtResumeProcess(ref['handle']) == 0, 'Could not resume held dispatcher')


def qualifier_descendants(native, launcher_pid):
    """Find surviving children even after the direct launcher has exited."""
    parents, found, pending = native.parents(), set(), [launcher_pid]
    while pending:
        parent = pending.pop()
        for pid, actual_parent in parents.items():
            if actual_parent == parent and pid not in found and pid != launcher_pid:
                found.add(pid); pending.append(pid)
    return found


def require_qualifier_cleanup(native, process):
    require(process is None or process.poll() is not None,
            'Qualification launcher remains alive; preserve exclusive boundary')
    if process is not None:
        require(not qualifier_descendants(native, process.pid),
                'Qualification descendant remains alive; preserve exclusive boundary')
    require(not (S.OUT / 'execution_acceleration_v1/sync_qualification_owner.json').exists(),
            'Qualification process lease remains; cleanup is not proven')
    gpu_idle()


def arm(expected_descriptor):
    expected = S.read(S.checked(expected_descriptor))
    require(expected['source'] == S.desc(SOURCE), 'Inspected guardian source changed')
    # Seal only the additional experimental sources; no training registration
    # or source is modified. They are checked again before each probe starts.
    prototype_paths = [S.WORK / 'tridep_ccml/eeg_all_candidates' / name for name in
                       ('sync_reduction_v1.py', 'sync_subject_update_v1.py', 'sync_state_digest_v1.py', 'sync_qualification_gate_v1.py')]
    experimental_sources = [S.desc(path) for path in prototype_paths] + [S.desc(S.REPORT / name) for name in QUALIFIERS]
    native = backend(); refs = []; suspended = False; qualification_started = False; active_process = None
    lease_path = ROOT / 'guardian_owner.json'; state_path = ROOT / 'state.json'
    state = {'phase': 'capturing_exact_handles', 'source': S.desc(SOURCE), 'passed': False,
             'gpu_children_suspended_or_terminated': False, 'qualification_sources': experimental_sources,
             'formal_training_protocol_changed': False, 'dispatcher_suspended': False}
    guardian_identity = self_identity(native)
    S.save(lease_path, {'identity': guardian_identity, 'expected': expected_descriptor,
                        'started_at': S.now()}, immutable=True)
    try:
        require_no_user_pause()
        owner, _ = boundary()
        require(expected['owner'] == S.desc(Q4.ROOT / 'owner.json')
                and expected['registration'] == S.desc(Q4.ROOT / 'registration.json'), 'Exact inspected boundary bytes changed')
        for role in PIDS:
            ref = native.capture(role, arm=role == 'dispatcher'); refs.append(ref)
            require(H2.public_identity(ref) == expected['roles'][role], 'Held process object differs from inspection')
        require(all(not native.exited(ref) for ref in refs), 'Child exit raced with capture')
        native.suspend_dispatcher(refs[0]); suspended = True
        state.update(phase='waiting_for_CSBrain_full15', dispatcher_suspended=True,
                     guardian_identity=guardian_identity, held_process_identities=expected['roles'],
                     current_GPU_training_continues=True, waiting_started_at=S.now())
        S.save(state_path, state)
        while not (native.exited(refs[1]) and native.exited(refs[2])):
            require_no_user_pause()
            require(boundary()[0] == owner, 'Suspended queue boundary changed')
            require(native.descendants().issubset({PIDS['launcher'], PIDS['gpu_child']}), 'Unexpected training descendant')
            time.sleep(2)
        require_no_user_pause()
        require(native.exit_code(refs[1]) == native.exit_code(refs[2]) == 0, 'Full CSBrain child failed; skip probe')
        require(Q1.all_complete((CONDITION,)), 'Actual full fifteen CSBrain fit evidence must verify')
        registration = Q4.register()
        for source in registration['sources'] + experimental_sources: S.checked(source)
        gpu_idle()
        folder = R.fit_directory(CONDITION, 22, 5)
        proof = {'schema': 'sync_qualification_safe_boundary_v1', 'passed': True,
                 'source': S.desc(SOURCE), 'queue': 'continuation_queue_v4',
                 'registration': S.desc(Q4.ROOT / 'registration.json'), 'original_training_registration': R.registration_descriptor(),
                 'queue_owner': S.desc(Q4.ROOT / 'owner.json'), 'held_process_identities': expected['roles'],
                 'gpu_children_exit_verified': True, 'launcher_exit_code': 0, 'gpu_child_exit_code': 0,
                 'held_dispatcher_running_suspended': True, 'gpu_idle': True, 'zero_optimizer_probe_only': True,
                 'completed_CSBrain_fits_verified': 15, 'condition': CONDITION, 'seed': 22, 'fold': 5,
                 'registered_fold_ids': R.load_registration()['folds']['fold_5'],
                 'checkpoint': S.desc(folder / 'latest.pt'), 'train_history': S.desc(folder / 'train_history.json'),
                 'guardian_identity': guardian_identity, 'guardian_owner': S.desc(lease_path),
                 'guardian_state_path': state_path.relative_to(S.WORK).as_posix(),
                 'experimental_sources': experimental_sources, 'created_at': S.now()}
        proof_path = ROOT / ('boundary_' + str(time.time_ns()) + '.json')
        S.save(proof_path, proof, immutable=True); proof_descriptor = S.desc(proof_path)
        state.update(phase='qualification_running', current_GPU_training_continues=False,
                     boundary_proof=proof_descriptor, qualification_started_at=S.now(), outcomes=[])
        S.save(state_path, state)
        verify_live_boundary(proof_descriptor)
        for name in QUALIFIERS:
            require_no_user_pause()
            for source in registration['sources'] + experimental_sources: S.checked(source)
            require_qualifier_cleanup(native, active_process)
            require_no_user_pause()
            qualification_started = True
            log_path = ROOT / (name.removesuffix('.py') + '.log')
            arguments = [str(S.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', str(S.REPORT / name),
                         '--run', '--boundary-proof', str(proof_path), '--boundary-sha', proof_descriptor['sha256']]
            with log_path.open('ab') as log:
                process = subprocess.Popen(arguments, cwd=S.WORK, stdout=log, stderr=subprocess.STDOUT)
                active_process = process
                state.update(active_qualifier=name, active_qualifier_launcher_pid=process.pid)
                S.save(state_path, state)
                # A timeout keeps Q4 held until this experiment's exact process
                # and CUDA cleanup can be inspected. It never starts a competitor.
                code = process.wait(timeout=1200)
            require_no_user_pause()
            require_qualifier_cleanup(native, active_process)
            active_process = None
            state['outcomes'].append({'qualifier': name, 'returncode': code, 'log': S.desc(log_path),
                                      'qualification_passed': code == 0,
                                      'qualification_declined': code == 2,
                                      'execution_failed': code not in (0, 2)})
            state.update(active_qualifier=None, active_qualifier_launcher_pid=None)
            S.save(state_path, state)
            require(code in (0, 2), 'Qualifier execution failed; skip remaining probes')
        require_no_user_pause()
        require_qualifier_cleanup(native, active_process)
        resume_dispatcher(native, refs[0]); suspended = False
        state.update(phase='qualification_finished_Q4_resumed', dispatcher_suspended=False,
                     passed=True, boundary_workflow_completed=True,
                     all_qualifications_passed=all(item['qualification_passed'] for item in state['outcomes']),
                     finished_at=S.now(), no_automatic_runtime_installation=True,
                     formal_optimizer_updates=0)
        S.save(state_path, state)
        return state
    except BaseException as error:
        state.update(error_type=type(error).__name__, error=str(error), passed=False)
        if not isinstance(error, UserPauseRequested):
            try:
                require_no_user_pause()
            except UserPauseRequested as pause_error:
                state.update(interrupted_error_type=type(error).__name__, interrupted_error=str(error))
                error = pause_error
            except BaseException as control_error:
                state.update(phase='invalid_user_pause_marker_Q4_held' if suspended else 'invalid_user_pause_marker_before_suspend',
                             dispatcher_suspended=suspended, control_error=str(control_error))
                S.save(state_path, state)
                raise
        if isinstance(error, UserPauseRequested):
            state.update(phase='user_pause_Q4_held' if suspended else 'user_pause_before_suspend',
                         dispatcher_suspended=suspended, automatic_resume_revoked_by_user=True,
                         error_type=type(error).__name__, error=str(error))
            S.save(state_path, state)
            return state
        if suspended:
            safe_to_resume = False
            try:
                if not qualification_started and not native.exited(refs[2]):
                    # The original GPU child still owns its existing run;
                    # resuming its waiting parent introduces no new probe.
                    safe_to_resume = True
                else:
                    # An alive CPU-only qualifier could allocate CUDA later.
                    # After the original GPU child exits, GPU-idle and lease
                    # cleanup are required even if no probe was launched yet.
                    require_qualifier_cleanup(native, active_process)
                    safe_to_resume = True
            except BaseException:
                safe_to_resume = False
            if safe_to_resume:
                try:
                    resume_dispatcher(native, refs[0]); suspended = False
                    state.update(phase='probe_failed_Q4_resumed', dispatcher_suspended=False)
                except BaseException as resume_error:
                    state.update(phase='resume_unverified_Q4_held', resume_error=str(resume_error))
            else:
                state.update(phase='probe_cleanup_unverified_Q4_held', dispatcher_suspended=True)
        else:
            state['phase'] = 'refused_before_suspend'
        S.save(state_path, state)
        raise
    finally:
        for ref in refs: native.close(ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--expected')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    if args.arm:
        require(args.expected and args.expected_sha256, 'Exact inspection path and SHA required')
        print(json.dumps(arm({'path': args.expected, 'sha256': args.expected_sha256})))
    else:
        print(json.dumps(inspect()))
