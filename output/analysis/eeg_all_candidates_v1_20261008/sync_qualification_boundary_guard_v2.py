"""Natural Q4-to-Q5 handoff; optional timing never blocks mandatory fits.

Current GPU children continue unchanged. Only their waiting dispatcher is held.
After all fifteen CSBrain fits exit0, profiles run only with a truly quiet GPU.
Known desktop/game contexts defer those optional profiles; they do not cancel
required fitting. Unknown compute contexts keep the exclusive boundary held.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R, continuation_queue as Q1
from tridep_ccml.eeg_all_candidates import continuation_queue_v4 as Q4, continuation_queue_v5 as Q5
from tridep_ccml.eeg_all_candidates import acceleration_prepare_v2 as AP2
import sync_qualification_boundary_guard_v1 as G1

G1_SHA = '3a40f7b1d13b0bb85e868376f72c809e3d938d676ad9270be894e28a768308d3'
if hashlib.sha256(Path(G1.__file__).read_bytes()).hexdigest() != G1_SHA:
    raise ValueError('Reviewed native identity/pause delegate changed')

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'execution_acceleration_v1/sync_boundary_v2'
QUALIFIERS = ('qualify_sync_reduction_v2.py', 'qualify_training_sync_v2.py')
OBSERVED_GRAPHICS_PATHS = frozenset(path.replace('\\', '/').casefold() for path in (
    'C:/Riot Games/League of Legends/LeagueClientUxRender.exe',
    'C:/Riot Games/League of Legends/Game/League of Legends.exe',
    'C:/Program Files/NVIDIA Corporation/NVIDIA app/CEF/NVIDIA Overlay.exe'))


def require(value, message):
    if not value: raise ValueError(message)


def require_no_user_pause():
    G1.require_no_user_pause()
    path = ROOT / 'user_pause.json'
    if path.exists():
        value = S.read(path)
        require(type(value) is dict and set(value) == {'paused'} and value['paused'] is True,
                'Invalid V2 user pause control; explicit operator resolution required')
        raise G1.UserPauseRequested('User revoked natural-boundary work and automatic continuation')


def gpu_observation():
    return AP2.load_qualifier().gpu_observation()


def training_gpu_scope(observation):
    """Permit only observed graphics workloads, never another compute worker."""
    require(type(observation.get('contexts')) is list, 'GPU process inventory is unavailable')
    for context in observation['contexts']:
        kind = context.get('type')
        path = str(context.get('process_name', '')).replace('\\', '/').casefold()
        require(kind == 'G' or (kind == 'C+G' and path in OBSERVED_GRAPHICS_PATHS),
                'Another or unknown GPU compute worker remains: ' + str(context.get('pid')))
    return {'passed': True, 'contexts': observation['contexts'],
            'observation': observation,
            'scope': 'single experiment CUDA worker; observed desktop graphics may coexist',
            'optional_performance_measurement_quiet': not observation['contexts']}


def guardian_identity(native):
    value = G1.self_identity(native)
    return {**value, 'source': S.desc(SOURCE), 'delegated_native_source': S.desc(Path(G1.__file__))}


def inspect():
    native = G1.backend(); refs = []
    try:
        owner, _ = G1.boundary()
        for role in G1.PIDS: refs.append(native.capture(role))
        require(all(not native.exited(ref) for ref in refs), 'Current CSBrain process already exited')
        require(native.descendants() == {G1.PIDS['launcher'], G1.PIDS['gpu_child']}, 'Unexpected child tree')
        require(G1.boundary()[0] == owner, 'Q4 boundary changed during inspection')
        path = ROOT / ('inspection_' + str(time.time_ns()) + '.json')
        S.save(path, {'source': S.desc(SOURCE), 'owner': S.desc(Q4.ROOT / 'owner.json'),
                      'q4_registration': S.desc(Q4.ROOT / 'registration.json'),
                      'q5_registration': S.desc(Q5.ROOT / 'registration.json'),
                      'roles': {ref['role']: G1.H2.public_identity(ref) for ref in refs},
                      'process_actions': 0, 'checked_at': S.now()}, immutable=True)
        return S.desc(path)
    finally:
        for ref in refs: native.close(ref)


def verify_live_boundary(binding):
    proof = S.read(S.checked(binding)); require_no_user_pause()
    require(proof.get('schema') == 'sync_qualification_safe_boundary_v2'
            and proof.get('source') == S.desc(SOURCE) and proof.get('passed') is True, 'Reviewed V2 boundary required')
    state = S.read(ROOT / 'state.json')
    lease = S.read(S.checked(proof['guardian_owner']))
    require(state.get('phase') == 'qualification_running' and state.get('dispatcher_suspended') is True
            and state.get('boundary_proof') == binding and lease['identity'] == proof['guardian_identity']
            and state.get('source') == proof['source']
            and state.get('guardian_identity') == proof['guardian_identity']
            and state.get('held_process_identities') == proof['held_process_identities']
            and proof['guardian_identity'].get('source') == proof['source'], 'V2 guardian lease is not live')
    require(proof['queue_owner'] == S.desc(Q4.ROOT / 'owner.json')
            and proof['registration'] == S.desc(Q4.ROOT / 'registration.json'), 'Actual held Q4 bytes changed')
    native = G1.backend(); ref = None
    try:
        G1.current_process_identity(native, proof['guardian_identity'])
        ref = native.capture('dispatcher')
        require(not native.exited(ref) and G1.H2.public_identity(ref) == proof['held_process_identities']['dispatcher']
                and not native.descendants(), 'Held dispatcher or drained child boundary changed')
        require(proof.get('gpu_children_exit_verified') is True
                and type(proof.get('launcher_exit_code')) is int and type(proof.get('gpu_child_exit_code')) is int
                and proof['launcher_exit_code'] == proof['gpu_child_exit_code'] == 0,
                'Actual two normal child exits required')
        require_no_user_pause()
        return {'passed': True, 'gpu_children_exit_verified': True, 'held_dispatcher_running_suspended': True,
                'process_control_calls': 0, 'suspension_basis': 'source-bound live guardian holds exact suspended dispatcher handle'}
    finally:
        if ref is not None: native.close(ref)


def require_probe_cleanup(native, process):
    if process is not None:
        require(process.poll() is not None, 'Optional qualifier still alive')
        require(not G1.qualifier_descendants(native, process.pid), 'Orphan qualifier remains')
    require(not (S.OUT / 'execution_acceleration_v1/sync_qualification_owner.json').exists(), 'Optional qualification lease remains')
    return training_gpu_scope(gpu_observation())


def completed_artifacts():
    require(Q1.all_complete(('CSBrain-lastblock-finetune',)), 'Actual CSBrain fifteen completion evidence missing')
    records = []
    for seed in Q1.P.SEEDS:
        for fold in Q1.P.FOLDS:
            directory = R.fit_directory('CSBrain-lastblock-finetune', seed, fold)
            status = R.fit_status('CSBrain-lastblock-finetune', seed, fold)
            R._validate_complete(status['evidence'], directory)
            records.extend(status['evidence'][key] for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'))
            records.extend(S.desc(directory / name) for name in ('complete.json', 'status.json'))
    return records


def spawn_q5():
    with (ROOT / 'q5_dispatcher.log').open('ab') as log:
        return subprocess.Popen([str(S.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', '-m',
                    'tridep_ccml.eeg_all_candidates.continuation_queue_v5', '--resume'], cwd=S.WORK,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000)


def arm(expected_binding):
    expected = S.read(S.checked(expected_binding)); require_no_user_pause()
    require(expected['source'] == S.desc(SOURCE), 'Inspected guardian V2 changed')
    Q5.register()
    require(expected['q5_registration'] == S.desc(Q5.ROOT / 'registration.json'), 'Inspected Q5 registration differs')
    require(expected['q4_registration'] == S.desc(Q4.ROOT / 'registration.json'), 'Inspected Q4 registration differs')
    experimental = [S.desc(S.REPORT / name) for name in QUALIFIERS]
    experimental += [S.desc(S.WORK / 'tridep_ccml/eeg_all_candidates' / name) for name in
                     ('sync_qualification_gate_v2.py', 'sync_reduction_v1.py', 'sync_state_digest_v1.py', 'sync_subject_update_v1.py')]
    native = G1.backend(); refs = []; held = False; termination_started = False; process = None
    lease_path = ROOT / 'guardian_owner.json'; state_path = ROOT / 'state.json'
    identity = guardian_identity(native)
    S.save(lease_path, {'identity': identity, 'expected': expected_binding, 'started_at': S.now()}, immutable=True)
    state = {'phase': 'capturing', 'source': S.desc(SOURCE), 'passed': False, 'dispatcher_suspended': False,
             'GPU_training_child_suspend_or_terminate_calls': 0, 'experimental_sources': experimental,
             'formal_training_protocol_changed': False, 'guardian_identity': identity}
    try:
        owner, attempt_path = G1.boundary()
        require(expected['owner'] == S.desc(Q4.ROOT / 'owner.json'), 'Inspected owner bytes changed')
        for role in G1.PIDS:
            ref = native.capture(role, arm=role == 'dispatcher'); refs.append(ref)
            require(G1.H2.public_identity(ref) == expected['roles'][role], 'Exact current process differs')
        require(all(not native.exited(ref) for ref in refs), 'Natural boundary raced with capture')
        require_no_user_pause()
        native.suspend_dispatcher(refs[0]); held = True
        state.update(phase='waiting_for_CSBrain_full15', dispatcher_suspended=True,
                     current_GPU_training_continues=True, held_process_identities=expected['roles'])
        S.save(state_path, state)
        while not (native.exited(refs[1]) and native.exited(refs[2])):
            require_no_user_pause()
            require(G1.boundary()[0] == owner, 'Held Q4 owner changed')
            require(native.descendants().issubset({G1.PIDS['launcher'], G1.PIDS['gpu_child']}), 'Unexpected original child')
            time.sleep(2)
        require_no_user_pause()
        require(native.exit_code(refs[1]) == native.exit_code(refs[2]) == 0, 'Original full CSBrain child failed')
        artifacts = completed_artifacts()
        protected = S.read(Q4.ROOT / 'registration.json')['sources'] + [S.desc(Q4.ROOT / 'registration.json')]
        for item in protected + experimental + artifacts: S.checked(item)
        observation = gpu_observation(); continuation_scope = training_gpu_scope(observation)
        outcomes = []
        if observation['contexts']:
            deferred_path = ROOT / ('optional_sync_qualification_deferred_' + str(time.time_ns()) + '.json')
            S.save(deferred_path, {'status': 'deferred_external_GPU_contexts', 'source': S.desc(SOURCE),
                                   'created_at': S.now(), 'observed_GPU_contexts': observation,
                                   'experimental_sources': experimental, 'model_constructions': 0,
                                   'GPU_forward_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
                                   'formal_fits': 0, 'GPU_speedup_claimed': False,
                                   'mandatory_fits_continue_with_original_F3': True}, immutable=True)
            state['deferred_qualification'] = S.desc(deferred_path)
        else:
            folder = R.fit_directory('CSBrain-lastblock-finetune', 22, 5)
            proof_path = ROOT / ('boundary_' + str(time.time_ns()) + '.json')
            proof = {'schema': 'sync_qualification_safe_boundary_v2', 'passed': True, 'source': S.desc(SOURCE),
                     'queue': 'continuation_queue_v4', 'registration': S.desc(Q4.ROOT / 'registration.json'),
                     'original_training_registration': R.registration_descriptor(), 'queue_owner': S.desc(Q4.ROOT / 'owner.json'),
                     'held_process_identities': expected['roles'], 'gpu_children_exit_verified': True,
                     'launcher_exit_code': 0, 'gpu_child_exit_code': 0, 'held_dispatcher_running_suspended': True,
                     'gpu_idle': True, 'zero_optimizer_probe_only': True, 'completed_CSBrain_fits_verified': 15,
                     'condition': 'CSBrain-lastblock-finetune', 'seed': 22, 'fold': 5,
                     'registered_fold_ids': R.load_registration()['folds']['fold_5'],
                     'checkpoint': S.desc(folder / 'latest.pt'), 'train_history': S.desc(folder / 'train_history.json'),
                     'guardian_identity': identity, 'guardian_owner': S.desc(lease_path),
                     'guardian_state_path': state_path.relative_to(S.WORK).as_posix(),
                     'experimental_sources': experimental, 'created_at': S.now()}
            S.save(proof_path, proof, immutable=True); binding = S.desc(proof_path)
            state.update(phase='qualification_running', boundary_proof=binding, current_GPU_training_continues=False,
                         outcomes=outcomes); S.save(state_path, state)
            for name in QUALIFIERS:
                require_no_user_pause(); require_probe_cleanup(native, process)
                for item in protected + experimental: S.checked(item)
                if gpu_observation()['contexts']:
                    state['remaining_optional_qualification_deferred_after_GPU_race'] = True
                    break
                log_path = ROOT / (name.removesuffix('.py') + '.log')
                with log_path.open('ab') as log:
                    process = subprocess.Popen([str(S.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', str(S.REPORT / name),
                                                '--run', '--boundary-proof', str(proof_path), '--boundary-sha', binding['sha256']],
                                               cwd=S.WORK, stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000)
                    state.update(active_qualifier=name, active_qualifier_launcher_pid=process.pid); S.save(state_path, state)
                    code = process.wait(timeout=1200)
                require_no_user_pause(); require_probe_cleanup(native, process); process = None
                outcomes.append({'qualifier': name, 'returncode': code, 'log': S.desc(log_path),
                                 'qualification_passed': code == 0, 'qualification_declined': code == 2,
                                 'execution_failed': code not in (0, 2)})
                state.update(active_qualifier=None, active_qualifier_launcher_pid=None); S.save(state_path, state)
                if code not in (0, 2): break
        require_no_user_pause(); continuation_scope = require_probe_cleanup(native, process)
        Q5.register()
        require(expected['q5_registration'] == S.desc(Q5.ROOT / 'registration.json'), 'Q5 registration changed before handoff')
        for item in protected + artifacts: S.checked(item)
        snapshot = ROOT / ('handoff_snapshot_' + str(time.time_ns())); snapshot.mkdir(parents=True)
        for original, name in ((Q4.ROOT / 'owner.json', 'q4_owner.json'), (attempt_path, 'q4_attempt_status.json')):
            shutil.copy2(original, snapshot / name)
            require(S.digest(original) == S.digest(snapshot / name), 'Old Q4 byte archive differs')
        state.update(phase='verified_before_Q4_parent_termination', current_GPU_training_continues=False,
                     outcomes=outcomes, completed_CSBrain_fits_verified=15); S.save(state_path, state)
        require_no_user_pause(); termination_started = True
        require(native.kernel.TerminateProcess(refs[0]['handle'], 0xEED80008), 'Exact old Q4 parent termination failed')
        require(native.kernel.WaitForSingleObject(refs[0]['handle'], 15000) == 0, 'Exact Q4 parent exit unverified')
        held = False; state.update(dispatcher_suspended=False, old_Q4_exit_verified=True)
        require_no_user_pause(); continuation_scope = require_probe_cleanup(native, process)
        old_owner = S.desc(snapshot / 'q4_owner.json')
        require(S.digest(Q4.ROOT / 'owner.json') == old_owner['sha256'], 'Old owner changed before archive')
        (Q4.ROOT / 'owner.json').unlink()
        handoff = {'passed': True, 'version': 5, 'source': S.desc(SOURCE), 'finished_at': S.now(),
                   'q5_registration': S.desc(Q5.ROOT / 'registration.json'), 'old_q4_registration': S.desc(Q4.ROOT / 'registration.json'),
                   'old_dispatcher_terminated_after_completed_children': True,
                   'old_dispatcher_exit_code': native.exit_code(refs[0]), 'launcher_exit_code': 0, 'gpu_child_exit_code': 0,
                   'completed_CSBrain_fits_verified': 15, 'GPU_training_child_suspend_or_terminate_calls': 0,
                   'held_process_identities': expected['roles'], 'old_owner_snapshot': old_owner,
                   'old_attempt_snapshot': S.desc(snapshot / 'q4_attempt_status.json'), 'protected_sources': protected,
                   'completed_CSBrain_artifacts': artifacts, 'GPU_training_scope': continuation_scope,
                   'optional_qualification_outcomes': outcomes, 'deferred_qualification': state.get('deferred_qualification'),
                   'formal_optimizer_updates_this_guardian': 0, 'old_sources_and_reports_preserved': True}
        S.save(Q5.ROOT / 'handoff_complete.json', handoff, immutable=True)
        state.update(phase='handoff_complete_starting_Q5', handoff=S.desc(Q5.ROOT / 'handoff_complete.json'))
        S.save(state_path, state)
        require_no_user_pause(); child = spawn_q5()
        state.update(phase='Q5_started', passed=True, dispatcher_suspended=False,
                     q5_launcher_pid=child.pid, all_qualifications_passed=len(outcomes) == len(QUALIFIERS) and all(x['qualification_passed'] for x in outcomes),
                     actual_sync_optimization_applied=False); S.save(state_path, state)
        # Keep the explicit tool job and stdout guardian alive for the real new queue.
        code = child.wait(); state.update(phase='Q5_finished', passed=code == 0, q5_exit_code=code,
                                         handoff_completed=True, q5_finished_at=S.now())
        S.save(state_path, state); return handoff
    except BaseException as error:
        state.update(passed=False, error_type=type(error).__name__, error=str(error))
        if not isinstance(error, G1.UserPauseRequested):
            try:
                require_no_user_pause()
            except G1.UserPauseRequested as pause_error:
                state.update(interrupted_error_type=type(error).__name__, interrupted_error=str(error))
                error = pause_error
            except BaseException as control_error:
                state.update(phase='invalid_user_pause_control', dispatcher_suspended=held,
                             control_error=str(control_error), automatic_resume_revoked=True)
                S.save(state_path, state)
                raise
        if isinstance(error, G1.UserPauseRequested):
            phase = 'user_pause_Q4_held' if held else 'user_pause_after_Q4_exit' if termination_started else 'user_pause_before_suspend'
            state.update(phase=phase, dispatcher_suspended=held, automatic_resume_revoked=True,
                         error_type=type(error).__name__, error=str(error))
        elif termination_started:
            state.update(phase='old_Q4_exit_or_Q5_start_pending_inspection', automatic_resume_old_parent_forbidden=True)
        elif held:
            try:
                require_no_user_pause()
                if native.exited(refs[2]): require_probe_cleanup(native, process)
                require_no_user_pause()
                G1.resume_dispatcher(native, refs[0]); held = False
                state.update(phase='optional_guard_failed_Q4_resumed', dispatcher_suspended=False)
            except BaseException as cleanup_error:
                state.update(phase='cleanup_unverified_Q4_held', cleanup_error=str(cleanup_error), dispatcher_suspended=True)
        else: state['phase'] = 'refused_before_suspend'
        S.save(state_path, state); raise
    finally:
        for ref in refs: native.close(ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--expected')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    if args.arm:
        require(args.expected and args.expected_sha256, 'Exact inspected boundary descriptor required')
        print(json.dumps(arm({'path': args.expected, 'sha256': args.expected_sha256})))
    else: print(json.dumps(inspect()))
