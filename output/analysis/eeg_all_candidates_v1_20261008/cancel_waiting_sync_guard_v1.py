"""Cancel only this waiting optional probe guardian; keep current training live."""
import argparse
import ctypes
from pathlib import Path
import shutil
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S
import sync_qualification_boundary_guard_v1 as G

SOURCE = Path(__file__).resolve()


def run(*, execute=False):
    G.require(S.digest(Path(G.__file__)) == '3a40f7b1d13b0bb85e868376f72c809e3d938d676ad9270be894e28a768308d3', 'Reviewed waiting guardian differs')
    G.require_no_user_pause()
    owner = S.read(G.ROOT / 'guardian_owner.json')
    identity = owner['identity']
    G.require(identity['pid'] == 36216 and identity['parent_pid'] == 37380
              and identity['creation_100ns'] == '134360249272116468', 'Exact optional guardian object differs')
    native = G.backend(); refs = []; handle = None; guard_suspended = False; termination_started = False
    try:
        handle = native.kernel.OpenProcess(0x1000 | 0x100000 | (0x0800 | 0x0001 if execute else 0), False, identity['pid'])
        G.require(handle and native.kernel.WaitForSingleObject(handle, 0) == 0x102, 'Guardian is not live')
        times = [ctypes.c_ulonglong() for _ in range(4)]
        G.require(native.kernel.GetProcessTimes(handle, *(ctypes.byref(x) for x in times)), 'Cannot inspect guardian object')
        G.require(str(times[0].value) == identity['creation_100ns']
                  and native.parents().get(identity['pid']) == identity['parent_pid'], 'Guardian PID reused')
        for role in G.PIDS:
            refs.append(native.capture(role, arm=execute and role == 'dispatcher'))
        G.require(all(not native.exited(ref) for ref in refs), 'Training child exited; waiting-only cancellation is refused')
        G.boundary()
        if execute:
            G.require(native.ntdll.NtSuspendProcess(handle) == 0, 'Could not hold only optional guardian')
            guard_suspended = True
        state = S.read(G.ROOT / 'state.json')
        G.require(state.get('phase') == 'waiting_for_CSBrain_full15'
                  and state.get('dispatcher_suspended') is True and state.get('current_GPU_training_continues') is True
                  and not state.get('active_qualifier'), 'Optional guardian reached a probe; do not cancel blindly')
        G.require(state.get('source') == S.desc(Path(G.__file__))
                  and identity.get('source') == state['source']
                  and state.get('guardian_identity') == identity
                  and state.get('held_process_identities') == {ref['role']: G.H2.public_identity(ref) for ref in refs},
                  'Published waiting state is not bound to the exact guardian/training objects')
        G.require(not (S.OUT / 'execution_acceleration_v1/sync_qualification_owner.json').exists(),
                  'An optional qualification lease remains; waiting-only cancellation is refused')
        G.require(all(not native.exited(ref) for ref in refs), 'Training exit raced with cancellation')
        G.require(not any(parent == identity['pid'] for parent in native.parents().values()), 'Guardian already launched a child')
        result = {'source': S.desc(SOURCE), 'guardian_source': S.desc(Path(G.__file__)),
                  'guardian_identity': identity, 'training_identities': {ref['role']: G.H2.public_identity(ref) for ref in refs},
                  'qualification_model_optimizer_GPU_calls': 0, 'training_child_process_actions': 0,
                  'reason': 'Optional global-GPU-idle probe may block healthy training when unrelated C+G applications remain',
                  'observed_state': state, 'checked_at': S.now(), 'process_control_requested': execute}
        if not execute:
            return result
        snapshot = G.ROOT / ('cancel_waiting_' + str(time.time_ns()))
        snapshot.mkdir(parents=True)
        for name in ('guardian_owner.json', 'state.json'):
            shutil.copy2(G.ROOT / name, snapshot / name)
            G.require(S.digest(G.ROOT / name) == S.digest(snapshot / name), 'Waiting state archive differs')
        G.require_no_user_pause()
        termination_started = True
        G.require(native.kernel.TerminateProcess(handle, 0xEED70007), 'Exact optional guardian termination failed')
        G.require(native.kernel.WaitForSingleObject(handle, 15000) == 0, 'Optional guardian exit unverified')
        G.require_no_user_pause()
        G.require(all(not native.exited(ref) for ref in refs), 'Training children no longer live; inspect boundary before resume')
        G.resume_dispatcher(native, refs[0])
        result.update(passed=True, optional_guardian_exit_verified=True, original_Q4_dispatcher_resumed=True,
                      original_GPU_training_continues=True, archived_owner=S.desc(snapshot / 'guardian_owner.json'),
                      archived_state=S.desc(snapshot / 'state.json'), finished_at=S.now())
        path = G.ROOT / ('cancel_waiting_complete_' + str(time.time_ns()) + '.json')
        S.save(path, result, immutable=True)
        return S.desc(path)
    finally:
        if guard_suspended and not termination_started:
            native.ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
            native.ntdll.NtResumeProcess.restype = ctypes.c_long
            native.ntdll.NtResumeProcess(handle)
        for ref in refs: native.close(ref)
        if handle is not None: native.kernel.CloseHandle(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    print(run(execute=args.run))
