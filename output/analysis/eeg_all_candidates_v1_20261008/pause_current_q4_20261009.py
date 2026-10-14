"""Explicit user pause of exact Q4/F3 objects; no training or optimizer work."""
import ctypes
from pathlib import Path
import shutil
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R
import sync_qualification_boundary_guard_v1 as G


def run():
    for path in (G.ROOT / 'user_pause.json', S.OUT / 'execution_acceleration_v1/sync_boundary_v2/user_pause.json'):
        G.require(S.read(path) == {'paused': True}, 'Exact user pause marker required')
    guard_state = S.read(S.OUT / 'execution_acceleration_v1/sync_boundary_v2/state.json')
    G.require(guard_state.get('dispatcher_suspended') is True and guard_state.get('automatic_resume_revoked') is True,
              'Old dispatcher must already be held with automatic resume revoked')
    G.require(not (S.OUT / 'continuation_queue_v5/owner.json').exists()
              and not (S.OUT / 'execution_acceleration_v1/sync_qualification_owner.json').exists(),
              'Unexpected newer queue/qualification owner; inspect before pausing')
    native = G.backend(); refs = []; controls = []
    folder = S.OUT / 'pause_events' / ('q4_user_pause_' + str(time.time_ns()))
    folder.mkdir(parents=True)
    state = {'status': 'pause_preflight', 'source': S.desc(Path(__file__)), 'requested_by': 'explicit human user',
             'started_at': S.now(), 'actions': [], 'dispatcher_already_held_no_additional_suspend': True,
             'automatic_resume_authorized': False, 'model_optimizer_GPU_operations': 0, 'processes_terminated': 0}
    try:
        owner, _ = G.boundary()
        for role in G.PIDS:
            ref = native.capture(role); refs.append(ref)
            G.require(not native.exited(ref), 'Current training object exited before pause')
            if role != 'dispatcher':
                handle = native.kernel.OpenProcess(0x1000 | 0x100000 | 0x0800, False, ref['pid'])
                G.require(handle, 'Cannot capture exact reversible suspension handle')
                values = [ctypes.c_ulonglong() for _ in range(4)]
                G.require(native.kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)), 'Creation unavailable')
                G.require(str(values[0].value) == ref['creation_100ns'], 'PID changed during control capture')
                controls.append({**ref, 'handle': handle})
        state.update(identities={ref['role']: G.H2.public_identity(ref) for ref in refs}, owner_snapshot=owner)
        S.save(folder / 'state.json', state)
        G.require(G.boundary()[0] == owner, 'Queue boundary changed')
        # Freeze the actual GPU Python before its launcher. The dispatcher
        # already has one native suspend from the exited guardian; do not add two.
        for ref in reversed(controls):
            status = native.ntdll.NtSuspendProcess(ref['handle'])
            state['actions'].append({'role': ref['role'], 'pid': ref['pid'], 'native_status': int(status), 'at': S.now()})
            S.save(folder / 'state.json', state)
            G.require(status == 0, 'Native pause failed; retain exact partial actions')
        active = [R.fit_directory('CSBrain-lastblock-finetune', seed, fold) for seed in G.Q1.P.SEEDS for fold in G.Q1.P.FOLDS
                  if R.fit_status('CSBrain-lastblock-finetune', seed, fold)['status'] == 'training']
        G.require(len(active) == 1, 'Expected one current fit')
        fit = active[0]; retained = []
        for name in ('latest.pt', 'train_history.json', 'validation_history.json', 'progress.json', 'started.json', 'status.json'):
            path = fit / name
            if path.exists():
                copy = folder / name
                shutil.copy2(path, copy)
                G.require(S.digest(path) == S.digest(copy), 'Pause byte copy differs')
                retained.append(S.desc(copy))
        attempts = []
        seed, fold = int(fit.parent.name.split('_')[1]), int(fit.name.split('_')[1])
        for path in (S.OUT / 'finetune_v2_attempts/CSBrain-lastblock-finetune').glob('*/attempt.json'):
            value = S.read(path)
            if value.get('kind') == 'formal' and value.get('seed') == seed and value.get('fold') == fold and value.get('status') == 'running':
                copy = folder / 'buffered_attempt.json'; shutil.copy2(path, copy)
                G.require(S.digest(copy) == S.digest(path), 'Buffered observation copy differs')
                attempts.append({'original': S.desc(path), 'snapshot': S.desc(copy),
                                 'unflushed_tail_unknown_if_runtime_lost': True})
        state.update(status='paused_by_user', finished_at=S.now(), registry_status=R.status_summary(),
                     condition='CSBrain-lastblock-finetune', seed=seed, fold=fold,
                     active_progress=S.read(fit / 'progress.json'), retained=retained, attempt_snapshots=attempts,
                     training_runtime_memory_preserved=True, checkpoint_tensors_not_loaded=True,
                     pause_wall_excluded_from_training=True,
                     resume_note='Only after explicit user continuation: recheck all births/owner. Resume GPU then launcher once; dispatcher already held once and must be resumed or handed off only through a new verified control. Never duplicate queues. If runtime is lost, use complete checkpoint and keep unflushed physical tail unknown.')
        S.save(folder / 'state.json', state)
        S.save(folder / 'pause_complete.json', state, immutable=True)
        print({'paused': True, 'proof': S.desc(folder / 'pause_complete.json'), 'registry_status': state['registry_status'],
               'active_fit': {'condition': state['condition'], 'seed': seed, 'fold': fold}, 'progress': state['active_progress']})
    except BaseException as error:
        state.update(status='pause_incomplete_preserve_actions', error_type=type(error).__name__, error=str(error))
        S.save(folder / 'state.json', state)
        raise
    finally:
        for ref in controls: native.close(ref)
        for ref in refs: native.close(ref)


if __name__ == '__main__': run()
