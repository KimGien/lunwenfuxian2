"""Scoped checkpoint handoff; inspect first, act only with exact held handles."""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import types

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R, protocol as P
from tridep_ccml.eeg_all_candidates import handoff_supervisor_v2 as H2

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'continuation_queue_v4/handoff'
QUEUE = S.OUT / 'continuation_queue_v3'
OWNER_ID = 'a45f0f1e767e44dc83150ab00a3971bc'
PIDS = {'dispatcher': 29932, 'launcher': 47936, 'gpu_child': 40360}
PARENTS = {'dispatcher': 51120, 'launcher': 29932, 'gpu_child': 47936}
BIRTHS = {'dispatcher': '2026-10-09T08:48:04.339447+08:00',
          'launcher': '2026-10-09T12:38:36.093393+08:00',
          'gpu_child': '2026-10-09T12:38:36.105085+08:00'}


def require(value, message):
    if not value:
        raise ValueError(message)


def backend():
    module = types.ModuleType('scoped_acceleration_native')
    module.__package__ = H2.__package__
    exec(compile(Path(H2.__file__).read_text(encoding='utf-8'), H2.__file__, 'exec'), module.__dict__)
    module.ROLE_PIDS, module.ROLE_PARENTS, module.CREATED_LOCAL = PIDS, PARENTS, BIRTHS
    return module.WindowsBackend()


def boundary():
    owner = S.read(QUEUE / 'owner.json')
    require(owner['id'] == OWNER_ID and owner['pid'] == PIDS['dispatcher']
            and owner['child_pid'] == PIDS['launcher'] and owner['child_job'] == 'CSBrain-lastblock-finetune-fit',
            'Scoped CSBrain queue identity changed')
    attempt = (S.WORK / owner['linux_owner']).resolve().with_name('status.json')
    require(attempt.is_relative_to((QUEUE / 'attempts').resolve()), 'Attempt outside original Q3 root')
    value = S.read(attempt)
    require(value['status'] == 'running' and value['child_pid'] == PIDS['launcher']
            and value['job']['module'] == 'finetune_v2', 'Scoped child job changed')
    return owner, attempt


def inspect():
    native = backend()
    refs = []
    try:
        owner, _ = boundary()
        for role in PIDS:
            refs.append(native.capture(role))
        require(all(not native.exited(ref) for ref in refs) and native.descendants() == {PIDS['launcher'], PIDS['gpu_child']},
                'Unexpected live descendant or child exited')
        require(boundary()[0] == owner, 'Boundary raced with inspection')
        path = ROOT / ('inspection_' + str(time.time_ns()) + '.json')
        S.save(path, {'source': S.desc(SOURCE), 'owner': S.desc(QUEUE / 'owner.json'),
                      'roles': {ref['role']: H2.public_identity(ref) for ref in refs},
                      'checked_at': S.now(), 'process_actions': 0}, immutable=True)
        return S.desc(path)
    finally:
        for ref in refs:
            native.close(ref)


def _integer(value):
    return type(value) is int and value >= 0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_checkpoint_payloads(initial, saved, contract, registration, *, condition, seed, fold, lock, registry, torch, F):
    """Check actual CPU tensors without constructing a model or CUDA generator."""
    names = contract['trainable_parameter_names']
    heads = [name for name in names if name.startswith('head.')]
    backbone = [name for name in names if name.startswith('encoder.')]
    require(heads and backbone and len(names) == len(set(names)) == len(heads) + len(backbone),
            'Trainable parameter group coverage differs')
    ordered = heads + backbone
    split = registration['folds'][f'fold_{fold}']
    raw_by_id = {row['subject_id']: row['raw'] for row in registration['raw_sources']}
    expected_raw = {sid: raw_by_id[sid] for sid in split['train'] + split['val']}
    expected_keys = set(initial['state_dict'])
    require(set(names).issubset(expected_keys) and set(contract['frozen_parameter_names']).issubset(expected_keys)
            and not set(names).intersection(contract['frozen_parameter_names']), 'Initial parameter coverage differs')
    for value in (initial, saved):
        require(all(value[k] == v for k, v in {'condition': condition, 'seed': seed, 'fold': fold,
                'registry': registry, 'condition_source_lock': lock, 'raw_sources': expected_raw}.items()), 'Checkpoint identity differs')
        require(type(value['seed']) is int and type(value['fold']) is int, 'Checkpoint seed/fold types differ')
        require(set(value['state_dict']) == expected_keys, 'Checkpoint tensor keys differ')
        require(all(torch.is_tensor(t) and t.device.type == 'cpu' and bool(torch.isfinite(t).all())
                    and t.shape == initial['state_dict'][name].shape and t.dtype == initial['state_dict'][name].dtype
                    for name, t in value['state_dict'].items()), 'Checkpoint tensor shape/dtype/finiteness differs')
        require(all(_integer(value[key]) for key in ('epoch', 'optimizer_updates', 'backward_calls')),
                'Checkpoint counters must be strict nonnegative integers')
        require(type(value['scheduler_state_dict']['extended']) is bool, 'Scheduler extension must be boolean')
    require(initial['epoch'] == initial['optimizer_updates'] == initial['backward_calls'] == 0
            and initial['history'] == initial['subject_order'] == []
            and initial['best_selected_state_dict'] is None and initial['best_selected_state_sha256'] is None
            and initial['sampler_position'] == {'epoch_completed': 0, 'next_batch': 0}
            and initial['scheduler_state_dict'] == {'policy': 'finetune', 'completed_epoch': 0, 'extended': False},
            'Initial checkpoint is not complete epoch-zero state')
    epoch = saved['epoch']
    n = len(split['train'])
    require(1 <= epoch <= P.POLICIES['finetune'].final_epochs and len(saved['history']) == len(saved['subject_order']) == epoch,
            'Complete durable epoch/history required')
    require(saved['optimizer_updates'] == math.ceil(n / 8) * epoch and saved['backward_calls'] == n * epoch,
            'Durable optimizer/backward counts differ')
    require(saved['sampler_position'] == {'epoch_completed': epoch, 'next_batch': 0}
            and saved['scheduler_state_dict'] == {'policy': 'finetune', 'completed_epoch': epoch,
                                                  'extended': saved['scheduler_state_dict']['extended']}, 'Full sampler/scheduler missing')
    frozen = expected_keys - set(names)  # Includes frozen buffers, as sealed F1 does.
    frozen_digest = F.state_digest({name: value for name, value in initial['state_dict'].items() if name in frozen})
    require(saved['initial_state_sha256'] == initial['initial_state_sha256'] == F.state_digest(initial['state_dict'])
            and saved['frozen_initial_sha256'] == initial['frozen_initial_sha256'] == frozen_digest, 'Initial state differs')
    for name in frozen:
        require(torch.equal(saved['state_dict'][name], initial['state_dict'][name]), 'Frozen parameter changed')
    best = saved['best_selected_state_dict']
    require(isinstance(best, dict) and set(best) == set(names)
            and all(torch.is_tensor(t) and t.device.type == 'cpu' and bool(torch.isfinite(t).all())
                    and t.shape == initial['state_dict'][name].shape and t.dtype == initial['state_dict'][name].dtype
                    for name, t in best.items())
            and F.state_digest(best) == saved['best_selected_state_sha256'], 'Embedded best state differs/nonfinite')
    for value in (initial, saved):
        optimizer = value['optimizer_state_dict']
        groups = optimizer['param_groups']
        require(len(groups) == 2 and [g['group_name'] for g in groups] == ['head', 'backbone'], 'Optimizer groups differ')
        ids = [pid for group in groups for pid in group['params']]
        require(all(type(pid) is int for pid in ids) and ids == list(range(len(ordered)))
                and len(groups[0]['params']) == len(heads) and len(groups[1]['params']) == len(backbone),
                'Optimizer parameter IDs/order/coverage differ')
        expected_states = set()
        for index, (group, steps) in enumerate(zip(groups, (value['optimizer_updates'], math.ceil(n / 8) * max(0, value['epoch'] - 10)))):
            base = P.POLICIES['finetune'].head_lr if index == 0 else P.POLICIES['finetune'].backbone_lr
            expected_lr = (P.POLICIES['finetune'].head_lr if index == 0 else 0.) if value['epoch'] == 0 else base * P.learning_rate_multiplier('finetune', value['epoch'], parameter_group=group['group_name'])
            require(_finite(group['lr']) and math.isclose(group['lr'], expected_lr, rel_tol=1e-12, abs_tol=1e-15)
                    and group['weight_decay'] == P.POLICIES['finetune'].weight_decay
                    and group['betas'] == (0.9, 0.999) and group['eps'] == 1e-8, 'Optimizer hyperparameters differ')
            for pid in group['params']:
                state = optimizer['state'].get(pid, {})
                if not steps:
                    require(not state, 'Unused optimizer parameter has state')
                    continue
                expected_states.add(pid)
                require(state and torch.is_tensor(state['step']) and state['step'].numel() == 1
                        and math.isfinite(float(state['step'])) and float(state['step']) == steps, 'Optimizer step differs')
                parameter = value['state_dict'][ordered[pid]]
                for key in ('exp_avg', 'exp_avg_sq'):
                    require(key in state and torch.is_tensor(state[key]) and state[key].device.type == 'cpu'
                            and state[key].shape == parameter.shape and state[key].dtype == parameter.dtype
                            and bool(torch.isfinite(state[key]).all()), 'Optimizer moment shape/dtype/finiteness differs')
        require(set(optimizer['state']) == expected_states, 'Optimizer state IDs missing/extra')
        rng = value['rng']
        require(set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}, 'Complete RNG missing')
        random.Random().setstate(rng['python'])
        F.np.random.RandomState().set_state(rng['numpy'])
        cpu_rng = rng['torch_cpu']
        require(torch.is_tensor(cpu_rng) and cpu_rng.device.type == 'cpu' and cpu_rng.dtype == torch.uint8
                and len(cpu_rng.shape) == 1 and cpu_rng.numel() > 0, 'CPU RNG byte state malformed')
        torch.Generator(device='cpu').set_state(cpu_rng)
        cuda_rng = rng['torch_cuda']
        initial_cuda = initial['rng']['torch_cuda']
        require(isinstance(cuda_rng, list) and len(cuda_rng) == len(initial_cuda) >= 1, 'CUDA RNG device coverage differs')
        require(all(torch.is_tensor(item) and item.device.type == 'cpu' and item.dtype == torch.uint8
                    and len(item.shape) == 1 and item.numel() > 0 and item.shape == initial_cuda[i].shape
                    for i, item in enumerate(cuda_rng)), 'CUDA RNG saved byte shapes differ')
    extended = False
    for i, (row, people) in enumerate(zip(saved['history'], saved['subject_order']), 1):
        require(all(_integer(row[key]) for key in ('epoch', 'optimizer_updates', 'backward_calls'))
                and row['epoch'] == i and row['optimizer_updates'] == math.ceil(n / 8) * i
                and row['backward_calls'] == n * i and sorted(people) == sorted(split['train'])
                and row['window_indices'] == F.bag_indices(i).tolist() and row['unfrozen'] is (i > 10)
                and all(_finite(row[key]) for key in ('train_bag_ce', 'train_eval_ce', 'val_eval_ce')),
                'Epoch identity/coverage/numerical history differs')
        decision = P.training_decision('finetune', [item['val_eval_ce'] for item in saved['history'][:i]], extended=extended)
        require(not decision['stop'] or i == epoch, 'Checkpoint contains epochs after original stopping decision')
        if decision['extend']:
            extended = True
    require(saved['scheduler_state_dict']['extended'] is extended, 'Scheduler extension/history differs')
    return {'condition': condition, 'seed': seed, 'fold': fold, 'epoch': epoch,
            'optimizer_updates': saved['optimizer_updates'], 'backward_calls': saved['backward_calls'],
            'best_selected_state_sha256': saved['best_selected_state_sha256'], 'resume_full_state_verified': True,
            'model_operations': 0, 'checkpoint_load_device': 'cpu',
            'optimizer_group_order_and_moment_shapes_verified': True,
            'best_selected_parameter_finiteness_verified': True,
            'cpu_rng_local_restore_and_cuda_saved_byte_shapes_verified': True,
            'raw_role_descriptors_verified_without_loading_raw': True}


def validate_checkpoint(folder):
    import torch
    from tridep_ccml.eeg_all_candidates import finetune as F
    condition, seed, fold = folder.parent.parent.name, int(folder.parent.name.split('_')[1]), int(folder.name.split('_')[1])
    lock = R._condition_lock(condition)
    contract = S.read(S.checked(lock))['contract']
    paths = [folder / 'initial.pt', folder / 'latest.pt']
    before = [S.desc(path) for path in paths]
    initial, saved = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
    result = validate_checkpoint_payloads(initial, saved, contract, R.load_registration(), condition=condition,
                seed=seed, fold=fold, lock=lock, registry=R.registration_descriptor(), torch=torch, F=F)
    require([S.desc(path) for path in paths] == before, 'Held checkpoint bytes changed during CPU validation')
    return {**result, 'cpu_checkpoint_loads': 2, 'original_checkpoint_bindings': before}


def terminate_held(native, refs, state, state_path):
    """After any termination request, never automatically resume the old tree."""
    state.update(termination_started=True, termination_requested_roles=[], exit_verified_roles=[])
    S.save(state_path, state)
    for ref in reversed(refs):
        # Persist intent before the API: a call whose return is interrupted is
        # unknown, not proof that all three original processes remain healthy.
        state['termination_request_inflight_role'] = ref['role']
        S.save(state_path, state)
        require(native.kernel.TerminateProcess(ref['handle'], 0xEED40004), 'Scoped termination failed')
        state['termination_requested_roles'].append(ref['role'])
        require(native.kernel.WaitForSingleObject(ref['handle'], 15000) == 0, 'Scoped process exit not proven')
        state['exit_verified_roles'].append(ref['role'])
        state['termination_request_inflight_role'] = None
        S.save(state_path, state)


def resume_unterminated_paused(native, paused, state):
    """Child-first recovery; a failed child resume leaves the parent held."""
    require(not state.get('termination_started'), 'Cannot resume a partially terminated process tree')
    native.ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
    native.ntdll.NtResumeProcess.restype = ctypes.c_long
    state['resume_returned_roles'] = []
    for ref in reversed(paused):
        try:
            require(native.ntdll.NtResumeProcess(ref['handle']) == 0, 'Scoped resume failed')
        except BaseException as error:
            state.update(resume_failed_role=ref['role'], resume_error=str(error))
            return False
        state['resume_returned_roles'].append(ref['role'])
    return True


def arm(expected_binding):
    expected = S.read(S.checked(expected_binding))
    require(expected['source'] == S.desc(SOURCE), 'Inspected controller source changed')
    native = backend()
    refs = []
    paused = []
    state_path = ROOT / ('state_' + str(time.time_ns()) + '.json')
    state = {'phase': 'capturing', 'passed': False, 'source': S.desc(SOURCE), 'expected': expected_binding}
    try:
        owner, old_attempt = boundary()
        require(expected['owner'] == S.desc(QUEUE / 'owner.json'), 'Inspected owner bytes changed')
        for role in PIDS:
            read_ref = native.capture(role)
            try:
                require(H2.public_identity(read_ref) == expected['roles'][role], 'Process object identity changed')
                handle = native.kernel.OpenProcess(0x1000 | 0x100000 | 0x0800 | 0x0001, False, read_ref['pid'])
                require(handle, 'Could not capture scoped control handle')
                try:
                    times = [ctypes.c_ulonglong() for _ in range(4)]
                    require(native.kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)), 'Cannot inspect held handle')
                    require(str(times[0].value) == read_ref['creation_100ns'], 'PID reuse crossed control capture')
                    refs.append({**read_ref, 'handle': handle})
                except BaseException:
                    native.kernel.CloseHandle(handle)
                    raise
            finally:
                native.close(read_ref)
        require(native.descendants() == {PIDS['launcher'], PIDS['gpu_child']}, 'Unexpected child; no control performed')
        # Dispatcher first prevents the old route launching a new job during the handoff.
        for ref in refs:
            require(native.ntdll.NtSuspendProcess(ref['handle']) == 0, 'Could not suspend scoped process')
            paused.append(ref)
        require(boundary()[0] == owner, 'Owner changed while capturing checkpoint')
        state.update(phase='held_checkpoint_verification', held_identities=expected['roles'])
        S.save(state_path, state)
        active = [R.fit_directory('CSBrain-lastblock-finetune', seed, fold) for seed in P.SEEDS for fold in P.FOLDS
                  if R.fit_status('CSBrain-lastblock-finetune', seed, fold)['status'] == 'training']
        require(len(active) == 1, 'Expected exactly one current formal fit')
        folder = active[0]
        # Full torch checkpoint is atomically renamed; the checkpoint itself is authoritative.
        verified = validate_checkpoint(folder)
        snapshot = ROOT / ('snapshot_' + str(time.time_ns()))
        snapshot.mkdir(parents=True)
        retained = []
        for name in ('initial.pt', 'latest.pt', 'train_history.json', 'validation_history.json', 'progress.json', 'started.json', 'status.json'):
            if (folder / name).exists():
                target = snapshot / name
                shutil.copy2(folder / name, target)
                require(S.digest(target) == S.digest(folder / name), 'Snapshot copy differs')
                retained.append(S.desc(target))
        latest = S.desc(snapshot / 'latest.pt')
        matched = []
        for path in (S.OUT / 'finetune_v2_attempts/CSBrain-lastblock-finetune').glob('*/attempt.json'):
            record = S.read(path)
            if record.get('kind') == 'formal' and record.get('seed') == verified['seed'] and record.get('fold') == verified['fold']:
                matched.append((path, record))
        require(len(matched) == 1 and matched[0][1]['status'] == 'running', 'Exact current F2 trace required')
        trace_path, trace = matched[0]
        preserved_trace = snapshot / 'f2_attempt.json'
        shutil.copy2(trace_path, preserved_trace)
        require(S.digest(preserved_trace) == S.digest(trace_path), 'Held F2 trace snapshot differs')
        extra_updates = trace['optimizer_step_returned'] - verified['optimizer_updates']
        extra_backwards = trace['backward_returned'] - verified['backward_calls']
        require(extra_updates >= 0 and extra_backwards >= 0, 'Physical counters precede durable checkpoint')
        protected = list(S.read(QUEUE / 'registration.json')['sources']) + [S.desc(QUEUE / 'registration.json')]
        for item in protected:
            S.checked(item)
        state.update(phase='verified_before_scoped_termination', checkpoint=latest, checkpoint_validation=verified)
        S.save(state_path, state)
        terminate_held(native, refs, state, state_path)
        processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, check=True).stdout.strip()
        require(not processes, 'Another CUDA compute process remains; qualification refused')
        archived_owner = snapshot / 'q3_owner.json'
        shutil.copy2(QUEUE / 'owner.json', archived_owner)
        owner_descriptor = S.desc(archived_owner)
        # The original bytes remain in the archive; remove only the now-dead exclusive owner.
        require(S.digest(QUEUE / 'owner.json') == owner_descriptor['sha256'], 'Owner changed before archive')
        (QUEUE / 'owner.json').unlink()
        shutil.copy2(old_attempt, snapshot / 'q3_attempt_status.json')
        proof = {'schema': 'acceleration_safe_stop_v1', 'mode': 'stopped_training', 'passed': True,
                 'source': S.desc(SOURCE), 'finished_at': S.now(), 'all_training_processes_stopped': True, 'gpu_idle': True,
                 'stopped_process_ids': list(PIDS.values()), 'held_process_identities': expected['roles'],
                 'exit_verified_roles': state['exit_verified_roles'],
                 'checkpoint': latest, 'checkpoints': [latest], **verified,
                 'train_history': S.desc(snapshot / 'train_history.json'), 'registration': R.registration_descriptor(),
                 'f2_attempt': S.desc(trace_path), 'f2_attempt_snapshot': S.desc(preserved_trace),
                 'old_owner_snapshot': owner_descriptor, 'old_q3_attempt_status': S.desc(snapshot / 'q3_attempt_status.json'),
                 'protected_sources': protected, 'checkpoint_evidence': retained,
                 'recovery_extra_optimizer_updates': extra_updates, 'recovery_extra_backward_calls': extra_backwards,
                 'unreturned_calls_remain_unknown': True, 'old_attempt_seconds_remain_unknown': True,
                 'checkpoint_history_is_authoritative_if_json_write_lagged': True,
                 'formal_training_batch_precision_selection_budget_unchanged': True}
        path = ROOT / ('safe_stop_' + str(time.time_ns()) + '.json')
        S.save(path, proof, immutable=True)
        state.update(phase='safe_stop_complete', passed=True, proof=S.desc(path))
        S.save(state_path, state)
        return S.desc(path)
    except BaseException as error:
        state.update(error_type=type(error).__name__, error=str(error))
        if state.get('termination_started'):
            state.update(phase='stopped_pending_recovery', old_training_resumed=False,
                         automatic_resume_forbidden_after_termination_request=True)
        else:
            resumed = resume_unterminated_paused(native, paused, state)
            state.update(phase='verification_failed_old_training_resumed' if resumed else 'verification_failed_resume_incomplete',
                         old_training_resumed=resumed)
        S.save(state_path, state)
        raise
    finally:
        for ref in refs:
            native.close(ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected', help='workspace-relative exact inspection JSON')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    if bool(args.expected) != bool(args.expected_sha256):
        parser.error('Both expected path and SHA are required')
    print(json.dumps(arm({'path': args.expected, 'sha256': args.expected_sha256}) if args.expected else inspect()))
