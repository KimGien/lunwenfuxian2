"""Opt-in real diagnostic batching qualification after verified GPU safe stop.

No --run: write pending without torch/model/checkpoint/raw reads.
--run --stop-proof <JSON>: require acceleration_safe_stop_v1, stopped PIDs,
idle compute GPU, stable checkpoint/history hashes and an exclusive lease.
Default eligibility is BITWISE logits AND CE only. Tolerances are descriptive.
"""
from datetime import datetime, timezone
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
import uuid

REPORT = Path(__file__).resolve().parent
WORK = REPORT.parents[2]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
ROOT = OUT / 'execution_acceleration_v1'
OWNER = ROOT / 'qualification_owner.json'
REGISTRY = OUT / 'registration.json'
REGISTRY_SHA = '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'
DELEGATE_HASHES = {'finetune.py': 'cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24',
                   'finetune_v2.py': 'fa53183ea4927daed5dcec0204e8a28ab09940f589742debda884402581c1a8b'}
SAFE_BUILDERS = {'stopped_training': REPORT / 'safe_stop_acceleration_v1.py',
                 'queue_exclusive_post_prepare': WORK / 'tridep_ccml/eeg_all_candidates/acceleration_prepare_v1.py'}
THRESHOLDS = {'eligibility': 'bitwise_only', 'analysis_logit_atol': 1e-6, 'analysis_logit_rtol': 1e-6,
              'analysis_ce_atol': 1e-7, 'analysis_tolerances_do_not_authorize_runtime': True}


def require(value, message):
    if not value:
        raise ValueError(message)


def atomic_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        return
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex[:8] + '.part')
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def descriptor(path, *, binary=False):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK), 'Evidence outside workspace')
    require(binary or path.suffix in ('.json', '.py'), 'Binary access is gated by explicit run')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest}


def checked(binding, *, binary=False):
    path = (WORK / binding['path']).resolve()
    require(descriptor(path, binary=binary) == binding, 'Stable evidence changed: ' + binding['path'])
    return path


def json_read(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix == '.json', 'Expected workspace JSON')
    return json.loads(path.read_text(encoding='utf-8'))


def pid_alive(pid):
    require(type(pid) is int and pid > 0, 'Invalid PID')
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            require(ctypes.get_last_error() == 87, 'Cannot establish stopped process')
            return False
        try:
            code = ctypes.c_ulong()
            require(kernel.GetExitCodeProcess(handle, ctypes.byref(code)), 'PID inspection failed')
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def gpu_idle():
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20, check=True)
    require(not query.stdout.strip(), 'Existing GPU compute process: qualification refused')
    return {'query': 'nvidia-smi compute-apps', 'stdout': query.stdout, 'stderr': query.stderr, 'checked_at': datetime.now(timezone.utc).isoformat()}


def windows_identity(pid):
    require(type(pid) is int and pid > 0, 'Invalid Windows process identity')
    kwargs = {'creationflags': 0x08000000} if os.name == 'nt' else {}
    query = subprocess.run(['powershell' if os.name == 'nt' else 'powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
            f'Get-CimInstance Win32_Process -Filter "ProcessId = {pid}" | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath | ConvertTo-Json -Compress'],
            capture_output=True, text=True, timeout=20, check=True, **kwargs)
    if not query.stdout.strip():
        return None
    identity = json.loads(query.stdout)
    require(identity['ProcessId'] == pid and identity['ExecutablePath'], 'Windows CIM identity unavailable')
    return identity


def preflight(proof_path):
    proof = json_read(proof_path)
    require(proof.get('schema') == 'acceleration_safe_stop_v1' and proof.get('passed') is True
            and proof.get('gpu_idle') is True, 'Explicit safe-stop proof required')
    mode = proof.get('mode', 'stopped_training')
    require(mode in ('stopped_training', 'queue_exclusive_post_prepare'), 'Unknown safety gate mode')
    windows_live = pid_alive if os.name == 'nt' else lambda pid: windows_identity(pid) is not None
    require(type(proof.get('stopped_process_ids')) is list and all(not windows_live(pid) for pid in proof['stopped_process_ids']), 'Training process still alive')
    require(checked(proof['source']) == SAFE_BUILDERS[mode].resolve(), 'Safety proof builder differs')
    queue_owner_path = None
    if mode == 'stopped_training':
        require(proof.get('all_training_processes_stopped') is True and proof['stopped_process_ids'], 'Stopped-training proof needs exited process IDs')
    else:
        queue_owner_path = checked(proof['queue_owner'] if os.name == 'nt' else proof['windows_queue_owner'])
        q = json_read(queue_owner_path)
        require(windows_live(q['pid']), 'Queue parent is not live')
        if os.name == 'nt':
            require(q.get('child_pid') in (None, os.getpid(), os.getppid()), 'Queue ownership is not exclusive to qualification')
        else:
            linux_owner = json_read(checked(proof['linux_owner']))
            require(linux_owner['pid'] in (os.getpid(), os.getppid()) and pid_alive(linux_owner['pid']), 'Linux qualification owner is not this process or launcher')
            require(q.get('child_pid') is not None, 'Windows WSL launcher missing')
        job = q.get('child_job') or proof.get('queue_child_job', '')
        require(proof['condition'] in job and 'acceleration' in job and 'prepare' in job, 'Queue job is not this condition acceleration preparation')
        identity = windows_identity(q.get('child_pid') or os.getpid())
        require(identity is not None, 'Queue child is not live')
        proof = {**proof, '_actual_queue_child_identity': identity}
    for folder in ('continuation_queue', 'continuation_queue_v2', 'continuation_queue_v3', 'continuation_queue_v4'):
        owner = OUT / folder / 'owner.json'
        if owner.exists():
            if owner.resolve() == queue_owner_path:
                continue
            value = json_read(owner)
            for key in ('pid', 'child_pid'):
                if value.get(key) is not None:
                    require(not windows_live(value[key]), 'A registered dispatcher/child is still active')
    registry = json_read(REGISTRY)
    require(descriptor(REGISTRY)['sha256'] == REGISTRY_SHA and proof['registration'] == descriptor(REGISTRY), 'Fixed registration differs')
    for name, digest in DELEGATE_HASHES.items():
        require(descriptor(WORK / 'tridep_ccml/eeg_all_candidates' / name)['sha256'] == digest, 'Sealed mathematical delegate changed')
    condition, seed, fold = proof['condition'], proof['seed'], proof['fold']
    require(type(seed) is int and seed in (11, 22, 33) and type(fold) is int and fold in range(1, 6), 'Invalid strict seed/fold identity')
    fit_id = f'{condition}/seed_{seed}/fold_{fold}'
    fit, = [f for f in registry['fits'] if f['fit_id'] == fit_id and f['stage'] == '3']
    split = registry['folds']['fold_' + str(fold)]
    folder = (OUT / 'runs' / fit_id).resolve()
    checkpoints = proof.get('checkpoints', [proof['checkpoint']] if proof.get('checkpoint') else [])
    require(type(checkpoints) is list and all(type(x) is dict for x in checkpoints), 'Bad checkpoint descriptor list')
    require(len({x['path'] for x in checkpoints}) == len(checkpoints), 'Duplicate checkpoint paths')
    for binding in checkpoints:
        path = (WORK / binding['path']).resolve()
        snapshot = path.is_relative_to((OUT / 'continuation_queue_v4/handoff').resolve()) and path.parent.name.startswith('snapshot_')
        require((path.parent == folder or (snapshot and binding in proof.get('checkpoint_evidence', [])))
                and path.name in ('latest.pt', 'best.pt'), 'Only bound fit/snapshot latest/best weights allowed')
    if checkpoints:
        history_path = checked(proof['train_history'])
        require(history_path == folder / 'train_history.json' or (history_path.parent == (WORK / checkpoints[0]['path']).resolve().parent
                and proof['train_history'] in proof.get('checkpoint_evidence', [])), 'History must belong to the specified fold/snapshot')
        history = json_read(history_path)
        require(history and all(r['epoch'] == i for i, r in enumerate(history, 1)), 'Stable complete history prefix required')
    else:
        require(seed == 11 and fold == 1, 'Fresh engineering qualification uses seed11/fold1 only')
        history = None
    return proof, registry, fit, split, checkpoints, history, gpu_idle()


def controller(P, losses):
    extended, extensions = False, []
    for epoch in range(1, len(losses) + 1):
        decision = P.training_decision('finetune', losses[:epoch], extended=extended)
        if decision['extend']:
            extensions.append(epoch); extended = True
    return {'decision': decision, 'extended': extended, 'extension_epochs': extensions}


def run(args, attempt_path, trace):
    proof, registry, fit, split, checkpoints, history, idle = preflight(args.stop_proof)
    trace.update(safe_stop_proof=descriptor(args.stop_proof), idle_gpu_check=idle, condition=fit['condition'], seed=fit['seed'], fold=fit['fold'],
                 queue_child_identity=proof.get('_actual_queue_child_identity'), checkpoint_history_prefix_from_json=history)
    atomic_json(attempt_path, trace)
    owner = {'id': uuid.uuid4().hex, 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(), 'attempt_path': attempt_path.relative_to(WORK).as_posix()}
    atomic_json(OWNER, owner, exclusive=True)
    model = raw = None
    try:
        gpu_idle()
        for binding in checkpoints:
            checked(binding, binary=True)
        sys.path.insert(0, str(WORK))
        import torch
        from tridep_ccml.eeg_all_candidates import finetune as F1, finetune_v2 as F2, protocol as P
        from tridep_ccml.eeg_all_candidates import validation_batching_v1 as B
        require(torch.cuda.is_available(), 'Real CUDA runtime unavailable')
        source_bindings = {name: descriptor(path) for name, path in [('qualifier_source', Path(__file__)),
                           ('diagnostic_wrapper_source', Path(B.__file__)), ('original_diagnostics_source', Path(F1.__file__)),
                           ('delegated_f2_source', Path(F2.__file__))]}
        trace.update(source_bindings=source_bindings, gpu_stage_entered=True)
        atomic_json(attempt_path, trace)
        weight_mode = 'stable_formal_checkpoint' if checkpoints else 'fresh_official_initialization'
        probe_split = split if checkpoints else {'train': split['train'][:-1], 'val': split['train'][-1:], 'test': split['val'] + split['test']}
        raw, raw_sources = F1._raw_sources(split, people=None if checkpoints else split['train'])
        B.validate_roles(raw, probe_split, registry['labels'])
        F1.C.runtime(fit['seed'])
        model, adapter_audit = F2.build_model(fit['condition'], seed=fit['seed'], device='cuda', window_chunk=8)
        condition_lock = F2._check_v2_lock(fit['condition'])
        if not checkpoints:
            initial_path = attempt_path.parent / 'initial_state.json'
            atomic_json(initial_path, {'kind': 'fresh_official_initialization', 'condition': fit['condition'], 'seed': fit['seed'],
                        'fold': fit['fold'], 'state_sha256': F1.state_digest(model.state_dict()), 'adapter_audit': adapter_audit,
                        'optimizer_updates': 0, 'original_fold_train_only': True}, exclusive=True)
            checkpoints = [descriptor(initial_path)]
        runtime = {'device_type': 'cuda', 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'gpu_name': torch.cuda.get_device_name(), 'python': sys.version}
        comparisons = {chunk: [] for chunk in (16, 32, 60)}
        times = {chunk: [] for chunk in B.CHUNKS}
        measurements = []
        for checkpoint in checkpoints:
            if weight_mode == 'stable_formal_checkpoint':
                saved = torch.load(checked(checkpoint, binary=True), map_location='cpu', weights_only=False)
                require((saved['condition'], saved['seed'], saved['fold']) == (fit['condition'], fit['seed'], fit['fold']), 'Checkpoint identity differs')
                require(saved['registry'] == descriptor(REGISTRY), 'Checkpoint registration differs')
                require(saved['condition_source_lock'] == condition_lock, 'Checkpoint condition source lock differs')
                epoch = saved.get('epoch', saved.get('best_epoch'))
                state_history = saved.get('history', history)
                require(type(epoch) is int and 1 <= epoch <= len(state_history), 'Checkpoint epoch outside stable history')
                if 'history' in saved:
                    require(len(state_history) == epoch and all(r['epoch'] == i for i, r in enumerate(state_history, 1)), 'Embedded checkpoint history is incomplete')
                    require(history[:epoch] == state_history[:len(history[:epoch])], 'JSON history is not the matching embedded-history prefix')
                    if len(history) < epoch:
                        require(proof.get('checkpoint_history_is_authoritative_if_json_write_lagged') is True
                                and proof['epoch'] == epoch, 'Lagged history lacks CPU-verified durable epoch evidence')
                model.load_state_dict(saved['state_dict'], strict=True)
                del saved
            else:
                epoch = 1
                state_history = None
            F1.set_train_phase(model, epoch)
            baseline_state = F1.state_digest(model.state_dict())
            baseline_rng = F1.rng_state(model.device)
            baseline_modes = [(name, module.training) for name, module in model.named_modules()]
            original_forward = model.forward
            original_encode = model.adapter.encode_windows
            original_backward, original_step = torch.Tensor.backward, torch.optim.AdamW.step
            captured = []

            def forward(value):
                trace['forward_started'] += 1
                result = original_forward(value)
                trace['forward_returned'] += 1
                captured.append(result.detach().cpu().clone())
                return result

            def encode(*a, **kw):
                trace['encoder_forward_started'] += 1; trace['encoder_windows_started'] += len(a[1])
                result = original_encode(*a, **kw)
                trace['encoder_forward_returned'] += 1; trace['encoder_windows_returned'] += len(a[1])
                return result

            def forbidden_backward(*a, **kw):
                trace['forbidden_backward_attempts'] += 1
                raise RuntimeError('Qualification cannot call backward')

            def forbidden_step(*a, **kw):
                trace['forbidden_optimizer_attempts'] += 1
                raise RuntimeError('Qualification cannot update optimizer')

            model.forward = forward; model.adapter.encode_windows = encode
            torch.Tensor.backward = forbidden_backward; torch.optim.AdamW.step = forbidden_step
            outputs = {}
            try:
                for chunk in B.CHUNKS:
                    runs = []
                    trace.update(current_checkpoint=checkpoint, current_chunk=chunk, phase='diagnostics', phase_finished=False)
                    atomic_json(attempt_path, trace)
                    for repetition in range(args.repeats + 1):
                        captured.clear()
                        F1.restore_rng(baseline_rng)
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start_event.record(); start = time.perf_counter()
                        with B.diagnostic_chunk(model, raw, probe_split, registry['labels'], chunk):
                            losses = F1.diagnostics(model, raw, probe_split, registry['labels'])
                        end_event.record(); torch.cuda.synchronize()
                        wall = time.perf_counter() - start
                        logits = torch.cat(captured)
                        state_ok = F1.state_digest(model.state_dict()) == baseline_state
                        rng_ok = F1.rng_equal(baseline_rng, F1.rng_state(model.device))
                        modes_ok = baseline_modes == [(n, m.training) for n, m in model.named_modules()]
                        require(state_ok and rng_ok and modes_ok and model.window_chunk == 8, 'Diagnostics changed state/RNG/modes/default chunk')
                        require(len(logits) == len(probe_split['train']) + len(probe_split['val']) and torch.isfinite(logits).all(), 'Missing/nonfinite diagnostic logits')
                        record = {'checkpoint': checkpoint, 'chunk': chunk, 'repetition': repetition, 'warmup': repetition == 0,
                                  'synchronized_wall_seconds': wall, 'gpu_event_interval_ms_not_busy_time': start_event.elapsed_time(end_event),
                                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'state_unchanged': state_ok, 'rng_restored': rng_ok, 'modes_restored': modes_ok}
                        measurements.append(record); runs.append((logits, losses))
                        if repetition:
                            times[chunk].append(wall)
                        trace.update(measurements=measurements, phase_finished=True)
                        atomic_json(attempt_path, trace)
                    require(all(torch.equal(runs[0][0], item[0]) and runs[0][1] == item[1] for item in runs[1:]), 'Same chunk repeated diagnostics not bitwise stable')
                    outputs[chunk] = runs[-1]
                baseline_logits, baseline_losses = outputs[8]
                prefix = [r['val_eval_ce'] for r in state_history[:epoch - 1]] if state_history else []
                baseline_policy = controller(P, prefix + [baseline_losses['val_eval_ce']])
                for chunk in (16, 32, 60):
                    logits, losses = outputs[chunk]
                    bitwise = torch.equal(baseline_logits, logits)
                    logit_error = float((baseline_logits - logits).abs().max())
                    ce_error = max(abs(losses[k] - baseline_losses[k]) for k in baseline_losses)
                    policy_equal = controller(P, prefix + [losses['val_eval_ce']]) == baseline_policy
                    exact = bitwise and ce_error == 0 and policy_equal
                    comparisons[chunk].append({'checkpoint': checkpoint, 'epoch': epoch, 'passed': exact,
                        'bitwise_logits': bitwise, 'logits_max_abs': logit_error, 'ce_max_abs': ce_error,
                        'strict_tolerance_pass_analysis_only': bool(torch.allclose(baseline_logits, logits, atol=1e-6, rtol=1e-6)) and ce_error <= 1e-7,
                        'policy_decisions_equal': policy_equal, 'baseline_controller': baseline_policy,
                        'controller_scope': 'actual_stable_history_prefix' if state_history else 'train_only_probe_scalar_no_formal_validation_policy_claim',
                        'candidate_controller': controller(P, prefix + [losses['val_eval_ce']]),
                        'state_unchanged': True, 'rng_restored': True, 'modes_restored': True})
            finally:
                model.forward = original_forward; model.adapter.encode_windows = original_encode
                torch.Tensor.backward = original_backward; torch.optim.AdamW.step = original_step
                model.window_chunk = 8
            checked(checkpoint, binary=True)
        candidates = [{'chunk': chunk, 'qualified': all(c['passed'] for c in comparisons[chunk]),
                       'equivalence_level': 'bitwise' if all(c['passed'] for c in comparisons[chunk]) else 'not_qualified',
                       'comparisons': comparisons[chunk], 'median_wall_seconds': statistics.median(times[chunk])}
                      for chunk in (16, 32, 60)]
        eligible = [c for c in candidates if c['qualified'] and c['median_wall_seconds'] < statistics.median(times[8])]
        chosen = min(eligible, key=lambda c: (c['median_wall_seconds'], c['chunk'])) if eligible else None
        for binding in checkpoints:
            checked(binding, binary=True)
        if history:
            checked(proof['train_history'])
        trace.update(status='complete', phase='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                     seconds=time.perf_counter() - trace.pop('_timer_start'), measurements=measurements)
        atomic_json(attempt_path, trace)
        result = {'schema': 'diagnostic_batching_qualification_v1', 'passed': chosen is not None,
                  'reason': 'bitwise candidate with measured wall improvement' if chosen else 'No faster candidate satisfies bitwise logits and CE; retain diagnostic chunk8',
                  'run_requested': True, 'qualification_mode': 'bitwise_only', 'created_at': datetime.now(timezone.utc).isoformat(),
                  'condition': fit['condition'], 'seed': fit['seed'], 'fold': fit['fold'], 'registration': descriptor(REGISTRY),
                  'safe_stop_proof': descriptor(args.stop_proof), 'checkpoints': checkpoints, 'train_history': proof.get('train_history'),
                  'weight_mode': weight_mode, 'role_ids': split, 'probe_split': probe_split, 'condition_source_lock': condition_lock,
                  'raw_sources': raw_sources, 'actual_runtime': runtime, **source_bindings,
                  'baseline_chunk': 8, 'candidates': candidates, 'selected_chunk': chosen['chunk'] if chosen else None,
                  'baseline_median_wall_seconds': statistics.median(times[8]), 'measurements': measurements,
                  'measured_repeats': args.repeats,
                  'thresholds_declared_before_execution': THRESHOLDS, 'selection_rule': 'fastest_qualified_median_wall_no_validation_performance_ranking',
                  'optimizer_updates': 0, 'backward_calls': 0, 'test_forward_count': 0,
                  'attempt_trace': descriptor(attempt_path), 'scope_limit': ('Measured stable weight states and original fold train/val only; no training, test, or universal mathematical equivalence claim.'
                      if weight_mode == 'stable_formal_checkpoint' else 'Fresh official initialization and originalfold1 training people only; internal train-only probe roles are not formal validation; no future policy or universal equivalence claim.')}
        path = REPORT / ('diagnostic_batching_qualification_' + str(time.time_ns()) + '.json')
        atomic_json(path, result, exclusive=True)
        return {'artifact': descriptor(path), 'passed': result['passed'], 'selected_chunk': result['selected_chunk']}
    except BaseException:
        if trace.get('gpu_stage_entered'):
            try:
                trace['peak_allocated_bytes_observed_at_failure'] = torch.cuda.max_memory_allocated()
            except BaseException as resource_error:
                trace['failure_resource_read_error'] = str(resource_error)
        raise
    finally:
        if model is not None:
            del model
        if raw is not None:
            del raw
        if OWNER.exists() and json_read(OWNER).get('id') == owner['id']:
            OWNER.unlink()


def execute_explicit(args):
    require(args.stop_proof is not None, 'Explicit execution requires exact safe-stop proof')
    require(2 <= args.repeats <= 5, 'Use2..5 measured repeats after one warmup')
    try:
        condition = json_read(args.stop_proof)['condition']
        require(type(condition) is str and '/' not in condition and '\\' not in condition and '..' not in condition, 'Invalid condition path')
    except (KeyError, ValueError, OSError):
        condition = '_unclassified'
    attempt_path = ROOT / condition / 'qualification_attempts' / ('attempt_' + str(time.time_ns())) / 'attempt.json'
    trace = {'schema': 'diagnostic_batching_qualification_attempt_v1', 'kind': 'engineering_qualification', 'status': 'running',
             'started_at': datetime.now(timezone.utc).isoformat(), '_timer_start': time.perf_counter(),
             'qualifier_source': descriptor(Path(__file__)), 'thresholds': THRESHOLDS,
             'forward_started': 0, 'forward_returned': 0, 'encoder_forward_started': 0, 'encoder_forward_returned': 0,
             'encoder_windows_started': 0, 'encoder_windows_returned': 0, 'backward_calls': 0, 'optimizer_updates': 0,
             'forbidden_backward_attempts': 0, 'forbidden_optimizer_attempts': 0, 'test_forward_count': 0,
             'counter_durability': 'Phase/repetition snapshots; hard-crash unflushed tail is unknown, never zero-filled',
             'unflushed_crash_tail_exact': None, 'gpu_stage_entered': False,
             'peak_allocated_bytes_observed_at_failure': None}
    atomic_json(attempt_path, trace, exclusive=True)
    try:
        result = run(args, attempt_path, trace)
        return result
    except BaseException as error:
        trace.update(status='failed', error_type=type(error).__name__, error=str(error), finished_at=datetime.now(timezone.utc).isoformat())
        timer = trace.pop('_timer_start', None)
        if timer is not None:
            trace['seconds'] = time.perf_counter() - timer
        atomic_json(attempt_path, trace)
        print(json.dumps({'passed': False, 'attempt': descriptor(attempt_path), 'error': str(error)}))
        raise


def run_qualification(condition, seed, fold, checkpoint=None, *, safe_stop_proof, repeats=2):
    """Q4 Python API; explicit call has the same safety gates as CLI --run.

    safe_stop_proof is a path+SHA descriptor. No checkpoint means fresh official
    initialization only when the proof itself has no checkpoint descriptors.
    """
    path = checked(safe_stop_proof)
    proof = json_read(path)
    require((proof['condition'], proof['seed'], proof['fold']) == (condition, seed, fold), 'API identity differs from proof')
    declared = proof.get('checkpoints', [proof['checkpoint']] if proof.get('checkpoint') else [])
    if checkpoint is not None:
        require(declared == ([checkpoint] if isinstance(checkpoint, dict) else checkpoint), 'API checkpoint differs from proof')
    return execute_explicit(types.SimpleNamespace(stop_proof=path, repeats=repeats))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stop-proof', type=Path)
    parser.add_argument('--condition')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args(argv)
    require(2 <= args.repeats <= 5, 'Use2..5 measured repeats after one warmup')
    if not args.run:
        result = {'schema': 'diagnostic_batching_qualification_pending_v1', 'passed': False, 'status': 'pending_explicit_safe_gpu_run',
                  'qualifier_source': descriptor(Path(__file__)), 'thresholds': THRESHOLDS, 'torch_imports': 0,
                  'checkpoint_reads': 0, 'raw_reads': 0, 'GPU': 0, 'training_actions': 0}
        path = REPORT / ('diagnostic_batching_qualification_pending_' + str(time.time_ns()) + '.json')
        atomic_json(path, result, exclusive=True)
        print(json.dumps({'artifact': descriptor(path), 'status': result['status']})); return 0
    require(args.stop_proof is not None, '--run requires exact root safe-stop proof')
    proof = json_read(args.stop_proof)
    for key in ('condition', 'seed', 'fold'):
        if getattr(args, key) is not None:
            require(getattr(args, key) == proof[key], 'CLI identity differs from safe proof')
    result = execute_explicit(args)
    print(json.dumps(result)); return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
