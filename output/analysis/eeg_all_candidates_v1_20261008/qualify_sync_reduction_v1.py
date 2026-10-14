"""Opt-in zero-update real diagnostics synchronization comparison, chunk8 only.

Default writes pending. Explicit execution requires a source-bound natural
full-CSBrain15 boundary with the dispatcher held and its GPU children exited.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT = SOURCE.parent
BASE_SOURCE = SOURCE.with_name('qualify_acceleration_20261009.py')
BASE_SHA = '878fda0f97d0a6ec801501baf1bad084d77d2d319897609a2a39600e01568a5e'
if hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError('Reviewed safe IO/identity helper source changed')
spec = importlib.util.spec_from_file_location('_sync_qualification_safety_helpers', BASE_SOURCE)
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)
require, desc, checked, read, save = Q.require, Q.descriptor, Q.checked, Q.json_read, Q.atomic_json
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import sync_qualification_gate_v1 as G
OWNER = G.OWNER


def exact_ce(left, right):
    return (set(left) == set(right) and all(type(left[key]) is float and type(right[key]) is float
            and math.isfinite(left[key]) and math.isfinite(right[key]) and left[key].hex() == right[key].hex() for key in left))


def exact_logits(torch, left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def preflight(proof_binding, builder_binding):
    result = G.preflight(proof_binding, builder_binding)
    require(not (OUT / 'execution_acceleration_v1/qualification_owner.json').exists(), 'Other qualification owner exists')
    for name, digest in Q.DELEGATE_HASHES.items():
        require(desc(WORK / 'tridep_ccml/eeg_all_candidates' / name)['sha256'] == digest, 'Sealed F1/F2 source changed')
    return result


def qualify(proof_binding, builder_binding, *, repeats=2):
    require(type(repeats) is int and 2 <= repeats <= 5, 'Use2..5 measured repetitions after warmup')
    token = str(time.time_ns())
    attempt_path = OUT / 'supervised_engineering/CSBrain-lastblock-finetune' / ('diagnostic_sync_attempt_' + token + '.json')
    trace = {'schema': 'diagnostic_sync_qualification_attempt_v1', 'subtype': 'diagnostic_sync_zero_optimizer_qualification',
             'kind': 'engineering', 'condition': 'CSBrain-lastblock-finetune', 'seed': 22, 'fold': 5,
             'status': 'running', 'source': desc(SOURCE),
             'safe_stop_proof': proof_binding, 'proof_builder': builder_binding, 'gpu_stage_entered': False,
             'started_at': Q.datetime.now(Q.timezone.utc).isoformat(), 'seconds': None,
             'forward_started': 0, 'forward_returned': 0, 'encoder_forward_started': 0, 'encoder_forward_returned': 0,
             'encoder_windows_started': 0, 'encoder_windows_returned': 0, 'formal_fits': 0,
             'optimizer_updates': 0, 'backward_calls': 0, 'forbidden_optimizer_attempts': 0, 'forbidden_backward_attempts': 0,
             'test_forward_count': 0, 'unflushed_hard_crash_tail_exact': None,
             'peak_allocated_bytes': None, 'model_constructions_started': 0, 'model_constructions_returned': 0,
             'latest_checkpoint_load_started': 0, 'latest_checkpoint_load_returned': 0,
             'counter_durability': 'Persist before and after each repetition; caught failures retain in-memory calls; hard-crash tail remains unknown.',
             'measurement_scope': 'Engineering qualification only; total attempt wall counted once, nested per-repetition times not added.'}
    save(attempt_path, trace, exclusive=True)
    began = time.perf_counter(); owner = None; torch = model = None; old_backward = old_step = None
    original_forward = original_encode = None
    measurements, comparisons = [], []
    try:
        proof, registry, fit, history, idle = preflight(proof_binding, builder_binding)
        trace.update(condition=fit['condition'], seed=fit['seed'], fold=fit['fold'], idle_gpu_check=idle)
        save(attempt_path, trace)
        owner = {'id': uuid.uuid4().hex, 'pid': os.getpid(), 'attempt': attempt_path.relative_to(WORK).as_posix()}
        save(OWNER, owner, exclusive=True)
        preflight(proof_binding, builder_binding)
        checkpoint_path = checked(proof['checkpoint'], binary=True)
        sys.path.insert(0, str(WORK))
        import torch
        from tridep_ccml.eeg_all_candidates import finetune as F, finetune_v2 as F2, protocol as P, sync_reduction_v1 as Y, sync_state_digest_v1 as Z
        require(desc(Path(Z.__file__))['sha256'] == '0e82e52882326403320ab5fea06ba8550faad60e3a6e410ef94d11a3ab3db2ad', 'Reviewed packed digest prototype changed')
        require(torch.cuda.is_available(), 'Actual CUDA required')
        trace['gpu_stage_entered'] = True
        source_bindings = {key: desc(path) for key, path in [('sync_reduction_source', Path(Y.__file__)),
                           ('original_diagnostics_source', Path(F.__file__)), ('delegated_f2_source', Path(F2.__file__)),
                           ('qualification_source', SOURCE), ('safety_helpers_source', BASE_SOURCE),
                           ('shared_boundary_gate_source', Path(G.__file__)), ('packed_digest_source', Path(Z.__file__))]}
        trace['source_bindings'] = source_bindings; save(attempt_path, trace)
        split = registry['folds']['fold_' + str(fit['fold'])]
        raw, raw_sources = F._raw_sources(split)
        F.C.runtime(fit['seed'])
        trace['model_constructions_started'] += 1; save(attempt_path, trace)
        model, adapter_audit = F2.build_model(fit['condition'], seed=fit['seed'], device='cuda', window_chunk=8)
        trace['model_constructions_returned'] += 1; save(attempt_path, trace)
        lock = F2._check_v2_lock(fit['condition'])
        trace['latest_checkpoint_load_started'] += 1; save(attempt_path, trace)
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        trace['latest_checkpoint_load_returned'] += 1; save(attempt_path, trace)
        require((saved['condition'], saved['seed'], saved['fold']) == (fit['condition'], fit['seed'], fit['fold'])
                and saved['registry'] == proof['original_training_registration'] and saved['condition_source_lock'] == lock
                and saved['raw_sources'] == raw_sources, 'Stable checkpoint identity/source differs')
        epoch = saved['epoch']; authoritative_history = saved['history']
        require(type(epoch) is int and len(authoritative_history) == epoch and history == authoritative_history
                and all(row['epoch'] == i for i, row in enumerate(authoritative_history, 1)), 'Stable history prefix differs')
        model.load_state_dict(saved['state_dict'], strict=True)
        F.set_train_phase(model, epoch); F.restore_rng(saved['rng']); del saved
        state = F.state_digest(model.state_dict()); rng = F.rng_state(model.device)
        modes = [(name, module.training) for name, module in model.named_modules()]
        captured, sync_observations, digest_observations = [], [], []
        original_forward, original_encode = model.forward, model.adapter.encode_windows
        def observe_forward(call):
            def invoke(raw_value):
                trace['forward_started'] += 1
                result = call(raw_value)
                trace['forward_returned'] += 1; captured.append(result.detach())
                return result
            return invoke
        def encode(*args, **kwargs):
            trace['encoder_forward_started'] += 1
            trace['encoder_windows_started'] += len(args[1])
            result = original_encode(*args, **kwargs)
            trace['encoder_forward_returned'] += 1
            trace['encoder_windows_returned'] += len(args[1])
            return result
        model.forward = observe_forward(original_forward); model.adapter.encode_windows = encode
        old_backward, old_step = torch.Tensor.backward, torch.optim.AdamW.step
        def forbid_backward(*args, **kwargs):
            trace['forbidden_backward_attempts'] += 1; raise RuntimeError('No backward permitted during diagnostic qualification')
        def forbid_step(*args, **kwargs):
            trace['forbidden_optimizer_attempts'] += 1; raise RuntimeError('No optimizer update permitted during diagnostic qualification')
        torch.Tensor.backward, torch.optim.AdamW.step = forbid_backward, forbid_step
        packed_digest = Z.make_state_digest(F.state_digest, torch_module=torch, observer=digest_observations.append)
        variants = [('original', F.diagnostics),
                    ('combined_sync_coalesced_chunk8', Y.make_diagnostics(F.diagnostics, forward_observer=observe_forward,
                        observer=sync_observations.append, state_digest=packed_digest,
                        coalesce_input_transfer=True, coalesce_targets=True))]
        outputs, timings = {}, {}
        try:
            for name, diagnostics in variants:
                outputs[name], timings[name] = [], []
                for rep in range(repeats + 1):
                    captured.clear(); sync_observations.clear(); digest_observations.clear(); F.restore_rng(rng)
                    trace['current_digest_observations'] = digest_observations
                    trace.update(current_variant=name, repetition=rep, phase='diagnostics_started'); save(attempt_path, trace)
                    torch.cuda.synchronize()
                    trace['peak_allocated_bytes'] = max(trace['peak_allocated_bytes'] or 0, torch.cuda.max_memory_allocated())
                    torch.cuda.reset_peak_memory_stats()
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start_event.record(); start = time.perf_counter()
                    losses = diagnostics(model, raw, split, registry['labels'])
                    end_event.record(); torch.cuda.synchronize(); wall = time.perf_counter() - start
                    logits = torch.cat(captured).cpu().clone()
                    unchanged = state == F.state_digest(model.state_dict())
                    rng_restored = F.rng_equal(rng, F.rng_state(model.device))
                    modes_restored = modes == [(n, m.training) for n, m in model.named_modules()]
                    require(unchanged and rng_restored and modes_restored and model.window_chunk == 8,
                            'Qualification changed state/RNG/modes/chunk')
                    require(len(logits) == len(split['train']) + len(split['val']) and torch.isfinite(logits).all(), 'Incomplete/nonfinite captured logits')
                    if name != 'original':
                        require(len(sync_observations) == 1 and sync_observations[0]['all_finite_checks_passed_before_return'] is True
                                and sync_observations[0]['scalar_device_to_host_transfers_returned'] == 2, 'Actual finite-check transfer evidence missing')
                    record = {'variant': name, 'repetition': rep, 'warmup': rep == 0, 'synchronized_wall_seconds': wall,
                              'gpu_event_interval_ms_not_busy_time': start_event.elapsed_time(end_event),
                              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'state_unchanged': unchanged,
                              'rng_restored': rng_restored, 'modes_restored': modes_restored,
                              'sync_observations': list(sync_observations), 'digest_observations': list(digest_observations)}
                    measurements.append(record); outputs[name].append((logits, losses))
                    trace['peak_allocated_bytes'] = max(trace['peak_allocated_bytes'], record['peak_allocated_bytes'])
                    if rep: timings[name].append(wall)
                    trace.update(phase='diagnostics_returned', measurements=measurements); save(attempt_path, trace)
                require(all(exact_logits(torch, outputs[name][0][0], logits) and exact_ce(outputs[name][0][1], losses) for logits, losses in outputs[name][1:]),
                        'Repeated same-variant output is not bitwise stable')
            baseline_logits, baseline_losses = outputs['original'][-1]
            prefix = [row['val_eval_ce'] for row in authoritative_history[:-1]]
            baseline_policy = Q.controller(P, prefix + [baseline_losses['val_eval_ce']])
            for name, _ in variants[1:]:
                logits, losses = outputs[name][-1]
                bitwise, ce_equal = exact_logits(torch, logits, baseline_logits), exact_ce(losses, baseline_losses)
                policy = Q.controller(P, prefix + [losses['val_eval_ce']])
                comparisons.append({'variant': name, 'bitwise_logits': bitwise, 'exact_ce': ce_equal,
                                    'logits_max_abs': float((logits - baseline_logits).abs().max()),
                                    'ce_max_abs': max(abs(losses[k] - baseline_losses[k]) for k in baseline_losses),
                                    'policy_decisions_equal': policy == baseline_policy,
                                    'median_wall_seconds': statistics.median(timings[name]),
                                    'eligible': bitwise and ce_equal and policy == baseline_policy,
                                    'baseline_policy': baseline_policy, 'candidate_policy': policy})
        finally:
            model.forward = original_forward; model.adapter.encode_windows = original_encode
        checked(proof['checkpoint'], binary=True); checked(proof['train_history']); checked(proof_binding)
        baseline_wall = statistics.median(timings['original'])
        eligible = [row for row in comparisons if row['eligible'] and row['median_wall_seconds'] < baseline_wall]
        chosen = min(eligible, key=lambda row: row['median_wall_seconds']) if eligible else None
        trace.update(status='complete', seconds=time.perf_counter() - began, measurements=measurements)
        save(attempt_path, trace)
        report = {'schema': 'diagnostic_sync_qualification_v1', 'passed': chosen is not None,
                  'source': desc(SOURCE), **source_bindings, 'safe_stop_proof': proof_binding, 'proof_builder': builder_binding,
                  'registration': proof['original_training_registration'], 'q4_registration': proof['registration'],
                  'condition': fit['condition'], 'seed': fit['seed'], 'fold': fit['fold'],
                  'checkpoint': proof['checkpoint'], 'train_history': proof['train_history'], 'condition_source_lock': lock,
                  'role_ids': split, 'raw_sources': raw_sources, 'adapter_audit': adapter_audit,
                  'actual_runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu_name': torch.cuda.get_device_name()},
                  'baseline': 'original_F1_diagnostics', 'diagnostic_window_chunk': 8, 'measured_repeats': repeats,
                  'combined_components': ['CE and finite scalar observations per role', 'exact packed state digest with recorded fallback',
                                          'one full60 adapted H2D transfer per person with unchanged8/last4 encoder views',
                                          'all train+validation target labels transferred once then original per-person views'],
                  'baseline_median_wall_seconds': baseline_wall, 'comparisons': comparisons, 'measurements': measurements,
                  'selected_variant': chosen['variant'] if chosen else None,
                  'eligibility': 'bitwise logits AND exactly equal Python sequential CE AND same controller policy',
                  'selection': 'fastest eligible measured wall below baseline; no validation-performance ranking',
                  'optimizer_updates': 0, 'backward_calls': 0, 'test_forward_count': 0,
                  'attempt': desc(attempt_path), 'not_formal_training_authorization': True,
                  'scope': 'Measured stable checkpoint only; no universal future-state numerical claim; no automatic runtime installation.'}
        path = REPORT / ('diagnostic_sync_qualification_' + token + '.json'); save(path, report, exclusive=True)
        return {'artifact': desc(path), 'passed': report['passed'], 'selected_variant': report['selected_variant']}
    except BaseException as error:
        trace.update(status='failed', seconds=time.perf_counter() - began, error_type=type(error).__name__, error=str(error))
        if torch is not None and trace['gpu_stage_entered']:
            try:
                trace['peak_allocated_bytes_observed_at_failure'] = torch.cuda.max_memory_allocated()
                trace['peak_allocated_bytes'] = max(trace['peak_allocated_bytes'] or 0, trace['peak_allocated_bytes_observed_at_failure'])
            except BaseException as resource_error: trace['resource_read_error'] = str(resource_error)
        save(attempt_path, trace)
        raise
    finally:
        if model is not None and original_forward is not None: model.forward = original_forward
        if model is not None and original_encode is not None: model.adapter.encode_windows = original_encode
        if old_backward is not None: torch.Tensor.backward = old_backward
        if old_step is not None: torch.optim.AdamW.step = old_step
        if owner is not None and OWNER.exists() and read(OWNER).get('id') == owner['id']: OWNER.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--safe-proof', type=Path)
    parser.add_argument('--boundary-proof', type=Path)
    parser.add_argument('--boundary-sha')
    parser.add_argument('--proof-builder', type=Path)
    parser.add_argument('--proof-builder-sha256')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args(argv)
    if not args.run:
        path = REPORT / ('diagnostic_sync_qualification_pending_' + str(time.time_ns()) + '.json')
        value = {'schema': 'diagnostic_sync_qualification_pending_v1', 'source': desc(SOURCE),
                 'prototype_source': desc(WORK / 'tridep_ccml/eeg_all_candidates/sync_reduction_v1.py'),
                 'status': 'pending_exclusive_safe_stop_and_explicit_run', 'passed': False, 'GPU': 0,
                 'torch_imports': 0, 'checkpoint_reads': 0, 'raw_reads': 0, 'model_constructions': 0}
        save(path, value, exclusive=True); print(json.dumps({'artifact': desc(path), 'status': value['status']})); return 0
    proof_path = args.boundary_proof or args.safe_proof
    require(proof_path is not None, 'Exact boundary proof required')
    binding = desc(proof_path)
    if args.boundary_proof:
        require(args.boundary_sha == binding['sha256'], 'Boundary proof SHA differs')
    builder = read(proof_path)['source'] if not args.proof_builder else desc(args.proof_builder)
    if args.proof_builder:
        require(builder['sha256'] == args.proof_builder_sha256, 'Reviewed builder SHA differs')
    result = qualify(binding, builder, repeats=args.repeats)
    print(json.dumps(result)); return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
