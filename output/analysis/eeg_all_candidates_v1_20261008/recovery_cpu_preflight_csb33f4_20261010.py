"""Actual CPU-only two-checkpoint recovery audit for the Oct9 paused Q4 fit.

Read original initial/latest once each; no model, raw data, predictions, test
gate, GPU query or process operation. Old evidence and source bytes preserved.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R
from tridep_ccml.eeg_all_candidates import execution_acceleration_v1 as A

HELPER = SOURCE.with_name('safe_stop_acceleration_v1.py')
HELPER_SHA = '6c0c6d084b5463ed3cfa2326cc17bb105fa3a9369fd75527d4912d198489071c'
PAUSE = S.OUT / 'pause_events/q4_user_pause_1791558499460173100'
FOLDER = S.OUT / 'runs/CSBrain-lastblock-finetune/seed_33/fold_4'
LATEST_SHA = '8844db8f6443ecb7777730fd06aa33655fa6be8b1aa0fee46b4b447983b43782'
TRACE_SHA = 'e190ee531b30275322398edeeab6a93c4db4a28dc239510e28a59ea4939fd52f'


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    began = time.perf_counter()
    value = {'schema': 'actual_cpu_recovery_preflight_csb33f4_v1', 'source': S.desc(SOURCE), 'passed': False,
             'scope': 'Resume safety for the partial CSBrain seed33/fold4 only, not completed fit/full-plan acceptance.',
             'condition': 'CSBrain-lastblock-finetune', 'seed': 33, 'fold': 4,
             'checkpoint_loads': 0, 'model_constructions': 0, 'GPU_operations': 0, 'GPU_queries': 0,
             'raw_arrays_loaded': 0, 'test_probability_reads': 0, 'test_gate_calls': 0,
             'optimizer_updates': 0, 'backward_calls': 0, 'process_operations': 0,
             'original_artifacts_modified': False, 'process_exit_verification': 'Handled separately by root; not established by this CPU audit.'}
    path = S.REPORT / ('recovery_cpu_preflight_csb33f4_' + str(time.time_ns()) + '.json')
    try:
        require(S.desc(HELPER)['sha256'] == HELPER_SHA, 'Sealed CPU validator changed')
        spec = importlib.util.spec_from_file_location('sealed_cpu_checkpoint_validator', HELPER)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        pause = S.read(PAUSE / 'pause_complete.json')
        require((pause['condition'], pause['seed'], pause['fold']) == ('CSBrain-lastblock-finetune', 33, 4), 'Wrong paused fit')
        snapshot = next(item for item in pause['retained'] if Path(item['path']).name == 'latest.pt')
        require(snapshot['sha256'] == LATEST_SHA and S.desc(S.checked(snapshot)) == snapshot
                and S.desc(FOLDER / 'latest.pt')['sha256'] == LATEST_SHA, 'Original and paused latest checkpoint differ')
        pair, = pause['attempt_snapshots']
        require(pair['original']['sha256'] == pair['snapshot']['sha256'] == TRACE_SHA, 'Wrong buffered pause trace')
        trace_path = S.checked(pair['original']); S.checked(pair['snapshot'])
        trace = S.read(trace_path)
        registry = R.load_registration()
        fit, = [row for row in registry['fits'] if row['fit_id'] == 'CSBrain-lastblock-finetune/seed_33/fold_4']
        observer_check = A.verify_observer_record(trace, pair['original'], fit)
        source_sets = []
        for queue, count in [('continuation_queue_v4', 59), ('continuation_queue_v5', 76)]:
            registration_path = S.OUT / queue / 'registration.json'
            registration = S.read(registration_path)
            require(len(registration['sources']) == count, 'Wrong sealed source coverage')
            for binding in registration['sources']:
                S.checked(binding)
            source_sets.append({'queue': queue, 'registration': S.desc(registration_path), 'sources_verified': count, 'differences': []})

        import torch
        require(not torch.cuda.is_initialized(), 'CPU audit unexpectedly has initialized CUDA')
        old_load, old_lazy = torch.load, torch.cuda._lazy_init
        allowed = {FOLDER / 'initial.pt', FOLDER / 'latest.pt'}
        loaded = {}
        def cpu_load(file, *args, **kwargs):
            target = Path(file).resolve()
            require(target in allowed and target not in loaded and kwargs.get('map_location') == 'cpu', 'Only two distinct CPU checkpoints allowed')
            payload = old_load(file, *args, **kwargs)
            loaded[target] = payload; value['checkpoint_loads'] += 1
            return payload
        def forbidden_cuda(*args, **kwargs):
            raise AssertionError('GPU initialization is prohibited in CPU recovery audit')
        try:
            torch.load, torch.cuda._lazy_init = cpu_load, forbidden_cuda
            checked = helper.validate_checkpoint(FOLDER)
            initial, latest = loaded[FOLDER / 'initial.pt'], loaded[FOLDER / 'latest.pt']
            require(value['checkpoint_loads'] == checked['cpu_checkpoint_loads'] == 2, 'Expected exactly initial/latest CPU loads')
            require((checked['epoch'], checked['optimizer_updates'], checked['backward_calls']) == (8, 32, 216), 'Unexpected durable recovery boundary')
            current_history = S.read(FOLDER / 'train_history.json')
            require(current_history == latest['history'] == S.read(PAUSE / 'train_history.json'), 'Current/snapshot/embedded durable histories differ')
            decision = helper.P.training_decision('finetune', [row['val_eval_ce'] for row in latest['history']], extended=False)
            require(1 <= decision['best_epoch'] <= 8 and decision['stop'] is False, 'Warmup partial checkpoint has unexpected policy decision')
            contract = S.read(S.checked(R._condition_lock(fit['condition'])))['contract']
            encoder_names = [name for name in contract['trainable_parameter_names'] if name.startswith('encoder.')]
            head_names = [name for name in contract['trainable_parameter_names'] if name.startswith('head.')]
            require(len(encoder_names) == 20 and len(head_names) == 6, 'Original CSBrain groups differ')
            require(all(torch.equal(initial['state_dict'][name], latest['state_dict'][name])
                        and torch.equal(initial['state_dict'][name], latest['best_selected_state_dict'][name])
                        for name in encoder_names), 'Warmup current/best encoder state differs from initialization')
            best_head_changes = sum(not torch.equal(initial['state_dict'][name], latest['best_selected_state_dict'][name]) for name in head_names)
            latest_head_changes = sum(not torch.equal(initial['state_dict'][name], latest['state_dict'][name]) for name in head_names)
            require(not torch.cuda.is_initialized(), 'GPU initialization occurred')
            value.update(checkpoint_validation=checked, best_epoch_from_durable_validation_history=decision['best_epoch'],
                         warmup_state={'epoch': 8, 'selected_phase': 'head_warmup', 'encoder_current_and_best_equal_initial': True,
                                       'head_optimizer_steps': 32, 'backbone_optimizer_states': 0,
                                       'selected_head_parameters_changed': best_head_changes,
                                       'latest_head_parameters_changed': latest_head_changes},
                         durable_history_matches_current_and_pause_snapshot=True)
        finally:
            torch.load, torch.cuda._lazy_init = old_load, old_lazy

        progress = S.read(FOLDER / 'progress.json')
        require(progress == S.read(PAUSE / 'progress.json') == pause['active_progress'], 'Paused progress changed')
        require(trace['latest_durable_checkpoint']['epoch'] == 8 and trace['latest_durable_checkpoint']['optimizer_updates'] == 32
                and trace['latest_durable_checkpoint']['backward_calls'] == 216, 'F3 durable observation and checkpoint differ')
        require(trace['optimizer_step_returned'] == progress['optimizer_updates_completed'] == 36
                and trace['backward_returned'] == progress['backward_calls_completed'] == 243, 'Returned-call/progress boundary differs')
        require(trace['last_flush_reason'] == 'diagnostics_before' and trace['graceful_terminal_flush'] is False
                and trace['unflushed_tail_unknown'] is True and trace['unflushed_tail_counters'] is None, 'Lost-runtime uncertainty must remain explicit')
        value.update(passed=True, helper_source=S.desc(HELPER), original_registration=R.registration_descriptor(),
                     pause_evidence=S.desc(PAUSE / 'pause_complete.json'), latest_pause_snapshot=snapshot,
                     latest_original_matches_pause_sha=True, buffered_trace_pair=pair, observer_validation=observer_check,
                     sealed_source_verification=source_sets,
                     recovery_boundary={'durable_epoch': 8, 'durable_optimizer_updates': 32, 'durable_backward_calls': 216,
                         'resume_next_epoch': 9, 'known_returned_optimizer_updates_beyond_checkpoint_lower_bound': 4,
                         'known_returned_backward_calls_beyond_checkpoint_lower_bound': 27,
                         'persisted_forward_returns': trace['forward_returned'], 'persisted_encoder_forward_returns': trace['encoder_forward_returned'],
                         'persisted_encoder_window_returns': trace['encoder_windows_returned'], 'last_flush_reason': trace['last_flush_reason'],
                         'unflushed_runtime_tail_exact': None, 'unflushed_runtime_tail_unknown': True,
                         'historical_physical_call_totals_exact': False, 'historical_attempt_wall_seconds_exact': None,
                         'pause_or_overnight_wall_added_as_compute': False,
                         'known_returned_extra_work_is_not_added_to_durable_completed_fit_counts': True},
                     training_complete_promoted=False, full_plan_completed_claimed=False)
        # Recheck all paused/original bindings after validation without reloading.
        S.checked(snapshot); S.checked(pair['original']); S.checked(pair['snapshot'])
        for binding in checked['original_checkpoint_bindings']: S.checked(binding)
        for source_set in source_sets:
            for binding in S.read(S.checked(source_set['registration']))['sources']: S.checked(binding)
    except BaseException as error:
        value.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        value['seconds'] = time.perf_counter() - began
        S.save(path, value, immutable=True)
        print(json.dumps({'artifact': S.desc(path), 'passed': value['passed'], 'checkpoint_loads': value['checkpoint_loads']}), flush=True)


if __name__ == '__main__':
    main()
