"""Q5 optional batching preparation; every outcome keeps original chunk8.

External GPU contexts defer only this engineering probe. Mandatory F3 fitting
uses its existing logging-only A1 sidecar. A faster bitwise V2 measurement is
retained as evidence for future versioned integration, never passed to B1.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time

from . import storage as S, registry as R
from . import execution_acceleration_v1 as A

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'execution_acceleration_v1'
QUALIFIER = S.REPORT / 'qualify_acceleration_v2.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def load_qualifier():
    spec = importlib.util.spec_from_file_location('q5_optional_batching_v2', QUALIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registration_path(condition):
    return ROOT / condition / 'registration.json'


def queue_exclusive_proof(condition, *, linux_owner=None):
    qualifier = load_qualifier()
    context, _, _ = qualifier.queue_context(condition, linux_owner=linux_owner)
    observed = qualifier.gpu_observation()
    value = {'schema': 'acceleration_safe_stop_v2', 'mode': 'queue_exclusive_post_prepare',
             'passed': observed['idle'], 'source': S.desc(SOURCE), 'condition': condition, 'seed': 11, 'fold': 1,
             'registration': R.registration_descriptor(), 'gpu_idle': observed['idle'], 'gpu_observation': observed,
             'stopped_process_ids': [], 'checkpoints': [], 'created_at': S.now(),
             'zero_optimizer_qualification_only': True, **context}
    path = ROOT / condition / ('queue_exclusive_proof_v2_' + str(time.time_ns()) + '.json')
    S.save(path, value, immutable=True)
    return S.desc(path)


def classify_result(condition, result, report):
    require(report.get('condition') == condition and report.get('qualifier_source') == S.desc(QUALIFIER), 'Actual V2 measurement identity/source differs')
    require(report.get('expanded_batch_activation_authorized') is False and report.get('production_diagnostic_chunk') == 8, 'V2 evidence cannot activate expanded chunks')
    for key in ('optimizer_updates', 'backward_calls', 'test_forward_count'):
        require(type(report.get(key)) is int and report[key] == 0, 'Optional probe performed forbidden work')
    if result.get('status') == 'deferred':
        require(result.get('passed') is False and report.get('schema') == 'diagnostic_batching_qualification_attempt_v2'
                and report.get('status') == 'deferred' and report.get('gpu_stage_entered') is False
                and report.get('qualification_performed') is False, 'Invalid deferral outcome')
        for key in ('model_constructions', 'raw_reads', 'checkpoint_reads', 'torch_imports', 'forward_started',
                    'forward_returned', 'encoder_forward_started', 'encoder_forward_returned', 'encoder_windows_started',
                    'encoder_windows_returned', 'forbidden_backward_attempts', 'forbidden_optimizer_attempts'):
            require(type(report.get(key)) is int and report[key] == 0, 'Deferred qualification must perform zero model/data work')
        observed = report.get('external_gpu_observation', {})
        require(observed.get('idle') is False and type(observed.get('contexts')) is list and observed['contexts'], 'Actual external GPU observation missing')
        return 'retained_original8_optional_probe_deferred_external_gpu'
    if result.get('status') == 'failed_candidate_oom':
        require(result.get('passed') is False and report.get('schema') == 'diagnostic_batching_qualification_attempt_v2'
                and report.get('status') == 'failed' and report.get('gpu_stage_entered') is True
                and 'cuda' in report.get('error', '').lower() and 'out of memory' in report.get('error', '').lower(), 'Invalid optional CUDA OOM fallback')
        for key in ('forbidden_backward_attempts', 'forbidden_optimizer_attempts'):
            require(type(report.get(key)) is int and report[key] == 0, 'Forbidden update attempt')
        return 'retained_original8_candidate_oom'
    require(result.get('status') == 'complete' and report.get('schema') == 'diagnostic_batching_qualification_v2'
            and type(report.get('passed')) is bool and result.get('passed') is report['passed'], 'Actual completed measurement required')
    return ('expanded_batch_activation_pending_versioned_contract' if report['passed']
            else 'retained_original8_no_bitwise_speedup')


def create_registration(condition, result):
    path = registration_path(condition)
    if path.exists():
        binding = S.desc(path)
        A.load_execution_acceleration(binding, condition=condition, seed=11, fold=1)
        return binding
    report = S.read(S.checked(result['artifact']))
    status = classify_result(condition, result, report)
    entry = {'condition_source_lock': R._condition_lock(condition), 'diagnostic_window_chunk': 8,
             'qualification': None, 'tested_qualification': result['artifact'], 'batching_status': status,
             'expanded_batch_activation_authorized': False}
    sources = {name: S.desc(SOURCE.parent / (name + '.py')) for name in
               ('finetune_v3', 'finetune_v2', 'finetune', 'validation_batching_v1', 'execution_acceleration_v1')}
    S.save(path, {'schema': 'eeg_finetune_execution_acceleration_v1', 'version': 1, 'status': 'qualified',
                 'status_scope': 'existing buffered F3 observer with unchanged diagnostic chunk8 only',
                 'registry': R.registration_descriptor(), 'registered_at': S.now(),
                 'previous_execution_registration': S.desc(S.OUT / 'continuation_queue_v3/registration.json'),
                 'single_gpu': True, 'training_math_unchanged': True, 'training_micro_subjects': 1,
                 'training_window_chunk': 8, 'sources': sources, 'conditions': {condition: entry},
                 'preparation_source': S.desc(SOURCE), 'qualifier_source': S.desc(QUALIFIER),
                 'actual_preparation_queue': 'continuation_queue_v5',
                 'preparation_queue_registration': S.desc(S.OUT / 'continuation_queue_v5/registration.json'),
                 'logging_status': 'buffered_call_observer_qualified',
                 'optional_probe_outcome': result['status'], 'expanded_batch_activation_authorized': False,
                 'optional_measurement_not_counted_as_formal_fit': True}, immutable=True)
    binding = S.desc(path)
    A.load_execution_acceleration(binding, condition=condition, seed=11, fold=1)
    return binding


def run(condition, *, stop_proof=None, linux_owner=None):
    path = registration_path(condition)
    if path.exists():
        value = S.desc(path)
        A.load_execution_acceleration(value, condition=condition, seed=11, fold=1)
        return value
    R._condition_lock(condition)
    qualifier = load_qualifier()
    proof = stop_proof or queue_exclusive_proof(condition, linux_owner=linux_owner)
    attempts = ROOT / condition / 'qualification_attempts_v2'
    previous = set(attempts.glob('*/attempt.json'))
    try:
        result = qualifier.run_qualification(condition, 11, 1, safe_stop_proof=proof)
    except RuntimeError as error:
        if 'cuda' not in str(error).lower() or 'out of memory' not in str(error).lower():
            raise
        created = set(attempts.glob('*/attempt.json')) - previous
        require(len(created) == 1, 'Cannot identify actual failed optional probe')
        result = {'artifact': S.desc(created.pop()), 'status': 'failed_candidate_oom', 'passed': False, 'selected_chunk': None}
    return create_registration(condition, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True)
    args = parser.parse_args()
    print(run(args.condition), flush=True)
