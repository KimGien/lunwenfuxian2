"""Source-registered buffered observation of the unchanged F2/F1 training.

No tensor library is imported until an authorized fit begins.  The original F2
condition contract remains the mathematics contract; execution acceleration has
its own immutable registration and provenance.  A running snapshot is a lower
bound, never an assertion that a killed process performed no further work.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import importlib
from pathlib import Path
import time
import uuid

from . import storage as S

SOURCE = Path(__file__).resolve()
COUNTER_KINDS = ('forward', 'encoder_forward', 'encoder_windows', 'encode',
                 'backward', 'optimizer_step')
COUNTERS = tuple(kind + '_' + phase for kind in COUNTER_KINDS
                 for phase in ('started', 'returned'))
OBSERVED_KINDS = ('forward', 'encoder_forward', 'backward', 'optimizer_step')
FLUSH_BOUNDARIES = ('subject_update_before', 'subject_update_after',
                    'diagnostics_before', 'diagnostics_after',
                    'latest_checkpoint_committed', 'fit_before',
                    'fit_returned', 'exception')
_ACTIVE = False


def require(value, message):
    if not value:
        raise ValueError(message)


def observer_provenance(execution):
    delegated = execution['delegated_finetune_implementation']
    return {'version': 3, 'wrapper_source': S.desc(SOURCE),
            'delegated_wrapper_source': delegated['wrapper_source'],
            'delegated_runner_source': delegated['delegated_runner_source'],
            'delegated_finetune_implementation': copy.deepcopy(delegated),
            'training_math_sampler_warmup_optimizer_stop_and_resume_unchanged': True,
            'no_validation_or_test_score_configuration_selection': True}


class BufferedAttempt:
    """Every call is observed, but only declared boundaries perform durable IO.

    All counters describe this process invocation, not the resumed trajectory.
    ``dirty_since_last_flush`` on disk describes the snapshot instant only.
    In-memory work after that instant is unknown after a hard kill.  Even a
    graceful exception only proves started/returned observations, not that an
    interrupted callable did or did not perform an optimizer update.
    """

    def __init__(self, condition, seed, fold, execution):
        require(execution['condition'] == condition and execution['seed'] == seed
                and execution['fold'] == fold, 'Acceleration/attempt identity differs')
        stem = 'f_' + str(time.time_ns()) + '_' + uuid.uuid4().hex[:8]
        self.folder = S.OUT / 'finetune_v2_attempts' / condition / stem
        self.path = self.folder / 'attempt.json'
        self.started = time.monotonic()
        self.adapters = []
        self.dirty = False
        self.terminal = False
        self._save = S.save
        self.record = {
            'condition': condition, 'kind': 'formal', 'seed': seed, 'fold': fold,
            'started_at': S.now(), 'status': 'running',
            'implementation': observer_provenance(execution),
            'execution_acceleration': copy.deepcopy(execution),
            **dict.fromkeys(COUNTERS, 0),
            'flush_sequence': 0, 'flush_counts': {},
            'counter_semantics': 'persisted_lower_bounds_until_graceful_terminal_flush',
            'dirty_flag_scope': 'snapshot_instant_only',
            'latest_durable_checkpoint': None,
            'counter_scope': 'this_process_invocation_not_resumed_trajectory',
            'encode_is_encoder_forward_alias': True,
        }
        self.flush('attempt_created', immutable=True)

    def flush(self, reason, *, immutable=False, terminal_status=None, error=None,
              latest_checkpoint=None):
        require(reason in FLUSH_BOUNDARIES + ('attempt_created',), 'Unknown flush boundary')
        require(not self.terminal, 'Attempt already terminal')
        require(terminal_status in (None, 'complete', 'failed'), 'Invalid terminal status')
        require((terminal_status == 'complete') == (reason == 'fit_returned'),
                'Complete trace requires a returned fit')
        require((terminal_status == 'failed') == (reason == 'exception'),
                'Failed trace requires an exception boundary')
        if terminal_status == 'complete':
            require(all(self.record[kind + '_started'] == self.record[kind + '_returned']
                        for kind in COUNTER_KINDS), 'Returned fit retains an unreturned observed call')
        value = copy.deepcopy(self.record)
        value['flush_sequence'] += 1
        value['flush_counts'][reason] = value['flush_counts'].get(reason, 0) + 1
        value.update(last_flush_reason=reason, last_flushed_at=S.now(),
                     durable_counter_snapshot={key: value[key] for key in COUNTERS},
                     dirty_since_last_flush=False,
                     graceful_terminal_flush=terminal_status is not None,
                     counter_snapshot_exact_for_process=terminal_status is not None,
                     unflushed_tail_unknown=terminal_status is None,
                     unflushed_tail_counters=None if terminal_status is None
                     else dict.fromkeys(COUNTERS, 0),
                     started_without_observed_return={
                         kind: value[kind + '_started'] - value[kind + '_returned']
                         for kind in OBSERVED_KINDS})
        if latest_checkpoint is not None:
            value['latest_durable_checkpoint'] = copy.deepcopy(latest_checkpoint)
        if terminal_status is not None:
            value.update(status=terminal_status, finished_at=S.now(),
                         seconds=time.monotonic() - self.started)
            if error is not None:
                value.update(failure_type=type(error).__name__, failure=str(error))
        # Commit our in-memory snapshot state only after the atomic save returns.
        # If storage itself fails, the last successful disk snapshot remains a
        # lower bound.  Never silently turn an unsuccessful flush into success.
        self._save(self.path, value, immutable=immutable)
        self.record = value
        self.dirty = False
        self.terminal = terminal_status is not None

    def observe(self, kind, function):
        require(kind in OBSERVED_KINDS, 'Unsupported observer kind')

        def wrapped(*args, **kwargs):
            require(not self.terminal, 'Observed call after terminal flush')
            self.record[kind + '_started'] += 1
            self.dirty = True
            if kind == 'encoder_forward':
                self.record['encoder_windows_started'] += len(args[1])
                self.record['encode_started'] += 1
            result = function(*args, **kwargs)
            self.record[kind + '_returned'] += 1
            if kind == 'encoder_forward':
                self.record['encoder_windows_returned'] += len(args[1])
                self.record['encode_returned'] += 1
            return result

        return wrapped

    def bind_adapter(self, adapter):
        if any(saved[0] is adapter for saved in self.adapters):
            return
        original = adapter.encode_windows
        self.adapters.append((adapter, original))
        adapter.encode_windows = self.observe('encoder_forward', original)

    def restore(self):
        for adapter, original in self.adapters:
            adapter.encode_windows = original
        self.adapters = []

    def finish(self, error=None):
        self.flush('exception' if error is not None else 'fit_returned',
                   terminal_status='failed' if error is not None else 'complete',
                   error=error)


def _boundary(original, trace, name):
    def wrapped(*args, **kwargs):
        trace.flush(name + '_before')
        result = original(*args, **kwargs)
        trace.flush(name + '_after')
        return result
    return wrapped


@contextmanager
def activate(trace, v2, batching):
    """Compose wrappers around original callables; never replace their math."""
    global _ACTIVE
    require(not _ACTIVE and not v2._ACTIVE and v2._TRACE is None,
            'F3 must own an isolated execution context')
    original_trace = v2._TRACE
    v1 = v2.V1
    original_update, original_diagnostics = v1.subject_update, v1.diagnostics
    original_save_torch = S.save_torch
    previous_save = None
    _ACTIVE = True
    v2._TRACE = trace
    try:
        with v2.activate():
            previous_save = S.save  # F2 metadata selection/provenance stays active.
            execution = trace.record['execution_acceleration']
            diagnostic = execution['diagnostics']
            selected_diagnostics = original_diagnostics
            if diagnostic['window_chunk'] != 8:
                selected_diagnostics = batching.make_diagnostics(
                    original_diagnostics, qualification=diagnostic['qualification'],
                    condition=trace.record['condition'])
            v1.subject_update = _boundary(original_update, trace, 'subject_update')
            v1.diagnostics = _boundary(selected_diagnostics, trace, 'diagnostics')
            fit_folder = v1.R.fit_directory(trace.record['condition'],
                                            trace.record['seed'], trace.record['fold']).resolve()

            def save_with_acceleration(path, value, *, immutable=False):
                target = Path(path).resolve()
                if target == fit_folder / 'metadata.json':
                    require(isinstance(value, dict), 'Metadata must be a mapping')
                    value['execution_acceleration'] = copy.deepcopy(execution)
                return previous_save(path, value, immutable=immutable)

            def save_torch_with_boundary(path, value, *, immutable=False):
                result = original_save_torch(path, value, immutable=immutable)
                target = Path(path).resolve()
                if target == fit_folder / 'latest.pt':
                    trace.flush('latest_checkpoint_committed', latest_checkpoint={
                        'path': target.relative_to(S.WORK).as_posix(),
                        'epoch': value['epoch'], 'optimizer_updates': value['optimizer_updates'],
                        'backward_calls': value['backward_calls'],
                        'checkpoint_save_returned': True,
                        'checkpoint_hash': None,
                        'hash_scope': 'not_rehashed_each_epoch; final_checkpoint_audit_binds_bytes'})
                return result

            S.save = save_with_acceleration
            S.save_torch = save_torch_with_boundary
            try:
                yield
            finally:
                v1.subject_update, v1.diagnostics = original_update, original_diagnostics
                S.save_torch = original_save_torch
                S.save = previous_save
    finally:
        v1.subject_update, v1.diagnostics = original_update, original_diagnostics
        S.save_torch = original_save_torch
        v2._TRACE = original_trace
        _ACTIVE = False


def fit(condition, seed, fold, *, acceleration_registration, resume=False,
        micro_subjects=1, window_chunk=8):
    require(type(micro_subjects) is int and micro_subjects == 1
            and type(window_chunk) is int and window_chunk == 8,
            'F3 keeps training micro_subjects=1 and window_chunk=8')
    # JSON/source-only sidecar validation precedes importing tensor code or
    # constructing an attempt/model.  No unregistered acceleration fallback.
    verifier = importlib.import_module('.execution_acceleration_v1', __package__)
    execution = verifier.load_execution_acceleration(
        acceleration_registration, condition=condition, seed=seed, fold=fold)
    v2 = importlib.import_module('.finetune_v2', __package__)
    batching = importlib.import_module('.validation_batching_v1', __package__)
    require(execution['delegated_finetune_implementation'] == v2.provenance(),
            'Registered F2 mathematics provenance differs')
    require(execution['source'] == S.desc(SOURCE), 'F3 source differs from registration')
    v2._check_v2_lock(condition)
    trace = BufferedAttempt(condition, seed, fold, execution)
    try:
        with activate(trace, v2, batching):
            trace.flush('fit_before')
            # F2's captured _FIT is the original F1 fit.  Its activate context
            # supplies the exact locked F2 model/LUNA repair/selection evidence.
            result = v2._FIT(condition, seed, fold, resume=resume,
                             micro_subjects=micro_subjects, window_chunk=window_chunk)
        trace.finish()
        return result
    except BaseException as error:
        if not trace.terminal:
            try:
                trace.finish(error)
            except BaseException as flush_error:
                raise flush_error from error
        raise
    finally:
        trace.restore()


def run(conditions, *, acceleration_registration, resume=False, limit=None,
        micro_subjects=1, window_chunk=8):
    from . import protocol as P
    require(limit is None or type(limit) is int and limit > 0, 'Invalid limit')
    require(conditions and len(set(conditions)) == len(conditions), 'Empty/duplicate conditions')
    completed = 0
    for condition in conditions:
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                fit(condition, seed, fold, acceleration_registration=acceleration_registration,
                    resume=resume, micro_subjects=micro_subjects, window_chunk=window_chunk)
                completed += 1
                if limit is not None and completed >= limit:
                    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs='+', required=True)
    parser.add_argument('--acceleration-registration', required=True)
    parser.add_argument('--acceleration-registration-sha256', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--micro-subjects', type=int, choices=(1,), default=1)
    parser.add_argument('--window-chunk', type=int, choices=(8,), default=8)
    args = parser.parse_args()
    binding = {'path': args.acceleration_registration,
               'sha256': args.acceleration_registration_sha256}
    run(args.conditions, acceleration_registration=binding, resume=args.resume,
        limit=args.limit, micro_subjects=args.micro_subjects, window_chunk=args.window_chunk)


if __name__ == '__main__':
    main()
