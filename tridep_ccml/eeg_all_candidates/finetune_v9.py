"""Qualified deterministic-backward repair runner; never auto-prepare or probe.

Original F1 training/selection/resume and F2 metadata code are delegated.
Buffered counters/flush boundaries reuse F3. Only the model's private adapter
route differs, after actual engineering and complete new-source registration.
This is actual observer9, not an F3/Q5 completion or a speed improvement.
Default CLI performs no tensor import/model/raw/GPU/fit/output operation.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import importlib
from pathlib import Path
import types
from . import storage as S
from . import finetune_v3 as TEMPLATE
from . import deterministic_repair_execution_v6 as D
from . import qualify_deterministic_repairs_v8 as Q
SOURCE = Path(__file__).resolve()
_ACTIVE = False

def require(ok, message):
    if not ok:
        raise ValueError(message)

def provenance(execution):
    return {'version': 9, 'wrapper_source': S.desc(SOURCE), 'delegated_observer_source': S.desc(Path(TEMPLATE.__file__)), 'delegated_wrapper_source': execution['delegated_finetune_implementation']['wrapper_source'], 'delegated_runner_source': execution['delegated_finetune_implementation']['delegated_runner_source'], 'delegated_finetune_implementation': copy.deepcopy(execution['delegated_finetune_implementation']), 'pipeline_sampler_warmup_optimizer_stop_and_resume_unchanged': True, 'backward_graph_privately_changed': True, 'original_CUDA_backward_equivalence_established': False, 'numerical_accumulation_order_changed': execution['deterministic_backward_repair']['accumulation_order_changed'], 'qualification': execution['qualification'], 'repair_registration': execution['registration'], 'no_validation_or_test_score_configuration_selection': True, 'performance_improvement_claimed': False}

class RepairAttempt(TEMPLATE.BufferedAttempt):
    """Original per-call counters and persistence, with true version4 sources."""

    def flush(self, reason, **kwargs):
        self.record['implementation'] = provenance(self.record['execution_acceleration'])
        self.record['actual_deterministic_repair_execution'] = copy.deepcopy(self.record['execution_acceleration'])
        return super().flush(reason, **kwargs)

@contextmanager
def activate(trace, v2, batching):
    global _ACTIVE
    require(not _ACTIVE, 'No nested formal repair context')
    v1 = v2.V1
    original_builder = Q.isolated_builder(types.SimpleNamespace(build_model=v2._BUILD))
    _ACTIVE = True
    try:
        with TEMPLATE.activate(trace, v2, batching):
            previous_builder = v1.build_model
            previous_save = S.save
            execution = trace.record['execution_acceleration']
            folder = v1.R.fit_directory(trace.record['condition'], trace.record['seed'], trace.record['fold']).resolve()

            def private_builder(condition, *, seed=11, device='cuda', window_chunk=8):
                require(condition == trace.record['condition'] and seed == trace.record['seed'] and (window_chunk == 8), 'Actual source-bound model identity/chunk differs')
                model, audit = original_builder(condition, seed=seed, device=device, window_chunk=8)
                audit['versioned_finetune_implementation'] = v2.provenance()
                audit['actual_deterministic_repair_execution'] = copy.deepcopy(execution)
                model.forward = trace.observe('forward', model.forward)
                trace.bind_adapter(model.adapter)
                return (model, audit)

            def save_repair_metadata(path, value, *, immutable=False):
                if Path(path).resolve() == folder / 'metadata.json':
                    require(isinstance(value, dict), 'Metadata mapping required')
                    value['actual_finetune_implementation'] = provenance(execution)
                    value['actual_deterministic_repair_execution'] = copy.deepcopy(execution)
                    value['backward_repair_qualification'] = execution['qualification']
                return previous_save(path, value, immutable=immutable)
            v1.build_model = private_builder
            S.save = save_repair_metadata
            try:
                yield
            finally:
                v1.build_model = previous_builder
                S.save = previous_save
    finally:
        _ACTIVE = False

def fit(condition, seed, fold, *, repair_registration, resume=False):
    Q.check_pause()
    execution = D.load_execution(repair_registration, condition=condition, seed=seed, fold=fold)
    require(execution['source'] == S.desc(SOURCE) and execution['observer_version'] == 9, 'Actual observer9 source differs')
    v2 = importlib.import_module('.finetune_v2', __package__)
    batching = importlib.import_module('.validation_batching_v1', __package__)
    require(v2.provenance() == execution['delegated_finetune_implementation'], 'Sealed training/metadata delegate changed')
    status = v2.V1.R.fit_status(condition, seed, fold)
    if status['status'] == 'training_complete':
        folder = v2.V1.R.fit_directory(condition, seed, fold)
        metadata = S.read(folder / 'metadata.json')
        require(metadata.get('actual_deterministic_repair_execution') == execution and metadata.get('actual_finetune_implementation') == provenance(execution), 'Cannot relabel historical completion as actual observer9')
        v2.V1.R._validate_complete(S.read(folder / 'complete.json'), folder)
        return metadata
    trace = RepairAttempt(condition, seed, fold, execution)
    try:
        with activate(trace, v2, batching):
            trace.flush('fit_before')
            result = v2._FIT(condition, seed, fold, resume=resume, micro_subjects=1, window_chunk=8)
        trace.finish()
        return result
    except BaseException as error:
        if not trace.terminal:
            trace.finish(error)
        raise
    finally:
        trace.restore()

def run(condition, *, repair_registration, resume=False, limit=None):
    require(condition in D.REPAIR_CONDITIONS and (limit is None or (type(limit) is int and limit > 0)), 'Exact repair condition/limit required')
    completed = 0
    for seed in (11, 22, 33):
        for fold in (1, 2, 3, 4, 5):
            Q.check_pause()
            fit(condition, seed, fold, repair_registration=repair_registration, resume=resume)
            completed += 1
            if limit is not None and completed >= limit:
                return

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--condition', choices=D.REPAIR_CONDITIONS)
    parser.add_argument('--repair-registration')
    parser.add_argument('--repair-registration-sha256')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    if not args.run:
        require(args.repair_registration is None and (not args.resume) and (args.limit is None), 'Actual execution options require --run')
        import json
        print(json.dumps({'status': 'preparation_only', 'actual_observer_version': 9, 'model_calls': 0, 'GPU_calls': 0, 'formal_fits': 0, 'registration_or_fit_writes': 0, 'actual_GPU_qualification_required': True, 'new_complete_source_registration_required': True}))
        return
    require(args.condition is not None and args.repair_registration is not None and (args.repair_registration_sha256 is not None), 'Actual source-bound repair registration required')
    return run(args.condition, repair_registration={'path': args.repair_registration, 'sha256': args.repair_registration_sha256}, resume=args.resume, limit=args.limit)
if __name__ == '__main__':
    main()
