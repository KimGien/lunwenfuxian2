"""Unqualified, opt-in subject-update sync prototype; never patches live code.

The original micro1 forward/backward, CE checks, parameter/gradient clipping,
optimizer, and eight-window chunk remain unchanged. Only detached CE copies and
gradient-finite host observations are coalesced. Importing this module has no
tensor dependency. This prototype is not an execution registration or a GPU
qualification and must not be used by a formal runner without both.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import types

SOURCE = Path(__file__).resolve()
ORIGINAL_SOURCE = SOURCE.with_name('finetune.py')
ORIGINAL_SHA256 = 'cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24'


def _require(value, message):
    if not value:
        raise ValueError(message)


def explicit_host_observation_counts(microbatches, active_parameters):
    """Successful-path source counts, not timings or hidden-library counts."""
    _require(type(microbatches) is int and 1 <= microbatches <= 8
             and type(active_parameters) is int and active_parameters > 0,
             'Positive strict integer microbatch/active-parameter counts required')
    return {'original':3 * microbatches + active_parameters,
            'candidate':2 * microbatches + 1,
            'removed':microbatches + active_parameters - 1,
            'representation_and_ce_pre_backward_checks_retained':2 * microbatches,
            'candidate_combined_ce_and_gradient_flag_host_copy':1,
            'scope':'Successful subject_update plus original representation finite checks; excludes adapter/library/optimizer internal observations',
            'elapsed_seconds_saved':None,'gpu_busy_fraction':None}


def _subject_update(model, optimizer, people, raw, labels, window_ids, *, micro_subjects):
    """Candidate body; globals are copied from the source-verified original."""
    require(micro_subjects == 1 and model.window_chunk == 8,
            'Sync prototype supports only registered micro1/train window_chunk8')
    require(1 <= len(people) <= 8 and micro_subjects in (1, 2), "Logical batch must contain 1..8 people and microbatch 1/2")
    require(len(people) == len(set(people)), "A person must contribute exactly one CE to this logical batch")
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    backward_calls = 0
    detached_ce = []
    for start in range(0, len(people), micro_subjects):
        chosen = people[start:start + micro_subjects]
        batch = np.stack([raw[sid][window_ids] for sid in chosen])
        target = torch.tensor([labels[sid] for sid in chosen], device=model.device, dtype=torch.long)
        ce_sum = torch.nn.functional.cross_entropy(model(batch), target, reduction="sum")
        require(torch.isfinite(ce_sum), "Nonfinite raw-bag subject CE")
        (ce_sum / len(people)).backward()
        detached_ce.append(ce_sum.detach())
        backward_calls += 1
    finite_flags = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None, "Nonfinite or disconnected trainable gradient: " + name)
            finite_flags.append(torch.isfinite(parameter.grad).all())
        else:
            require(parameter.grad is None, "Frozen parameter acquired a gradient: " + name)
    require(bool(finite_flags), 'No trainable gradient flags available')
    # Bool-to-CE-dtype casts are exact 0/1. No CE/gradient floating sum runs on
    # the device: stack preserves each original detached scalar unchanged.
    finite = torch.stack(finite_flags).all()
    host_values = torch.stack(detached_ce + [finite.to(dtype=detached_ce[0].dtype)]).cpu().tolist()
    require(host_values[-1] == 1., 'Nonfinite trainable gradient: coalesced finite check')
    for scalar_ce in host_values[:-1]:
        total += float(scalar_ce)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], POLICY.gradient_clip)
    optimizer.step()
    return {"subject_ce_sum": total, "subjects": len(people), "backward_calls": backward_calls}


def make_subject_update(original):
    """Return a prototype callable; caller owns isolation and future provenance.

    Pass the original F1 function before observer wrapping. A future registered
    runner must install its logical-update observer around this callable, so F3
    boundary and every actual forward/backward/step observation remain intact.
    The returned function does not modify F1/F2/F3 globals or model.forward.
    """
    raw = ORIGINAL_SOURCE.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA256, 'Original F1 source changed')
    _require(isinstance(original, types.FunctionType) and original.__name__ == 'subject_update'
             and Path(original.__code__.co_filename).resolve() == ORIGINAL_SOURCE,
             'Expected original unwrapped F1 subject_update')
    source_tree = ast.parse(raw.decode('utf-8-sig'))
    definitions = [node for node in source_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'subject_update']
    actual = ast.parse(inspect.getsource(original)).body
    _require(len(definitions) == len(actual) == 1 and ast.dump(definitions[0], include_attributes=False)
             == ast.dump(actual[0], include_attributes=False), 'Original subject_update syntax differs')
    context = dict(original.__globals__)
    _require(all(key in context for key in ('require','torch','np','POLICY')), 'Original execution globals missing')
    candidate = types.FunctionType(_subject_update.__code__, context, 'subject_update_sync_candidate_v1')
    candidate.__kwdefaults__ = dict(_subject_update.__kwdefaults__ or {})
    candidate.__doc__ = _subject_update.__doc__
    candidate.__module__ = __name__
    candidate.__qualname__ = 'subject_update_sync_candidate_v1'
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    candidate.sync_candidate_provenance = {
        'schema':'subject_update_sync_prototype_v1','source_sha256':source_digest,
        'source':{'path':SOURCE.relative_to(SOURCE.parents[2]).as_posix(),'sha256':source_digest},
        'original_math_source':{'path':ORIGINAL_SOURCE.relative_to(SOURCE.parents[2]).as_posix(),'sha256':ORIGINAL_SHA256},
        'original_source_sha256':ORIGINAL_SHA256,'gpu_qualified':False,'formal_execution_authorized':False,
        'micro_subjects':1,'training_window_chunk':8,'forward_representation_finite_check_unchanged':True,
        'ce_finite_checked_before_each_backward':True,'python_ce_float_addition_order_unchanged':True,
        'gradient_finite_failure_stops_before_clip_and_step':True,
        'gradient_error_text_may_differ':'coalesced nonfinite gradients do not identify a single parameter; disconnected/frozen names retained',
        'detached_ce_host_copy_failure_boundary':'A transfer failure occurs after all micro backward calls and before clip/step; it must remain a failed physical attempt',
    }
    return candidate
