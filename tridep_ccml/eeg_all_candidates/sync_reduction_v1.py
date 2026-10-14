"""Unregistered diagnostic-only synchronization prototype; stdlib at import.

Eligibility requires a separate real, source-bound bitwise qualification.
No formal queue, model source, global Tensor method, or batch size is patched.
"""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import types

SOURCE = Path(__file__).resolve()
F1_SOURCE = SOURCE.with_name('finetune.py')
F1_SHA = 'cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24'
_MISSING = object()


def require(value, message):
    if not value:
        raise ValueError(message)


def sequential_role_ce(rows, expected):
    """Host-only validation and the exact original left-to-right Python sum."""
    require(len(rows) == expected and expected > 0, 'Missing diagnostic scalar rows')
    total = 0.
    for row in rows:
        require(len(row) in (2, 3), 'Invalid diagnostic scalar row')
        if len(row) == 3:
            require(row[2] == 1., 'Encoder returned nonfinite representations')
        require(row[1] == 1. and math.isfinite(float(row[0])), 'Nonfinite all-window diagnostic CE')
        total += float(row[0])
    return total / expected


def forward_ast(*, coalesce_input_transfer=False):
    """Change only where the original representation finite scalar is checked."""
    data = F1_SOURCE.read_bytes()
    require(hashlib.sha256(data).hexdigest() == F1_SHA, 'Sealed F1 source changed')
    tree = ast.parse(data)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RawSubjectModel')
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    derived = copy.deepcopy(original)
    guards = [n for n in ast.walk(derived) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Name) and n.value.func.id == 'require'
              and len(n.value.args) == 2 and isinstance(n.value.args[1], ast.Constant)
              and n.value.args[1].value == 'Encoder returned nonfinite representations']
    require(len(guards) == 1, 'Expected exactly one representation finite guard')
    guard = guards[0]
    require(ast.dump(guard.value.args[0]) == ast.dump(ast.parse('torch.isfinite(representation).all()', mode='eval').body),
            'Original finite guard arithmetic differs')
    guard.value = ast.Call(func=ast.Name(id='_sync_capture_representation_finite', ctx=ast.Load()), args=[guard.value.args[0]], keywords=[])
    if coalesce_input_transfer:
        loops = [n for n in derived.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'start']
        require(len(loops) == 1, 'Expected original window loop')
        loop = loops[0]
        expected = ast.parse('tensor = torch.as_tensor(adapted[start:start + self.window_chunk], device=self.device)').body[0]
        require(ast.dump(loop.body[0]) == ast.dump(expected), 'Original per-window input transfer differs')
        prefix = ast.parse("require(adapted.flags.c_contiguous, 'Input transfer coalescing requires contiguous unchanged adapted data')\n_sync_adapted_device = torch.as_tensor(adapted, device=self.device)").body
        index = derived.body.index(loop)
        derived.body[index:index] = prefix
        loop.body[0] = ast.parse('tensor = _sync_adapted_device[start:start + self.window_chunk]').body[0]
    return original, ast.fix_missing_locations(derived)


@contextmanager
def deferred_representation_guard(model, F, flags, *, forward_observer=None, coalesce_input_transfer=False):
    """Local forward replacement. Existing call observers must be reattached explicitly."""
    require(type(model.window_chunk) is int and model.window_chunk == 8, 'Synchronization prototype keeps window_chunk8')
    original = model.forward
    native = getattr(original, '__func__', None) is F.RawSubjectModel.forward
    require(native or forward_observer is not None, 'An existing forward observer cannot be silently bypassed')
    original_ast, candidate_ast = forward_ast(coalesce_input_transfer=coalesce_input_transfer)
    namespace = dict(F.RawSubjectModel.forward.__globals__)
    namespace['_sync_capture_representation_finite'] = flags.append
    exec(compile(ast.Module(body=[candidate_ast], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    candidate = types.MethodType(namespace['forward'], model)
    if forward_observer is not None:
        candidate = forward_observer(candidate)
    own = model.__dict__.get('forward', _MISSING)
    model.forward = candidate
    try:
        yield
    finally:
        if own is _MISSING:
            del model.forward
        else:
            model.forward = own


@contextmanager
def unchanged_forward():
    yield


def _make_diagnostics(F, *, defer_representation=True, forward_observer=None, state_digest=None, observer=None,
                      coalesce_input_transfer=False, coalesce_targets=False):
    """Implementation also accepts a fake module in stdlib-only unit tests."""
    digest = state_digest or F.state_digest

    def diagnostics(model, raw, split, labels):
        from .validation_batching_v1 import validate_roles
        require(type(model.window_chunk) is int and model.window_chunk == 8, 'Synchronization prototype keeps window_chunk8')
        validate_roles(raw, split, labels)
        state = F.rng_state(model.device)
        modes = [(module, module.training) for module in model.modules()]
        before = digest(model.state_dict())
        output, representation_flags = {}, []
        observations = {'schema': 'diagnostic_sync_reduction_observation_v1', 'diagnostic_window_chunk': 8,
                        'representation_guard_deferred': defer_representation,
                        'adapted_all60_input_transfer_once_then_original_chunk8_views': coalesce_input_transfer,
                        'target_labels_transferred_once': coalesce_targets,
                        'scalar_device_to_host_transfers_started': 0, 'scalar_device_to_host_transfers_returned': 0,
                        'role_subject_counts': {}, 'all_finite_checks_passed_before_return': False,
                        'optimizer_updates': 0, 'test_forward_count': 0,
                        'gpu_ce_sum_reduction_used': False, 'python_sequential_double_sum_preserved': True}
        context = deferred_representation_guard(model, F, representation_flags, forward_observer=forward_observer,
                    coalesce_input_transfer=coalesce_input_transfer) if defer_representation else unchanged_forward()
        try:
            model.eval()
            with context, F.torch.inference_mode():
                if coalesce_targets:
                    people_in_order = split['train'] + split['val']
                    target_positions = {sid: i for i, sid in enumerate(people_in_order)}
                    packed_targets = F.torch.tensor([labels[sid] for sid in people_in_order], device=model.device)
                for role in ('train', 'val'):
                    people = split[role]
                    losses, finite_losses = [], []
                    representation_flags.clear()
                    for sid in people:
                        require(sid in raw and sid not in split['test'], 'Outer-test person reached diagnostics')
                        logits = model(raw[sid][None])
                        target = (packed_targets[target_positions[sid]:target_positions[sid] + 1] if coalesce_targets
                                  else F.torch.tensor([labels[sid]], device=model.device))
                        loss = F.torch.nn.functional.cross_entropy(logits, target)
                        losses.append(loss.detach())
                        finite_losses.append(F.torch.isfinite(loss))
                    columns = [F.torch.stack(losses), F.torch.stack(finite_losses).to(dtype=losses[0].dtype)]
                    if defer_representation:
                        require(len(representation_flags) == len(people), 'Missing representation finite flags')
                        columns.append(F.torch.stack(representation_flags).to(dtype=losses[0].dtype))
                    observations['scalar_device_to_host_transfers_started'] += 1
                    rows = F.torch.stack(columns, dim=1).cpu().tolist()
                    observations['scalar_device_to_host_transfers_returned'] += 1
                    output[role + '_eval_ce'] = sequential_role_ce(rows, len(people))
                    observations['role_subject_counts'][role] = len(people)
            observations['state_unchanged_on_exit'] = before == digest(model.state_dict())
            require(observations['state_unchanged_on_exit'], 'Diagnostics mutated parameter/buffer state')
            observations['all_finite_checks_passed_before_return'] = True
        finally:
            for module, mode in modes:
                module.training = mode
            F.restore_rng(state)
            if 'state_unchanged_on_exit' not in observations:
                observations['state_unchanged_on_exit'] = before == digest(model.state_dict())
            observations['rng_restored_on_exit'] = F.rng_equal(state, F.rng_state(model.device))
            observations['modes_restored_on_exit'] = all(module.training is mode for module, mode in modes)
            if observer is not None:
                observer(observations)
        require(observations['state_unchanged_on_exit'] and observations['rng_restored_on_exit']
                and observations['modes_restored_on_exit'], 'Diagnostics failed to preserve state/RNG/modes')
        return output

    return diagnostics


def make_diagnostics(original, *, defer_representation_guard=True, forward_observer=None, state_digest=None, observer=None,
                     coalesce_input_transfer=False, coalesce_targets=False):
    """Prototype factory. Runtime registration and real qualification are external gates.

    state_digest can be a separately qualified exact digest implementation.
    It is never installed globally. The caller must register all actual sources.
    """
    require(hashlib.sha256(F1_SOURCE.read_bytes()).hexdigest() == F1_SHA, 'Sealed F1 source changed')
    require(original.__name__ == 'diagnostics' and Path(original.__code__.co_filename).resolve() == F1_SOURCE,
            'Factory must receive the original sealed F1 diagnostics function')
    require(type(defer_representation_guard) is bool, 'Explicit representation guard mode required')
    require(type(coalesce_input_transfer) is bool and type(coalesce_targets) is bool
            and (not coalesce_input_transfer or defer_representation_guard), 'Explicit isolated input/target transfer mode required')
    namespace = types.SimpleNamespace(**original.__globals__)
    return _make_diagnostics(namespace, defer_representation=defer_representation_guard,
                             forward_observer=forward_observer, state_digest=state_digest, observer=observer,
                             coalesce_input_transfer=coalesce_input_transfer, coalesce_targets=coalesce_targets)
