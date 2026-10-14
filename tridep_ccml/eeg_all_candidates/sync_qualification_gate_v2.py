"""Guardian2 compatibility over the unchanged Q4 natural-boundary checks.

This wrapper changes the expected guardian source, proof schema, and requires
an empty XML table of all GPU process types (including pure graphics).
Original Q4/360 lineage, fifteen completed CSBrain identities, held dispatcher,
exited children, fixed completed seed22/fold5 and strict GPU-idle checks remain.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import types

SOURCE = Path(__file__).resolve()
DELEGATE = SOURCE.with_name('sync_qualification_gate_v1.py')
DELEGATE_SHA = '0fdb296e7b77e0d1ece1613de66e91884a508cc81126da361cb406edeb8f3b8e'
REPLACEMENTS = {'sync_qualification_boundary_guard_v1.py': 'sync_qualification_boundary_guard_v2.py',
                'sync_qualification_safe_boundary_v1': 'sync_qualification_safe_boundary_v2'}


def compatibility_tree(data):
    tree = ast.parse(data)
    counts = {key: 0 for key in REPLACEMENTS}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in REPLACEMENTS:
            old = node.value; node.value = REPLACEMENTS[old]; counts[old] += 1
    if counts != {key: 1 for key in REPLACEMENTS}:
        raise ValueError('Reviewed gate compatibility anchors changed')
    return tree


def load_delegate():
    data = DELEGATE.read_bytes()
    if hashlib.sha256(data).hexdigest() != DELEGATE_SHA:
        raise ValueError('Sealed natural-boundary gate changed')
    module = types.ModuleType('isolated_natural_boundary_gate_v2')
    module.__file__ = str(SOURCE)
    exec(compile(compatibility_tree(data), str(DELEGATE), 'exec'), module.__dict__)
    return module


D = load_delegate()
OWNER, GUARDIAN, WORK, OUT, REPORT = D.OWNER, D.GUARDIAN, D.WORK, D.OUT, D.REPORT
descriptor, read_bound, require = D.descriptor, D.read_bound, D.require
verify_shape = D.verify_shape
GPU_OBSERVER = REPORT / 'qualify_acceleration_v2.py'
GPU_OBSERVER_SHA = '3f63b8eb63c33782a972494af0278e8924fa0b866511c7b217f16a2d6e4416e2'


def load_gpu_observer():
    require(descriptor(GPU_OBSERVER)['sha256'] == GPU_OBSERVER_SHA, 'Reviewed XML GPU observer changed')
    spec = importlib.util.spec_from_file_location('_sync_v2_readonly_all_GPU_contexts', GPU_OBSERVER)
    observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
    return observer


def gpu_idle():
    observed = load_gpu_observer().gpu_observation()
    require(observed.get('idle') is True and observed.get('contexts') == [],
            'Performance qualification requires zero GPU contexts of every type')
    return {**observed, 'gpu_observer_source': descriptor(GPU_OBSERVER)}


# The original preflight invokes this stronger read-only observation after the
# exact original held-parent/finished-children checks. No tensor path changes.
D.gpu_idle = gpu_idle


def preflight(proof_binding, builder_binding=None):
    proof, registration, fit, history, evidence = D.preflight(proof_binding, builder_binding)
    return proof, registration, fit, history, {**evidence,
        'actual_shared_gate_source': descriptor(SOURCE), 'delegated_shared_gate_source': descriptor(DELEGATE),
        'gpu_observer_source': descriptor(GPU_OBSERVER),
        'compatibility_changes': ['guardian2_source', 'proof_schema_v2', 'strict_XML_all_GPU_context_types_idle'],
        'original_Q4_full_CSBrain15_and_completed_seed22fold5_checks_unchanged': True}
