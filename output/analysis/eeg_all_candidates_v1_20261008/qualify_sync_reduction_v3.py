"""Recovered-Q5 control gate around the unchanged V1 diagnostic probe."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import types

SOURCE = Path(__file__).resolve()
DELEGATE = SOURCE.with_name('qualify_sync_reduction_v1.py')
DELEGATE_SHA = 'c97df4301857a9511a0d2469eaa2a88324a6d5df5544eaf6f09f81477b61bea8'


def compatibility_tree(data):
    tree = ast.parse(data); count = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'tridep_ccml.eeg_all_candidates':
            for alias in node.names:
                if alias.name == 'sync_qualification_gate_v1' and alias.asname == 'G':
                    alias.name = 'sync_qualification_gate_v3'; count += 1
    if count != 1: raise ValueError('Expected exactly one original gate import')
    return tree


def load_delegate():
    data = DELEGATE.read_bytes()
    if hashlib.sha256(data).hexdigest() != DELEGATE_SHA: raise ValueError('Sealed V1 diagnostic probe changed')
    module = types.ModuleType('_isolated_diagnostic_sync_qualification_v3'); module.__file__ = str(SOURCE)
    exec(compile(compatibility_tree(data), str(DELEGATE), 'exec'), module.__dict__)
    return module


D = load_delegate()
ORIGINAL_SAVE = D.save


def provenance(value):
    value = dict(value)
    if value.get('schema') in ('diagnostic_sync_qualification_attempt_v1','diagnostic_sync_qualification_v1',
                               'diagnostic_sync_qualification_pending_v1'):
        value.update(schema=value['schema'][:-2]+'v3', source=D.desc(SOURCE),
                     delegated_qualification_source=D.desc(DELEGATE), shared_boundary_gate_source=D.desc(Path(D.G.__file__)),
                     measurement_variants_roles_math_and_repetitions_unchanged=True, formal_activation_authorized=False,
                     qualification_execution_version=3)
        if 'q4_registration' in value: value['queue_registration'] = value.pop('q4_registration')
        if 'source_bindings' in value:
            value['source_bindings'] = {**value['source_bindings'], 'qualification_source':D.desc(SOURCE),
                'delegated_qualification_source':D.desc(DELEGATE), 'shared_boundary_gate_source':D.desc(Path(D.G.__file__))}
    return value


def save_actual(path, value, exclusive=False):
    return ORIGINAL_SAVE(path, provenance(value), exclusive=exclusive)


D.save = save_actual
OWNER = D.OWNER
exact_ce, exact_logits, preflight, qualify, main = D.exact_ce, D.exact_logits, D.preflight, D.qualify, D.main


if __name__ == '__main__': raise SystemExit(main())
