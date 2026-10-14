"""Guardian2 entry for the unchanged two-variant diagnostic sync measurement.

No --run produces only pending JSON. Explicit --run still needs live natural
full-CSBrain15 Q4 boundary proof, exclusive idle GPU, fixed completed22/fold5,
the same warmup/repeats/bitwise comparisons and zero training/test updates.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import types

SOURCE = Path(__file__).resolve()
DELEGATE = SOURCE.with_name('qualify_sync_reduction_v1.py')
DELEGATE_SHA = 'c97df4301857a9511a0d2469eaa2a88324a6d5df5544eaf6f09f81477b61bea8'
GATE_DELEGATE_SHA = '0fdb296e7b77e0d1ece1613de66e91884a508cc81126da361cb406edeb8f3b8e'


def compatibility_tree(data):
    tree = ast.parse(data)
    changed = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'tridep_ccml.eeg_all_candidates':
            for alias in node.names:
                if alias.name == 'sync_qualification_gate_v1' and alias.asname == 'G':
                    alias.name = 'sync_qualification_gate_v2'; changed += 1
    if changed != 1:
        raise ValueError('Expected exactly one reviewed boundary-gate import')
    return tree


def load_delegate():
    data = DELEGATE.read_bytes()
    if hashlib.sha256(data).hexdigest() != DELEGATE_SHA:
        raise ValueError('Sealed diagnostic sync qualifier changed')
    module = types.ModuleType('isolated_diagnostic_sync_qualifier_v2')
    module.__file__ = str(SOURCE)
    exec(compile(compatibility_tree(data), str(DELEGATE), 'exec'), module.__dict__)
    return module


D = load_delegate()
G = D.G
if G.DELEGATE_SHA != GATE_DELEGATE_SHA:
    raise ValueError('Gate must delegate the reviewed original safety checks')
ORIGINAL_SAVE = D.save


def provenance(value):
    value = dict(value)
    schema = value.get('schema')
    if schema in ('diagnostic_sync_qualification_attempt_v1', 'diagnostic_sync_qualification_v1',
                  'diagnostic_sync_qualification_pending_v1'):
        value['schema'] = schema[:-2] + 'v2'
        value['source'] = D.desc(SOURCE)
        value['delegated_qualification_source'] = D.desc(DELEGATE)
        value['shared_boundary_gate_source'] = D.desc(Path(G.__file__))
        value['delegated_boundary_gate_source'] = D.desc(G.DELEGATE)
        value['measurement_variants_roles_math_and_repetitions_unchanged'] = True
        value['formal_activation_authorized'] = False
        if 'source_bindings' in value:
            value['source_bindings'] = {**value['source_bindings'],
                'qualification_source': D.desc(SOURCE), 'delegated_qualification_source': D.desc(DELEGATE),
                'shared_boundary_gate_source': D.desc(Path(G.__file__)), 'delegated_boundary_gate_source': D.desc(G.DELEGATE)}
    return value


def save_actual(path, value, exclusive=False):
    return ORIGINAL_SAVE(path, provenance(value), exclusive=exclusive)


D.save = save_actual
OWNER = D.OWNER
exact_ce, exact_logits, preflight = D.exact_ce, D.exact_logits, D.preflight
qualify, main = D.qualify, D.main


if __name__ == '__main__':
    raise SystemExit(main())
