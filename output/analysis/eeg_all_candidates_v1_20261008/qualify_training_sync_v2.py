"""Natural-boundary V2 gate around the unchanged zero-update training probe.

The 96-backward/zero-optimizer tensor probe is the sealed V1 implementation.
Only guardian schema, shared gate and truthful emitted provenance are adapted.
No --run writes a new pending JSON; importing this file performs no GPU work.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import types

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
DELEGATE = SOURCE.with_name('qualify_training_sync_v1.py')
DELEGATE_SHA = '4507196d0d159b94174249d0e62bc8e88a4e8bb3b00a00a6387a04e9778bb019'
GATE_SOURCE = WORK / 'tridep_ccml/eeg_all_candidates/sync_qualification_gate_v2.py'
GATE_SHA = 'f14f2b09fc14b1dd26c62a4f541cf560337e632830bdee17aa58a0c43dd06223'


def load_delegate():
    raw = DELEGATE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DELEGATE_SHA:
        raise ValueError('Sealed zero-optimizer training qualification changed')
    module = types.ModuleType('_isolated_training_sync_qualification_v2')
    module.__file__ = str(SOURCE)
    exec(compile(raw, str(DELEGATE), 'exec'), module.__dict__)
    module.GATE_SOURCE, module.GATE_SHA = GATE_SOURCE, GATE_SHA
    # This one schema literal is the only change to the identity validator.
    # No checkpoint, RNG, optimizer, forward, backward or comparison code changes.
    tree = ast.parse(raw)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_fixed_identity')
    literals = [n for n in ast.walk(function) if isinstance(n, ast.Constant) and n.value == 'sync_qualification_safe_boundary_v1']
    if len(literals) != 1:
        raise ValueError('Reviewed identity schema guard changed')
    literals[0].value = 'sync_qualification_safe_boundary_v2'
    isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(isolated, str(SOURCE), 'exec'), module.__dict__)
    return module


D = load_delegate()
descriptor = D.descriptor
_original_helpers = D.helpers


def provenance(value):
    value = dict(value)
    if value.get('schema') in ('training_sync_qualification_attempt_v1', 'training_sync_qualification_v1',
                              'training_sync_qualification_pending_v1'):
        D.require(descriptor(DELEGATE)['sha256']==DELEGATE_SHA and descriptor(GATE_SOURCE)['sha256']==GATE_SHA,
                  'Reviewed delegate or G2 source changed')
        value['schema'] = value['schema'][:-2] + 'v2'
        value['source'] = descriptor(SOURCE)
        value['qualification_execution_version'] = 2
        value['delegated_training_qualification_source'] = descriptor(DELEGATE)
        value['actual_gate_source'] = descriptor(GATE_SOURCE)
        value['tensor_probe_and_budget_unchanged'] = True
    return value


def helpers():
    module = _original_helpers()
    original = module.atomic_json
    def write(path, value, exclusive=False):
        return original(path, provenance(value), exclusive=exclusive)
    module.atomic_json = write
    return module


D.helpers = helpers
qualify = D.qualify
main = D.main


if __name__ == '__main__':
    raise SystemExit(main())
