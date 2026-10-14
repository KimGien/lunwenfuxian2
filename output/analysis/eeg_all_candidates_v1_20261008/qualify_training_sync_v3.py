"""Recovered-Q5 gate/provenance around the sealed 96-backward V1 probe.

The original two phases, eight train people, repeats, exact comparisons and
zero real optimizer steps are unchanged. Import/default mode runs no model.
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
GATE_SOURCE = WORK/'tridep_ccml/eeg_all_candidates/sync_qualification_gate_v3.py'
GATE_SHA = 'c206ea734195eda173893fcfc3363172d72c7892f4a830a875bf894e0041c47d'


def load_delegate():
    raw = DELEGATE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DELEGATE_SHA: raise ValueError('Sealed zero-step tensor probe changed')
    module = types.ModuleType('_isolated_training_sync_qualification_v3'); module.__file__ = str(SOURCE)
    exec(compile(raw,str(DELEGATE),'exec'),module.__dict__)
    module.GATE_SOURCE, module.GATE_SHA = GATE_SOURCE, GATE_SHA
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'validate_fixed_identity')
    anchors = [n for n in ast.walk(node) if isinstance(n,ast.Constant) and n.value == 'sync_qualification_safe_boundary_v1']
    if len(anchors) != 1: raise ValueError('Expected one original boundary-schema literal')
    anchors[0].value = 'sync_qualification_safe_boundary_v3'
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),module.__dict__)
    return module


D = load_delegate()
descriptor = D.descriptor
_original_helpers = D.helpers


def provenance(value):
    value = dict(value)
    if value.get('schema') in ('training_sync_qualification_attempt_v1','training_sync_qualification_v1',
                               'training_sync_qualification_pending_v1'):
        D.require(descriptor(DELEGATE)['sha256'] == DELEGATE_SHA and descriptor(GATE_SOURCE)['sha256'] == GATE_SHA,
                  'Reviewed tensor delegate or actual gate3 source changed')
        value.update(schema=value['schema'][:-2]+'v3', source=descriptor(SOURCE), qualification_execution_version=3,
                     delegated_training_qualification_source=descriptor(DELEGATE), actual_gate_source=descriptor(GATE_SOURCE),
                     tensor_probe_and_budget_unchanged=True, formal_activation_authorized=False)
    return value


def helpers():
    module = _original_helpers(); original = module.atomic_json
    def write(path,value,exclusive=False): return original(path,provenance(value),exclusive=exclusive)
    module.atomic_json = write
    return module


D.helpers = helpers
qualify, main = D.qualify, D.main


if __name__ == '__main__': raise SystemExit(main())
