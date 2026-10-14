"""Source-bound observer12 verifier reusing the unchanged F3 counter guard.

No actual12 input is relabelled as actual3. An exact original AST suffix only
checks counter/flush arithmetic; actual repair qualification/provenance is
verified separately. Imports/default construction use no tensor/model/GPU.
"""
from __future__ import annotations
import ast
import copy
from pathlib import Path
import types
from . import storage as S, execution_acceleration_v1 as X
from . import deterministic_repair_execution_v9 as D
from . import finetune_v12 as F
SOURCE = Path(__file__).resolve()
X_SHA = 'e9ed83f34ed788cb34b25bf526d637032c1e1080be45166dcf8fd02f75ca1c4f'
F4_SHA = '99636d7c22296a3d49ad0f99e5bd1a047c9fa4b22251b6d97489fd84f922fec4'
D_SHA = '6dd5d4d42f5b10989483da1a1a6da97d147765b57f82537ebccc86e94214bce9'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def source_check():
    require(S.digest(Path(X.__file__)) == X_SHA and S.digest(F.SOURCE) == F4_SHA and (S.digest(D.SOURCE) == D_SHA), 'Reviewed counter guard/formal/qualification verifier source changed')

def counter_guard():
    source_check()
    raw = Path(X.__file__).read_text(encoding='utf-8')
    tree = ast.parse(raw)
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_observer_record'))
    matches = [i for i, n in enumerate(function.body) if isinstance(n, ast.Expr) and ast.get_source_segment(raw, n).startswith('require(record.get("counter_semantics")')]
    require(len(matches) == 1 and isinstance(function.body[-1], ast.Return), 'Exact original counter-guard boundary absent')
    tail = copy.deepcopy(function.body[matches[0]:-1])
    return_node = ast.parse("return {'terminal':terminal,'state':state,'unflushed_tail_unknown':not terminal}").body[0]
    new = ast.FunctionDef(name='_original_buffer_guard', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='record'), ast.arg(arg='fit')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=tail + [return_node], decorator_list=[])
    namespace = dict(vars(X))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[new], type_ignores=[])), str(Path(X.__file__)) + ':counter_only', 'exec'), namespace)
    return namespace['_original_buffer_guard']

def verify_execution(value, *, condition, seed, fold):
    source_check()
    execution = value.get('actual_deterministic_repair_execution', value.get('execution_acceleration', value))
    require(isinstance(execution, dict) and type(execution.get('observer_version')) is int and (execution['observer_version'] == 12), 'Actual observer12 execution required')
    expected = D.load_execution(execution['registration'], condition=condition, seed=seed, fold=fold)
    require(execution == expected, 'Actual observer12 complete registration/qualification differs')
    if 'actual_deterministic_repair_execution' in value:
        require(value.get('execution_acceleration') == expected, 'Legacy-named execution field differs from actual repair')
    if 'actual_finetune_implementation' in value:
        require(value['actual_finetune_implementation'] == F.provenance(expected) and value['versioned_finetune_implementation'] == expected['delegated_finetune_implementation'], 'Actual12 and delegated2 metadata scope differ')
    return expected

def verify_observer(record, artifact, fit):
    require(S.desc(S.checked(artifact)) == artifact and S.read(S.checked(artifact)) == record, 'Actual observer12 input bytes differ')
    require((record.get('condition'), record.get('kind'), record.get('seed'), record.get('fold')) == (fit['condition'], 'formal', fit['seed'], fit['fold']) and type(record.get('seed')) is int and (type(record.get('fold')) is int), 'Actual12 strict formal identity differs')
    execution = verify_execution(record, condition=fit['condition'], seed=fit['seed'], fold=fit['fold'])
    require(record.get('implementation') == F.provenance(execution), 'Actual observer12 full provenance differs')
    proof = counter_guard()(record, fit)
    return {'passed': True, 'observer_version': 12, 'actual_execution': execution, 'counter_guard_source': S.desc(Path(X.__file__)), 'repair_observer_verifier_source': S.desc(SOURCE), 'counter_snapshot_exact_for_process': proof['terminal'], 'unflushed_tail_unknown': proof['unflushed_tail_unknown'], 'original_CUDA_backward_equivalence_claimed': False, 'actual_observer_version_not_normalized': True}

def trace_boundary(legacy, fit, metadata, inputs):
    execution = verify_execution(metadata, condition=fit['condition'], seed=fit['seed'], fold=fit['fold'])
    require(metadata.get('actual_finetune_implementation') == F.provenance(execution), 'Actual completed F12 metadata required')
    normalized = []
    observations = []
    for item in inputs:
        record = item['record']
        require(type(record.get('implementation', {}).get('version')) is int and record['implementation']['version'] == 12, 'No unreviewed mixed observer in repaired fit')
        proof = verify_observer(record, item['artifact'], fit)
        require(proof['actual_execution'] == execution, 'Physical trace and completed repair registration differ')
        if record['status'] == 'complete' and record['optimizer_step_returned'] > 0:
            durable = record.get('latest_durable_checkpoint')
            require(durable is not None and all((durable[k] == metadata[m] for k, m in (('epoch', 'epochs'), ('optimizer_updates', 'optimizer_updates'), ('backward_calls', 'backward_calls')))), 'Actual F12 returned updates lack matching terminal durable checkpoint')
        clone = copy.deepcopy(item)
        clone['record']['implementation'] = X.f2_provenance()
        normalized.append(clone)
        observations.append({'artifact': item['artifact'], 'actual_implementation': record['implementation'], **proof})
    require(any((x['record']['optimizer_step_returned'] > 0 for x in inputs)), 'No actual physical training in completed repaired fit')
    result = legacy(fit, metadata, normalized)
    by_path = {x['artifact']['path']: x for x in observations}
    for row in result['attempts']:
        row['actual_execution_observation'] = by_path[row['artifact']['path']]
    result.update(actual_observer_versions=[12], actual_execution_observations=observations, execution_acceleration_verified=execution, repair_execution_verified=execution, legacy_counter_arithmetic_reused_after_strict_actual_source_verification=True, unflushed_buffered_tails_unknown=any((x['unflushed_tail_unknown'] for x in observations)), actual_observer_source_and_version_preserved=True, historical_or_unflushed_physical_work_zero_filled=False)
    return result
