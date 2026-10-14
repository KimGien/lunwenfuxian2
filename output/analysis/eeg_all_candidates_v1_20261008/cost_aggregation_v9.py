"""Cost V9 compatibility for separately bound F3 process-loss closures.

Reuse the sealed V6 counter aggregation function exactly once on the original
V5 inventory. Do not sum recovery deltas into persisted physical counters.
Unknown process elapsed and unflushed tails stay null. No collect occurs at import.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
REPORT = SOURCE.parent
DELEGATE = REPORT / 'cost_aggregation_v6.py'
DELEGATE_SHA = 'a16f6562dd9809991639fe77df7fa7cd0f9ca94a264c6a8f31f80f7a1012aa8b'
VERIFIER = WORK / 'tridep_ccml/eeg_all_candidates/lost_f3_closure_v3.py'
VERIFIER_SHA = 'b7067820b9ff0fb3f6cfddf62991acb918d6a6198cdcf6ead07d3f0f5cffbaf1'
PREVIOUS_VERIFIER = WORK / 'tridep_ccml/eeg_all_candidates/lost_f3_closure_v2.py'
PREVIOUS_VERIFIER_SHA = '89d5a04868c07d377f50712f23dcf271425c060831254862eb2ee27edb5761e6'
PREVIOUS_COST = REPORT / 'cost_aggregation_v7.py'
PREVIOUS_COST_SHA = 'ee232c5e231cc665e0173db27c5a9890b5b421167410ada1b6198f99af1f46a1'


def require(ok, message):
    if not ok: raise ValueError(message)


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK), 'Cost artifact outside workspace')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_source(path, digest):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'Sealed source changed: '+Path(path).name)
    return raw


def validate_disclosure(proof):
    """Proof producer verifies each event's own delta, not a blanket old4/27."""
    for key in ('passed', 'persisted_counters_are_lower_bounds', 'unflushed_tail_unknown', 'extra_work_not_added_again'):
        require(proof.get(key) is True, 'Lost-process disclosure differs: '+key)
    require(proof.get('original_status') == 'running' and proof.get('graceful_terminal_flush') is False,
            'Absence cannot manufacture a graceful terminal state')
    for key in ('unflushed_tail_counters', 'final_elapsed_seconds'):
        require(key in proof and proof[key] is None, 'Unobserved tail/time must be explicit null')
    for key in ('known_extra_optimizer_returns_outside_checkpoint', 'known_extra_backward_returns_outside_checkpoint'):
        require(key in proof and (proof[key] is None or (type(proof[key]) is int and proof[key] >= 0)),
                'Known event-specific lower bound must be a nonnegative integer or unknown')
    for key in ('closure', 'recovery_registration', 'closure_source', 'checkpoint_cpu_resume_proof'):
        value = proof.get(key)
        require(type(value) is dict and set(value) == {'path', 'sha256'}
                and type(value['path']) is str and type(value['sha256']) is str and len(value['sha256']) == 64,
                'Source-bound closure evidence missing: '+key)


def aggregation_delegate(verifier, reader, verifier_source):
    """Compile only the unchanged sealed pure aggregation body, no project import."""
    checked_source(PREVIOUS_COST, PREVIOUS_COST_SHA)
    checked_source(PREVIOUS_VERIFIER, PREVIOUS_VERIFIER_SHA)
    source = checked_source(DELEGATE, DELEGATE_SHA)
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'augment']
    require(len(nodes) == 1, 'Exact sealed augmentation function absent')
    def desc(path):
        if Path(path).resolve() == VERIFIER.resolve(): return verifier_source
        return descriptor(path)
    namespace = {'L': types.SimpleNamespace(verify_lost_f3_attempt=verifier, read_json=reader,
                    descriptor=desc, __file__=str(VERIFIER)),
                 'SOURCE': SOURCE, 'Path': Path, 'json': json, 'validate_disclosure': validate_disclosure}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(DELEGATE), 'exec'), namespace)
    return namespace['augment']


def augment(result, *, verifier, reader, verifier_source):
    require('source_bound_lost_f3_attempts' not in result and result.get('schema_version') not in (6, 7, 8, 9),
            'Start from the original V5 inventory; never reaugment previously accounted closures')
    value = aggregation_delegate(verifier, reader, verifier_source)(result, verifier=verifier, reader=reader)
    value.update(schema_version=9, source=descriptor(SOURCE), previous_cost_source_v6=descriptor(DELEGATE), previous_cost_source_v7=descriptor(PREVIOUS_COST),
                 delegated_counter_aggregation_source=descriptor(DELEGATE),
                 lost_f3_closure_verifier_source=verifier_source,
                 previous_lost_f3_closure_verifier_source={'path': PREVIOUS_VERIFIER.relative_to(WORK).as_posix(), 'sha256': PREVIOUS_VERIFIER_SHA},
                 process_loss_closures_aggregated_once=True,
                 process_loss_known_deltas_are_subsets_of_persisted_primary_counters=True,
                 no_closed_attempt_elapsed_fabricated=True)
    bind_lost_parent_scopes(value, reader)
    value['rules'].extend([
        'Each independently verified loss event retains its own checkpoint boundary. The historical4/27, F2handoff0/2 and subsequent losses are not aliases.',
        'A known recovery delta is already a subset of that attempt\'s persisted physical counters; replay work appears only in the new real attempt, never as an added synthetic update.',
        'Same-memory native pauses, system low-power event scope and CPU observer cost remain separate evidence. V9 does not turn raw recorded elapsed into exact active training time.',
        'The bounded low-power window with an unmatched later entry still needs final event-range scope closure; no collector claims all future pause scopes complete.'])
    return value


def bind_lost_parent_scopes(value, reader):
    """Parent observer elapsed is separate, non-additive and unknown after loss."""
    parents = {}; errors = []
    for proof in value['source_bound_lost_f3_attempts']:
        if 'original_queue_attempt' not in proof: continue
        try:
            binding = proof['original_queue_attempt']
            require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Bound original parent descriptor absent')
            require('queue_attempt_final_elapsed_seconds' in proof and proof['queue_attempt_final_elapsed_seconds'] is None,
                    'Loss cannot supply a parent elapsed endpoint')
            key = binding['path']
            require(key not in parents or parents[key]['artifact'] == binding, 'Same lost parent bound to conflicting bytes')
            if key in parents: continue
            rows = [r for r in value['cost_records'] if r['artifact']['path'] == key]
            require(len(rows) == 1 and rows[0]['artifact'] == binding
                    and rows[0]['category'] == 'dispatcher_elapsed'
                    and rows[0]['aggregation_role'] == 'supplemental_nested_or_unmeasured', 'Original lost parent missing/duplicated/misclassified')
            row = rows[0]; original = reader(binding)
            require(original.get('status') == row.get('status') == 'running', 'Original lost parent status changed')
            row['seconds_observed'] = None
            row['seconds_scope'] = 'Source-bound lost parent observer elapsed unknown; not added to formal attempts or engineering'
            row['source_bound_process_loss_closure'] = proof['closure']
            row['parent_elapsed_added_to_formal_or_primary_work'] = False
            parents[key] = {'artifact': binding, 'closure': proof['closure'], 'original_status': 'running',
                            'final_elapsed_seconds': None, 'added_again': False, 'aggregation_role': row['aggregation_role']}
            if binding not in value['inputs']: value['inputs'].append(binding)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({'artifact': proof.get('original_queue_attempt'), 'error_type': type(error).__name__, 'error': str(error)})
    value['source_bound_lost_parent_observer_scopes'] = list(parents.values())
    value['lost_parent_observer_verification'] = {'passed_for_observed_closures': not errors, 'errors': errors,
                                               'parent_seconds_added_to_formal_cost': False}
    value['input_read_errors'].extend(errors)


def load_module(path, digest, name):
    checked_source(path, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_v5():
    """Expose the unchanged field normalizer expected by the finalizer family."""
    if str(WORK) not in sys.path: sys.path.insert(0, str(WORK))
    return load_module(DELEGATE, DELEGATE_SHA, '_cost_v9_sealed_v6_normalizer').load_v5()


def collect():
    if str(WORK) not in sys.path: sys.path.insert(0, str(WORK))
    verifier = load_module(VERIFIER, VERIFIER_SHA, 'tridep_ccml.eeg_all_candidates._cost_v9_bound_closure_v3')
    # Calling previous.collect() would apply old fixed4/27 validation first.
    # Instead preserve its original source-checked V5 inventory and run one pass.
    result = load_v5().collect()
    return augment(result, verifier=verifier.verify_lost_f3_attempt, reader=verifier.read_json,
                   verifier_source=descriptor(VERIFIER))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({'status': 'pending_explicit_run', 'source': descriptor(SOURCE), 'actual_cost_collection': False}))
        return 2
    value = collect()
    target = (args.output or REPORT/('actual_cost_v9_'+str(time.time_ns())+'.json')).resolve()
    require(target.parent == REPORT, 'Write a new standalone analysis report only')
    with target.open('x', encoding='utf-8', newline='\n') as stream: json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'artifact': descriptor(target), 'lost_f3_closures': len(value['source_bound_lost_f3_attempts']),
                      'read_errors': value['input_read_errors'], 'draft': True}))


if __name__ == '__main__': raise SystemExit(main())
