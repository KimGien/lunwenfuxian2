"""REVE seed11 five-fold JSON/source review; pending until real five-fit CPU proof.

Delegates immutable reviewed semantic functions through explicit AST adaptation.
Only .py/.json evidence bytes are read. Weight/document/config descriptors outside
that set are compared with the existing actual CPU proof, never opened here.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve()
REPORT = SOURCE.parent
WORK = SOURCE.parents[3]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
BASE_PATH = REPORT / 'review_csbrain_full_finetune_15_v3.py'
BASE_SHA = '9f7eaec7fb4a00787512b2cde064051d7e75d03f5947e6728c03040da6ccbec3'
CONDITION = 'REVE-lastblock-finetune'
EXPECTED = {f'{CONDITION}/seed_11/fold_{fold}' for fold in range(1, 6)}
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError('Reviewed JSON/trace delegate changed')
spec = importlib.util.spec_from_file_location('_private_csbrain_v3_for_reve', BASE_PATH)
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
H = B.H
require, Reader = H.require, H.Reader
NONCODE = []


def check_source(reader, binding):
    suffix = Path(binding['path']).suffix
    if suffix in ('.py', '.json'):
        return reader.checked(binding)
    require(suffix in ('.md', '.yaml') or Path(binding['path']).name == 'LICENSE', 'Unexpected non-text implementation source')
    if binding not in NONCODE: NONCODE.append(binding)
    return None


def check_adapter(reader, metadata, contract):
    adapter, original = metadata['adapter_audit'], contract['input_contract']['adapter']
    keys = ('repository', 'revision', 'implementation_sources', 'weight_sources', 'position_bank', 'input_contract',
            'loaded_backbone_keys', 'native_dimension', 'encoder_parameters', 'trainable_parameter_names',
            'frozen_parameter_names', 'last_block_prefix', 'output_normalization', 'unused_pretrained_query_token')
    require(all(adapter[k] == original[k] for k in keys), 'REVE adapter differs from original engineering qualification')
    require(adapter['repository'] == 'brain-bzh/reve-base' and adapter['revision'] == 'dc2a075c287bb2f6c04ee5875bd79535a0f7dba6'
            and adapter['formal_pretrained_strict_loaded'] is original['formal_pretrained_strict_loaded'] is True
            and adapter['loaded_backbone_keys'] == 140 and adapter['native_dimension'] == 512
            and adapter['encoder_parameters'] == 69189632, 'Official REVE source/strict-load shape differs')
    require(len(adapter['weight_sources']) == 1 and adapter['weight_sources'][0]['sha256'] ==
            '8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe', 'REVE official weight descriptor differs')
    bank = adapter['position_bank']
    require(bank['repository'] == 'brain-bzh/reve-positions' and bank['revision'] == 'befa5b57a455b77cf302daf610c2e9ed8140bace'
            and bank['units'] == 'metres' and bank['frame'] == 'mri'
            and bank['source_is_nominal_montage_not_measured_subject_positions'] is True, 'Nominal position provenance differs')
    bank_weights = [v for v in bank['sources'] if Path(v['path']).suffix == '.safetensors']
    require(len(bank_weights) == 1 and bank_weights[0]['sha256'] ==
            '4b793820b9df0998667deb6c8ce2dbb86b38221d5165c1844bc4971941b13f13', 'Official position weight descriptor differs')
    require(all(v in contract['weight_sources'] for v in adapter['weight_sources'] + bank['sources']), 'Official provenance missing in sealed contract')
    for binding in bank['sources']:
        if Path(binding['path']).suffix != '.safetensors': check_source(reader, binding)
    require(adapter['last_block_prefix'] == 'backbone.transformer.layers.21.'
            and len(adapter['trainable_parameter_names']) == 6 and len(adapter['frozen_parameter_names']) == 134,
            'REVE last-block contract differs')


def physical(fit, metadata, boundary, traces, implementation):
    value = B.verify_traces(fit, metadata, boundary, traces, implementation)
    # A new, single complete invocation has an exact original diagnostic formula.
    if len(value['attempts']) == 1 and value['attempts'][0]['status'] == 'complete':
        n, v, epochs = len(fit['train_ids']), len(fit['val_ids']), metadata['epochs']
        expected = {'forward': epochs * (2*n+v)+2, 'encoder_forward': epochs*(n+(n+v)*8)+16,
                    'encode': epochs*(n+(n+v)*8)+16, 'encoder_windows': epochs*(n*8+(n+v)*60)+120,
                    'backward': epochs*n, 'optimizer_step': epochs*math.ceil(n/8)}
        require(all(value['recorded_physical_call_subtotals'][k+s] == count
                    for k, count in expected.items() for s in ('_started', '_returned')), 'Pristine physical call formula differs')
        value['single_complete_attempt_math_checked'] = True
        value['single_complete_attempt_expected_calls'] = expected
    return value


def semantic_functions():
    tree = ast.parse(B.DELEGATE.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('review_fit', 'verify_curve')]
    changes = Counter()
    class Adapt(ast.NodeTransformer):
        def visit_Expr(self, node):
            text = ast.unparse(node)
            if text.startswith('require(') and 'AUTHOR_SHA' in text:
                changes['adapter_check'] += 1
                return ast.copy_location(ast.Expr(ast.Call(ast.Name('check_adapter', ast.Load()),
                    [ast.Name('reader', ast.Load()), ast.Name('metadata', ast.Load()), ast.Name('contract', ast.Load())], [])), node)
            return self.generic_visit(node)
        def visit_Call(self, node):
            if ast.unparse(node) == 'reader.checked(value)':
                changes['implementation_source_read'] += 1
                return ast.copy_location(ast.Call(ast.Name('check_source', ast.Load()),
                    [ast.Name('reader', ast.Load()), ast.Name('value', ast.Load())], []), node)
            return self.generic_visit(node)
        def visit_Subscript(self, node):
            if ast.unparse(node) == "adapter['source']['weight']":
                changes['weight_descriptor'] += 1
                return ast.copy_location(ast.Subscript(ast.Name('adapter', ast.Load()), ast.Constant('weight_sources'), ast.Load()), node)
            return self.generic_visit(node)
        def visit_Constant(self, node):
            if type(node.value) is int and node.value in (20, 273):
                changes[str(node.value)] += 1
                return ast.copy_location(ast.Constant({20: 6, 273: 134}[node.value]), node)
            if isinstance(node.value, str):
                text = node.value.replace('backbone.encoder.layers.11', 'backbone.transformer.layers.21').replace('CSBrain', 'REVE')
                return ast.copy_location(ast.Constant(text), node)
            return node
    # Only review_fit has architecture-count constants; curve's last20 policy stays unchanged.
    modified = []
    for node in nodes:
        if node.name == 'review_fit': modified.append(Adapt().visit(node))
        else:
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    child.value = child.value.replace('backbone.encoder.layers.11', 'backbone.transformer.layers.21').replace('CSBrain', 'REVE')
            modified.append(node)
    require(changes == Counter({'adapter_check': 1, 'implementation_source_read': 2, 'weight_descriptor': 1, '20': 4, '273': 2}),
            'Reviewed semantic adaptation anchors differ: ' + str(changes))
    namespace = dict(H.__dict__)
    namespace.update(CONDITION=CONDITION, check_adapter=check_adapter, check_source=check_source, verify_traces=physical)
    exec(compile(ast.fix_missing_locations(ast.Module(body=modified, type_ignores=[])), str(SOURCE), 'exec'), namespace)
    B.CONDITION = CONDITION
    return namespace['review_fit']


def gate(reader):
    registry, binding = reader.snapshot(OUT / 'registration.json')
    require(binding['sha256'] == H.REGISTRY_SHA and registry['formal_fits'] == len(registry['fits']) == 360, 'Original360 differs')
    fits = [f for f in registry['fits'] if f['fit_id'] in EXPECTED]
    require(len(fits) == 5 and {f['fit_id'] for f in fits} == EXPECTED, 'Exact original seed11 five-fold identity set required')
    statuses = []
    for fit in fits:
        value, desc = reader.snapshot(OUT / 'runs' / fit['fit_id'] / 'status.json')
        require((value['fit_id'], value['condition'], value['seed'], value['fold']) ==
                (fit['fit_id'], CONDITION, 11, fit['fold']) and value['registration'] == binding, 'Current status identity differs')
        statuses.append({'fit_id': fit['fit_id'], 'status': value['status'], 'artifact': desc, 'value': value})
    return registry, binding, fits, statuses


def pending(registry, statuses, reason, audit_binding=None):
    return {'status': 'pending', 'passed': False, 'reason': reason, 'registration': registry,
            'seed': 11, 'REVE_first_seed_registered': 5,
            'completed_statuses_observed': sum(row['status'] == 'training_complete' for row in statuses),
            'statuses': [{k:v for k,v in row.items() if k != 'value'} for row in statuses],
            'actual_CPU_report_read': audit_binding is not None, 'actual_CPU_proof': audit_binding,
            'full_REVE15_or_FT135_or_plan360_completion_claimed': False}


def inventory(reader):
    formal, engineering = defaultdict(list), []
    for path in sorted((OUT / 'finetune_v2_attempts' / CONDITION).glob('*/attempt.json')):
        data, binding = reader.snapshot(path, bind=False)
        require(data['condition'] == CONDITION and data['kind'] in ('engineering', 'formal'), 'Unrecognized REVE attempt')
        if data['kind'] == 'formal' and data['seed'] != 11:
            continue  # Other seeds can progress; they are outside this stage and never rebound.
        reader.snapshot(path)
        require(data['status'] in ('complete', 'failed'), 'Unclosed REVE seed11/engineering attempt; no invented unknown closure')
        if data['kind'] == 'engineering': engineering.append((data, binding))
        else:
            identity = f'{CONDITION}/seed_{data["seed"]}/fold_{data["fold"]}'
            require(identity in EXPECTED and type(data['seed']) is type(data['fold']) is int, 'Unknown seed11 trace identity')
            formal[identity].append((data, binding))
    require(set(formal) == EXPECTED, 'Missing stage physical trace')
    return formal, engineering


def review(cpu_path=None, cpu_sha=None, check_only=False):
    NONCODE.clear()
    reader = Reader()
    registry, binding, fits, statuses = gate(reader)
    if any(row['status'] != 'training_complete' for row in statuses):
        return pending(binding, statuses, 'All five real completions required before CPU proof is read'), reader
    if check_only or cpu_path is None:
        return pending(binding, statuses, 'All five status gate reached; await supplied genuine CPU5 proof'), reader
    require(cpu_sha is not None, 'Caller must supply actual CPU report hash')
    audit, audit_binding = reader.snapshot(cpu_path)
    require(audit_binding['sha256'] == cpu_sha and audit['schema_version'] == 5
            and audit['auditor_source']['sha256'] == B.CPU_SHA, 'Actual reviewed CPU5 proof differs')
    reader.checked(audit['auditor_source'])
    for source in audit['delegated_sources'].values(): reader.checked(source)
    require(audit['registration'] == binding and audit['status'] in ('partial_verified', 'all_finetune_verified')
            and audit['model_constructions'] == audit['gpu_operations'] == 0
            and all(audit[k] is False for k in ('raw_arrays_loaded', 'outer_prediction_artifacts_read', 'test_gate_called', 'evaluation_driver_called')),
            'Actual CPU proof operation scope differs')
    shape = H.verify_cpu_evidence_shape(audit, registry)
    by_id = {row['fit_id']: row for row in audit['completed_rows']}
    if not EXPECTED.issubset(by_id):
        return pending(binding, statuses, 'Existing genuine CPU proof has not yet covered all five REVE identities', audit_binding), reader
    require(shape['verified_global_fits'] >= 35, 'Full seed11 CPU proof requires original30 plus new5')
    cohort = reader.checked(registry['sources']['cohort'])
    original = reader.checked(registry['sources']['folds'])
    copy = reader.checked(registry['sources']['folds_copy'])
    official38 = reader.checked(registry['sources']['official38_folds'])
    labels = reader.checked(registry['sources']['labels'])
    require(len(set(cohort['subject_ids'])) == 37 and registry['subject_ids'] == cohort['subject_ids']
            and registry['folds'] == original == copy, 'Original37/folds changed')
    require(original == {f:{role:[sid for sid in ids if sid != '02010004'] for role,ids in split.items()} for f,split in official38.items()}, 'Official38 exclusion changed other roles')
    require(registry['labels'] == {r['subject_id']:r['label'] for r in labels if r['subject_id'] in cohort['subject_ids']}
            and Counter(registry['labels'].values()) == {0:21, 1:16}, 'Original37 labels changed')
    require(Counter(s for split in original.values() for s in split['test']) == Counter(cohort['subject_ids']), 'Outer roles do not partition37')
    for split in original.values(): require(sum(map(len, split.values())) == len(set().union(*map(set, split.values()))) == 37, 'Overlapping roles')
    for fit in registry['fits']:
        require(all(fit[r+'_ids'] == original['fold_'+str(fit['fold'])][r] for r in ('train','val','test')), 'Registered condition changed original folds')
    formal, engineering = inventory(reader)
    B.ACTIVE['reader'] = reader
    review_fit = semantic_functions()
    statuses_by_id = {row['fit_id']:row['value'] for row in statuses}
    rows = [review_fit(reader, fit, by_id[fit['fit_id']], registry, binding, statuses_by_id[fit['fit_id']], formal[fit['fit_id']]) for fit in fits]
    require(len({row['initial_full_state_sha256'] for row in rows}) == len({row['initial_encoder_state_sha256'] for row in rows}) ==
            len({row['frozen_initial_state_sha256'] for row in rows}) == 1, 'Same-seed fresh initializers differ')
    lock = by_id[fits[0]['fit_id']]['condition_source_lock']
    require(all(by_id[f['fit_id']]['condition_source_lock'] == lock for f in fits), 'Shared REVE source lock differs')
    contract = reader.checked(lock)['contract']
    engineering_cost = H.engineering_summary(reader, engineering, contract, contract['versioned_finetune_implementation'], original)
    totals = Counter()
    for row in rows: totals.update(row['physical_attempts']['recorded_physical_call_subtotals'])
    formal_after, engineering_after = inventory(reader)
    require({k:[b for _,b in v] for k,v in formal.items()} == {k:[b for _,b in v] for k,v in formal_after.items()}
            and [b for _,b in engineering] == [b for _,b in engineering_after], 'Stage attempt inventory changed')
    result = {'status':'verified_REVE_first_seed5_only', 'passed':True, 'registration':binding, 'actual_CPU_proof':audit_binding,
        'actual_CPU_source':audit['auditor_source'], 'actual_CPU_snapshot_global_fits':shape['verified_global_fits'],
        'actual_CPU_snapshot_global_loads':shape['checkpoint_loads'], 'actual_CPU_snapshot_pending_FT':shape['pending_global_fits'],
        'scope':'REVE seed11 original five folds only, backed by existing actual CPU5; other seeds/architectures not independently re-reviewed here.',
        'seed':11, 'REVE_fits_reviewed':5, 'REVE_registered':15, 'all_FT_registered':135,
        'full_REVE15_or_FT135_or_plan360_completion_claimed':False, 'cohort_people':37,
        'fresh_same_seed_initializers_equal':True, 'three_seed_differences_claimed':False,
        'fits':rows, 'epochs':[r['epochs'] for r in rows], 'best_epochs':[r['best_epoch'] for r in rows],
        'selected_phase_counts':dict(Counter(r['selected_phase'] for r in rows)),
        'extensions':[r['fit_id'] for r in rows if r['extended']],
        'selection_policy':'Validation subject-mean CE; minimum improvement1e-4; warmup1..10 eligible; earliest stop40 with patience30; extend at120 once only if best>96 and last20 relative improvement>=1%; cap200.',
        'parameter_scope':{'encoder_trainable':6,'head_trainable':6,'frozen':134,'positions':'author nominal montage, not subject measured'},
        'durable_optimizer_updates':sum(r['durable_optimizer_updates'] for r in rows),
        'durable_backward_calls':sum(r['durable_backward_calls'] for r in rows),
        'head_group_updates':sum(r['head_group_updates'] for r in rows), 'backbone_group_updates':sum(r['backbone_group_updates'] for r in rows),
        'group_updates_not_added_as_independent_optimizer_calls':True,
        'physical_recorded_call_subtotals':dict(totals),
        'formal_attempt_seconds_known_sum':sum(r['physical_attempts']['recorded_attempt_seconds_known_sum'] for r in rows),
        'nested_metadata_seconds_not_added':sum(r['last_success_metadata_seconds_not_added_to_attempt_seconds'] for r in rows),
        'peak_allocated_bytes_max':max(r['peak_allocated_bytes'] for r in rows), 'engineering_separate':engineering_cost,
        'unknown_elapsed_attempts':[a for r in rows for a in r['physical_attempts']['unknown_elapsed_attempts']],
        'all_historical_exact_wall_seconds':None, 'all_historical_exact_physical_counts':None,
        'physical_durable_and_engineering_scopes_not_added':True,
        'non_JSON_python_text_source_descriptors_not_rehashed_this_review':list(NONCODE),
        'non_rehashed_descriptor_basis':'Compared with source-bound actual CPU5 metadata/engineering lock. Official binary weights and position bank descriptors also carried without reads.',
        'no_sync_optimization_or_architecture_speedup_claim':True}
    reader.recheck()
    return result, reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-audit', type=Path); parser.add_argument('--cpu-sha')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    value, reader = review(args.cpu_audit, args.cpu_sha, args.check_only)
    for path in (SOURCE, BASE_PATH, B.DELEGATE): reader.snapshot(path)
    require('torch' not in sys.modules, 'Tensor library unexpectedly imported')
    value.update(schema='REVE_first_seed5_JSON_stage_review_v1', source=reader.snapshot(SOURCE)[1],
        created_at=datetime.now(timezone.utc).isoformat(), semantic_delegate=reader.snapshot(B.DELEGATE)[1],
        physical_delegate=reader.snapshot(BASE_PATH)[1], input_bindings=list(reader.inputs.values()),
        operations={'binary_reads':0,'tensor_loads':0,'models_GPU_process_actions':0,'raw_features_test_probabilities':0,
                    'X_L_registry_verifier_executions':0,'sealed_source_changes':False})
    target = REPORT / ('REVE_first_seed5_' + ('passed_' if value['passed'] else 'pending_') + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'artifact':reader.snapshot(target)[1], 'passed':value['passed'], 'status':value['status']}))


if __name__ == '__main__': main()
