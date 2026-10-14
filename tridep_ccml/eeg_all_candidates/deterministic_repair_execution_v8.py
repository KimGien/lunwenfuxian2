"""Read-only actual repair qualification/contract verifier; no tensor imports.

The registered original15 identities, sampler/budgets/selection, micro1,
logical8/chunk8/FP32 remain fixed. Changed backward implementation and Mamba
accumulation order are explicit. Old source/fit records are never rewritten.
Source and engineering-checkpoint bytes may be hashed; this is not JSON-only IO.
Registration creation belongs to a separately reviewed future queue entry.
"""
from __future__ import annotations
import ast
import argparse
import copy
import json
from pathlib import Path
from . import storage as S
from . import qualify_deterministic_repairs_v9 as Q
from . import execution_acceleration_v1 as X
from . import qualify_deterministic_repairs_v5 as PREVIOUS_Q
SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'deterministic_repair_execution_v8'
SCHEMA = 'eeg_finetune_deterministic_repair_execution_v8'
QUALIFIER_SHA = '0c66d96313760334802e86da99771e14cf468f68f2908aa62f783ab4fa74f559'
F3_SHA = '13601bd5f78b5c543bbc9d6c4dcb856fd9e72a6d09c55d18a5a38febeb670e59'
REPAIR_CONDITIONS = Q.CONDITIONS
NEW_ENTRY_NAMES = ('finetune_v11.py', 'continuation_queue_v13.py')
NEW_ANALYSIS_NAMES = ('audit_completed_finetune_checkpoints_v15.py', 'cost_aggregation_v16.py')
FLUSH_BOUNDARIES = ('subject_update_before', 'subject_update_after', 'diagnostics_before', 'diagnostics_after', 'latest_checkpoint_committed', 'fit_before', 'fit_returned', 'exception')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_bound(binding):
    return S.read(S.checked(binding))

def validate_report_data(value, condition, fit):
    require(condition in REPAIR_CONDITIONS and value.get('condition') == condition and (value.get('status') == 'engineering_verified') and (value.get('kind') == 'deterministic_repair_engineering'), 'Actual condition engineering success required')
    require(value.get('GPU_engineering_checks_passed') is True and value.get('new_formal_execution_required') is True and (value.get('formal_fits') == 0) and (value.get('native_CUDA_gradient_equivalence_established') is False) and (value.get('performance_improvement_claimed') is False) and (value.get('accumulation_order_changed') is (condition == REPAIR_CONDITIONS[1])), 'Actual GPU checks/changed numeric order/future scope must be explicit')
    require(value.get('outer_test_read') is False and value.get('original_native_forward_bitwise_equal') is True and (value.get('zero_step_optimizer_and_rng_resume_verified') is True), 'GPU forward, zero-step resume or test-role proof missing')
    for kind, total in (('forward', 6), ('backward', 4), ('encoder_forward', 6), ('encoder_windows', 48), ('optimizer_step', 0)):
        require(all((type(value.get(kind + '_' + suffix)) is int and value[kind + '_' + suffix] == total for suffix in ('started', 'returned'))), 'Actual observed engineering count differs: ' + kind)
    require(type(value.get('primary_engineering_seconds')) in (int, float) and value['primary_engineering_seconds'] > 0, 'Actual engineering elapsed measurement required')
    person = fit['train_ids'][0]
    require(value.get('probe_person') == person and person not in fit['val_ids'] + fit['test_ids'] and (set(value.get('raw_sources', {})) == {person}), 'Fixed training-only qualification role differs')
    contract = value['runtime_contract']
    input_contract = contract['input_contract']
    require(all((input_contract.get(k) == v for k, v in {'micro_subjects': 1, 'window_chunk': 8, 'effective_subjects': 8, 'windows_per_training_subject': 8, 'all_window_validation': 60, 'coverage_cycle_epochs': 8, 'precision': 'float32'}.items())), 'Original training/input constants differ')
    selected = contract['trainable_parameter_names']
    frozen = contract['frozen_parameter_names']
    require(selected and len(selected) == len(set(selected)) and (len(frozen) == len(set(frozen))) and (not set(selected) & set(frozen)) and any((n.startswith('encoder.') for n in selected)) and any((n.startswith('head.') for n in selected)), 'Exact parameter partition missing')
    phases = value['checks']
    require(len(phases) == 2 and [x['epoch_phase'] for x in phases] == [1, 11] and all((x['repeat_bitwise_equal'] is True for x in phases)), 'Actual warm/lastblock repeated gradient checks missing')
    heads = [n for n in selected if n.startswith('head.')]
    require(set(phases[0]['gradient_parameter_names']) == set(heads) and set(phases[1]['gradient_parameter_names']) == set(selected), 'Warm/lastblock parameter scopes differ')
    require(all((contract.get(k) is True for k in ('forward_checked', 'gradient_checked', 'resume_checked', 'training_only_engineering_checked', 'frozen_buffers_and_parameters_zero_update_verified', 'separate_new_formal_execution_and_auditor_required'))), 'Qualification contract incomplete')
    require(contract.get('engineering_backward_calls') == 4 and contract.get('engineering_optimizer_updates') == 0 and (contract.get('probe_subject_ids') == [person]) and (contract['raw_sources'] == value['raw_sources']) and (contract.get('native_CUDA_gradient_equivalence_established') is False) and (contract.get('accumulation_order_changed') is (condition == REPAIR_CONDITIONS[1])), 'Qualification contract or numerical disclosure differs')
    return contract

def qualification(binding, condition):
    require(condition == REPAIR_CONDITIONS[0], 'This partial execution admits only the genuinely qualified CodeBrain15')
    provider = PREVIOUS_Q if condition == REPAIR_CONDITIONS[0] else Q
    provider_sha = 'f038cdace54f7ce7ff2a3286b3d948726fb065cb08b39ab28464b152e5c5f0f2' if condition == REPAIR_CONDITIONS[0] else QUALIFIER_SHA

    def builder_ast(module):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'isolated_builder'))
        # Only Mamba's private adapter generation changes. CodeBrain mapping,
        # dispatch and builder bytecode must remain the qualified Qual5 path.
        mapping = next((node.value for node in function.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'mapping' for t in node.targets)), None)
        require(isinstance(mapping, ast.Dict), 'Exact private builder mapping required')
        values = ast.literal_eval(mapping)
        expected_mamba = '.adapters.eegmamba_v4' if module is Q else '.adapters.eegmamba_v2'
        require(values == {'.adapters.codebrain_v2': '.adapters.codebrain_v3',
                           '.adapters.eegmamba': expected_mamba}, 'Unknown condition routing change')
        if module is Q:
            found = [value for key, value in zip(mapping.keys, mapping.values)
                     if isinstance(key, ast.Constant) and key.value == '.adapters.eegmamba']
            require(len(found) == 1 and isinstance(found[0], ast.Constant), 'One Mamba route required')
            found[0].value = '.adapters.eegmamba_v2'
        return ast.dump(function, include_attributes=False)
    require(builder_ast(Q) == builder_ast(PREVIOUS_Q), 'CodeBrain builder AST or unexpected routing changed; old qualification cannot be reused')
    require(S.desc(Path(provider.__file__))['sha256'] == provider_sha, 'Reviewed qualifier source changed')
    path = S.checked(binding)
    require(path.is_relative_to((provider.ROOT / condition).resolve()) and path.name == 'report.json', 'Exact condition report path required')
    value = read_bound(binding)
    folder = path.parent
    require(value.get('source') == S.desc(Path(provider.__file__)), 'Actual qualified entry source differs')
    require(S.read(folder / 'attempt.json') == value, 'Actual completed worker attempt/report differs')
    controller = S.read(folder / 'controller.json')
    require(controller.get('returncode') == 0 and controller.get('status') == 'worker_returned' and (controller.get('condition') == condition), 'Actual controller/worker return required')
    released = S.read(folder / 'released_owner.json')
    require(released.get('condition') == condition and released['pid'] in {x['pid'] for x in controller['admission']['allowed_host']}, 'Exact released controller owner proof missing')
    intent = read_bound(released['intent'])
    require(intent['folder'] == folder.relative_to(S.WORK).as_posix() and intent['condition'] == condition and (intent['source'] == value['source']), 'Permanent intent binds another actual attempt')
    admission = read_bound(value['admission'])
    require(admission == controller['admission'] and admission['source'] == value['source'] and (admission['condition'] == condition) and (admission['GPU_admission']['context_count'] == 0), 'Actual worker admission differs')
    require(value['admission'] == S.desc(folder / 'admission.json'), 'Admission bytes changed')
    provider.source_check()
    require(admission['terminal'] == S.desc(provider.QUEUE / 'status.json') and (not (provider.QUEUE / 'owner.json').exists()), 'Old Q5 terminal not released or changed')
    provider.check_boundary()
    complete = admission.get('original330_complete_proof')
    require(type(complete) is dict and complete.get('source') == value['source'] and (complete.get('registry') == S.desc(S.OUT / 'registration.json')) and (complete.get('recovery_registration') == S.desc(provider.QUEUE / 'recovery_registration_v3.json')) and (complete.get('formal_fits_verified') == 330), 'Actual source-bound full330 completion proof required')
    rows = complete['rows']
    require(len(rows) == 330 and len({r['fit_id'] for r in rows}) == 330, 'Full330 unique completion rows required')
    for row in rows:
        current = read_bound(row['status'])
        read_bound(row['complete'])
        require(current.get('fit_id') == row['fit_id'] and current.get('status') == 'training_complete' and (current.get('evidence') == row['evidence']), 'Original completion changed after qualification')
    registry = S.read(S.OUT / 'registration.json')
    fits = [x for x in registry['fits'] if x['condition'] == condition]
    require(len(fits) == 15, 'Original15 formal identities required')
    fit = next((x for x in fits if x['seed'] == 11 and x['fold'] == 1))
    contract = validate_report_data(value, condition, fit)
    require(contract['qualification_source'] == value['source'], 'Contract qualifier differs')
    for record in contract['implementation_sources'] + contract['weight_sources']:
        S.checked(record)
    require(contract['weight_sources'], 'Actual official pretrained resources required')
    S.checked(value['resume_probe'])
    require(value['raw_sources'][value['probe_person']] == next((x['raw'] for x in registry['raw_sources'] if x['subject_id'] == value['probe_person'])), 'Raw-source descriptor differs from immutable registration')
    return {'report': binding, 'contract': contract, 'observed_engineering': value, 'original_fit_ids': [x['fit_id'] for x in fits], 'controller': S.desc(folder / 'controller.json'), 'released_owner': S.desc(folder / 'released_owner.json'), 'worker_owner': S.desc(folder / 'worker_owner.json'), 'intent': released['intent'], 'successful_qualification_reused_without_new_model_work': condition == REPAIR_CONDITIONS[0]}

def implementation_sources():
    prior_binding = {'path': 'output/analysis/eeg_all_candidates_v1_20261008/independent_full_module_compile_repair_chain_source_review_1791783895805780100.json', 'sha256': '40a39257cf551b776c90843f8ed3778e13a7184dd3a106f1f0fe38d220ee9ea9'}
    prior = read_bound(prior_binding)
    require(prior.get('passed') is True and len(prior['reviewed_sources']) == 166, 'Previous166 reviewed source lineage required')
    records = {x['path']: x for x in prior['reviewed_sources']}
    records[prior_binding['path']] = prior_binding
    for name in NEW_ENTRY_NAMES:
        path = SOURCE.with_name(name)
        require(path.exists(), 'New formal entry not prepared: ' + name)
        record = S.desc(path)
        records[record['path']] = record
    for name in NEW_ANALYSIS_NAMES:
        path = S.REPORT / name
        require(path.exists(), 'New audit/cost entry not prepared: ' + name)
        record = S.desc(path)
        records[record['path']] = record
    for path in (SOURCE, Path(Q.__file__)):
        record = S.desc(path)
        records[record['path']] = record
    for path in Q.PINS:
        record = S.desc(S.WORK / path)
        records[record['path']] = record
    require(S.desc(SOURCE.with_name('finetune_v3.py'))['sha256'] == F3_SHA, 'Original buffered observer changed')
    for record in records.values():
        S.checked(record)
    return list(records.values())

def lock_contract(proof, registration):
    contract = copy.deepcopy(proof['contract'])
    sources = {x['path']: x for x in contract['implementation_sources']}
    for record in read_bound(registration)['sources']:
        sources[record['path']] = record
    contract.update(implementation_sources=list(sources.values()), versioned_finetune_implementation=X.f2_provenance(), deterministic_backward_repair={'registration': registration, 'qualification': proof['report'], 'contract_verifier_source': S.desc(SOURCE), 'actual_formal_source': S.desc(SOURCE.with_name('finetune_v11.py')), 'pipeline_sampler_warmup_optimizer_stop_and_resume_unchanged': True, 'original_backward_bitwise_equivalence_claimed': False, 'accumulation_order_changed': proof['contract']['accumulation_order_changed']})
    return contract

def load_execution(binding, *, condition, seed, fold):
    require(condition == REPAIR_CONDITIONS[0], 'Mamba has no qualified partial execution; evaluation remains prohibited')
    require(condition in REPAIR_CONDITIONS and type(seed) is int and (seed in (11, 22, 33)) and (type(fold) is int) and (fold in (1, 2, 3, 4, 5)), 'Original strict15 fit identity required')
    expected = ROOT / condition / 'registration.json'
    require(S.checked(binding) == expected.resolve(), 'Actual condition repair registration required')
    value = read_bound(binding)
    require(value.get('schema') == SCHEMA and value.get('version') == 8 and (value.get('status') == 'qualified') and (value.get('condition') == condition) and (value.get('registry') == S.desc(S.OUT / 'registration.json')), 'Qualified new execution registration differs')
    require(value.get('sources') == implementation_sources(), 'Complete new formal/audit/cost source binding differs')
    proof = qualification(value['qualification'], condition)
    require(value['original_fit_ids'] == proof['original_fit_ids'], 'New registration changed original15 fit identities')
    review = read_bound(value['independent_engineering_review'])
    require(review.get('passed') is True and review.get('qualification_report') == proof['report'] and (review.get('controller') == proof['controller']) and (review.get('released_owner') == proof['released_owner']) and (review.get('numerical_order_change_acknowledged') is proof['contract']['accumulation_order_changed']), 'Actual source-bound independent engineering review required')
    S.checked(review['reviewer_source'])
    from . import registry as R
    lock = R._condition_lock(condition)
    require(S.read(S.checked(lock))['contract'] == lock_contract(proof, binding), 'New exact engineering source contract differs')
    return {'schema': SCHEMA, 'version': 11, 'observer_version': 11, 'source': S.desc(SOURCE.with_name('finetune_v11.py')), 'delegated_finetune_implementation': X.f2_provenance(), 'registration': binding, 'condition_source_lock': lock, 'condition': condition, 'seed': seed, 'fold': fold, 'qualification': proof['report'], 'independent_engineering_review': value['independent_engineering_review'], 'deterministic_backward_repair': {'applied': True, 'accumulation_order_changed': proof['contract']['accumulation_order_changed'], 'original_CUDA_backward_equivalence_established': False, 'performance_improvement_claimed': False}, 'diagnostics': {'window_chunk': 8, 'qualification': None, 'bitwise_equivalence_required': True}, 'training': {'micro_subjects': 1, 'window_chunk': 8, 'effective_subject_batch': 8, 'window_bag_size': 8, 'precision': 'float32', 'pipeline_sampler_budgets_selection_unchanged': True}, 'observer': {'per_call_persistence': False, 'flush_boundaries': list(FLUSH_BOUNDARIES), 'hard_kill_unflushed_tail': 'unknown'}, 'new_actual_observer_version': 11, 'original_CUDA_backward_equivalence_established': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps({'status': 'preparation_only', 'model_calls': 0, 'GPU_calls': 0, 'checkpoint_tensor_loads': 0, 'registration_or_fit_writes': 0, 'actual_formal_execution_registered': False}))
if __name__ == '__main__':
    main()
