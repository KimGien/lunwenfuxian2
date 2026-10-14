"""Independent five-fit source/JSON checks, never project or tensor execution."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from statistics import mean
import hashlib
import json
import math
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
REPORT = SOURCE.parent
INPUTS = {}


def read(path, sha=None):
    path = (WORK / path).resolve()
    assert path.is_relative_to(WORK) and path.suffix in ('.py', '.json')
    raw = path.read_bytes()
    binding = {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert sha is None or binding['sha256'] == sha
    assert binding['path'] not in INPUTS or INPUTS[binding['path']] == binding
    INPUTS[binding['path']] = binding
    return (json.loads(raw.decode('utf-8-sig')) if path.suffix == '.json' else raw.decode('utf-8-sig')), binding


def checked(binding):
    return read(binding['path'], binding['sha256'])[0]


def decision(policy, losses, extended):
    best, best_loss = 1, losses[0]
    for epoch, loss in enumerate(losses[1:], 2):
        if best_loss - loss >= policy['minimum_improvement']:
            best, best_loss = epoch, loss
    epoch = len(losses)
    cap = policy['final_epochs'] if extended else policy['initial_epochs']
    assert epoch <= cap
    result = {'epoch': epoch, 'best_epoch': best, 'stop': False, 'extend': False, 'reason': 'continue'}
    if epoch >= policy['minimum_stop_epoch'] and epoch - best >= policy['patience']:
        result.update(stop=True, reason='validation_patience')
    elif epoch == cap:
        previous, recent = mean(losses[-40:-20]), mean(losses[-20:])
        relative = (previous - recent) / previous if previous > 0 else 0.
        eligible = not extended and best > .8 * policy['initial_epochs'] and relative >= .01 and policy['final_epochs'] > policy['initial_epochs']
        result.update(stop=not eligible, extend=eligible, reason='registered_extension' if eligible else 'epoch_cap',
                      previous20_mean=previous, recent20_mean=recent, relative_improvement=relative)
    return result


def lr(policy, epoch, group):
    start = policy['warmup_epochs'] if group == 'backbone' else 0
    warm = policy['backbone_warmup_epochs'] if start else policy['warmup_epochs']
    if epoch <= start: factor = 0.
    elif epoch <= start + warm: factor = (epoch - start) / warm
    elif epoch >= policy['initial_epochs']: factor = .1
    else: factor = .1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - start - warm) / (policy['initial_epochs'] - start - warm)))
    return policy[group + '_lr'] * factor


def main():
    _, own = read(SOURCE)
    stage, stage_binding = read(REPORT / 'REVE_first_seed5_passed_1791606547152873800.json', 'afbf52aa09aa1161525b80e97088d912ff170decc5e7fc13b953b2876311338b')
    source = checked(stage['source'])
    assert stage['source']['sha256'] == 'b5b32889a76bef07b96d43d347f48e6b6a705a768f187c42a56d5fab5a9cfdcc'
    checked(stage['semantic_delegate']); checked(stage['physical_delegate'])
    tests, tests_binding = read(REPORT / 'REVE_first_seed5_guard_tests_1791606600913999400.json', '6df9f0afc56c17b75cbf6c2dca857ac8615bf13f50cf92e3fde7763f4cc6c81c')
    assert tests['target'] == stage['source'] and tests['passed'] is True and tests['tests_run'] == 8
    assert tests['failures'] == tests['errors'] == 0 and tests['torch_imported'] is False
    checked(tests['source'])
    cpu = checked(stage['actual_CPU_proof']); checked(stage['actual_CPU_source'])
    assert stage['actual_CPU_proof']['sha256'] == '7dc8c0fb82fd60b218e41889bc116f145f24bad84ac815b3d59b33e1adbed72c'
    assert cpu['schema_version'] == 5 and cpu['verified_completed_finetune_fits'] == 35 and cpu['checkpoint_loads'] == 105
    assert cpu['training_complete_finetune_observed'] == 35 and not cpu['errors'] and not cpu['deferred_completed_fits']
    assert cpu['auditor_source'] == stage['actual_CPU_source'] and stage['actual_CPU_source']['sha256'] == '58b101d301f92da4283890e0a4fa80bef40cdb38eeace1001d6fa18e583a3eba'
    registry = checked(stage['registration'])
    _, protocol = read(WORK / 'tridep_ccml/eeg_all_candidates/protocol.py', '7449f21e82e5a81c71f46fe7ac98681f1308c670dab4598750fbd0192e88d114')
    _, adapter_source = read(WORK / 'tridep_ccml/eeg_all_candidates/adapters/reve.py', '24e33d12ca6700ad82c89182b87d941e56ae0a65683e640ed0828e64ac726146')
    original = checked(registry['sources']['folds'])
    assert original == registry['folds'] == checked(registry['sources']['folds_copy'])
    expected_ids = {f'REVE-lastblock-finetune/seed_11/fold_{f}' for f in range(1, 6)}
    fits = {f['fit_id']: f for f in registry['fits'] if f['fit_id'] in expected_ids}
    assert len(stage['fits']) == 5 and {r['fit_id'] for r in stage['fits']} == expected_ids
    cpu_rows = {r['fit_id']: r for r in cpu['completed_rows']}
    traces, notes, total, phases = [], [], Counter(), Counter()
    initials, encoder_initials, frozen_initials = set(), set(), set()
    for row in stage['fits']:
        fit, cr = fits[row['fit_id']], cpu_rows[row['fit_id']]
        metadata = checked(row['metadata']); complete = checked(row['completion'])
        assert row['metadata'] == cr['metadata'] == complete['metadata'] and row['completion'] == cr['completion']
        status = checked(cr['status_artifact'])
        assert status['status'] == 'training_complete' and status['evidence'] == complete and status['test_evaluation_status'] == 'frozen'
        assert checked(status['event_record']) == {k:v for k,v in status.items() if k != 'event_record'}
        assert metadata['condition_source_lock'] == cr['condition_source_lock'] == status['condition_source_lock']
        lock = checked(cr['condition_source_lock']); contract = lock['contract']
        assert lock['registration'] == complete['registry'] == stage['registration']
        assert metadata['fresh_head'] is metadata['fresh_official_pretrained'] is True and metadata['frozen_best_head_inherited'] is False
        assert metadata['initial_state_sha256'] == contract['initial_state_sha256']
        initials.add(metadata['initial_state_sha256']); encoder_initials.add(metadata['selected_encoder_initial_state_sha256']); frozen_initials.add(metadata['frozen_initial_sha256'])
        for role in ('train', 'val', 'test'): assert metadata[role + '_ids'] == fit[role + '_ids'] == original['fold_' + str(fit['fold'])][role]
        assert set(metadata['raw_sources']) == set(fit['train_ids'] + fit['val_ids']) and not set(metadata['raw_sources']) & set(fit['test_ids'])
        assert metadata['test_forward_count'] == 0 and metadata['reload_probe_subject_ids'] == fit['train_ids'][:1]
        adapter, qualified = metadata['adapter_audit'], contract['input_contract']['adapter']
        for key in ('repository','revision','implementation_sources','weight_sources','position_bank','input_contract','loaded_backbone_keys','native_dimension','trainable_parameter_names','frozen_parameter_names'):
            assert adapter[key] == qualified[key]
        assert adapter['formal_pretrained_strict_loaded'] is True and adapter['loaded_backbone_keys'] == 140 and adapter['native_dimension'] == 512
        assert adapter['weight_sources'][0]['sha256'] == '8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe'
        assert adapter['position_bank']['source_is_nominal_montage_not_measured_subject_positions'] is True
        enc = [x for x in contract['trainable_parameter_names'] if x.startswith('encoder.')]
        heads = [x for x in contract['trainable_parameter_names'] if x.startswith('head.')]
        assert len(enc) == len(heads) == 6 and len(contract['frozen_parameter_names']) == 134
        assert all(x.startswith('encoder.backbone.transformer.layers.21.') for x in enc)
        for key in ('wrapper_source','delegated_runner_source','luna_adapter_source'): checked(metadata['versioned_finetune_implementation'][key])
        checked(metadata['execution_acceleration']['source'])
        assert metadata['execution_acceleration']['diagnostics']['window_chunk'] == 8 and metadata['execution_acceleration']['diagnostics']['qualification'] is None
        history, validation = checked(complete['train_history']), checked(complete['validation_history'])
        epoch_count, policy = metadata['epochs'], fit['policy']
        assert len(history) == len(validation) == epoch_count
        assert (policy['initial_epochs'],policy['minimum_stop_epoch'],policy['patience'],policy['final_epochs']) == (120,40,30,200)
        assert policy['minimum_improvement'] == .0001 and policy['warmup_epochs'] == 10 and policy['backbone_warmup_epochs'] == 5
        losses, extension, extended = [], None, False
        n, v = len(fit['train_ids']), len(fit['val_ids'])
        for epoch, (train, val) in enumerate(zip(history, validation), 1):
            assert train['epoch'] == val['epoch'] == epoch and train['val_eval_ce'] == val['val_ce']
            assert type(val['val_ce']) in (int,float) and math.isfinite(val['val_ce']) and val['val_ce'] >= 0
            assert (train['optimizer_updates'],train['backward_calls']) == (math.ceil(n/8)*epoch,n*epoch)
            assert train['unfrozen'] is (epoch>10) and train['training_module_subtrees'] == (['backbone.transformer.layers.21'] if epoch>10 else [])
            assert train['window_indices'] == [(8*((epoch-1)%8)+j)%60 for j in range(8)]
            assert train['cycle'] == (epoch-1)//8+1 and train['cycle_epoch'] == (epoch-1)%8+1
            assert train['repeated_padding_windows_this_epoch'] == ([0,1,2,3] if epoch%8 == 0 else [])
            for group in ('head','backbone'): assert math.isclose(train[group+'_lr'],lr(policy,epoch,group),rel_tol=1e-12,abs_tol=1e-16)
            losses.append(val['val_ce']); verdict = decision(policy,losses,extended)
            assert verdict['stop'] is (epoch==epoch_count)
            if verdict['extend']:
                assert not extended and epoch == 120 and extension is None
                extended = True; extension = verdict
        assert (verdict['best_epoch'],verdict['reason'],extended) == (metadata['best_epoch'],metadata['stop_reason'],metadata['extended'])
        assert (row['epochs'],row['best_epoch'],row['extended']) == (epoch_count,metadata['best_epoch'],extended)
        phase = 'head_only_warmup' if metadata['best_epoch']<=10 else 'lastblock_finetune'
        assert row['selected_phase'] == metadata['selected_checkpoint_phase'] == phase
        assert metadata['terminal_encoder_changed_parameter_names'] == enc and metadata['terminal_head_changed_parameter_names'] == heads
        assert metadata['selected_head_changed_parameter_count'] == 6 and metadata['terminal_encoder_changed_parameter_count'] == 6
        assert metadata['selected_encoder_changed_parameter_count'] == (0 if phase=='head_only_warmup' else 6)
        if phase=='head_only_warmup': assert metadata['selected_encoder_state_sha256'] == metadata['selected_encoder_initial_state_sha256']
        assert metadata['frozen_initial_sha256'] == metadata['frozen_final_sha256']
        assert metadata['terminal_updates_are_not_claimed_as_selected_encoder_updates'] is True
        assert cr['passed'] is cr['training_json_verified'] is True and cr['cpu_checkpoint_audit']['checkpoint_loads'] == 3
        proof = cr['cpu_checkpoint_audit']; assert complete['checkpoint'] in proof['checkpoints']
        bounds = {x['path']:x for x in proof['durable_counter_boundaries']}
        assert bounds['latest.pt']['epoch'] == epoch_count and bounds['best.pt']['epoch'] == metadata['best_epoch']
        assert bounds['latest.pt']['optimizer_groups']['serialized_id_parameter_names'] == heads + enc
        assert bounds['latest.pt']['optimizer_groups']['moment_shapes_verified'] is True
        physical = row['physical_attempts']; assert len(physical['attempts']) == 1
        item = physical['attempts'][0]; attempt = checked(item['artifact'])
        assert attempt['status'] == 'complete' and attempt['seed'] == 11 and attempt['fold'] == fit['fold']
        assert item['zero_model_or_training_call_recheck'] is False
        assert attempt['execution_acceleration'] == metadata['execution_acceleration']
        assert attempt['graceful_terminal_flush'] is True and attempt['counter_snapshot_exact_for_process'] is True
        assert attempt['unflushed_tail_unknown'] is False and all(type(x) is int and x==0 for x in attempt['unflushed_tail_counters'].values())
        expected = {'forward':epoch_count*(2*n+v)+2,'encoder_forward':epoch_count*(n+(n+v)*8)+16,'encode':epoch_count*(n+(n+v)*8)+16,
                    'encoder_windows':epoch_count*(n*8+(n+v)*60)+120,'backward':epoch_count*n,'optimizer_step':epoch_count*math.ceil(n/8)}
        assert physical['single_complete_attempt_math_checked'] is True and physical['single_complete_attempt_expected_calls'] == expected
        for key,count in expected.items():
            for suffix in ('_started','_returned'):
                assert type(attempt[key+suffix]) is int and attempt[key+suffix] == physical['recorded_physical_call_subtotals'][key+suffix] == count
        assert physical['recorded_physical_call_subtotals'] == cr['f2_physical_durable_boundary']['recorded_physical_call_subtotals']
        assert attempt['seconds'] == item['seconds'] >= metadata['seconds_this_attempt'] > 0
        assert metadata['head_optimizer_updates'] == row['head_group_updates'] == expected['optimizer_step']
        assert metadata['backbone_optimizer_updates'] == row['backbone_group_updates'] == math.ceil(n/8)*(epoch_count-10)
        assert metadata['backward_calls'] == row['durable_backward_calls'] == expected['backward']
        total.update(physical['recorded_physical_call_subtotals']); phases[phase]+=1; traces.append(item)
        notes.append({'fold':fit['fold'],'epochs':epoch_count,'best_epoch':metadata['best_epoch'],'stop_reason':verdict['reason'],
            'extension120':extension,'selected_phase':phase,'selected_encoder_changed':metadata['selected_encoder_changed_parameter_count'],
            'terminal_encoder_changed':6,'selected_and_terminal_encoder_sha_equal':metadata['selected_encoder_state_sha256']==metadata['terminal_encoder_state_sha256'],
            'pristine_expected_calls':expected,'all_epoch_policy_LR_sampler_checks_passed':True})
    assert len(initials) == len(encoder_initials) == len(frozen_initials) == 1
    assert dict(phases) == stage['selected_phase_counts'] == {'lastblock_finetune':4,'head_only_warmup':1}
    assert dict(total) == stage['physical_recorded_call_subtotals']
    assert total['optimizer_step_returned'] == stage['durable_optimizer_updates'] == stage['head_group_updates'] == 2324
    assert total['backward_returned'] == stage['durable_backward_calls'] == 15447
    assert stage['backbone_group_updates'] == 2124
    assert math.isclose(sum(x['seconds'] for x in traces),stage['formal_attempt_seconds_known_sum'],rel_tol=1e-12)
    assert not stage['unknown_elapsed_attempts'] and stage['all_historical_exact_wall_seconds'] is stage['all_historical_exact_physical_counts'] is None
    engineering = stage['engineering_separate']; assert len(engineering['unique_attempts']) == 1
    item = engineering['unique_attempts'][0]; record = checked(item['artifact'])
    assert record['condition'] == 'REVE-lastblock-finetune' and record['kind'] == 'engineering' and record['status'] == 'complete'
    assert record['seed'] is record['fold'] is None
    assert record['seconds'] == item['seconds'] == engineering['known_seconds_separate_from_formal'] == 24.18075930001214
    assert record['backward_started'] == record['backward_returned'] == 2
    assert record['optimizer_step_started'] == record['optimizer_step_returned'] == 0
    assert contract['engineering_attempt_path'] == item['artifact']['path'] and contract['engineering_optimizer_updates'] == 0 and contract['engineering_backward_calls'] == 2
    assert contract['probe_subject_ids'] == original['fold_1']['train'][:1]
    assert stage['full_REVE15_or_FT135_or_plan360_completion_claimed'] is stage['three_seed_differences_claimed'] is False
    for binding in list(INPUTS.values()): checked(binding)
    result = {'schema':'independent_REVE_first_seed5_JSON_review_v1','created_at':datetime.now(timezone.utc).isoformat(),
        'source':own,'passed':True,'reviewed_stage':stage_binding,'existing_CPU_proof':stage['actual_CPU_proof'],
        'independent_stdlib_tests':tests_binding,'scope':'Five original REVE seed11 folds only; existing CPU snapshot35/135 and105 loads. No tensor proof rerun.',
        'source_review':{'helper':stage['source'],'semantic_delegate':stage['semantic_delegate'],'physical_delegate':stage['physical_delegate'],
            'protocol':protocol,'adapter':adapter_source,'no_definite_blockers':True,
            'findings':['Explicit AST architecture substitutions preserve original policy decision/LR functions and leave old CSBrain CPU superset identity gate unchanged.',
                        'Pending status gate precedes CPU evidence read; eight independently rerun stdlib tests pass.',
                        'Reader restricts actual opened files to .py/.json; official safetensors and README/LICENSE/yaml descriptors are compared/carried, not rehashed.',
                        'Source/JSON-only physical delegate does not execute live X/L or registry._condition_lock.']},
        'fold_checks':notes,'same_seed_full_initial_sha':next(iter(initials)),'same_official_encoder_initial_sha':next(iter(encoder_initials)),
        'parameter_scope':{'encoder_trainable':6,'head_trainable':6,'frozen':134},'selected_phase_counts':dict(phases),
        'durable_optimizer_updates':2324,'durable_backwards':15447,'head_group_updates':2324,'backbone_group_updates':2124,
        'physical_counter_subtotals':dict(total),'formal_attempt_seconds_known_sum':stage['formal_attempt_seconds_known_sum'],
        'engineering_separate':engineering,'nested_metadata_seconds_not_added':stage['nested_metadata_seconds_not_added'],
        'baseline_zero_call_scope':'These five actual REVE attempts are single positive-work completed invocations, not zero-call reuse. Any prior baseline zero-call statement applies only to already-completed-fit rechecks, not these fits, engineering, all history, or a new global cost snapshot.',
        'actual_five_fit_zero_call_rechecks':0,'global_recheck_inventory_reaudited':False,'global_cost_snapshot_created':False,
        'all_historical_exact_wall_seconds':None,'all_historical_exact_physical_counts':None,
        'full_REVE15_FT135_plan360_claimed':False,'old_sources_reports_modified':False,
        'operations':{'JSON_source_reads_only':True,'binary_hash_reads':0,'tensor_loads':0,'model_GPU_process_operations':0,
                      'raw_feature_test_probability_reads':0,'project_verifier_or_actual_helper_executions':0},
        'input_bindings':list(INPUTS.values())}
    target = REPORT / ('independent_REVE_first_seed5_review_' + str(time.time_ns()) + '.json')
    with target.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'path':target.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'passed':True,'folds':notes}))


if __name__ == '__main__': main()
