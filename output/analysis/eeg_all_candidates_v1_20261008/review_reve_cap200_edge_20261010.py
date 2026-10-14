"""One completed REVE edge case: JSON/source only; CPU proof is not rerun."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
REPORT = SOURCE.parent
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
FIT = 'REVE-lastblock-finetune/seed_11/fold_2'
INPUTS = {}


def read(path, sha=None):
    path = (WORK / path).resolve()
    assert path.is_relative_to(WORK) and path.suffix in ('.json', '.py')
    raw = path.read_bytes()
    binding = {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert sha is None or sha == binding['sha256']
    assert binding['path'] not in INPUTS or INPUTS[binding['path']] == binding
    INPUTS[binding['path']] = binding
    return (json.loads(raw.decode('utf-8-sig')) if path.suffix == '.json' else raw.decode('utf-8-sig')), binding


def checked(binding):
    return read(binding['path'], binding['sha256'])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-proof', required=True)
    parser.add_argument('--cpu-sha', required=True)
    args = parser.parse_args()
    _, own = read(SOURCE)
    registration, registry = read(OUT / 'registration.json', '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090')
    fit = next(f for f in registration['fits'] if f['fit_id'] == FIT)
    policy = fit['policy']
    assert (policy['initial_epochs'], policy['minimum_stop_epoch'], policy['patience'], policy['final_epochs']) == (120, 40, 30, 200)
    assert (policy['warmup_epochs'], policy['backbone_warmup_epochs'], policy['minimum_improvement']) == (10, 5, .0001)
    assert policy['checkpoint_selection'] == 'validation_subject_mean_ce' and policy['optimizer'] == 'AdamW'
    protocol, protocol_binding = read(WORK / 'tridep_ccml/eeg_all_candidates/protocol.py', '7449f21e82e5a81c71f46fe7ac98681f1308c670dab4598750fbd0192e88d114')
    # Only these three pure mathematical functions are extracted, without project imports.
    names = {'best_validation_epoch', 'training_decision', 'learning_rate_multiplier'}
    nodes = [node for node in ast.parse(protocol).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == 3
    namespace = {'math': math, 'mean': mean, 'Sequence': list, 'POLICIES': {'finetune': SimpleNamespace(**policy)}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), protocol_binding['path'], 'exec'), namespace)
    folder = OUT / 'runs' / FIT
    complete, complete_binding = read(folder / 'complete.json')
    metadata = checked(complete['metadata'])
    history = checked(complete['train_history']); validation = checked(complete['validation_history'])
    status, status_binding = read(folder / 'status.json')
    assert status['status'] == 'training_complete' and status['test_evaluation_status'] == 'frozen'
    assert status['evidence'] == complete and status['registration'] == complete['registry'] == registry
    assert checked(status['event_record']) == {k: v for k, v in status.items() if k != 'event_record'}
    lock = checked(metadata['condition_source_lock'])
    assert status['condition_source_lock'] == metadata['condition_source_lock'] and lock['registration'] == registry
    contract = lock['contract']
    assert metadata['versioned_finetune_implementation'] == contract['versioned_finetune_implementation']
    for key in ('wrapper_source', 'delegated_runner_source', 'luna_adapter_source'):
        checked(metadata['versioned_finetune_implementation'][key])
    checked(metadata['execution_acceleration']['source'])
    sidecar = checked(metadata['execution_acceleration']['registration'])
    assert metadata['execution_acceleration']['diagnostics'] == {'window_chunk': 8, 'qualification': None, 'bitwise_equivalence_required': True}
    assert sidecar['conditions']['REVE-lastblock-finetune']['condition_source_lock'] == metadata['condition_source_lock']
    for role in ('train', 'val', 'test'):
        assert metadata[role + '_ids'] == fit[role + '_ids'] == registration['folds']['fold_2'][role]
    assert [len(fit[role + '_ids']) for role in ('train', 'val', 'test')] == [26, 3, 8]
    assert set(metadata['raw_sources']) == set(fit['train_ids'] + fit['val_ids'])
    assert not set(metadata['raw_sources']) & set(fit['test_ids'])
    assert metadata['test_forward_count'] == 0 and metadata['reload_probe_subject_ids'] == fit['train_ids'][:1]
    raw_bindings = {row['subject_id']: row['raw'] for row in registration['raw_sources']}
    assert all(binding == raw_bindings[sid] for sid, binding in metadata['raw_sources'].items())
    assert len(history) == len(validation) == metadata['epochs'] == complete['epochs'] == 200
    losses, extended, extensions, decisions = [], False, [], []
    for epoch, (row, val) in enumerate(zip(history, validation), 1):
        assert row['epoch'] == val['epoch'] == epoch
        assert type(val['val_ce']) in (int, float) and math.isfinite(val['val_ce']) and val['val_ce'] >= 0
        assert row['val_eval_ce'] == val['val_ce']
        assert (row['optimizer_updates'], row['backward_calls']) == (4 * epoch, 26 * epoch)
        assert row['unfrozen'] is (epoch > 10)
        assert row['training_module_subtrees'] == (['backbone.transformer.layers.21'] if epoch > 10 else [])
        assert row['window_indices'] == [(8 * ((epoch - 1) % 8) + i) % 60 for i in range(8)]
        assert row['cycle'] == (epoch - 1) // 8 + 1 and row['cycle_epoch'] == (epoch - 1) % 8 + 1
        assert row['repeated_padding_windows_this_epoch'] == ([0, 1, 2, 3] if epoch % 8 == 0 else [])
        for group in ('head', 'backbone'):
            expected = policy[group + '_lr'] * namespace['learning_rate_multiplier']('finetune', epoch, parameter_group=group)
            assert math.isclose(row[group + '_lr'], expected, rel_tol=1e-12, abs_tol=1e-16)
        losses.append(val['val_ce'])
        result = namespace['training_decision']('finetune', losses, extended=extended)
        # Independently recompute the effective-improvement best instead of assuming global argmin.
        best, best_loss = 1, losses[0]
        for index, loss in enumerate(losses[1:], 2):
            if best_loss - loss >= policy['minimum_improvement']:
                best, best_loss = index, loss
        assert result['best_epoch'] == best
        assert result['stop'] is (epoch == 200)
        if result['extend']:
            assert not extended and epoch == 120
            extended = True; extensions.append(epoch)
        decisions.append(result)
    prefix = decisions[119]; final = decisions[-1]
    assert extensions == [120] and extended is metadata['extended'] is True
    assert prefix['best_epoch'] == 120 and prefix['extend'] is True and prefix['stop'] is False
    previous, recent = mean(losses[80:100]), mean(losses[100:120])
    relative = (previous - recent) / previous
    assert prefix['last_20_relative_validation_improvement'] == relative and relative >= .01
    assert final['best_epoch'] == metadata['best_epoch'] == 200 and final['reason'] == metadata['stop_reason'] == complete['stop_reason'] == 'epoch_cap'
    assert (metadata['head_optimizer_updates'], metadata['backbone_optimizer_updates'], metadata['backward_calls']) == (800, 760, 5200)
    assert complete['optimizer_updates'] == metadata['optimizer_updates'] == 800
    enc = [n for n in contract['trainable_parameter_names'] if n.startswith('encoder.')]
    head = [n for n in contract['trainable_parameter_names'] if n.startswith('head.')]
    assert len(enc) == len(head) == 6 and len(contract['frozen_parameter_names']) == 134
    assert all(n.startswith('encoder.backbone.transformer.layers.21.') for n in enc)
    assert metadata['selected_checkpoint_phase'] == 'lastblock_finetune'
    for phase in ('selected', 'terminal'):
        assert metadata[phase + '_encoder_changed_parameter_names'] == enc
        assert metadata[phase + '_head_changed_parameter_names'] == head
        assert metadata[phase + '_encoder_changed_parameter_count'] == metadata[phase + '_head_changed_parameter_count'] == 6
    assert metadata['selected_encoder_state_sha256'] == metadata['terminal_encoder_state_sha256']
    assert metadata['selected_encoder_state_sha256'] != metadata['selected_encoder_initial_state_sha256']
    assert metadata['frozen_initial_sha256'] == metadata['frozen_final_sha256']
    assert metadata['terminal_updates_are_not_claimed_as_selected_encoder_updates'] is True
    audit, audit_binding = read(args.cpu_proof, args.cpu_sha)
    checked(audit['auditor_source'])
    assert audit['auditor_source']['sha256'] == '58b101d301f92da4283890e0a4fa80bef40cdb38eeace1001d6fa18e583a3eba'
    assert audit['schema_version'] == 5 and audit['verified_completed_finetune_fits'] == 32 and audit['checkpoint_loads'] == 96
    assert audit['registration'] == registry and not audit['errors'] and not audit['deferred_completed_fits']
    matches = [row for row in audit['completed_rows'] if row['fit_id'] == FIT]
    assert len(matches) == 1
    row = matches[0]
    assert row['passed'] is row['training_json_verified'] is True
    assert row['metadata'] == complete['metadata'] and row['completion'] == complete_binding and row['status_artifact'] == status_binding
    assert (row['epochs'], row['updates'], row['best_epoch'], row['stop']) == (200, 800, 200, 'epoch_cap')
    cpu = row['cpu_checkpoint_audit']
    assert cpu['cpu_tensors_verified'] is True and cpu['checkpoint_loads'] == 3 and cpu['only_map_location_cpu'] is True
    assert cpu['model_constructions'] == cpu['gpu_operations'] == 0
    assert complete['checkpoint'] in cpu['checkpoints']
    bounds = {v['path']: v for v in cpu['durable_counter_boundaries']}
    assert bounds['best.pt']['epoch'] == bounds['latest.pt']['epoch'] == 200
    assert (bounds['latest.pt']['optimizer_updates'], bounds['latest.pt']['backward_calls']) == (800, 5200)
    groups = bounds['latest.pt']['optimizer_groups']
    assert groups['group_names'] == ['head', 'backbone'] and groups['moment_shapes_verified'] is True
    assert groups['serialized_id_parameter_names'] == head + enc
    assert groups['head_parameters'] == groups['backbone_parameters'] == 6
    physical = row['f2_physical_durable_boundary']
    assert physical['durable_optimizer_updates_counted_once'] == 800 and physical['durable_backward_calls_counted_once'] == 5200
    for binding in list(INPUTS.values()): checked(binding)
    result = {
        'schema': 'REVE_first_cap200_selected_terminal_JSON_edge_review_v1', 'source': own,
        'created_at': datetime.now(timezone.utc).isoformat(), 'passed': True, 'fit_id': FIT,
        'scope': 'One REVE seed11/fold2 completed edge case, backed by existing actual CPU5 32/135 and96 loads; no other REVE fits or full plan completion asserted.',
        'registration': registry, 'protocol': protocol_binding, 'metadata': complete['metadata'], 'completion': complete_binding,
        'status': status_binding, 'condition_source_lock': metadata['condition_source_lock'], 'actual_CPU_proof': audit_binding,
        'original_roles': {role: fit[role + '_ids'] for role in ('train', 'val', 'test')},
        'training_batch_tail': [8, 8, 8, 2], 'prefix120_decision': prefix,
        'extension_evidence': {'previous20_mean': previous, 'recent20_mean': recent, 'relative_improvement': relative,
            'best_in_last20percent': True, 'threshold_1percent_met': True, 'extension_epochs': extensions},
        'final200_decision': final, 'all200_prefix_decisions_verified': True, 'earlier_stop_ignored': False,
        'selection_includes_warmup_epochs_1_to_10': True, 'head_warmup_epochs': 10, 'backbone_warmup_epochs_11_to_15': 5,
        'registered_LR_all200_verified': True, 'extended_LR_held_at_10percent': True,
        'selected': {'epoch': 200, 'encoder_changed': 6, 'head_changed': 6, 'encoder_state_sha256': metadata['selected_encoder_state_sha256']},
        'terminal': {'epoch': 200, 'encoder_changed': 6, 'head_changed': 6, 'encoder_state_sha256': metadata['terminal_encoder_state_sha256']},
        'selected_and_terminal_distinct_evidence_fields_preserved': True, 'equal_encoder_digests_observed_at_same_epoch': True,
        'frozen_parameters_unchanged': 134, 'head_updates': 800, 'backbone_updates': 760, 'backward_calls': 5200,
        'CPU_checkpoint_descriptors_not_reopened': cpu['checkpoints'], 'CPU_optimizer_groups': groups,
        'physical_durable_boundary_from_existing_CPU_proof': physical,
        'cost_scope': {'metadata_seconds': metadata['seconds_this_attempt'], 'peak_allocated_bytes': metadata['peak_allocated_bytes'],
            'nested_time_not_added_to_attempt_time': True, 'physical_plus_durable_not_added': True},
        'test_forward_count': 0, 'test_evaluation_status': 'frozen',
        'operations_this_review': {'JSON_source_only': True, 'binary_hash_reads': 0, 'tensor_loads': 0, 'models_GPU_process_operations': 0,
            'raw_feature_test_probability_reads': 0, 'X_L_registry_verifier_executions': 0, 'sealed_files_modified': False},
        'full_REVE15_or_FT135_or_plan360_completion_claimed': False, 'input_bindings': list(INPUTS.values())}
    target = REPORT / ('REVE_first_cap200_selected_terminal_edge_review_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'path': target.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'passed': True,
                      'epoch120_improvement': relative, 'selected_epoch': 200, 'scope': '1 REVE fit; existing CPU32/96'}))


if __name__ == '__main__': main()
