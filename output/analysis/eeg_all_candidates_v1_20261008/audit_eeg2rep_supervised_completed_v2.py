"""Version2 CPU artifact audit adds exact AdamW/head/selected-state scope checks.

The original audit and its three-fit report remain preserved. No model,
prediction forward, signal data, GPU or optimizer execution is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))

import torch
from tridep_ccml.eeg_all_candidates import protocol as P, registry as R, storage as S

CONDITION = 'EEG2Rep-supervised'
SSL = 'EEG2Rep-fold-local-ssl'
POLICY = 'ssl_supervised'


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def audit_fit(seed, fold, registration, lock, contract):
    folder = R.fit_directory(CONDITION, seed, fold)
    split = registration['folds'][f'fold_{fold}']
    status = R.fit_status(CONDITION, seed, fold)
    require(status['status'] == 'training_complete' and status['condition_source_lock'] == lock,
            'Completed status/source lock differs')
    evidence = status['evidence']
    R._validate_complete(evidence, folder)
    require(S.read(folder / 'complete.json') == evidence, 'Final completion/status evidence differs')
    meta = S.read(S.checked(evidence['metadata']))
    history = S.read(S.checked(evidence['train_history']))
    validation = S.read(S.checked(evidence['validation_history']))
    require(meta['train_ids'] == split['train'] and meta['val_ids'] == split['val'], 'Fold roles differ')
    require(set(meta['raw_sources']) == set(split['train'] + split['val'])
            and not set(meta['raw_sources']) & set(split['test']), 'Recorded raw roles overlap test')
    require(meta['statistics'] is None and not (folder / 'train_statistics.json').exists(),
            'Unexpected fitted preprocessing in author Norm=False route')
    binding = meta['ssl_binding']
    require(binding['condition'] == SSL and binding['seed'] == seed and binding['fold'] == fold
            and binding['train_ids'] == binding['encoder_fit_subject_ids'] == split['train']
            and binding['val_ids'] == split['val'] and binding['cross_fold_weights_shared'] is False,
            'SSL initialization binding differs from exact fold roles')
    ssl_folder = R.fit_directory(SSL, seed, fold)
    ssl_status = R.fit_status(SSL, seed, fold)
    require(ssl_status['status'] == 'training_complete', 'SSL dependency incomplete')
    R._validate_complete(ssl_status['evidence'], ssl_folder)
    for key in ('checkpoint', 'metadata', 'completion', 'status', 'settings', 'validation_mask_manifest',
                'condition_source_lock'):
        S.checked(binding[key])
    require(binding['checkpoint'] == ssl_status['evidence']['checkpoint']
            and binding['metadata'] == ssl_status['evidence']['metadata']
            and binding['condition_source_lock'] == R._condition_lock(SSL), 'SSL dependency bytes/lock differ')
    ssl_best = torch.load(S.checked(binding['checkpoint']), map_location='cpu', weights_only=False)
    require(ssl_best['epoch'] == binding['best_epoch']
            and tensor_hash(ssl_best['state_dict']) == binding['selected_state_sha256'], 'SSL best state differs')
    require(len(history) == len(validation) == meta['epochs'], 'Curve lengths differ')
    losses, extended = [], False
    policy = P.POLICIES[POLICY]
    for epoch, row in enumerate(history, 1):
        require(row['epoch'] == epoch and row['optimizer_updates'] == 4 * epoch
                and row['backward_calls'] == len(split['train']) * epoch, 'Actual update/backward count differs')
        windows = [((epoch - 1) % 8 * 8 + index) % 60 for index in range(8)]
        require(row['window_indices'] == windows and row['cycle_epoch'] == (epoch - 1) % 8 + 1
                and row['repeated_padding_windows_this_epoch'] == ([0, 1, 2, 3] if epoch % 8 == 0 else []),
                'Window-bag schedule differs')
        require(row['supervised_ema_updates'] == 0
                and validation[epoch - 1] == {'epoch': epoch, 'val_ce': row['val_eval_ce']}, 'EMA/validation differs')
        for group, base in (('head', policy.head_lr), ('backbone', policy.backbone_lr)):
            expected = base * P.learning_rate_multiplier(POLICY, epoch, parameter_group=group)
            require(abs(row[group + '_lr'] - expected) < 1e-15, 'Warmup/cosine schedule differs')
        losses.append(row['val_eval_ce'])
        decision = P.training_decision(POLICY, losses, extended=extended)
        require(not decision['stop'] or epoch == len(history), 'Continued after registered stopping rule')
        if decision['extend']:
            extended = True
    require(decision['stop'] and decision['reason'] == meta['stop_reason'] == evidence['stop_reason']
            and decision['best_epoch'] == meta['best_epoch'] and extended == meta['extended'], 'Stop/best/extension differs')
    require(policy.minimum_stop_epoch <= meta['epochs'] <= policy.final_epochs, 'Epoch outside registered bounds')
    require(meta['optimizer_updates'] == meta['head_optimizer_updates'] == meta['encoder_optimizer_updates']
            == 4 * meta['epochs'] and meta['backward_calls'] == len(split['train']) * meta['epochs'],
            'Parameter-group update count differs')
    require(meta['test_forward_count'] == meta['supervised_ema_updates'] == 0, 'Forbidden test/EMA update')
    require(meta['micro_subjects'] == 1 and meta['window_chunk'] == 8 and meta['effective_subjects'] == 8
            and meta['window_bag_size'] == 8 and meta['eval_windows'] == 60 and meta['cycle_unique_windows'] == 60
            and meta['cycle_repeats'] == [0, 1, 2, 3] and meta['fresh_subject_head'] is True
            and meta['batchnorm_microbatch_statistics_not_true_batch8'] is False and meta['precision'] == 'float32',
            'Actual batching/initialization/precision disclosure differs')
    initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=False)
    best = torch.load(S.checked(evidence['checkpoint']), map_location='cpu', weights_only=False)
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    require(all(bool(torch.isfinite(value).all()) for saved in (initial, best, latest)
                for value in saved['state_dict'].values()), 'Nonfinite completed checkpoint')
    require(tensor_hash(initial['state_dict']) == meta['initial_state_sha256']
            and tensor_hash(best['state_dict']) == meta['selected_state_sha256'], 'CPU tensor fingerprints differ')
    for saved in (initial, best, latest):
        identity = saved['identity']
        require(identity['condition'] == CONDITION and identity['seed'] == seed and identity['fold'] == fold
                and identity['condition_source_lock'] == lock and identity['registry'] == R.registration_descriptor()
                and identity['ssl_binding'] == binding and identity['statistics'] is None,
                'Checkpoint identity/binding differs')
    initial_encoder = {name.removeprefix('encoder.'): value for name, value in initial['state_dict'].items()
                       if name.startswith('encoder.')}
    require(set(initial_encoder) == set(ssl_best['state_dict'])
            and all(torch.equal(value, ssl_best['state_dict'][name]) for name, value in initial_encoder.items()),
            'Actual initial encoder does not exactly equal same-fold SSL best')
    require(best['best_epoch'] == meta['best_epoch'] and best['statistics'] is None, 'Best epoch/statistics differs')
    require(latest['epoch'] == meta['epochs'] and latest['history'] == history
            and len(latest['subject_order']) == meta['epochs']
            and all(sorted(order) == sorted(split['train']) for order in latest['subject_order']),
            'Recovery epoch/subject coverage differs')
    require(latest['sampler_position'] == {'epoch_completed': meta['epochs'], 'next_batch': 0}
            and latest['scheduler_state_dict'] == {'policy': POLICY, 'completed_epoch': meta['epochs'],
                                                   'extended': meta['extended']}, 'Recovery scheduler/sampler differs')
    require(latest['optimizer_updates'] == meta['optimizer_updates']
            and latest['backward_calls'] == meta['backward_calls'], 'Recovery cost differs')
    require(set(latest['rng']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
            and len(latest['rng']['torch_cuda']) == 1, 'Full saved RNG coverage differs')
    changes = meta['parameter_changed_after_training']
    require(set(changes) == set(contract['trainable_parameter_names']) and all(changes.values())
            and all(not torch.equal(initial['state_dict'][name], latest['state_dict'][name]) for name in changes),
            'An intended parameter did not update or names differ')
    require(meta['active_running_stat_buffer_keys'] == [], 'Unexpected mutable buffer in EEG2Rep supervised state')
    optimizer = latest['optimizer_state_dict']
    groups, states = optimizer['param_groups'], optimizer['state']
    require([group['group_name'] for group in groups] == ['head', 'backbone'], 'Saved optimizer group names differ')
    parameter_ids = [identity for group in groups for identity in group['params']]
    require(len(parameter_ids) == len(set(parameter_ids)) == len(changes) == len(states)
            and set(parameter_ids) == set(states), 'Saved optimizer parameter IDs/states are missing or duplicated')
    head_names = sorted(name for name in changes if name.startswith('head.'))
    head_shapes = {'head.window.0.weight': (16,), 'head.window.0.bias': (16,),
                   'head.window.1.weight': (128, 16), 'head.window.1.bias': (128,),
                   'head.classifier.weight': (2, 128), 'head.classifier.bias': (2,)}
    require(head_names == sorted(head_shapes) and len(groups[0]['params']) == 6
            and len(groups[1]['params']) == len(changes) - 6, 'Native subject head/optimizer group scope differs')
    for name, shape in head_shapes.items():
        require(all(tuple(saved['state_dict'][name].shape) == shape for saved in (initial, best, latest)),
                'Native-D head tensor shape differs: ' + name)
    for group in groups:
        require(group['weight_decay'] == policy.weight_decay
                and abs(group['lr'] - history[-1][group['group_name'] + '_lr']) < 1e-15,
                'Stored terminal optimizer LR/weight decay differs')
    for value in states.values():
        require(set(value) == {'step', 'exp_avg', 'exp_avg_sq'}
                and float(value['step']) == meta['optimizer_updates'], 'Stored AdamW state/step count differs')
        require(all(value[key].dtype == torch.float32 and bool(torch.isfinite(value[key]).all())
                    for key in ('exp_avg', 'exp_avg_sq')), 'Stored AdamW moments are not finite FP32')
    require(any(name.startswith('encoder.core.InputEmbedding.') for name in changes)
            and any(name.startswith('encoder.core.contex_encoder.') for name in changes)
            and any(name.startswith('head.') for name in changes), 'Input/context/head update scope missing')
    selected_changed = [name for name in changes
                        if not torch.equal(initial['state_dict'][name], best['state_dict'][name])]
    require(any(name.startswith('encoder.core.InputEmbedding.') for name in selected_changed)
            and any(name.startswith('encoder.core.contex_encoder.') for name in selected_changed),
            'Selected best lacks learned online input/context changes')
    require(set(latest['best_selected_state_dict']) == set(changes)
            and tensor_hash(latest['best_selected_state_dict']) == latest['best_selected_state_sha256'],
            'Saved selected-state exact scope/hash differs')
    for name, value in latest['best_selected_state_dict'].items():
        require(torch.equal(value, best['state_dict'][name]), 'Best checkpoint differs from saved selected state')
    excluded = set(changes) | set(meta['active_running_stat_buffer_keys'])
    frozen = [{name: value for name, value in saved['state_dict'].items() if name not in excluded}
              for saved in (initial, best, latest)]
    require(tensor_hash(frozen[0]) == meta['frozen_initial_sha256'] == meta['frozen_final_sha256']
            == tensor_hash(frozen[1]) == tensor_hash(frozen[2]), 'Frozen predictor/target/source state differs')
    require(any(name.startswith('encoder.core.target_encoder.') for name in frozen[0])
            and any(name.startswith('encoder.core.Predictor.') for name in frozen[0]), 'Frozen target/predictor missing')
    require(tuple(best['state_dict']['head.window.1.weight'].shape) == (128, 16)
            and tuple(best['state_dict']['head.classifier.weight'].shape) == (2, 128), 'Native-D subject head differs')
    require(set(meta['reload_probe_subject_ids']).issubset(set(split['train']))
            and evidence['checkpoint_reload_verified'], 'Recorded prediction reload used nontraining people')
    return {'seed': seed, 'fold': fold, 'passed': True, 'epochs': meta['epochs'], 'best_epoch': meta['best_epoch'],
            'stop_reason': meta['stop_reason'], 'optimizer_updates': meta['optimizer_updates'],
            'backward_calls': meta['backward_calls'], 'actual_terminal_changed_parameters': len(changes),
            'actual_selected_changed_parameters': len(selected_changed),
            'terminal_head_changed_names': head_names,
            'selected_head_changed_names': [name for name in head_names if name in selected_changed],
            'selected_head_changed_count': sum(name in selected_changed for name in head_names),
            'exact_adamw_parameter_groups_states_steps_and_fp32_moments_verified': True,
            'exact_best_selected_scope_and_digest_verified': True,
            'same_fold_ssl_best_bitwise_initialization_verified': True,
            'selected_and_terminal_input_context_updates_verified': True,
            'target_predictor_frozen_at_initial_best_terminal': True,
            'full_curves_stop_and_recovery_artifacts_verified': True,
            'independent_prediction_forward_performed': False,
            'recorded_train_only_prediction_reload_proof_verified': True,
            'evidence': evidence, 'ssl_binding': binding, 'initial_checkpoint': S.desc(folder / 'initial.pt'),
            'latest_checkpoint': S.desc(folder / 'latest.pt'),
            'prior_failure_evidence': [S.desc(path) for path in sorted(folder.glob('failure_*.json'))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    registration = R.load_registration()
    lock = R._condition_lock(CONDITION)
    contract = S.read(S.checked(lock))['contract']
    require(all(contract[key] is True for key in ('forward_checked', 'gradient_checked', 'resume_checked',
                                                  'training_only_engineering_checked')), 'Engineering checks incomplete')
    require(contract['engineering_optimizer_updates'] == 0 and contract['engineering_backward_calls'] == 1
            and set(contract['probe_subject_ids']).issubset(set(registration['folds']['fold_1']['train'])),
            'Engineering input/cost differs')
    S.checked(contract['probe_checkpoint'])
    report = {'captured_at_Asia_Shanghai': S.now(), 'source': S.desc(Path(__file__)), 'condition': CONDITION,
              'registration': R.registration_descriptor(), 'condition_source_lock': lock,
              'engineering_verified': True, 'gpu_calls': 0, 'model_constructions': 0,
              'optimizer_executions': 0, 'outer_test_signal_reads': 0, 'outer_test_probability_reads': 0,
              'status_snapshot': [], 'completed_fit_audits': []}
    for seed in P.SEEDS:
        for fold in P.FOLDS:
            status = R.fit_status(CONDITION, seed, fold)
            report['status_snapshot'].append({'seed': seed, 'fold': fold, 'status': status['status']})
    for identity in report['status_snapshot']:
        if identity['status'] == 'training_complete':
            report['completed_fit_audits'].append(audit_fit(identity['seed'], identity['fold'], registration, lock, contract))
    count = len(report['completed_fit_audits'])
    report.update(completed_fits_verified=count, planned_fits=15, all15_complete=count == 15,
                  cumulative_optimizer_updates=sum(row['optimizer_updates'] for row in report['completed_fit_audits']),
                  cumulative_backward_calls=sum(row['backward_calls'] for row in report['completed_fit_audits']),
                  passed=count == 15 if args.require_all else count > 0)
    path = S.REPORT / ('eeg2rep_supervised_cumulative_audit_' + str(count) + 'fits_' + uuid.uuid4().hex[:12] + '.json')
    S.save(path, report, immutable=True)
    print(json.dumps({'passed': report['passed'], 'audit_path': path.relative_to(S.WORK).as_posix(),
                      'completed_fits_verified': count, 'optimizer_updates': report['cumulative_optimizer_updates'],
                      'all15_complete': count == 15, 'gpu_calls': 0, 'model_constructions': 0,
                      'outer_test_reads': 0}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
