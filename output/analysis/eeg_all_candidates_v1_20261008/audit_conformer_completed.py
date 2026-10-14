"""Read-only CPU audit of completed Conformer fits; never loads EEG arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))

import torch
from tridep_ccml.eeg_all_candidates import protocol as P, registry as R, storage as S

CONDITION = 'EEGConformer-scratch'


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def audit_fit(seed, fold, registration, lock):
    folder = R.fit_directory(CONDITION, seed, fold)
    split = registration['folds'][f'fold_{fold}']
    status = R.fit_status(CONDITION, seed, fold)
    require(status['status'] == 'training_complete' and status['condition_source_lock'] == lock, 'Completed status/source lock differs')
    evidence = status['evidence']
    R._validate_complete(evidence, folder)
    require(S.read(folder / 'complete.json') == evidence, 'Final complete/status evidence differs')
    meta = S.read(S.checked(evidence['metadata']))
    history = S.read(S.checked(evidence['train_history']))
    validation = S.read(S.checked(evidence['validation_history']))
    require(meta['train_ids'] == split['train'] and meta['val_ids'] == split['val'], 'Fold roles differ')
    require(set(meta['raw_sources']) == set(split['train']+split['val']) and not set(meta['raw_sources']) & set(split['test']), 'Recorded training/validation raw roles overlap test')
    require(len(history) == len(validation) == meta['epochs'], 'Curve lengths differ')
    losses, extended = [], False
    for epoch, row in enumerate(history, 1):
        require(row['epoch'] == epoch and row['optimizer_updates'] == 4*epoch
                and row['backward_calls'] == len(split['train'])*epoch, 'Actual epoch/update/backward count differs')
        windows = [((epoch-1)%8*8+i)%60 for i in range(8)]
        require(row['window_indices'] == windows and row['cycle_epoch'] == (epoch-1)%8+1
                and row['repeated_padding_windows_this_epoch'] == ([0,1,2,3] if epoch%8 == 0 else []), 'Window-bag schedule differs')
        require(row['supervised_ema_updates'] == 0 and validation[epoch-1] == {'epoch':epoch,'val_ce':row['val_eval_ce']}, 'EMA/validation curve differs')
        for group in ('head','backbone'):
            expected = 3e-4*P.learning_rate_multiplier('scratch', epoch, parameter_group=group)
            require(abs(row[group+'_lr']-expected)<1e-15, 'Warmup/cosine schedule differs')
        losses.append(row['val_eval_ce'])
        decision = P.training_decision('scratch', losses, extended=extended)
        require(not decision['stop'] or epoch == len(history), 'Recorded fit continued after its stopping rule')
        if decision['extend']:
            extended = True
    require(decision['stop'] and decision['reason'] == meta['stop_reason'] == evidence['stop_reason']
            and decision['best_epoch'] == meta['best_epoch'] and extended == meta['extended'], 'Terminal stop/best/extension differs')
    require(P.POLICIES['scratch'].minimum_stop_epoch <= meta['epochs'] <= P.POLICIES['scratch'].final_epochs, 'Epoch outside registered bounds')
    require(meta['optimizer_updates'] == meta['head_optimizer_updates'] == meta['encoder_optimizer_updates'] == 4*meta['epochs'], 'Parameter-group update count differs')
    require(meta['test_forward_count'] == meta['supervised_ema_updates'] == 0, 'Forbidden test/EMA update recorded')
    require(meta['micro_subjects'] == 1 and meta['window_chunk'] == 8 and meta['window_bag_size'] == 8 and meta['eval_windows'] == 60
            and meta['fresh_subject_head'] and meta['batchnorm_microbatch_statistics_not_true_batch8'], 'Actual batching/initialization disclosure differs')
    stats = S.read(S.checked(meta['statistics']))
    require(stats['statistics']['fit_subject_ids'] == split['train'] and set(stats['fit_raw_sources']) == set(split['train'])
            and stats['validation_or_test_fit_people'] == [], 'Normalizer source includes nontraining roles')
    initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=False)
    best = torch.load(S.checked(evidence['checkpoint']), map_location='cpu', weights_only=False)
    latest = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
    require(all(bool(torch.isfinite(v).all()) for saved in (initial,best,latest) for v in saved['state_dict'].values()), 'Nonfinite completed checkpoint')
    require(tensor_hash(initial['state_dict']) == meta['initial_state_sha256'] and tensor_hash(best['state_dict']) == meta['selected_state_sha256'], 'Reloaded CPU tensor fingerprints differ')
    require(best['best_epoch'] == meta['best_epoch'] and best['statistics'] == stats['statistics']
            and best['identity']['condition_source_lock'] == lock and best['identity']['registry'] == R.registration_descriptor(), 'Best checkpoint source/stats identity differs')
    require(latest['epoch'] == meta['epochs'] and latest['history'] == history
            and len(latest['subject_order']) == meta['epochs']
            and all(sorted(order)==sorted(split['train']) for order in latest['subject_order']), 'Recovery epoch/subject coverage differs')
    require(latest['sampler_position'] == {'epoch_completed':meta['epochs'],'next_batch':0}
            and latest['scheduler_state_dict'] == {'policy':'scratch','completed_epoch':meta['epochs'],'extended':meta['extended']}, 'Recovery scheduler/sampler differs')
    require(latest['optimizer_updates'] == meta['optimizer_updates'] and latest['backward_calls'] == meta['backward_calls'], 'Recovery actual cost differs')
    require(set(latest['rng']) == {'python','numpy','torch_cpu','torch_cuda'} and len(latest['rng']['torch_cuda']) == 1, 'Full RNG coverage differs')
    require(all(float(v['step']) == meta['optimizer_updates'] for v in latest['optimizer_state_dict']['state'].values()), 'AdamW stored step counts differ')
    changes = meta['parameter_changed_after_training']
    require(all(changes.values()) and all(not torch.equal(initial['state_dict'][name],latest['state_dict'][name]) for name in changes), 'An intended parameter did not actually update')
    require(any(n.startswith('encoder.patch.') for n in changes) and any(n.startswith('encoder.transformer.') for n in changes), 'CNN/attention update evidence missing')
    bn = 'encoder.patch.shallownet.2.num_batches_tracked'
    require(int(initial['state_dict'][bn]) == 0 and int(best['state_dict'][bn]) == meta['best_epoch']*len(split['train'])
            and int(latest['state_dict'][bn]) == meta['epochs']*len(split['train']), 'BN counter differs from actual bag forwards')
    excluded = set(changes)|set(meta['active_running_stat_buffer_keys'])
    frozen_before = {n:v for n,v in initial['state_dict'].items() if n not in excluded}
    frozen_best = {n:v for n,v in best['state_dict'].items() if n not in excluded}
    require(tensor_hash(frozen_before) == meta['frozen_initial_sha256'] == meta['frozen_final_sha256'] == tensor_hash(frozen_best), 'Frozen source state differs')
    require(set(meta['reload_probe_subject_ids']).issubset(set(split['train'])) and evidence['checkpoint_reload_verified'], 'Reload proof used nontraining people')
    failures = [S.desc(p) for p in sorted(folder.glob('failure_*.json'))]
    return {'seed':seed,'fold':fold,'passed':True,'epochs':meta['epochs'],'best_epoch':meta['best_epoch'],
            'stop_reason':meta['stop_reason'],'optimizer_updates':meta['optimizer_updates'],'backward_calls':meta['backward_calls'],
            'actual_changed_parameters':len(changes),'cnn_and_attention_parameters_both_changed':True,
            'bn_running_stats_in_best_checkpoint':True,'bn_counter_matches_actual_window_bag_forwards':True,
            'train_only_standardization_verified':True,'full_curves_and_stop_rule_verified':True,
            'cpu_checkpoint_reload_tensor_hash_verified':True,'recorded_training_only_prediction_reload_proof_verified':True,
            'independent_prediction_forward_performed':False,'rng_optimizer_sampler_recovery_complete':True,
            'evidence':evidence,'initial_checkpoint':S.desc(folder/'initial.pt'),'latest_checkpoint':S.desc(folder/'latest.pt'),
            'statistics':meta['statistics'],'prior_failure_evidence':failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-all', action='store_true', help='Require15/15 actual completed fits')
    args = parser.parse_args()
    torch.set_num_threads(2)
    registration = R.load_registration()
    lock = R._condition_lock(CONDITION)
    contract = S.read(S.checked(lock))['contract']
    require(all(contract[k] is True for k in ('forward_checked','gradient_checked','resume_checked','training_only_engineering_checked')), 'Engineering evidence incomplete')
    require(contract['engineering_optimizer_updates'] == 0 and contract['engineering_backward_calls'] == 1
            and set(contract['probe_subject_ids']).issubset(set(registration['folds']['fold_1']['train'])), 'Engineering input/cost differs')
    require(all(contract['buffer_changes_during_probe_reset_before_exit'].values()), 'Engineering BN reset missing')
    S.checked(contract['probe_checkpoint'])
    report = {'captured_at_Asia_Shanghai':S.now(),'source':S.desc(Path(__file__)), 'condition':CONDITION,
              'registration':R.registration_descriptor(),'condition_source_lock':lock,'engineering_verified':True,
              'gpu_calls':0,'model_constructions':0,'optimizer_executions':0,'outer_test_signal_reads':0,
              'outer_test_probability_reads':0,'candidate_ranking_performed':False,
              'status_snapshot':[], 'completed_fit_audits':[], 'actual_error_lines':[]}
    for seed in P.SEEDS:
        for fold in P.FOLDS:
            status = R.fit_status(CONDITION,seed,fold)
            report['status_snapshot'].append({'seed':seed,'fold':fold,'status':status['status']})
    for identity in report['status_snapshot']:
        if identity['status'] == 'training_complete':
            report['completed_fit_audits'].append(audit_fit(identity['seed'],identity['fold'],registration,lock))
    for folder in sorted((S.OUT/'continuation_queue/attempts').glob('*conformer-*')):
        path = folder/'process.log'
        if not path.exists():
            continue
        data = path.read_bytes()
        errors = [line for line in data.decode('utf-8',errors='replace').splitlines()
                  if re.search(r'Traceback \(most recent call last\)|(?:RuntimeError|ValueError|AssertionError|OutOfMemoryError):|CUDA out of memory',line)]
        report.setdefault('mutable_log_snapshots',[]).append({'path':path.relative_to(S.WORK).as_posix(),
                      'observed_bytes':len(data),'observed_prefix_sha256':hashlib.sha256(data).hexdigest(),'error_lines':errors})
        report['actual_error_lines'].extend(errors)
    count = len(report['completed_fit_audits'])
    report.update(completed_fits_verified=count,planned_fits=15,all15_complete=count==15,
                  cumulative_optimizer_updates=sum(r['optimizer_updates'] for r in report['completed_fit_audits']),
                  cumulative_backward_calls=sum(r['backward_calls'] for r in report['completed_fit_audits']),
                  passed=not report['actual_error_lines'] and (count==15 if args.require_all else count>0))
    path = S.REPORT / ('conformer_cumulative_audit_'+str(count)+'fits_'+uuid.uuid4().hex[:12]+'.json')
    S.save(path, report, immutable=True)
    print(json.dumps({'passed':report['passed'],'audit_path':path.relative_to(S.WORK).as_posix(),
                     'completed_fits_verified':count,'all15_complete':count==15,'actual_error_lines':len(report['actual_error_lines']),
                     'optimizer_updates':report['cumulative_optimizer_updates'],'gpu_calls':0,'model_constructions':0,'outer_test_reads':0},indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
