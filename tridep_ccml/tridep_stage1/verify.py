"""Independent completion verification for the preregistered stage-1 EEG run.

Importing this module only defines functions. Verification reads existing CPU
checkpoint states and saved GPU reload evidence; it never trains or infers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             recall_score, roc_auc_score)
import torch

WORK = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = WORK / 'tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext'
DEFAULT_REPORT = WORK / 'output/analysis/tridep_stage1_20261005'
DEFAULT_STAGE0 = WORK / 'output/analysis/tridep_stage0_20261004'
CONDITIONS = ('CB-mean', 'CB-spatial', 'CB-temporal', 'CB-spatiotemporal',
              'CB-originalCNN', 'LaBraM-originalCNN')
VIEWS = {'eeg': 'p_eeg', 'est_weighted': 'p_est_weighted', 'est_equal': 'p_est_equal'}
SEEDS = (11, 22, 33)
SPLITS = ('train', 'val', 'test')
KEYS = ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Stage 1 is not ready for final verification; required evidence missing: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Stage 1 is not ready for final verification; required evidence missing: ' + str(path))
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK), 'Evidence path outside workspace: ' + str(path))
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest(path)}


def check(record):
    path = (WORK / record['path']).resolve()
    require(path.is_relative_to(WORK), 'Evidence binding outside workspace')
    require(path.is_file(), 'Bound evidence missing: ' + str(path))
    require(digest(path) == record['sha256'], 'Evidence SHA-256 mismatch: ' + record['path'])
    return path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def state_digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        require(bool(torch.isfinite(cpu).all()), 'Nonfinite checkpoint state: ' + name)
        value.update(name.encode())
        value.update(str(cpu.dtype).encode())
        value.update(str(tuple(cpu.shape)).encode())
        value.update(cpu.numpy().tobytes())
    return value.hexdigest()


def verify_registration(experiment):
    registration_path = experiment / 'registration.json'
    cfg = read(registration_path)
    require(cfg['stage'] == 1 and tuple(cfg['conditions']) == CONDITIONS and tuple(cfg['seeds']) == SEEDS,
            'Registered stage/condition/seed contract drift')
    require(cfg['formal_fits'] == 90 and cfg['formal_updates'] == 18000, 'Registered fit/update budget drift')
    for name in ('inner_fits', 'calibrator_fits', 'threshold_searches'):
        require(cfg[name] == 0, 'Unregistered additional fit/search: ' + name)
    require(cfg['early_stopping'] is False and cfg['head_selection'] == 'none; final epoch60', 'Checkpoint selection policy drift')
    source = descriptor(__file__)
    require(cfg['sources'].get(source['path']) == source['sha256'], 'Independent verifier must have been source-locked before formal training')
    for path, expected in cfg['sources'].items():
        check({'path': path, 'sha256': expected})
    cohort = read(WORK / 'S00/configs/cohort.json')
    folds_path = WORK / 'S00' / cohort['fold_file']
    require(digest(folds_path) == cfg['folds_sha256'] == cohort['folds_sha256'], 'Active fold provenance drift')
    folds = read(folds_path)
    subjects = {r['subject_id']: r['label'] for r in read(WORK / 'S00/manifests/subjects.json')}
    require(set(cfg['subject_ids']) == set(subjects) and len(subjects) == 37, 'Registered active cohort drift')
    for fold in folds.values():
        train, val, test = (set(fold[split]) for split in SPLITS)
        require(not (train & val or train & test or val & test) and train | val | test == set(subjects), 'Personnel isolation failed')
    return cfg, folds, subjects


def verify_fits(experiment, cfg, folds):
    ledger = read(experiment / 'training_ledger.json')
    require(ledger['fits_completed'] == ledger['planned_fits'] == 90 and
            ledger['updates_completed'] == ledger['planned_updates'] == 18000, 'Incomplete formal training ledger')
    ledger_by_key = {(r['condition'], r['seed'], r['fold']): r for r in ledger['fits']}
    require(len(ledger['fits']) == len(ledger_by_key) == 90, 'Repeated or missing completed fit ledger entry')
    checkpoints = {}
    artifacts = []
    pairing_checks = 0
    total_updates = 0
    for seed in SEEDS:
        for fold in range(1, 6):
            pairs = {'token': [], 'cnn': []}
            for condition in CONDITIONS:
                folder = experiment / f'runs/{condition}/seed_{seed}/fold_{fold}'
                complete_path = folder / 'complete.json'
                complete = read(complete_path)
                expected_names = {'checkpoint.pt', 'metadata.json', 'train_history.json'}
                require(len(complete['artifacts']) == 3 and
                        {Path(item['path']).name for item in complete['artifacts']} == expected_names,
                        'Fit completion manifest must bind checkpoint, metadata and history')
                for item in complete['artifacts']:
                    require(check(item).parent == folder.resolve(), 'Fit completion artifact bound to another folder')
                    artifacts.append(item)
                artifacts.append(descriptor(complete_path))
                metadata = read(folder / 'metadata.json')
                key = (condition, seed, fold)
                require((metadata['condition'], metadata['seed'], metadata['fold']) == key, 'Fit identity mismatch')
                require(metadata == ledger_by_key[key], 'Fit ledger and actual checkpoint metadata differ')
                require(metadata['registration'] == descriptor(experiment / 'registration.json'), 'Fit belongs to a different registration')
                require(metadata['folds_sha256'] == cfg['folds_sha256'] and metadata['cohort_id'] == cfg['cohort_id'], 'Fit cohort provenance mismatch')
                for split in SPLITS:
                    require(metadata[split + '_ids'] == folds[f'fold_{fold}'][split], 'Fit personnel/order differs from original fold')
                training = cfg['training'][condition]
                require(metadata['training'] == training and training['epochs'] == 60 and
                        training['checkpoint_selection'] == 'fixed_final_epoch', 'Fit hyperparameters/selection drift')
                expected_updates = 120 if condition.endswith('originalCNN') else 240
                require(metadata['updates'] == training['updates_per_model'] == expected_updates, 'Fit update budget mismatch')
                require(metadata['fit_subjects_unique'] == len(metadata['train_ids']) and
                        metadata['no_val_test_forward_during_training'] is True, 'Training exposure contract violated')
                history = read(folder / 'train_history.json')
                require(len(history) == 60 and [r['epoch'] for r in history] == list(range(1, 61)) and
                        all(np.isfinite(r['train_ce']) and r['train_ce'] >= 0 for r in history), 'Training history incomplete/nonfinite')
                checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
                require(checkpoint['metadata'] == metadata, 'Embedded checkpoint metadata differs')
                require(state_digest(checkpoint['state_dict']) == metadata['final_state_sha256'], 'Final checkpoint state hash mismatch')
                checkpoints[key] = descriptor(folder / 'checkpoint.pt')
                group = 'cnn' if condition.endswith('originalCNN') else 'token'
                pairs[group].append((metadata['initial_state_sha256'], metadata['batch_order_sha256']))
                total_updates += metadata['updates']
                del checkpoint
            require(len(pairs['token']) == 4 and len(set(pairs['token'])) == 1, 'Four token heads differ in paired initialization/batch order')
            require(len(pairs['cnn']) == 2 and len(set(pairs['cnn'])) == 1, 'Two CNN backbones differ in paired initialization/batch order')
            pairing_checks += 2
    require(total_updates == 18000, 'Actual fit update total differs from ledger')
    return {'passed': True, 'checkpoint_metadata_history_complete_sets': 90,
            'epochs_per_fit': 60, 'updates_verified': total_updates,
            'initial_state_and_batch_order_pair_groups': pairing_checks,
            'original_fold_personnel_verified': True, 'CPU_checkpoint_state_hashes_verified': 90}, checkpoints, artifacts


def verify_fixed_st(stage0, predictions, cfg):
    stage0_status = read(stage0 / 'status.json')
    require(stage0_status['passed'] and stage0_status['status'] == 'complete', 'Primary stage0 control incomplete')
    for name, expected_key in (('source_manifest.json', 'source_manifest_sha256'), ('output_manifest.json', 'output_manifest_sha256')):
        require(digest(stage0 / name) == stage0_status[expected_key], 'Primary stage0 lock changed')
    outputs = {r['path']: r for r in read(stage0 / 'output_manifest.json')}
    for name in ('branch_probabilities.jsonl', 'control_predictions.jsonl', 'metrics.json'):
        check(outputs[descriptor(stage0 / name)['path']])
    branches = rows(stage0 / 'branch_probabilities.jsonl')
    by_key = {(r['seed'], r['outer_fold'], r['subject_id']): r for r in branches}
    require(len(branches) == len(by_key) == 555, 'Primary frozen branch coverage mismatch')
    source_manifest = {r['path']: r['sha256'] for r in read(stage0 / 'source_manifest.json')}
    historical = {}
    for branch in branches:
        historical_prediction = WORK / f'S00/runs/B0-37-fulltext/seed_{branch["seed"]}/probabilities.jsonl'
        source = {'path': historical_prediction.relative_to(WORK).as_posix(),
                  'sha256': branch['source_prediction_file_sha256']}
        require(source_manifest.get(source['path']) == source['sha256'], 'Historical frozen prediction source unbound')
        historical[source['path']] = source
        for modality in ('speech', 'text'):
            source = branch['parent_checkpoints'][modality]
            require(source_manifest.get(source['path']) == source['sha256'], 'Frozen S/T checkpoint lacks stage0 source lock')
            historical[source['path']] = source
    for source in historical.values():
        check(source)
    for row in predictions:
        parent = by_key[(row['seed'], row['outer_fold'], row['subject_id'])]
        require(row['split'] == parent['split'] and row['label'] == parent['label'] and
                row['p_speech'] == parent['p_speech'] and row['p_text'] == parent['p_text'], 'Original fixed S/T probabilities changed')
    return {'passed': True, 'frozen_ST_exact_prediction_checks': len(predictions),
            'historical_ST_checkpoints_and_prediction_sources_verified': len(historical),
            'primary_stage0_main_test': {condition: value['main_test'] for condition, value in read(stage0 / 'metrics.json').items()}}, by_key


def sklearn_metrics(group, probability):
    label = np.asarray([r['label'] for r in group], dtype=np.int64)
    p = np.asarray([r[probability] for r in group], dtype=np.float64)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid emitted probability')
    decision = (p > .5).astype(np.int64)
    true_class = np.where(label == 1, p, 1 - p).clip(1e-15, 1 - 1e-15)
    nll = float(-np.log(true_class).mean())
    # Preserve the preregistered true-class clipping definition at saturated p.
    # sklearn additionally clips p before forming (1-p), which changes eps tails.
    if bool(((p >= 1e-12) & (p <= 1 - 1e-12)).all()):
        require(abs(nll - float(log_loss(label, p, labels=[0, 1]))) <= TOLERANCE, 'Independent sklearn NLL disagreement')
    return {'macro_f1': float(f1_score(label, decision, labels=[0, 1], average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(label, decision)),
            'mdd_recall': float(recall_score(label, decision, pos_label=1, zero_division=0)),
            'auc': float(roc_auc_score(label, p)), 'nll': nll,
            'brier': float(brier_score_loss(label, p)), 'n_subjects': len(label),
            'confusion_hc_mdd': confusion_matrix(label, decision, labels=[0, 1]).tolist()}


def verify_metrics(predictions, stored, folds, subjects):
    maximum = 0.0
    comparisons = 0
    metric_groups = 0
    calculated = {}

    def compare(value, expected):
        nonlocal maximum, comparisons
        for key in KEYS:
            error = abs(value[key] - expected[key])
            maximum = max(maximum, error)
            comparisons += 1
            require(error <= TOLERANCE, 'Independent metric mismatch: ' + key + f' error={error}')
        for key in ('n_subjects', 'confusion_hc_mdd'):
            if key in value and key in expected:
                require(value[key] == expected[key], 'Independent count/confusion mismatch')

    for condition in CONDITIONS:
        selected = [r for r in predictions if r['condition'] == condition]
        calculated[condition] = {}
        for view, probability in VIEWS.items():
            reference = stored[condition][view]
            by_group = {(r['seed'], r['fold'], r['split']): r for r in reference['fold_metrics']}
            require(len(reference['fold_metrics']) == len(by_group) == 45, 'Metric groups missing/repeated')
            fold_values, means, pooled = [], {}, {}
            for seed in SEEDS:
                for fold in range(1, 6):
                    for split in SPLITS:
                        group = [r for r in selected if (r['seed'], r['outer_fold'], r['split']) == (seed, fold, split)]
                        expected_ids = set(folds[f'fold_{fold}'][split])
                        require(len(group) == len(expected_ids) and {r['subject_id'] for r in group} == expected_ids,
                                'Prediction personnel coverage failed')
                        require(all(r['label'] == subjects[r['subject_id']] for r in group), 'Prediction diagnosis mismatch')
                        value = {'seed': seed, 'fold': fold, 'split': split, **sklearn_metrics(group, probability)}
                        compare(value, by_group[(seed, fold, split)])
                        fold_values.append(value)
                        metric_groups += 1
                means[str(seed)] = {split: {key: float(np.mean([r[key] for r in fold_values if r['seed'] == seed and r['split'] == split]))
                                           for key in KEYS} for split in SPLITS}
                for split in SPLITS:
                    compare(means[str(seed)][split], reference['seed_means'][str(seed)][split])
                test = [r for r in selected if r['seed'] == seed and r['split'] == 'test']
                require(len(test) == len({r['subject_id'] for r in test}) == 37, 'Each seed must have 37 distinct OOF people')
                pooled[str(seed)] = sklearn_metrics(test, probability)
                compare(pooled[str(seed)], reference['pooled_oof'][str(seed)])
            main = {key: float(np.mean([means[str(seed)]['test'][key] for seed in SEEDS])) for key in KEYS}
            compare(main, reference['main_test'])
            calculated[condition][view] = {'main_test': main, 'seed_means': means,
                                            'fold_metrics': fold_values, 'pooled_oof': pooled}
    require(metric_groups == 810, 'Expected 810 condition/view/seed/fold/split metric groups')
    return {'passed': True, 'sklearn_version': sklearn.__version__, 'tolerance': TOLERANCE,
            'metric_groups': metric_groups, 'scalar_comparisons': comparisons, 'max_absolute_error': maximum,
            'decision_rule': 'p > 0.5; tie is HC', 'nll_rule': 'true-class probability clipped [1e-15,1-1e-15]',
            'aggregation': 'mean of 5 outer folds within seed, then mean of 3 seeds; pooled OOF separately verified',
            'calculated_metrics': calculated}


def verify_fusion(predictions, selections, folds):
    path = WORK / 'S00/scripts/metrics.py'
    spec = importlib.util.spec_from_file_location('_stage1_original_validation_fusion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    by_key = {(r['condition'], r['seed'], r['fold']): r for r in selections}
    require(len(selections) == len(by_key) == 90, 'Fusion selection ledger missing/repeated')
    for condition in CONDITIONS:
        for seed in SEEDS:
            for fold in range(1, 6):
                group = [r for r in predictions if (r['condition'], r['seed'], r['outer_fold']) == (condition, seed, fold)]
                val_ids = folds[f'fold_{fold}']['val']
                val = [next(r for r in group if r['subject_id'] == sid) for sid in val_ids]
                require(all(r['split'] == 'val' for r in val) and len(val) == 3, 'Fusion requires original 3-person validation')
                selected = module.select_weights(val)
                saved = by_key[(condition, seed, fold)]
                require(saved['val_ids'] == val_ids and saved['weights'] == selected['weights'] and
                        saved['validation_macro_f1'] == selected['validation_macro_f1'], 'Original 231-grid validation fusion changed')
                for row in group:
                    expected = sum(weight * row[name] for weight, name in zip(selected['weights'], ('p_eeg', 'p_speech', 'p_text')))
                    equal = sum(row[name] for name in ('p_eeg', 'p_speech', 'p_text')) / 3
                    require(row['fusion_weights'] == selected['weights'] and row['p_est_weighted'] == expected,
                            'Emitted weighted EST differs from selected validation weights')
                    require(abs(row['p_est_equal'] - equal) <= TOLERANCE, 'Emitted equal EST differs from equal fusion')
    return {'passed': True, 'validation_selections': 90, 'candidates_per_selection': 231,
            'candidate_scores': 20790, 'original_selector_source': descriptor(path),
            'only_original_validation_people_used': True}


def verify_paired(predictions, changes, stage0):
    current = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']): r for r in predictions if r['split'] == 'test'}
    baseline = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']): r for r in rows(stage0 / 'control_predictions.jsonl') if r['split'] == 'test'}
    seen = set()
    counts = {}
    for row in changes:
        key = (row['condition'], row['seed'], row['outer_fold'], row['subject_id'])
        unique = (*key, row['reference'])
        require(unique not in seen, 'Repeated paired comparison')
        seen.add(unique)
        candidate = current[key]
        if row['reference'] == 'fixed-ST':
            reference = baseline[('ST-val-weighted', *key[1:])]['p_final']
        elif row['reference'] == 'original-EST':
            reference = baseline[('EST-original-val-weighted', *key[1:])]['p_final']
        elif row['reference'] == 'direct-est-weighted':
            reference_condition = 'CB-originalCNN' if row['condition'].endswith('originalCNN') else 'CB-mean'
            reference = current[(reference_condition, *key[1:])]['p_est_weighted']
        else:
            raise ValueError('Unknown paired reference: ' + row['reference'])
        p = candidate['p_est_weighted']
        label = candidate['label']
        prediction, old_prediction = int(p > .5), int(reference > .5)
        change = ('add_correct' if prediction == label else 'add_error') if prediction != old_prediction else (
                  'unchanged_correct' if prediction == label else 'unchanged_error')
        require(row['label'] == label and row['p_candidate'] == p and row['p_reference'] == reference and
                row['prediction_candidate'] == prediction and row['prediction_reference'] == old_prediction and row['change'] == change,
                'Paired person-level probability/classification/change mismatch')
        count_key = row['condition'] + '/' + row['reference']
        counts.setdefault(count_key, {name: 0 for name in ('add_correct', 'add_error', 'unchanged_correct', 'unchanged_error')})[change] += 1
    expected = {(*key, reference) for key in current for reference in ('fixed-ST', 'original-EST', 'direct-est-weighted')}
    require(seen == expected and len(seen) == 1998, 'Paired comparisons must cover all 666 test candidate rows x 3 references')
    return {'passed': True, 'paired_person_seed_comparisons': len(seen), 'change_counts': counts}


def verify_reloads(saved, checkpoints):
    require(saved['passed'] is True and len(saved['checks']) == 90, 'Saved final checkpoint reload evidence incomplete')
    seen = set()
    for row in saved['checks']:
        key = (row['condition'], row['seed'], row['fold'])
        require(key in checkpoints and key not in seen, 'Repeated/unregistered reload check')
        seen.add(key)
        require(row['exact_probabilities'] is True and row['exact_representations'] is True and
                row['frozen_eval_state_unchanged'] is True, 'Final checkpoint GPU reload check failed')
        require(row['checkpoint'] == checkpoints[key], 'Reload evidence refers to another checkpoint')
    require(seen == set(checkpoints), 'Missing final checkpoint GPU reload check')
    return {'passed': True, 'saved_GPU_reload_checks_verified': len(seen), 'new_GPU_reload_or_inference_in_this_verifier': False}


def verify(experiment=DEFAULT_EXPERIMENT, report=DEFAULT_REPORT, stage0=DEFAULT_STAGE0, *, save_result=True):
    """Validate completed evidence and save independent_verification.json."""
    experiment, report, stage0 = map(lambda path: Path(path).resolve(), (experiment, report, stage0))
    cfg, folds, subjects = verify_registration(experiment)
    fit_audit, checkpoints, fit_artifacts = verify_fits(experiment, cfg, folds)
    prediction_path = report / 'prediction_rows.jsonl'
    predictions = rows(prediction_path)
    identities = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']) for r in predictions}
    require(len(predictions) == len(identities) == 3330, 'Expected 90 fits x 37 unique subject prediction rows')
    require(all(r['condition'] in CONDITIONS and r['seed'] in SEEDS and r['outer_fold'] in range(1, 6) and
                r['subject_id'] in subjects and r['split'] in SPLITS for r in predictions), 'Unregistered prediction identity')
    fixed_audit, _ = verify_fixed_st(stage0, predictions, cfg)
    metric_audit = verify_metrics(predictions, read(report / 'metrics.json'), folds, subjects)
    fusion_audit = verify_fusion(predictions, read(report / 'fusion_selections.json'), folds)
    paired_audit = verify_paired(predictions, rows(report / 'paired_changes.jsonl'), stage0)
    reload_audit = verify_reloads(read(report / 'reload_validation.json'), checkpoints)
    evidence_paths = [experiment / 'registration.json', experiment / 'training_ledger.json',
                      report / 'prediction_rows.jsonl', report / 'metrics.json', report / 'fusion_selections.json',
                      report / 'paired_changes.jsonl', report / 'reload_validation.json']
    artifact_bindings = [descriptor(path) for path in evidence_paths] + fit_artifacts
    result = {'passed': True, 'stage': 1,
              'completed_at_Asia_Shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'verifier_source': descriptor(__file__), 'registration': descriptor(experiment / 'registration.json'),
              'registered_source_hashes_verified': len(cfg['sources']), 'checkpoint_and_training_audit': fit_audit,
              'fixed_original_ST_and_control0': fixed_audit, 'independent_metrics': metric_audit,
              'original_validation_weight_grid': fusion_audit, 'paired_test_changes': paired_audit,
              'saved_final_checkpoint_reload': reload_audit, 'artifact_bindings': artifact_bindings,
              'no_new_training_or_inference': True, 'physical_calibration_unconfirmed': True,
              'evidence_role': 'same previously observed 37-person development cohort; not independent external validation'}
    if save_result:
        source_path = report / 'independent_verifier_source.json'
        source_value = {'source': result['verifier_source'], 'preregistered_sha256': cfg['sources'][result['verifier_source']['path']],
                        'registration': result['registration']}
        if source_path.exists():
            require(read(source_path) == source_value, 'Existing independent verifier source descriptor changed')
        else:
            write(source_path, source_value)
        result['independent_source_descriptor'] = descriptor(source_path)
        result_path = report / 'independent_verification.json'
        if result_path.exists():
            previous = read(result_path)
            require({key: value for key, value in previous.items() if key != 'completed_at_Asia_Shanghai'} ==
                    {key: value for key, value in result.items() if key != 'completed_at_Asia_Shanghai'},
                    'Completed independent evidence changed; preserving original result')
            result = previous
        else:
            write(result_path, result)
    print(json.dumps({'passed': True, 'fits': 90, 'updates': 18000, 'metric_groups': metric_audit['metric_groups'],
                      'max_absolute_error': metric_audit['max_absolute_error'], 'paired_comparisons': 1998}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--stage0', type=Path, default=DEFAULT_STAGE0)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    verify(args.experiment, args.report, args.stage0, save_result=not args.no_save)
