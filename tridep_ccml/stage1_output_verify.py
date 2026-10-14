"""Supplemental stage-1 output integrity audit; no training or model inference.

This module lives outside the preregistered training source directory. It
checks saved outputs against their frozen producer, checkpoints and inputs,
without changing registered files or any acceptance rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent.parent
EXPERIMENT = WORK / 'tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext'
REPORT = WORK / 'output/analysis/tridep_stage1_20261005'
STAGE0 = WORK / 'output/analysis/tridep_stage0_20261004'
CONDITIONS = ('CB-mean', 'CB-spatial', 'CB-temporal', 'CB-spatiotemporal',
              'CB-originalCNN', 'LaBraM-originalCNN')
SEEDS = (11, 22, 33)
METRICS = ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')
REFERENCES = ('fixed-ST', 'original-EST', 'direct-est-weighted')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Supplemental audit awaits uniformly evaluated outputs: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Supplemental audit awaits uniformly evaluated outputs: ' + str(path))
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK), 'Output audit path outside workspace')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': sha(path)}


def check_descriptor(value):
    path = (WORK / value['path']).resolve()
    require(path.is_relative_to(WORK) and path.is_file(), 'Invalid bound output/source path')
    require(sha(path) == value['sha256'], 'Output/source SHA mismatch: ' + value['path'])
    return path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def direct_control(condition):
    return 'CB-originalCNN' if condition.endswith('originalCNN') else 'CB-mean'


def registered_gate(candidate, control, candidate_seeds, control_seeds):
    """Exactly the already-frozen evaluator rule, without selecting a new rule."""
    delta = {key: candidate[key] - control[key] for key in METRICS}
    improving = sum(candidate_seeds[str(seed)]['test']['macro_f1'] > control_seeds[str(seed)]['test']['macro_f1']
                    for seed in SEEDS)
    checks = {'f1_delta_at_least_0_01': delta['macro_f1'] >= .01,
              'at_least2_seeds_strictly_improve': improving >= 2,
              'BA_not_lower': delta['balanced_accuracy'] >= 0.,
              'MDD_recall_drop_at_most0_05': delta['mdd_recall'] >= -.05}
    return {'passed': all(checks.values()), 'checks': checks, 'improving_seeds': improving,
            'delta': delta, 'probability_improvement': delta['nll'] < 0 and delta['brier'] < 0}


def verify_semantic_summaries(report, stage0, predictions):
    metrics = read(report / 'metrics.json')
    original = read(stage0 / 'metrics.json')
    controls = read(report / 'fixed_controls.json')
    expected_controls = {'stage0': descriptor(stage0 / 'metrics.json'),
                         'ST': original['ST-val-weighted'], 'EST': original['EST-original-val-weighted']}
    require(controls == expected_controls, 'Fixed-controls output differs from original stage0 controls')
    acceptance = read(report / 'acceptance.json')
    expected_acceptance = {}
    baseline = original['EST-original-val-weighted']
    for condition in CONDITIONS:
        direct = direct_control(condition)
        eeg_gate = registered_gate(metrics[condition]['eeg']['main_test'], metrics[direct]['eeg']['main_test'],
                                   metrics[condition]['eeg']['seed_means'], metrics[direct]['eeg']['seed_means'])
        est_gate = registered_gate(metrics[condition]['est_weighted']['main_test'], metrics[direct]['est_weighted']['main_test'],
                                   metrics[condition]['est_weighted']['seed_means'], metrics[direct]['est_weighted']['seed_means'])
        original_gate = registered_gate(metrics[condition]['est_weighted']['main_test'], baseline['main_test'],
                                        metrics[condition]['est_weighted']['seed_means'], baseline['train_val_test_seed_means'])
        fixed_st = original['ST-val-weighted']['main_test']
        expected_acceptance[condition] = {
            'direct_control': direct, 'eeg_vs_direct_control': eeg_gate,
            'est_vs_direct_control': est_gate, 'est_vs_original_EST': original_gate,
            'est_minus_fixed_ST': {key: metrics[condition]['est_weighted']['main_test'][key] - fixed_st[key] for key in METRICS},
            'EEG_structure_or_backbone_value_supported': eeg_gate['passed'],
            'fusion_classification_keep': est_gate['passed'] and original_gate['passed'],
            'accepted_combination': False, 'next_phase_started': False}
    require(acceptance == expected_acceptance, 'Acceptance output differs from frozen evaluator rules')
    paired = rows(report / 'paired_changes.jsonl')
    expected_keys = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id'], reference)
                     for r in predictions if r['split'] == 'test' for reference in REFERENCES}
    require(len(paired) == 1998 and
            {(r['condition'], r['seed'], r['outer_fold'], r['subject_id'], r['reference']) for r in paired} == expected_keys,
            'Paired summary source coverage mismatch')
    expected_summary = {condition: {reference: dict(Counter(row['change'] for row in paired
                                      if row['condition'] == condition and row['reference'] == reference))
                                    for reference in REFERENCES} for condition in CONDITIONS}
    require(read(report / 'paired_summary.json') == expected_summary, 'Paired summary counts differ from person-level changes')
    return {'passed': True, 'acceptance_conditions_verified': 6, 'original_fixed_controls_exact': True,
            'paired_summary_counts_verified': 18, 'acceptance_rules_changed': False}


def post_closeout(report):
    """Read-only check after closeout; excluded from the pre-closeout saved result."""
    status = read(report / 'status.json')
    require(status['status'] == 'complete' and status['passed'] is True and status['stage'] == 1,
            'Post-closeout hash audit requires completed stage1 status')
    manifest_path = report / 'deliverable_hashes.json'
    bindings = read(manifest_path)
    require(descriptor(manifest_path)['path'] not in bindings, 'Deliverable manifest must exclude itself')
    for path, expected in bindings.items():
        check_descriptor({'path': path, 'sha256': expected})
    required = ['status.json', 'metrics.json', 'prediction_rows.jsonl', 'independent_verification.json',
                'supplemental_output_verification.json', 'supplemental_output_verifier_source.json']
    require(all(descriptor(report / name)['path'] in bindings for name in required), 'Closeout manifest omits essential evidence')
    return {'passed': True, 'deliverable_files_rehashed': len(bindings), 'deliverable_manifest': descriptor(manifest_path),
            'new_files_or_changes': False, 'self_reference_avoided': True}


def verify(experiment=EXPERIMENT, report=REPORT, stage0=STAGE0, *, save_result=True, check_closeout=False):
    experiment, report, stage0 = (Path(value).resolve() for value in (experiment, report, stage0))
    cfg = read(experiment / 'registration.json')
    require(tuple(cfg['conditions']) == CONDITIONS and tuple(cfg['seeds']) == SEEDS and
            cfg['formal_fits'] == 90 and cfg['formal_updates'] == 18000, 'Stage1 registered output contract mismatch')
    ledger = read(experiment / 'training_ledger.json')
    require(ledger['fits_completed'] == 90 and ledger['updates_completed'] == 18000, 'Uniform output audit cannot run before all fits complete')
    registration = descriptor(experiment / 'registration.json')
    producer = descriptor(WORK / 'tridep_ccml/tridep_stage1/evaluate.py')
    require(cfg['sources'].get(producer['path']) == producer['sha256'], 'Registered output producer changed')
    input_path = experiment / 'input_manifest.json'
    require(cfg['sources'].get(descriptor(input_path)['path']) == sha(input_path), 'Registered input manifest changed')
    inputs = read(input_path)
    cohort = read(WORK / 'S00/configs/cohort.json')
    folds_path = WORK / 'S00' / cohort['fold_file']
    require(sha(folds_path) == cfg['folds_sha256'], 'Output fold assignment hash mismatch')
    folds = read(folds_path)
    ids = cfg['subject_ids']
    labels = {row['subject_id']: row['label'] for row in read(WORK / 'S00/manifests/subjects.json')}
    require(len(ids) == len(set(ids)) == 37 and ids == inputs['subject_ids'] and set(ids) == set(labels), 'Output/input subject order mismatch')
    fixed_source = descriptor(stage0 / 'branch_probabilities.jsonl')
    require(cfg['sources'].get(fixed_source['path']) == fixed_source['sha256'], 'Registered fixed S/T probability file changed')
    original_rows = rows(stage0 / 'branch_probabilities.jsonl')
    original = {(r['seed'], r['outer_fold'], r['subject_id']): r for r in original_rows}
    require(len(original_rows) == len(original) == 555, 'Fixed S/T source coverage mismatch')
    prior_source_hashes = {r['path']: r['sha256'] for r in read(stage0 / 'source_manifest.json')}
    predictions = rows(report / 'prediction_rows.jsonl')
    lookup = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']): r for r in predictions}
    require(len(predictions) == len(lookup) == 3330, 'Output rows missing/repeated')
    reloads = read(report / 'reload_validation.json')
    reload_lookup = {(r['condition'], r['seed'], r['fold']): r for r in reloads['checks']}
    require(reloads['passed'] is True and len(reloads['checks']) == len(reload_lookup) == 90, 'Uniform reload evidence incomplete')
    selections = read(report / 'fusion_selections.json')
    selection_lookup = {(r['condition'], r['seed'], r['fold']): r for r in selections}
    require(len(selections) == len(selection_lookup) == 90, 'Fusion selection coverage mismatch')
    bindings = {}

    def bind(value):
        old = bindings.get(value['path'])
        require(old is None or old == value, 'One output/source path has conflicting SHA bindings')
        if old is None:
            check_descriptor(value)
            bindings[value['path']] = value

    for path in (experiment / 'registration.json', experiment / 'training_ledger.json', input_path,
                 folds_path, WORK / 'S00/manifests/subjects.json', stage0 / 'source_manifest.json',
                 stage0 / 'metrics.json', report / 'prediction_rows.jsonl', report / 'reload_validation.json',
                 report / 'fusion_selections.json', report / 'metrics.json', report / 'acceptance.json',
                 report / 'fixed_controls.json', report / 'paired_changes.jsonl', report / 'paired_summary.json'):
        bind(descriptor(path))
    bind(producer)
    bind(fixed_source)
    bundle_count, row_count = 0, 0
    expected_identities = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            for fold in range(1, 6):
                identity = (condition, seed, fold)
                folder = experiment / f'runs/{condition}/seed_{seed}/fold_{fold}'
                complete_path = folder / 'complete.json'
                complete = read(complete_path)
                require(len(complete['artifacts']) == 3 and {Path(d['path']).name for d in complete['artifacts']} ==
                        {'checkpoint.pt', 'metadata.json', 'train_history.json'}, 'Invalid completed fit artifact inventory')
                for value in complete['artifacts']:
                    require((WORK / value['path']).resolve().parent == folder, 'Completed artifact belongs to another fit')
                    bind(value)
                bind(descriptor(complete_path))
                checkpoint = descriptor(folder / 'checkpoint.pt')
                metadata = read(folder / 'metadata.json')
                require((metadata['condition'], metadata['seed'], metadata['fold']) == identity and
                        metadata['registration'] == registration, 'Output checkpoint metadata provenance mismatch')
                reload = reload_lookup[identity]
                require(reload['checkpoint'] == checkpoint and reload['exact_probabilities'] is True and
                        reload['exact_representations'] is True and reload['frozen_eval_state_unchanged'] is True,
                        'Output bundle lacks exact saved reload evidence for its checkpoint')
                bundle_path = experiment / f'evaluation/{condition}/seed_{seed}/fold_{fold}.npz'
                sidecar_path = bundle_path.with_suffix('.json')
                sidecar = read(sidecar_path)
                bundle = descriptor(bundle_path)
                expected_sidecar = {'checkpoint': checkpoint, 'bundle': bundle, 'representation_dim': 128,
                                    'fixed_ST_source': fixed_source, 'registration': registration}
                require(sidecar == expected_sidecar, 'Evaluation bundle sidecar/source linkage mismatch')
                bind(bundle)
                bind(descriptor(sidecar_path))
                with np.load(bundle_path, allow_pickle=False) as saved:
                    require(set(saved.files) == {'features', 'p_eeg', 'subject_ids', 'labels', 'splits'}, 'Unexpected evaluation bundle fields')
                    features, probabilities = saved['features'], saved['p_eeg']
                    require(features.shape == (37, 128) and features.dtype == np.float32 and np.isfinite(features).all(), 'Invalid 128D saved representation')
                    require(probabilities.shape == (37,) and probabilities.dtype == np.float64 and
                            np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), 'Invalid saved EEG probabilities')
                    require(saved['subject_ids'].tolist() == ids, 'Saved subject representation order differs from frozen cohort')
                    require(saved['labels'].tolist() == [labels[sid] for sid in ids], 'Saved representation diagnosis mismatch')
                    split = folds[f'fold_{fold}']
                    expected_splits = [next(part for part in ('train', 'val', 'test') if sid in split[part]) for sid in ids]
                    require(saved['splits'].tolist() == expected_splits, 'Saved representation personnel isolation mismatch')
                    for index, sid in enumerate(ids):
                        key = (*identity, sid)
                        expected_identities.add(key)
                        row = lookup[key]
                        parent = original[(seed, fold, sid)]
                        expected_st = {modality: parent['parent_checkpoints'][modality] for modality in ('speech', 'text')}
                        require(row['eeg_checkpoint'] == checkpoint and row['fixed_ST_checkpoints'] == expected_st and
                                row['fixed_ST_probability_file'] == fixed_source and row['registration'] == registration,
                                'Prediction source descriptors disagree with checkpoint/fixed S/T/registration')
                        for value in expected_st.values():
                            require(prior_source_hashes.get(value['path']) == value['sha256'], 'Original S/T checkpoint lacks primary source binding')
                            bind(value)
                        require(row['cohort_id'] == cfg['cohort_id'] and row['folds_sha256'] == cfg['folds_sha256'] and
                                row['label'] == labels[sid] == parent['label'] and row['split'] == expected_splits[index] == parent['split'],
                                'Prediction output subject/label/split provenance mismatch')
                        require(row['p_eeg'] == float(probabilities[index]) and row['p_speech'] == parent['p_speech'] and
                                row['p_text'] == parent['p_text'], 'Prediction probabilities are not exact saved EEG/frozen S/T outputs')
                        selection = selection_lookup[identity]
                        require(row['fusion_weights'] == selection['weights'] and selection['val_ids'] == split['val'], 'Prediction fusion weight provenance mismatch')
                        weighted = sum(weight * row['p_' + modality] for weight, modality in
                                       zip(selection['weights'], ('eeg', 'speech', 'text')))
                        equal = sum((1 / 3) * row['p_' + modality] for modality in ('eeg', 'speech', 'text'))
                        require(row['p_est_weighted'] == weighted and row['p_est_equal'] == equal, 'Derived EST probabilities differ from frozen output producer')
                        row_count += 1
                bundle_count += 1
    require(expected_identities == set(lookup) and bundle_count == 90 and row_count == 3330, 'Saved bundle/prediction coverage mismatch')
    summaries = verify_semantic_summaries(report, stage0, predictions)
    source = descriptor(__file__)
    result = {'passed': True, 'stage': 1,
              'verified_at_Asia_Shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'audit_role': 'read-only supplemental output integrity and frozen-rule semantics; no model selection or inference',
              'source': source, 'registered_output_producer': producer, 'registration': registration,
              'evaluation_npz_and_sidecar_pairs_verified': bundle_count,
              'saved_representation_dim': 128, 'saved_subject_order_label_split_verified': True,
              'exact_prediction_output_and_source_rows_verified': row_count,
              'same_checkpoint_saved_exact_reload_links_verified': 90,
              'exact_fixed_ST_and_derived_EST_probabilities': True,
              'semantic_summaries': summaries, 'new_training_or_model_forward_calls': 0,
              'artifact_bindings': [bindings[path] for path in sorted(bindings)]}
    if save_result and not check_closeout:
        source_path = report / 'supplemental_output_verifier_source.json'
        source_value = {'source': source, 'registration': registration, 'registered_producer': producer,
                        'training_registration_not_changed': True, 'output_audit_written_after_registration': True}
        if source_path.exists():
            require(read(source_path) == source_value, 'Saved supplemental verifier source descriptor changed')
        else:
            write(source_path, source_value)
        result['source_descriptor'] = descriptor(source_path)
        result_path = report / 'supplemental_output_verification.json'
        if result_path.exists():
            previous = read(result_path)
            require({key: value for key, value in previous.items() if key != 'verified_at_Asia_Shanghai'} ==
                    {key: value for key, value in result.items() if key != 'verified_at_Asia_Shanghai'},
                    'Existing supplemental output evidence differs; preserving original result')
            result = previous
        else:
            write(result_path, result)
    if check_closeout:
        result['post_closeout'] = post_closeout(report)
    print(json.dumps({'passed': True, 'bundle_pairs': 90, 'exact_prediction_rows': 3330,
                      'semantic_summaries_verified': True, 'saved': save_result and not check_closeout,
                      'post_closeout_verified': check_closeout}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=EXPERIMENT)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--stage0', type=Path, default=STAGE0)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--post-closeout', action='store_true', help='Also check closeout hashes, without saving/changing evidence')
    args = parser.parse_args()
    verify(args.experiment, args.report, args.stage0, save_result=not args.no_save, check_closeout=args.post_closeout)
