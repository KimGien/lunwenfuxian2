"""Verify all paired TC05 ablations and publish their fixed-budget results."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from .ccml import ThreeViewHeads
from .complete_tc01 import read_rows, write
from .complete_tc03 import metric_crosscheck
from .complete_tc04 import PAIRED_FIELDS, METRICS
from .metrics import summarize
from .records import ROOT, MODALITIES, RecordContext, digest, read_json, require, validate_prediction_records
from .train import load_features, make_initial_state, predict_records, run_directory, state_digest, validate_config

OUTPUT = ROOT / 'reports/tc05'
VARIANTS = {'a': ('rho', 1.), 'b': ('delta', 0.), 'c': ('gamma', 0.)}


def protected_artifacts():
    snapshot = read_json(ROOT / 'reports/tc05_existing_artifacts_before.json')
    observed = {}
    for name, expected in snapshot['protected_files'].items():
        observed[name] = digest(ROOT.parent / name)
        require(observed[name] == expected, f'Protected file changed: {name}')
    return observed


def verify(context):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = validate_config(context)
    unchanged = protected_artifacts()
    controls, control_metadata = {}, {}
    for stage in ('TC-01', 'TC-02', 'TC-04'):
        controls[stage] = []
        previous = read_json(ROOT / f'reports/{stage.lower().replace("-", "")}/verification.json')
        require(previous['passed'], f'Unverified control {stage}')
        prior = {(v['seed'], v['fold']): v for v in previous['folds']}
        for seed in config['seeds']:
            for fold in config['folds']:
                directory = run_directory(ROOT, f'{stage}-37-fulltext', seed, fold)
                run = read_json(directory / 'run.json')
                checkpoint = directory / 'checkpoint.pt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                sidecar = read_json(checkpoint.with_suffix('.json'))
                require(run['status'] == 'complete' and run['training_performed'], 'Control incomplete')
                require(saved['metadata'] == sidecar['metadata'] == run['source_metadata'], 'Control metadata differs')
                require(digest(checkpoint) == prior[seed, fold]['checkpoint_sha256'] == run['checkpoint_sha256'] == sidecar['checkpoint_sha256'], 'Control weights changed')
                require(digest(directory / 'predictions.jsonl') == prior[seed, fold]['predictions_sha256'], 'Control predictions changed')
                control_metadata[stage, seed, fold] = saved['metadata']
                controls[stage].extend(read_rows(directory / 'predictions.jsonl'))
        validate_prediction_records(controls[stage], f'{stage}-37-fulltext', context=context)
    rows = {name: [] for name in VARIANTS}
    audits, validations = [], {}
    common_parameters = {'rho': 1.25, 'delta': 1., 'gamma': 1., 'anneal_epochs': 50}
    for seed in config['seeds']:
        initial = state_digest(make_initial_state(seed))
        for fold in config['folds']:
            data = load_features(context, seed, fold)
            train_ids = context.folds[f'fold_{fold}']['train']
            generator = torch.Generator().manual_seed(seed)
            batch_hash = hashlib.sha256()
            for _ in range(config['epochs']):
                batch_hash.update(torch.randperm(len(train_ids), generator=generator).numpy().astype('<i8').tobytes())
            for variant, (changed_key, changed_value) in VARIANTS.items():
                experiment = f'TC-05{variant}-37-fulltext'
                parameters = {**common_parameters, changed_key: changed_value}
                directory = run_directory(ROOT, experiment, seed, fold)
                run = read_json(directory / 'run.json')
                checkpoint = directory / 'checkpoint.pt'
                require(run['status'] == 'complete' and run['training_performed'], 'TC05 run incomplete')
                require((run['experiment'], run['seed'], run['fold']) == (experiment, seed, fold), 'Run identity mismatch')
                require(Path(run['checkpoint_path']).resolve() == checkpoint.resolve(), 'Wrong checkpoint path')
                require(not (directory / 'failure.json').exists(), 'Failure record present')
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                metadata = saved['metadata']
                sidecar = read_json(checkpoint.with_suffix('.json'))
                checksum = digest(checkpoint)
                require(metadata == sidecar['metadata'] == run['source_metadata'], 'Checkpoint metadata differs')
                require(checksum == sidecar['checkpoint_sha256'] == run['checkpoint_sha256'], 'Checkpoint digest differs')
                expected = {'experiment': experiment, 'seed': seed, 'fold': fold,
                    'cohort_id': context.lock['cohort_id'], 'cohort_sha256': context.cohort_sha256,
                    'folds_sha256': context.folds_sha256, 'train_ids': train_ids,
                    'feature_manifest_sha256': data['manifest_hash'], 'feature_npz_sha256': data['manifest']['npz_sha256'],
                    'parent_checkpoints': data['manifest']['parent_checkpoints'], 'training_config_sha256': context.config_sha256,
                    'initialization_sha256': initial, 'batch_order_sha256': batch_hash.hexdigest(),
                    'completed_epochs': 60, 'optimizer_updates': 60 * math.ceil(len(train_ids) / 8),
                    'selection': 'last_epoch', 'trainable_parameters': 774, 'optimizer': 'Adamax',
                    'class_weights': None, 'resampling': False, 'upstream_mode': 'frozen_eval_cached_features',
                    'device': 'cpu', 'torch_version': str(torch.__version__), 'effective_loss_parameters': parameters,
                    'runner_sha256': digest(ROOT / 'train.py'), 'ccml_sha256': digest(ROOT / 'ccml.py'),
                    'records_sha256': digest(ROOT / 'records.py')}
                for key, value in expected.items():
                    require(metadata.get(key) == value, f'{experiment} protocol mismatch: {key}')
                for stage in controls:
                    paired = control_metadata[stage, seed, fold]
                    require(all(metadata[k] == paired[k] for k in PAIRED_FIELDS), f'{experiment} differs from {stage} paired protocol')
                tc04_parameters = control_metadata['TC-04', seed, fold]['effective_loss_parameters']
                require(tc04_parameters == common_parameters, 'TC04 objective mismatch')
                require([k for k in parameters if parameters[k] != tc04_parameters[k]] == [changed_key], 'More than one ablation parameter changed')
                history = read_json(directory / 'train_history.json')
                require([v['epoch'] for v in history] == list(range(1, 61)), 'Incomplete epoch history')
                require(all(math.isfinite(v['train_loss']) and v['train_loss'] >= 0 for v in history), 'Invalid training loss')
                require(state_digest(saved['state_dict']) != initial, 'Weights did not change from initialization')
                model = ThreeViewHeads().eval()
                model.load_state_dict(saved['state_dict'], strict=True)
                require(all(torch.isfinite(v).all().item() for v in model.parameters()), 'Nonfinite saved parameters')
                recorded = read_rows(directory / 'predictions.jsonl')
                require(len(recorded) == 37 and {r['subject_id'] for r in recorded} == set(context.ids), 'Incomplete fold coverage')
                reproduced = predict_records(context, data, model, experiment, seed, fold, checkpoint, metadata, torch.device('cpu'))
                require(recorded == reproduced, 'Reloaded evidence or prediction differs')
                validate_prediction_records(recorded, experiment, context=context, require_complete=False)
                if variant == 'a':
                    for row in recorded:
                        evidence = np.array([row['views'][m]['e'] for m in MODALITIES])
                        common = evidence.min(0)
                        require(np.allclose(row['e_con'], 3 * common, rtol=2e-6, atol=2e-6), 'rho=1 consistent evidence identity failed')
                        require(np.allclose(row['e_final'], evidence.mean(0) + 2 * common, rtol=2e-6, atol=2e-6), 'rho=1 final evidence identity failed')
                if variant == 'c':
                    require(any(sum(row['e_cmp']) > 0 for row in recorded), 'Complementary inference evidence unexpectedly absent')
                rows[variant].extend(recorded)
                audits.append({'experiment': experiment, 'variant': variant, 'seed': seed, 'fold': fold,
                    'checkpoint_sha256': checksum, 'predictions_sha256': digest(directory / 'predictions.jsonl'),
                    'history_sha256': digest(directory / 'train_history.json'), 'effective_loss_parameters': parameters,
                    'only_changed_parameter_from_TC04': changed_key, 'initialization_sha256': initial,
                    'batch_order_sha256': batch_hash.hexdigest(), 'epochs': 60, 'optimizer_updates': metadata['optimizer_updates'],
                    'paired_TC01_TC02_TC04_protocol_equal': True, 'all_37_records_reproduced_exactly': True,
                    'first_train_loss': history[0]['train_loss'], 'last_train_loss': history[-1]['train_loss']})
    for variant, values in rows.items():
        require(len(values) == 555, 'Missing ablation predictions')
        validations[variant] = validate_prediction_records(values, f'TC-05{variant}-37-fulltext', context=context)
        metric_crosscheck(values)
    require(len(audits) == 45, 'Missing training jobs')
    require(protected_artifacts() == unchanged, 'Protected artifact snapshot changed during verification')
    verification = {'passed': True, 'completed_training_jobs': 45, 'completed_epochs_per_job': 60,
        'checkpoints_verified': 45, 'prediction_records_reproduced_exactly': 1665, 'maximum_probability_error': 0.,
        'total_optimizer_updates': sum(v['optimizer_updates'] for v in audits), 'validation': validations,
        'paired_TC01_jobs': 45, 'paired_TC02_jobs': 45, 'paired_TC04_jobs': 45, 'paired_fields_checked': list(PAIRED_FIELDS),
        'protected_files_unchanged': len(unchanged), 'rho1_identity_records_checked': 555,
        'TC05c_retains_complementary_inference': True, 'independent_metrics_verifier': 'sklearn',
        'config_sha256': context.config_sha256, 'completion_script_sha256': digest(__file__),
        'helpers_sha256': {name: digest(ROOT / name) for name in ('complete_tc01.py', 'complete_tc03.py', 'complete_tc04.py', 'metrics.py')},
        'protection_snapshot_sha256': digest(ROOT / 'reports/tc05_existing_artifacts_before.json'), 'jobs': audits}
    return rows, controls, verification, unchanged


def build_summary(rows, controls):
    summaries = {name: summarize(values) for name, values in rows.items()}
    control_summaries = {name: summarize(values) for name, values in controls.items()}
    branches = {}
    for name, values in rows.items():
        branches[name] = {}
        for modality in MODALITIES:
            derived = deepcopy(values)
            for row in derived:
                row.update(p_final=row['views'][modality]['p'], u_final=row['views'][modality]['u'])
                row['prediction'] = int(row['p_final'][1] > .5)
            metric_crosscheck(derived)
            branches[name][modality] = summarize(derived)
            branches[name][modality].update(output=modality, interpretation='branch of the same trained ablation; no separate retraining')
    comparisons = {}
    for name, result in summaries.items():
        comparisons[name] = {}
        for stage in ('TC-04', 'TC-01'):
            reference = control_summaries[stage]
            delta = {key: result['primary_seed_mean_of_fold_means'][key] - reference['primary_seed_mean_of_fold_means'][key] for key in METRICS}
            seed_delta = {seed: data['unweighted_fold_mean']['macro_f1'] - reference['seed_reports'][seed]['unweighted_fold_mean']['macro_f1'] for seed, data in result['seed_reports'].items()}
            positive = sum(value > 0 for value in seed_delta.values())
            comparisons[name][stage] = {'primary_metric_differences': delta, 'per_seed_macro_f1_differences': seed_delta,
                'seeds_with_higher_macro_f1': positive, 'engineering_reference_met': delta['macro_f1'] >= .01 and positive >= 2 and delta['mdd_recall'] >= -.05,
                'nll_worsened': delta['nll'] > 0, 'brier_worsened': delta['brier'] > 0}
    return {'variants': summaries, 'branches': branches, 'controls': control_summaries, 'comparisons': comparisons}


def main():
    rows, controls, verification, unchanged = verify(RecordContext())
    summary = build_summary(rows, controls)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT / 'existing_artifacts_after.json', {'protected_files': unchanged})
    write(OUTPUT / 'verification.json', verification)
    write(OUTPUT / 'summary.json', summary)
    for name, values in rows.items():
        path = OUTPUT / f'predictions_{name}.jsonl'
        content = ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values)
        if path.exists():
            require(path.read_text(encoding='utf-8') == content, 'Existing combined predictions differ')
        else:
            path.write_text(content, encoding='utf-8')
    from .report_tc05 import generate
    generate(summary, verification)
    status = read_json(ROOT / 'reports/experiment_status.json')
    status['utc'] = datetime.now(timezone.utc).isoformat()
    status['TC-05'] = {'status': 'complete', 'completed_training_jobs': 45, 'completed_epochs_per_job': 60,
        'verified': True, 'paired_TC01_TC02_TC04_verified': True, 'total_optimizer_updates': 10800,
        'report': 'reports/tc05/TC-05实验完成报告.md',
        'variants': {name: {'experiment': f'TC-05{name}-37-fulltext', 'jobs': 15,
            'primary_metrics': result['primary_seed_mean_of_fold_means'],
            'effective_loss_parameters': {**{'rho': 1.25, 'delta': 1., 'gamma': 1., 'anneal_epochs': 50}, VARIANTS[name][0]: VARIANTS[name][1]}}
            for name, result in summary['variants'].items()}}
    (ROOT / 'reports/experiment_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'TC05': status['TC-05'], 'comparisons': summary['comparisons']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
