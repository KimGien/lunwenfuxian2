"""Prepare fold-specific TC representations without fitting new parameters."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PARENT = WORK / 'S00'
sys.path.insert(0, str(PARENT / 'scripts'))
import numpy as np
import torch
from cohort import validate_active_cohort, validate_prediction_records
from fulltext_b0 import feature_paths, source_checks
from run_b0 import check_inputs
from metrics import metrics as parent_metrics
from smoke_models import seed as set_seed
sys.path.insert(0, str(PARENT / 'upstream-dev'))
from lib.models import models


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, obj, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != obj:
            raise ValueError('Existing locked artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def configure():
    lock = validate_active_cohort(require_data=True)
    config = {
        'schema_version': 1, 'cohort_id': lock['cohort_id'],
        'cohort_sha256': digest(PARENT / 'configs/cohort.json'),
        'folds_sha256': lock['folds_sha256'],
        'plan_path': 'output/plans/MODMA_TRI-DEP_CCML_实验计划.md',
        'plan_sha256': digest(WORK / 'output/plans/MODMA_TRI-DEP_CCML_实验计划.md'),
        'python': '../S00/.venv/Scripts/python.exe',
        'seeds': [11, 22, 33], 'folds': [1, 2, 3, 4, 5],
        'modalities': ['eeg', 'speech', 'text'], 'epochs': 60,
        'batch_size': 8, 'optimizer': 'Adamax', 'learning_rate': .001,
        'weight_decay': 0., 'gradient_clip': 1., 'rho': 1.25,
        'delta': 1., 'gamma': 1., 'anneal_epochs': 50,
        'threshold': .5, 'tie_class': 0, 'trainable_parameters': 774,
        'checkpoint_selection': 'fixed_final_epoch',
        'representation_policy': 'same_fold_same_seed_parent_frozen_eval',
        'class_weights': None, 'resampling': False, 'calibration_fitting': False,
        'validation_search': False, 'test_interpretation': 'fixed_split_development_comparison',
        'tc06': {'status': 'deferred_second_phase', 'epochs': 60, 'batch_size': 8,
                 'head_lr': .001, 'representation_lr': .0001},
        'synthetic_perturbation': {'modality': 'eeg', 'operation': 'multiply_evidence',
                                   'factors': [0., .5, 1., 2.], 'clinical_or_ood_claim': False},
    }
    names = ['00-weighted', '00-mean', '01', '02', '03-mean', '03-sum', '03-ccml', '04', '05a', '05b', '05c']
    registry = {
        'cohort_id': lock['cohort_id'], 'source_cohort_path': 'S00/configs/cohort.json',
        'source_cohort_sha256': config['cohort_sha256'],
        'folds_sha256': lock['folds_sha256'], 'subject_ids': lock['subject_ids'],
        'experiment_ids': ['TC-' + n + '-37-fulltext' for n in names],
        'seeds': config['seeds'],
        'registration_policy': 'Independent TC namespace; revalidate unchanged S00 cohort and folds; never write TC rows through legacy experiment registry.',
    }
    write(ROOT / 'configs/experiment.json', config, immutable=True)
    write(ROOT / 'configs/registry.json', registry, immutable=True)
    write(ROOT / 'manifests/cohort.json', lock, immutable=True)
    write(ROOT / 'manifests/fold_assignments.json', read(PARENT / lock['fold_file']), immutable=True)
    return config, lock


def export():
    config, lock = configure()
    lock, fullconfig, folds = source_checks()
    textpaths = feature_paths(lock)
    baseconfig = read(PARENT / 'configs/b0.json')
    labels = {r['subject_id']: r['label'] for r in read(PARENT / 'manifests/subjects.json')}
    ids = lock['subject_ids']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    set_seed(11)
    report = {'passed': False, 'new_model_training_started': False,
              'cohort_id': lock['cohort_id'], 'cohort_sha256': config['cohort_sha256'],
              'folds_sha256': lock['folds_sha256'], 'device': device,
              'checkpoints_verified': 0, 'probabilities_reproduced': 0,
              'max_absolute_probability_error': 0., 'bundles': []}
    baseline_rows = []
    baseline_metrics = []
    for value in config['seeds']:
        paths = check_inputs(PARENT / 'features/B0-37', value, folds, baseconfig)
        basefolder = PARENT / f'runs/B0-37/seed_{value}'
        textfolder = PARENT / f'runs/B0-37-fulltext/seed_{value}'
        base_run = read(basefolder / 'run.json')
        text_run = read(textfolder / 'run.json')
        rows = [json.loads(s) for s in (textfolder / 'probabilities.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        validate_prediction_records(rows, 'B0-37-fulltext')
        if len(rows) != 185:
            raise ValueError('Incomplete parent predictions')
        for name, split in folds.items():
            fold = int(name.split('_')[1])
            predictions = {r['subject_id']: r for r in rows if r['outer_fold'] == fold}
            features = np.empty((37, 3, 128), dtype=np.float32)
            metadata = {'seed': value, 'fold': fold, 'cohort_id': lock['cohort_id'],
                        'cohort_sha256': config['cohort_sha256'], 'folds_sha256': lock['folds_sha256'],
                        'train_ids': split['train'], 'subject_ids': ids,
                        'modalities': config['modalities'], 'parent_checkpoints': {}, 'source_features': {},
                        'upstream_model_source_sha256': digest(PARENT / 'upstream-dev/lib/models/models.py'),
                        'exporter_sha256': digest(__file__), 'frozen_eval': True,
                        'parent_prediction_file_sha256': digest(textfolder / 'probabilities.jsonl')}
            for mi, modality in enumerate(config['modalities']):
                branch = fullconfig['branches'][modality]
                checkpoint = (textfolder if modality == 'text' else basefolder) / f'{name}_{modality}.pt'
                expected = (text_run['folds'][name]['text_checkpoint_sha256'] if modality == 'text'
                            else base_run['folds'][name]['branches'][modality]['checkpoint_sha256'])
                sha = digest(checkpoint)
                if sha != expected:
                    raise ValueError('Parent checkpoint hash changed: ' + str(checkpoint))
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                expected_exp = 'B0-37-fulltext' if modality == 'text' else 'B0-37'
                checks = {'seed': value, 'fold': fold, 'train_ids': split['train'], 'configuration': branch,
                          'cohort_id': lock['cohort_id'], 'folds_sha256': lock['folds_sha256'], 'experiment': expected_exp}
                if any(saved.get(key) != expected_value for key, expected_value in checks.items()):
                    raise ValueError('Parent fit provenance mismatch: ' + str(checkpoint))
                model = getattr(models, branch['model_class'])(**branch['model_kwargs'])
                model.load_state_dict(saved['state_dict'], strict=True)
                model.requires_grad_(False).to(device).eval()
                captured = []
                handle = model.classifier[-1].register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().cpu()))
                metadata['parent_checkpoints'][modality] = {'path': checkpoint.relative_to(WORK).as_posix(), 'sha256': sha}
                metadata['source_features'][modality] = {}
                with torch.inference_mode():
                    for i, sid in enumerate(ids):
                        path = textpaths[sid] if modality == 'text' else paths[(fold, sid, modality)]
                        tensor = torch.as_tensor(np.load(path, allow_pickle=False), dtype=torch.float32, device=device).unsqueeze(0)
                        p = float(torch.softmax(model(tensor), dim=-1)[0, 1].cpu())
                        delta = abs(p - predictions[sid]['p_' + modality])
                        if not np.isfinite(p) or delta > 1e-6:
                            raise ValueError(f'Parent probability mismatch: {value}/{fold}/{modality}/{sid}: {delta}')
                        h = captured.pop().numpy()[0]
                        if h.shape != (128,) or not np.isfinite(h).all():
                            raise ValueError('Invalid supervised representation')
                        features[i, mi] = h
                        report['probabilities_reproduced'] += 1
                        report['max_absolute_probability_error'] = max(report['max_absolute_probability_error'], delta)
                        metadata['source_features'][modality][sid] = {'path': path.relative_to(WORK).as_posix(), 'sha256': digest(path)}
                handle.remove()
                report['checkpoints_verified'] += 1
                del model, saved
            splits = [next(part for part in ('train', 'val', 'test') if sid in split[part]) for sid in ids]
            path = ROOT / f'features/seed_{value}/{name}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {'features': features, 'subject_ids': np.array(ids), 'labels': np.array([labels[s] for s in ids]), 'splits': np.array(splits)}
            if path.exists():
                with np.load(path, allow_pickle=False) as previous:
                    if set(previous.files) != set(arrays) or any(not np.array_equal(previous[k], v) for k, v in arrays.items()):
                        raise ValueError('Refusing to replace different representation cache')
            else:
                temporary = path.with_suffix('.npz.part')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(path)
            metadata['npz_sha256'] = digest(path)
            write(path.with_suffix('.json'), metadata, immutable=True)
            report['bundles'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path), 'metadata_sha256': digest(path.with_suffix('.json')), 'shape': list(features.shape)})
            for fusion in ('weighted', 'mean'):
                for sid in ids:
                    row = predictions[sid]
                    weights = row['fusion_weights'] if fusion == 'weighted' else [1 / 3] * 3
                    p = sum(w * row['p_' + m] for w, m in zip(weights, config['modalities']))
                    if fusion == 'weighted' and abs(p - row['p_final']) > 1e-12:
                        raise ValueError('Historical weighted fusion inconsistent')
                    baseline_rows.append({
                        'experiment': f'TC-00-{fusion}-37-fulltext', 'cohort_id': lock['cohort_id'],
                        'cohort_sha256': config['cohort_sha256'], 'folds_sha256': lock['folds_sha256'],
                        'subject_id': sid, 'seed': value, 'outer_fold': fold, 'split': row['split'],
                        'label': row['label'], 'parent_checkpoints': metadata['parent_checkpoints'],
                        'parent_features': {m: metadata['source_features'][m][sid] for m in config['modalities']},
                        'train_ids': split['train'], 'e': None, 'alpha': None,
                        'p': [[1-row['p_'+m], row['p_'+m]] for m in config['modalities']], 'u': None,
                        'm': None, 'e_con': None, 'e_cmp': None, 'e_final': None, 'u_final': None,
                        'p_final': p, 'prediction': int(p > .5),
                        'fusion_method': 'weighted_probability' if fusion == 'weighted' else 'mean_probability',
                        'fusion_weights': weights if fusion == 'weighted' else None,
                        'rho': None, 'delta': None, 'gamma': None,
                        'training_config_sha256': digest(ROOT / 'configs/experiment.json'),
                        'parent_predictions_sha256': metadata['parent_prediction_file_sha256'],
                        'status': 'parent_reuse_no_new_training',
                    })
                selected = [r for r in baseline_rows if r['seed'] == value and r['outer_fold'] == fold and r['split'] == 'test' and r['experiment'] == f'TC-00-{fusion}-37-fulltext']
                baseline_metrics.append({'seed': value, 'fold': fold, 'fusion': fusion, **parent_metrics([r['label'] for r in selected], [r['p_final'] for r in selected])})
            print(f'Prepared seed={value} fold={fold}: 37 x 3 x 128; parent probabilities verified', flush=True)
    report['passed'] = report['checkpoints_verified'] == 45 and report['probabilities_reproduced'] == 1665 and len(report['bundles']) == 15
    if not report['passed']:
        raise ValueError('Incomplete TC preparation')
    output = ROOT / 'reports/tc00_probabilities.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in baseline_rows), encoding='utf-8')
    baseline_summary = {'interpretation': 'Historical parent recheck and equal-weight recomputation; no new trained TC result.', 'fold_metrics': baseline_metrics,
                        'mean_over_seed_means_of_five_folds': {method: {metric: float(np.mean([r[metric] for r in baseline_metrics if r['fusion'] == method])) for metric in ('macro_f1', 'mdd_recall', 'balanced_accuracy')} for method in ('weighted', 'mean')}}
    write(ROOT / 'reports/tc00_summary.json', baseline_summary)
    report['tc00_predictions_sha256'] = digest(output)
    report['utc'] = datetime.now(timezone.utc).isoformat()
    write(ROOT / 'reports/representation_preparation.json', report)
    environment = {'python': sys.version, 'python_executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
                   'cuda_available': torch.cuda.is_available(), 'device': device,
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    write(ROOT / 'reports/environment.json', environment)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (ROOT / 'requirements-prepared.lock.txt').write_text(freeze.stdout, encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'bundles'}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['configure', 'export'], nargs='?', default='configure')
    args = parser.parse_args()
    try:
        configure() if args.action == 'configure' else export()
    except Exception as exc:
        write(ROOT / 'reports/preparation_failure.json', {'passed': False, 'utc': datetime.now(timezone.utc).isoformat(), 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
