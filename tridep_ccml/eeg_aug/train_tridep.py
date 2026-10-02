"""Original TRI-DEP EEG training with train-only, precomputed EEG views.

This is an additive runner: historical code/checkpoints are read, never edited.
The default command verifies inputs only. Formal training requires --execute.
Epoch-boundary progress stores optimizer and Torch RNG state for exact resume.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import os

from .common import WORK, OUT, load_views, epoch_views
from tridep_ccml import tc00_eeg as original
from tridep_ccml.records import RecordContext, read_json, require, digest
from tridep_ccml.train import state_digest
from tridep_ccml.metrics import binary_metrics
import numpy as np
import torch


ROOT = OUT / 'tridep'
EXPERIMENT = 'EEG-AUG-TRIDEP-37-fulltext'
BASE = WORK / 'tridep_ccml/experiments/TC-00-E-37-fulltext'
CONFIG = ROOT / 'configuration.json'
SEEDS = (11, 22, 33)
FOLDS = (1, 2, 3, 4, 5)
ROUTES = {'eeg': 'p_eeg', 'weighted': 'p_final', 'mean': 'p_mean'}
METRICS = ('macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier')


def descriptor(path):
    path = Path(path)
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest(path)}


def verify(item):
    path = (WORK / item['path']).resolve()
    require(path.is_relative_to(WORK.resolve()), 'Source is outside workspace')
    require(digest(path) == item['sha256'], 'Source hash changed: ' + item['path'])
    return path


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        require(read_json(path) == value, 'Refusing different existing JSON: ' + str(path))
        return
    atomic_json(path, value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def rng_state():
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': [value.clone() for value in torch.cuda.get_rng_state_all()]}


def same_rng(a, b):
    return (torch.equal(a['cpu'], b['cpu']) and len(a['cuda']) == len(b['cuda']) and
            all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'])))


def restore_rng(value):
    require(len(value['cuda']) == torch.cuda.device_count(), 'CUDA device count changed on resume')
    torch.set_rng_state(value['cpu'])
    torch.cuda.set_rng_state_all(value['cuda'])


def load_parent(context, seed, fold):
    """Read the completed corrected-input comparator and unchanged A/T sources."""
    status = read_json(BASE / 'status.json')
    require(status['status'] == 'complete', 'Corrected TRI-DEP comparator is incomplete')
    verify(status['configuration'])
    verify(status['audit'])
    folder = BASE / f'runs/seed_{seed}/fold_{fold}'
    complete = read_json(folder / 'complete.json')
    for item in complete.values():
        verify(item)
    rows = read_json(folder / 'predictions.json')
    side = read_json(folder / 'checkpoint.json')
    verify(side['checkpoint'])
    metadata = side['metadata']
    require(metadata['seed'] == seed and metadata['fold'] == fold and
            metadata['train_ids'] == context.folds[f'fold_{fold}']['train'] and
            metadata['folds_sha256'] == context.folds_sha256 and
            metadata['cohort_sha256'] == context.cohort_sha256,
            'Historical EEG model has wrong training provenance')
    require(metadata['objective'] == 'original_TRI_DEP_cross_entropy' and
            metadata['updates'] == 120, 'Wrong historical EEG comparator')
    require([row['subject_id'] for row in rows] == context.ids, 'Comparator subject order differs')
    parent_rows, checkpoints, _ = original.parent_bundle(context, seed, fold)
    for row, parent in zip(rows, parent_rows):
        sid = row['subject_id']
        require(row['seed'] == seed and row['outer_fold'] == fold and
                row['label'] == context.labels[sid] and
                sid in context.folds[f'fold_{fold}'][row['split']], 'Comparator identity/split differs')
        require(all(row['p_' + m] == parent['p_' + m] for m in ('speech', 'text')),
                'Comparator no longer has unchanged speech/text predictions')
        for m in ('speech', 'text'):
            require(row['unchanged_branch_checkpoints'][m] == checkpoints[m],
                    'Speech/text checkpoint source mismatch')
            verify(checkpoints[m])
    return rows, {'clean_comparator_predictions': descriptor(folder / 'predictions.json'),
                  'clean_comparator_checkpoint': side['checkpoint'],
                  'unchanged_branch_checkpoints': {m: checkpoints[m] for m in ('speech', 'text')},
                  'original_probability_file': descriptor(
                      WORK / f'S00/runs/B0-37-fulltext/seed_{seed}/probabilities.jsonl')}


def preflight(context=None, views=None):
    """Read-only checks; this function does not create a model or train."""
    context = context or RecordContext()
    views = load_views() if views is None else views
    require(set(views) == set(context.ids), 'Augmentation cohort mismatch')
    manifest_path = WORK / 'tridep_ccml/experiments/CCML-v2-E-37-fulltext/eeg_manifest.json'
    clean_sources = {row['provenance']['subject_id']: row for row in read_json(manifest_path)}
    arrays = {}
    for sid in context.ids:
        x = views[sid]
        require(isinstance(x, np.ndarray) and x.shape == (3, 60, 200) and
                x.dtype == np.float32 and np.isfinite(x).all(), 'Bad augmented views for ' + sid)
        row = clean_sources[sid]
        require(row['provenance']['fit_subject_ids'] == [], 'Clean backbone was MODMA-supervised')
        clean = np.load(verify(row['feature']), allow_pickle=False)
        require(np.array_equal(x[0], clean), 'Clean view is not exact corrected EEG input: ' + sid)
        arrays[sid] = array_digest(x)
    baseline = read_json(BASE / 'configuration.json')
    branch = baseline['eeg_branch']
    require(branch['shape'] == [60, 200] and branch['model_class'] == 'EEGConvPoolClassifier' and
            branch['epochs'] == 60 and branch['batch_size'] == 16 and branch['lr'] == .005 and
            branch['optimizer'] == 'Adamax' and branch['weight_decay'] == 0,
            'Original TRI-DEP branch configuration changed')
    require(baseline['seeds'] == list(SEEDS), 'Original seeds changed')
    parents = {f'{seed}/{fold}': load_parent(context, seed, fold)[1]
               for seed in SEEDS for fold in FOLDS}
    sources = [Path(__file__), Path(__file__).with_name('common.py'), Path(__file__).with_name('prepare.py'),
               WORK / 'tridep_ccml/tc00_eeg.py', WORK / 'tridep_ccml/train.py',
               WORK / 'tridep_ccml/records.py', WORK / 'tridep_ccml/metrics.py',
               WORK / 'S00/scripts/metrics.py', WORK / 'S00/scripts/smoke_models.py',
               WORK / 'S00/upstream-dev/lib/models/models.py']
    configuration = {
        'experiment': EXPERIMENT, 'version': 1, 'cohort_sha256': context.cohort_sha256,
        'folds_sha256': context.folds_sha256, 'subject_ids': context.ids,
        'seeds': list(SEEDS), 'folds': list(FOLDS), 'eeg_branch': branch,
        'clean_comparator': descriptor(BASE / 'configuration.json'),
        'clean_eeg_manifest': descriptor(manifest_path),
        'augmented_feature_manifest': descriptor(OUT / 'feature_manifest.json'),
        'view_array_sha256': arrays,
        'augmentation': 'epoch_views; view0 clean, view1/2 augmented; train subjects only',
        'views_per_subject': 3, 'inference_view': 0, 'test_time_augmentation': False,
        'objective': 'original_TRI_DEP_cross_entropy', 'dropout': '0.4_train_eval_at_inference',
        'gradient_clipping': None, 'selection': 'fixed_last_epoch', 'threshold': .5,
        'tie_class': 0, 'fusion': baseline['fusion'], 'device': 'cuda',
        'matmul_tf32': False, 'cudnn_tf32': True, 'formal_runs': 15, 'updates': 1800,
        'augmentation_rng': 'must not advance CPU or CUDA Torch RNG',
        'sources': [descriptor(path) for path in sources], 'parents': parents,
        'evaluation_scope': '37 previously viewed subjects; development comparison',
    }
    return context, views, configuration


def training_views(ids, views, seed, fold, epoch):
    before = rng_state()
    x = epoch_views(ids, views, seed, fold, epoch)
    require(same_rng(before, rng_state()), 'Augmentation advanced model/dropout RNG')
    require(isinstance(x, np.ndarray) and x.dtype == np.float32 and
            x.shape == (len(ids), 60, 200) and np.isfinite(x).all(), 'Bad epoch view tensor')
    matches = []
    for i, sid in enumerate(ids):
        choices = [j for j in range(3) if np.array_equal(x[i], views[sid][j])]
        require(choices, 'Epoch input is not a registered view: ' + sid)
        matches.append(choices)
    return x, {'epoch': epoch, 'training_input_sha256': array_digest(x),
               'view_index_matches': matches, 'subject_ids': list(ids)}


def run_metadata(context, cfg, seed, fold, initial, parent_sources):
    return {'experiment': EXPERIMENT, 'seed': seed, 'fold': fold,
            'train_ids': context.folds[f'fold_{fold}']['train'],
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'configuration': descriptor(CONFIG), 'initial_state_sha256': initial,
            'parent_sources': parent_sources, 'objective': cfg['objective'],
            'trainable_parameters': 498434, 'epochs': 60, 'updates': 120,
            'device': 'cuda', 'torch': str(torch.__version__),
            'cuda_runtime': str(torch.version.cuda), 'gpu': torch.cuda.get_device_name(0),
            'matmul_tf32': False, 'cudnn_tf32': True,
            'evaluation_input': 'view0_only', 'checkpoint_selection': 'fixed_last_epoch'}


def fit_one(context, views, cfg, seed, fold):
    original.runtime(seed)
    branch = cfg['eeg_branch']
    model = getattr(original.models, branch['model_class'])(**branch['model_kwargs']).to('cuda')
    initial = state_digest(model.state_dict())
    require(sum(p.numel() for p in model.parameters()) == 498434, 'Original architecture changed')
    parent_rows, parent_sources = load_parent(context, seed, fold)
    metadata = run_metadata(context, cfg, seed, fold, initial, parent_sources)
    folder = ROOT / f'runs/seed_{seed}/fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoint.pt'
    progress = folder / 'progress.pt'
    ids = metadata['train_ids']
    assignment = context.folds[f'fold_{fold}']
    require(not set(ids) & set(assignment['val'] + assignment['test']), 'Training split leakage')
    y = torch.tensor([context.labels[sid] for sid in ids], dtype=torch.long)
    clean = {sid: views[sid][0] for sid in context.ids}
    dataset = original.TrackingDataset(torch.tensor(np.stack([clean[sid] for sid in ids])), y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adamax(model.parameters(), lr=.005, weight_decay=0)
    history, view_history, updates, start_epoch = [], [], 0, 1
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['metadata']['contract'] == metadata, 'Final checkpoint contract mismatch')
        model.load_state_dict(saved['state_dict'], strict=True)
        history = saved['history']
        view_history = saved['view_history']
        require(saved['metadata']['completed_epochs'] == 60 and saved['metadata']['actual_updates'] == 120,
                'Incomplete final checkpoint')
    else:
        if progress.exists():
            saved = torch.load(progress, map_location='cpu', weights_only=True)
            require(saved['contract'] == metadata, 'Resume source/configuration/runtime changed')
            model.load_state_dict(saved['state_dict'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            history, view_history = saved['history'], saved['view_history']
            updates, start_epoch = saved['updates'], saved['completed_epochs'] + 1
            dataset.seen = saved['sample_order']
            require(0 <= saved['completed_epochs'] <= 60 and updates == 2 * saved['completed_epochs'] and
                    len(history) == len(view_history) == saved['completed_epochs'] and
                    len(dataset.seen) == saved['completed_epochs'] * len(ids), 'Invalid resume progress')
            restore_rng(saved['rng'])
        for epoch in range(start_epoch, 61):
            x, view_row = training_views(ids, views, seed, fold, epoch)
            dataset.tensors = (torch.from_numpy(np.ascontiguousarray(x)), y)
            model.train()
            total, n = 0., 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(xb.to('cuda')), yb.to('cuda'))
                require(bool(torch.isfinite(loss)), 'Nonfinite original TRI-DEP loss')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            for p in model.parameters()), 'Missing/nonfinite EEG gradient')
                optimizer.step()
                require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'Nonfinite EEG weights')
                updates += 1
                total += float(loss.detach().cpu()) * len(yb)
                n += len(yb)
            require(n == len(ids), 'Epoch did not visit every training subject exactly once')
            history.append({'epoch': epoch, 'train_cross_entropy': total / n})
            view_history.append(view_row)
            atomic_torch(progress, {'contract': metadata, 'state_dict': cpu_state(model),
                         'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'completed_epochs': epoch,
                         'updates': updates, 'history': history, 'view_history': view_history,
                         'sample_order': list(dataset.seen)})
        require(updates == 120 and len(dataset.seen) == 60 * len(ids), 'Original training budget changed')
        require(state_digest(model.state_dict()) != initial, 'Model did not update')
        end_rng = rng_state()
        final_metadata = {**metadata, 'contract': metadata, 'completed_epochs': 60, 'actual_updates': updates,
                          'batch_order_sha256': hashlib.sha256(
                              np.array(dataset.seen, dtype='<i8').tobytes()).hexdigest(),
                          'final_cpu_rng_sha256': array_digest(end_rng['cpu'].numpy()),
                          'final_cuda_rng_sha256': [array_digest(x.numpy()) for x in end_rng['cuda']]}
        atomic_torch(checkpoint, {'state_dict': cpu_state(model), 'metadata': final_metadata,
                                 'history': history, 'view_history': view_history})
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require([r['epoch'] for r in history] == list(range(1, 61)), 'Final training history is incomplete')
    before = original.probabilities(model, clean, context.ids)
    require(state_digest(model.state_dict()) == state_digest(saved['state_dict']), 'Final state reload differs')
    model.load_state_dict(saved['state_dict'], strict=True)
    after = original.probabilities(model, clean, context.ids)
    require(before == after, 'Reloaded clean probabilities differ')
    rows, selection = original.fuse(parent_rows, after)
    for row in rows:
        row.update(experiment=EXPERIMENT, eeg_checkpoint=descriptor(checkpoint),
                   configuration=descriptor(CONFIG), augmentation_inference_view=0,
                   augmentation_applied_at_inference=False,
                   eeg_source_array_sha256=cfg['view_array_sha256'][row['subject_id']],
                   clean_comparator=parent_sources['clean_comparator_predictions'])
        require(row['p_speech'] == parent_rows[context.ids.index(row['subject_id'])]['p_speech'] and
                row['p_text'] == parent_rows[context.ids.index(row['subject_id'])]['p_text'],
                'Unchanged modality probability differs')
    metrics = {}
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split]
        metrics[split] = {route: binary_metrics([r['label'] for r in subset], [r[key] for r in subset])
                          for route, key in ROUTES.items()}
    write_new(folder / 'checkpoint.json', {'checkpoint': descriptor(checkpoint), 'metadata': saved['metadata']})
    write_new(folder / 'train_history.json', history)
    write_new(folder / 'augmentation_history.json', view_history)
    write_new(folder / 'predictions.json', rows)
    write_new(folder / 'fusion_selection.json', selection)
    write_new(folder / 'metrics.json', metrics)
    write_new(folder / 'reload_check.json', {'exact_state': True, 'exact_37_clean_probabilities': True,
                                             'inference_view': 0, 'weights_only': True})
    names = ('checkpoint.pt', 'checkpoint.json', 'train_history.json', 'augmentation_history.json',
             'predictions.json', 'fusion_selection.json', 'metrics.json', 'reload_check.json')
    write_new(folder / 'complete.json', {name: descriptor(folder / name) for name in names})
    print(f'TRI-DEP augmentation complete seed={seed} fold={fold} updates=120', flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return rows


def finish(context):
    rows, completed = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            folder = ROOT / f'runs/seed_{seed}/fold_{fold}'
            path = folder / 'complete.json'
            if not path.exists():
                continue
            for item in read_json(path).values():
                verify(item)
            group = read_json(folder / 'predictions.json')
            require(len(group) == 37 and [r['subject_id'] for r in group] == context.ids,
                    'Completed run subject coverage differs')
            rows.extend(group)
            completed.append({'seed': seed, 'fold': fold, 'complete': descriptor(path)})
    status = {'experiment': EXPERIMENT, 'status': 'complete' if len(completed) == 15 else 'partial',
              'completed_runs': len(completed), 'updates': 120 * len(completed), 'runs': completed,
              'configuration': descriptor(CONFIG)}
    if len(completed) == 15:
        summaries = {}
        for route, key in ROUTES.items():
            seeds = {}
            for seed in SEEDS:
                test = [r for r in rows if r['seed'] == seed and r['split'] == 'test']
                require(Counter(r['subject_id'] for r in test) == Counter(context.ids), 'OOF coverage differs')
                folds = {str(f): binary_metrics([r['label'] for r in test if r['outer_fold'] == f],
                                                [r[key] for r in test if r['outer_fold'] == f]) for f in FOLDS}
                means = {m: sum(v[m] for v in folds.values()) / 5 for m in METRICS}
                seeds[str(seed)] = {'fold_metrics': folds, 'fold_mean': means,
                                    'pooled_oof': binary_metrics([r['label'] for r in test], [r[key] for r in test])}
            summaries[route] = {'seed_reports': seeds,
                               'primary_seed_mean_of_fold_means': {
                                   m: sum(s['fold_mean'][m] for s in seeds.values()) / 3 for m in METRICS}}
        write_new(ROOT / 'summary.json', {'experiment': EXPERIMENT, 'independent_subjects': 37,
                  'configuration': descriptor(CONFIG), 'routes': summaries,
                  'scope': 'fixed-split development comparison; 3 seeds are not 111 independent people'})
        status['summary'] = descriptor(ROOT / 'summary.json')
    atomic_json(ROOT / 'status.json', status)
    return status


def run(seed=None, fold=None):
    context, views, cfg = preflight()
    require(seed is None or seed in SEEDS, 'Invalid seed')
    require(fold is None or fold in FOLDS, 'Invalid fold')
    write_new(CONFIG, cfg)
    for s in (SEEDS if seed is None else [seed]):
        for f in (FOLDS if fold is None else [fold]):
            fit_one(context, views, cfg, s, f)
    return finish(context)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Run/resume the authorized formal training')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--fold', type=int, choices=FOLDS)
    args = parser.parse_args()
    if args.execute:
        result = run(args.seed, args.fold)
        print(json.dumps({'status': result['status'], 'completed_runs': result['completed_runs']}, ensure_ascii=False))
    else:
        preflight()
        print('TRI-DEP augmentation read-only preflight passed; no training started.', flush=True)
