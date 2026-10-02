"""EEG augmentation only: the fixed S-control model and full CCML objective.

This module never alters the completed v2 experiment. Run ``prepare`` only
after the common augmentation cache is complete; ``train`` explicitly starts
the 15 fixed runs, and ``complete`` independently reloads their predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .common import WORK, OUT, load_views, epoch_views
from tridep_ccml.v2.common import (
    RecordContext, LOSS, descriptor, verify_descriptor, read_json, require,
    write_new, set_seed,
)
from tridep_ccml.v2.scale import ScaleModel, MODALITIES
from tridep_ccml.v2.rebuild_eeg import loss_parts, validate_row
from tridep_ccml.v2.complete_eeg import mean_scores, crosscheck_metrics
from tridep_ccml.ccml import ccml_loss, evidence_dc, dirichlet_from_evidence
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.metrics import summarize


EXPERIMENT = 'CCML-v2-S-control-EEG-aug-37-fulltext'
BASE = WORK/'tridep_ccml/experiments/CCML-v2-S-37-fulltext'
CONFIG = OUT/'v2_configuration.json'
INPUT = OUT/'v2_input_manifest.json'
SEEDS = [11, 22, 33]
FOLDS = [1, 2, 3, 4, 5]


def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def run_folder(seed, fold):
    return OUT/f'runs/v2/seed_{seed}/fold_{fold}'


def old_folder(seed, fold):
    return BASE/f'runs/seed_{seed}/fold_{fold}/outer_control'


def inputs(context):
    """Check the clean control is identical before exposing any augmentation."""
    old_manifest = read_json(BASE/'input_manifest.json')
    require(old_manifest['fit_subject_ids'] == [] and not old_manifest['labels_used'],
            'The baseline cache must be label-free')
    require(old_manifest['subject_ids'] == context.ids, 'Baseline subject order changed')
    with np.load(verify_descriptor(old_manifest['cache']), allow_pickle=False) as source:
        require(source['subject_ids'].tolist() == context.ids, 'Baseline cache identity changed')
        data = {m: source[m].copy() for m in MODALITIES}
    views = load_views()
    require(set(views) == set(context.ids), 'Augmentation subject coverage differs')
    hashes = {}
    for sid in context.ids:
        value = views[sid]
        require(value.shape == (3, 60, 200) and value.dtype == np.float32
                and np.isfinite(value).all(), 'Invalid EEG views: '+sid)
        hashes[sid] = array_digest(value)
    clean = np.stack([views[s][0].mean(0, dtype=np.float32) for s in context.ids])
    require(np.array_equal(clean, data['eeg']), 'Clean view 0 does not reproduce S-control EEG')
    for m, dim in [('eeg', 200), ('speech', 768), ('text', 768)]:
        require(data[m].shape == (37, dim) and data[m].dtype == np.float32
                and np.isfinite(data[m]).all(), 'Invalid baseline input: '+m)
    info = {'baseline_cache': old_manifest['cache'],
            'baseline_input_manifest': descriptor(BASE/'input_manifest.json'),
            'augmentation_configuration': descriptor(OUT/'configuration.json'),
            'augmentation_feature_manifest': descriptor(OUT/'feature_manifest.json'),
            'subject_ids': context.ids, 'cohort_sha256': context.cohort_sha256,
            'folds_sha256': context.folds_sha256, 'view_array_sha256': hashes,
            'views_per_subject': 3, 'shape_per_subject': [3, 60, 200],
            'clean_eeg_mean_bitwise_equal_to_S_control': True,
            'speech_text_copied_from_S_control_cache': True,
            'training': 'one deterministic view per train subject per epoch, then float32 time mean',
            'evaluation': 'clean view 0 only for every split; no test-time augmentation'}
    return {m: torch.from_numpy(v) for m, v in data.items()}, views, info


def configuration(context):
    return {'experiment': EXPERIMENT, 'version': 1,
            'baseline': descriptor(BASE/'configuration.json'),
            'input_manifest': descriptor(INPUT), 'subject_ids': context.ids,
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'seeds': SEEDS, 'folds': FOLDS, 'epochs': 60, 'batch_size': 8,
            'optimizer': 'Adamax', 'projection_lr': 1e-4, 'head_lr': 1e-3,
            'weight_decay': 0., 'gradient_clip': 1., 'ccml': LOSS,
            'model': 'ScaleModel(seed, learned=False)', 'parameters': 223366,
            'threshold': .5, 'tie_class': 0, 'checkpoint_selection': 'fixed_last_epoch',
            'device': 'cpu', 'threads': 1, 'dtype': 'float32',
            'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'training_runs': 15, 'updates_per_run': 240, 'total_updates': 3600,
            'supervised_parent_loaded': False, 'inner_selection': False,
            'outer_val_test_used_for_selection': False,
            'evaluation_scope': 'paired developmental comparison on the existing viewed cohort',
            'sources': [descriptor(p) for p in [Path(__file__), Path(__file__).with_name('common.py'),
                WORK/'tridep_ccml/v2/scale.py', WORK/'tridep_ccml/v2/common.py',
                WORK/'tridep_ccml/v2/rebuild_eeg.py', WORK/'tridep_ccml/v2/complete_eeg.py',
                WORK/'tridep_ccml/ccml.py', WORK/'tridep_ccml/train.py',
                WORK/'tridep_ccml/metrics.py', WORK/'tridep_ccml/records.py']]}


def historical_control(seed, fold):
    side = read_json(old_folder(seed, fold)/'checkpoint.json')
    verify_descriptor(side['checkpoint'])
    require(side['metadata']['configuration'] == descriptor(BASE/'configuration.json')
            and side['metadata']['inputs'] == descriptor(BASE/'input_manifest.json'),
            'Historical control provenance changed')
    return side


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    context = RecordContext()
    data, views, info = inputs(context)
    write_new(INPUT, info)
    write_new(CONFIG, configuration(context))
    checks = []
    for seed in SEEDS:
        for fold in FOLDS:
            set_seed(seed)
            model = ScaleModel(seed, learned=False)
            old_side = historical_control(seed, fold)
            old = old_side['metadata']
            require(state_digest(model.state_dict()) == old['initial_state_sha256'],
                    'S-control initialization is not paired')
            ids = context.folds[f'fold_{fold}']['train']
            require(ids == old['train_ids'], 'Training subject order changed')
            rng = torch.random.get_rng_state().clone()
            sample = epoch_views(ids, views, seed, fold, 1)
            require(torch.equal(rng, torch.random.get_rng_state()), 'Augmentation modified torch RNG')
            require(sample.shape == (len(ids), 60, 200) and sample.dtype == np.float32
                    and np.isfinite(sample).all(), 'Invalid epoch EEG input')
            require(np.array_equal(sample, epoch_views(ids, views, seed, fold, 1)),
                    'Epoch augmentation is not deterministic')
            saved = torch.load(verify_descriptor(old_side['checkpoint']), map_location='cpu', weights_only=True)
            require(saved['metadata'] == old, 'Historical sidecar differs from checkpoint')
            model.load_state_dict(saved['state_dict'], strict=True)
            control = predictions(model, data, context, old, verify_descriptor(old_side['checkpoint']))
            archived = read_json(old_folder(seed, fold)/'predictions.json')
            require(len(control) == len(archived) == 37, 'Historical prediction coverage mismatch')
            for new_row, old_row in zip(control, archived):
                for key in ['subject_id', 'label', 'split', 'p_final', 'u_final', 'prediction',
                            'm', 'e_con', 'e_cmp', 'e_final']:
                    require(new_row[key] == old_row[key], 'Historical clean prediction differs: '+key)
                for modality in MODALITIES:
                    for key in ['e', 'alpha', 'p', 'u']:
                        require(new_row['views'][modality][key] == old_row['views'][modality][key],
                                'Historical clean branch output differs')
            checks.append({'seed': seed, 'fold': fold, 'initial_state_paired': True,
                           'historical_clean_predictions_exactly_replayed': 37})
    write_new(OUT/'v2_preflight.json', {'passed': True, 'configuration': descriptor(CONFIG),
        'checks': checks, 'clean_control_equal': True, 'runs': 15, 'updates': 3600})
    print('v2 augmentation prepared: 15 runs / 3600 updates; no training started.', flush=True)


def load_setup():
    context = RecordContext()
    cfg = read_json(CONFIG)
    require(cfg == configuration(context), 'Locked v2 augmentation configuration changed')
    require(read_json(OUT/'v2_preflight.json')['configuration'] == descriptor(CONFIG),
            'Missing or stale preflight')
    data, views, info = inputs(context)
    require(info == read_json(INPUT), 'Augmentation input changed after freeze')
    return context, cfg, data, views


def epoch_input(train_ids, views, seed, fold, epoch):
    value = epoch_views(train_ids, views, seed, fold, epoch)
    require(value.shape == (len(train_ids), 60, 200) and value.dtype == np.float32
            and np.isfinite(value).all(), 'Invalid training EEG augmentation')
    return value, torch.from_numpy(value.mean(1, dtype=np.float32))


def train_model(model, train_data, labels, train_ids, views, seed, fold):
    """Only outer-train tensors/IDs are passed to the optimizer."""
    optimizer = torch.optim.Adamax([
        {'params': model.projections.parameters(), 'lr': 1e-4},
        {'params': model.heads.parameters(), 'lr': 1e-3}], weight_decay=0.)
    generator = torch.Generator().manual_seed(seed)
    history, batch_hash, updates = [], hashlib.sha256(), 0
    model.train()
    for epoch in range(1, 61):
        value, eeg = epoch_input(train_ids, views, seed, fold, epoch)
        order = torch.randperm(len(labels), generator=generator)
        batch_hash.update(order.numpy().astype('<i8').tobytes())
        total, terms, gradients = 0., {}, []
        for start in range(0, len(labels), 8):
            batch = order[start:start+8]
            batch_inputs = {'eeg': eeg[batch],
                            **{m: train_data[m][batch] for m in ('speech', 'text')}}
            optimizer.zero_grad(set_to_none=True)
            evidence = model(batch_inputs)
            loss = ccml_loss(evidence, labels[batch], epoch, **LOSS)
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters()), 'Missing/nonfinite gradient')
            gradients.append(float(nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)))
            optimizer.step()
            require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameter')
            updates += 1
            total += float(loss.detach())*len(batch)
            with torch.no_grad():
                for key, value_term in loss_parts(evidence.detach(), labels[batch], epoch).items():
                    terms[key] = terms.get(key, 0.)+value_term*len(batch)
        history.append({'epoch': epoch, 'train_objective': total/len(labels),
                        'ccml_loss_terms': {key: value_term/len(labels) for key, value_term in terms.items()},
                        'preclip_gradient_mean': float(np.mean(gradients)),
                        'epoch_views_sha256': array_digest(value)})
    return history, updates, batch_hash.hexdigest()


def predictions(model, data, context, metadata, checkpoint):
    model.eval()
    with torch.no_grad():
        evidence = torch.cat([model({m: data[m][start:start+8] for m in MODALITIES})
                              for start in range(0, len(context.ids), 8)])
        views = dirichlet_from_evidence(evidence)
        final = evidence_dc(evidence, 1.25)
    rows = []
    for i, sid in enumerate(context.ids):
        row = {'experiment': EXPERIMENT, 'seed': metadata['seed'], 'outer_fold': metadata['fold'],
               'subject_id': sid, 'label': context.labels[sid], 'split': metadata['split_map'][sid],
               'train_ids': metadata['train_ids'], 'checkpoint': descriptor(checkpoint),
               'configuration': descriptor(CONFIG), 'feature_source': descriptor(INPUT),
               'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
               'fusion_method': 'ccml', 'rho': 1.25, 'delta': 1., 'gamma': 1.,
               'evaluation_eeg_view': 0, 'scales': [1., 1., 1.],
               'views': {m: {'e': evidence[i, j].tolist(), 'alpha': views['alpha'][i, j].tolist(),
                              'p': views['p'][i, j].tolist(), 'u': float(views['u'][i, j])}
                         for j, m in enumerate(MODALITIES)},
               **{key: final[key][i].tolist() for key in ['m', 'e_con', 'e_cmp', 'e_final']},
               'min_sources': [[m for j, m in enumerate(MODALITIES)
                                if evidence[i, j, k] == final['m'][i, k]] for k in range(2)],
               'p_final': final['p'][i].tolist(), 'u_final': float(final['u'][i]),
               'prediction': int(final['p'][i, 1] > .5)}
        validate_row(row)
        rows.append(row)
    return rows


def run_metadata(context, model, seed, fold):
    assignment = context.folds[f'fold_{fold}']
    old = historical_control(seed, fold)
    initial = state_digest(model.state_dict())
    require(initial == old['metadata']['initial_state_sha256'], 'Initial state not paired')
    require(sum(p.numel() for p in model.parameters()) == 223366, 'Model parameter count changed')
    require(assignment['train'] == old['metadata']['train_ids'], 'Training order not paired')
    return {'experiment': EXPERIMENT, 'seed': seed, 'fold': fold,
            'train_ids': assignment['train'], 'evaluation_ids': context.ids,
            'split_map': {s: split for split, ids in assignment.items() for s in ids},
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'configuration': descriptor(CONFIG), 'inputs': descriptor(INPUT),
            'initial_state_sha256': initial, 'head_initialization_sha256': state_digest(make_initial_state(seed)),
            'historical_control': old['checkpoint'], 'historical_control_metadata': descriptor(old_folder(seed, fold)/'checkpoint.json'),
            'epochs': 60, 'updates': 60*math.ceil(len(assignment['train'])/8),
            'trainable_parameters': 223366, 'objective': 'full_CCML',
            'supervised_parent': None, 'device': 'cpu', 'torch': str(torch.__version__),
            'checkpoint_selection': 'fixed_last_epoch', 'evaluation_eeg_view': 0}


def fit_run(context, data, views, seed, fold):
    started = time.perf_counter()
    set_seed(seed)
    model = ScaleModel(seed, learned=False)
    metadata = run_metadata(context, model, seed, fold)
    folder = run_folder(seed, fold)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'checkpoint.pt'
    if checkpoint.exists():
        side = read_json(folder/'checkpoint.json')
        saved = torch.load(verify_descriptor(side['checkpoint']), map_location='cpu', weights_only=True)
        require(saved['metadata'] == side['metadata'] and
                all(saved['metadata'].get(k) == v for k, v in metadata.items()), 'Existing checkpoint differs')
        metadata = saved['metadata']
        model.load_state_dict(saved['state_dict'], strict=True)
    else:
        require(not (folder/'train_history.json').exists(), 'Incomplete run requires inspection')
        ids = metadata['train_ids']
        require(not set(ids)&set(context.folds[f'fold_{fold}']['val']+context.folds[f'fold_{fold}']['test']),
                'Training split leakage')
        idx = [context.ids.index(s) for s in ids]
        train_views = {s: views[s] for s in ids}
        history, updates, batch_hash = train_model(model, {m: data[m][idx].clone() for m in MODALITIES},
                torch.tensor([context.labels[s] for s in ids], dtype=torch.long), ids, train_views, seed, fold)
        require(updates == metadata['updates'] == 240, 'Update count changed')
        require(batch_hash == historical_control(seed, fold)['metadata']['batch_order_sha256'],
                'Historical batch order is not paired')
        metadata.update(batch_order_sha256=batch_hash, final_scales=model.scales().detach().tolist())
        write_new(folder/'train_history.json', history)
        torch.save({'state_dict': model.state_dict(), 'metadata': metadata}, checkpoint)
        write_new(folder/'checkpoint.json', {'checkpoint': descriptor(checkpoint), 'metadata': metadata})
    rows = predictions(model, data, context, metadata, checkpoint)
    write_new(folder/'predictions.json', rows)
    write_new(folder/'complete.json', {name: descriptor(folder/name) for name in
              ['checkpoint.pt', 'checkpoint.json', 'train_history.json', 'predictions.json']})
    print(f'v2 EEG augmentation seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-started:.1f}', flush=True)
    return rows


def train():
    context, _, data, views = load_setup()
    for seed in SEEDS:
        for fold in FOLDS:
            fit_run(context, data, views, seed, fold)


def complete():
    context, _, data, views = load_setup()
    rows, checks = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            folder = run_folder(seed, fold)
            for value in read_json(folder/'complete.json').values():
                verify_descriptor(value)
            side = read_json(folder/'checkpoint.json')
            saved = torch.load(verify_descriptor(side['checkpoint']), map_location='cpu', weights_only=True)
            metadata = saved['metadata']
            set_seed(seed)
            model = ScaleModel(seed, learned=False)
            expected = run_metadata(context, model, seed, fold)
            require(metadata == side['metadata'] and all(metadata.get(k) == v for k, v in expected.items()),
                    'Checkpoint metadata differs')
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(saved['state_dict'], strict=True)
            require(model.scales().detach().tolist() == metadata['final_scales'] == [1., 1., 1.],
                    'S-control scales changed')
            for modality in MODALITIES:
                prefix = 'projections.'+modality+'.'
                require(any(not torch.equal(initial[k], saved['state_dict'][k])
                            for k in initial if k.startswith(prefix)), 'Projection was not trained')
            history = read_json(folder/'train_history.json')
            require([r['epoch'] for r in history] == list(range(1, 61)), 'Incomplete epoch history')
            generator, batch_hash = torch.Generator().manual_seed(seed), hashlib.sha256()
            train_ids = metadata['train_ids']
            for item in history:
                require(math.isfinite(item['train_objective']) and math.isclose(
                    item['train_objective'], sum(item['ccml_loss_terms'].values()), abs_tol=2e-5, rel_tol=2e-5),
                    'Full CCML loss decomposition differs')
                value, _ = epoch_input(train_ids, {s: views[s] for s in train_ids}, seed, fold, item['epoch'])
                require(array_digest(value) == item['epoch_views_sha256'], 'Epoch augmentation cannot be reproduced')
                batch_hash.update(torch.randperm(len(train_ids), generator=generator).numpy().astype('<i8').tobytes())
            require(batch_hash.hexdigest() == metadata['batch_order_sha256'] ==
                    historical_control(seed, fold)['metadata']['batch_order_sha256'], 'Batch order mismatch')
            reproduced = predictions(model, data, context, metadata, folder/'checkpoint.pt')
            require(reproduced == read_json(folder/'predictions.json'), 'Checkpoint predictions do not reproduce exactly')
            rows.extend(reproduced)
            checks.append({'seed': seed, 'fold': fold, 'updates': 240,
                           'replayed_subjects': 37, 'epoch_views_replayed': 60,
                           'paired_initial_state_and_batch_order': True})
    require(len(rows) == 555 and len([r for r in rows if r['split'] == 'test']) == 111,
            'Incomplete prediction coverage')
    for seed in SEEDS:
        require(Counter(r['subject_id'] for r in rows if r['seed'] == seed and r['split'] == 'test')
                == Counter(context.ids), 'Test OOF coverage mismatch')
    crosschecks = crosscheck_metrics(rows)
    baseline = read_json(BASE/'outer_predictions_control.json')
    report = {'summary': summarize(rows),
              'by_split': {split: {'fusion': mean_scores(rows, split),
                  'branches': {m: mean_scores(rows, split, m) for m in MODALITIES}}
                  for split in ['train', 'val', 'test']},
              'historical_control_predictions': descriptor(BASE/'outer_predictions_control.json'),
              'historical_control_summary': summarize(baseline),
              'comparison': {}, 'branch_interpretation': 'EEG branch under complete CCML joint training, not EEG-only CE'}
    for modality in [None]+MODALITIES:
        new = mean_scores(rows, 'test', modality)
        old = mean_scores(baseline, 'test', modality)
        report['comparison'][modality or 'fusion'] = {'augmented': new, 'control': old,
            'delta': {key: new[key]-old[key] for key in new}}
    write_new(OUT/'v2_predictions.json', rows)
    write_new(OUT/'v2_results.json', report)
    write_new(OUT/'v2_verification.json', {'passed': True, 'checks': checks,
        'independent_metric_crosschecks': crosschecks, 'training_runs': 15,
        'updates': 3600, 'predictions': 555, 'test_predictions': 111,
        'configuration': descriptor(CONFIG), 'inputs': descriptor(INPUT)})
    write_new(OUT/'v2_status.json', {'status': 'complete', 'experiment': EXPERIMENT,
        'results': descriptor(OUT/'v2_results.json'), 'verification': descriptor(OUT/'v2_verification.json'),
        'predictions': descriptor(OUT/'v2_predictions.json')})
    print('v2 EEG augmentation complete: 15 checkpoints, 555 predictions exactly reloaded.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'complete'])
    action = parser.parse_args().action
    {'prepare': prepare, 'train': train, 'complete': complete}[action]()
