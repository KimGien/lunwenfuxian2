"""CCML-S: bounded modality scales, with fully refitted nested subject splits."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import time

from .common import (WORK, TC, RecordContext, descriptor, verify_descriptor, read_json,
                     require, write_new, digest, set_seed, LOSS)
from .rebuild_eeg import loss_parts, validate_row
from .short_audio import load_pool
from tridep_ccml.ccml import ThreeViewHeads, ccml_loss, evidence_dc, dirichlet_from_evidence
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.metrics import binary_metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = TC/'experiments/CCML-v2-S-37-fulltext'
CONFIG = OUT/'configuration.json'
PLAN = WORK/'output/plans/CCML_v2_S_模态证据尺度校准固定计划.md'
MODALITIES = ['eeg', 'speech', 'text']
DIMENSIONS = {'eeg': 200, 'speech': 768, 'text': 768}
LAMBDAS = [0.01, 0.1]
STAGES = {'control': 'CCML-v2-S-control-37-fulltext', 'S': OUT.name}
METRICS = ['macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier']


class ScaleModel(nn.Module):
    def __init__(self, seed, learned=True):
        super().__init__()
        torch.manual_seed(seed)
        self.projections = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, 128), nn.ReLU())
                                         for m, d in DIMENSIONS.items()})
        self.normalizer = nn.LayerNorm(128, eps=1e-5, elementwise_affine=False)
        self.heads = ThreeViewHeads()
        self.heads.load_state_dict(make_initial_state(seed), strict=True)
        if learned:
            self.scale_raw = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer('scale_raw', torch.zeros(3))
        self.eval()

    def log_scales(self):
        return math.log(2)*self.scale_raw.tanh()

    def scales(self):
        return self.log_scales().exp()

    def penalty(self):
        return self.log_scales().square().mean()

    def raw_evidence(self, inputs):
        reps = torch.stack([self.projections[m](inputs[m]) for m in MODALITIES], 1)
        return F.softplus(self.heads(self.normalizer(reps)))

    def forward(self, inputs):
        return self.raw_evidence(inputs)*self.scales()[None, :, None]


def require_label_free(value):
    require(value['fit_subject_ids'] == [], 'A MODMA-supervised parent cannot enter inner validation')
    require(value.get('labels_used', False) is False, 'Input uses MODMA labels')


def protect_history():
    path = OUT/'history_before.json'
    if not path.exists():
        frozen = read_json(TC/'reports/ccml_v2_closeout/verified_sources.json')
        for p in (TC/'reports/ccml_v2_closeout').rglob('*'):
            if p.is_file():
                frozen[p.relative_to(WORK).as_posix()] = digest(p)
        frozen['tridep_ccml/closeout_v2.py'] = digest(TC/'closeout_v2.py')
        write_new(path, frozen)
    frozen = read_json(path)
    for name, sha in frozen.items():
        require(digest(WORK/name) == sha, 'Historical artifact changed: '+name)
    return len(frozen)


def build_inputs(context):
    sources, eeg, text = [], [], []
    text_root = WORK/'S00/features/B0-37-fulltext'
    manifest = read_json(text_root/'features_manifest.json')
    require(len(manifest) == 37 and len({r['subject_id'] for r in manifest}) == 37, 'Text coverage mismatch')
    by_id = {r['subject_id']: r for r in manifest}
    for sid in context.ids:
        side_path = TC/f'experiments/CCML-v2-E-37-fulltext/eeg/{sid}.json'
        side = read_json(side_path)
        provenance = side['provenance']
        require_label_free(provenance)
        require(provenance['subject_id'] == sid and provenance['cohort_sha256'] == context.cohort_sha256,
                'EEG identity mismatch')
        verify_descriptor(provenance['producer'])
        verify_descriptor(provenance['revision'])
        x = np.load(verify_descriptor(side['feature']), allow_pickle=False)
        require(x.shape == (60, 200) and np.isfinite(x).all(), 'Invalid EEG backbone input')
        eeg.append(x.mean(0, dtype=np.float32))
        row = by_id[sid]
        require_label_free(row)
        require(row['outer_fold'] is None and row['seed'] is None and row['modality'] == 'text' and
                row['folds_sha256'] == context.folds_sha256, 'Text supervised split binding found')
        for p, key in [(WORK/'S00/scripts/fulltext_b0.py', 'producer_sha256'),
                       (text_root/'protocol_resolution.json', 'protocol_sha256'),
                       (WORK/'S00/manifests/backbones.json', 'backbones_sha256')]:
            require(digest(p) == row[key], 'Text provenance changed')
        text_path = (text_root/row['path']).resolve()
        require(text_path.is_relative_to(text_root.resolve()) and digest(text_path) == row['sha256'], 'Text feature changed')
        for q, sha in enumerate(row['transcript_sha256'], 1):
            require(digest(WORK/f'S00/transcripts/B0-37-fulltext/{sid}/{q:02d}.json') == sha, 'Fulltext transcript changed')
        x = np.load(text_path, allow_pickle=False)
        require(x.shape == (29, 768) and np.isfinite(x).all(), 'Invalid fulltext backbone input')
        text.append(x.mean(0, dtype=np.float32))
        sources.append({'subject_id': sid, 'eeg': side['feature'], 'eeg_manifest': descriptor(side_path),
                        'text': descriptor(text_path)})
    speech, speech_source = load_pool(context)
    arrays = {'eeg': np.stack(eeg), 'speech': speech, 'text': np.stack(text), 'subject_ids': np.array(context.ids)}
    for m, d in DIMENSIONS.items():
        require(arrays[m].shape == (37, d) and arrays[m].dtype == np.float32 and np.isfinite(arrays[m]).all(), 'Bad pooled input')
    path = OUT/'backbone_mean_inputs.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            require(set(old.files) == set(arrays) and all(np.array_equal(old[k], v) for k, v in arrays.items()), 'Pooled inputs changed')
    else:
        with path.open('xb') as f:
            np.savez(f, **arrays)
    info = {'version': 'S-backbone-mean-v1', 'cache': descriptor(path), 'subject_ids': context.ids,
            'cohort_sha256': context.cohort_sha256, 'fit_subject_ids': [], 'labels_used': False,
            'input_dimensions': DIMENSIONS, 'sources': sources, 'speech_pool': speech_source,
            'text_manifest': descriptor(text_root/'features_manifest.json'),
            'pooling': 'EEG time mean; speech 29 equal question means; text 29 CLS means; per subject only',
            'supervised_parent_weights_loaded': False}
    write_new(OUT/'input_manifest.json', info)
    return info


def nested_splits(context):
    folds = {}
    for f in range(1, 6):
        outer = context.folds[f'fold_{f}']
        ids = outer['train']
        y = [context.labels[s] for s in ids]
        parts = []
        for j, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=9000+f).split(ids, y), 1):
            train, val = [ids[i] for i in a], [ids[i] for i in b]
            require(not set(train)&set(val) and set(train+val) == set(ids), 'Invalid nested partition')
            require(not set(train+val)&set(outer['val']+outer['test']), 'Outer holdout entered nested fitting')
            require(set(context.labels[s] for s in train) == {0, 1} and set(context.labels[s] for s in val) == {0, 1}, 'Non-stratified split')
            parts.append({'inner_fold': j, 'train_ids': train, 'val_ids': val,
                          'updates': 60*math.ceil(len(train)/8)})
        require(Counter(s for p in parts for s in p['val_ids']) == Counter(ids), 'Inner OOF coverage mismatch')
        folds[str(f)] = parts
    return {'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'method': 'StratifiedKFold(3,shuffle=True,random_state=9000+outer_fold)', 'folds': folds}


def configuration(context):
    return {'experiment': OUT.name, 'date': '2026-09-24', 'version': 1, 'plan': descriptor(PLAN),
            'input_version': 'S-backbone-mean-v1', 'input_manifest': descriptor(OUT/'input_manifest.json'),
            'nested_splits': descriptor(OUT/'nested_splits.json'), 'cohort_sha256': context.cohort_sha256,
            'folds_sha256': context.folds_sha256, 'subject_ids': context.ids,
            'seeds': [11, 22, 33], 'folds': [1, 2, 3, 4, 5], 'lambdas': LAMBDAS,
            'selection': 'minimum unweighted three-inner-fold final NLL; within 1e-12 prefer lambda=0.1',
            'outer_val_usage': 'descriptive scoring only after all selections are locked',
            'scale': {'parameters': 3, 'bounds': [.5, 2.], 'initial': [1., 1., 1.],
                      'formula': 'exp(log(2)*tanh(a))', 'penalty': 'lambda*mean(log(s)^2)', 'lr': .001},
            'epochs': 60, 'batch_size': 8, 'projection_lr': 1e-4, 'head_lr': 1e-3,
            'optimizer': 'Adamax', 'weight_decay': 0., 'gradient_clip': 1., 'ccml': LOSS,
            'normalization': 'per subject and modality LayerNorm128 no affine eps1e-5',
            'dropout': 'none', 'threshold': .5, 'tie_class': 0, 'checkpoint_selection': 'fixed_last_epoch',
            'parameters': {'control': 223366, 'S': 223369}, 'device': 'cpu', 'threads': 1,
            'budget': {'inner_runs': 90, 'outer_control_runs': 15, 'outer_S_runs': 15,
                       'total_runs': 120, 'inner_updates': 16200, 'outer_updates': 7200, 'total_updates': 23400},
            'historical_references': ['CCML-v2-AS-NP-37-fulltext', 'CCML-v2-NP-37-fulltext'],
            'scope': 'CCML extension on a new label-free backbone input version; previously viewed development cohort',
            'sources': [descriptor(TC/name) for name in ['v2/scale.py', 'v2/complete_scale.py', 'v2/test_scale.py',
                        'v2/common.py', 'v2/rebuild_eeg.py', 'v2/short_audio.py', 'ccml.py', 'metrics.py', 'records.py', 'train.py']]}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    count = protect_history()
    context = RecordContext()
    require(read_json(OUT/'test_results.json')['passed'], 'S tests must pass before configuration freeze')
    build_inputs(context)
    splits = nested_splits(context)
    require(sum(p['updates'] for parts in splits['folds'].values() for p in parts)*3*2 == 16200, 'Inner budget changed')
    write_new(OUT/'nested_splits.json', splits)
    cfg = configuration(context)
    write_new(CONFIG, cfg)
    for seed in cfg['seeds']:
        a, b = ScaleModel(seed, False), ScaleModel(seed, True)
        require(state_digest(a.state_dict()) == state_digest(b.state_dict()), 'Initial control/S states differ')
        for model, kind in [(a, 'control'), (b, 'S')]:
            require(sum(p.numel() for p in model.parameters()) == cfg['parameters'][kind], 'Parameter count changed')
    write_new(OUT/'preflight.json', {'passed': True, 'configuration': descriptor(CONFIG),
              'input_manifest': descriptor(OUT/'input_manifest.json'), 'nested_splits': descriptor(OUT/'nested_splits.json'),
              'test_results': descriptor(OUT/'test_results.json'), 'protected_files_unchanged': count,
              'no_supervised_MODMA_parent_loaded': True, 'paired_initial_states': True, 'budget': cfg['budget']})
    print('S preflight passed: label-free pooled inputs; nested disjoint subjects; 120 runs / 23400 updates.', flush=True)


def load_data(context):
    manifest = read_json(OUT/'input_manifest.json')
    require_label_free(manifest)
    require(manifest['subject_ids'] == context.ids, 'Input identity mismatch')
    with np.load(verify_descriptor(manifest['cache']), allow_pickle=False) as source:
        require(source['subject_ids'].tolist() == context.ids, 'Cached identity mismatch')
        return {m: torch.from_numpy(source[m].copy()) for m in MODALITIES}


def predictions(model, data, context, meta, cp, split_map):
    ids = [s for s in context.ids if s in split_map]
    lookup = {s: i for i, s in enumerate(context.ids)}
    raw = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), 8):
            idx = [lookup[s] for s in ids[start:start+8]]
            raw.append(model.raw_evidence({m: data[m][idx] for m in MODALITIES}))
        raw = torch.cat(raw)
        scales = model.scales()
        e = raw*scales[None, :, None]
        views, final = dirichlet_from_evidence(e), evidence_dc(e)
        unscaled = evidence_dc(raw)
    rows = []
    for i, sid in enumerate(ids):
        r = {'experiment': meta['experiment'], 'seed': meta['seed'], 'outer_fold': meta['fold'],
             'inner_fold': meta['inner_fold'], 'subject_id': sid, 'label': context.labels[sid], 'split': split_map[sid],
             'train_ids': meta['train_ids'], 'checkpoint': descriptor(cp), 'configuration': descriptor(CONFIG),
             'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
             'feature_source': descriptor(OUT/'input_manifest.json'), 'lambda': meta['lambda'],
             'fusion_method': 'ccml_with_bounded_modality_scales' if meta['kind'] == 'S' else 'ccml',
             'rho': 1.25, 'delta': 1., 'gamma': 1., 'scales': scales.tolist(),
             'views': {m: {'e_raw': raw[i, j].tolist(), 'e': e[i, j].tolist(),
                           'alpha': views['alpha'][i, j].tolist(), 'p': views['p'][i, j].tolist(),
                           'u': float(views['u'][i, j])} for j, m in enumerate(MODALITIES)},
             **{k: final[k][i].tolist() for k in ['m', 'e_con', 'e_cmp', 'e_final']},
             'min_sources': [[m for j, m in enumerate(MODALITIES) if e[i, j, k] == final['m'][i, k]] for k in range(2)],
             'p_final': final['p'][i].tolist(), 'u_final': float(final['u'][i]),
             'prediction': int(final['p'][i, 1] > .5),
             'same_trained_model_unit_scale_diagnostic': {'p_final': unscaled['p'][i].tolist(),
                  'u_final': float(unscaled['u'][i]), 'prediction': int(unscaled['p'][i, 1] > .5)}}
        validate_row(r)
        rows.append(r)
    return rows


def run_folder(seed, fold, inner, kind, strength):
    base = OUT/f'runs/seed_{seed}/fold_{fold}'
    return base/(f'inner_{inner}/lambda_{strength:g}' if inner is not None else f'outer_{kind}')


def train_model(model, inputs, y, seed, strength):
    """Only fitting tensors enter this function; no validation/test labels or arrays."""
    groups = [{'params': model.projections.parameters(), 'lr': 1e-4},
              {'params': model.heads.parameters(), 'lr': 1e-3}]
    if isinstance(model.scale_raw, nn.Parameter):
        groups.append({'params': [model.scale_raw], 'lr': .001})
    optimizer = torch.optim.Adamax(groups, weight_decay=0.)
    generator = torch.Generator().manual_seed(seed)
    history, batch_hash, updates = [], hashlib.sha256(), 0
    for epoch in range(1, 61):
        order = torch.randperm(len(y), generator=generator)
        batch_hash.update(order.numpy().astype('<i8').tobytes())
        total, regularizer, terms, grad, scale_grad = 0., 0., {}, [], []
        for start in range(0, len(y), 8):
            batch = order[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            e = model({m: inputs[m][batch] for m in MODALITIES})
            base_loss = ccml_loss(e, y[batch], epoch, **LOSS)
            penalty = strength*model.penalty()
            loss = base_loss+penalty
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'Missing/nonfinite gradient')
            scale_grad.append(float(model.scale_raw.grad.norm()) if model.scale_raw.grad is not None else 0.)
            grad.append(float(nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)))
            optimizer.step()
            require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameter')
            updates += 1
            total += float(loss.detach())*len(batch)
            regularizer += float(penalty.detach())*len(batch)
            with torch.no_grad():
                for k, v in loss_parts(e.detach(), y[batch], epoch).items():
                    terms[k] = terms.get(k, 0.)+v*len(batch)
        history.append({'epoch': epoch, 'train_objective': total/len(y),
                        'ccml_loss_terms': {k: v/len(y) for k, v in terms.items()},
                        'scale_shrinkage': regularizer/len(y), 'scales': model.scales().detach().tolist(),
                        'preclip_gradient_mean': float(np.mean(grad)), 'scale_gradient_mean': float(np.mean(scale_grad))})
    return history, updates, batch_hash.hexdigest()


def fit_run(context, data, seed, fold, inner, kind, strength, train_ids, split_map, selection=None):
    start = time.perf_counter()
    set_seed(seed)
    model = ScaleModel(seed, learned=kind == 'S')
    folder = run_folder(seed, fold, inner, kind, strength)
    folder.mkdir(parents=True, exist_ok=True)
    cp = folder/'checkpoint.pt'
    require(set(train_ids) == {s for s, split in split_map.items() if split == 'train'}, 'Fitting list differs from split')
    outer = context.folds[f'fold_{fold}']
    require(set(train_ids) <= set(outer['train']) and not set(train_ids)&set(outer['val']+outer['test']), 'Fitting leakage')
    meta = {'experiment': STAGES[kind], 'seed': seed, 'fold': fold, 'inner_fold': inner, 'kind': kind,
            'lambda': strength, 'train_ids': train_ids, 'evaluation_ids': [s for s in context.ids if s in split_map],
            'split_map': split_map, 'configuration': descriptor(CONFIG), 'inputs': descriptor(OUT/'input_manifest.json'),
            'selection': selection, 'initial_state_sha256': state_digest(model.state_dict()),
            'head_initialization_sha256': state_digest(make_initial_state(seed)),
            'epochs': 60, 'updates': 60*math.ceil(len(train_ids)/8),
            'trainable_parameters': sum(p.numel() for p in model.parameters()),
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'objective': 'full_CCML_plus_log_scale_shrinkage' if kind == 'S' else 'full_CCML',
            'torch': str(torch.__version__), 'device': 'cpu', 'supervised_parent': None}
    if cp.exists():
        side = read_json(folder/'checkpoint.json')
        saved = torch.load(verify_descriptor(side['checkpoint']), map_location='cpu', weights_only=True)
        require(saved['metadata'] == side['metadata'] and all(saved['metadata'][k] == v for k, v in meta.items()), 'Existing checkpoint provenance mismatch')
        model.load_state_dict(saved['state_dict'], strict=True)
    else:
        require(not (folder/'train_history.json').exists(), 'Incomplete prior run needs inspection')
        lookup = {s: i for i, s in enumerate(context.ids)}
        idx = [lookup[s] for s in train_ids]
        history, updates, batch_hash = train_model(model, {m: data[m][idx].clone() for m in MODALITIES},
                                                  torch.tensor([context.labels[s] for s in train_ids]), seed, strength)
        require(updates == meta['updates'], 'Update budget mismatch')
        meta['batch_order_sha256'] = batch_hash
        meta['final_scales'] = model.scales().detach().tolist()
        write_new(folder/'train_history.json', history)
        torch.save({'state_dict': model.state_dict(), 'metadata': meta}, cp)
        write_new(folder/'checkpoint.json', {'checkpoint': descriptor(cp), 'metadata': meta})
    rows = predictions(model, data, context, meta, cp, split_map)
    write_new(folder/'predictions.json', rows)
    write_new(folder/'complete.json', {name: descriptor(folder/name) for name in ['checkpoint.pt', 'checkpoint.json', 'train_history.json', 'predictions.json']})
    print(f'S run seed={seed} outer={fold} inner={inner} kind={kind} lambda={strength:g} updates={meta["updates"]} seconds={time.perf_counter()-start:.1f}', flush=True)
    return rows


def select_candidate(candidate_rows, outer_train_ids):
    """Only the legal inner held-out records enter selection."""
    require(set(candidate_rows) == set(LAMBDAS), 'Wrong candidate set')
    scores = {}
    for strength, rows in candidate_rows.items():
        require(all(r['split'] == 'inner_val' and r['inner_fold'] in [1, 2, 3] and
                    r['subject_id'] in outer_train_ids and r['subject_id'] not in r['train_ids'] and
                    set(r['train_ids']) <= set(outer_train_ids) for r in rows), 'Illegal selection subjects')
        require(Counter(r['subject_id'] for r in rows) == Counter(outer_train_ids), 'Incomplete inner OOF selection')
        metrics = {str(j): binary_metrics([r['label'] for r in rows if r['inner_fold'] == j],
                                          [r['p_final'][1] for r in rows if r['inner_fold'] == j]) for j in [1, 2, 3]}
        scores[strength] = {'fold_metrics': metrics,
                            'mean_nll': sum(v['nll'] for v in metrics.values())/3}
    best = min(v['mean_nll'] for v in scores.values())
    chosen = max(k for k, v in scores.items() if v['mean_nll'] <= best+1e-12)
    return {'selected_lambda': chosen, 'criterion': 'three inner validation fold mean final NLL; ties prefer stronger shrinkage',
            'candidate_scores': {str(k): v for k, v in scores.items()}, 'selection_subject_ids': outer_train_ids}


def load_setup():
    context = RecordContext()
    cfg = read_json(CONFIG)
    require(cfg == configuration(context), 'Locked configuration/source changed')
    require(read_json(OUT/'preflight.json')['passed'], 'Preflight missing')
    return context, cfg, load_data(context)


def train_inner():
    context, cfg, data = load_setup()
    splits = read_json(OUT/'nested_splits.json')['folds']
    selections = []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            selected_rows, sources = {}, []
            for strength in LAMBDAS:
                selected_rows[strength] = []
                for part in splits[str(fold)]:
                    split_map = {**dict.fromkeys(part['train_ids'], 'train'), **dict.fromkeys(part['val_ids'], 'inner_val')}
                    rows = fit_run(context, data, seed, fold, part['inner_fold'], 'S', strength, part['train_ids'], split_map)
                    selected_rows[strength] += [r for r in rows if r['split'] == 'inner_val']
                    sources.append(descriptor(run_folder(seed, fold, part['inner_fold'], 'S', strength)/'predictions.json'))
            choice = select_candidate(selected_rows, context.folds[f'fold_{fold}']['train'])
            choice.update(seed=seed, outer_fold=fold, configuration=descriptor(CONFIG), sources=sources,
                          outer_val_used=False, outer_test_used=False)
            target = OUT/f'selections/seed_{seed}/fold_{fold}.json'
            write_new(target, choice)
            selections.append(descriptor(target))
            print(f'S selection locked: seed={seed} outer={fold} lambda={choice["selected_lambda"]}', flush=True)
    write_new(OUT/'all_selections_locked.json', {'configuration': descriptor(CONFIG), 'selections': selections,
              'inner_training_runs': 90, 'inner_updates': 16200, 'outer_predictions_computed': False})


def train_outer():
    context, cfg, data = load_setup()
    lock = read_json(OUT/'all_selections_locked.json')
    require(len(lock['selections']) == 15 and lock['configuration'] == descriptor(CONFIG), 'All selections must be locked before outer evaluation')
    for item in lock['selections']:
        verify_descriptor(item)
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            f = context.folds[f'fold_{fold}']
            selection_path = OUT/f'selections/seed_{seed}/fold_{fold}.json'
            strength = read_json(selection_path)['selected_lambda']
            split_map = {s: split for split, ids in f.items() for s in ids}
            for kind, lam in [('control', 0.), ('S', strength)]:
                fit_run(context, data, seed, fold, None, kind, lam, f['train'], split_map, descriptor(selection_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'inner', 'outer', 'train'])
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'inner':
        train_inner()
    elif action == 'outer':
        train_outer()
    else:
        train_inner()
        train_outer()
