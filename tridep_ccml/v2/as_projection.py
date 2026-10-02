"""Finite three-projection adaptation on the coverage-complete AS parent."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from . import p as original_projection
from . import short_audio as as_parent
from . import short_train as as_train
from . import np_combo as old_np
from .common import (WORK, TC, RecordContext, descriptor, verify_descriptor, read_json,
                     require, write_new, set_seed, digest, LOSS)
from .rebuild_eeg import loss_parts, validate_row
from .resume_eeg import load_checkpoint as load_eeg_checkpoint
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ThreeViewHeads, ccml_loss, evidence_dc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = TC/'experiments/CCML-v2-AS-NP-37-fulltext'
CONFIG = OUT/'configuration.json'
PLAN = WORK/'output/plans/CCML_v2_AS_NP_同版本投影适配固定计划.md'
EXPERIMENT = OUT.name
PARENT = as_parent.OUT
CONTROL = as_parent.STAGES['N']
MODALITIES = ['eeg', 'speech', 'text']
DIMENSIONS = {'eeg': 1920, 'speech': 768, 'text': 1024}
PROJECTION_COUNT = 475520
TOTAL_COUNT = PROJECTION_COUNT+774
logits_all = original_projection.logits_all


class NormalizedProjectionHeads(nn.Module):
    def __init__(self, initial, seed, normalize=True):
        super().__init__()
        self.projections = nn.ModuleDict({
            'eeg': nn.Sequential(nn.Linear(1920, 128), nn.ReLU(), nn.Dropout(.4)),
            'speech': nn.Sequential(nn.Linear(768, 128), nn.ReLU()),
            'text': nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(.3))})
        self.projections.load_state_dict(initial, strict=True)
        self.heads = ThreeViewHeads()
        self.heads.load_state_dict(make_initial_state(seed), strict=True)
        self.normalizer = nn.LayerNorm(128, eps=1e-5, elementwise_affine=False) if normalize else nn.Identity()
        self.eval()

    def representations(self, data):
        return torch.stack([self.projections[m](data[m]) for m in MODALITIES], dim=1)

    def forward(self, data):
        return self.heads(self.normalizer(self.representations(data)))


def configuration(context):
    return {'experiment': EXPERIMENT, 'version': 1, 'date': '2026-09-23', 'plan': descriptor(PLAN),
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'subject_ids': context.ids, 'seeds': [11, 22, 33], 'folds': [1, 2, 3, 4, 5],
            'parent_family': PARENT.name, 'primary_comparator': CONTROL,
            'secondary_comparator': as_parent.STAGES['control'], 'old_N_NP': 'historical development references',
            'parent_status': descriptor(PARENT/'status.json'), 'parent_audit': descriptor(PARENT/'completion_audit.json'),
            'projection_dimensions': DIMENSIONS, 'projection_parameters': PROJECTION_COUNT,
            'head_parameters': 774, 'trainable_parameters': TOTAL_COUNT,
            'normalization': 'per_subject_per_view_LayerNorm_128_no_affine', 'eps': 1e-5,
            'projection_lr': 1e-4, 'head_lr': 1e-3, 'epochs': 60, 'batch_size': 8,
            'optimizer': 'Adamax', 'weight_decay': 0., 'gradient_clip': 1., 'ccml': LOSS,
            'dropout': {'eeg': '0.4_eval', 'speech': 'none', 'text': '0.3_eval'},
            'device': 'cpu', 'threads': 1, 'threshold': .5, 'tie_class': 0,
            'selection': 'fixed_last_epoch', 'formal_runs': 15, 'updates': 3600, 'search': False,
            'initialization': 'same-fold AS parent projections; fresh common seed heads; no trained AS-N or NP heads',
            'representation_rtol': 2e-5, 'representation_atol': 2e-5, 'probability_atol': 2e-6,
            'stage_gate_vs_AS_N': {'f1_delta_min': .01, 'positive_seeds_min': 2, 'recall_delta_min': -.05,
                                  'nll_brier': 'report deterioration; no automatic promotion on tradeoffs'},
            'scope': 'previously viewed 37-subject development cohort; frozen upstream supervised ancestry retained',
            'sources': {p.relative_to(WORK).as_posix(): descriptor(p) for p in [Path(__file__),
                TC/'v2/complete_as_projection.py', TC/'v2/test_as_projection.py', TC/'v2/p.py',
                TC/'v2/short_audio.py', TC/'v2/short_train.py', TC/'v2/audio.py', TC/'v2/nr.py',
                TC/'v2/common.py', TC/'v2/rebuild_eeg.py', TC/'v2/resume_eeg.py', TC/'ccml.py',
                TC/'train.py', TC/'records.py', TC/'metrics.py']}}


def protect_history():
    path = OUT/'history_before.json'
    if not path.exists():
        tracked = read_json(PARENT/'history_before.json')
        for p in PARENT.rglob('*'):
            if p.is_file():
                tracked[p.relative_to(WORK).as_posix()] = digest(p)
        for p in [TC/'v2/short_audio.py', TC/'v2/short_train.py', TC/'v2/complete_short.py',
                  TC/'v2/test_short.py', as_parent.PLAN, *list((TC/'experiments').glob('CCML-v2-AS-*.log'))]:
            tracked[p.relative_to(WORK).as_posix()] = digest(p)
        write_new(path, tracked)
    tracked = read_json(path)
    for name, value in tracked.items():
        require(digest(WORK/name) == value, f'Historical artifact changed: {name}')
    return len(tracked)


def setup():
    context = RecordContext()
    OUT.mkdir(parents=True, exist_ok=True)
    protect_history()
    status = read_json(PARENT/'status.json')
    require(status['status'] == 'complete', 'AS stage incomplete')
    verify_descriptor(status['report'])
    verify_descriptor(status['audit'])
    cfg = configuration(context)
    write_new(CONFIG, cfg)
    return context, cfg


def load_data(context, seed, fold):
    previous = original_projection.load_data(context, seed, fold)
    frozen = as_train.load_data(context, seed, fold, 'N')
    pooled, pool_source = as_parent.load_pool(context)
    cp = verify_descriptor(frozen['source']['parent_checkpoint'])
    saved = torch.load(cp, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    require(meta['experiment'] == as_parent.STAGES['adapt'] and meta['seed'] == seed and meta['fold'] == fold and
            meta['train_ids'] == context.folds[f'fold_{fold}']['train'], 'Speech projection from wrong parent/fold')
    require(meta['source']['speech_pool'] == pool_source, 'Speech parent used different coverage')
    initial = {k: v.clone() for k, v in previous['initial'].items() if k.startswith(('eeg.', 'text.'))}
    for key in ['weight', 'bias']:
        initial[f'speech.0.{key}'] = saved['state_dict'][f'speech_projection.0.{key}'].clone()
    require(np.array_equal(previous['labels'], frozen['labels']) and np.array_equal(previous['splits'], frozen['splits']), 'Parent labels/splits differ')
    data = {m: previous[m] for m in ['eeg', 'text']}
    data.update(speech=pooled, initial=initial, labels=frozen['labels'], splits=frozen['splits'],
                subject_ids=np.array(context.ids),
                source={'eeg_text_preprojection': previous['source'], 'speech_pool': pool_source,
                        'AS_parent_checkpoint': frozen['source']['parent_checkpoint'], 'AS_frozen_bundle': frozen['source']})
    for m, d in DIMENSIONS.items():
        require(data[m].shape == (37, d) and data[m].dtype == np.float32 and np.isfinite(data[m]).all(), 'Invalid projection input')
    require(sum(v.numel() for v in initial.values()) == PROJECTION_COUNT, 'Projection count changed')
    return data


def verify_initial_parents(context, seed, fold, data):
    original = read_json(verify_descriptor(data['source']['eeg_text_preprojection']['manifest']))
    for modality in ['eeg', 'text']:
        cp = verify_descriptor(original['sources'][modality]['checkpoint'])
        saved = (load_eeg_checkpoint(cp) if modality == 'eeg' else
                 torch.load(cp, map_location='cpu', weights_only=True))
        meta = saved['metadata'] if modality == 'eeg' else saved
        require(meta['seed'] == seed and meta['fold'] == fold and meta['train_ids'] == context.folds[f'fold_{fold}']['train'], 'Initial projection ancestry mismatch')
        prefix = 'eeg.classifier.2.' if modality == 'eeg' else 'classifier.0.'
        for key in ['weight', 'bias']:
            require(torch.equal(data['initial'][f'{modality}.0.{key}'], saved['state_dict'][prefix+key]), 'Projection differs from actual legal parent')
    saved = torch.load(verify_descriptor(data['source']['AS_parent_checkpoint']), map_location='cpu', weights_only=True)
    require(original['sources']['eeg']['checkpoint'] == saved['metadata']['source']['frozen_parent']['parent_checkpoint'],
            'EEG projection ancestry differs from AS frozen parent')
    for key in ['weight', 'bias']:
        require(torch.equal(data['initial'][f'speech.0.{key}'], saved['state_dict'][f'speech_projection.0.{key}']), 'Speech projection differs from AS parent')


def preflight():
    context, cfg = setup()
    require(read_json(OUT/'test_results.json')['passed'], 'Tests incomplete')
    checks = []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed)
            data = load_data(context, seed, fold)
            verify_initial_parents(context, seed, fold, data)
            frozen = as_train.load_data(context, seed, fold, 'N')
            model = NormalizedProjectionHeads(data['initial'], seed)
            require(sum(p.numel() for p in model.parameters()) == TOTAL_COUNT, 'Trainable count mismatch')
            reps = original_projection.project_all(model, data).numpy()
            require(np.allclose(reps, frozen['features'], rtol=cfg['representation_rtol'], atol=cfg['representation_atol']), 'AS representation reconstruction mismatch')
            checks_for_fold = {}
            for stage in ['control', 'N']:
                folder = PARENT/'runs'/as_parent.STAGES[stage]/f'seed_{seed}/fold_{fold}'
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                saved = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
                head_hash = state_digest(make_initial_state(seed))
                require(saved['metadata']['head_initialization_sha256'] == head_hash, 'Fresh initialization pairing mismatch')
                reconstructed = NormalizedProjectionHeads(data['initial'], seed, normalize=stage == 'N')
                reconstructed.heads.load_state_dict(saved['state_dict'], strict=True)
                probabilities = evidence_dc(F.softplus(logits_all(reconstructed, data)))['p'].numpy()
                rows = read_json(folder/'predictions.json')
                require([r['subject_id'] for r in rows] == context.ids, 'Comparator identity mismatch')
                error = float(np.max(np.abs(probabilities-np.array([r['p_final'] for r in rows]))))
                require(error <= cfg['probability_atol'], 'AS probabilities not reconstructed')
                require((probabilities[:, 1] > .5).astype(int).tolist() == [r['prediction'] for r in rows], 'AS reconstructed decisions changed')
                checks_for_fold[stage] = {'max_probability_error': error, 'all_decisions_identical': True,
                                          'checkpoint': descriptor(folder/'checkpoint.pt')}
            checks.append({'seed': seed, 'fold': fold, 'source': data['source'],
                           'initial_projection_sha256': state_digest(data['initial']),
                           'representation_max_abs_error': float(np.max(np.abs(reps-frozen['features']))),
                           'actual_parent_weights_verified': True, 'comparators': checks_for_fold})
    write_new(OUT/'preflight.json', {'passed': True, 'configuration': descriptor(CONFIG), 'checks': checks,
              'source_bundles': 15, 'reconstructed_control_records': 1110, 'protected_files_unchanged': protect_history()})
    print('AS-NP preflight passed: 15 legal source bundles; 1110 AS-control/N records reconstructed.', flush=True)


def predictions(model, data, context, seed, fold, checkpoint):
    rows = original_projection.predictions(model, data, context, seed, fold, checkpoint)
    for row in rows:
        row.update(experiment=EXPERIMENT, configuration=descriptor(CONFIG))
        validate_row(row)
    return rows


def fit(context, cfg, seed, fold):
    start = time.perf_counter()
    set_seed(seed)
    data = load_data(context, seed, fold)
    model = NormalizedProjectionHeads(data['initial'], seed)
    initial_sha = state_digest(model.state_dict())
    folder = OUT/f'runs/seed_{seed}/fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    cp = folder/'checkpoint.pt'
    train_ids = context.folds[f'fold_{fold}']['train']
    base_meta = read_json(PARENT/'runs'/CONTROL/f'seed_{seed}/fold_{fold}/checkpoint.json')['metadata']
    meta = {'experiment': EXPERIMENT, 'seed': seed, 'fold': fold, 'train_ids': train_ids,
            'source': data['source'], 'cohort_sha256': context.cohort_sha256,
            'folds_sha256': context.folds_sha256, 'configuration': descriptor(CONFIG),
            'initial_state_sha256': initial_sha, 'initial_projection_sha256': state_digest(data['initial']),
            'head_initialization_sha256': state_digest(make_initial_state(seed)),
            'normalization': cfg['normalization'], 'eps': cfg['eps'],
            'projection_parameters': PROJECTION_COUNT, 'trainable_parameters': TOTAL_COUNT,
            'projection_lr': cfg['projection_lr'], 'head_lr': cfg['head_lr'], 'weight_decay': 0.,
            'objective': 'full_CCML_only', 'epochs': 60, 'updates': 240,
            'selection': 'last_epoch', 'device': 'cpu', 'torch': str(torch.__version__), 'dropout': 'eval'}
    if cp.exists():
        side = read_json(cp.with_suffix('.json'))
        verify_descriptor(side['checkpoint'])
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        require(saved['metadata'] == side['metadata'] and
                all(saved['metadata'].get(k) == v for k, v in meta.items()), 'Existing AS-NP checkpoint mismatch')
        model.load_state_dict(saved['state_dict'], strict=True)
    else:
        lookup = {s: i for i, s in enumerate(context.ids)}
        selected = [lookup[s] for s in train_ids]
        require(not (set(train_ids) & set(context.folds[f'fold_{fold}']['val'] + context.folds[f'fold_{fold}']['test'])), 'Training leakage')
        x = {m: torch.from_numpy(data[m][selected]) for m in MODALITIES}
        y = torch.from_numpy(data['labels'][selected])
        optimizer = torch.optim.Adamax([
            {'params': model.projections.parameters(), 'lr': cfg['projection_lr']},
            {'params': model.heads.parameters(), 'lr': cfg['head_lr']}], weight_decay=0.)
        generator = torch.Generator().manual_seed(seed)
        batch_hash = hashlib.sha256()
        history = []
        updates = 0
        for epoch in range(1, cfg['epochs']+1):
            order = torch.randperm(len(selected), generator=generator)
            batch_hash.update(order.numpy().astype('<i8').tobytes())
            total, terms, norms = 0., {}, []
            project_grad, head_grad = ({m: [] for m in MODALITIES} for _ in range(2))
            for i in range(0, len(order), cfg['batch_size']):
                b = order[i:i+cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                e = F.softplus(model({m: x[m][b] for m in MODALITIES}))
                loss = ccml_loss(e, y[b], epoch, **LOSS)
                loss.backward()
                for modules, record in [(model.projections, project_grad), (model.heads.heads, head_grad)]:
                    for m, module in modules.items():
                        require(all(p.grad is not None for p in module.parameters()), 'Missing gradient')
                        record[m].append(float(torch.sqrt(sum((p.grad**2).sum() for p in module.parameters()))))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameters')
                total += float(loss.detach()) * len(b)
                norms.append(float(norm))
                with torch.no_grad():
                    parts = loss_parts(e.detach(), y[b], epoch)
                for k, v in parts.items():
                    terms[k] = terms.get(k, 0.) + v*len(b)
            history.append({'epoch': epoch, 'train_loss': total/len(selected),
                            'loss_terms': {k: v/len(selected) for k, v in terms.items()},
                            'preclip_grad_norm_mean': float(np.mean(norms)),
                            'projection_grad_norm_mean': {m: float(np.mean(v)) for m, v in project_grad.items()},
                            'head_grad_norm_mean': {m: float(np.mean(v)) for m, v in head_grad.items()}})
        require(updates == meta['updates'], 'Update budget changed')
        meta['batch_order_sha256'] = batch_hash.hexdigest()
        require(meta['batch_order_sha256'] == base_meta['batch_order_sha256'] and
                meta['head_initialization_sha256'] == base_meta['head_initialization_sha256'], 'Control pairing mismatch')
        require(state_digest(model.state_dict()) != initial_sha, 'No parameter updates')
        with cp.open('xb') as stream:
            torch.save({'state_dict': model.state_dict(), 'metadata': meta}, stream)
        write_new(cp.with_suffix('.json'), {'checkpoint': descriptor(cp), 'metadata': meta})
        write_new(folder/'train_history.json', history)
        before = logits_all(model, data)
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        require(state_digest(saved['state_dict']) == state_digest(model.state_dict()), 'Checkpoint state changed')
        model.load_state_dict(saved['state_dict'], strict=True)
        require(torch.equal(before, logits_all(model, data)), 'Checkpoint predictions changed')
        write_new(folder/'reload_check.json', {'exact_state': True, 'exact_all_37_logits': True, 'restricted_loader': True})
    rows = predictions(model, data, context, seed, fold, cp)
    write_new(folder/'predictions.json', rows)
    write_new(folder/'complete.json', {name: descriptor(folder/name) for name in
              ['checkpoint.pt', 'checkpoint.json', 'train_history.json', 'predictions.json', 'reload_check.json']})
    print(f'Completed AS-NP seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-start:.1f}', flush=True)


def train():
    context, cfg = setup()
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG), 'AS-NP preflight incomplete')
    require(read_json(OUT/'test_results.json')['passed'], 'Tests incomplete')
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            fit(context, cfg, seed, fold)
    print(f'AS-NP training complete; {protect_history()} historical files unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'train'])
    (preflight if parser.parse_args().action == 'preflight' else train)()
