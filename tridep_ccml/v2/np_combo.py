"""Fixed N+P combination, preserving all historical experiment code and artifacts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from . import p as projection
from .nr import CandidateHeads, OUT as NR, VARIANTS, load_data as load_frozen_data
from .common import (WORK, TC, OUT as PARENT, CONTROL, LOSS, RecordContext, descriptor,
                     verify_descriptor, read_json, require, write_new, set_seed, digest)
from .resume_eeg import load_checkpoint
from .rebuild_eeg import loss_parts, validate_row
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ccml_loss, evidence_dc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = TC / 'experiments/CCML-v2-NP-37-fulltext'
CONFIG = OUT / 'configuration_v2.json'
PLAN = WORK / 'output/plans/CCML_v2_NP_固定组合实验计划.md'
EXPERIMENT = OUT.name
MODALITIES = projection.MODALITIES
PROJECTION_COUNT = projection.PROJECTION_COUNT
load_data = projection.load_data
logits_all = projection.logits_all


class NormalizedProjectionHeads(projection.ProjectionHeads):
    def __init__(self, projection_state, seed):
        super().__init__(projection_state, seed)
        self.normalizer = nn.LayerNorm(128, eps=1e-5, elementwise_affine=False)
        self.eval()

    def forward(self, features):
        return self.heads(self.normalizer(self.representations(features)))


def configuration(context):
    return {
        'experiment': EXPERIMENT, 'version': 2, 'registered_plan': descriptor(PLAN),
        'preflight_revision_reason': 'Use restricted generic torch loader for N; EEG legacy loader is scoped to its own experiment. No model, tolerance, or training change.',
        'previous_configuration': descriptor(OUT/'configuration.json'),
        'preserved_preflight_sources': {name: descriptor(OUT/'recovery'/name) for name in
                                       ['np_combo_preflight_v1.py', 'complete_np_v1.py']},
        'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
        'subject_ids': context.ids, 'seeds': [11, 22, 33], 'folds': [1, 2, 3, 4, 5],
        'parent_family': PARENT.name, 'primary_comparator': VARIANTS['N']['experiment'],
        'secondary_comparators': [CONTROL, projection.EXPERIMENT],
        'prior_status': {k: descriptor(root / 'status.json') for k, root in
                         [('E', PARENT), ('NR', NR), ('P', projection.OUT)]},
        'normalization': 'per_subject_per_view_LayerNorm_128_no_affine', 'eps': 1e-5,
        'projection_dimensions': projection.DIMENSIONS,
        'projection_parameters': PROJECTION_COUNT, 'head_parameters': 774,
        'trainable_parameters': 836742, 'projection_lr': 1e-4, 'head_lr': 1e-3,
        'epochs': 60, 'batch_size': 8, 'optimizer': 'Adamax', 'weight_decay': 0.,
        'gradient_clip': 1., 'ccml': LOSS, 'threshold': .5, 'tie_class': 0,
        'selection': 'fixed_last_epoch', 'dropout': 'eval', 'device': 'cpu', 'threads': 1,
        'formal_runs': 15, 'updates': 3600, 'search': False,
        'initialization': 'same-fold initial parent projections and fresh shared seed-specific heads; no N/P trained weights',
        'preflight_representation_rtol': 2e-5, 'preflight_representation_atol': 2e-5,
        'control_probability_atol': 2e-6, 'control_decisions_must_match': True,
        'stage_gate_vs_N': {'mean_f1_delta_min': .01, 'positive_f1_seeds_min': 2,
                            'recall_delta_min': -.05, 'nll_brier': 'report tradeoffs'},
        'interaction': 'NP - N - P + E-control; descriptive, not a significance test',
        'evaluation_scope': 'previously viewed 37-subject developmental cohort',
        'sources': {p.name: descriptor(p) for p in [Path(__file__), TC/'v2/p.py',
                    TC/'v2/nr.py', TC/'v2/common.py', TC/'v2/resume_eeg.py',
                    TC/'v2/rebuild_eeg.py', TC/'ccml.py', TC/'train.py', TC/'records.py',
                    TC/'v2/complete_np.py', TC/'v2/test_np.py']}}


def protect_history():
    path = OUT / 'history_before.json'
    if not path.exists():
        tracked = read_json(projection.OUT / 'history_before.json')
        for p in projection.OUT.rglob('*'):
            if p.is_file():
                tracked[p.relative_to(WORK).as_posix()] = digest(p)
        for p in [TC/'v2/p.py', TC/'v2/complete_p.py', TC/'v2/test_p.py',
                  *list((TC/'experiments').glob('CCML-v2-P-*.log'))]:
            tracked[p.relative_to(WORK).as_posix()] = digest(p)
        write_new(path, tracked)
    tracked = read_json(path)
    for path, expected in tracked.items():
        require(digest(WORK/path) == expected, f'Historical artifact changed: {path}')
    return len(tracked)


def setup():
    context = RecordContext()
    OUT.mkdir(parents=True, exist_ok=True)
    protect_history()
    cfg = configuration(context)
    for item in cfg['prior_status'].values():
        status = read_json(verify_descriptor(item))
        require(status['status'] == 'complete', 'Preceding stage incomplete')
        verify_descriptor(status['report'])
        verify_descriptor(status['audit'])
    write_new(CONFIG, cfg)
    return context, cfg


def preflight():
    context, cfg = setup()
    checks = []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed)
            data = load_data(context, seed, fold)
            frozen = load_frozen_data(context, seed, fold)
            model = NormalizedProjectionHeads(data['initial'], seed)
            require(sum(p.numel() for p in model.parameters()) == 836742, 'Wrong trainable count')
            require(state_digest(model.heads.state_dict()) == state_digest(make_initial_state(seed)), 'Head initialization changed')
            initial_hash = state_digest(model.state_dict())
            p_folder = projection.OUT / f'runs/seed_{seed}/fold_{fold}'
            p_meta = read_json(p_folder/'checkpoint.json')['metadata']
            require(initial_hash == p_meta['initial_state_sha256'], 'Initial N+P parameters differ from P')
            reps = projection.project_all(model, data).numpy()
            require(np.allclose(reps, frozen['features'], rtol=2e-5, atol=2e-5), 'Parent reconstruction mismatch')
            normalized = model.normalizer(torch.from_numpy(reps))
            original_normalized = model.normalizer(torch.from_numpy(frozen['features']))
            require(torch.allclose(normalized, original_normalized, rtol=2e-5, atol=2e-5), 'Normalized parent mismatch')
            row = {'seed': seed, 'fold': fold, 'manifest': data['source']['manifest'],
                   'initial_state_sha256': initial_hash,
                   'max_representation_error': float(np.max(np.abs(reps-frozen['features']))),
                   'max_normalized_error': float((normalized-original_normalized).abs().max()),
                   'comparators': {}}
            for name, folder in [('C', PARENT/'runs'/CONTROL/f'seed_{seed}/fold_{fold}'),
                                 ('N', NR/'runs'/VARIANTS['N']['experiment']/f'seed_{seed}/fold_{fold}')]:
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                cp = (load_checkpoint(folder/'checkpoint.pt') if name == 'C' else
                      torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True))
                head = CandidateHeads(name)
                head.load_state_dict(cp['state_dict'], strict=True)
                if name == 'N':
                    model.heads.load_state_dict(head.state_dict(), strict=True)
                    outputs = logits_all(model, data)
                else:
                    plain = projection.ProjectionHeads(data['initial'], seed)
                    plain.heads.load_state_dict(head.state_dict(), strict=True)
                    outputs = logits_all(plain, data)
                fused = evidence_dc(F.softplus(outputs))
                reference = read_json(folder/'predictions.json')
                require([r['subject_id'] for r in reference] == context.ids, 'Comparator identity mismatch')
                error = float(np.max(np.abs(fused['p'].numpy()-np.array([r['p_final'] for r in reference]))))
                require(error <= cfg['control_probability_atol'], 'Comparator probability mismatch')
                require((fused['p'][:, 1] > .5).int().tolist() == [r['prediction'] for r in reference], 'Comparator decisions changed')
                row['comparators'][name] = {'max_probability_error': error, 'all_decisions_identical': True}
            for item in read_json(p_folder/'complete.json').values():
                verify_descriptor(item)
            saved = torch.load(p_folder/'checkpoint.pt', map_location='cpu', weights_only=True)
            plain = projection.ProjectionHeads(data['initial'], seed)
            plain.load_state_dict(saved['state_dict'], strict=True)
            require(projection.predictions(plain, data, context, seed, fold, p_folder/'checkpoint.pt')
                    == read_json(p_folder/'predictions.json'), 'P reference replay failed')
            row['P_exact_replayed_rows'] = 37
            checks.append(row)
    write_new(OUT/'preflight.json', {'passed': True, 'configuration': descriptor(CONFIG),
              'cache_bundles': 15, 'comparator_records': 1665, 'checks': checks,
              'protected_files_unchanged': protect_history()})
    print('N+P preflight passed: 15 legal caches, 1665 comparator records.', flush=True)


def predictions(model, data, context, seed, fold, checkpoint):
    rows = projection.predictions(model, data, context, seed, fold, checkpoint)
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
            'projection_parameters': PROJECTION_COUNT, 'trainable_parameters': 836742,
            'projection_lr': cfg['projection_lr'], 'head_lr': cfg['head_lr'], 'weight_decay': 0.,
            'objective': 'full_CCML_only', 'epochs': 60, 'updates': 240,
            'selection': 'last_epoch', 'device': 'cpu', 'torch': str(torch.__version__), 'dropout': 'eval'}
    if cp.exists():
        side = read_json(cp.with_suffix('.json'))
        verify_descriptor(side['checkpoint'])
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        require(saved['metadata'] == side['metadata'] and
                all(saved['metadata'].get(k) == v for k, v in meta.items()), 'Existing N+P checkpoint mismatch')
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
    print(f'Completed NP seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-start:.1f}', flush=True)


def train():
    context, cfg = setup()
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG), 'N+P preflight incomplete')
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            fit(context, cfg, seed, fold)
    print(f'N+P training complete; {protect_history()} historical files unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'train'])
    action = parser.parse_args().action
    (preflight if action == 'preflight' else train)()
