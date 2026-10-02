"""One fixed Adamax regularization contrast on the completed AS-NP family."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from . import as_projection as baseline
from .common import (WORK, TC, RecordContext, descriptor, verify_descriptor, read_json,
                     require, write_new, set_seed, digest, LOSS)
from .rebuild_eeg import loss_parts, validate_row
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ccml_loss
import numpy as np
import torch
from torch.nn import functional as F

OUT = TC/'experiments/CCML-v2-AS-NP-R-37-fulltext'
CONFIG = OUT/'configuration.json'
PLAN = WORK/'output/plans/CCML_v2_AS_NP_R_固定正则对照计划.md'
EXPERIMENT = OUT.name
PARENT = baseline.OUT
MODALITIES = baseline.MODALITIES
PROJECTION_COUNT, TOTAL_COUNT = baseline.PROJECTION_COUNT, baseline.TOTAL_COUNT
NormalizedProjectionHeads = baseline.NormalizedProjectionHeads
load_data, verify_initial_parents = baseline.load_data, baseline.verify_initial_parents
logits_all = baseline.logits_all
PAIR_KEYS = ['cohort_sha256', 'folds_sha256', 'subject_ids', 'seeds', 'folds',
             'projection_dimensions', 'projection_parameters', 'head_parameters', 'trainable_parameters',
             'normalization', 'eps', 'projection_lr', 'head_lr', 'epochs', 'batch_size', 'optimizer',
             'gradient_clip', 'ccml', 'dropout', 'device', 'threads', 'threshold', 'tie_class',
             'selection', 'formal_runs', 'updates', 'search', 'initialization']


def configuration(context):
    old = read_json(baseline.CONFIG)
    require(old == baseline.configuration(context), 'AS-NP source/configuration drift')
    return {**{k: old[k] for k in PAIR_KEYS}, 'experiment': EXPERIMENT, 'version': 1,
            'date': '2026-09-23', 'plan': descriptor(PLAN), 'primary_comparator': PARENT.name,
            'baseline_configuration': descriptor(baseline.CONFIG), 'baseline_status': descriptor(PARENT/'status.json'),
            'baseline_audit': descriptor(PARENT/'completion_audit.json'), 'weight_decay': 1e-4,
            'regularization_scope': 'all trainable weights and biases; coupled Adamax decay after CCML gradient clipping',
            'gate': {'f1_delta_min': .01, 'positive_seeds_min': 2, 'recall_delta_min': -.05},
            'scope': 'fixed development comparison on the same previously viewed 37 people',
            'sources': {p.relative_to(WORK).as_posix(): descriptor(p) for p in
                [Path(__file__), TC/'v2/complete_as_np_r.py', TC/'v2/test_as_np_r.py',
                 *[verify_descriptor(d) for d in old['sources'].values()]]}}


def protect_history():
    path = OUT/'history_before.json'
    if not path.exists():
        tracked = read_json(PARENT/'history_before.json')
        for p in [*PARENT.rglob('*'), baseline.PLAN, TC/'v2/as_projection.py',
                  TC/'v2/complete_as_projection.py', TC/'v2/test_as_projection.py',
                  *[(TC/'experiments')/f'CCML-v2-AS-NP-{stage}.log' for stage in ['preflight', 'train', 'completion']]]:
            if p.is_file():
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
    require(status['status'] == 'complete', 'AS-NP incomplete')
    for key in ['report', 'audit', 'configuration', 'plan']:
        verify_descriptor(status[key])
    cfg = configuration(context)
    write_new(CONFIG, cfg)
    tests = read_json(OUT/'test_results.json')
    require(tests['passed'] and tests['failures'] == tests['errors'] == 0, 'Tests incomplete')
    for item in tests['sources'].values():
        verify_descriptor(item)
    return context, cfg


def make_optimizer(model, cfg):
    return torch.optim.Adamax([
        {'params': model.projections.parameters(), 'lr': cfg['projection_lr']},
        {'params': model.heads.parameters(), 'lr': cfg['head_lr']}], weight_decay=cfg['weight_decay'])


def predictions(model, data, context, seed, fold, checkpoint):
    rows = baseline.predictions(model, data, context, seed, fold, checkpoint)
    for row in rows:
        row.update(experiment=EXPERIMENT, configuration=descriptor(CONFIG))
        validate_row(row)
    return rows


def preflight():
    context, cfg = setup()
    checks = []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed)
            data = load_data(context, seed, fold)
            verify_initial_parents(context, seed, fold, data)
            model = NormalizedProjectionHeads(data['initial'], seed)
            folder = PARENT/f'runs/seed_{seed}/fold_{fold}'
            for d in read_json(folder/'complete.json').values():
                verify_descriptor(d)
            cp = folder/'checkpoint.pt'
            saved = torch.load(cp, map_location='cpu', weights_only=True)
            meta = saved['metadata']
            require(meta['initial_state_sha256'] == state_digest(model.state_dict()) and
                    meta['source'] == data['source'] and meta['weight_decay'] == 0., 'Baseline pairing mismatch')
            initial_hash = state_digest(model.state_dict())
            model.load_state_dict(saved['state_dict'], strict=True)
            rows = baseline.predictions(model, data, context, seed, fold, cp)
            require(rows == read_json(folder/'predictions.json') and len(rows) == 37, 'Baseline replay mismatch')
            checks.append({'seed': seed, 'fold': fold, 'source': data['source'], 'checkpoint': descriptor(cp),
                           'initial_state_sha256': initial_hash, 'exact_reproduced_rows': 37})
    write_new(OUT/'preflight.json', {'passed': True, 'configuration': descriptor(CONFIG), 'checks': checks,
              'reconstructed_control_records': 555, 'protected_files_unchanged': protect_history()})
    print('AS-NP-R preflight passed: 15 matched initial states; 555 AS-NP records exactly replayed.', flush=True)


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
    base_meta = read_json(PARENT/f'runs/seed_{seed}/fold_{fold}/checkpoint.json')['metadata']
    meta = {'experiment': EXPERIMENT, 'seed': seed, 'fold': fold, 'train_ids': train_ids,
            'source': data['source'], 'cohort_sha256': context.cohort_sha256,
            'folds_sha256': context.folds_sha256, 'configuration': descriptor(CONFIG),
            'initial_state_sha256': initial_sha, 'initial_projection_sha256': state_digest(data['initial']),
            'head_initialization_sha256': state_digest(make_initial_state(seed)),
            'normalization': cfg['normalization'], 'eps': cfg['eps'],
            'projection_parameters': PROJECTION_COUNT, 'trainable_parameters': TOTAL_COUNT,
            'projection_lr': cfg['projection_lr'], 'head_lr': cfg['head_lr'], 'weight_decay': cfg['weight_decay'],
            'objective': 'full_CCML_only', 'epochs': 60, 'updates': 240,
            'selection': 'last_epoch', 'device': 'cpu', 'torch': str(torch.__version__), 'dropout': 'eval'}
    if cp.exists():
        side = read_json(cp.with_suffix('.json'))
        verify_descriptor(side['checkpoint'])
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        require(saved['metadata'] == side['metadata'] and
                all(saved['metadata'].get(k) == v for k, v in meta.items()), 'Existing AS-NP-R checkpoint mismatch')
        model.load_state_dict(saved['state_dict'], strict=True)
    else:
        lookup = {s: i for i, s in enumerate(context.ids)}
        selected = [lookup[s] for s in train_ids]
        require(not (set(train_ids) & set(context.folds[f'fold_{fold}']['val'] + context.folds[f'fold_{fold}']['test'])), 'Training leakage')
        x = {m: torch.from_numpy(data[m][selected]) for m in MODALITIES}
        y = torch.from_numpy(data['labels'][selected])
        optimizer = make_optimizer(model, cfg)
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
    print(f'Completed AS-NP-R seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-start:.1f}', flush=True)


def train():
    context, cfg = setup()
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG), 'AS-NP-R preflight incomplete')
    require(read_json(OUT/'test_results.json')['passed'], 'Tests incomplete')
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            fit(context, cfg, seed, fold)
    print(f'AS-NP-R training complete; {protect_history()} historical files unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'train'])
    (preflight if parser.parse_args().action == 'preflight' else train)()
