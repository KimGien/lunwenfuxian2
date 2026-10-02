"""AS training: the fixed A runner, forked only for new artifact bindings and labels."""
from __future__ import annotations

import hashlib
import time
from .short_audio import OUT, CONFIG, STAGES, setup, protect_history, load_pool
from .common import descriptor, verify_descriptor, read_json, require, write_new, set_seed, LOSS
from .audio import (EEG, OLD_CONTROL, MODALITIES, load_old_data, make_model, forward_all,
                    save_npz, predict_rows, validate_row, loss_parts)
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ccml_loss
import numpy as np
import torch
from torch.nn import functional as F

def load_data(context, seed, fold, stage):
    old = load_old_data(context, seed, fold)
    if stage == 'adapt':
        pooled, source = load_pool(context)
        return {**old, 'pooled': pooled, 'source': {'frozen_parent': old['source'], 'speech_pool': source}}
    info = read_json(OUT/f'features/seed_{seed}/fold_{fold}.json')
    require(info['fit_subject_ids'] == context.folds[f'fold_{fold}']['train'] and
            info['seed'] == seed and info['fold'] == fold, 'New speech parent fitted on wrong subjects')
    verify_descriptor(info['parent_checkpoint'])
    require(info['configuration'] == descriptor(CONFIG), 'New parent configuration changed')
    with np.load(verify_descriptor(info['features']), allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    require(data['subject_ids'].tolist() == context.ids and np.array_equal(data['labels'], old['labels']) and
            np.array_equal(data['splits'], old['splits']), 'New parent labels/splits mismatch')
    require(data['features'].shape == (37, 3, 128) and np.isfinite(data['features']).all(), 'Invalid revised features')
    require(np.array_equal(data['features'][:, [0, 2]], old['features'][:, [0, 2]]), 'EEG/text representations changed')
    data['source'] = {'features': info['features'], 'manifest': descriptor(OUT/f'features/seed_{seed}/fold_{fold}.json'),
                      'parent_checkpoint': info['parent_checkpoint']}
    return data


def predictions(model, data, context, seed, fold, stage, checkpoint):
    rows = predict_rows(forward_all(model, data, stage), data, context, seed, fold,
                        STAGES[stage], descriptor(checkpoint), data['source'])
    for row in rows:
        row.update(configuration=descriptor(CONFIG), split=str(row['split']))
        validate_row(row)
    return rows


def fit(context, cfg, seed, fold, stage):
    start = time.perf_counter()
    set_seed(seed)
    data = load_data(context, seed, fold, stage)
    model = make_model(seed, stage)
    folder = OUT/'runs'/STAGES[stage]/f'seed_{seed}/fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    cp = folder/'checkpoint.pt'
    initial_sha = state_digest(model.state_dict())
    head_state = model.heads.state_dict() if stage == 'adapt' else model.state_dict()
    require(state_digest(head_state) == state_digest(make_initial_state(seed)), 'Fresh head mismatch')
    parameter_count = sum(p.numel() for p in model.parameters())
    require(parameter_count == (99206 if stage == 'adapt' else 774), 'Wrong trainable parameters')
    meta = {'experiment': STAGES[stage], 'stage': stage, 'seed': seed, 'fold': fold,
            'train_ids': context.folds[f'fold_{fold}']['train'], 'source': data['source'],
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'configuration': descriptor(CONFIG), 'initial_state_sha256': initial_sha,
            'head_initialization_sha256': state_digest(make_initial_state(seed)),
            'initial_projection_sha256': state_digest(model.speech_projection.state_dict()) if stage == 'adapt' else None,
            'trainable_parameters': parameter_count, 'normalization': cfg['normalization'][stage],
            'projection_lr': cfg['projection_lr'] if stage == 'adapt' else None, 'head_lr': cfg['head_lr'],
            'weight_decay': 0., 'epochs': 60, 'updates': 240, 'objective': 'full_CCML_only',
            'selection': 'last_epoch', 'device': 'cpu', 'torch': str(torch.__version__)}
    if cp.exists():
        side = read_json(cp.with_suffix('.json'))
        verify_descriptor(side['checkpoint'])
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        require(saved['metadata'] == side['metadata'] and all(saved['metadata'].get(k) == v for k, v in meta.items()), 'Checkpoint provenance mismatch')
        model.load_state_dict(saved['state_dict'], strict=True)
    else:
        lookup = {s: i for i, s in enumerate(context.ids)}
        selected = [lookup[s] for s in meta['train_ids']]
        require(not (set(meta['train_ids']) & set(context.folds[f'fold_{fold}']['val'] + context.folds[f'fold_{fold}']['test'])), 'Training leakage')
        x = torch.from_numpy(data['features'][selected])
        y = torch.from_numpy(data['labels'][selected])
        pooled = torch.from_numpy(data['pooled'][selected]) if stage == 'adapt' else None
        groups = ([{'params': model.speech_projection.parameters(), 'lr': cfg['projection_lr']},
                   {'params': model.heads.parameters(), 'lr': cfg['head_lr']}] if stage == 'adapt' else
                  [{'params': model.parameters(), 'lr': cfg['head_lr']}])
        optimizer = torch.optim.Adamax(groups, weight_decay=0.)
        generator = torch.Generator().manual_seed(seed)
        batch_hash = hashlib.sha256()
        history, updates = [], 0
        for epoch in range(1, 61):
            order = torch.randperm(len(selected), generator=generator)
            batch_hash.update(order.numpy().astype('<i8').tobytes())
            total, terms, norms, project_grads = 0., {}, [], []
            head_grads = {m: [] for m in MODALITIES}
            for i in range(0, len(order), 8):
                b = order[i:i+8]
                optimizer.zero_grad(set_to_none=True)
                logits = model(pooled[b], x[b]) if stage == 'adapt' else model(x[b])
                e = F.softplus(logits)
                loss = ccml_loss(e, y[b], epoch, **LOSS)
                loss.backward()
                heads = model.heads.heads if stage == 'adapt' else model.heads
                for name, head in heads.items():
                    require(all(p.grad is not None for p in head.parameters()), 'Missing head gradient')
                    head_grads[name].append(float(torch.sqrt(sum((p.grad**2).sum() for p in head.parameters()))))
                if stage == 'adapt':
                    require(all(p.grad is not None for p in model.speech_projection.parameters()), 'Missing speech gradient')
                    project_grads.append(float(torch.sqrt(sum((p.grad**2).sum() for p in model.speech_projection.parameters()))))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameters')
                total += float(loss.detach())*len(b)
                norms.append(float(norm))
                with torch.no_grad():
                    part = loss_parts(e.detach(), y[b], epoch)
                for k, v in part.items():
                    terms[k] = terms.get(k, 0.)+v*len(b)
            history.append({'epoch': epoch, 'train_loss': total/len(selected),
                            'loss_terms': {k: v/len(selected) for k, v in terms.items()},
                            'preclip_grad_norm_mean': float(np.mean(norms)),
                            'speech_projection_grad_norm_mean': float(np.mean(project_grads)) if project_grads else None,
                            'head_grad_norm_mean': {m: float(np.mean(v)) for m, v in head_grads.items()}})
        require(updates == 240, 'Update budget changed')
        meta['batch_order_sha256'] = batch_hash.hexdigest()
        base_meta = read_json(EEG/'runs'/OLD_CONTROL/f'seed_{seed}/fold_{fold}/checkpoint.json')['metadata']
        require(meta['batch_order_sha256'] == base_meta['batch_order_sha256'] and
                meta['head_initialization_sha256'] == base_meta['head_initialization_sha256'], 'Old control pairing failed')
        require(state_digest(model.state_dict()) != initial_sha, 'No model update')
        with cp.open('xb') as stream:
            torch.save({'state_dict': model.state_dict(), 'metadata': meta}, stream)
        write_new(cp.with_suffix('.json'), {'checkpoint': descriptor(cp), 'metadata': meta})
        write_new(folder/'train_history.json', history)
        before = forward_all(model, data, stage)
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        require(state_digest(saved['state_dict']) == state_digest(model.state_dict()), 'State reload changed')
        model.load_state_dict(saved['state_dict'], strict=True)
        require(torch.equal(before, forward_all(model, data, stage)), 'Output reload changed')
        write_new(folder/'reload_check.json', {'exact_state': True, 'exact_all_37_logits': True, 'restricted_loader': True})
    write_new(folder/'predictions.json', predictions(model, data, context, seed, fold, stage, cp))
    if stage == 'adapt':
        reps = forward_all(model, data, stage, representations=True).numpy()
        require(np.array_equal(reps[:, [0, 2]], data['features'][:, [0, 2]]), 'EEG/text changed')
        require(state_digest(model.speech_projection.state_dict()) != meta['initial_projection_sha256'], 'Speech projection did not change')
        path = OUT/f'features/seed_{seed}/fold_{fold}.npz'
        save_npz(path, {'features': reps, 'labels': data['labels'], 'splits': data['splits'], 'subject_ids': np.array(context.ids)})
        write_new(path.with_suffix('.json'), {'features': descriptor(path), 'parent_checkpoint': descriptor(cp),
                  'configuration': descriptor(CONFIG), 'seed': seed, 'fold': fold,
                  'fit_subject_ids': context.folds[f'fold_{fold}']['train'], 'source': data['source'],
                  'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
                  'eeg_text_exactly_unchanged': True, 'fitted_objective': 'full_CCML_only'})
    write_new(folder/'complete.json', {name: descriptor(folder/name) for name in
              ['checkpoint.pt', 'checkpoint.json', 'train_history.json', 'predictions.json', 'reload_check.json']})
    print(f'Completed AS-{stage} seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-start:.1f}', flush=True)


def train():
    context, cfg = setup()
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG), 'Preflight incomplete')
    require(read_json(OUT/'test_results.json')['passed'], 'Unit tests incomplete')
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            for stage in STAGES:
                fit(context, cfg, seed, fold, stage)
    print(f'Short-audio stage complete: 45 runs; {protect_history()} historical files unchanged.', flush=True)


if __name__ == '__main__':
    train()
