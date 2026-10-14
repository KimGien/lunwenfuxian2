"""Fold-local EEG2Rep SSL. `--prepare` performs train-only CUDA engineering.

No external test input or label is forwarded. A real logical batch is scored
once; gradients are never accumulated from separately computed VICReg losses.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import math
from pathlib import Path
import random
import time

from tridep_ccml.tridep_stage1 import common as C
from tridep_ccml.tridep_eegpt import train_eval as T
from . import protocol as P, registry as R, storage as S
from .adapters import eeg2rep as E
from .paired_replay import rng_state, restore_rng, rng_equal

np, torch = C.np, C.torch
CONDITION = 'EEG2Rep-fold-local-ssl'
RAW_MANIFEST = C.OUT / 'input_manifest.json'
SETTINGS = S.OUT / 'ssl/engineering_settings.json'
ENGINEERING_CONTRACT = S.OUT / 'ssl/engineering_contract.json'
SNAPSHOTS = (60, 120, 200, 300, 500)
ONLINE_PREFIXES = ('core.InputEmbedding.', 'core.Norm.', 'core.Norm2.', 'core.contex_encoder.')


def require(value, message):
    if not value:
        raise ValueError(message)


def stream_seed(seed, fold, role):
    raw = f'{P.EXPERIMENT_ID}|{CONDITION}|{seed}|{fold}|{role}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)


def sampler_streams(seed, fold):
    return (torch.Generator(device='cpu').manual_seed(stream_seed(seed, fold, 'window_sampling')),
            random.Random(stream_seed(seed, fold, 'training_ssp')))


def training_order(person_count, generator):
    """Every train person contributes all60 windows, interleaved in each round."""
    require(isinstance(person_count, int) and person_count > 0, 'Positive train person count required')
    windows = [torch.randperm(60, generator=generator).tolist() for _ in range(person_count)]
    order = []
    for turn in range(60):
        people = torch.randperm(person_count, generator=generator).tolist()
        order.extend(person * 60 + windows[person][turn] for person in people)
    require(sorted(order) == list(range(person_count * 60)), 'SSL sampling lost/repeated a window')
    return order


def logical_batches(order, batch_size):
    require(isinstance(batch_size, int) and batch_size >= 4, 'Actual logical SSL batch must be at least4')
    result = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if result and len(result[-1]) == 1:
        # Preserve every window without a one-sample variance loss or duplication.
        final = result.pop()
        final.insert(0, result[-1].pop())
        result.append(final)
    require(result and all(2 <= len(batch) <= batch_size for batch in result), 'Invalid final logical SSL batch')
    return result


def order_digest(order):
    return hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest()


def mask_record(mask):
    """Two starts losslessly reconstruct the unchanged author SSP mask."""
    starts = [int(chunk[0]) for chunk in mask['chunk_indices']]
    return {'token_count': mask['token_count'], 'chunk_starts': starts,
            'visible_count': len(mask['visible_indices']), 'masked_count': len(mask['masked_indices'])}


def restore_mask(record):
    require(record['token_count'] == 500 and len(record['chunk_starts']) == 2
            and all(isinstance(s, int) and 0 <= s <= 375 for s in record['chunk_starts']), 'Invalid stored SSP chunk starts')
    chunks = [list(range(start, start + 125)) for start in record['chunk_starts']]
    visible = np.asarray(chunks, dtype=np.int64).ravel()
    masked = np.setdiff1d(np.arange(500), visible).tolist()
    require(record['visible_count'] == 250 and record['masked_count'] == len(masked), 'Stored SSP population differs')
    return {'token_count': 500, 'chunk_indices': chunks, 'visible_indices': visible.tolist(), 'masked_indices': masked}


class RawData:
    def __init__(self, manifest=RAW_MANIFEST):
        self.manifest = S.desc(manifest)
        data = S.read(manifest)
        self.ids = data['subject_ids']
        self.records = {r['subject_id']: r for r in data['records']}
        require(len(self.ids) == len(self.records) == 37 and set(self.records) == set(self.ids), 'Raw cohort coverage differs')
        self.cache = {}

    def raw(self, subject):
        if subject not in self.cache:
            self.cache[subject] = E.adapt_raw(np.load(S.checked(self.records[subject]['raw']), allow_pickle=False))
            require(self.cache[subject].shape == (60, 19, 1000), 'Raw person must retain all60 windows')
        return self.cache[subject]

    def batch(self, people, indices, device):
        require(all(isinstance(i, int) and 0 <= i < len(people) * 60 for i in indices), 'Window index outside declared people')
        x = np.stack([self.raw(people[i // 60])[i % 60] for i in indices])
        return torch.as_tensor(x, dtype=torch.float32, device=device)


def autocast(precision, device):
    require(precision in ('float32', 'bfloat16'), 'Unsupported registered SSL precision')
    if precision == 'float32':
        return nullcontext()
    require(str(device).startswith('cuda'), 'Formal bfloat16 engineering requires CUDA')
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def validation_manifest(folder, split, seed, fold, batch_size, source):
    path = folder / 'validation_masks.json'
    groups = logical_batches(list(range(len(split['val']) * 60)), batch_size)
    records = []
    for group, indices in enumerate(groups):
        mask = E.make_ssp_mask(rng=random.Random(stream_seed(seed, fold, f'fixed_val_mask_{group}')))
        records.append({'batch': group, 'window_indices': indices, 'mask': mask_record(mask)})
    value = {'condition': CONDITION, 'seed': seed, 'fold': fold, 'val_ids': split['val'],
             'windows_per_person': 60, 'actual_logical_batch': batch_size, 'records': records,
             'source': source, 'registry': R.registration_descriptor(),
             'aggregation': 'arithmetic mean of fixed logical-batch author SSL losses',
             'validation_updates': 0, 'validation_ema_updates': 0}
    S.save(path, value, immutable=True)
    return value


def component_values(output, loss):
    context = output['context'].detach().float().mean(dim=1)
    predicted = output['prediction_masked'].detach().float().mean(dim=1)
    return {**{key: float(loss[key].detach().cpu()) for key in ('loss', 'alignment_mse', 'variance_hinge', 'offdiagonal_covariance')},
            'context_feature_std_mean': float(context.std(dim=0, unbiased=True).mean().cpu()),
            'prediction_feature_std_mean': float(predicted.std(dim=0, unbiased=True).mean().cpu())}


def validation(model, data, split, manifest, precision):
    """Fixed masks and all180 validation windows; no target update or test access."""
    state = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    before = C.state_digest(model.state_dict())
    model.eval()
    rows = []
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for record in manifest['records']:
                x = data.batch(split['val'], record['window_indices'], device)
                with autocast(precision, device):
                    output = model.pretrain_forward(x, mask=restore_mask(record['mask']))
                    loss = E.ssl_loss(output)
                rows.append({'batch': record['batch'], 'windows': len(x), **component_values(output, loss)})
        require(sum(r['windows'] for r in rows) == len(split['val']) * 60, 'Validation window coverage differs')
        require(before == C.state_digest(model.state_dict()), 'SSL validation changed model/EMA state')
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(state)
    require(rng_equal(state, rng_state()), 'SSL validation RNG restore differs')
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ('batch', 'windows')}


def optimizer_for(model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)


def online_state(state):
    selected = {name: value for name, value in state.items() if name.startswith(ONLINE_PREFIXES)}
    require(selected, 'EEG2Rep online representation parameters missing')
    return selected


def changed_online_names(initial, selected):
    before, after = online_state(initial), online_state(selected)
    require(before.keys() == after.keys(), 'Online representation parameter names differ')
    return [name for name in before if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())]


def require_online_change(initial, selected):
    names = changed_online_names(initial, selected)
    require(names, 'SSL online backbone did not change; predictor/target EMA changes alone are insufficient')
    return names


def sampler_state(generator, mask_rng):
    return {'window_generator': generator.get_state().clone(), 'mask_rng': mask_rng.getstate()}


def restore_sampler(state, generator, mask_rng):
    generator.set_state(state['window_generator'])
    mask_rng.setstate(state['mask_rng'])


def state_fragment(model, optimizer, generator, mask_rng, epoch, updates):
    return {'state_dict': T._cpu(model.state_dict()), 'optimizer_state_dict': T._cpu(optimizer.state_dict()),
            'rng': rng_state(), 'sampling': sampler_state(generator, mask_rng), 'epoch': epoch,
            'online_state_sha256': C.state_digest(online_state(model.state_dict())),
            'optimizer_updates': updates, 'ema_updates': updates}


def checkpoint(model, optimizer, generator, mask_rng, epoch, history, order, best, extended, identity):
    return {**state_fragment(model, optimizer, generator, mask_rng, epoch, history[-1]['updates_completed'] if history else 0),
            'history': history, 'order': order, 'best': best, 'extended': extended,
            'scheduler_state_dict': {'policy': 'ssl', 'completed_epoch': epoch, 'extended': extended},
            'sampler_position': {'epoch_completed': epoch, 'next_epoch': epoch + 1, 'next_batch': 0},
            'identity': identity, 'configuration': E.DEFAULT_CONFIG, 'ssl_contract': E.SSL_CONTRACT}


def best_checkpoint(best, history, order, identity):
    return {**best, 'identity': identity, 'configuration': E.DEFAULT_CONFIG, 'ssl_contract': E.SSL_CONTRACT,
            'history': history[:best['epoch']], 'order': order[:best['epoch']], 'stage': 'ssl'}


def lock_prepared_contract(settings):
    require(settings['passed'] is True, 'Successful CUDA engineering check missing')
    for source in settings['prepared_sources'] + [settings['probe_checkpoint']]:
        S.checked(source)
    if not ENGINEERING_CONTRACT.exists():
        # The real GPU probe was durably accepted; reconstruct only its static
        # parameter names after an interrupted contract write, with no new fit.
        model, audit = E.build_model('cpu', stage='ssl')
        require(C.state_digest(torch.load(S.checked(settings['probe_checkpoint']), map_location='cpu', weights_only=False)['state_dict'])
                == settings['model_state_sha256'], 'Accepted engineering checkpoint changed')
        contract = {'input_contract': settings, 'trainable_parameter_names': audit['trainable_parameter_names'],
                    'frozen_parameter_names': audit['frozen_parameter_names'],
                    'implementation_sources': settings['prepared_sources'] + [S.desc(SETTINGS), settings['probe_checkpoint']],
                    'weight_sources': [], 'forward_checked': True, 'gradient_checked': True, 'resume_checked': True,
                    'training_only_engineering_checked': True, 'author_defect_repairs': E.AUTHOR_DEFECT_REPAIRS,
                    'target_frozen_gradient_but_ema_state_mutable': True}
        S.save(ENGINEERING_CONTRACT, contract, immutable=True)
        del model
    contract = S.read(ENGINEERING_CONTRACT)
    for source in contract['implementation_sources']:
        S.checked(source)
    R.lock_condition(CONDITION, contract)
    for seed in P.SEEDS:
        for fold in P.FOLDS:
            R.record_fit_status(CONDITION, seed, fold, 'ready')


def validate_resume(saved, model, identity, people, batch_size):
    epoch = saved['epoch']
    steps = len(logical_batches(list(range(len(people) * 60)), batch_size))
    require(saved['identity'] == identity and isinstance(epoch, int) and 0 <= epoch <= 500,
            'SSL resume identity/epoch differs')
    require(saved['configuration'] == E.DEFAULT_CONFIG and saved['ssl_contract'] == E.SSL_CONTRACT,
            'SSL resume author method hyperparameters differ')
    require(len(saved['history']) == len(saved['order']) == epoch and saved['optimizer_updates'] == saved['ema_updates'] == epoch * steps,
            'SSL resume counts/EMA coverage differ')
    require(saved['sampler_position'] == {'epoch_completed': epoch, 'next_epoch': epoch + 1, 'next_batch': 0}
            and saved['scheduler_state_dict'] == {'policy': 'ssl', 'completed_epoch': epoch, 'extended': saved['extended']},
            'SSL resume sampler/scheduler differs')
    for index, row in enumerate(saved['history']):
        require(row['epoch'] == index + 1 and row['updates_completed'] == row['ema_updates_completed'] == (index + 1) * steps
                and sorted(saved['order'][index]) == list(range(len(people) * 60)), 'SSL resume window/epoch coverage differs')
        require(len(row['masks']) == steps and all(math.isfinite(row[k]) for k in ('train_ssl_loss', 'val_ssl_loss')),
                'SSL resume mask/loss history invalid')
        for mask in row['masks']:
            restore_mask(mask)
    require(all(torch.isfinite(v).all() for v in saved['state_dict'].values()), 'Nonfinite SSL resume model')
    require(C.state_digest(online_state(saved['state_dict'])) == saved['online_state_sha256'], 'SSL resume online state hash differs')
    states = saved['optimizer_state_dict']['state']
    require((epoch == 0 and not states) or (len(states) == len([p for p in model.parameters() if p.requires_grad])
            and all(float(v['step']) == epoch * steps for v in states.values())), 'SSL resume optimizer state count differs')
    require(len(saved['rng']['torch_cuda']) == torch.cuda.device_count(), 'SSL resume CUDA RNG device coverage differs')
    if epoch:
        decision = P.training_decision('ssl', [row['val_ssl_loss'] for row in saved['history']], extended=saved['extended'])
        require(saved['best'] is not None and saved['best']['epoch'] == decision['best_epoch']
                and C.state_digest(saved['best']['state_dict']) == saved['best']['state_sha256']
                and C.state_digest(online_state(saved['best']['state_dict'])) == saved['best']['online_state_sha256'],
                'SSL resume selected-best state differs')
    else:
        require(saved['best'] is None, 'Initial SSL checkpoint has a selected state')


def validate_fit_context(seed, fold, data, split):
    registration = R.load_registration()
    expected = next(item for item in registration['fits']
                    if item['condition'] == CONDITION and item['seed'] == seed and item['fold'] == fold)
    require(data.ids == registration['subject_ids'] and split == {role: expected[role + '_ids'] for role in ('train', 'val', 'test')},
            'SSL fit people or original fold assignments differ')


def prepare(*, logical_batch=64, precision='bfloat16'):
    """Only a successful real CUDA logical-batch probe may mark this condition ready."""
    R.load_registration()
    engineering = S.OUT / 'engineering' / (CONDITION + '.json')
    if engineering.exists():
        R._condition_lock(CONDITION)
        settings = S.read(SETTINGS)
        require(settings['actual_logical_batch'] == logical_batch and settings['precision'] == precision,
                'Existing SSL resource lock differs; preserve it')
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                if not (R.fit_directory(CONDITION, seed, fold) / 'started.json').exists() and R.fit_status(CONDITION, seed, fold)['status'] != 'ready':
                    R.record_fit_status(CONDITION, seed, fold, 'ready')
        return settings
    if SETTINGS.exists():
        settings = S.read(SETTINGS)
        require(settings['actual_logical_batch'] == logical_batch and settings['precision'] == precision,
                'Prepared SSL resource settings differ')
        lock_prepared_contract(settings)
        return settings
    require(4 <= logical_batch <= 64, 'SSL actual batch must be4..64; requested true logical batch is64')
    lock, folds, _ = C.context()
    data = RawData()
    require(data.ids == lock['subject_ids'], 'SSL raw cohort differs')
    generator, mask_rng = sampler_streams(11, 1)
    order = training_order(len(folds['fold_1']['train']), generator)
    ii = logical_batches(order, logical_batch)[0]
    model = optimizer = x = None
    costs = {'encoder_window_forwards_started': 0, 'encoder_window_forwards_completed': 0,
             'engineering_gradient_passes_started': 0, 'engineering_gradient_passes_completed': 0,
             'optimizer_updates': 0, 'ema_updates': 0}
    began = time.perf_counter()
    try:
        C.runtime(11)
        require(precision != 'bfloat16' or torch.cuda.is_bf16_supported(), 'CUDA bfloat16 not supported')
        model, audit = E.build_model('cuda', stage='ssl')
        optimizer = optimizer_for(model)
        torch.cuda.reset_peak_memory_stats()
        x = data.batch(folds['fold_1']['train'], ii, 'cuda')
        costs['encoder_window_forwards_started'] = len(ii)
        with autocast(precision, 'cuda'):
            output = model.pretrain_forward(x, mask_rng=mask_rng)
            loss = E.ssl_loss(output)
        costs['encoder_window_forwards_completed'] = len(ii)
        costs['engineering_gradient_passes_started'] = 1
        loss['loss'].backward()
        costs['engineering_gradient_passes_completed'] = 1
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad),
                'SSL logical-batch engineering gradients invalid')
        require(all(p.grad is None and not p.requires_grad for p in model.core.target_encoder.parameters()), 'SSL target received gradient')
        before = C.state_digest(model.state_dict())
        state = rng_state()
        saved = state_fragment(model, optimizer, generator, mask_rng, 0, 0)
        probe = S.OUT / 'ssl/engineering_resume.pt'
        S.save_torch(probe, saved)
        restored = torch.load(probe, map_location='cpu', weights_only=False)
        model.load_state_dict(restored['state_dict'], strict=True)
        optimizer.load_state_dict(restored['optimizer_state_dict'])
        restore_rng(restored['rng']); restore_sampler(restored['sampling'], generator, mask_rng)
        require(before == C.state_digest(model.state_dict()) and rng_equal(state, rng_state()), 'SSL full-state engineering restore differs')
        settings = {'passed': True, 'actual_logical_batch': logical_batch, 'requested_logical_batch': 64,
                    'precision': precision, 'loss_precision': 'float32', 'gradient_accumulation': False,
                    'logical_batch64_equivalent_claimed': logical_batch == 64,
                    'method_difference': None if logical_batch == 64 else 'registered smaller real logical batch; VICReg differs from batch64',
                    'raw_manifest': data.manifest, 'configuration': E.DEFAULT_CONFIG, 'ssl_contract': E.SSL_CONTRACT,
                    'forward_backward_seconds': time.perf_counter() - began, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'engineering_optimizer_updates': 0, 'engineering_gradient_passes': 1, 'engineering_ema_updates': 0,
                    'actual_engineering_cost': costs,
                    'probe_windows': [{'subject_id': folds['fold_1']['train'][i // 60], 'window_index': i % 60} for i in ii]}
        sources = [S.desc(p) for p in (Path(__file__), Path(E.__file__), Path(P.__file__), Path(S.__file__),
                   S.WORK / 'tridep_ccml/eeg_all_candidates/paired_replay.py', Path(C.__file__),
                   Path(T.__file__), E.ROOT / 'author_source_binding.json', RAW_MANIFEST)]
        sources.extend(S.desc(E.ROOT / name) for name in E.SOURCE_SHA256)
        settings.update(prepared_sources=sources, probe_checkpoint=S.desc(probe), model_state_sha256=before)
        S.save(SETTINGS, settings, immutable=True)
        lock_prepared_contract(settings)
        return settings
    except BaseException as error:
        reason = f'{type(error).__name__}: {error}'
        S.save(S.OUT / 'ssl/engineering_attempts' / f'{time.time_ns()}.json', {
            'passed': False, 'actual_batch_attempted': logical_batch, 'precision': precision,
            'error': reason, 'optimizer_updates': 0, 'actual_engineering_cost': costs, 'formal_fits': 0, 'time': S.now()})
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                R.record_fit_status(CONDITION, seed, fold, 'blocked', reason=reason)
        raise
    finally:
        del model, optimizer, x
        gc.collect()
        torch.cuda.empty_cache()


def fit_one(seed, fold, data, split, *, resume=False, device='cuda'):
    R._condition_lock(CONDITION)
    validate_fit_context(seed, fold, data, split)
    settings = S.read(SETTINGS)
    batch_size, precision = settings['actual_logical_batch'], settings['precision']
    folder = R.fit_directory(CONDITION, seed, fold)
    if (folder / 'complete.json').exists():
        evidence = S.read(folder / 'complete.json'); R._validate_complete(evidence, folder)
        if R.fit_status(CONDITION, seed, fold)['status'] != 'training_complete':
            R.record_fit_status(CONDITION, seed, fold, 'training_complete', evidence=evidence)
        return S.read(folder / 'metadata.json')
    partial = (folder / 'started.json').exists()
    require(not partial or resume, 'Partial SSL fit requires complete-state --resume')
    identity = {'condition': CONDITION, 'seed': seed, 'fold': fold, 'train_ids': split['train'], 'val_ids': split['val'],
                'registry': R.registration_descriptor(), 'source': data.manifest, 'settings': S.desc(SETTINGS),
                'encoder_fit_subject_ids': split['train'], 'test_forward_count': 0}
    R.record_fit_status(CONDITION, seed, fold, 'training')
    if not partial:
        S.save(folder / 'started.json', {**identity, 'started_at': S.now()}, immutable=True)
    vm = validation_manifest(folder, split, seed, fold, batch_size, data.manifest)
    identity['validation_mask_manifest'] = S.desc(folder / 'validation_masks.json')
    C.runtime(seed)
    model, audit = E.build_model(device, stage='ssl')
    optimizer = optimizer_for(model)
    generator, mask_rng = sampler_streams(seed, fold)
    initial = C.state_digest(model.state_dict())
    initial_online = T._cpu(online_state(model.state_dict()))
    unused_initial = C.state_digest(model.core.predict_head.state_dict())
    identity['initial_state_sha256'] = initial
    identity['initial_online_state_sha256'] = C.state_digest(initial_online)
    history, order, best, epoch, extended = [], [], None, 0, False
    try:
        if partial:
            path = folder / 'latest.pt' if (folder / 'latest.pt').exists() else folder / 'initial.pt'
            if not path.exists():
                require(not (folder / 'progress.json').exists(), 'Missing SSL state after optimizer work was started')
                S.save_torch(path, checkpoint(model, optimizer, generator, mask_rng, epoch, history, order, best, extended, identity), immutable=True)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            validate_resume(saved, model, identity, split['train'], batch_size)
            model.load_state_dict(saved['state_dict'], strict=True); optimizer.load_state_dict(saved['optimizer_state_dict'])
            restore_rng(saved['rng']); restore_sampler(saved['sampling'], generator, mask_rng)
            require(rng_equal(saved['rng'], rng_state()), 'SSL RNG failed to restore')
            history, order, best, epoch, extended = saved['history'], saved['order'], saved['best'], saved['epoch'], saved['extended']
            if epoch in SNAPSHOTS:
                S.save_torch(folder / f'epoch_{epoch}.pt', saved, immutable=True)
            if best is not None:
                S.save_torch(folder / 'best.pt', best_checkpoint(best, history, order, identity))
        else:
            S.save_torch(folder / 'initial.pt', checkpoint(model, optimizer, generator, mask_rng, epoch, history, order, best, extended, identity), immutable=True)
        durable_epoch = epoch
        updates = history[-1]['updates_completed'] if history else 0
        start_updates = updates
        started = time.perf_counter()
        decision = P.training_decision('ssl', [r['val_ssl_loss'] for r in history], extended=extended) if history else {'stop': False}
        while not decision['stop']:
            epoch += 1
            model.train()
            for group in optimizer.param_groups:
                group['lr'] = 1e-4 * P.learning_rate_multiplier('ssl', epoch)
            indices = training_order(len(split['train']), generator)
            batches = logical_batches(indices, batch_size)
            values, masks = [], []
            for batch_index, ii in enumerate(batches):
                S.save(folder / 'progress.json', {'epoch': epoch, 'logical_batch': batch_index,
                       'updates_started': updates + 1, 'updates_completed': updates, 'ema_updates_completed': updates,
                       'train_windows': [{'subject_id': split['train'][i // 60], 'window_index': i % 60} for i in ii]})
                x = data.batch(split['train'], ii, device)
                mask = E.make_ssp_mask(rng=mask_rng)
                optimizer.zero_grad(set_to_none=True)
                with autocast(precision, device):
                    output = model.pretrain_forward(x, mask=mask)
                    loss = E.ssl_loss(output)
                loss['loss'].backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad),
                        'Nonfinite/disconnected SSL trainable gradient')
                require(all(p.grad is None for p in model.core.target_encoder.parameters()), 'SSL target received a training gradient')
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step()
                model.momentum_update()  # Exactly once per logical optimizer step, never per microbatch.
                updates += 1
                require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite SSL model/EMA parameter')
                values.append(component_values(output, loss)); masks.append(mask_record(mask))
                S.save(folder / 'progress.json', {'epoch': epoch, 'updates_completed': updates,
                       'gradient_passes_completed': updates, 'ema_updates_completed': updates})
            val = validation(model, data, split, vm, precision)
            row = {'epoch': epoch, 'updates_completed': updates, 'gradient_passes_completed': updates,
                   'ema_updates_completed': updates, 'train_windows': len(indices), 'logical_batch_sizes': [len(b) for b in batches],
                   'training_order_sha256': order_digest(indices), 'masks': masks,
                   'train_ssl_loss': float(np.mean([v['loss'] for v in values])), 'val_ssl_loss': val['loss'],
                   'lr': optimizer.param_groups[0]['lr'],
                   'train_components': {k: float(np.mean([v[k] for v in values])) for k in values[0]}, 'val_components': val}
            history.append(row); order.append(indices)
            decision = P.training_decision('ssl', [r['val_ssl_loss'] for r in history], extended=extended)
            if decision['best_epoch'] == epoch:
                best = {**state_fragment(model, optimizer, generator, mask_rng, epoch, updates),
                        'state_sha256': C.state_digest(model.state_dict()), 'validation_loss': val['loss']}
            if decision['extend']:
                extended = True
            payload = checkpoint(model, optimizer, generator, mask_rng, epoch, history, order, best, extended, identity)
            S.save_torch(folder / 'latest.pt', payload)
            durable_epoch = epoch
            if best['epoch'] == epoch:
                S.save_torch(folder / 'best.pt', best_checkpoint(best, history, order, identity))
            S.save(folder / 'train_history.json', history)
            S.save(folder / 'validation_history.json', [{'epoch': r['epoch'], 'val_ssl_loss': r['val_ssl_loss'], **r['val_components']} for r in history])
            if epoch in SNAPSHOTS:
                S.save_torch(folder / f'epoch_{epoch}.pt', payload, immutable=True)
            if epoch % 20 == 0:
                print(f'TRAIN {CONDITION} seed={seed} fold={fold} epoch={epoch} best={decision["best_epoch"]} steps={updates}', flush=True)
        require(best is not None and best['epoch'] == decision['best_epoch'], 'SSL selected state missing')
        require(best['state_sha256'] != initial, 'Selected SSL encoder unchanged from initialization')
        online_changed = require_online_change(initial_online, best['state_dict'])
        terminal_online = C.state_digest(online_state(model.state_dict()))
        terminal_online_changed = changed_online_names(initial_online, model.state_dict())
        selected_payload = best_checkpoint(best, history, order, identity)
        S.save_torch(folder / 'best.pt', selected_payload, immutable=True)
        model.load_state_dict(best['state_dict'], strict=True); model.eval()
        require(C.state_digest(model.core.predict_head.state_dict()) == unused_initial, 'Unused original window head unexpectedly changed')
        with torch.inference_mode(), autocast(precision, device):
            probe = data.batch(split['train'], [0, 1], device)
            expected = E.encode_windows(model, probe)
            reloaded = copy.deepcopy(model)
            reloaded.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)['state_dict'], strict=True)
            require(torch.equal(expected, E.encode_windows(reloaded, probe)), 'Selected SSL encoder reload differs')
        del reloaded
        S.save(folder / 'train_history.json', history)
        S.save(folder / 'validation_history.json', [{'epoch': r['epoch'], 'val_ssl_loss': r['val_ssl_loss'], **r['val_components']} for r in history])
        meta = {**identity, 'epochs': epoch, 'updates': updates, 'gradient_passes': updates, 'ema_updates': updates,
                'best_epoch': best['epoch'], 'selected_state_sha256': best['state_sha256'], 'stopping_reason': decision['reason'],
                'extended': extended, 'actual_logical_batch': batch_size, 'precision': precision,
                'loss_precision': 'float32', 'parameters': audit['parameters'], 'author_defect_repairs': E.AUTHOR_DEFECT_REPAIRS,
                'noncollapse_validation_context_std_mean': history[best['epoch'] - 1]['val_components']['context_feature_std_mean'],
                'selected_online_state_sha256': best['online_state_sha256'],
                'terminal_online_state_sha256': terminal_online,
                'online_changed_parameter_names': online_changed, 'online_changed_parameter_count': len(online_changed),
                'terminal_online_changed_parameter_count': len(terminal_online_changed),
                'online_backbone_update_verified': True,
                'training_order_sha256': order_digest(np.asarray(order, dtype=np.int64)),
                'seconds_this_attempt': time.perf_counter() - started, 'sampler_rng_saved': True, 'validation_updates': 0}
        if (folder / 'metadata.json').exists():
            previous = S.read(folder / 'metadata.json')
            require(all(previous[k] == meta[k] for k in ('condition', 'seed', 'fold', 'epochs', 'updates', 'best_epoch', 'selected_state_sha256')),
                    'Existing selected SSL metadata differs')
            meta = previous
        else:
            S.save(folder / 'metadata.json', meta, immutable=True)
        evidence = {'passed': True, 'checkpoint_reload_verified': True, 'parameter_update_verified': True,
                    'online_backbone_update_verified': True, 'online_changed_parameter_count': len(online_changed),
                    'frozen_parameters_verified': True, 'frozen_verification_basis': 'unused predict_head immutable; target has no gradients and only registered EMA updates',
                    'epochs': epoch, 'optimizer_updates': updates, 'stop_reason': decision['reason'], 'registry': R.registration_descriptor(),
                    **{key: S.desc(folder / name) for key, name in (('checkpoint', 'best.pt'), ('train_history', 'train_history.json'),
                                                                 ('validation_history', 'validation_history.json'), ('metadata', 'metadata.json'))}}
        S.save(folder / 'complete.json', evidence, immutable=True)
        R.record_fit_status(CONDITION, seed, fold, 'training_complete', evidence=evidence)
        return meta
    except BaseException as error:
        S.save(folder / f'failure_{time.time_ns()}.json', {'error_type': type(error).__name__, 'error': str(error),
               'last_durable_epoch': locals().get('durable_epoch', 0), 'updates_completed': locals().get('updates', 0),
               'updates_completed_this_process': locals().get('updates', 0) - locals().get('start_updates', 0), 'time': S.now()})
        R.record_fit_status(CONDITION, seed, fold, 'failed', reason=str(error))
        raise
    finally:
        del model, optimizer
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def run(*, seed=None, fold=None, resume=False):
    R.load_registration(); R._condition_lock(CONDITION)
    lock, folds, _ = C.context()
    data = RawData()
    require(data.ids == lock['subject_ids'], 'SSL data people differ')
    for value in (P.SEEDS if seed is None else (seed,)):
        for index in (P.FOLDS if fold is None else (fold,)):
            fit_one(value, index, data, folds[f'fold_{index}'], resume=resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--logical-batch', type=int, default=64)
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--seed', type=int, choices=P.SEEDS)
    parser.add_argument('--fold', type=int, choices=P.FOLDS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(logical_batch=args.logical_batch, precision=args.precision)
    else:
        run(seed=args.seed, fold=args.fold, resume=args.resume)
