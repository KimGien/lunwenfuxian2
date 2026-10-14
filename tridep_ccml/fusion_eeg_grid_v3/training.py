"""Durable, epoch-boundary fusion training for the registered v3 experiments.

No test forward is performed here. The 120-epoch prefix and the conditional
240-epoch continuation share one absolute-epoch schedule and selection policy.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import common as C
from .models import build_model
from . import preprocessing

SCHEMA = 'fusion_eeg_grid_v3_training_1'
FIXED = {'batch_size': 8, 'weight_decay': 1e-3, 'gradient_clip': 1.,
         'warmup_epochs': 10, 'prefix_epochs': 120, 'min_epochs': 60,
         'patience': 60, 'min_delta': 1e-4, 'precision': 'FP32',
         'selection': 'strict_minimum_validation_NLL_earlier_on_exact_tie'}


def configuration(identity):
    value = dict(FIXED, base_lr=float(identity.get('base_lr', 1e-3)),
                 normalization=identity.get('normalization', 'N1'),
                 qmf_variant=identity.get('qmf_variant', 'Q1'),
                 budget=int(identity.get('budget', 120)))
    C.require(value['base_lr'] in (3e-4, 1e-3, 3e-3), 'Unregistered learning rate')
    C.require(value['normalization'] in ('N0', 'N1'), 'Unknown normalization')
    C.require(value['qmf_variant'] in ('Q0', 'Q1'), 'Unknown QMF variant')
    C.require(value['budget'] in (120, 240), 'Unregistered training budget')
    C.require(value['budget'] != 240 or identity['method'] in ('CCML', 'ECML'),
              'Only CCML/ECML are registered for the extension')
    return value


def lr_for_epoch(epoch, base_lr=1e-3):
    C.require(1 <= epoch <= 240, 'Learning-rate epoch outside registered schedule')
    if epoch <= 10:
        return base_lr * epoch / 10
    if epoch > 120:
        return base_lr * .1
    return base_lr * (.1 + .9 * (1 + math.cos(math.pi * (epoch - 10) / 110)) / 2)


def scheduler_state(epoch, config):
    return {'kind': 'absolute_epoch_warmup10_cosine120_then_constant',
            'base_lr': config['base_lr'], 'last_completed_epoch': epoch,
            'next_epoch': epoch + 1,
            'next_lr': lr_for_epoch(epoch + 1, config['base_lr']) if epoch < config['budget'] else None,
            'budget': config['budget'], 'extension_lr': .1 * config['base_lr']}


def nll(prob, y):
    return -prob[torch.arange(len(y), device=y.device), y].clamp_min(1e-15).double().log().mean()


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def setup(method, views, ntrain, seed, device, base_lr=1e-3, qmf_variant='Q1'):
    C.runtime(seed, device)
    modalities = ('S', 'T') if views == 2 else ('E', 'S', 'T')
    C.require(views in (2, 3), 'Only ST or EST modalities are registered')
    model = build_model(method, modalities, qmf_variant=qmf_variant, n_train=ntrain).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-3, foreach=False)
    return model, optimizer


def train_epoch(model, optimizer, x, y, epoch, base_lr=1e-3):
    """One indivisible resume boundary; the final incomplete batch is retained."""
    start = time.perf_counter()
    model.train()
    rate = lr_for_epoch(epoch, base_lr)
    for group in optimizer.param_groups:
        group['lr'] = rate
    order = torch.randperm(len(y), device=x.device)
    initial = [p.detach().clone() for p in model.parameters()]
    losses, grads, terms = [], [], {}
    for offset in range(0, len(order), 8):
        indices = order[offset:offset + 8]
        optimizer.zero_grad(set_to_none=True)
        loss, components = model.objective(x[indices], y[indices], epoch, indices)
        C.require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        C.require(all(bool(torch.isfinite(v).all()) for v in components.values()),
                  'Nonfinite core objective component')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.detach() * len(indices))
        grads.append(grad.detach())
        for key, value in components.items():
            terms.setdefault(key, []).append(value.detach() * len(indices))
    C.require(all(bool(p.detach().isfinite().all()) for p in model.parameters()),
              'Nonfinite parameter update')
    update_sq = sum((p.detach() - before).double().square().sum()
                    for p, before in zip(model.parameters(), initial))
    diagnostic = torch.stack([sum(losses) / len(y), torch.stack(grads).max(),
                              *[sum(terms[k]) / len(y) for k in sorted(terms)]]).cpu().tolist()
    return {'epoch': epoch, 'lr': rate, 'loss': diagnostic[0], 'max_gradient_norm': diagnostic[1],
            'components': dict(zip(sorted(terms), diagnostic[2:])), 'updates': math.ceil(len(y) / 8),
            'train_seconds': time.perf_counter() - start,
            'parameter_update_l2': float(update_sq.sqrt()),
            'train_order': order.cpu().tolist(),
            'batch_sizes': [min(8, len(y) - i) for i in range(0, len(y), 8)]}


def history_state(model):
    if getattr(model, 'history', None) is None:
        return None
    return {'cumulative_ce': model.history.copy(), 'confidence': model.confidence_history.copy()}


def restore_history(model, state):
    if getattr(model, 'history', None) is None:
        C.require(state is None, 'Unexpected quality history')
        return
    C.require(state is not None and set(state) == {'cumulative_ce', 'confidence'},
              'QMF history is incomplete')
    for key in state:
        C.require(state[key].shape == model.history.shape and np.isfinite(state[key]).all(),
                  'QMF train-history shape/finiteness differs')
    model.history = state['cumulative_ce'].copy()
    model.confidence_history = state['confidence'].copy()


def load_inputs(identity, device, feature_root=None):
    C.require(feature_root is None, 'Feature source override is not registered')
    return preprocessing.load_inputs(identity, device, variant=configuration(identity)['normalization'])


def selection_update(best_nll, best_epoch, patience_ref, stale, vnll, epoch):
    """The absolute minimum selects best; min_delta only controls patience."""
    C.require(math.isfinite(vnll), 'Nonfinite validation NLL')
    selected = best_nll is None or vnll < best_nll
    if selected:
        best_nll, best_epoch = vnll, epoch
    if patience_ref is None or vnll <= patience_ref - 1e-4:
        patience_ref, stale = vnll, 0
    else:
        stale += 1
    return best_nll, best_epoch, patience_ref, stale, selected


def should_stop(epoch, stale, budget):
    return epoch >= budget or (epoch >= 60 and stale >= 60)


def _path(directory, identity):
    return Path(directory).resolve() if directory is not None else C.fitdir(identity).resolve()


def complete_verified(identity, directory=None):
    directory = _path(directory, identity)
    path = directory / 'complete.json'
    if not path.exists():
        return None
    value = C.read(path)
    budget = configuration(identity)['budget']
    C.require(value['identity'] == identity and value['passed'] and 60 <= value['epochs'] <= budget,
              'Completed fit identity/budget invalid')
    for key in ('best', 'latest', 'metadata', 'trajectory'):
        C.checked(value[key])
    C.require(value['updates'] > 0 and value['parameter_change_verified'] and value['test_forwards'] == 0,
              'Completed fit lacks actual training/isolation evidence')
    return value


def _core_identity(identity):
    return {key: value for key, value in identity.items() if key not in ('stage', 'budget')}


def _parent(identity, directory, device, binding, config):
    directory = Path(directory).resolve()
    complete = C.read(directory / 'complete.json')
    C.require(complete['passed'] and complete['test_forwards'] == 0,
              'Extension parent is not accepted/isolation-preserving')
    parent_identity = complete['identity']
    C.require(_core_identity(parent_identity) == _core_identity(identity), 'Extension parent fit differs')
    C.require(configuration(parent_identity)['budget'] == 120, 'Extension parent must be the 120 policy')
    descriptors = {'complete': C.desc(directory / 'complete.json')}
    for key in ('latest', 'best', 'trajectory', 'metadata'):
        expected = directory / {'latest': 'latest.pt', 'best': 'best.pt',
                                'trajectory': 'trajectory.json', 'metadata': 'metadata.json'}[key]
        actual = C.checked(complete[key])
        C.require(actual.resolve() == expected.resolve(), 'Extension parent artifact belongs to another directory')
        descriptors[key] = complete[key]
    saved = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
    C.require(saved['schema'] == SCHEMA and saved['epoch'] == 120 and saved['stale'] < 60,
              'Extension requires an actual epoch-120 latest whose patience is not exhausted')
    C.require(saved['identity'] == parent_identity and saved['input_binding'] == binding and saved['device'] == device,
              'Extension parent input/backend differs')
    C.require({k: v for k, v in saved['configuration'].items() if k != 'budget'} ==
              {k: v for k, v in config.items() if k != 'budget'}, 'Extension configuration differs')
    parent_metadata = C.read(C.checked(descriptors['metadata']))
    C.require(parent_metadata['identity'] == parent_identity and
              parent_metadata['configuration'] == configuration(parent_identity),
              'Extension parent metadata identity/configuration differs')
    _validate_snapshot(saved, parent_identity, parent_metadata['registration'], binding,
                       device, configuration(parent_identity))
    C.require(saved['ledger'] == C.read(directory / 'trajectory.json'), 'Parent trajectory/latest differ')
    C.require(saved['epoch'] == complete['epochs'] and saved['updates'] == complete['updates'],
              'Parent completed epoch/update counts differ')
    best = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    C.require(equal_state(best, saved['best_checkpoint']), 'Parent best differs from durable selection')
    return saved, descriptors


def _validate_snapshot(saved, identity, registration, binding, device, config):
    required = {'schema', 'identity', 'registration', 'configuration', 'input_binding', 'device',
                'precision', 'epoch', 'updates', 'state_dict', 'optimizer', 'scheduler', 'rng',
                'quality_history', 'best_nll', 'best_epoch', 'patience_ref', 'stale', 'initial_sha',
                'ledger', 'best_prob', 'best_checkpoint', 'sampling', 'parent_binding',
                'prefix_epochs', 'prefix_updates'}
    C.require(required <= set(saved), 'Resume checkpoint fields are incomplete: ' +
              ', '.join(sorted(required - set(saved))))
    C.require({'python', 'numpy', 'torch', 'cuda'} <= set(saved['rng']) and
              {'state', 'param_groups'} <= set(saved['optimizer']),
              'Resume RNG/optimizer state is incomplete')
    C.require(saved['schema'] == SCHEMA and saved['identity'] == identity and
              saved['registration'] == registration and saved['input_binding'] == binding and
              saved['device'] == device and saved['configuration'] == config,
              'Resume identity/source/backend/configuration changed')
    C.require(saved['scheduler'] == scheduler_state(saved['epoch'], config), 'Resume scheduler is incomplete')
    C.require(saved['sampling'] == {'boundary': 'completed_epoch', 'completed_epoch': saved['epoch'],
                                    'next_epoch': saved['epoch'] + 1, 'next_batch_offset': 0,
                                    'generator': 'restored_torch_device_rng'},
              'Resume sampler is not at the recorded epoch boundary')
    C.require(len(saved['ledger']) == saved['epoch'], 'Checkpoint trajectory prefix is incomplete')
    C.require(saved['updates'] == sum(row['updates'] for row in saved['ledger']),
              'Checkpoint update count differs from trajectory')


def _process_evidence(engineering_only=False):
    try:
        return {'native_tree': C.own_tree(), 'unavailable': None}
    except Exception as error:
        if not engineering_only:
            raise
        # A synthetic/restricted-shell recovery may lack CIM. Never invent a PID birth.
        return {'native_tree': None, 'unavailable': repr(error)}


def _recover_milestone(directory, saved, identity, registration, binding, device, config):
    """Close the crash window after latest commits but before snapshot writes."""
    epoch = saved['epoch']
    if epoch not in (60, 120, 180, 240) or epoch <= saved['prefix_epochs']:
        return
    target = directory / f'epoch_{epoch}.pt'
    if target.exists():
        snapshot = torch.load(target, map_location='cpu', weights_only=False)
        _validate_snapshot(snapshot, identity, registration, binding, device, config)
        C.require(equal_state({k: v for k, v in snapshot.items() if k != 'ledger'},
                              {k: v for k, v in saved.items() if k != 'ledger'}),
                  'Existing milestone differs from its durable state')
        # A later latest commit may know the already measured write duration.
        def rows_without_write(rows):
            return [{k: v for k, v in row.items() if k != 'write_seconds'} for row in rows]
        C.require(equal_state(rows_without_write(snapshot['ledger']), rows_without_write(saved['ledger'])),
                  'Existing milestone trajectory differs')
        return
    C.save_torch(target, saved)
    C.save(directory / f'milestone_recovery_epoch_{epoch}_{time.time_ns()}.json',
           {'at': C.now(), 'epoch': epoch, 'source': C.desc(directory / 'latest.pt'),
            'snapshot': C.desc(target), 'reason': 'latest committed before missing milestone snapshot',
            'additional_optimizer_updates': 0})


def fit(identity, device, registration, *, directory=None, feature_root=None,
        parent_directory=None, engineering_stop_after=None):
    """Fit a registered identity or continue its exact durable epoch boundary.

    engineering_stop_after is restricted to synthetic gates in the report folder;
    it never creates a completion certificate before the registered endpoint.
    """
    config = configuration(identity)
    directory = _path(directory, identity)
    if engineering_stop_after is not None:
        C.require(directory.is_relative_to((C.REPORT / 'training_gates').resolve()) and
                  registration.get('engineering_only') is True,
                  'Partial-run control is reserved for synthetic engineering gates')
    done = complete_verified(identity, directory)
    if done:
        accepted_metadata = C.read(C.checked(done['metadata']))
        C.require(accepted_metadata['registration'] == registration and
                  accepted_metadata['configuration'] == config and accepted_metadata['device'] == device,
                  'Completed fit registration/configuration/backend changed')
        return done
    C.check_pause()
    directory.mkdir(parents=True, exist_ok=True)
    x, y, ix, binding = load_inputs(identity, device, feature_root)
    train = torch.as_tensor(ix['train'], device=device)
    val = torch.as_tensor(ix['val'], device=device)
    train_x, train_y = x[train], y[train]
    val_x, val_y = x[val], y[val]
    C.require(bool(torch.isfinite(x).all()), 'Input features are not finite')
    model, optimizer = setup(identity['method'], x.shape[1], len(train), identity['seed'], device,
                             config['base_lr'], config['qmf_variant'])
    initial_sha = C.state_sha(model.state_dict())
    epoch = updates = best_epoch = stale = prefix_epochs = prefix_updates = 0
    best_nll = patience_ref = best_checkpoint = best_prob = None
    ledger = []
    parent_binding = None
    resume = directory / 'latest.pt'
    saved = None
    restoring = resume.exists()
    if restoring:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        _validate_snapshot(saved, identity, registration, binding, device, config)
        _recover_milestone(directory, saved, identity, registration, binding, device, config)
        parent_binding = saved['parent_binding']
        if parent_binding:
            for descriptor in parent_binding.values():
                C.checked(descriptor)
    elif config['budget'] == 240:
        C.require(parent_directory is not None, 'Extension requires its explicit read-only 120-epoch parent')
        parent_path = Path(parent_directory).resolve()
        C.require(directory != parent_path and not directory.is_relative_to(parent_path),
                  'Extension must use an independent output directory')
        saved, parent_binding = _parent(identity, parent_path, device, binding, config)
    else:
        C.require(parent_directory is None, 'A 120-epoch fit cannot import a parent state')
    if saved is not None:
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        restore_history(model, saved['quality_history'])
        C.restore_rng(saved['rng'])
        epoch, updates, best_nll, best_epoch, patience_ref, stale = (saved[k] for k in
            ('epoch', 'updates', 'best_nll', 'best_epoch', 'patience_ref', 'stale'))
        initial_sha, ledger = saved['initial_sha'], copy.deepcopy(saved['ledger'])
        best_checkpoint, best_prob = saved['best_checkpoint'], saved['best_prob']
        if restoring:
            prefix_epochs, prefix_updates = saved['prefix_epochs'], saved['prefix_updates']
            C.save(directory / f'recovery_epoch_{epoch}_{time.time_ns()}.json',
                   {'at': C.now(), 'latest': C.desc(resume), 'restored_epoch': epoch,
                    'restored_updates': updates, 'unknown_uncommitted_tail_updates': None,
                    **_process_evidence(registration.get('engineering_only') is True)})
        elif parent_binding:
            prefix_epochs, prefix_updates = epoch, updates
            C.save(directory / 'parent_binding.json', parent_binding, immutable=True)
    committed_epoch = epoch

    def payload(include_best=True):
        value = {'schema': SCHEMA, 'identity': identity, 'registration': registration,
                 'configuration': config, 'input_binding': binding, 'device': device,
                 'precision': 'FP32', 'epoch': epoch, 'updates': updates,
                 'state_dict': C.cpu_state(model.state_dict()),
                 'optimizer': C.cpu_state(optimizer.state_dict()),
                 'scheduler': scheduler_state(epoch, config), 'rng': C.rng_state(),
                 'quality_history': history_state(model), 'best_nll': best_nll,
                 'best_epoch': best_epoch, 'patience_ref': patience_ref, 'stale': stale,
                 'initial_sha': initial_sha, 'ledger': copy.deepcopy(ledger), 'best_prob': best_prob,
                 'sampling': {'boundary': 'completed_epoch', 'completed_epoch': epoch,
                              'next_epoch': epoch + 1, 'next_batch_offset': 0,
                              'generator': 'restored_torch_device_rng'},
                 'parent_binding': parent_binding, 'prefix_epochs': prefix_epochs,
                 'prefix_updates': prefix_updates}
        if include_best:
            value['best_checkpoint'] = best_checkpoint
        return value

    if not restoring:
        C.save_torch(resume, payload())
    try:
        while not should_stop(epoch, stale, config['budget']):
            C.check_pause()
            if engineering_stop_after is not None and epoch >= engineering_stop_after:
                C.save_torch(resume, payload())
                return {'identity': identity, 'passed': False, 'engineering_boundary': True,
                        'epochs': epoch, 'updates': updates, 'latest': C.desc(resume), 'test_forwards': 0}
            epoch += 1
            record = train_epoch(model, optimizer, train_x, train_y, epoch, config['base_lr'])
            updates += record['updates']
            started = time.perf_counter()
            model.eval()
            with torch.inference_mode():
                probability = model(val_x)['p']
                C.require(bool(torch.isfinite(probability).all()) and bool((probability >= 0).all()) and
                          bool((probability <= 1).all()) and
                          torch.allclose(probability.sum(-1), torch.ones(len(val), device=probability.device),
                                         atol=2e-6, rtol=2e-6), 'Invalid validation probabilities')
                vnll = float(nll(probability, val_y))
                vp = probability.cpu().clone()
            record.update(val_nll=vnll, validation_seconds=time.perf_counter() - started,
                          cumulative_updates=updates, write_seconds=None)
            best_nll, best_epoch, patience_ref, stale, selected = selection_update(
                best_nll, best_epoch, patience_ref, stale, vnll, epoch)
            record.update(best_epoch=best_epoch, best_val_nll=best_nll,
                          patience_reference_nll=patience_ref, stale=stale)
            ledger.append(record)
            if selected:
                best_prob = vp
                best_checkpoint = payload(include_best=False)
            started = time.perf_counter()
            current = payload()
            C.save_torch(resume, current)
            committed_epoch = epoch
            if selected:
                C.save_torch(directory / 'best.pt', best_checkpoint)
            if epoch in (60, 120, 180, 240):
                C.save_torch(directory / f'epoch_{epoch}.pt', current)
            # Each committed epoch has an atomically flushed readable trajectory.
            C.save(directory / 'trajectory.json', ledger)
            C.save(directory / 'progress.json', {'identity': identity, 'epoch': epoch, 'updates': updates,
                    'best_epoch': best_epoch, 'best_val_nll': best_nll, 'val_nll': vnll, 'stale': stale,
                    'at': C.now(), 'status': 'training', 'test_forwards': 0})
            record['write_seconds'] = time.perf_counter() - started
        C.check_pause()
        # Latest contains the entire selected epoch checkpoint, including its
        # optimizer/RNG. It repairs a crash between latest and best transactions.
        C.save_torch(resume, payload())
        bestpath = directory / 'best.pt'
        best = torch.load(bestpath, map_location='cpu', weights_only=False) if bestpath.exists() else None
        if not equal_state(best, best_checkpoint):
            C.save_torch(bestpath, best_checkpoint)
        selected, _ = setup(identity['method'], x.shape[1], len(train), identity['seed'], device,
                            config['base_lr'], config['qmf_variant'])
        selected.load_state_dict(best_checkpoint['state_dict'], strict=True)
        restore_history(selected, best_checkpoint['quality_history'])
        selected.eval()
        with torch.inference_mode():
            restored_probability = selected(val_x)['p'].cpu()
        C.require(torch.equal(restored_probability, best_prob), 'Best checkpoint validation replay differs')
        terminal_sha = C.state_sha(model.state_dict())
        C.require(terminal_sha != initial_sha and updates == len(ledger) * math.ceil(len(train) / 8),
                  'No parameter change or update count mismatch')
        C.require(all(row['parameter_update_l2'] > 0 for row in ledger), 'An epoch did not update parameters')
        C.save(directory / 'trajectory.json', ledger)

        def cost(rows):
            return {'training_seconds': sum(row['train_seconds'] for row in rows),
                    'validation_seconds': sum(row['validation_seconds'] for row in rows),
                    'durable_write_seconds': sum(row['write_seconds'] for row in rows)
                    if all(row['write_seconds'] is not None for row in rows) else None}

        metadata = {'identity': identity, 'configuration': config, 'input_binding': binding,
                    'registration': registration, 'device': device,
                    'parameters': sum(p.numel() for p in model.parameters()), 'epochs': epoch,
                    'updates': updates, 'additional_epochs': epoch - prefix_epochs,
                    'additional_updates': updates - prefix_updates, 'prefix_epochs': prefix_epochs,
                    'prefix_updates': prefix_updates, 'new_initializations': int(parent_binding is None),
                    'best_epoch': best_epoch, 'best_val_nll': best_nll, 'patience_ref': patience_ref,
                    'stale': stale, 'initial_state_sha256': initial_sha,
                    'terminal_state_sha256': terminal_sha,
                    'best_state_sha256': C.state_sha(best_checkpoint['state_dict']),
                    'local_costs': cost(ledger[prefix_epochs:]), 'readonly_prefix_costs': cost(ledger[:prefix_epochs]),
                    'final_acceptance_write_and_reload_seconds': None,
                    'stopping_reason': 'validation_patience' if stale >= 60 else f'max_{config["budget"]}_epochs',
                    'parent_binding': parent_binding, 'validation_replay_bitwise': True,
                    'test_forwards': 0, 'batch8_last_kept': True, 'precision': 'FP32',
                    'resume_uncommitted_tail_is_unknown': True}
        C.save(directory / 'metadata.json', metadata)
        value = {'identity': identity, 'passed': True, 'epochs': epoch, 'updates': updates,
                 'additional_epochs': epoch - prefix_epochs, 'additional_updates': updates - prefix_updates,
                 'new_initializations': int(parent_binding is None), 'best_epoch': best_epoch,
                 'parameter_change_verified': True, 'test_forwards': 0, 'validation_replay_bitwise': True,
                 'metadata': C.desc(directory / 'metadata.json'), 'trajectory': C.desc(directory / 'trajectory.json'),
                 'best': C.desc(bestpath), 'latest': C.desc(resume), 'finished_at': C.now()}
        C.save(directory / 'complete.json', value, immutable=True)
        C.save(directory / 'progress.json', dict(identity=identity, status='complete', epoch=epoch,
                                                updates=updates, at=C.now(), test_forwards=0))
        return value
    except BaseException as error:
        C.save(directory / f'event_{time.time_ns()}.json',
               {'at': C.now(), 'error': repr(error), 'epoch_attempt': epoch,
                'latest_durable': C.desc(resume), 'latest_committed_epoch': committed_epoch,
                'unknown_uncommitted_tail_updates': None if epoch > committed_epoch else 0,
                'unknown_tail_seconds': None if epoch > committed_epoch else 0,
                'paused_at_boundary': isinstance(error, InterruptedError) and epoch == committed_epoch})
        raise
