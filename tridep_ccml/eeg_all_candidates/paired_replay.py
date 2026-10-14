"""New 300-epoch single-factor replay. Never scores the external test split."""
from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path

from tridep_ccml.tridep_eegpt import train_eval as T
from tridep_ccml.tridep_eegpt import common as C
from . import storage as S
from . import registry as R

np, torch = C.np, C.torch
CONDITIONS = tuple(c + '-300ep' for c in T.CONDITIONS)
SNAPSHOTS = (60, 120, 200, 300)


def require(value, message):
    if not value:
        raise ValueError(message)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def rng_equal(a, b):
    return (a['python'] == b['python'] and a['numpy'][0] == b['numpy'][0]
            and np.array_equal(a['numpy'][1], b['numpy'][1]) and a['numpy'][2:] == b['numpy'][2:]
            and torch.equal(a['torch_cpu'], b['torch_cpu'])
            and len(a['torch_cuda']) == len(b['torch_cuda'])
            and all(torch.equal(x, y) for x, y in zip(a['torch_cuda'], b['torch_cuda'])))


def diagnostic(model, features, ids, split, labels):
    """Separate eval indexing, with all RNG and module modes restored."""
    state = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    before = C.state_digest(model.state_dict())
    result = {}
    try:
        model.eval()
        with torch.inference_mode():
            for role in ('train', 'val'):
                chosen = split[role]
                ii = torch.tensor([ids.index(s) for s in chosen], device='cuda')
                yy = torch.tensor([labels[s] for s in chosen], device='cuda')
                # Do not construct another randomized DataLoader.
                logits = torch.cat([model(features[ii[start:start + 16]])
                                    for start in range(0, len(chosen), 16)])
                ce = torch.nn.functional.cross_entropy(logits, yy)
                require(torch.isfinite(ce), 'Nonfinite read-only diagnostic CE')
                result[role + '_eval_ce'] = float(ce.cpu())
        require(before == C.state_digest(model.state_dict()), 'Diagnostic changed model state')
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(state)
    require(rng_equal(state, rng_state()), 'Diagnostic RNG restore failed')
    return result


def prepare():
    R.load_registration()
    path = S.OUT / 'paired_replay/source_lock.json'
    if path.exists():
        evidence = S.read(path)
        for descriptor in evidence['sources']:
            S.checked(descriptor)
        if (S.OUT / 'paired_replay/engineering_checks.json').exists():
            for condition in CONDITIONS:
                require((S.OUT / 'engineering' / (condition + '.json')).exists(), 'Missing condition engineering lock')
            return evidence
    lock, folds, labels = C.context()
    T.validate_context(lock, folds, labels)
    sources = [S.desc(p) for p in (
        Path(__file__), Path(S.__file__), Path(T.__file__), Path(C.__file__),
        C.WORK / 'tridep_ccml/tridep_stage1/common.py',
        C.WORK / 'tridep_ccml/tridep_stage1/run.py',
        C.WORK / 'S00/upstream-dev/lib/models/models.py',
        C.OUT / 'features_manifest.json',
        C.REPORT / 'input_independent_verification.json')]
    require(C.read(C.REPORT / 'input_independent_verification.json')['passed'],
            'Historical frozen input verification missing')
    for seed in C.SEEDS:
        for fold in range(1, 6):
            for old in T.CONDITIONS:
                folder = T._folder(old, seed, fold)
                meta = C.read(folder / 'metadata.json')
                require(meta['epochs'] == 60 and meta['updates'] == 120, 'Historical model budget differs')
                sources.extend(S.desc(folder / name) for name in (
                    'initial.pt', 'checkpoint.pt', 'metadata.json', 'train_history.json', 'batch_order.npy'))
    evidence = {'registered_at': S.now(), 'sources': sources, 'conditions': list(CONDITIONS),
                'epochs': 300, 'formal_fits': 30, 'updates_per_fit': 600, 'early_stopping': False,
                'feature_cache_reused': True, 'new_encoder_forwards': 0,
                'snapshots': list(SNAPSHOTS), 'test_forward_allowed': False,
                'diagnostic_splits': ['train', 'val'], 'rng_isolated_diagnostics': True,
                'historical_first_60_required': 'exact batch order, every batch loss, initial and final model states',
                'registry': R.registration_descriptor()}
    if not path.exists():
        S.save(path, evidence, immutable=True)
    arrays, _ = T.load_features(lock)
    probes = []
    for condition in CONDITIONS:
        engineering_path = S.OUT / 'engineering' / (condition + '.json')
        if engineering_path.exists():
            probes.append(S.read(engineering_path)['contract'])
            for seed in C.SEEDS:
                for fold in range(1, 6):
                    if R.fit_status(condition, seed, fold)['status'] == 'engineering_pending':
                        R.record_fit_status(condition, seed, fold, 'ready', evidence={'source_lock': S.desc(path)})
            continue
        # Resource configuration uses training people only, with no optimizer step.
        old = condition.removesuffix('-300ep')
        C.runtime(11)
        model = T.original_model().to('cuda')
        chosen = folds['fold_1']['train'][:16]
        ii = [lock['subject_ids'].index(s) for s in chosen]
        x = torch.as_tensor(arrays[T.FEATURE_KEYS[old]][ii], device='cuda')
        y = torch.tensor([labels[s] for s in chosen], device='cuda')
        start = time.perf_counter()
        model.train()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        require(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()), 'Training-only gradient engineering check failed')
        before = C.state_digest(model.state_dict())
        state = rng_state()
        # Resume serialization must roundtrip RNG, model and Adamax state.
        optimizer = torch.optim.Adamax(model.parameters(), lr=.005, weight_decay=0.)
        saved = checkpoint_value(model, optimizer, 0, [], [], before)
        probe_path = S.OUT / 'paired_replay' / (condition + '_engineering_resume.pt')
        S.save_torch(probe_path, saved, immutable=True)
        restored = torch.load(probe_path, map_location='cpu', weights_only=False)
        model.load_state_dict(restored['state_dict'], strict=True)
        optimizer.load_state_dict(restored['optimizer_state_dict'])
        restore_rng(restored['rng'])
        require(before == C.state_digest(model.state_dict()) and rng_equal(state, rng_state()),
                'Engineering resume roundtrip failed')
        contract = {'input_contract': {'frozen_cache': S.desc(C.OUT / 'features_manifest.json'),
                    'shape_per_subject': [60, 200], 'physical_seconds_per_window': 4,
                    'normalization': 'identical historical cache; no newly fitted transform',
                    'head': 'original EEGConvPoolClassifier', 'parameters': 498434,
                    'dtype': 'float32', 'batch_size': 16, 'optimizer': 'Adamax',
                    'learning_rate': .005, 'weight_decay': 0., 'scheduler': None},
                    'trainable_parameter_names': [n for n, p in model.named_parameters() if p.requires_grad],
                    'frozen_parameter_names': [], 'implementation_sources': sources[:7],
                    'weight_sources': [S.desc(C.OUT / 'features_manifest.json')],
                    'forward_checked': True, 'gradient_checked': True, 'resume_checked': True,
                    'training_only_engineering_checked': True,
                    'probe_subject_ids': chosen, 'engineering_optimizer_updates': 0,
                    'engineering_gradient_passes': 1, 'forward_backward_seconds': time.perf_counter()-start}
        R.lock_condition(condition, contract)
        probes.append(contract)
        for seed in C.SEEDS:
            for fold in range(1, 6):
                R.record_fit_status(condition, seed, fold, 'ready', evidence={'source_lock': S.desc(path)})
        del model, optimizer, x, y
        gc.collect()
        torch.cuda.empty_cache()
    S.save(S.OUT / 'paired_replay/engineering_checks.json', {'passed': True, 'checks': probes,
           'optimizer_updates': 0, 'gradient_passes': 2, 'test_forward_count': 0}, immutable=True)
    return evidence


def checkpoint_value(model, optimizer, epoch, history, order, initial):
    return {'state_dict': T._cpu(model.state_dict()),
            'optimizer_state_dict': T._cpu(optimizer.state_dict()),
            'scheduler_state_dict': None, 'rng': rng_state(), 'epoch': epoch,
            'history': history, 'order': order, 'initial_state_sha256': initial,
            'sampler_position': {'epoch_completed': epoch, 'next_epoch': epoch + 1, 'next_batch': 0},
            'checkpoint_selection': 'fixed_epoch_300', 'registry': R.registration_descriptor()}


def fit_one(condition, seed, fold, tensors, ids, split, labels, *, resume=False):
    old = condition.removesuffix('-300ep')
    folder = R.fit_directory(condition, seed, fold)
    oldfolder = T._folder(old, seed, fold)
    if (folder / 'complete.json').exists():
        completion = S.read(folder / 'complete.json')
        for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'):
            S.checked(completion[key])
        if R.fit_status(condition, seed, fold)['status'] != 'training_complete':
            R.record_fit_status(condition, seed, fold, 'training_complete', evidence=completion)
        return S.read(folder / 'metadata.json')
    partial = (folder / 'started.json').exists()
    require(not partial or resume, 'Partial fit preserved; use --resume with full saved RNG state')
    oldmeta = C.read(oldfolder / 'metadata.json')
    oldhistory = C.read(oldfolder / 'train_history.json')
    oldorder = np.load(oldfolder / 'batch_order.npy', allow_pickle=False)
    indices, dataset = T.training_dataset(ids, split['train'], labels)
    R.record_fit_status(condition, seed, fold, 'training')
    if not partial:
        S.save(folder / 'started.json', {'condition': condition, 'seed': seed, 'fold': fold,
               'started_at': S.now(), 'planned_updates': 600, 'train_ids': split['train'],
               'source_lock': S.desc(S.OUT / 'paired_replay/source_lock.json')}, immutable=True)
    C.runtime(seed)
    model = T.original_model().to('cuda')
    require(sum(p.numel() for p in model.parameters()) == 498434, 'Original head parameters differ')
    initial = C.state_digest(model.state_dict())
    require(initial == oldmeta['initial_state_sha256'], 'Historical initialization does not replay')
    # Construction must consume the identical random draws; do not merely load initial.pt.
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adamax(model.parameters(), lr=.005, weight_decay=0.)
    history, order, start_epoch = [], [], 0
    if partial:
        resume_path = folder / 'latest.pt' if (folder / 'latest.pt').exists() else folder / 'initial.pt'
        if not resume_path.exists():
            # The durable start exists, but no optimizer step could have completed
            # without a preceding initial checkpoint; restart the deterministic setup.
            require(not (folder / 'progress.json').exists(), 'Missing state after an optimizer step was started')
            S.save_torch(folder / 'initial.pt', checkpoint_value(model, optimizer, 0, [], [], initial), immutable=True)
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        require(saved['registry'] == R.registration_descriptor(), 'Resume registry differs')
        require(isinstance(saved['epoch'], int) and 0 <= saved['epoch'] <= 300
                and len(saved['history']) == len(saved['order']) == saved['epoch']
                and saved['initial_state_sha256'] == initial, 'Resume epoch/history/initial identity differs')
        require(saved['sampler_position'] == {'epoch_completed': saved['epoch'],
                'next_epoch': saved['epoch'] + 1, 'next_batch': 0}, 'Resume sampler state differs')
        require(all(torch.isfinite(v).all() for v in saved['state_dict'].values()), 'Nonfinite resume model')
        for index, row in enumerate(saved['history']):
            require(row['epoch'] == index + 1 and row['updates_completed'] == row['gradient_passes_completed'] == (index + 1)*2,
                    'Resume update/epoch history differs')
            require(sorted(saved['order'][index]) == sorted(indices), 'Resume sample population differs')
            if index < 60:
                require({k: row[k] for k in oldhistory[index]} == oldhistory[index]
                        and saved['order'][index] == oldorder[index].tolist(), 'Resume historical first60 evidence differs')
        require((not saved['optimizer_state_dict']['state'] and saved['epoch'] == 0) or
                (len(saved['optimizer_state_dict']['state']) == len(list(model.parameters()))
                 and all(float(value['step']) == saved['epoch']*2 for value in saved['optimizer_state_dict']['state'].values())),
                'Resume optimizer step count differs')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history, order, start_epoch = saved['history'], saved['order'], saved['epoch']
        restore_rng(saved['rng'])
        if start_epoch:
            S.save(folder / 'train_history.json', history)
            S.save(folder / 'validation_history.json', [
                {'epoch': r['epoch'], 'val_ce': r['val_eval_ce']} for r in history])
            if start_epoch in SNAPSHOTS:
                S.save_torch(folder / f'epoch_{start_epoch}.pt', saved, immutable=True)
    else:
        S.save_torch(folder / 'initial.pt', checkpoint_value(model, optimizer, 0, [], [], initial), immutable=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    updates = start_epoch * 2
    durable_epoch = start_epoch
    features = tensors[T.FEATURE_KEYS[old]]
    try:
        for epoch in range(start_epoch, 300):
            model.train()
            epoch_order, losses = [], []
            for batch, (ii, yy) in enumerate(loader):
                require(all(int(i) in indices for i in ii), 'Nontrain subject entered optimizer step')
                require(yy.tolist() == [labels[ids[int(i)]] for i in ii], 'Training label mismatch')
                S.save(folder / 'progress.json', {'epoch': epoch + 1, 'updates_completed': updates,
                       'updates_started': updates + 1, 'batch_subject_ids': [ids[int(i)] for i in ii]})
                optimizer.zero_grad(set_to_none=True)
                logits = model(features[ii.to('cuda')])
                loss = torch.nn.functional.cross_entropy(logits, yy.to('cuda'))
                require(torch.isfinite(loss), 'Nonfinite train CE')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                        'Nonfinite or disconnected head gradient')
                optimizer.step()
                require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite model parameter')
                updates += 1
                epoch_order.extend(ii.tolist())
                losses.append({'batch_size': len(ii), 'train_ce': float(loss.detach().cpu())})
                if epoch < 60:
                    require(losses[-1] == oldhistory[epoch]['batches'][batch],
                            f'Historical batch CE differs at epoch {epoch+1}, batch {batch+1}')
            require(len(epoch_order) == len(indices), 'Epoch subject count differs')
            if epoch < 60:
                require(epoch_order == oldorder[epoch].tolist(), 'Historical sample order differs')
            row = {'epoch': epoch + 1, 'updates_completed': updates,
                   'gradient_passes_completed': updates, 'batches': losses,
                   'train_ce': sum(v['train_ce'] * v['batch_size'] for v in losses) / len(indices)}
            if epoch < 60:
                require(row == oldhistory[epoch], 'Historical full training curve differs')
            row.update(diagnostic(model, features, ids, split, labels))
            history.append(row)
            order.append(epoch_order)
            if epoch == 59:
                require(C.state_digest(model.state_dict()) == oldmeta['final_state_sha256'],
                        'Historical epoch60 final state differs')
                saved60 = torch.load(oldfolder / 'checkpoint.pt', map_location='cpu', weights_only=False)
                from tridep_ccml.tridep_eegpt.recovery_io import _same
                require(_same(T._cpu(optimizer.state_dict()), saved60['optimizer_state_dict']),
                        'Historical Adamax state differs at epoch60')
                S.save(folder / 'first60_replay_verification.json', {
                    'passed': True, 'exact_initial_state': True, 'exact_all_120_batch_losses': True,
                    'exact_batch_order': True, 'exact_epoch60_state': True, 'exact_optimizer_state': True,
                    'historical_checkpoint': S.desc(oldfolder / 'checkpoint.pt')}, immutable=True)
            payload = checkpoint_value(model, optimizer, epoch + 1, history, order, initial)
            S.save_torch(folder / 'latest.pt', payload)
            durable_epoch = epoch + 1
            if epoch + 1 in SNAPSHOTS:
                S.save_torch(folder / f'epoch_{epoch+1}.pt', payload, immutable=True)
            S.save(folder / 'train_history.json', history)
            S.save(folder / 'validation_history.json', [
                {'epoch': r['epoch'], 'val_ce': r['val_eval_ce']} for r in history])
            S.save(folder / 'progress.json', {'epoch': epoch + 1, 'updates_completed': updates,
                   'gradient_passes_completed': updates, 'epoch_checkpoint': 'latest.pt'})
            if (epoch + 1) % 60 == 0:
                print(f'TRAIN {condition} seed={seed} fold={fold} epoch={epoch+1}/300 exact60={epoch>=59}', flush=True)
        final = C.state_digest(model.state_dict())
        require(final != initial and updates == 600, 'Incomplete or unchanged trained head')
        model.eval()
        with torch.inference_mode():
            ii = torch.tensor(indices, device='cuda')
            reference = model(features[ii]).cpu()
            saved = torch.load(folder / 'epoch_300.pt', map_location='cpu', weights_only=False)
            # deepcopy avoids drawing random numbers for a new initialization.
            import copy
            reloaded = copy.deepcopy(model)
            reloaded.load_state_dict(saved['state_dict'], strict=True)
            require(torch.equal(reference, reloaded(features[ii]).cpu()), 'Full checkpoint reload differs')
        del reloaded
        meta = {'condition': condition, 'seed': seed, 'fold': fold, 'epochs': 300,
                'updates': updates, 'gradient_passes': updates, 'parameters': 498434,
                'initial_state_sha256': initial, 'final_state_sha256': final,
                'batch_order_sha256': T.batch_order_digest(np.asarray(order, dtype=np.int64)),
                'train_ids': split['train'], 'val_ids': split['val'], 'test_ids': split['test'],
                'training': {**T.TRAINING, 'epochs': 300, 'updates_per_model': 600},
                'stopping_reason': 'fixed_300_epoch_diagnostic', 'checkpoint_selection': 'fixed_epoch_300',
                'early_stopping': False, 'test_forward_count': 0, 'optimizer_rng_isolated_validation': True,
                'encoder_forwards': 0, 'frozen_features': S.desc(C.OUT / 'features_manifest.json'),
                'reload_verification_subject_ids': split['train'], 'seconds_this_attempt': time.perf_counter()-started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'first60_replay_exact': True,
                'snapshots': {str(e): S.desc(folder / f'epoch_{e}.pt') for e in SNAPSHOTS}}
        if (folder / 'metadata.json').exists():
            previous = S.read(folder / 'metadata.json')
            require(all(previous[k] == meta[k] for k in ('condition', 'seed', 'fold', 'epochs',
                    'updates', 'initial_state_sha256', 'final_state_sha256', 'batch_order_sha256')),
                    'Existing final metadata differs')
            meta = previous
        else:
            S.save(folder / 'metadata.json', meta, immutable=True)
        completion = {'passed': True, 'checkpoint_reload_verified': True,
                      'parameter_update_verified': True, 'frozen_parameters_verified': True,
                      'epochs': 300, 'optimizer_updates': updates, 'stop_reason': 'fixed_300_epoch_diagnostic',
                      'frozen_verification_basis': 'hash-bound immutable frozen features; no encoder executed',
                      'registry': R.registration_descriptor(),
                      **{key: S.desc(folder / name) for key, name in (
                          ('checkpoint', 'epoch_300.pt'), ('train_history', 'train_history.json'),
                          ('validation_history', 'validation_history.json'), ('metadata', 'metadata.json'))}}
        S.save(folder / 'complete.json', completion, immutable=True)
        R.record_fit_status(condition, seed, fold, 'training_complete', evidence=completion)
        return meta
    except BaseException as error:
        S.save(folder / ('failure_' + str(time.time_ns()) + '.json'), {
            'time': S.now(), 'updates_completed_this_process': updates - start_epoch * 2,
            'updates_total_process': updates, 'last_durable_epoch': durable_epoch,
            'error_type': type(error).__name__, 'error': str(error)})
        R.record_fit_status(condition, seed, fold, 'failed', reason=str(error))
        raise
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def run(*, limit=None, resume=False):
    prepare()
    lock, folds, labels = C.context()
    arrays, manifest = T.load_features(lock)
    tensors = {key: torch.as_tensor(x, device='cuda') for key, x in arrays.items()}
    completed = []
    for seed in C.SEEDS:
        for fold in range(1, 6):
            paired = []
            for condition in CONDITIONS:
                meta = fit_one(condition, seed, fold, tensors, lock['subject_ids'], folds[f'fold_{fold}'], labels, resume=resume)
                completed.append(meta)
                paired.append((meta['initial_state_sha256'], meta['batch_order_sha256']))
                S.save(S.OUT / 'paired_replay/training_ledger.json', {
                    'planned_fits': 30, 'fits_completed': len(completed),
                    'planned_updates': 18000, 'updates_completed': sum(m['updates'] for m in completed),
                    'fits': completed, 'metrics_released': False, 'test_forward_count': 0})
                print(f'COMPLETE {condition} seed={seed} fold={fold} fits={len(completed)}/30', flush=True)
                if limit is not None and len(completed) >= limit:
                    return
            require(paired[0] == paired[1], 'Paired conditions differ in initialization or full300 order')
    print('All30 paired300 fits complete; external tests remain locked until all360 registered fits finish.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare_only else run(limit=args.limit, resume=args.resume)
