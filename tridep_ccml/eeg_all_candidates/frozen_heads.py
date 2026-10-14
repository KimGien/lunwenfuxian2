"""Native-dimensional subject readouts, with validation-only stop decisions."""
from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
import time

from tridep_ccml.tridep_stage1 import common as C
from tridep_ccml.tridep_eegpt import train_eval as T
from . import protocol as P, registry as R, storage as S
from .paired_replay import rng_state, restore_rng, rng_equal

np, torch = C.np, C.torch
REUSE_CONDITIONS = ('EEGPT-frozen-smallhead', 'CBraMod-frozen-smallhead',
                    'LaBraM-native5s-frozen-smallhead', 'LaBraM-native4s-frozen-smallhead')


class SubjectHead(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.window = torch.nn.Sequential(torch.nn.LayerNorm(dimension), torch.nn.Linear(dimension, 128),
                                         torch.nn.GELU(), torch.nn.Dropout(.2))
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, windows):
        if windows.ndim != 3 or windows.shape[1] != 60:
            raise ValueError('Frozen readout requires all60 windows per subject')
        return self.classifier(self.window(windows).mean(dim=1))


def require(value, message):
    if not value:
        raise ValueError(message)


def manifest_path(condition):
    return S.OUT / 'frozen_features' / condition / 'manifest.json'


def register_existing_features(condition):
    """Reuse only immutable external-encoder outputs, never fold-fitted features."""
    require(condition in REUSE_CONDITIONS, 'Unsupported historical frozen interface')
    path = manifest_path(condition)
    if path.exists():
        value = S.read(path)
        for source in value['sources']:
            S.checked(source)
        return value
    lock, folds, labels = C.context()
    T.validate_context(lock, folds, labels)
    from tridep_ccml.tridep_eegpt.common import OUT as OLD4
    source = OLD4 / 'features_manifest.json' if condition in ('EEGPT-frozen-smallhead',
              'LaBraM-native4s-frozen-smallhead') else C.OUT / ('input_manifest.json'
              if condition.startswith('CBraMod') else 'labram_manifest.json')
    records = S.read(source)
    records = records['records'] if isinstance(records, dict) else records
    by_id = {row['subject_id']: row for row in records}
    sources = [S.desc(source)]
    rows = []
    for sid in lock['subject_ids']:
        row = by_id[sid]
        if condition in ('EEGPT-frozen-smallhead', 'LaBraM-native4s-frozen-smallhead'):
            descriptor = row['features']
            key = 'eegpt_native512' if condition.startswith('EEGPT') else 'labram'
            with np.load(S.checked(descriptor), allow_pickle=False) as data:
                array = data[key].copy()
        else:
            descriptor, key = row['mean'], None
            array = np.load(S.checked(descriptor), allow_pickle=False)
        dimension = 512 if condition.startswith('EEGPT') else 200
        require(array.shape == (60, dimension) and array.dtype == np.float32
                and np.isfinite(array).all(), 'Invalid native frozen features')
        sources.append(descriptor)
        rows.append({'subject_id': sid, 'features': descriptor, 'archive_key': key})
    value = {'condition': condition, 'passed': True, 'dimension': dimension, 'subject_ids': lock['subject_ids'],
             'records': rows, 'sources': sources, 'encoder_fit_subject_ids': [], 'new_encoder_forwards': 0,
             'native_dimension_retained': True, 'projection_fitted': False,
             'pooling': 'mean native tokens within each window; all60 windows retained for learned readout',
             'input_seconds': P.CONDITION_BY_ID[condition].input_seconds,
             'reference': 'unchanged accepted E2v2 common/Cz reference; matches historical raw caches',
             'author_reference_difference': 'EEGPT average-reference pretraining differs; explicitly retained to compare output/training adaptation',
             'amplitude': 'historically audited reader numeric scaling; physical calibration unconfirmed',
             'registry': R.registration_descriptor()}
    S.save(path, value, immutable=True)
    return value


def load_features(condition):
    path = manifest_path(condition)
    value = S.read(path)
    require(value['passed'] and value['condition'] == condition and value['encoder_fit_subject_ids'] == [],
            'Feature manifest is not a fixed external frozen encoder')
    rows = []
    for row in value['records']:
        source = S.checked(row['features'])
        if row.get('archive_key'):
            with np.load(source, allow_pickle=False) as bundle:
                array = bundle[row['archive_key']].copy()
        else:
            array = np.load(source, allow_pickle=False)
        require(array.shape == (60, value['dimension']) and array.dtype == np.float32
                and np.isfinite(array).all(), 'Frozen feature shape or dtype mismatch')
        rows.append(array)
    require([r['subject_id'] for r in value['records']] == value['subject_ids'], 'Feature person order differs')
    return torch.as_tensor(np.stack(rows), device='cuda'), value


def prepare_condition(condition):
    R.load_registration()
    if condition in REUSE_CONDITIONS:
        register_existing_features(condition)
    engineering = S.OUT / 'engineering' / (condition + '.json')
    if engineering.exists():
        R._condition_lock(condition)
        return
    features, manifest = load_features(condition)
    lock, folds, labels = C.context()
    require(manifest['subject_ids'] == lock['subject_ids'], 'Feature cohort differs')
    C.runtime(11)
    model = SubjectHead(manifest['dimension']).to('cuda')
    chosen = folds['fold_1']['train'][:8]
    indices = [lock['subject_ids'].index(s) for s in chosen]
    yy = torch.tensor([labels[s] for s in chosen], device='cuda')
    logits = model(features[indices])
    loss = torch.nn.functional.cross_entropy(logits, yy)
    loss.backward()
    require(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()), 'Native head engineering gradient failure')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    state = rng_state()
    value = {'state_dict': T._cpu(model.state_dict()), 'optimizer_state_dict': T._cpu(optimizer.state_dict()),
             'rng': state, 'scheduler': {'policy': 'frozen', 'epoch': 0}}
    target = S.OUT / 'frozen_features' / condition / 'engineering_resume.pt'
    S.save_torch(target, value, immutable=True)
    saved = torch.load(target, map_location='cpu', weights_only=False)
    before = C.state_digest(model.state_dict())
    model.load_state_dict(saved['state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    restore_rng(saved['rng'])
    require(before == C.state_digest(model.state_dict()) and rng_equal(state, rng_state()), 'Head resume probe failed')
    contract = {'input_contract': {k: manifest.get(k) for k in ('dimension', 'input_seconds', 'reference',
                  'author_reference_difference', 'amplitude', 'pooling')},
                'trainable_parameter_names': [n for n, p in model.named_parameters()],
                'frozen_parameter_names': [], 'implementation_sources': [S.desc(Path(__file__)), S.desc(Path(S.__file__)),
                    S.desc(S.WORK / 'tridep_ccml/eeg_all_candidates/paired_replay.py')],
                'weight_sources': [S.desc(manifest_path(condition))],
                'forward_checked': True, 'gradient_checked': True, 'resume_checked': True,
                'training_only_engineering_checked': True, 'probe_subject_ids': chosen,
                'parameters': sum(p.numel() for p in model.parameters()),
                'engineering_optimizer_updates': 0, 'engineering_gradient_passes': 1,
                'frozen_basis': 'hash-bound fixed native feature arrays; external encoder not invoked while fitting'}
    R.lock_condition(condition, contract)
    for seed in P.SEEDS:
        for fold in P.FOLDS:
            R.record_fit_status(condition, seed, fold, 'ready')
    del features, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def validation(model, features, ids, split, labels):
    state = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    before = C.state_digest(model.state_dict())
    model.eval()
    result = {}
    try:
        with torch.inference_mode():
            for role in ('train', 'val'):
                chosen = split[role]
                ii = [ids.index(s) for s in chosen]
                yy = torch.tensor([labels[s] for s in chosen], device=features.device)
                logits = torch.cat([model(features[ii[start:start+8]]) for start in range(0, len(ii), 8)])
                loss = torch.nn.functional.cross_entropy(logits, yy)
                require(torch.isfinite(loss), 'Nonfinite validation loss')
                result[role + '_eval_ce'] = float(loss.cpu())
        require(before == C.state_digest(model.state_dict()), 'Validation changed model state')
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(state)
    require(rng_equal(state, rng_state()), 'Validation RNG restore failed')
    return result


def fit(condition, seed, fold, features, manifest, split, labels, *, resume=False):
    folder = R.fit_directory(condition, seed, fold)
    if (folder / 'complete.json').exists():
        evidence = S.read(folder / 'complete.json')
        R._validate_complete(evidence, folder)
        if R.fit_status(condition, seed, fold)['status'] != 'training_complete':
            R.record_fit_status(condition, seed, fold, 'training_complete', evidence=evidence)
        return S.read(folder / 'metadata.json')
    partial = (folder / 'started.json').exists()
    require(not partial or resume, 'Partial fit requires full-state --resume')
    R.record_fit_status(condition, seed, fold, 'training')
    if not partial:
        S.save(folder / 'started.json', {'condition': condition, 'seed': seed, 'fold': fold,
               'started_at': S.now(), 'train_ids': split['train'], 'feature_manifest': S.desc(manifest_path(condition))}, immutable=True)
    C.runtime(seed)
    model = SubjectHead(manifest['dimension']).to(features.device)
    initial = C.state_digest(model.state_dict())
    ids = manifest['subject_ids']
    indices, dataset = T.training_dataset(ids, split['train'], labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history, order, best, epoch, extended = [], [], None, 0, False

    def payload():
        return {'state_dict': T._cpu(model.state_dict()), 'optimizer_state_dict': T._cpu(optimizer.state_dict()),
                'scheduler_state_dict': {'policy': 'frozen', 'completed_epoch': epoch, 'extended': extended},
                'epoch': epoch, 'history': history, 'order': order, 'best_state_dict': best, 'rng': rng_state(),
                'best_state_sha256': C.state_digest(best) if best is not None else None,
                'initial_state_sha256': initial, 'sampler_position': {'epoch_completed': epoch, 'next_batch': 0},
                'registry': R.registration_descriptor(), 'feature_manifest': S.desc(manifest_path(condition))}

    if partial:
        target = folder / 'latest.pt' if (folder / 'latest.pt').exists() else folder / 'initial.pt'
        if not target.exists():
            require(not (folder / 'progress.json').exists(), 'Missing state after an optimizer step started')
            S.save_torch(folder / 'initial.pt', payload(), immutable=True)
        saved = torch.load(target, map_location='cpu', weights_only=False)
        require(saved['registry'] == R.registration_descriptor() and saved['feature_manifest'] == S.desc(manifest_path(condition))
                and saved['initial_state_sha256'] == initial, 'Resume identity/source differs')
        epoch, history, order, best = saved['epoch'], saved['history'], saved['order'], saved['best_state_dict']
        require(0 <= epoch <= 500 and len(history) == len(order) == epoch, 'Resume history/epoch differs')
        extended = saved['scheduler_state_dict']['extended']
        require(saved['scheduler_state_dict']['completed_epoch'] == epoch
                and saved['scheduler_state_dict']['policy'] == 'frozen'
                and saved['sampler_position'] == {'epoch_completed': epoch, 'next_batch': 0}, 'Resume scheduler/sampler differs')
        for i, row in enumerate(history):
            require(row['epoch'] == i+1 and row['updates_completed'] == row['gradient_passes_completed'] == (i+1)*4
                    and sorted(order[i]) == sorted(indices), 'Resume epoch/update/sample population differs')
        states = saved['optimizer_state_dict']['state']
        require((epoch == 0 and not states) or (len(states) == len(list(model.parameters()))
                and all(float(v['step']) == epoch*4 for v in states.values())), 'Resume optimizer step count differs')
        require(all(torch.isfinite(v).all() for v in saved['state_dict'].values()), 'Nonfinite resume head')
        require((epoch == 0 and best is None) or (best is not None
                and C.state_digest(best) == saved['best_state_sha256']), 'Resume validation-selected state differs')
        require(len(saved['rng']['torch_cuda']) == torch.cuda.device_count(), 'Resume CUDA RNG coverage differs')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        restore_rng(saved['rng'])
        require(rng_equal(saved['rng'], rng_state()), 'Resume RNG differs after restoration')
        S.save(folder / 'train_history.json', history)
        S.save(folder / 'validation_history.json', [{'epoch': r['epoch'], 'val_ce': r['val_eval_ce']} for r in history])
        if epoch in (60, 120, 200, 300):
            S.save_torch(folder / f'epoch_{epoch}.pt', saved, immutable=True)
    else:
        S.save_torch(folder / 'initial.pt', payload(), immutable=True)
    started = time.perf_counter()
    durable_epoch = epoch
    updates = history[-1]['updates_completed'] if history else 0
    start_updates = updates
    try:
        decision = P.training_decision('frozen', [r['val_eval_ce'] for r in history], extended=extended) if history else {'stop': False}
        while not decision['stop']:
            epoch += 1
            model.train()
            for group in optimizer.param_groups:
                group['lr'] = .001 * P.learning_rate_multiplier('frozen', epoch)
            losses, epoch_order = [], []
            for ii, yy in loader:
                require(all(int(i) in indices for i in ii), 'Nontrain subject reached optimizer')
                S.save(folder / 'progress.json', {'epoch': epoch, 'updates_started': updates+1,
                       'updates_completed': updates, 'batch_subject_ids': [ids[int(i)] for i in ii]})
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(features[ii.to(features.device)]), yy.to(features.device))
                require(torch.isfinite(loss), 'Nonfinite native-head training CE')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'Invalid head gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                updates += 1
                S.save(folder / 'progress.json', {'epoch': epoch, 'updates_completed': updates,
                       'gradient_passes_completed': updates})
                epoch_order.extend(ii.tolist())
                losses.append((len(ii), float(loss.detach().cpu())))
            require(sorted(epoch_order) == sorted(indices), 'Full subject coverage differs')
            order.append(epoch_order)
            row = {'epoch': epoch, 'updates_completed': updates, 'gradient_passes_completed': updates,
                   'train_ce': sum(n*v for n,v in losses)/len(indices), 'lr': optimizer.param_groups[0]['lr'],
                   **validation(model, features, ids, split, labels)}
            history.append(row)
            decision = P.training_decision('frozen', [r['val_eval_ce'] for r in history], extended=extended)
            if decision['best_epoch'] == epoch:
                best = T._cpu(model.state_dict())
            if decision['extend']:
                extended = True
            S.save_torch(folder / 'latest.pt', payload())
            durable_epoch = epoch
            S.save(folder / 'train_history.json', history)
            S.save(folder / 'validation_history.json', [{'epoch': r['epoch'], 'val_ce': r['val_eval_ce']} for r in history])
            if epoch in (60, 120, 200, 300):
                S.save_torch(folder / f'epoch_{epoch}.pt', payload(), immutable=True)
                S.save(folder / f'epoch_{epoch}_index.json', {'epoch': epoch, 'state_sha256': C.state_digest(model.state_dict()),
                       'best_epoch': decision['best_epoch'], 'checkpoint': S.desc(folder / f'epoch_{epoch}.pt')})
            if epoch % 60 == 0:
                print(f'TRAIN {condition} seed={seed} fold={fold} epoch={epoch} best={decision["best_epoch"]}', flush=True)
        require(best is not None, 'Validation-selected checkpoint missing')
        best_final = C.state_digest(best)
        require(best_final != initial, 'Selected head did not update')
        S.save_torch(folder / 'best.pt', {'state_dict': best, 'epoch': decision['best_epoch'],
                     'registry': R.registration_descriptor()}, immutable=True)
        model.load_state_dict(best, strict=True)
        model.eval()
        with torch.inference_mode():
            ii = indices[:8]
            reference = model(features[ii])
            reloaded = copy.deepcopy(model)
            reloaded.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)['state_dict'], strict=True)
            require(torch.equal(reference, reloaded(features[ii])), 'Native-head reload prediction differs')
        del reloaded
        meta = {'condition': condition, 'seed': seed, 'fold': fold, 'epochs': epoch, 'updates': updates,
                'gradient_passes': updates, 'best_epoch': decision['best_epoch'], 'stopping_reason': decision['reason'],
                'extended': extended, 'dimension': manifest['dimension'], 'parameters': sum(p.numel() for p in model.parameters()),
                'input_seconds': manifest['input_seconds'], 'initial_state_sha256': initial, 'selected_state_sha256': best_final,
                'feature_manifest': S.desc(manifest_path(condition)), 'train_ids': split['train'], 'val_ids': split['val'],
                'test_ids': split['test'], 'test_forward_count': 0, 'new_encoder_forwards': 0,
                'reload_probe_subject_ids': split['train'][:8], 'seconds_this_attempt': time.perf_counter()-started}
        if (folder / 'metadata.json').exists():
            previous = S.read(folder / 'metadata.json')
            require(all(previous[k] == meta[k] for k in ('condition', 'seed', 'fold', 'epochs', 'updates',
                    'best_epoch', 'stopping_reason', 'initial_state_sha256', 'selected_state_sha256')),
                    'Existing selected-head metadata differs')
            meta = previous
        else:
            S.save(folder / 'metadata.json', meta, immutable=True)
        evidence = {'passed': True, 'checkpoint_reload_verified': True, 'parameter_update_verified': True,
                    'frozen_parameters_verified': True, 'epochs': epoch, 'optimizer_updates': updates,
                    'stop_reason': decision['reason'], 'registry': R.registration_descriptor(),
                    **{key: S.desc(folder/name) for key,name in (('checkpoint','best.pt'), ('train_history','train_history.json'),
                       ('validation_history','validation_history.json'), ('metadata','metadata.json'))}}
        S.save(folder / 'complete.json', evidence, immutable=True)
        R.record_fit_status(condition, seed, fold, 'training_complete', evidence=evidence)
        print(f'COMPLETE {condition} seed={seed} fold={fold} epochs={epoch} updates={updates}', flush=True)
        return meta
    except BaseException as error:
        S.save(folder / ('failure_'+str(time.time_ns())+'.json'), {'error': str(error), 'updates_completed': updates,
               'updates_completed_this_process': updates-start_updates,
               'last_durable_epoch': durable_epoch, 'epoch_started': epoch, 'time': S.now()})
        R.record_fit_status(condition, seed, fold, 'failed', reason=str(error))
        raise
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def run(conditions=REUSE_CONDITIONS, *, resume=False, limit=None):
    completed = 0
    lock, folds, labels = C.context()
    for condition in conditions:
        require(P.CONDITION_BY_ID[condition].policy_key == 'frozen', 'Only frozen small-head conditions supported')
        prepare_condition(condition)
        features, manifest = load_features(condition)
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                fit(condition, seed, fold, features, manifest, folds[f'fold_{fold}'], labels, resume=resume)
                completed += 1
                if limit is not None and completed >= limit:
                    return
        del features
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs='+', default=list(REUSE_CONDITIONS))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.conditions, resume=args.resume, limit=args.limit)
