"""Independent CPU-only audit of completed paired300 artifacts, with no scoring."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch

from replay_readonly_audit import WORK, OLD, independent_initial, independent_orders, state_digest

NEW = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(record):
    path = WORK / record['path']
    assert path.resolve().is_relative_to(WORK) and digest(path) == record['sha256']
    return path


def equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    source_lock = read(NEW / 'paired_replay/source_lock.json')
    for descriptor in source_lock['sources']:
        checked(descriptor)
    lock = read(WORK / 'S00/configs/cohort.json')
    folds = read(WORK / 'S00' / lock['fold_file'])
    registry = {'path': (NEW / 'registration.json').relative_to(WORK).as_posix(), 'sha256': digest(NEW / 'registration.json')}
    checks = []
    pair_states = {}
    for completion_path in sorted((NEW / 'runs').rglob('complete.json')):
        folder = completion_path.parent
        completion = read(completion_path)
        meta = read(folder / 'metadata.json')
        if not meta['condition'].endswith('-300ep'):
            continue
        assert completion['registry'] == registry and completion['passed']
        for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'):
            checked(completion[key])
        condition, seed, fold = meta['condition'], meta['seed'], meta['fold']
        old = OLD / f"runs/{condition.removesuffix('-300ep')}/seed_{seed}/fold_{fold}"
        oldmeta, oldhistory = read(old / 'metadata.json'), read(old / 'train_history.json')
        oldcheckpoint = torch.load(old / 'checkpoint.pt', map_location='cpu', weights_only=True)
        expected_initial, cpu_after_init = independent_initial(seed)
        split = folds[f'fold_{fold}']
        indices = [lock['subject_ids'].index(s) for s in split['train']]
        expected_orders, expected_cpu = independent_orders(cpu_after_init, indices)
        initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=False)
        assert equal(initial['state_dict'], expected_initial)
        assert torch.equal(initial['rng']['torch_cpu'], cpu_after_init)
        history, validation = read(folder / 'train_history.json'), read(folder / 'validation_history.json')
        assert len(history) == len(validation) == 300
        assert meta['epochs'] == 300 and meta['updates'] == meta['gradient_passes'] == 600
        assert meta['train_ids'] == split['train'] and meta['val_ids'] == split['val'] and meta['test_ids'] == split['test']
        assert meta['test_forward_count'] == meta['encoder_forwards'] == 0
        for epoch, row in enumerate(history, 1):
            assert row['epoch'] == epoch and row['updates_completed'] == row['gradient_passes_completed'] == epoch * 2
            assert [b['batch_size'] for b in row['batches']] == [16, len(indices) - 16]
            assert all(math.isfinite(b['train_ce']) and b['train_ce'] >= 0 for b in row['batches'])
            assert abs(row['train_ce'] - sum(b['batch_size'] * b['train_ce'] for b in row['batches']) / len(indices)) <= 1e-12
            assert math.isfinite(row['train_eval_ce']) and math.isfinite(row['val_eval_ce'])
            assert validation[epoch - 1] == {'epoch': epoch, 'val_ce': row['val_eval_ce']}
            if epoch <= 60:
                assert {k: row[k] for k in oldhistory[epoch - 1]} == oldhistory[epoch - 1]
        snapshots = {}
        for epoch in (60, 120, 200, 300):
            payload = torch.load(checked(meta['snapshots'][str(epoch)]), map_location='cpu', weights_only=False)
            assert payload['registry'] == registry and payload['epoch'] == len(payload['history']) == len(payload['order']) == epoch
            assert payload['history'] == history[:epoch] and np.array_equal(payload['order'], expected_orders[:epoch])
            assert hashlib.sha256(payload['rng']['torch_cpu'].numpy().tobytes()).hexdigest() == expected_cpu[str(epoch)]
            assert payload['sampler_position'] == {'epoch_completed': epoch, 'next_epoch': epoch + 1, 'next_batch': 0}
            assert len(payload['optimizer_state_dict']['state']) == 8
            assert all(float(value['step']) == epoch * 2 for value in payload['optimizer_state_dict']['state'].values())
            assert payload['optimizer_state_dict']['param_groups'] == oldcheckpoint['optimizer_state_dict']['param_groups']
            if epoch == 60:
                assert equal(payload['state_dict'], oldcheckpoint['state_dict'])
                assert equal(payload['optimizer_state_dict'], oldcheckpoint['optimizer_state_dict'])
            if epoch == 300:
                assert state_digest(payload['state_dict']) == meta['final_state_sha256']
            snapshots[epoch] = payload['rng']['torch_cuda']
        order_hash = hashlib.sha256(expected_orders.astype('<i8').tobytes()).hexdigest()
        assert order_hash == meta['batch_order_sha256']
        first60 = read(folder / 'first60_replay_verification.json')
        assert all(first60[k] is True for k in ('passed', 'exact_initial_state', 'exact_all_120_batch_losses',
                                              'exact_batch_order', 'exact_epoch60_state', 'exact_optimizer_state'))
        assert checked(first60['historical_checkpoint']) == old / 'checkpoint.pt'
        checks.append({'condition': condition, 'seed': seed, 'fold': fold, 'epochs': 300, 'optimizer_updates': 600,
                       'exact_120_historical_minibatch_losses': True, 'exact_old_epoch60_model_and_optimizer': True,
                       'all300_orders_independently_exact': True, 'all4_snapshot_cpu_rng_independently_exact': True,
                       'all4_snapshot_optimizer_counts_exact': True, 'bound_completion_artifacts_checked': 4,
                       'test_forward_count': 0, 'new_auditor_neural_forwards': 0})
        pair_states.setdefault((seed, fold), []).append((condition, state_digest(expected_initial), order_hash, snapshots))
    pairs = []
    for (seed, fold), states in pair_states.items():
        if len(states) == 2:
            assert states[0][1:3] == states[1][1:3]
            assert equal(states[0][3], states[1][3])
            pairs.append({'seed': seed, 'fold': fold, 'initial_and_300epoch_order_exact': True,
                          'cuda_rng_all4_snapshots_exact_between_conditions': True})
    assert checks, 'No completed paired fit to audit'
    result = {'passed': True, 'completed_fits_checked': len(checks), 'paired_groups_checked': len(pairs),
              'new_neural_forwards': 0, 'new_optimizer_updates': 0, 'external_test_scores_computed': 0,
              'source_lock_artifacts_checked': len(source_lock['sources']), 'checks': checks, 'pairs': pairs}
    target = REPORT / f"paired_completed_readonly_audit_{len(checks):02d}fits.json"
    assert not target.exists(), 'Preserve previous independent completion audit'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'completed_fits_checked', 'paired_groups_checked', 'new_neural_forwards')}))


if __name__ == '__main__':
    main()
