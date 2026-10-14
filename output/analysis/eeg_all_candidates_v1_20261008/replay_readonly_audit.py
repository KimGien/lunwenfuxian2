"""CPU-only, independent source/init/sampling audit. No model forward or fitting."""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import torch

WORK = Path(__file__).resolve().parents[3]
OLD = WORK / 'tridep_ccml/experiments/TriDep-Stage4-EEGPT-Matched4s-37-fulltext-20261006'
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        x = tensor.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def independent_initial(seed):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    state = {}
    gain = math.sqrt(2. / (1. + math.sqrt(5.) ** 2))
    for prefix, shape in (('conv.0', (256, 200, 3)), ('conv.3', (128, 256, 3)),
                          ('classifier.2', (128, 1920)), ('classifier.5', (2, 128))):
        fan = math.prod(shape[1:])
        bound = math.sqrt(3.) * gain / math.sqrt(fan)
        state[prefix + '.weight'] = torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound, generator=gen)
        state[prefix + '.bias'] = torch.empty((shape[0],), dtype=torch.float32).uniform_(-1 / math.sqrt(fan), 1 / math.sqrt(fan), generator=gen)
    assert sum(v.numel() for v in state.values()) == 498434
    return state, gen.get_state().clone()


def independent_orders(cpu_after_init, indices, epochs=300):
    gen = torch.Generator(device='cpu')
    gen.set_state(cpu_after_init)
    orders = []
    states = {}
    for epoch in range(1, epochs + 1):
        # DataLoader._BaseDataLoaderIter consumes one seed even num_workers=0.
        _base_seed = int(torch.empty((), dtype=torch.int64).random_(generator=gen).item())
        # Default RandomSampler consumes its local seed from the global CPU stream.
        sampler_seed = int(torch.empty((), dtype=torch.int64).random_(generator=gen).item())
        local = torch.Generator(device='cpu').manual_seed(sampler_seed)
        perm = torch.randperm(len(indices), generator=local).tolist()
        # Pinned PyTorch 2.8 implementation also makes a second empty-tail perm.
        _empty_tail = torch.randperm(len(indices), generator=local).tolist()[:0]
        orders.append([indices[i] for i in perm])
        if epoch in (60, 120, 200, 300):
            states[str(epoch)] = hashlib.sha256(gen.get_state().numpy().tobytes()).hexdigest()
    return np.asarray(orders, dtype=np.int64), states


def main():
    lock = read(WORK / 'S00/configs/cohort.json')
    folds_path = WORK / 'S00' / lock['fold_file']
    assert digest(folds_path) == lock['folds_sha256']
    folds = read(folds_path)
    reg = read(OLD / 'registration.json')
    assert reg['subject_ids'] == lock['subject_ids'] and reg['folds_sha256'] == lock['folds_sha256']
    checks = []
    source_records = {}
    for seed in (11, 22, 33):
        expected, cpu_after_init = independent_initial(seed)
        expected_initial_hash = state_digest(expected)
        for fold in range(1, 6):
            split = folds[f'fold_{fold}']
            indices = [lock['subject_ids'].index(s) for s in split['train']]
            expected_order, cpu_endpoint_hashes = independent_orders(cpu_after_init, indices)
            pair = []
            for condition in reg['conditions']:
                folder = OLD / f'runs/{condition}/seed_{seed}/fold_{fold}'
                complete = read(folder / 'complete.json')
                assert complete['passed'] and complete['updates'] == 120
                assert len(complete['artifacts']) == 7
                for artifact in complete['artifacts']:
                    path = WORK / artifact['path']
                    assert path.parent == folder and digest(path) == artifact['sha256']
                    source_records[artifact['path']] = artifact['sha256']
                initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
                saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
                meta = read(folder / 'metadata.json')
                assert set(initial['state_dict']) == set(expected)
                assert all(torch.equal(initial['state_dict'][k], expected[k]) for k in expected)
                assert meta['initial_state_sha256'] == expected_initial_hash
                assert saved['metadata'] == meta and state_digest(saved['state_dict']) == meta['final_state_sha256']
                assert meta['train_ids'] == split['train'] and meta['val_ids'] == split['val'] and meta['test_ids'] == split['test']
                order = np.load(folder / 'batch_order.npy', allow_pickle=False)
                assert order.dtype == np.int64 and np.array_equal(order, expected_order[:60])
                assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == meta['batch_order_sha256']
                optimizer = saved['optimizer_state_dict']
                assert len(optimizer['state']) == 8 and len(optimizer['param_groups']) == 1
                assert all(float(v['step']) == 120 for v in optimizer['state'].values())
                assert all(set(v) == {'step', 'exp_avg', 'exp_inf'} for v in optimizer['state'].values())
                assert optimizer['param_groups'][0]['lr'] == .005 and optimizer['param_groups'][0]['weight_decay'] == 0.
                history = read(folder / 'train_history.json')
                assert len(history) == 60
                for epoch, row in enumerate(history, 1):
                    assert row['epoch'] == epoch and row['updates_completed'] == row['gradient_passes_completed'] == epoch * 2
                    assert [b['batch_size'] for b in row['batches']] == [16, len(indices) - 16]
                    assert all(math.isfinite(b['train_ce']) for b in row['batches'])
                    expected_mean = sum(b['batch_size'] * b['train_ce'] for b in row['batches']) / len(indices)
                    assert abs(expected_mean - row['train_ce']) <= 1e-12
                checks.append({'condition': condition, 'seed': seed, 'fold': fold,
                               'initial_tensor_only_recipe_exact': True,
                               'all_60_epoch_orders_independently_exact': True,
                               'cpu_rng_after_initialization_sha256': hashlib.sha256(cpu_after_init.numpy().tobytes()).hexdigest(),
                               'independent_cpu_rng_endpoint_sha256': cpu_endpoint_hashes,
                               'epochs': 60, 'saved_optimizer_steps': 120,
                               'old_checkpoint_has_rng': any('rng' in k.lower() for k in saved),
                               'optimizer_param_group': optimizer['param_groups'][0]})
                pair.append((expected_initial_hash, meta['batch_order_sha256']))
            assert pair[0] == pair[1]
    result = {'passed': True, 'scope': 'CPU-only static checkpoint and independent sampling audit',
              'new_model_constructions': 0, 'new_neural_forwards': 0,
              'new_training_fits': 0, 'new_optimizer_updates': 0,
              'fits_checked': len(checks), 'paired_fits_checked': 15,
              'independent_epoch_orders_checked': 1800,
              'bound_artifacts_checked': len(source_records),
              'runtime': {'python_executable': sys.executable, 'python': sys.version,
                          'torch': torch.__version__, 'numpy': np.__version__},
              'checks': checks, 'historical_artifact_sha256': source_records}
    path = OUT / 'replay_readonly_audit.json'
    assert not path.exists(), 'Preserve prior reviewer audit'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'fits_checked', 'independent_epoch_orders_checked',
                                          'bound_artifacts_checked', 'new_neural_forwards', 'new_training_fits')}))


if __name__ == '__main__':
    main()
