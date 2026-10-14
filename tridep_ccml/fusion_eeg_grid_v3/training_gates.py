"""Synthetic CPU gates for selection, schedule, durable recovery and extension.

These are engineering updates only. No cohort cache or test prediction is read.
Run with ``python -m tridep_ccml.fusion_eeg_grid_v3.training_gates``.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from . import common as C
from . import training as T
from . import preprocessing as P


def fixture(views, device='cpu'):
    generator = torch.Generator().manual_seed(20261014 + views)
    raw = torch.randn(37, views, 128, generator=generator).numpy()
    raw[:27, :, 0] = 2.
    raw[27:, :, 0] = 7.
    labels = np.asarray([i % 2 for i in range(37)], dtype=np.int64)
    ids = [f'synthetic_{i:03}' for i in range(37)]
    indexes = {'train': np.arange(27), 'val': np.arange(27, 30), 'test': np.arange(30, 37)}
    roles = {key: [ids[i] for i in values] for key, values in indexes.items()}
    state = P.fit_preprocessor(raw[:27], variant='N1', fit_people=roles['train'])
    x = P.apply_preprocessor(raw, state)
    binding = {'synthetic_only': True, 'subject_ids': ids, 'labels': labels.tolist(),
               'roles': roles, 'sources': [], 'normalization': state}
    return torch.tensor(x, device=device), torch.tensor(labels, device=device), indexes, binding


def fixture_loader(identity, device, feature_root=None):
    C.require(feature_root is None, 'Synthetic fixture does not accept source overrides')
    return fixture(2 if identity['eeg'] == 'ST' else 3, device)


def identity(method, views, qmf_variant='Q1', budget=120):
    return {'stage': 'synthetic_engineering', 'eeg': 'ST' if views == 2 else 'E00',
            'method': method, 'seed': 11, 'fold': 1, 'normalization': 'N1',
            'qmf_variant': qmf_variant, 'base_lr': 1e-3, 'budget': budget}


def selection_schedule_gate():
    values = {}
    for lr in (3e-4, 1e-3, 3e-3):
        C.require(T.lr_for_epoch(1, lr) == lr / 10 and T.lr_for_epoch(10, lr) == lr,
                  'Warmup endpoints differ')
        C.require(abs(T.lr_for_epoch(120, lr) - .1 * lr) < 1e-18 and
                  all(T.lr_for_epoch(epoch, lr) == .1 * lr for epoch in range(121, 241)),
                  'Extension must retain the prefix endpoint learning rate')
        values[str(lr)] = {str(epoch): T.lr_for_epoch(epoch, lr) for epoch in (1, 10, 60, 120, 121, 180, 240)}
    best, be, reference, stale = None, 0, None, 0
    for epoch, nll in enumerate((1., .99995, .99995, .9998), 1):
        best, be, reference, stale, selected = T.selection_update(best, be, reference, stale, nll, epoch)
        if epoch == 2:
            C.require(selected and be == 2 and stale == 1 and reference == 1.,
                      'Sub-min_delta minimum must select best without resetting patience')
        if epoch == 3:
            C.require(not selected and be == 2 and stale == 2, 'Exact best ties must keep the earlier epoch')
        if epoch == 4:
            C.require(selected and stale == 0 and reference == .9998, 'Accumulated min_delta failed')
    C.require(not T.should_stop(59, 60, 120) and T.should_stop(60, 60, 120) and
              T.should_stop(120, 0, 120) and not T.should_stop(120, 0, 240), 'Stopping policy differs')
    return {'passed': True, 'rates': values, 'strict_best_separate_from_patience': True,
            'min_epoch_60_and_patience_60': True}


def _without_cost(value):
    if isinstance(value, dict):
        return {k: _without_cost(v) for k, v in value.items()
                if k not in ('train_seconds', 'validation_seconds', 'write_seconds')}
    if isinstance(value, list):
        return [_without_cost(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_without_cost(v) for v in value)
    return value


def _compare(first, second):
    keys = ('state_dict', 'optimizer', 'scheduler', 'rng', 'quality_history', 'epoch', 'updates',
            'best_nll', 'best_epoch', 'best_prob', 'patience_ref', 'stale', 'sampling',
            'input_binding', 'configuration', 'initial_sha')
    for key in keys:
        C.require(T.equal_state(first[key], second[key]), 'Recovery is not bitwise equal: ' + key)
    for key in ('ledger', 'best_checkpoint'):
        C.require(T.equal_state(_without_cost(first[key]), _without_cost(second[key])),
                  'Recovery selection/trajectory differs: ' + key)
    return list(keys) + ['trajectory_excluding_wall_clock', 'complete_selected_state_excluding_wall_clock']


def recovery_gate(method, views, variant, run_root, registration):
    name = f'{method}_{views}_{variant}'
    ident = identity(method, views, variant)
    direct = run_root / name / 'continuous'
    interrupted = run_root / name / 'resumed'
    T.fit(ident, 'cpu', registration, directory=direct, engineering_stop_after=4)
    T.fit(ident, 'cpu', registration, directory=interrupted, engineering_stop_after=2)
    halfway = torch.load(interrupted / 'latest.pt', map_location='cpu', weights_only=False)
    C.require(halfway['epoch'] == 2 and halfway['updates'] == 8,
              'Engineering interruption not at the intended boundary')
    T.fit(ident, 'cpu', registration, directory=interrupted, engineering_stop_after=4)
    a = torch.load(direct / 'latest.pt', map_location='cpu', weights_only=False)
    b = torch.load(interrupted / 'latest.pt', map_location='cpu', weights_only=False)
    compared = _compare(a, b)
    C.require(all(row['batch_sizes'] == [8, 8, 8, 3] and row['parameter_update_l2'] > 0
                  for row in a['ledger']), 'Last batch or actual parameter update evidence missing')
    C.require(not (direct / 'complete.json').exists() and not (interrupted / 'complete.json').exists(),
              'Engineering prefix must not claim a successful formal fit')
    return {'method': method, 'views': views, 'qmf_variant': variant, 'device': 'cpu', 'passed': True,
            'bitwise_fields': compared, 'optimizer_updates_per_trajectory': 16,
            'formal_fits': 0, 'latest': C.desc(interrupted / 'latest.pt')}


def pause_gate(run_root, registration):
    ident = identity('QMF', 3)
    direct, resumed = run_root / 'pause' / 'continuous', run_root / 'pause' / 'resumed'
    T.fit(ident, 'cpu', registration, directory=direct, engineering_stop_after=4)
    original_pause = C.check_pause
    calls = 0
    def boundary_pause():
        nonlocal calls
        original_pause()
        calls += 1
        if calls == 4:
            raise InterruptedError('Synthetic user-pause boundary after epoch 2')
    try:
        C.check_pause = boundary_pause
        caught = False
        try:
            T.fit(ident, 'cpu', registration, directory=resumed, engineering_stop_after=4)
        except InterruptedError:
            caught = True
        C.require(caught, 'Pause did not stop the engine')
    finally:
        C.check_pause = original_pause
    stopped = torch.load(resumed / 'latest.pt', map_location='cpu', weights_only=False)
    C.require(stopped['epoch'] == 2 and stopped['updates'] == 8 and not (resumed / 'complete.json').exists(),
              'Pause passed its durable epoch boundary')
    events = [C.read(path) for path in resumed.glob('event_*.json')]
    C.require(len(events) == 1 and events[0]['paused_at_boundary'] and
              events[0]['unknown_uncommitted_tail_updates'] == 0, 'Pause boundary was not recorded accurately')
    T.fit(ident, 'cpu', registration, directory=resumed, engineering_stop_after=4)
    a = torch.load(direct / 'latest.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed / 'latest.pt', map_location='cpu', weights_only=False)
    compared = _compare(a, b)
    return {'passed': True, 'synthetic_pause_only': True, 'real_user_pause_file_modified': False,
            'paused_epoch': 2, 'resumed_bitwise_fields': compared, 'formal_fits': 0}


def extension_gate(run_root, registration):
    """Actual 120 synthetic epochs; controlled validation only isolates policy.

    The imposed monotonically improving NLL is explicitly not a measured model
    metric. All parameter updates, optimizer states and replay probabilities are
    real, allowing a deterministic continuation and immutable-parent test.
    """
    original_nll = T.nll

    def schedule_from(epoch):
        counter = epoch
        def controlled(_prob, _labels):
            nonlocal counter
            counter += 1
            return torch.tensor(2. - counter * .002, dtype=torch.float64)
        return controlled

    parent = run_root / 'extension' / 'parent120'
    direct = run_root / 'extension' / 'continuous124'
    resumed = run_root / 'extension' / 'resumed124'
    base = identity('CCML', 2)
    extension = dict(base, stage='synthetic_extension', budget=240)
    try:
        original_save_torch = C.save_torch
        interrupted_snapshots = {}
        def fail_milestone_once(path, value):
            if path.parent == parent and path.name in ('epoch_60.pt', 'epoch_120.pt') and path.name not in interrupted_snapshots:
                interrupted_snapshots[path.name] = copy.deepcopy(value)
                raise OSError('Synthetic interruption after latest and before milestone snapshot')
            return original_save_torch(path, value)
        C.save_torch = fail_milestone_once
        try:
            while True:
                start_epoch = torch.load(parent / 'latest.pt', map_location='cpu', weights_only=False)['epoch'] if (parent / 'latest.pt').exists() else 0
                T.nll = schedule_from(start_epoch)
                try:
                    endpoint = T.fit(base, 'cpu', registration, directory=parent)
                    break
                except OSError as error:
                    C.require('Synthetic interruption' in str(error), 'Unexpected persistence failure')
        finally:
            C.save_torch = original_save_torch
        for name, intended in interrupted_snapshots.items():
            recovered = torch.load(parent / name, map_location='cpu', weights_only=False)
            C.require(T.equal_state(recovered, intended), 'Recovered milestone is not its exact durable checkpoint')
        C.require(len(interrupted_snapshots) == 2, 'Did not exercise both milestone transaction windows')
        C.require(endpoint['epochs'] == 120 and endpoint['updates'] == 480,
                  'Extension fixture did not actually reach 120 epochs')
        original = {name: C.desc(parent / name) for name in
                    ('latest.pt', 'best.pt', 'complete.json', 'trajectory.json', 'metadata.json')}
        T.nll = schedule_from(120)
        T.fit(extension, 'cpu', registration, directory=direct, parent_directory=parent,
              engineering_stop_after=124)
        T.nll = schedule_from(120)
        T.fit(extension, 'cpu', registration, directory=resumed, parent_directory=parent,
              engineering_stop_after=122)
        T.nll = schedule_from(122)
        T.fit(extension, 'cpu', registration, directory=resumed, parent_directory=parent,
              engineering_stop_after=124)
        a = torch.load(direct / 'latest.pt', map_location='cpu', weights_only=False)
        b = torch.load(resumed / 'latest.pt', map_location='cpu', weights_only=False)
        compared = _compare(a, b)
        C.require(a['prefix_epochs'] == 120 and a['prefix_updates'] == 480 and a['updates'] == 496,
                  'Extension reset or miscounted the original prefix')
        C.require(all(row['lr'] == 1e-4 for row in a['ledger'][120:]), 'Extension learning rate differs')
        C.require(all(C.desc(parent / name) == descriptor for name, descriptor in original.items()),
                  'Extension modified a parent artifact')
        # An exhausted parent must be rejected even if it physically reached 120.
        parent_saved = torch.load(parent / 'latest.pt', map_location='cpu', weights_only=False)
        original_load = torch.load
        def stale_parent_load(path, *args, **kwargs):
            result = original_load(path, *args, **kwargs)
            if str(path) == str(parent / 'latest.pt'):
                result = copy.deepcopy(result)
                result['stale'] = 60
            return result
        torch.load = stale_parent_load
        rejected = False
        try:
            T._parent(extension, parent, 'cpu', parent_saved['input_binding'], T.configuration(extension))
        except ValueError as error:
            rejected = 'patience' in str(error)
        finally:
            torch.load = original_load
        C.require(rejected, 'Patience-exhausted epoch-120 parent was accepted')
        return {'passed': True, 'method': 'CCML', 'views': 2, 'device': 'cpu',
                'actual_synthetic_prefix_epochs': 120, 'actual_synthetic_prefix_updates': 480,
                'actual_extension_updates_per_trajectory': 16, 'formal_fits': 0,
                'controlled_validation_metric': '2 - absolute_epoch * 0.002; policy-only fixture',
                'parent_immutable': True, 'exhausted_parent_rejected': True,
                'milestone_write_failure_recovered_bitwise': sorted(interrupted_snapshots),
                'bitwise_fields': compared, 'parent_artifacts': original,
                'latest': C.desc(resumed / 'latest.pt')}
    finally:
        T.nll = original_nll


def extension_patience_gate(run_root, registration):
    """Carry stale=59 and an old best across the 120/240 directory boundary."""
    parent = run_root / 'extension_patience' / 'parent120'
    extended = run_root / 'extension_patience' / 'extended240'
    ident = identity('ECML', 3)
    extension = dict(ident, stage='synthetic_extension', budget=240)
    original_nll = T.nll
    counter = 0
    def controlled(_prob, _labels):
        nonlocal counter
        counter += 1
        return torch.tensor(2. - min(counter, 61) * .002, dtype=torch.float64)
    try:
        T.nll = controlled
        first = T.fit(ident, 'cpu', registration, directory=parent)
        saved = torch.load(parent / 'latest.pt', map_location='cpu', weights_only=False)
        C.require(first['epochs'] == 120 and saved['stale'] == 59 and saved['best_epoch'] == 61,
                  'Nonzero-patience parent fixture is incorrect')
        hashes = {name: C.desc(parent / name) for name in ('latest.pt', 'best.pt', 'complete.json', 'trajectory.json')}
        continued = T.fit(extension, 'cpu', registration, directory=extended, parent_directory=parent)
        latest = torch.load(extended / 'latest.pt', map_location='cpu', weights_only=False)
        best = torch.load(extended / 'best.pt', map_location='cpu', weights_only=False)
        C.require(continued['epochs'] == 121 and continued['additional_epochs'] == 1 and
                  continued['additional_updates'] == 4 and latest['stale'] == 60 and latest['best_epoch'] == 61,
                  'Extension reset inherited patience or selected an equal validation value')
        C.require(T.equal_state(best, saved['best_checkpoint']) and
                  T.equal_state(latest['best_prob'], saved['best_prob']), 'Extension discarded the parent best')
        C.require(all(C.desc(parent / name) == value for name, value in hashes.items()), 'Patience extension changed parent')
        C.require(not (extended / 'epoch_180.pt').exists() and not (extended / 'epoch_240.pt').exists(),
                  'Extension manufactured unachieved epoch snapshots')
        return {'passed': True, 'method': 'ECML', 'views': 3, 'formal_fits': 0,
                'actual_synthetic_prefix_epochs': 120, 'actual_synthetic_prefix_updates': 480,
                'actual_extension_epochs': 1, 'actual_extension_updates': 4,
                'stale_at_120': 59, 'stale_at_121': 60, 'best_epoch_preserved': 61,
                'controlled_validation_metric': '2 - min(epoch, 61) * 0.002; policy-only fixture',
                'parent_immutable': True, 'unreached_snapshots_absent': True,
                'complete': C.desc(extended / 'complete.json')}
    finally:
        T.nll = original_nll


def run():
    C.check_pause()
    run_root = C.REPORT / 'training_gates' / f'run_{time.time_ns()}'
    run_root.mkdir(parents=True, exist_ok=True)
    registration = {'engineering_only': True, 'synthetic_only': True,
                    'sources': [C.desc(C.CODE / name) for name in
                                ('training.py', 'training_gates.py', 'models.py', 'preprocessing.py', 'common.py')]}
    C.save(run_root / 'registration.json', registration, immutable=True)
    for source in registration['sources']:
        source_path = C.checked(source)
        C.atomic(run_root / 'source_snapshots' / source_path.name, source_path.read_bytes(), immutable=True)
    original_loader = T.load_inputs
    started = time.perf_counter()
    try:
        T.load_inputs = fixture_loader
        policy = selection_schedule_gate()
        recoveries = []
        for method in C.METHODS:
            for views in (2, 3):
                C.check_pause()
                recoveries.append(recovery_gate(method, views, 'Q1', run_root, registration))
                print(f'Training gate passed: {method} {views} views Q1', flush=True)
        for views in (2, 3):
            recoveries.append(recovery_gate('QMF', views, 'Q0', run_root, registration))
        pause = pause_gate(run_root, registration)
        extension = extension_gate(run_root, registration)
        extension_patience = extension_patience_gate(run_root, registration)
        value = {'passed': True, 'formal_fits': 0, 'test_forwards': 0, 'device': 'cpu',
                 'synthetic_only': True, 'policy': policy, 'recoveries': recoveries, 'pause': pause, 'extension': extension,
                 'extension_patience': extension_patience,
                 'engineering_seconds': time.perf_counter() - started, 'sources': registration['sources'],
                 'at': C.now(), 'limitations': ['GPU/backend benchmark is a separate gate',
                     'Extension policy fixture uses a controlled validation metric, not a performance result']}
        C.save(run_root / 'complete.json', value, immutable=True)
        C.save(C.REPORT / 'training_gates' / 'latest.json', {'passed': True, 'report': C.desc(run_root / 'complete.json')})
        print(str(run_root / 'complete.json'), flush=True)
        return value
    except BaseException as error:
        C.save(run_root / 'failure.json', {'passed': False, 'error': repr(error),
                   'engineering_seconds': time.perf_counter() - started, 'sources': registration['sources'], 'at': C.now()})
        raise
    finally:
        T.load_inputs = original_loader


if __name__ == '__main__':
    run()
