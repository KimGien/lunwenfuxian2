"""Prepare, fit, and uniformly evaluate the registered paired EEG stage."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import unittest

from .common import *
from .models import CBraModTokenHead
from .labram_assets import (load_frozen_labram, extract_patch_tokens,
                            verify_assets, MANIFEST as LABRAM_MANIFEST,
                            ASSETS as LABRAM_ASSETS)


def original_model():
    sys.path.insert(0, str(WORK / 'S00/upstream-dev'))
    from lib.models.models import EEGConvPoolClassifier
    return EEGConvPoolClassifier(input_dim=200, seq_len=60)


def model_for(condition, adjacency):
    return original_model() if condition.endswith('originalCNN') else CBraModTokenHead(condition[3:], adjacency)


def optim_config(condition):
    cnn = condition.endswith('originalCNN')
    return {'epochs': 60, 'batch_size': 16 if cnn else 8, 'optimizer': 'Adamax',
            'learning_rate': .005 if cnn else .001, 'weight_decay': 0. if cnn else .0001,
            'gradient_clip': None if cnn else 1., 'parameters': 498434 if cnn else 12098,
            'updates_per_model': 120 if cnn else 240, 'checkpoint_selection': 'fixed_final_epoch'}


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(CODE), pattern='test_*.py', top_level_dir=str(WORK))
    with (OUT / 'tests.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError('Stage1 contract tests failed; see tests.log')
    save(OUT / 'test_results.json', {'passed': True, 'tests': result.testsRun,
                                    'failures': len(result.failures), 'errors': len(result.errors)})


def prepare():
    if (OUT / 'registration.json').exists():
        verify_registration()
        print('Registered stage1 input and assets already verified', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    runtime(11)
    # The completed phase0 evidence must still be intact before adding new fits.
    from tridep_ccml.stage0_verify import run as verify_phase0
    verify_phase0(True)
    lock, folds, labels = context()
    run_tests()
    from .inputs import prepare_inputs
    inputs = prepare_inputs(OUT, device='cuda')
    save(OUT / 'input_manifest.json', inputs, immutable=True)
    runtime(11)
    model, loading = load_frozen_labram('cuda')
    before = state_digest(model.state_dict())
    labram_records = []
    for rec in inputs['records']:
        sid = rec['subject_id']
        raw = np.load(checked(rec['raw']), allow_pickle=False)
        path = OUT / f'cache/labram/{sid}.npy'
        sidecar = path.with_suffix('.json')
        if sidecar.exists():
            old = read(sidecar)
            checked(old['mean'])
            expected_sidecar = {'subject_id': sid, 'raw': rec['raw'], 'asset_manifest': desc(LABRAM_MANIFEST),
                                'producer': desc(__file__), 'patch_token_shape': [60, 95, 200],
                                'pooling': 'mean of 19x5 patch tokens; exclude CLS',
                                'unit': 'native EEGLAB microvolt convention /100; physical calibration unconfirmed',
                                'channel_names': inputs['channel_names'], 'fit_subject_ids': [], 'frozen_eval': True}
            if any(old.get(k) != v for k, v in expected_sidecar.items()):
                raise ValueError('Stale LaBraM input cache')
            mean = np.load(path, allow_pickle=False)
        else:
            tokens = extract_patch_tokens(model, raw, inputs['channel_names'], batch_size=4)
            if tokens.shape != (60, 95, 200):
                raise ValueError('LaBraM must produce 19x5 patch tokens per window')
            mean = tokens.mean(axis=1, dtype=np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, mean, allow_pickle=False)
            save(sidecar, {'subject_id': sid, 'raw': rec['raw'], 'mean': desc(path),
                           'asset_manifest': desc(LABRAM_MANIFEST), 'producer': desc(__file__),
                           'patch_token_shape': [60, 95, 200], 'pooling': 'mean of 19x5 patch tokens; exclude CLS',
                           'unit': 'native EEGLAB microvolt convention /100; physical calibration unconfirmed',
                           'channel_names': inputs['channel_names'], 'fit_subject_ids': [], 'frozen_eval': True}, immutable=True)
        if mean.shape != (60, 200) or mean.dtype != np.float32 or not np.isfinite(mean).all():
            raise ValueError('Invalid LaBraM mean features')
        labram_records.append({'subject_id': sid, 'mean': desc(path), 'sidecar': desc(sidecar)})
        print(f'Prepared LaBraM {sid}: 60x200 frozen window representation', flush=True)
    if before != state_digest(model.state_dict()) or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('LaBraM encoder state changed')
    loading.update({'frozen_state_before': before, 'frozen_state_after': before,
                    'subjects_exported': 37, 'window_shape': [60, 200], 'state_unchanged': True})
    save(OUT / 'labram_loading_and_export.json', loading, immutable=True)
    save(OUT / 'labram_manifest.json', labram_records, immutable=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    adjacency = np.load(checked(inputs['adjacency']), allow_pickle=False)
    arrays = load_arrays(inputs, labram_records)
    preflight = []
    for condition in CONDITIONS:
        runtime(11)
        net = model_for(condition, adjacency).to('cuda')
        spec = optim_config(condition)
        if sum(p.numel() for p in net.parameters()) != spec['parameters']:
            raise ValueError('Registered head parameter budget differs')
        # Only the real outer train batch is used; no optimizer step or fitting.
        train_ids = folds['fold_1']['train'][:spec['batch_size']]
        idx = [lock['subject_ids'].index(s) for s in train_ids]
        xb = arrays[input_key(condition)][idx]
        yb = torch.tensor([labels[s] for s in train_ids], device='cuda')
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        net.train()
        loss = torch.nn.functional.cross_entropy(net(xb), yb)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise ValueError('Stage1 real train-input gradient probe failed')
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        net.eval()
        state = state_digest(net.state_dict())
        with torch.inference_mode():
            first = net(xb[:1])
            repeat = net(xb[:1])
        if not torch.equal(first, repeat) or state != state_digest(net.state_dict()):
            raise ValueError('Head evaluation changes outputs or state')
        preflight.append({'condition': condition, 'train_batch_subject_ids': train_ids,
                          'finite_gradients_all_parameters': True, 'optimizer_updates': 0,
                          'batch_forward_backward_seconds': elapsed,
                          'estimated_formal_fit_seconds': elapsed * spec['updates_per_model'],
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'parameters': spec['parameters'], 'repeat_input_equal': True, 'eval_state_unchanged': True})
        del net, xb, yb
    save(OUT / 'preflight.json', {'passed': True, 'checks': preflight,
                                  'test_or_val_data_used': False, 'optimizer_updates': 0}, immutable=True)
    sources = {}
    for source in inputs['sources']:
        sources[source['path']] = source['sha256']
    for p in [ANNEX, WORK / 'output/plans/TriDep_EEG与融合改进及语音文本泛化汇总计划_v1_20261004.md',
              STAGE0 / 'status.json', STAGE0 / 'source_manifest.json', STAGE0 / 'output_manifest.json',
              STAGE0 / 'independent_verification.json', STAGE0 / 'control_predictions.jsonl',
              STAGE0 / 'branch_probabilities.jsonl', STAGE0 / 'metrics.json',
              LABRAM_MANIFEST, WORK / 'S00/upstream-dev/lib/models/models.py',
              WORK / 'tridep_ccml/stage0.py', WORK / 'tridep_ccml/stage0_verify.py',
              WORK / 'S00/scripts/metrics.py', WORK / 'S00/scripts/smoke_models.py']:
        sources[desc(p)['path']] = digest(p)
    for p in CODE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.part',):
            sources[desc(p)['path']] = digest(p)
    for root in [OUT / 'cache', OUT]:
        candidates = root.rglob('*') if root.name == 'cache' else root.glob('*.json')
        for p in candidates:
            if p.is_file() and p.name != 'registration.json':
                sources[desc(p)['path']] = digest(p)
    registration = {'experiment': OUT.name, 'stage': 1, 'registered_at': now(), 'sources': sources,
                    'cohort_id': lock['cohort_id'], 'subject_ids': lock['subject_ids'], 'folds_sha256': lock['folds_sha256'],
                    'seeds': SEEDS, 'folds': [1, 2, 3, 4, 5], 'conditions': CONDITIONS,
                    'training': {c: optim_config(c) for c in CONDITIONS}, 'formal_fits': 90, 'formal_updates': 18000,
                    'inner_fits': 0, 'calibrator_fits': 0, 'threshold_searches': 0, 'early_stopping': False,
                    'head_selection': 'none; final epoch60', 'test_release': 'only after all90 fit checkpoints completed',
                    'fusion': {'method': 'original weighted probability', 'grid_step': .05, 'candidates': 231,
                               'selection_split': 'original3-person val', 'selections': 90, 'candidate_scores': 20790,
                               'tie_break': 'ascending EEG then speech', 'threshold': .5, 'tie_class': 0},
                    'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cuda',
                                'gpu': torch.cuda.get_device_name(0), 'float': 'float32', 'matmul_tf32': False,
                                'cudnn_tf32': True, 'deterministic': True, 'AMP': False},
                    'prior_stage': desc(STAGE0 / 'status.json'), 'appendix': desc(ANNEX),
                    'input_version': 'E2v2_corrected_channels_harmonic_guarded',
                    'fixed_ST': 'phase0 original same-seed/fold branch probabilities',
                    'physical_calibration_unconfirmed': True, 'new_independent_subjects': 0}
    save(OUT / 'registration.json', registration, immutable=True)
    save(OUT / 'registration_anchor.json', {'registration': desc(OUT / 'registration.json')}, immutable=True)
    save(REPORT / 'status.json', {'status': 'registered_ready_for_training', 'stage': 1,
                                'registration': desc(OUT / 'registration.json'), 'fits_completed': 0,
                                'planned_fits': 90, 'planned_updates': 18000, 'metrics_released': False})
    print('Stage1 fully registered: 6 conditions,90 fits,18000 updates; no external metrics released', flush=True)


def input_key(condition):
    return 'labram_mean' if condition.startswith('LaBraM') else 'cb_mean' if condition.endswith('originalCNN') else 'tokens'


def load_arrays(inputs=None, labram=None):
    inputs = read(OUT / 'input_manifest.json') if inputs is None else inputs
    labram = read(OUT / 'labram_manifest.json') if labram is None else labram
    arrays = {'tokens': np.stack([np.load(checked(r['tokens']), allow_pickle=False) for r in inputs['records']]),
              'cb_mean': np.stack([np.load(checked(r['mean']), allow_pickle=False) for r in inputs['records']]),
              'labram_mean': np.stack([np.load(checked(r['mean']), allow_pickle=False) for r in labram])}
    return {k: torch.as_tensor(v, dtype=torch.float32, device='cuda') for k, v in arrays.items()}


def verify_registration():
    checked(read(OUT / 'registration_anchor.json')['registration'])
    cfg = read(OUT / 'registration.json')
    for path, expected in cfg['sources'].items():
        if digest(WORK / path) != expected:
            raise ValueError('Stage1 registered source changed: ' + path)
    if (cfg['conditions'] != CONDITIONS or cfg['seeds'] != SEEDS or cfg['formal_fits'] != 90
            or cfg['formal_updates'] != 18000 or cfg['training'] != {c: optim_config(c) for c in CONDITIONS}
            or cfg['inner_fits'] != 0 or cfg['calibrator_fits'] != 0 or cfg['threshold_searches'] != 0
            or cfg['fusion']['candidates'] != 231 or cfg['fusion']['selections'] != 90
            or cfg['fusion']['candidate_scores'] != 20790):
        raise ValueError('Stage1 condition/budget drift')
    return cfg


def fit_all():
    cfg = verify_registration()
    lock, folds, labels = context()
    arrays = load_arrays()
    adjacency = np.load(checked(read(OUT / 'input_manifest.json')['adjacency']), allow_pickle=False)
    completed = []
    for seed in SEEDS:
        for fold in range(1, 6):
            split = folds[f'fold_{fold}']
            pairing = {}
            for condition in CONDITIONS:
                folder = OUT / f'runs/{condition}/seed_{seed}/fold_{fold}'
                if (folder / 'complete.json').exists():
                    complete = read(folder / 'complete.json')
                    for d in complete['artifacts']:
                        checked(d)
                    meta = read(folder / 'metadata.json')
                    expected_meta = {'condition': condition, 'seed': seed, 'fold': fold, 'train_ids': split['train'],
                                     'val_ids': split['val'], 'test_ids': split['test'], 'training': cfg['training'][condition],
                                     'registration': desc(OUT / 'registration.json'), 'folds_sha256': lock['folds_sha256'],
                                     'updates': cfg['training'][condition]['updates_per_model'],
                                     'no_val_test_forward_during_training': True}
                    if any(meta.get(k) != v for k, v in expected_meta.items()):
                        raise ValueError('Completed fit metadata differs from the frozen registration')
                    completed.append(meta)
                    print(f'Reusing completed fit {condition} seed={seed} fold={fold}', flush=True)
                else:
                    if folder.exists():
                        raise ValueError('Partial fit exists; preserve it and resolve before resuming: ' + str(folder))
                    runtime(seed)
                    model = model_for(condition, adjacency).to('cuda')
                    spec = cfg['training'][condition]
                    initial = state_digest(model.state_dict())
                    idx = [lock['subject_ids'].index(s) for s in split['train']]
                    dataset = torch.utils.data.TensorDataset(torch.tensor(idx, dtype=torch.long), torch.tensor([labels[s] for s in split['train']]))
                    loader = torch.utils.data.DataLoader(dataset, batch_size=spec['batch_size'], shuffle=True, num_workers=0)
                    opt = torch.optim.Adamax(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
                    order, history = [], []
                    updates = 0
                    start = time.perf_counter()
                    for epoch in range(60):
                        model.train()
                        total, count = 0., 0
                        for ii, yy in loader:
                            order.extend(ii.tolist())
                            xb = arrays[input_key(condition)][ii.to('cuda')]
                            opt.zero_grad(set_to_none=True)
                            loss = torch.nn.functional.cross_entropy(model(xb), yy.to('cuda'))
                            if not torch.isfinite(loss):
                                raise ValueError('Nonfinite CE loss')
                            loss.backward()
                            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                                raise ValueError('Nonfinite or disconnected head gradient')
                            if spec['gradient_clip'] is not None:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'], error_if_nonfinite=True)
                            opt.step()
                            total += float(loss.detach().cpu()) * len(ii)
                            count += len(ii)
                            updates += 1
                        history.append({'epoch': epoch + 1, 'train_ce': total / count})
                        if (epoch + 1) % 20 == 0:
                            print(f'TRAIN {condition} seed={seed} fold={fold} epoch={epoch+1}/60 loss={history[-1]["train_ce"]:.6f}', flush=True)
                    if updates != spec['updates_per_model']:
                        raise ValueError('Formal update budget differs')
                    model.eval()
                    metadata = {'condition': condition, 'seed': seed, 'fold': fold, 'train_ids': split['train'],
                                'val_ids': split['val'], 'test_ids': split['test'], 'training': spec,
                                'cohort_id': lock['cohort_id'], 'folds_sha256': lock['folds_sha256'],
                                'registration': desc(OUT / 'registration.json'), 'initial_state_sha256': initial,
                                'final_state_sha256': state_digest(model.state_dict()),
                                'batch_order_sha256': hashlib.sha256(np.array(order, dtype='<i8').tobytes()).hexdigest(),
                                'updates': updates, 'seconds': time.perf_counter() - start,
                                'fit_subjects_unique': len(set(order)), 'input_key': input_key(condition),
                                'no_val_test_forward_during_training': True}
                    folder.mkdir(parents=True)
                    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    torch.save({'state_dict': cpu_state, 'metadata': metadata}, folder / 'checkpoint.pt')
                    save(folder / 'metadata.json', metadata, immutable=True)
                    save(folder / 'train_history.json', history, immutable=True)
                    save(folder / 'complete.json', {'artifacts': [desc(folder / p) for p in ('checkpoint.pt', 'metadata.json', 'train_history.json')]}, immutable=True)
                    completed.append(metadata)
                    del model, opt
                    print(f'FIT COMPLETE {condition} seed={seed} fold={fold} updates={updates} seconds={metadata["seconds"]:.1f}', flush=True)
                meta = read(folder / 'metadata.json')
                group = 'cnn' if condition.endswith('originalCNN') else 'token'
                current = (meta['initial_state_sha256'], meta['batch_order_sha256'])
                if group in pairing and pairing[group] != current:
                    raise ValueError('Matched conditions differ in initialization or batch order')
                pairing[group] = current
                save(OUT / 'training_ledger.json', {'fits_completed': len(completed), 'updates_completed': sum(m['updates'] for m in completed),
                                                  'planned_fits': 90, 'planned_updates': 18000, 'fits': completed,
                                                  'metrics_released': False})
                save(REPORT / 'status.json', {'status': 'training' if len(completed) < 90 else 'all_fits_complete_pending_evaluation',
                                            'stage': 1, 'fits_completed': len(completed),
                                            'updates_completed': sum(m['updates'] for m in completed),
                                            'planned_fits': 90, 'planned_updates': 18000, 'metrics_released': False})
    if len(completed) != 90 or sum(m['updates'] for m in completed) != 18000:
        raise ValueError('All formal conditions must complete before evaluation')
    verify_registration()
    print('All90 fits complete; uniform evaluation is now permitted', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else fit_all()
