"""Paired TC-04 heads on corrected clean versus augmented TRI-DEP EEG parents.

Only this experiment's namespace is writable. Parent encoders are frozen; both
conditions export clean inputs only. --execute is required for formal training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .common import WORK, OUT, BASE_EEG, load_views
from tridep_ccml.v2.common import descriptor, verify_descriptor, write_new, set_seed
from tridep_ccml.records import RecordContext, MODALITIES, read_json, require, check_fusion
from tridep_ccml.train import make_initial_state, state_digest, load_features, validate_config
from tridep_ccml.ccml import ThreeViewHeads, ccml_loss, dirichlet_from_evidence, evidence_dc
from tridep_ccml.metrics import binary_metrics, summarize
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(WORK / 'S00/upstream-dev'))
from lib.models import models

TC = WORK / 'tridep_ccml'
CLEAN = TC / 'experiments/TC-00-E-37-fulltext'
REFERENCE = 'TC-04-37-fulltext'
CONDITIONS = ('tc_clean', 'tc_aug')
LOSS = {'rho': 1.25, 'delta': 1., 'gamma': 1., 'anneal_epochs': 50}
CONFIG_PATH = OUT / 'tc_configuration.json'


def configuration(context, export_device):
    validate_config(context)
    return {
        'family': OUT.name, 'conditions': list(CONDITIONS), 'reference': REFERENCE,
        'cohort_id': context.lock['cohort_id'], 'cohort_sha256': context.cohort_sha256,
        'folds_sha256': context.folds_sha256, 'subject_ids': context.ids,
        'seeds': [11, 22, 33], 'folds': [1, 2, 3, 4, 5],
        'modalities': list(MODALITIES), 'feature_shape': [37, 3, 128],
        'epochs': 60, 'batch_size': 8, 'optimizer': 'Adamax', 'learning_rate': .001,
        'weight_decay': 0., 'gradient_clip': 1., 'loss': LOSS,
        'trainable_parameters': 774, 'threshold': .5, 'tie_class': 0,
        'selection': 'fixed_last_epoch', 'head_device': 'cpu', 'export_device': str(export_device),
        'class_weights': None, 'resampling': False, 'validation_or_test_selection': False,
        'encoder_policy': 'same_fold_same_seed_frozen_eval',
        'inference_views': 'clean_only_for_all_37_people',
        'augmentation_scope': 'EEG_parent_training_only; no_augmentation_in_TC_head_training',
        'comparison': 'tc_aug_minus_tc_clean isolates parent EEG augmentation with corrected channels in both',
        'historical_TC04': 'reference_only; historical EEG electrode mapping differed',
        'source_descriptors': {str(p.relative_to(WORK)).replace('\\', '/'): descriptor(p)
            for p in [Path(__file__), Path(__file__).with_name('common.py'),
                      TC / 'train.py', TC / 'ccml.py', TC / 'records.py', TC / 'metrics.py',
                      TC / 'v2/common.py', WORK / 'S00/upstream-dev/lib/models/models.py',
                      WORK / 'S00/configs/b0_fulltext.json']},
        'augmentation_manifest': descriptor(OUT / 'feature_manifest.json'),
        'corrected_eeg_manifest': descriptor(BASE_EEG / 'eeg_manifest.json'),
        'historical_head_configuration': descriptor(TC / 'configs/experiment.json'),
    }


def parent_path(condition, seed, fold):
    require(condition in CONDITIONS, 'Unknown TC condition')
    base = CLEAN / 'runs' if condition == 'tc_clean' else OUT / 'tridep/runs'
    return base / f'seed_{seed}/fold_{fold}/checkpoint.pt'


def checked_eeg_parent(context, condition, seed, fold):
    path = parent_path(condition, seed, fold)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    require(set(saved) >= {'state_dict', 'metadata'}, 'Unexpected EEG checkpoint format')
    meta = saved['metadata']
    for key, expected in {
        'seed': seed, 'fold': fold, 'train_ids': context.folds[f'fold_{fold}']['train'],
        'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
    }.items():
        require(meta.get(key) == expected, f'EEG parent provenance mismatch: {key}')
    require(meta.get('epochs', meta.get('completed_epochs')) == 60, 'EEG parent must complete 60 epochs')
    sidecar = read_json(path.with_suffix('.json'))
    require(sidecar['checkpoint'] == descriptor(path) and sidecar['metadata'] == meta,
            'EEG parent checkpoint sidecar mismatch')
    complete = read_json(path.parent / 'complete.json')
    require(complete['checkpoint.pt'] == descriptor(path), 'EEG parent not complete or changed')
    parent_config = read_json(verify_descriptor(meta['configuration']))
    if condition == 'tc_clean':
        require(meta['experiment'] == 'TC-00-E-37-fulltext', 'Clean parent is not TC-00-E')
        require(set(meta['corrected_features']) == set(context.ids), 'Incomplete corrected input lineage')
        for sid in context.ids:
            require(meta['corrected_features'][sid] == descriptor(BASE_EEG / 'eeg' / f'{sid}.npy'),
                    'Clean parent was not trained with corrected inputs')
    else:
        require(meta['experiment'] == 'EEG-AUG-TRIDEP-37-fulltext', 'Unexpected augmented EEG parent')
        require(parent_config.get('augmented_feature_manifest') == descriptor(OUT / 'feature_manifest.json'),
                'Augmented parent must bind the current augmentation manifest')
        require(meta['evaluation_input'] == 'view0_only' and meta['actual_updates'] == 120,
                'Augmented parent training/inference protocol differs')
    return saved, path


def clean_inputs(context, views):
    require(set(views) == set(context.ids), 'Augmentation view cohort mismatch')
    manifest = {r['subject_id']: r for r in read_json(OUT / 'feature_manifest.json')}
    result, sources = {}, {}
    for sid in context.ids:
        source = manifest[sid]['clean_source']
        require(source == descriptor(BASE_EEG / 'eeg' / f'{sid}.npy'), 'Unexpected clean source')
        value = np.load(verify_descriptor(source), allow_pickle=False)
        require(np.array_equal(value, views[sid][0]), 'Clean inference view differs from corrected EEG')
        result[sid], sources[sid] = value, source
    return result, sources


def capture(model, arrays, ids, device):
    """Capture the exact 128-dimensional input to the frozen final classifier."""
    model.requires_grad_(False).to(device).eval()
    captured, representations, probabilities = [], [], []
    handle = model.classifier[-1].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach()))
    try:
        with torch.inference_mode():
            for sid in ids:
                logits = model(torch.as_tensor(arrays[sid], device=device).unsqueeze(0))
                require(len(captured) == 1, 'Expected exactly one classifier input per subject')
                hidden = captured.pop()
                require(hidden.shape == (1, 128) and torch.isfinite(hidden).all(), 'Invalid representation')
                # Replay only the last layer, without invoking the capture hook again.
                last = model.classifier[-1]
                replay = F.linear(hidden, last.weight, last.bias)
                require(torch.equal(replay, logits), 'Captured representation does not reconstruct parent logits')
                representations.append(hidden.cpu().numpy()[0].copy())
                probabilities.append(float(logits.softmax(-1)[0, 1].cpu()))
    finally:
        handle.remove()
    return np.stack(representations).astype(np.float32), probabilities


def export_features(context, condition, seed, fold, clean, clean_sources, device, persist=True):
    old = load_features(context, seed, fold)
    branches = read_json(WORK / 'S00/configs/b0_fulltext.json')['branches']
    saved_eeg, eeg_path = checked_eeg_parent(context, condition, seed, fold)
    old_rows = [json.loads(line) for line in
                (WORK / f'S00/runs/B0-37-fulltext/seed_{seed}/probabilities.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()]
    expected = {r['subject_id']: r for r in old_rows if r['outer_fold'] == fold}
    require(set(expected) == set(context.ids), 'Missing historical parent predictions')
    features = np.empty((37, 3, 128), dtype=np.float32)
    manifest = {
        'condition': condition, 'seed': seed, 'fold': fold, 'subject_ids': context.ids,
        'train_ids': context.folds[f'fold_{fold}']['train'], 'cohort_id': context.lock['cohort_id'],
        'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
        'modalities': list(MODALITIES), 'frozen_eval': True, 'inference_view': 'clean',
        'export_device': str(device), 'configuration': descriptor(CONFIG_PATH) if persist else None,
        'parent_checkpoints': {}, 'source_features': {}, 'checks': {},
    }
    for mi, modality in enumerate(MODALITIES):
        branch = branches[modality]
        if modality == 'eeg':
            saved, path, arrays, sources = saved_eeg, eeg_path, clean, clean_sources
        else:
            path = verify_descriptor(old['manifest']['parent_checkpoints'][modality])
            saved = torch.load(path, map_location='cpu', weights_only=True)
            for key, value in {'seed': seed, 'fold': fold,
                'train_ids': context.folds[f'fold_{fold}']['train'], 'configuration': branch,
                'cohort_id': context.lock['cohort_id'], 'folds_sha256': context.folds_sha256,
                'experiment': 'B0-37-fulltext' if modality == 'text' else 'B0-37'}.items():
                require(saved.get(key) == value, f'Historical {modality} checkpoint mismatch: {key}')
            sources = old['manifest']['source_features'][modality]
            arrays = {sid: np.load(verify_descriptor(sources[sid]), allow_pickle=False) for sid in context.ids}
        model = getattr(models, branch['model_class'])(**branch['model_kwargs'])
        model.load_state_dict(saved['state_dict'], strict=True)
        representation, probability = capture(model, arrays, context.ids, device)
        features[:, mi] = representation
        if modality != 'eeg':
            difference = float(np.max(np.abs(representation - old['features'][:, mi])))
            require(np.allclose(representation, old['features'][:, mi], atol=1e-6, rtol=1e-5),
                    f'Historical {modality} representation changed: {difference}')
            error = max(abs(p - expected[sid]['p_' + modality]) for sid, p in zip(context.ids, probability))
            require(error <= 1e-6, f'Historical {modality} parent probability changed: {error}')
            # Preserve bit-identical speech/text controls after verifying the fresh export.
            features[:, mi] = old['features'][:, mi]
            manifest['checks'][modality] = {'fresh_export_max_abs_error': difference,
                'parent_probability_max_abs_error': error, 'historical_cache_identical': True}
        else:
            prior = {r['subject_id']: r for r in read_json(eeg_path.parent / 'predictions.json')}
            error = max(abs(p - prior[sid]['p_eeg']) for sid, p in zip(context.ids, probability))
            require(error <= 1e-6, f'{condition} EEG probability changed: {error}')
            manifest['checks']['eeg'] = {'parent_probability_max_abs_error': error}
        manifest['parent_checkpoints'][modality] = descriptor(path)
        manifest['source_features'][modality] = sources
        del model
    require(features.shape == (37, 3, 128) and np.isfinite(features).all(), 'Bad exported features')
    data = {'features': features, 'subject_ids': context.ids, 'labels': old['labels'],
            'splits': old['splits'], 'manifest': manifest}
    if persist:
        path = OUT / f'features/{condition}/seed_{seed}/fold_{fold}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {'features': features, 'subject_ids': np.asarray(context.ids),
                  'labels': old['labels'], 'splits': np.asarray(old['splits'])}
        if path.exists():
            with np.load(path, allow_pickle=False) as previous:
                require(set(previous.files) == set(arrays) and
                        all(np.array_equal(previous[k], v) for k, v in arrays.items()), 'Existing feature export differs')
        else:
            with path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
        manifest['feature_bundle'] = descriptor(path)
        write_new(path.with_suffix('.json'), manifest)
        data['manifest_descriptor'] = descriptor(path.with_suffix('.json'))
    return data


def rows_from_logits(context, data, logits, condition, seed, fold, checkpoint):
    evidence = F.softplus(logits)
    views, fused = dirichlet_from_evidence(evidence), evidence_dc(evidence, LOSS['rho'])
    rows = []
    for i, sid in enumerate(context.ids):
        row = {
            'experiment': condition, 'reference_experiment': REFERENCE,
            'cohort_id': context.lock['cohort_id'], 'cohort_sha256': context.cohort_sha256,
            'folds_sha256': context.folds_sha256, 'seed': seed, 'outer_fold': fold,
            'subject_id': sid, 'split': data['splits'][i], 'label': int(data['labels'][i]),
            'train_ids': context.folds[f'fold_{fold}']['train'],
            'checkpoint': descriptor(checkpoint), 'configuration': descriptor(CONFIG_PATH),
            'feature_manifest': data['manifest_descriptor'],
            'parent_checkpoints': data['manifest']['parent_checkpoints'],
            'source_features': {m: data['manifest']['source_features'][m][sid] for m in MODALITIES},
            'inference_view': 'clean', 'fusion_method': 'ccml', 'fusion_weights': None,
            'rho': 1.25, 'delta': 1., 'gamma': 1.,
            'views': {m: {'e': evidence[i, v].tolist(), 'alpha': views['alpha'][i, v].tolist(),
                          'p': views['p'][i, v].tolist(), 'u': float(views['u'][i, v])}
                      for v, m in enumerate(MODALITIES)},
            **{k: fused[k][i].tolist() for k in ('m', 'e_con', 'e_cmp', 'e_final')},
            'p_final': fused['p'][i].tolist(), 'u_final': float(fused['u'][i]),
            'prediction': int(fused['p'][i, 1] > .5),
        }
        require(sid in context.folds[f'fold_{fold}'][row['split']] and
                row['label'] == context.labels[sid], 'Prediction identity mismatch')
        # Reuse the independent arithmetic verifier with the canonical TC-04
        # equation contract, without changing the historical experiment registry.
        check_fusion({**row, 'experiment': REFERENCE})
        rows.append(row)
    require(len(rows) == len({r['subject_id'] for r in rows}) == 37, 'Not one prediction per person')
    return rows


def fit(context, data, condition, seed, fold):
    folder = OUT / f'runs/{condition}/seed_{seed}/fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoint.pt'
    initial = make_initial_state(seed)
    model = ThreeViewHeads()
    model.load_state_dict(initial, strict=True)
    require(sum(p.numel() for p in model.parameters()) == 774, 'Wrong head size')
    training_ids = context.folds[f'fold_{fold}']['train']
    selected = [context.ids.index(sid) for sid in training_ids]
    require(not set(training_ids) & set(context.folds[f'fold_{fold}']['val'] +
                                       context.folds[f'fold_{fold}']['test']), 'Training leakage')
    meta = {'experiment': condition, 'reference_experiment': REFERENCE, 'seed': seed, 'fold': fold,
        'train_ids': training_ids, 'cohort_sha256': context.cohort_sha256,
        'folds_sha256': context.folds_sha256, 'configuration': descriptor(CONFIG_PATH),
        'feature_manifest': data['manifest_descriptor'], 'initialization_sha256': state_digest(initial),
        'parent_checkpoints': data['manifest']['parent_checkpoints'],
        'effective_loss_parameters': LOSS, 'completed_epochs': 60, 'selection': 'fixed_last_epoch',
        'trainable_parameters': 774, 'trainable_eeg_parameters': 0, 'device': 'cpu',
        'torch': str(torch.__version__), 'upstream_mode': 'frozen_eval_clean_cached_features'}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(all(saved['metadata'].get(k) == v for k, v in meta.items()), 'Existing TC checkpoint differs')
        require(read_json(folder / 'checkpoint.json') == {'checkpoint': descriptor(checkpoint),
                'metadata': saved['metadata']}, 'Existing TC sidecar differs')
        require((folder / 'train_history.json').is_file(), 'Existing TC run lacks training history')
        meta = saved['metadata']
        model.load_state_dict(saved['state_dict'], strict=True)
    else:
        require(not (folder / 'train_history.json').exists(), 'Incomplete TC run needs inspection')
        model.train()
        optimizer = torch.optim.Adamax(model.parameters(), lr=.001, weight_decay=0.)
        x, y = torch.from_numpy(data['features'][selected]), torch.from_numpy(data['labels'][selected])
        generator = torch.Generator(device='cpu').manual_seed(seed)
        batch_hash, history, updates = hashlib.sha256(), [], 0
        for epoch in range(1, 61):
            order = torch.randperm(len(selected), generator=generator)
            batch_hash.update(order.numpy().astype('<i8').tobytes())
            total = 0.
            for start in range(0, len(order), 8):
                batch = order[start:start + 8]
                optimizer.zero_grad(set_to_none=True)
                loss = ccml_loss(F.softplus(model(x[batch])), y[batch], epoch, **LOSS)
                require(torch.isfinite(loss).item(), 'Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameters')
                total += float(loss.detach()) * len(batch)
                updates += 1
            history.append({'epoch': epoch, 'train_loss': total / len(selected)})
        require(updates == 240, 'Unexpected optimizer budget')
        historical = read_json(TC / f'runs/{REFERENCE}/seed_{seed}/fold_{fold}/checkpoint.json')['metadata']
        require(meta['initialization_sha256'] == historical['initialization_sha256'] and
                batch_hash.hexdigest() == historical['batch_order_sha256'], 'TC-04 pairing mismatch')
        meta.update(batch_order_sha256=batch_hash.hexdigest(), optimizer_updates=updates)
        with checkpoint.open('xb') as stream:
            torch.save({'state_dict': model.state_dict(), 'metadata': meta}, stream)
        write_new(folder / 'checkpoint.json', {'checkpoint': descriptor(checkpoint), 'metadata': meta})
        write_new(folder / 'train_history.json', history)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(data['features']))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(state_digest(saved['state_dict']) == state_digest(model.state_dict()), 'Checkpoint reload differs')
    reloaded = ThreeViewHeads()
    reloaded.load_state_dict(saved['state_dict'], strict=True)
    reloaded.eval()
    with torch.inference_mode():
        replay = reloaded(torch.from_numpy(data['features']))
    require(torch.equal(logits, replay), 'Reload prediction differs')
    rows = rows_from_logits(context, data, replay, condition, seed, fold, checkpoint)
    write_new(folder / 'predictions.json', rows)
    write_new(folder / 'reload_check.json', {'passed': True, 'exact_state': True,
              'exact_logits': True, 'subjects': 37, 'evidence_arithmetic_verified': True})
    test = [r for r in rows if r['split'] == 'test']
    write_new(folder / 'metrics.json', {
        'fusion': binary_metrics([r['label'] for r in test], [r['p_final'][1] for r in test]),
        **{m: binary_metrics([r['label'] for r in test], [r['views'][m]['p'][1] for r in test]) for m in MODALITIES}})
    write_new(folder / 'complete.json', {name: descriptor(folder / name) for name in
        ('checkpoint.pt', 'checkpoint.json', 'train_history.json', 'predictions.json', 'reload_check.json', 'metrics.json')})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', action='append', choices=CONDITIONS)
    parser.add_argument('--seed', action='append', type=int, choices=[11, 22, 33])
    parser.add_argument('--fold', action='append', type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--export-device', default='cuda')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    context = RecordContext()
    conditions = [c for c in CONDITIONS if c in (args.condition or CONDITIONS)]
    seeds, folds = sorted(set(args.seed or [11, 22, 33])), sorted(set(args.fold or [1, 2, 3, 4, 5]))
    device = torch.device(args.export_device)
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable for parent export')
    cfg = configuration(context, device)
    views = load_views()
    clean, clean_sources = clean_inputs(context, views)
    # Complete all dependency checks before starting any requested head training.
    for condition in conditions:
        for seed in seeds:
            for fold in folds:
                checked_eeg_parent(context, condition, seed, fold)
                load_features(context, seed, fold)
    jobs = len(conditions) * len(seeds) * len(folds)
    print(json.dumps({'passed': True, 'mode': 'execute' if args.execute else 'dry_run',
        'jobs': jobs, 'optimizer_updates': jobs * 240, 'conditions': conditions,
        'training_performed': False}), flush=True)
    if not args.execute:
        return
    write_new(CONFIG_PATH, cfg)
    set_seed(11)
    # Match the original frozen encoder inference backend; heads remain CPU.
    torch.backends.cudnn.allow_tf32 = True
    for condition in conditions:
        rows = []
        for seed in seeds:
            for fold in folds:
                data = export_features(context, condition, seed, fold, clean, clean_sources, device)
                rows.extend(fit(context, data, condition, seed, fold))
                print(json.dumps({'condition': condition, 'seed': seed, 'fold': fold, 'status': 'complete'}), flush=True)
        if seeds == [11, 22, 33] and folds == [1, 2, 3, 4, 5]:
            for seed in seeds:
                test_ids = [r['subject_id'] for r in rows if r['seed'] == seed and r['split'] == 'test']
                require(len(test_ids) == 37 and set(test_ids) == set(context.ids), 'Incomplete or duplicate subject OOF')
            write_new(OUT / f'tc_{condition.removeprefix("tc_")}_predictions_all.json', rows)
            report = summarize(rows)
            report['branch_summaries'] = {m: summarize([{**r, 'p_final': r['views'][m]['p'],
                'prediction': int(r['views'][m]['p'][1] > .5), 'u_final': r['views'][m]['u']} for r in rows])
                for m in MODALITIES}
            write_new(OUT / f'{condition}_summary.json', report)


if __name__ == '__main__':
    main()
