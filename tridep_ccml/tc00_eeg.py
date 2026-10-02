"""Original TRI-DEP training with only label-corrected v2-E EEG inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from .v2.common import (WORK, TC, RecordContext, descriptor, verify_descriptor,
                       read_json, write_new, require, digest)
from .train import state_digest
import numpy as np
import torch

S00 = WORK/'S00'
OUT = TC/'experiments/TC-00-E-37-fulltext'
EEG_PARENT = TC/'experiments/CCML-v2-E-37-fulltext'
LATEST = TC/'experiments/CCML-v2-AS-NP-R-37-fulltext'
EXPERIMENT = OUT.name
CONFIG = OUT/'configuration.json'
PLAN = WORK/'output/plans/TC00_EEG修订_原始TRIDEP固定对照计划.md'
sys.path.insert(0, str(S00/'scripts'))
sys.path.insert(0, str(S00/'upstream-dev'))
from lib.models import models
from smoke_models import seed as original_seed
from metrics import select_weights, metrics as original_metrics


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def runtime(value):
    require(torch.cuda.is_available(), 'Original TRI-DEP comparator used CUDA')
    original_seed(value)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def configuration(context):
    old = read_json(S00/'configs/b0_fulltext.json')
    return {'experiment': EXPERIMENT, 'date': '2026-09-23', 'version': 1,
            'plan': descriptor(PLAN), 'cohort_sha256': context.cohort_sha256,
            'folds_sha256': context.folds_sha256, 'subject_ids': context.ids,
            'seeds': old['seeds'], 'folds': [1, 2, 3, 4, 5], 'eeg_branch': old['branches']['eeg'],
            'fusion': old['fusion'], 'threshold': .5, 'tie_class': 0,
            'original_configuration': descriptor(S00/'configs/b0_fulltext.json'),
            'corrected_features': descriptor(EEG_PARENT/'eeg_manifest.json'),
            'original_feature_manifest': descriptor(S00/'features/B0-37/features_manifest.json'),
            'fulltext_feature_manifest': descriptor(S00/'features/B0-37-fulltext/features_manifest.json'),
            'parent_probability_tolerance': 1e-6, 'new_eeg_runs': 15, 'new_updates': 1800,
            'old_eeg_training_replay_runs': 15, 'replay_updates': 1800,
            'training': 'original CE + Adamax; random initialization; train dropout; no gradient clipping',
            'device': 'cuda', 'matmul_tf32': False, 'cudnn_tf32': True,
            'explicit_user_scope': 'original TRI-DEP baseline with corrected EEG input; no CCML-fitted weights',
            'sources': {p.relative_to(WORK).as_posix(): descriptor(p) for p in [Path(__file__),
                TC/'complete_tc00_eeg.py', TC/'test_tc00_eeg.py', TC/'metrics.py', TC/'train.py',
                TC/'v2/common.py', S00/'scripts/run_b0.py', S00/'scripts/smoke_models.py',
                S00/'scripts/metrics.py', S00/'scripts/fulltext_b0.py',
                S00/'upstream-dev/lib/models/models.py', TC/'v2/extract_eeg.py']}}


def protect_history():
    path = OUT/'history_before.json'
    if not path.exists():
        tracked = read_json(LATEST/'history_before.json')
        folders = [LATEST, S00/'runs/B0-37', S00/'runs/B0-37-fulltext']
        files = [p for folder in folders for p in folder.rglob('*') if p.is_file()]
        files += [TC/'v2/as_np_r.py', TC/'v2/complete_as_np_r.py', TC/'v2/test_as_np_r.py',
                  WORK/'output/plans/CCML_v2_AS_NP_R_固定正则对照计划.md']
        files += [(TC/'experiments')/f'CCML-v2-AS-NP-R-{s}.log' for s in ['preflight', 'train', 'completion']]
        for p in files:
            tracked[p.relative_to(WORK).as_posix()] = digest(p)
        write_new(path, tracked)
    tracked = read_json(path)
    for name, value in tracked.items():
        require(digest(WORK/name) == value, f'Historical artifact changed: {name}')
    return len(tracked)


def setup():
    context = RecordContext()
    OUT.mkdir(parents=True, exist_ok=True)
    protect_history()
    tests = read_json(OUT/'test_results.json')
    require(tests['passed'] and tests['failures'] == tests['errors'] == 0, 'Tests incomplete')
    for d in tests['sources'].values():
        verify_descriptor(d)
    cfg = configuration(context)
    write_new(CONFIG, cfg)
    return context, cfg


def load_inputs(context):
    revised = read_json(EEG_PARENT/'eeg_manifest.json')
    require([r['provenance']['subject_id'] for r in revised] == context.ids, 'Corrected cohort mismatch')
    revision = read_json(verify_descriptor(revised[0]['provenance']['revision']))
    corrected, corrected_source, verified = {}, {}, set()
    for r in revised:
        p = r['provenance']
        sid = p['subject_id']
        require(p['fit_subject_ids'] == [] and p['cohort_sha256'] == context.cohort_sha256 and
                p['folds_sha256'] == context.folds_sha256, 'Corrected EEG fitted/cohort mismatch')
        require(p['native_scale'] and not p['normalization_added'] and not p['reference_changed'], 'Unexpected EEG transform')
        require(p['selection_by_label'] == [e['required_source_label'] for e in revision['selection']] and
                p['resolved_indices'] == [e['expected_zero_based_index'] for e in revision['selection']], 'Electrode mapping changed')
        for k in ['producer', 'common_source', 'backbone', 'revision']:
            identity = (p[k]['path'], p[k]['sha256'])
            if identity not in verified:
                verify_descriptor(p[k])
                verified.add(identity)
        x = np.load(verify_descriptor(r['feature']), allow_pickle=False)
        require(x.shape == (60, 200) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid corrected input')
        corrected[sid], corrected_source[sid] = x, r['feature']
    old_manifest = read_json(S00/'features/B0-37/features_manifest.json')
    full_manifest = read_json(S00/'features/B0-37-fulltext/features_manifest.json')
    original, source = {}, {}
    for row in old_manifest+full_manifest:
        if row['modality'] == 'text' and row not in full_manifest:
            continue
        root = S00/'features'/('B0-37-fulltext' if row['modality'] == 'text' else 'B0-37')
        path = root/row['path']
        require(row['fit_subject_ids'] == [] and digest(path) == row['sha256'], 'Parent feature changed/fitted')
        require(row['cohort_id'] == context.lock['cohort_id'] and row['folds_sha256'] == context.folds_sha256, 'Parent feature cohort mismatch')
        key = (row['modality'], row['seed'], row['subject_id'])
        require(key not in original, 'Duplicate feature')
        original[key] = np.load(path, allow_pickle=False)
        source[key] = descriptor(path)
    for sid in context.ids:
        require(not np.array_equal(original['eeg', None, sid], corrected[sid]), 'Corrected EEG equals old input')
    return {'corrected': corrected, 'corrected_source': corrected_source, 'original': original, 'source': source}


def parent_bundle(context, value, fold):
    base = S00/f'runs/B0-37/seed_{value}'
    full = S00/f'runs/B0-37-fulltext/seed_{value}'
    base_run, full_run = read_json(base/'run.json'), read_json(full/'run.json')
    require(base_run['training_script_sha256'] == digest(S00/'scripts/run_b0.py') and
            full_run['training_script_sha256'] == digest(S00/'scripts/fulltext_b0.py'), 'Original training code drift')
    require(full_run['parent_run_sha256'] == digest(base/'run.json') and
            full_run['parent_predictions_sha256'] == digest(base/'probabilities.jsonl'), 'Fulltext ancestry mismatch')
    all_rows = read_rows(full/'probabilities.jsonl')
    selected = {r['subject_id']: r for r in all_rows if r['outer_fold'] == fold}
    require(set(selected) == set(context.ids), 'Parent coverage mismatch')
    rows = [selected[s] for s in context.ids]
    checkpoints = {}
    cfg = read_json(S00/'configs/b0_fulltext.json')
    for modality in ['eeg', 'speech', 'text']:
        path = (full if modality == 'text' else base)/f'fold_{fold}_{modality}.pt'
        expected = (full_run['folds'][f'fold_{fold}']['text_checkpoint_sha256'] if modality == 'text' else
                    base_run['folds'][f'fold_{fold}']['branches'][modality]['checkpoint_sha256'])
        require(digest(path) == expected, 'Parent checkpoint hash mismatch')
        checkpoints[modality] = descriptor(path)
    for r in rows:
        require(r['seed'] == value and r['label'] == context.labels[r['subject_id']] and
                r['subject_id'] in context.folds[f'fold_{fold}'][r['split']], 'Parent row identity mismatch')
    return rows, checkpoints, base_run


class TrackingDataset(torch.utils.data.TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.seen = []

    def __getitem__(self, index):
        self.seen.append(int(index))
        return super().__getitem__(index)


def fit_eeg(context, cfg, value, fold, features):
    runtime(value)
    c = cfg['eeg_branch']
    model = getattr(models, c['model_class'])(**c['model_kwargs']).to('cuda')
    initial = state_digest(model.state_dict())
    require(sum(p.numel() for p in model.parameters()) == 498434, 'Original EEG architecture changed')
    opt = torch.optim.Adamax(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    ids = context.folds[f'fold_{fold}']['train']
    x = torch.tensor(np.stack([features[s] for s in ids]), dtype=torch.float32)
    y = torch.tensor([context.labels[s] for s in ids], dtype=torch.long)
    dataset = TrackingDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=c['batch_size'], shuffle=True, num_workers=0)
    curve, updates = [], 0
    for epoch in range(c['epochs']):
        model.train()
        loss_total, sample_count = 0., 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xb.to('cuda')), yb.to('cuda'))
            require(bool(torch.isfinite(loss)), 'Nonfinite original CE loss')
            loss.backward()
            opt.step()
            updates += 1
            loss_total += float(loss.detach().cpu())*len(yb)
            sample_count += len(yb)
        curve.append({'epoch': epoch+1, 'train_cross_entropy': loss_total/sample_count})
    model.eval()
    require(updates == 120 and len(dataset.seen) == 60*len(ids), 'Original budget changed')
    meta = {'seed': value, 'fold': fold, 'train_ids': ids, 'initial_state_sha256': initial,
            'batch_order_sha256': hashlib.sha256(np.array(dataset.seen, dtype='<i8').tobytes()).hexdigest(),
            'final_cpu_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'final_cuda_rng_sha256': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
            'updates': updates, 'epochs': c['epochs'], 'trainable_parameters': 498434}
    return model, curve, meta


def probabilities(model, features, ids):
    model.eval()
    with torch.no_grad():
        return {s: float(torch.softmax(model(torch.tensor(features[s], dtype=torch.float32, device='cuda').unsqueeze(0)), dim=-1)[0, 1].cpu()) for s in ids}


def fuse(parent_rows, eeg_probabilities):
    rows = copy.deepcopy(parent_rows)
    for r in rows:
        r['p_eeg'] = eeg_probabilities[r['subject_id']]
    selected = select_weights([r for r in rows if r['split'] == 'val'])
    for r in rows:
        r.update(experiment=EXPERIMENT, fusion_weights=selected['weights'])
        r['p_final'] = sum(w*r['p_'+m] for w, m in zip(selected['weights'], ['eeg', 'speech', 'text']))
        r['p_mean'] = sum((1/3)*r['p_'+m] for m in ['eeg', 'speech', 'text'])
        r['prediction'] = int(r['p_final'] > .5)
        r['mean_prediction'] = int(r['p_mean'] > .5)
        # These fields referred to the old parent EEG; source binding is provided separately below.
        r.pop('parent_experiment', None)
    return rows, selected


def preflight():
    context, cfg = setup()
    data = load_inputs(context)
    checks = []
    original_cfg = read_json(S00/'configs/b0_fulltext.json')
    for value in cfg['seeds']:
        for fold in cfg['folds']:
            runtime(value)
            rows, checkpoints, old_run = parent_bundle(context, value, fold)
            errors = {}
            for modality, cp in checkpoints.items():
                saved = torch.load(verify_descriptor(cp), map_location='cpu', weights_only=True)
                require(saved['seed'] == value and saved['fold'] == fold and
                        saved['train_ids'] == context.folds[f'fold_{fold}']['train'] and
                        saved['configuration'] == original_cfg['branches'][modality] and
                        saved['folds_sha256'] == context.folds_sha256, 'Parent fitting provenance mismatch')
                c = original_cfg['branches'][modality]
                model = getattr(models, c['model_class'])(**c['model_kwargs']).to('cuda')
                model.load_state_dict(saved['state_dict'], strict=True)
                source_seed = value if modality == 'speech' else None
                features = {s: data['original'][modality, source_seed, s] for s in context.ids}
                p = probabilities(model, features, context.ids)
                error = max(abs(p[r['subject_id']]-r['p_'+modality]) for r in rows)
                require(error <= cfg['parent_probability_tolerance'], 'Original branch probabilities not reproduced')
                errors[modality] = error
                del model, saved
            repeated, selected = fuse(rows, {r['subject_id']: r['p_eeg'] for r in rows})
            require(all(a['fusion_weights'] == b['fusion_weights'] and a['p_final'] == b['p_final']
                        for a, b in zip(rows, repeated)), 'Original val-weight fusion not reproduced')
            original_eeg = {s: data['original']['eeg', None, s] for s in context.ids}
            model, curve, meta = fit_eeg(context, cfg, value, fold, original_eeg)
            saved = torch.load(verify_descriptor(checkpoints['eeg']), map_location='cpu', weights_only=True)
            require(state_digest(model.state_dict()) == state_digest(saved['state_dict']), 'Original EEG training weights do not exactly reproduce')
            require(curve == old_run['folds'][f'fold_{fold}']['branches']['eeg']['training_curve'], 'Original EEG loss curve does not exactly reproduce')
            checks.append({**meta, 'checkpoints': checkpoints, 'parent_probability_max_errors': errors,
                           'exact_old_eeg_trained_state': True, 'exact_old_eeg_60_losses': True,
                           'old_fusion': selected})
            print(f'Original TRI-DEP replay passed seed={value} fold={fold}: exact EEG weights and 60 losses.', flush=True)
            del model, saved
    write_new(OUT/'preflight.json', {'passed': True, 'configuration': descriptor(CONFIG), 'checks': checks,
              'replayed_old_eeg_runs': 15, 'replay_updates': 1800, 'parent_checkpoints_verified': 45,
              'branch_probabilities_verified': 1665, 'corrected_features_verified': 37,
              'protected_files_unchanged': protect_history()})


def train():
    context, cfg = setup()
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG), 'Preflight incomplete')
    data = load_inputs(context)
    for value in cfg['seeds']:
        for fold in cfg['folds']:
            started = time.perf_counter()
            folder = OUT/f'runs/seed_{value}/fold_{fold}'
            require(not folder.exists(), 'Refusing to overwrite a corrected EEG run')
            rows, checkpoints, _ = parent_bundle(context, value, fold)
            model, curve, meta = fit_eeg(context, cfg, value, fold, data['corrected'])
            old = next(c for c in pre['checks'] if c['seed'] == value and c['fold'] == fold)
            require(all(old[k] == v for k, v in meta.items()), 'Initial state/batch/RNG/budget pairing differs')
            meta.update(experiment=EXPERIMENT, configuration=descriptor(CONFIG),
                        cohort_sha256=context.cohort_sha256, folds_sha256=context.folds_sha256,
                        corrected_features=data['corrected_source'], original_parent_checkpoints=checkpoints,
                        objective='original_TRI_DEP_cross_entropy', device='cuda', torch=str(torch.__version__),
                        branch_configuration=cfg['eeg_branch'], original_probability_file=descriptor(S00/f'runs/B0-37-fulltext/seed_{value}/probabilities.jsonl'))
            folder.mkdir(parents=True)
            cp = folder/'checkpoint.pt'
            with cp.open('xb') as stream:
                torch.save({'state_dict': model.state_dict(), 'metadata': meta}, stream)
            write_new(folder/'checkpoint.json', {'checkpoint': descriptor(cp), 'metadata': meta})
            write_new(folder/'train_history.json', curve)
            p = probabilities(model, data['corrected'], context.ids)
            saved = torch.load(cp, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['state_dict'], strict=True)
            require(p == probabilities(model, data['corrected'], context.ids), 'New checkpoint predictions changed')
            fused, selection = fuse(rows, p)
            for r in fused:
                r['eeg_checkpoint'] = descriptor(cp)
                r['corrected_eeg_feature'] = data['corrected_source'][r['subject_id']]
                r['unchanged_branch_checkpoints'] = {m: checkpoints[m] for m in ['speech', 'text']}
                r['configuration'] = descriptor(CONFIG)
            write_new(folder/'predictions.json', fused)
            write_new(folder/'fusion_selection.json', selection)
            write_new(folder/'reload_check.json', {'exact_state': state_digest(model.state_dict()) == state_digest(saved['state_dict']),
                      'exact_37_eeg_probabilities': True, 'restricted_loader': True})
            write_new(folder/'complete.json', {name: descriptor(folder/name) for name in
                ['checkpoint.pt', 'checkpoint.json', 'train_history.json', 'predictions.json', 'fusion_selection.json', 'reload_check.json']})
            print(f'TC-00-E complete seed={value} fold={fold} updates=120 seconds={time.perf_counter()-started:.1f}', flush=True)
            del model, saved
    print(f'Training complete; {protect_history()} historical files unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'train'])
    (preflight if parser.parse_args().action == 'preflight' else train)()
