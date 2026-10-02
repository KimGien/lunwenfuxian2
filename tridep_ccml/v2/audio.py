"""Fixed HuBERT mean-pooling speech revision, adaptation, and matched controls."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from .common import (WORK, TC, OUT as EEG, CONTROL as OLD_CONTROL, LOSS, RecordContext,
                     descriptor, verify_descriptor, read_json, require, write_new, set_seed, digest)
from .nr import OUT as NR, VARIANTS, CandidateHeads, load_data as load_old_data, logits_all as old_logits
from .np_combo import OUT as NP_OUT
from .resume_eeg import load_checkpoint as load_eeg_checkpoint
from .rebuild_eeg import loss_parts, predict_rows, validate_row
from tridep_ccml.ccml import ThreeViewHeads, ccml_loss, evidence_dc
from tridep_ccml.train import make_initial_state, state_digest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = TC/'experiments/CCML-v2-A-37-fulltext'
CONFIG = OUT/'configuration.json'
PLAN = WORK/'output/plans/CCML_v2_A_语音汇聚修订固定计划.md'
B0 = WORK/'S00/features/B0-37'
SEGMENTS = B0/'hubert_segments'
STAGES = {s: f'CCML-v2-A-{s}-37-fulltext' for s in ['adapt', 'control', 'N']}
MODALITIES = ['eeg', 'speech', 'text']


def pool_question(vectors, windows):
    """Undo lexicographic filenames; fixed mean, including duplicate tail windows."""
    require(vectors.ndim == 2 and vectors.shape[1] == 768 and vectors.dtype == np.float32
            and np.isfinite(vectors).all(), 'Invalid HuBERT vectors')
    require(len(vectors) == len(windows), 'Window/row count mismatch')
    if not len(vectors):
        return np.zeros(768, dtype=np.float32)
    # Producer rows are sorted by part1, part10, ..., part2; windows are chronological.
    lexical = sorted(range(len(windows)), key=lambda i: f'part{i+1}.wav')
    chronology = np.argsort(lexical)
    return vectors[chronology].mean(axis=0, dtype=np.float64).astype(np.float32)


def pool_subject(questions):
    require(questions.shape == (29, 768) and questions.dtype == np.float32 and
            np.isfinite(questions).all(), 'Expected 29 fixed questions')
    return questions.mean(axis=0, dtype=np.float64).astype(np.float32)


def initial_projection(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(768, 128), nn.ReLU())


class SpeechAdaptation(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.speech_projection = initial_projection(seed)
        self.heads = ThreeViewHeads()
        self.heads.load_state_dict(make_initial_state(seed), strict=True)
        self.eval()

    def representations(self, pooled, frozen):
        return torch.stack((frozen[:, 0], self.speech_projection(pooled), frozen[:, 2]), dim=1)

    def forward(self, pooled, frozen):
        return self.heads(self.representations(pooled, frozen))


def make_model(seed, stage):
    require(stage in STAGES, 'Unknown audio stage')
    if stage == 'adapt':
        return SpeechAdaptation(seed)
    model = CandidateHeads('N' if stage == 'N' else 'C')
    model.load_state_dict(make_initial_state(seed), strict=True)
    return model.eval()


def configuration(context):
    return {'family': OUT.name, 'version': 1, 'plan': descriptor(PLAN),
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'subject_ids': context.ids, 'seeds': [11, 22, 33], 'folds': [1, 2, 3, 4, 5],
            'stages': STAGES, 'formal_runs': 45, 'updates': 10800,
            'speech': {'pooling': 'equal segment mean per question, equal 29-question mean',
                       'accumulation': 'float64 mean then float32', 'dimension': 768,
                       'short_question': 'zero 768D vector, included in denominator 29',
                       'window_policy': 'retain historical 5s windows, duplicate tail and PCM16 path',
                       'sequence_order_learned': False, 'projection': 'Linear(768,128)+ReLU',
                       'projection_parameters': 98432, 'initialization': 'seed-specific torch Linear default'},
            'adaptation_parameters': 99206, 'control_parameters': 774,
            'projection_lr': 1e-4, 'head_lr': 1e-3, 'epochs': 60, 'batch_size': 8,
            'optimizer': 'Adamax', 'weight_decay': 0., 'gradient_clip': 1., 'ccml': LOSS,
            'normalization': {'adapt': 'none', 'control': 'none', 'N': 'LayerNorm128_no_affine_eps1e-5'},
            'device': 'cpu', 'threads': 1, 'dropout': 'none', 'selection': 'fixed_last_epoch',
            'threshold': .5, 'tie_class': 0, 'search': False,
            'comparisons': {'primary': 'A-N versus old N', 'within_version': 'A-N versus A-control',
                            'secondary': 'A-control versus E-control', 'NP': 'descriptive reference only'},
            'stage_gate': {'f1_delta_min': .01, 'positive_seeds_min': 2, 'recall_delta_min': -.05,
                           'nll_brier': 'report tradeoffs, no automatic promotion if worsened'},
            'prior_status': {n: descriptor(p/'status.json') for n, p in [('E', EEG), ('NR', NR), ('NP', NP_OUT)]},
            'historical_ancestry': 'frozen same-fold EEG/text parents retain prior supervised ancestry; new speech projection has no historical CE weights',
            'sources': {p.name: descriptor(p) for p in [Path(__file__), TC/'v2/complete_audio.py',
                TC/'v2/test_audio.py', TC/'v2/common.py', TC/'v2/nr.py', TC/'v2/resume_eeg.py',
                TC/'v2/rebuild_eeg.py', TC/'ccml.py', TC/'train.py', TC/'records.py', TC/'metrics.py']}}


def protect_history():
    path = OUT/'history_before.json'
    if not path.exists():
        tracked = read_json(NP_OUT/'history_before.json')
        for p in NP_OUT.rglob('*'):
            if p.is_file():
                tracked[p.relative_to(WORK).as_posix()] = digest(p)
        for p in [TC/'v2/np_combo.py', TC/'v2/complete_np.py', TC/'v2/test_np.py',
                  WORK/'output/plans/CCML_v2_NP_固定组合实验计划.md',
                  TC/'experiments/CCML-v2-NP-training.log']:
            tracked[p.relative_to(WORK).as_posix()] = digest(p)
        for p in SEGMENTS.rglob('*'):
            if p.is_file():
                tracked[p.relative_to(WORK).as_posix()] = digest(p)
        for p in [B0/'protocol_resolution.json', B0/'vendor_code_lock.json',
                  WORK/'S00/manifests/backbones.json', WORK/'S00/scripts/extract_b0.py']:
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
    cfg = configuration(context)
    for item in cfg['prior_status'].values():
        status = read_json(verify_descriptor(item))
        require(status['status'] == 'complete', 'Previous stage incomplete')
        verify_descriptor(status['audit'])
        verify_descriptor(status['report'])
    write_new(CONFIG, cfg)
    return context, cfg


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            require(set(old.files) == set(arrays) and all(np.array_equal(old[k], v) for k, v in arrays.items()), 'Existing cache changed')
    else:
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)


def extract_pool(context):
    expected = {'protocol_sha256': digest(B0/'protocol_resolution.json'),
                'producer_sha256': digest(WORK/'S00/scripts/extract_b0.py'),
                'backbones_sha256': digest(WORK/'S00/manifests/backbones.json'),
                'vendor_code_sha256': digest(B0/'vendor_code_lock.json')}
    backbone = read_json(WORK/'S00/manifests/backbones.json')['hubert_code']
    for filename, info in backbone['files'].items():
        require(digest(WORK/'S00/models/hubert_code'/filename) == info['sha256'], 'HuBERT files changed')
    # Verify the producer's vendor lock rather than inferring backbone provenance from filenames.
    for relative, value in read_json(B0/'vendor_code_lock.json').items():
        require(digest(WORK/'S00'/relative) == value, 'Historical vendor source changed')
    source_hashes = {r['path'].replace('\\', '/'): r['sha256'] for r in read_json(WORK/'S00/manifests/modma_raw_files.json')}
    subjects = {r['subject_id']: r for r in read_json(WORK/'S00/manifests/subjects.json')}
    question_arrays, pooled, counts, entries = [], [], [], []
    for sid in context.ids:
        questions, nsegments = [], []
        for q in range(1, 30):
            cache = SEGMENTS/sid/f'{q:02d}.npy'
            meta = read_json(cache.with_suffix('.json'))
            require(all(meta.get(k) == v for k, v in expected.items()), 'Stale HuBERT cache metadata')
            require(meta['sha256'] == digest(cache), 'Changed HuBERT cache')
            raw = WORK/'S00/staged'/sid/'audio'/f'{q:02d}.wav'
            original = (Path(subjects[sid]['audio_dir'])/f'{q:02d}.wav').as_posix()
            require(digest(raw) == meta['source_sha256'] == source_hashes[original], 'Audio ancestry mismatch')
            vectors = np.load(cache, allow_pickle=False)
            windows = meta['windows_samples']
            require(len(vectors) == meta['segment_count'] == len(windows), 'Segment count mismatch')
            require(meta['zero_feature'] == (len(vectors) == 0), 'Zero flag mismatch')
            samples = meta['processed_samples']
            expected_windows = []
            if samples >= 80000:
                starts = list(range(0, samples-80000+1, 40000))
                expected_windows = [[s, s+80000] for s in starts]
                if samples-starts[-1] > 40000:
                    expected_windows.append([samples-80000, samples])
            require(windows == expected_windows, 'Historical window policy mismatch')
            require(meta['duplicate_windows'] == len(windows)-len(set(map(tuple, windows))), 'Duplicate-window flag mismatch')
            questions.append(pool_question(vectors, windows))
            nsegments.append(len(vectors))
            entries.append({'subject_id': sid, 'question': q, 'cache': descriptor(cache),
                            'metadata': descriptor(cache.with_suffix('.json')), 'raw_audio': descriptor(raw),
                            'segment_count': len(vectors), 'duplicate_windows': meta['duplicate_windows'],
                            'processed_samples': samples, 'empty': not bool(len(vectors))})
        questions = np.stack(questions)
        question_arrays.append(questions)
        pooled.append(pool_subject(questions))
        counts.append(nsegments)
    arrays = {'question_means': np.stack(question_arrays), 'pooled': np.stack(pooled),
              'segment_counts': np.array(counts, dtype=np.int64), 'subject_ids': np.array(context.ids)}
    require(sum(r['empty'] for r in entries) == 96, 'Unexpected short-question coverage')
    save_npz(OUT/'speech_pool.npz', arrays)
    write_new(OUT/'speech_pool.json', {'cache': descriptor(OUT/'speech_pool.npz'),
              'configuration': descriptor(CONFIG), 'cohort_sha256': context.cohort_sha256,
              'subject_ids': context.ids, 'fit_subject_ids': [], 'labels_used': False,
              'backbone': backbone, 'sources': entries, 'questions': 1073, 'empty_questions': 96,
              'affected_subjects': len({r['subject_id'] for r in entries if r['empty']}),
              'mean_is_order_invariant': True, 'short_audio_repaired': False})
    print('Speech mean cache ready: 37 subjects, 1073 questions, 96 original empty questions retained.', flush=True)


def load_pool(context):
    info = read_json(OUT/'speech_pool.json')
    verify_descriptor(info['configuration'])
    require(info['subject_ids'] == context.ids and info['cohort_sha256'] == context.cohort_sha256 and
            info['fit_subject_ids'] == [] and info['labels_used'] is False, 'Pool provenance mismatch')
    path = verify_descriptor(info['cache'])
    with np.load(path, allow_pickle=False) as arrays:
        pooled = arrays['pooled'].copy()
        require(arrays['subject_ids'].tolist() == context.ids, 'Pool subject mismatch')
    require(pooled.shape == (37, 768) and pooled.dtype == np.float32 and np.isfinite(pooled).all(), 'Invalid pooled speech')
    return pooled, {'cache': info['cache'], 'manifest': descriptor(OUT/'speech_pool.json')}


def preflight():
    context, cfg = setup()
    extract_pool(context)
    checks = []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed)
            data = load_old_data(context, seed, fold)
            for name, folder in [('C', EEG/'runs'/OLD_CONTROL/f'seed_{seed}/fold_{fold}'),
                                 ('N', NR/'runs'/VARIANTS['N']['experiment']/f'seed_{seed}/fold_{fold}')]:
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                saved = (load_eeg_checkpoint(folder/'checkpoint.pt') if name == 'C' else
                         torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True))
                model = CandidateHeads(name)
                model.load_state_dict(saved['state_dict'], strict=True)
                result = evidence_dc(F.softplus(old_logits(model, data['features'])))
                rows = read_json(folder/'predictions.json')
                require([r['subject_id'] for r in rows] == context.ids, 'Old comparator identities changed')
                require(np.array_equal(result['p'].numpy(), np.array([r['p_final'] for r in rows])), 'Old probabilities not reproduced')
                require(saved['metadata']['train_ids'] == context.folds[f'fold_{fold}']['train'], 'Old train IDs changed')
                initial_key = 'head_initialization_sha256' if name == 'C' else 'initialization_sha256'
                require(saved['metadata'][initial_key] == state_digest(make_initial_state(seed)), 'Head initialization changed')
                checks.append({'seed': seed, 'fold': fold, 'comparator': name, 'exact_reproduced_rows': 37})
    write_new(OUT/'preflight.json', {'passed': True, 'configuration': descriptor(CONFIG),
              'pool': descriptor(OUT/'speech_pool.json'), 'checks': checks, 'exact_comparator_records': 1110,
              'protected_files_unchanged': protect_history()})
    print('Audio preflight passed: source checks and 1110 exact control/N predictions.', flush=True)


def load_data(context, seed, fold, stage):
    old = load_old_data(context, seed, fold)
    if stage == 'adapt':
        pooled, source = load_pool(context)
        return {**old, 'pooled': pooled, 'source': {'frozen_parent': old['source'], 'speech_pool': source}}
    info = read_json(OUT/f'features/seed_{seed}/fold_{fold}.json')
    require(info['fit_subject_ids'] == context.folds[f'fold_{fold}']['train'] and
            info['seed'] == seed and info['fold'] == fold, 'New speech parent fitted on wrong subjects')
    verify_descriptor(info['parent_checkpoint'])
    require(info['configuration'] == descriptor(CONFIG), 'New parent configuration changed')
    with np.load(verify_descriptor(info['features']), allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    require(data['subject_ids'].tolist() == context.ids and np.array_equal(data['labels'], old['labels']) and
            np.array_equal(data['splits'], old['splits']), 'New parent labels/splits mismatch')
    require(data['features'].shape == (37, 3, 128) and np.isfinite(data['features']).all(), 'Invalid revised features')
    require(np.array_equal(data['features'][:, [0, 2]], old['features'][:, [0, 2]]), 'EEG/text representations changed')
    data['source'] = {'features': info['features'], 'manifest': descriptor(OUT/f'features/seed_{seed}/fold_{fold}.json'),
                      'parent_checkpoint': info['parent_checkpoint']}
    return data


def forward_all(model, data, stage, representations=False):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(data['features']), 8):
            x = torch.from_numpy(data['features'][i:i+8])
            if stage == 'adapt':
                pooled = torch.from_numpy(data['pooled'][i:i+8])
                value = model.representations(pooled, x) if representations else model(pooled, x)
            else:
                require(not representations, 'Only adaptation creates representations')
                value = model(x)
            outputs.append(value)
    return torch.cat(outputs)


def predictions(model, data, context, seed, fold, stage, checkpoint):
    rows = predict_rows(forward_all(model, data, stage), data, context, seed, fold,
                        STAGES[stage], descriptor(checkpoint), data['source'])
    for row in rows:
        row.update(configuration=descriptor(CONFIG), split=str(row['split']))
        validate_row(row)
    return rows


def fit(context, cfg, seed, fold, stage):
    start = time.perf_counter()
    set_seed(seed)
    data = load_data(context, seed, fold, stage)
    model = make_model(seed, stage)
    folder = OUT/'runs'/STAGES[stage]/f'seed_{seed}/fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    cp = folder/'checkpoint.pt'
    initial_sha = state_digest(model.state_dict())
    head_state = model.heads.state_dict() if stage == 'adapt' else model.state_dict()
    require(state_digest(head_state) == state_digest(make_initial_state(seed)), 'Fresh head mismatch')
    parameter_count = sum(p.numel() for p in model.parameters())
    require(parameter_count == (99206 if stage == 'adapt' else 774), 'Wrong trainable parameters')
    meta = {'experiment': STAGES[stage], 'stage': stage, 'seed': seed, 'fold': fold,
            'train_ids': context.folds[f'fold_{fold}']['train'], 'source': data['source'],
            'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
            'configuration': descriptor(CONFIG), 'initial_state_sha256': initial_sha,
            'head_initialization_sha256': state_digest(make_initial_state(seed)),
            'initial_projection_sha256': state_digest(model.speech_projection.state_dict()) if stage == 'adapt' else None,
            'trainable_parameters': parameter_count, 'normalization': cfg['normalization'][stage],
            'projection_lr': cfg['projection_lr'] if stage == 'adapt' else None, 'head_lr': cfg['head_lr'],
            'weight_decay': 0., 'epochs': 60, 'updates': 240, 'objective': 'full_CCML_only',
            'selection': 'last_epoch', 'device': 'cpu', 'torch': str(torch.__version__)}
    if cp.exists():
        side = read_json(cp.with_suffix('.json'))
        verify_descriptor(side['checkpoint'])
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        require(saved['metadata'] == side['metadata'] and all(saved['metadata'].get(k) == v for k, v in meta.items()), 'Checkpoint provenance mismatch')
        model.load_state_dict(saved['state_dict'], strict=True)
    else:
        lookup = {s: i for i, s in enumerate(context.ids)}
        selected = [lookup[s] for s in meta['train_ids']]
        require(not (set(meta['train_ids']) & set(context.folds[f'fold_{fold}']['val'] + context.folds[f'fold_{fold}']['test'])), 'Training leakage')
        x = torch.from_numpy(data['features'][selected])
        y = torch.from_numpy(data['labels'][selected])
        pooled = torch.from_numpy(data['pooled'][selected]) if stage == 'adapt' else None
        groups = ([{'params': model.speech_projection.parameters(), 'lr': cfg['projection_lr']},
                   {'params': model.heads.parameters(), 'lr': cfg['head_lr']}] if stage == 'adapt' else
                  [{'params': model.parameters(), 'lr': cfg['head_lr']}])
        optimizer = torch.optim.Adamax(groups, weight_decay=0.)
        generator = torch.Generator().manual_seed(seed)
        batch_hash = hashlib.sha256()
        history, updates = [], 0
        for epoch in range(1, 61):
            order = torch.randperm(len(selected), generator=generator)
            batch_hash.update(order.numpy().astype('<i8').tobytes())
            total, terms, norms, project_grads = 0., {}, [], []
            head_grads = {m: [] for m in MODALITIES}
            for i in range(0, len(order), 8):
                b = order[i:i+8]
                optimizer.zero_grad(set_to_none=True)
                logits = model(pooled[b], x[b]) if stage == 'adapt' else model(x[b])
                e = F.softplus(logits)
                loss = ccml_loss(e, y[b], epoch, **LOSS)
                loss.backward()
                heads = model.heads.heads if stage == 'adapt' else model.heads
                for name, head in heads.items():
                    require(all(p.grad is not None for p in head.parameters()), 'Missing head gradient')
                    head_grads[name].append(float(torch.sqrt(sum((p.grad**2).sum() for p in head.parameters()))))
                if stage == 'adapt':
                    require(all(p.grad is not None for p in model.speech_projection.parameters()), 'Missing speech gradient')
                    project_grads.append(float(torch.sqrt(sum((p.grad**2).sum() for p in model.speech_projection.parameters()))))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameters')
                total += float(loss.detach())*len(b)
                norms.append(float(norm))
                with torch.no_grad():
                    part = loss_parts(e.detach(), y[b], epoch)
                for k, v in part.items():
                    terms[k] = terms.get(k, 0.)+v*len(b)
            history.append({'epoch': epoch, 'train_loss': total/len(selected),
                            'loss_terms': {k: v/len(selected) for k, v in terms.items()},
                            'preclip_grad_norm_mean': float(np.mean(norms)),
                            'speech_projection_grad_norm_mean': float(np.mean(project_grads)) if project_grads else None,
                            'head_grad_norm_mean': {m: float(np.mean(v)) for m, v in head_grads.items()}})
        require(updates == 240, 'Update budget changed')
        meta['batch_order_sha256'] = batch_hash.hexdigest()
        base_meta = read_json(EEG/'runs'/OLD_CONTROL/f'seed_{seed}/fold_{fold}/checkpoint.json')['metadata']
        require(meta['batch_order_sha256'] == base_meta['batch_order_sha256'] and
                meta['head_initialization_sha256'] == base_meta['head_initialization_sha256'], 'Old control pairing failed')
        require(state_digest(model.state_dict()) != initial_sha, 'No model update')
        with cp.open('xb') as stream:
            torch.save({'state_dict': model.state_dict(), 'metadata': meta}, stream)
        write_new(cp.with_suffix('.json'), {'checkpoint': descriptor(cp), 'metadata': meta})
        write_new(folder/'train_history.json', history)
        before = forward_all(model, data, stage)
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        require(state_digest(saved['state_dict']) == state_digest(model.state_dict()), 'State reload changed')
        model.load_state_dict(saved['state_dict'], strict=True)
        require(torch.equal(before, forward_all(model, data, stage)), 'Output reload changed')
        write_new(folder/'reload_check.json', {'exact_state': True, 'exact_all_37_logits': True, 'restricted_loader': True})
    write_new(folder/'predictions.json', predictions(model, data, context, seed, fold, stage, cp))
    if stage == 'adapt':
        reps = forward_all(model, data, stage, representations=True).numpy()
        require(np.array_equal(reps[:, [0, 2]], data['features'][:, [0, 2]]), 'EEG/text changed')
        require(state_digest(model.speech_projection.state_dict()) != meta['initial_projection_sha256'], 'Speech projection did not change')
        path = OUT/f'features/seed_{seed}/fold_{fold}.npz'
        save_npz(path, {'features': reps, 'labels': data['labels'], 'splits': data['splits'], 'subject_ids': np.array(context.ids)})
        write_new(path.with_suffix('.json'), {'features': descriptor(path), 'parent_checkpoint': descriptor(cp),
                  'configuration': descriptor(CONFIG), 'seed': seed, 'fold': fold,
                  'fit_subject_ids': context.folds[f'fold_{fold}']['train'], 'source': data['source'],
                  'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256,
                  'eeg_text_exactly_unchanged': True, 'fitted_objective': 'full_CCML_only'})
    write_new(folder/'complete.json', {name: descriptor(folder/name) for name in
              ['checkpoint.pt', 'checkpoint.json', 'train_history.json', 'predictions.json', 'reload_check.json']})
    print(f'Completed A-{stage} seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-start:.1f}', flush=True)


def train():
    context, cfg = setup()
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG), 'Preflight incomplete')
    require(read_json(OUT/'test_results.json')['passed'], 'Unit tests incomplete')
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            for stage in STAGES:
                fit(context, cfg, seed, fold, stage)
    print(f'Audio stage complete: 45 runs; {protect_history()} historical files unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'train'])
    (preflight if parser.parse_args().action == 'preflight' else train)()
