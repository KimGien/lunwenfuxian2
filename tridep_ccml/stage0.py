"""Lock original S00 controls and replay frozen branches, with zero fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
WORK = Path(__file__).resolve().parent.parent
S00 = WORK / 'S00'
OUT = WORK / 'output/analysis/tridep_stage0_20261004'
PLAN = WORK / 'output/plans/TriDep_EEG与融合改进及语音文本泛化汇总计划_v1_20261004.md'
ANNEX = WORK / 'output/plans/TriDep_阶段0_对照与来源锁定执行附表_v1_20261004.md'
sys.path.insert(0, str(S00 / 'scripts'))
import numpy as np
import torch
from cohort import validate_active_cohort, validate_prediction_records
from fulltext_b0 import feature_paths, source_checks, merge_segments
from run_b0 import check_inputs
from metrics import metrics as legacy_metrics, select_weights
from smoke_models import seed as set_seed
sys.path.insert(0, str(S00 / 'upstream-dev'))
from lib.models import models

MODALITIES = ('eeg', 'speech', 'text')
CONDITIONS = ('ST-val-weighted', 'EST-original-val-weighted', 'ST-equal', 'EST-equal')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(WORK).as_posix()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


class Sources:
    def __init__(self):
        self.items = {}

    def add(self, path, role, expected=None):
        path = Path(path).resolve()
        key = relative(path)
        if key not in self.items:
            self.items[key] = {'path': key, 'sha256': sha(path), 'bytes': path.stat().st_size, 'roles': []}
        item = self.items[key]
        if expected is not None and item['sha256'] != expected:
            raise ValueError('Source hash mismatch: ' + key)
        if role not in item['roles']:
            item['roles'].append(role)
        return {'path': key, 'sha256': item['sha256']}

    def rows(self):
        return [self.items[k] for k in sorted(self.items)]

    def verify_unchanged(self):
        for r in self.rows():
            if sha(WORK / r['path']) != r['sha256']:
                raise ValueError('Original source changed during replay: ' + r['path'])
        return len(self.items)


def binary_metrics(labels, ps):
    result = legacy_metrics(labels, ps)
    result['nll'] = -sum(math.log(max(1e-15, min(1 - 1e-15, p if y else 1 - p))) for y, p in zip(labels, ps)) / len(labels)
    result['brier'] = sum((p - y) ** 2 for y, p in zip(labels, ps)) / len(labels)
    pos = [p for y, p in zip(labels, ps) if y]
    neg = [p for y, p in zip(labels, ps) if not y]
    result['auc'] = sum((a > b) + .5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else None
    result['confusion_hc_mdd'] = [[sum(y == a and int(p > .5) == b for y, p in zip(labels, ps)) for b in (0, 1)] for a in (0, 1)]
    return result


def select_st(rows):
    if len(rows) != 3 or any(r['split'] != 'val' for r in rows):
        raise ValueError('ST selection requires the original three validation subjects')
    if len({(r['seed'], r['outer_fold']) for r in rows}) != 1 or len({r['subject_id'] for r in rows}) != 3:
        raise ValueError('ST validation provenance mismatch')
    candidates = []
    best = None
    for j in range(21):
        weights = [0., j / 20, (20 - j) / 20]
        score = legacy_metrics([r['label'] for r in rows], [sum(w * r['p_' + m] for w, m in zip(weights, MODALITIES)) for r in rows])['macro_f1']
        candidates.append({'weights': weights, 'validation_macro_f1': score})
        if best is None or score > best['validation_macro_f1']:
            best = candidates[-1]
    return {**best, 'candidates': candidates}


def summarize(rows, folds):
    result = {}
    keys = ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')
    for condition in CONDITIONS + MODALITIES:
        chosen = [r for r in rows if r['condition'] == condition]
        fold_metrics = []
        seed_means = {}
        pooled = {}
        for seed in (11, 22, 33):
            for fold in range(1, 6):
                for split in ('train', 'val', 'test'):
                    rr = [r for r in chosen if (r['seed'], r['outer_fold'], r['split']) == (seed, fold, split)]
                    expected = folds[f'fold_{fold}'][split]
                    if {r['subject_id'] for r in rr} != set(expected) or len(rr) != len(expected):
                        raise ValueError('Incomplete control split')
                    fold_metrics.append({'seed': seed, 'fold': fold, 'split': split, **binary_metrics([r['label'] for r in rr], [r['p_final'] for r in rr])})
            seed_means[str(seed)] = {split: {k: float(np.mean([r[k] for r in fold_metrics if r['seed'] == seed and r['split'] == split])) for k in keys} for split in ('train', 'val', 'test')}
            rr = [r for r in chosen if r['seed'] == seed and r['split'] == 'test']
            if len(rr) != 37 or len({r['subject_id'] for r in rr}) != 37:
                raise ValueError('Each seed must have exactly 37 unique OOF test subjects')
            pooled[str(seed)] = binary_metrics([r['label'] for r in rr], [r['p_final'] for r in rr])
        result[condition] = {'main_test': {k: float(np.mean([s['test'][k] for s in seed_means.values()])) for k in keys},
                             'train_val_test_seed_means': seed_means, 'fold_metrics': fold_metrics, 'pooled_oof_per_seed': pooled}
    return result


def audit_data(sources, lock, cfg):
    folds = read(S00 / lock['fold_file'])
    subjects = {r['subject_id']: r for r in read(S00 / 'manifests/subjects.json')}
    raw = {str(Path(r['path'])): r for r in read(S00 / 'manifests/modma_raw_files.json')}
    staged = {r['subject_id']: r for r in read(S00 / 'manifests/staged_dataset.json')}
    label_maps = read(S00 / 'manifests/modma_labels.json')
    for name, table in label_maps.items():
        matches = [r for r in raw.values() if r['archive'] == name and r['member'] == table['archive_member']]
        if len(matches) != 1:
            raise ValueError('Missing diagnosis workbook provenance')
        row = matches[0]
        sources.add(WORK / row['path'], 'diagnosis_workbook', row['sha256'])
        for sid, subject in subjects.items():
            if table['labels'][sid]['label'] != subject['label']:
                raise ValueError('EEG/audio diagnosis disagreement: ' + sid)
    transcripts = read(S00 / 'manifests/transcripts_sha256.json')
    speech_rows = [r for r in read(S00 / 'features/B0-37/features_manifest.json') if r['modality'] == 'speech']
    speech_by_id = {(r['subject_id'], r['seed']): r for r in speech_rows}
    empty = []
    changed_fulltext = []
    segment_count = 0
    duplicate_windows = 0
    for sid in lock['subject_ids']:
        subject = subjects[sid]
        speech_audit_path = S00 / 'features/B0-37/speech_audits' / f'{sid}.json'
        for seed in cfg['seeds']:
            sources.add(speech_audit_path, 'original_speech_audit', speech_by_id[(sid, seed)]['details']['audit_sha256'])
        speech_audit = read(speech_audit_path)
        if len(speech_audit) != 29:
            raise ValueError('Incomplete speech question audit')
        for key in ('eeg_set', 'eeg_fdt'):
            p = Path(subject[key])
            sources.add(WORK / p, 'raw_' + key, raw[str(p)]['sha256'])
        sources.add(S00 / staged[sid]['eeg_mat'], 'staged_eeg', staged[sid]['mat_sha256'])
        if staged[sid]['fdt_sha256'] != raw[str(Path(subject['eeg_fdt']))]['sha256']:
            raise ValueError('Staged EEG source mismatch')
        for q in range(1, 30):
            p = Path(subject['audio_dir']) / f'{q:02d}.wav'
            raw_sha = raw[str(p)]['sha256']
            sources.add(WORK / p, 'raw_audio', raw_sha)
            audio = S00 / 'staged' / sid / 'audio' / f'{q:02d}.wav'
            sources.add(audio, 'staged_audio', raw_sha)
            transcript = S00 / 'transcripts/B0-37/first_segment_code' / sid / f'{q:02d}.json'
            sources.add(transcript, 'original_asr', transcripts[f'{sid}/{q:02d}'])
            parent = read(transcript)
            if parent['source_sha256'] != raw_sha or parent['cohort_id'] != lock['cohort_id'] or parent['folds_sha256'] != lock['folds_sha256']:
                raise ValueError('ASR source mismatch')
            full = S00 / 'transcripts/B0-37-fulltext' / sid / f'{q:02d}.json'
            sources.add(full, 'fulltext_asr')
            merged = read(full)
            answer, segments = merge_segments(parent['segments'])
            if merged['answer_text'] != answer or merged['segments'] != segments or merged['parent_transcript_sha256'] != sha(transcript) or merged['source_sha256'] != raw_sha:
                raise ValueError('Fulltext source mismatch')
            if parent['answer_text'] != answer:
                changed_fulltext.append(f'{sid}/{q:02d}')
            for seed in cfg['seeds']:
                if speech_by_id[(sid, seed)]['details']['source_sha256_by_question'][str(q)] != raw_sha:
                    raise ValueError('Speech input audio provenance mismatch')
            segment_meta_path = S00 / 'features/B0-37/hubert_segments' / sid / f'{q:02d}.json'
            sources.add(segment_meta_path, 'original_hubert_segment_metadata')
            segment_meta = read(segment_meta_path)
            audit_row = speech_audit[q - 1]
            if audit_row != {'question_id': f'{q:02d}', **segment_meta} or segment_meta['source_sha256'] != raw_sha:
                raise ValueError('Speech question audit and segment provenance differ')
            for key, path in [('producer_sha256', S00 / 'scripts/extract_b0.py'),
                              ('protocol_sha256', S00 / 'features/B0-37/protocol_resolution.json'),
                              ('backbones_sha256', S00 / 'manifests/backbones.json'),
                              ('vendor_code_sha256', S00 / 'features/B0-37/vendor_code_lock.json')]:
                sources.add(path, 'hubert_source_' + key, segment_meta[key])
            segment_path = segment_meta_path.with_suffix('.npy')
            sources.add(segment_path, 'original_hubert_segment_array', segment_meta['sha256'])
            array = np.load(segment_path, allow_pickle=False)
            if array.shape != (segment_meta['segment_count'], 768) or not np.isfinite(array).all():
                raise ValueError('HuBERT segment shape or finite error')
            segment_count += segment_meta['segment_count']
            duplicate_windows += segment_meta['duplicate_windows']
        for seed in cfg['seeds']:
            array = np.load(S00 / 'features/B0-37' / speech_by_id[(sid, seed)]['path'], allow_pickle=False)
            for i, item in enumerate(speech_audit):
                if item['segment_count'] == 0:
                    if not item['zero_feature'] or np.any(array[i] != 0):
                        raise ValueError('Original short audio zero feature audit mismatch')
                    empty.append({'subject_id': sid, 'seed': seed, 'question': item['question_id']})
    split_rows = []
    for name, split in folds.items():
        train, val, test = (set(split[x]) for x in ('train', 'val', 'test'))
        if train & val or train & test or val & test or train | val | test != set(lock['subject_ids']):
            raise ValueError('Person isolation failure')
        split_rows.append({'fold': int(name[-1]), 'train_ids': split['train'], 'val_ids': split['val'], 'test_ids': split['test'],
                           'class_counts': {part: dict(Counter(str(subjects[s]['label']) for s in ids)) for part, ids in split.items()},
                           'original_val_seen_by_same_fold_st_supervised_fit': False,
                           'inner_val_carved_from_outer_train_seen_by_fixed_st': True,
                           'historical_val_fusion_selection': True})
    return {'passed': True, 'subjects': 37, 'recordings': 1073, 'class_counts': lock['class_counts'],
            'raw_eeg_files': 74, 'staged_eeg_files': 37, 'transcripts_per_version': 1073,
            'changed_fulltext_recordings': changed_fulltext,
            'hubert_segment_count': segment_count, 'duplicate_speech_windows': duplicate_windows,
            'speech_zero_segment_records_across_seeds': empty, 'folds': split_rows}


def run():
    if (OUT / 'status.json').exists():
        raise ValueError('Existing stage0 output: use --verify; refusing to overwrite completion')
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    sources = Sources()
    for path in [PLAN, ANNEX, Path(__file__)]:
        sources.add(path, 'stage0_registration_and_code')
    lock, cfg, folds = source_checks()
    if cfg['seeds'] != [11, 22, 33] or cfg['experiment'] != 'B0-37-fulltext':
        raise ValueError('Incorrect original parent experiment')
    for folder in ('configs', 'manifests'):
        for p in (S00 / folder).glob('*.json'):
            sources.add(p, 'original_' + folder)
    for p in (S00 / 'scripts').glob('*.py'):
        sources.add(p, 'original_source_code')
    for p in (S00 / 'features/B0-37').glob('*.json'):
        sources.add(p, 'base_feature_provenance')
    for p in (S00 / 'features/B0-37-fulltext').glob('*.json'):
        sources.add(p, 'fulltext_feature_provenance')
    for path, expected in read(S00 / 'features/B0-37/vendor_code_lock.json').items():
        sources.add(S00 / path, 'original_vendor_code', expected)
    for p in (S00 / 'reports').glob('*'):
        if p.is_file():
            sources.add(p, 'original_source_report')
    for name, model in read(S00 / 'manifests/backbones.json').items():
        for filename, item in model['files'].items():
            sources.add(S00 / 'models' / name / filename, 'original_' + name + '_asset', item['sha256'])
    cb = S00 / 'upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth'
    cb_expected = read(S00 / 'manifests/source_sha256.json')[str(cb.relative_to(WORK))]['sha256']
    sources.add(cb, 'original_cbramod_checkpoint', cb_expected)
    for seed in cfg['seeds']:
        p = S00 / f'features/B0-37/encoders/seed_{seed}.pt'
        info = sources.add(p, 'random_frozen_speech_encoder')
        saved = torch.load(p, map_location='cpu', weights_only=True)
        if saved['seed'] != seed or saved['fit_subject_ids'] or saved['protocol_sha256'] != sha(S00 / 'features/B0-37/protocol_resolution.json'):
            raise ValueError('Random frozen speech encoder provenance mismatch')
        for row in read(S00 / 'features/B0-37/features_manifest.json'):
            if row['modality'] == 'speech' and row['seed'] == seed and row['details']['encoder_sha256'] != info['sha256']:
                raise ValueError('Speech random encoder source mismatch')
    print('Checking raw, staged, ASR, label and feature provenance', flush=True)
    data_audit = audit_data(sources, lock, cfg)
    textpaths = feature_paths(lock)
    basecfg = read(S00 / 'configs/b0.json')
    all_paths = {seed: check_inputs(S00 / 'features/B0-37', seed, folds, basecfg) for seed in cfg['seeds']}
    for paths in all_paths.values():
        for p in paths.values():
            sources.add(p, 'original_branch_input')
    for p in textpaths.values():
        sources.add(p, 'fulltext_branch_input')
    labels = {r['subject_id']: r['label'] for r in read(S00 / 'manifests/subjects.json')}
    ids = lock['subject_ids']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    set_seed(11)
    branch_rows, controls, provenance, selections, bundles = [], [], [], [], []
    audit = {'checkpoints': 0, 'probabilities': 0, 'max_probability_error': 0., 'classification_changes': 0,
             'representations_bitwise_equal': 0, 'frozen_state_unchanged': 0, 'repeat_input_equal': 0,
             'last_linear_replay_max_error': 0., 'original_est_max_error': 0.}
    for seed in cfg['seeds']:
        base = S00 / f'runs/B0-37/seed_{seed}'
        full = S00 / f'runs/B0-37-fulltext/seed_{seed}'
        for folder in (base, full):
            for p in folder.glob('*'):
                if p.is_file():
                    sources.add(p, 'historical_run')
        base_run, full_run = read(base / 'run.json'), read(full / 'run.json')
        if read(base / 'configuration.json') != basecfg or read(full / 'configuration.json') != cfg:
            raise ValueError('Run configuration differs from original settings')
        for run_info, configpath, scriptpath, manifestpath in (
            (base_run, S00 / 'configs/b0.json', S00 / 'scripts/run_b0.py', S00 / 'features/B0-37/features_manifest.json'),
            (full_run, S00 / 'configs/b0_fulltext.json', S00 / 'scripts/fulltext_b0.py', S00 / 'features/B0-37-fulltext/features_manifest.json')):
            for key, p in [('config_sha256', configpath), ('training_script_sha256', scriptpath), ('feature_manifest_sha256', manifestpath)]:
                sources.add(p, 'historical_run_source', run_info[key])
        sources.add(base / 'run.json', 'fulltext_parent_run', full_run['parent_run_sha256'])
        sources.add(base / 'probabilities.jsonl', 'fulltext_parent_predictions', full_run['parent_predictions_sha256'])
        historical = read_rows(full / 'probabilities.jsonl')
        original_base = read_rows(base / 'probabilities.jsonl')
        validate_prediction_records(historical, cfg['experiment'])
        validate_prediction_records(original_base, basecfg['experiment'])
        if len(historical) != 185 or len(original_base) != 185:
            raise ValueError('Incomplete historical predictions')
        for name, split in folds.items():
            fold = int(name[-1])
            predictions = {r['subject_id']: r for r in historical if r['outer_fold'] == fold}
            oldbase = {r['subject_id']: r for r in original_base if r['outer_fold'] == fold}
            oldcache = WORK / f'tridep_ccml/features/seed_{seed}/{name}.npz'
            oldmeta = read(oldcache.with_suffix('.json'))
            sources.add(oldcache, 'historical_128d_cache', oldmeta['npz_sha256'])
            sources.add(oldcache.with_suffix('.json'), 'historical_128d_metadata')
            sources.add(S00 / 'upstream-dev/lib/models/models.py', 'historical_model_code', oldmeta['upstream_model_source_sha256'])
            sources.add(WORK / 'tridep_ccml/prepare.py', 'historical_cache_exporter', oldmeta['exporter_sha256'])
            if oldmeta['subject_ids'] != ids or oldmeta['train_ids'] != split['train'] or oldmeta['seed'] != seed or oldmeta['fold'] != fold or oldmeta['modalities'] != list(MODALITIES):
                raise ValueError('Historical representation metadata mismatch')
            sources.add(full / 'probabilities.jsonl', 'historical_cache_probabilities', oldmeta['parent_prediction_file_sha256'])
            features = np.empty((37, 3, 128), dtype=np.float32)
            probabilities = np.empty((37, 3), dtype=np.float64)
            meta = {'cohort_id': lock['cohort_id'], 'folds_sha256': lock['folds_sha256'], 'seed': seed, 'fold': fold,
                    'subject_ids': ids, 'train_ids': split['train'], 'val_ids': split['val'], 'test_ids': split['test'],
                    'modalities': list(MODALITIES), 'representation_endpoint': 'original final Linear input after original projection/ReLU/dropout(eval)',
                    'parent_checkpoints': {}, 'source_features': {}, 'frozen_eval': True, 'requires_grad': False,
                    'source_prediction_file': sources.add(full / 'probabilities.jsonl', 'source_predictions'),
                    'model_source': sources.add(S00 / 'upstream-dev/lib/models/models.py', 'original_model'),
                    'stage0_exporter': sources.add(Path(__file__), 'stage0_exporter'),
                    'historical_cache': sources.add(oldcache, 'historical_128d_cache'),
                    'legal_inner_oof': False, 'original_val_previously_used_for_fusion_selection': True}
            for mi, modality in enumerate(MODALITIES):
                branch = cfg['branches'][modality]
                checkpoint = (full if modality == 'text' else base) / f'{name}_{modality}.pt'
                expected = full_run['folds'][name]['text_checkpoint_sha256'] if modality == 'text' else base_run['folds'][name]['branches'][modality]['checkpoint_sha256']
                cp = sources.add(checkpoint, 'original_' + modality + '_checkpoint', expected)
                if oldmeta['parent_checkpoints'][modality] != cp:
                    raise ValueError('Historical cache parent checkpoint mismatch')
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                expected_meta = {'experiment': 'B0-37-fulltext' if modality == 'text' else 'B0-37', 'cohort_id': lock['cohort_id'],
                                 'folds_sha256': lock['folds_sha256'], 'seed': seed, 'fold': fold, 'train_ids': split['train'], 'configuration': branch}
                if any(saved.get(k) != v for k, v in expected_meta.items()):
                    raise ValueError('Checkpoint training provenance mismatch: ' + cp['path'])
                model = getattr(models, branch['model_class'])(**branch['model_kwargs']).to(device)
                model.load_state_dict(saved['state_dict'], strict=True)
                model.requires_grad_(False).eval()
                before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
                    raise ValueError('Frozen/eval contract failure')
                captured = []
                handle = model.classifier[-1].register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
                meta['parent_checkpoints'][modality] = cp
                meta['source_features'][modality] = {}
                with torch.inference_mode():
                    for i, sid in enumerate(ids):
                        path = textpaths[sid] if modality == 'text' else all_paths[seed][(fold, sid, modality)]
                        input_source = sources.add(path, 'frozen_' + modality + '_input')
                        if oldmeta['source_features'][modality][sid] != input_source:
                            raise ValueError('Historical cache input mismatch')
                        tensor = torch.as_tensor(np.load(path, allow_pickle=False), dtype=torch.float32, device=device).unsqueeze(0)
                        logits = model(tensor)
                        h = captured.pop()
                        p = float(torch.softmax(logits, dim=-1)[0, 1].cpu())
                        historical_p = predictions[sid]['p_' + modality]
                        delta = abs(p - historical_p)
                        if delta > 1e-6 or int(p > .5) != int(historical_p > .5) or h.shape != (1, 128) or not torch.isfinite(h).all():
                            raise ValueError('Frozen branch replay mismatch')
                        if modality != 'text' and historical_p != oldbase[sid]['p_' + modality]:
                            raise ValueError('Fulltext EEG/speech differs from original parent')
                        end_p = float(torch.softmax(model.classifier[-1](h), dim=-1)[0, 1].cpu())
                        audit['last_linear_replay_max_error'] = max(audit['last_linear_replay_max_error'], abs(end_p - p))
                        if end_p != p:
                            raise ValueError('Representation is not the exact final classifier input')
                        if i == 0:
                            repeated = model(tensor)
                            repeated_h = captured.pop()
                            if not torch.equal(repeated, logits) or not torch.equal(repeated_h, h):
                                raise ValueError('Repeated fixed input changed output')
                            audit['repeat_input_equal'] += 1
                        features[i, mi] = h.cpu().numpy()[0]
                        probabilities[i, mi] = p
                        audit['probabilities'] += 1
                        audit['max_probability_error'] = max(audit['max_probability_error'], delta)
                        meta['source_features'][modality][sid] = input_source
                handle.remove()
                if any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
                    raise ValueError('Frozen parameters or buffers changed')
                audit['frozen_state_unchanged'] += 1
                audit['checkpoints'] += 1
                provenance.append({'seed': seed, 'fold': fold, 'modality': modality, 'checkpoint': cp,
                                   'checkpoint_experiment': saved['experiment'], 'configuration': branch,
                                   'parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters_stage0': 0,
                                   'train_ids': split['train'], 'val_ids': split['val'], 'test_ids': split['test'],
                                   'val_seen_by_supervised_fit': False, 'test_seen_by_supervised_fit': False})
                del model, saved, before
            splits = np.array([next(part for part in ('train', 'val', 'test') if sid in split[part]) for sid in ids])
            ys = np.array([labels[sid] for sid in ids])
            with np.load(oldcache, allow_pickle=False) as old:
                if not np.array_equal(old['features'], features) or not np.array_equal(old['subject_ids'], np.array(ids)) or not np.array_equal(old['labels'], ys) or not np.array_equal(old['splits'], splits):
                    raise ValueError('Historical 128d representation cache changed')
            audit['representations_bitwise_equal'] += 1
            target = OUT / f'features/seed_{seed}/{name}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as f:
                np.savez_compressed(f, features=features, probabilities=probabilities, subject_ids=np.array(ids), labels=ys, splits=splits)
            meta['npz_sha256'] = sha(target)
            write(target.with_suffix('.json'), meta)
            bundles.append({'path': relative(target), 'sha256': sha(target), 'metadata_path': relative(target.with_suffix('.json')),
                            'metadata_sha256': sha(target.with_suffix('.json')), 'shape': [37, 3, 128]})
            val = [predictions[s] for s in split['val']]
            historical_selection = select_weights(val)
            if historical_selection != full_run['folds'][name]['fusion']:
                raise ValueError('Historical EST validation weights do not replay')
            selected_st = select_st(val)
            selections.append({'seed': seed, 'fold': fold, 'val_ids': split['val'], 'ST': selected_st,
                               'EST': historical_selection, 'ST_selection_role': 'stage0_new_deterministic_control',
                               'test_used_in_selection': False})
            weights = {'ST-val-weighted': selected_st['weights'], 'EST-original-val-weighted': historical_selection['weights'],
                       'ST-equal': [0., .5, .5], 'EST-equal': [1 / 3] * 3}
            for i, sid in enumerate(ids):
                source = predictions[sid]
                common = {'subject_id': sid, 'seed': seed, 'outer_fold': fold, 'split': str(splits[i]), 'label': labels[sid],
                          'cohort_id': lock['cohort_id'], 'folds_sha256': lock['folds_sha256'],
                          'source_prediction_file_sha256': meta['source_prediction_file']['sha256'], 'feature_bundle_sha256': meta['npz_sha256']}
                branch_rows.append({**common, **{f'p_{m}': float(probabilities[i, j]) for j, m in enumerate(MODALITIES)},
                                    'parent_checkpoints': meta['parent_checkpoints']})
                for condition in CONDITIONS:
                    p = sum(w * source['p_' + m] for w, m in zip(weights[condition], MODALITIES))
                    if condition == 'EST-original-val-weighted':
                        err = abs(p - source['p_final'])
                        audit['original_est_max_error'] = max(audit['original_est_max_error'], err)
                        if err > 1e-12 or int(p > .5) != source['prediction']:
                            raise ValueError('Original EST differs from historical prediction')
                    controls.append({**common, 'condition': condition, 'fusion_weights': weights[condition],
                                     'p_final': p, 'prediction': int(p > .5), 'new_model_fits': 0,
                                     'probability_policy': 'reuse_exact_historical_probabilities_after_frozen_replay'})
            print(f'Replayed seed={seed} fold={fold}: all original weights, probabilities and 128d representations verified', flush=True)
    metric_rows = list(controls)
    for r in branch_rows:
        for m in MODALITIES:
            metric_rows.append({**r, 'condition': m, 'p_final': r['p_' + m]})
    summaries = summarize(metric_rows, folds)
    historical_f1 = read(S00 / 'reports/fulltext_comparison.json')['aggregate']['final']['macro_f1']
    if abs(summaries['EST-original-val-weighted']['main_test']['macro_f1'] - historical_f1) > 1e-12:
        raise ValueError('Original EST main metric differs from historical report')
    paired = []
    lookup = {(r['seed'], r['outer_fold'], r['subject_id'], r['condition']): r for r in controls if r['split'] == 'test'}
    for r in controls:
        if r['condition'] != 'ST-val-weighted' or r['split'] != 'test':
            continue
        e = lookup[(r['seed'], r['outer_fold'], r['subject_id'], 'EST-original-val-weighted')]
        st_ok, est_ok = r['prediction'] == r['label'], e['prediction'] == e['label']
        paired.append({'seed': r['seed'], 'fold': r['outer_fold'], 'subject_id': r['subject_id'], 'label': r['label'],
                       'p_ST': r['p_final'], 'p_EST': e['p_final'], 'prediction_ST': r['prediction'], 'prediction_EST': e['prediction'],
                       'change': 'added_correct' if not st_ok and est_ok else 'added_error' if st_ok and not est_ok else 'unchanged_correct' if st_ok else 'unchanged_error'})
    paired_summary = {'counts': dict(Counter(r['change'] for r in paired)), 'unique_subjects': 37, 'repeated_seed_test_records': 111,
                      'delta_EST_minus_ST': {k: summaries['EST-original-val-weighted']['main_test'][k] - summaries['ST-val-weighted']['main_test'][k] for k in summaries['ST-val-weighted']['main_test']},
                      'est_zero_eeg_weight_folds': sum(s['EST']['weights'][0] == 0 for s in selections)}
    write(OUT / 'data_and_split_audit.json', data_audit)
    write(OUT / 'checkpoint_provenance.json', provenance)
    write(OUT / 'validation_weight_selections.json', selections)
    write(OUT / 'feature_bundles.json', bundles)
    write_rows(OUT / 'branch_probabilities.jsonl', branch_rows)
    write_rows(OUT / 'control_predictions.jsonl', controls)
    write_rows(OUT / 'paired_test_changes.jsonl', paired)
    write(OUT / 'metrics.json', summaries)
    write(OUT / 'paired_summary.json', paired_summary)
    write(OUT / 'source_manifest.json', sources.rows())
    print(f'Rechecking {len(sources.items)} original source hashes', flush=True)
    unchanged = sources.verify_unchanged()
    status = {'status': 'complete', 'phase': 0, 'passed': True, 'started_at_Asia_Shanghai': started,
              'completed_at_Asia_Shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'parent_experiment': cfg['experiment'], 'cohort_id': lock['cohort_id'], 'folds_sha256': lock['folds_sha256'],
              'plan': relative(PLAN), 'plan_sha256': sha(PLAN), 'execution_annex': relative(ANNEX), 'execution_annex_sha256': sha(ANNEX),
              'new_neural_fits': 0, 'new_inner_fits': 0, 'new_calibrator_fits': 0, 'optimizer_updates': 0,
              'ST_validation_selections': 15, 'ST_candidate_scores': 315, 'EST_historical_candidate_replay_scores': 3465,
              'frozen_replay': audit, 'historical_sources_unchanged': unchanged, 'bundles': 15,
              'branch_probability_records': len(branch_rows), 'control_prediction_records': len(controls),
              'test_records_per_condition': 111, 'next_phase_started': False,
              'environment': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
                              'cuda': torch.version.cuda, 'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None},
              'evidence_role': 'previously_observed_37_subject_development_control', 'independent_external_validation': False}
    if audit['checkpoints'] != 45 or audit['probabilities'] != 1665 or audit['representations_bitwise_equal'] != 15:
        raise ValueError('Incomplete replay')
    report(status, summaries, paired_summary, data_audit, selections)
    outputs = [{'path': relative(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('status.json', 'failure.json', 'verification.json')]
    write(OUT / 'output_manifest.json', outputs)
    status['source_manifest_sha256'] = sha(OUT / 'source_manifest.json')
    status['output_manifest_sha256'] = sha(OUT / 'output_manifest.json')
    write(OUT / 'status.json', status)
    print(json.dumps({'passed': True, 'stage': 0, 'sources': unchanged, 'replay': audit,
                      'ST': summaries['ST-val-weighted']['main_test'], 'EST': summaries['EST-original-val-weighted']['main_test']}, ensure_ascii=False), flush=True)


def report(status, metrics, paired, data, selections):
    lines = ['# TriDep 阶段0：对照与来源锁定完成报告', '', '完成日期：2026-10-04。第0阶段通过，新增训练和优化更新均为0；阶段1及以后尚未启动。', '',
             '来源固定为原 S00 B0-37-fulltext：S和E继承B0-37，T采用全文版。45个同种子同外折检查点均按原class严格加载，全部eval、停止梯度、inference_mode，原参数与buffer逐位不变。', '',
             f'1665个分支概率重放通过，最大绝对误差{status["frozen_replay"]["max_probability_error"]:.12g}；最后Linear接口重放误差为0。15份37×3×128表示与历史TC准备缓存逐位一致，45个分支重复同一输入输出逐位相同。{status["historical_sources_unchanged"]}个来源文件推理前后SHA-256相同。', '',
             '## 固定来源', '',
             '- S：`S00/runs/B0-37/seed_<seed>/fold_<fold>_speech.pt`；29×512输入，原70轮。',
             '- T：`S00/runs/B0-37-fulltext/seed_<seed>/fold_<fold>_text.pt`；29×768全文CLS输入，原50轮。',
             '- E：`S00/runs/B0-37/seed_<seed>/fold_<fold>_eeg.pt`；60×200原输入，原60轮。',
             '- 新缓存：`features/seed_<seed>/fold_<fold>.npz`；包含37×3×128表示、37×3 MDD概率、人员、标签和split；维序E/S/T。每份JSON绑定原权重、输入、模型代码、历史概率和导出脚本哈希。', '',
             '活动分折SHA-256：`' + status['folds_sha256'] + '`。原38人分折仅移除02010004，顺序保持。原始数据、staged副本、两种ASR转写、诊断工作簿、固定骨干资产、随机语音编码器、输入、权重和历史结果均已核对。完整逐文件来源见 [source_manifest.json](source_manifest.json)，逐折训练人员及权重见 [checkpoint_provenance.json](checkpoint_provenance.json)。', '',
             '## 原ST和EST控制', '',
             'EST主控制直接复用原全文版的val选权和精确历史概率，全部555条train/val/test记录及历史主F1一致。原流程没有独立保存ST控制；本阶段按事前附表构造确定性ST控制：令E权重为0，在原0.05步长的21个S/T权重候选中只按本折3人val Macro-F1选权，平局保留低S权重。共315个评分、15次选择、0次模型拟合。ST等权、EST等权仅作补充，不按test排序改主控制。', '',
             '主口径：每种子五外折测试均值，再取种子11/22/33均值；p>0.5判MDD，相等判HC。AUC/NLL/Brier也按同口径平均，每种子37人OOF合并值另列于 [metrics.json](metrics.json)。', '',
             '| 控制 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        m = metrics[c]['main_test']
        lines.append('| ' + c + ' | ' + ' | '.join(f'{m[k]:.6f}' for k in ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')) + ' |')
    lines += ['', '| 种子 | ST五折平均F1 | 原EST五折平均F1 |', '|---|---:|---:|']
    for seed in ('11', '22', '33'):
        lines.append(f'| {seed} | {metrics["ST-val-weighted"]["train_val_test_seed_means"][seed]["test"]["macro_f1"]:.6f} | {metrics["EST-original-val-weighted"]["train_val_test_seed_means"][seed]["test"]["macro_f1"]:.6f} |')
    counts = paired['counts']
    lines += ['', f'原EST相对固定ST的Macro-F1差为{paired["delta_EST_minus_ST"]["macro_f1"]:.12g}。111条重复种子测试记录中新增正确{counts.get("added_correct", 0)}条、新增错误{counts.get("added_error", 0)}条；仍仅有37位受试者，逐人变化见 [paired_test_changes.jsonl](paired_test_changes.jsonl)。', '',
              f'原EST有{paired["est_zero_eeg_weight_folds"]}/15组的EEG权重为0；只有seed22/fold3使用(0.15,0,0.85)，对应ST选(0,0,1)。因此原EST的名称不能证明EEG在所有折有贡献，本次也未验证新的EEG收益。全部21个ST候选与选择结果保存在 [validation_weight_selections.json](validation_weight_selections.json)。', '',
              '## 隔离边界及保留限制', '',
              '原每折train/val/test人员互斥，固定S/T监督训练没有使用本折原3人val或外折test；原val已经用于历史融合选权，人数极少。若从本折train再切内层val，固定S/T曾监督训练过那些人员，因此本缓存不能当作合法内层OOF，后续不得据此开展未登记选型。', '',
              '37人均为已反复观察的开发人群。新增ST控制和重放不恢复盲测身份，也不产生口音、词汇能力或跨域泛化证据。原S00随机冻结语音编码器、退化注意力、短音频空特征和EEG原幅值/导联适配不确定性继续保留；本阶段不替换E2v2、G1、M2或其他改过S/T的版本。', '',
              f'全文转写与原第一段文本不同的录音共{len(data["changed_fulltext_recordings"])}份；原语音无切片记录三种子共{len(data["speech_zero_segment_records_across_seeds"])}条（并非新增数据）。详情见 [data_and_split_audit.json](data_and_split_audit.json)。', '',
              '## 交付与后续', '',
              '阶段0已完成，可作为阶段1 EEG结构/骨干与阶段3 ST外部小头的原始控制入口。两阶段必须先各自登记条件、参数、训练预算和停止规则，本次没有启动后续训练。', '',
              '- [执行附表](../../plans/TriDep_阶段0_对照与来源锁定执行附表_v1_20261004.md)',
              '- [完成状态与推理核验](status.json)',
              '- [分支逐人概率](branch_probabilities.jsonl)',
              '- [四个控制逐人预测](control_predictions.jsonl)',
              '- [完整各折/种子/合并OOF指标](metrics.json)',
              '- [缓存入口](feature_bundles.json)',
              '- [产物哈希](output_manifest.json)', '',
              '复核入口：`S00/.venv/Scripts/python.exe tridep_ccml/stage0.py --verify`。它重新核对源文件及产物哈希、人员覆盖、全部指标、原EST历史结果、缓存和来源关系，不重训原S/T。', '']
    (OUT / 'TriDep_阶段0_对照与来源锁定完成报告.md').write_text('\n'.join(lines), encoding='utf-8')


def verify():
    status = read(OUT / 'status.json')
    sources = read(OUT / 'source_manifest.json')
    if status['source_manifest_sha256'] != sha(OUT / 'source_manifest.json') or status['output_manifest_sha256'] != sha(OUT / 'output_manifest.json'):
        raise ValueError('Completion manifest changed')
    for item in sources + read(OUT / 'output_manifest.json'):
        if sha(WORK / item['path']) != item['sha256']:
            raise ValueError('Locked source or output changed: ' + item['path'])
    lock = validate_active_cohort(require_data=True)
    folds = read(S00 / lock['fold_file'])
    controls = read_rows(OUT / 'control_predictions.jsonl')
    branches = read_rows(OUT / 'branch_probabilities.jsonl')
    rows = list(controls)
    for r in branches:
        for m in MODALITIES:
            rows.append({**r, 'condition': m, 'p_final': r['p_' + m]})
    calculated = summarize(rows, folds)
    if calculated != read(OUT / 'metrics.json'):
        raise ValueError('Metrics do not recompute')
    selection = {(x['seed'], x['fold']): x for x in read(OUT / 'validation_weight_selections.json')}
    branch_by_key = {(r['seed'], r['outer_fold'], r['subject_id']): r for r in branches}
    for seed in (11, 22, 33):
        old = read_rows(S00 / f'runs/B0-37-fulltext/seed_{seed}/probabilities.jsonl')
        old_by_key = {(r['outer_fold'], r['subject_id']): r for r in old}
        for fold in range(1, 6):
            selected = selection[(seed, fold)]
            val = [old_by_key[(fold, sid)] for sid in folds[f'fold_{fold}']['val']]
            if selected['ST'] != select_st(val) or selected['EST'] != select_weights(val):
                raise ValueError('Validation weights do not recompute')
            for r in controls:
                if r['seed'] != seed or r['outer_fold'] != fold:
                    continue
                src = old_by_key[(fold, r['subject_id'])]
                weights = {'ST-val-weighted': selected['ST']['weights'], 'EST-original-val-weighted': selected['EST']['weights'],
                           'ST-equal': [0., .5, .5], 'EST-equal': [1 / 3] * 3}[r['condition']]
                p = sum(w * src['p_' + m] for w, m in zip(weights, MODALITIES))
                if r['fusion_weights'] != weights or r['p_final'] != p or r['prediction'] != int(p > .5):
                    raise ValueError('Control probability or classification differs from original branches')
            target = OUT / f'features/seed_{seed}/fold_{fold}.npz'
            meta = read(target.with_suffix('.json'))
            if meta['npz_sha256'] != sha(target):
                raise ValueError('New cache metadata differs')
            with np.load(target, allow_pickle=False) as new, np.load(WORK / meta['historical_cache']['path'], allow_pickle=False) as oldcache:
                for k in ('features', 'subject_ids', 'labels', 'splits'):
                    if not np.array_equal(new[k], oldcache[k]):
                        raise ValueError('New cache differs from original supervised representation')
                for i, sid in enumerate(new['subject_ids']):
                    r = branch_by_key[(seed, fold, str(sid))]
                    if any(new['probabilities'][i, j] != r['p_' + m] for j, m in enumerate(MODALITIES)):
                        raise ValueError('Probability bundle and records differ')
    result = {'passed': True, 'source_hashes_verified': len(sources), 'output_hashes_verified': len(read(OUT / 'output_manifest.json')),
              'control_records': len(controls), 'branch_records': len(branches), 'metrics_recomputed': True,
              'validation_selections_recomputed': 15, 'representation_bundles_verified': 15, 'new_fits': 0}
    write(OUT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    try:
        verify() if args.verify else run()
    except Exception as exc:
        write(OUT / 'failure.json', {'passed': False, 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
