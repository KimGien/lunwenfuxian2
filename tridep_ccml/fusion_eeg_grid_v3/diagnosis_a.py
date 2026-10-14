"""Phase A: read-only diagnosis of sealed v2 predictions, features and trajectories.

No torch/model imports, model forwards, weight fusion, fitting, source mutation, or
new-candidate test reads. All outputs are confined to the v3 phase_a directory.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'tridep_ccml/experiments/Fusion-EEG15-Methods5-v2-37-20261013'
OLD_REPORT = ROOT / 'output/analysis/fusion_eeg15_methods5_v2_20261013'
BASE = ROOT / 'output/analysis/eeg_tridep_fusion_v1_20261013'
STAGE0 = ROOT / 'output/analysis/tridep_stage0_20261004'
OUT = ROOT / 'output/analysis/fusion_eeg15_methods5_v3_20261014/phase_a'
PAUSE = ROOT / 'tridep_ccml/experiments/Fusion-EEG15-Methods5-v3-37-20261014/control/user_pause.json'
PLAN = ROOT / 'output/plans/EEG十五版本与五种融合方法后续计划_v3_20261014.md'
EEG = [f'E{i:02d}' for i in range(14)] + ['E15']
METHODS = ['CCML', 'LMF', 'MMDynamics', 'QMF', 'ECML']
SEEDS = [11, 22, 33]
SOURCES: dict[str, dict] = {}
OUTPUTS: list[dict] = []


def check_pause():
    if PAUSE.exists():
        marker = json.loads(PAUSE.read_text('utf-8-sig'))
        assert set(marker) == {'paused'} and isinstance(marker['paused'], bool)
        if marker['paused']:
            raise RuntimeError('v3 user pause is active; no new diagnostic work')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def desc(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def source(path, expected=None):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT) and not path.is_relative_to(OUT)
    key = path.relative_to(ROOT).as_posix()
    if key not in SOURCES:
        SOURCES[key] = desc(path)
    if expected is not None:
        assert SOURCES[key]['sha256'] == expected, f'Source SHA mismatch: {path}'
    return path


def bound(record):
    return source(ROOT / record['path'], record['sha256'])


def read(path):
    return json.loads(source(path).read_text('utf-8-sig'))


def rows(path):
    return [json.loads(x) for x in source(path).read_text('utf-8-sig').splitlines() if x.strip()]


def write(name, payload):
    check_pause()
    path = (OUT / name).resolve()
    assert path.is_relative_to(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    with temp.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    OUTPUTS.append(desc(path))


def save(name, value):
    write(name, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def table(name, records):
    assert records, name
    stream = io.StringIO(newline='')
    keys = list(dict.fromkeys(k for r in records for k in r))
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for r in records:
        writer.writerow({k: ('null' if v is None else json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    write(name, stream.getvalue().encode('utf-8-sig'))


def avg(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def flags(prob, label):
    assert math.isfinite(prob) and 0 <= prob <= 1 and label in (0, 1)
    prediction = int(prob > .5)
    return {'prediction': prediction, 'correct': prediction == label,
            'mdd_missed': label == 1 and prediction == 0,
            'hc_false_alarm': label == 0 and prediction == 1,
            'boundary': .4 <= prob <= .6,
            'high_confidence_error': max(prob, 1-prob) >= .9 and prediction != label}


def macro_f1(records):
    scores = []
    for label in (0, 1):
        tp = sum(r['label'] == label and int(r['p'] > .5) == label for r in records)
        fp = sum(r['label'] != label and int(r['p'] > .5) == label for r in records)
        fn = sum(r['label'] == label and int(r['p'] > .5) != label for r in records)
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(scores)/2


def load_features(branches):
    exposures, bundles, mono, cache_meta = [], [], {}, {}
    for group in ['ST', *EEG]:
        for seed in SEEDS:
            for fold in range(1, 6):
                check_pause()
                path = OLD / 'features' / group / f'seed_{seed}' / f'fold_{fold}.npz'
                meta = read(path.with_suffix('.json'))
                bound(meta['cache'])
                with np.load(path, allow_pickle=False) as z:
                    x, ids, y, splits = [z[k].copy() for k in ('features', 'subject_ids', 'labels', 'splits')]
                if group != 'ST':
                    x = x[:, None, :]
                assert x.shape == (37, 2 if group == 'ST' else 1, 128)
                assert np.isfinite(x).all() and len(set(ids.tolist())) == 37 and int(y.sum()) == 16
                lookup = {str(s): i for i, s in enumerate(ids)}
                roles = meta['roles']
                index = {r: np.asarray([lookup[s] for s in people]) for r, people in roles.items()}
                assert set(roles) == {'train', 'val', 'test'}
                assert sorted(i for ix in index.values() for i in ix.tolist()) == list(range(37))
                assert all(all(str(splits[i]) == r for i in ix) for r, ix in index.items())
                mean = x[index['train']].astype(np.float64).mean(0)
                std = x[index['train']].astype(np.float64).std(0)
                norm = ((x.astype(np.float64)-mean)/np.maximum(std, 1e-6)).astype(np.float32)
                assert np.isfinite(norm).all()
                cache_meta[(group, seed, fold)] = {'meta': meta, 'mean': mean, 'std': std,
                                                  'ids': ids.tolist(), 'labels': y.tolist(), 'splits': splits.tolist()}
                completion = None
                if group not in ('ST', 'E00'):
                    completion = read(bound(meta['source_unit']['completion']))
                    assert completion['passed'] and completion['binding']['checkpoint'] == meta['source_unit']['checkpoint']
                    batch_rows = []
                    for b in completion['batches']:
                        batch = read(bound(b))
                        assert batch['binding']['checkpoint'] == meta['source_unit']['checkpoint']
                        batch_rows.extend(batch['rows'])
                    assert len(batch_rows) == 37 and len({r['subject_id'] for r in batch_rows}) == 37
                    for r in batch_rows:
                        i = lookup[r['subject_id']]
                        assert r['role'] == splits[i] and r['label'] == y[i]
                        mono[(group, seed, fold, r['subject_id'])] = float(r['p_mdd'])
                for v, modality in enumerate(['S', 'T'] if group == 'ST' else [group]):
                    record = {'modality': modality, 'seed': seed, 'fold': fold,
                              'train_zero_std_dimensions': int((std[v] == 0).sum()),
                              'train_std_le_1e6_dimensions': int((std[v] <= 1e-6).sum()),
                              'train_constant_dimensions': np.where(std[v] <= 1e-6)[0].tolist(),
                              'cache': meta['cache'], 'cache_metadata': desc(path.with_suffix('.json')),
                              'parent_source': meta.get('source_unit', meta.get('parent_checkpoints')),
                              'pooling': meta.get('pooling', 'stage0 original final Linear input'),
                              'known_selected_window_count': 60 if completion and completion.get('all60_windows_per_subject') else None,
                              'valid_quality_window_count': None, 'speech_recording_coverage': None,
                              'quality_missing_reason': 'Not present as per-subject quality metadata in the sealed feature/selected-prediction sources; fixed selected windows are not quality-valid windows.'}
                    for role, ix in index.items():
                        record[role + '_max_abs_n0'] = float(np.abs(norm[ix, v]).max())
                    bundles.append(record)
                    for i, sid in enumerate(ids.tolist()):
                        branch = branches[(seed, fold, sid)]
                        assert int(y[i]) == branch['label'] and str(splits[i]) == branch['split']
                        if group in ('ST', 'E00'):
                            key = {'S': 'p_speech', 'T': 'p_text', 'E00': 'p_eeg'}[modality]
                            mono[(modality, seed, fold, sid)] = float(branch[key])
                        absnorm = np.abs(norm[i, v]); maxdim = int(absnorm.argmax())
                        exposures.append({'modality': modality, 'seed': seed, 'fold': fold, 'subject_id': sid,
                                          'role': str(splits[i]), 'label': int(y[i]),
                                          'max_abs_n0': float(absnorm[maxdim]), 'max_dimension_zero_based': maxdim,
                                          'max_raw_value': float(x[i, v, maxdim]), 'max_train_mean': float(mean[v, maxdim]),
                                          'max_train_std': float(std[v, maxdim]),
                                          'n0_abs_gt100_coordinates': int((absnorm > 100).sum()),
                                          'n0_abs_gt1000_coordinates': int((absnorm > 1000).sum()),
                                          'nonzero_on_n1_mask_coordinates': int(((std[v] <= 1e-6) & (x[i, v] != mean[v])).sum()),
                                          'train_zero_std_dimensions': record['train_zero_std_dimensions'],
                                          'train_std_le_1e6_dimensions': record['train_std_le_1e6_dimensions'],
                                          'cache_path': meta['cache']['path'], 'cache_sha256': meta['cache']['sha256']})
    assert len(bundles) == 255 and len(exposures) == 9435 and len(mono) == 9435
    return exposures, bundles, mono, cache_meta


def main():
    started = time.perf_counter()
    check_pause()
    if (OUT / 'complete.json').exists():
        done = json.loads((OUT / 'complete.json').read_text('utf-8'))
        assert done['script']['sha256'] == sha(__file__), 'Completed diagnostic source changed; preserve prior completion'
        for r in done['outputs']:
            assert sha(ROOT / r['path']) == r['sha256'], f'Diagnostic output changed: {r["path"]}'
        manifest = json.loads((OUT / 'source_manifest.json').read_text('utf-8'))
        for r in manifest['sources']:
            check_pause()
            assert sha(ROOT / r['path']) == r['sha256'], f'Diagnostic input changed: {r["path"]}'
        print(json.dumps({'already_complete': True, 'counts': done['counts']}, ensure_ascii=False))
        return done
    source(PLAN)
    revision = read(PLAN.with_name(PLAN.stem + '_修订登记.json'))
    assert sha(PLAN) == revision['plan']['sha256']
    source(OLD_REPORT / 'diagnosis_20261013/诊断结论与验收勘误.txt')
    source(ROOT / 'tridep_ccml/fusion_eeg_grid_v2/training.py')
    old_scale = read(OLD_REPORT / 'diagnosis_20261013/feature_scale_diagnosis.json')
    source(OLD_REPORT / 'diagnosis_20261013/qmf_source_parity_diagnosis.json')
    old_summary = read(OLD_REPORT / 'results.json')
    base_results = read(BASE / 'results.json')
    # original_E_A is the precise 72.550% fixed baseline. historical_EST is a
    # different 72.418% historical condition and must not be substituted.
    assert base_results['baselines']['original_E_A']['primary']['macro_f1'] == .7255026455026455
    baseline = {(r['seed'], r['fold'], r['subject_id']): r for r in rows(BASE / 'baseline_predictions.jsonl') if r['unit'] == 'original_E_A'}
    branches = {(r['seed'], r['outer_fold'], r['subject_id']): r for r in rows(STAGE0 / 'branch_probabilities.jsonl')}
    assert len(baseline) == len(branches) == 555
    base_test = [r for r in baseline.values() if r['role'] == 'test']
    assert len(base_test) == 111 and len({r['subject_id'] for r in base_test}) == 37
    primary = avg(avg(macro_f1([r for r in base_test if r['seed'] == s and r['fold'] == f])
                      for f in range(1, 6)) for s in SEEDS)
    assert abs(primary-.7255026455026455) < 1e-15
    exposures, bundles, mono, cache_meta = load_features(branches)
    assert sum(r['train_zero_std_dimensions'] for r in bundles) == old_scale['global']['all_unique_modality_bundles']['raw_std_equal_zero'] == 2781
    exposure = {(r['modality'], r['seed'], r['fold'], r['subject_id']): r for r in exposures}
    bundle = {(r['modality'], r['seed'], r['fold']): r for r in bundles}
    fits, testrows, lookup = [], [], {}
    evaluation_count = 0
    for eeg in ['ST', *EEG]:
        for method in METHODS:
            for seed in SEEDS:
                for fold in range(1, 6):
                    check_pause()
                    p = OLD / 'runs' / f'{eeg}-{method}' / f'seed_{seed}' / f'fold_{fold}'
                    trajectory = read(p / 'trajectory.json'); m = read(p / 'metadata.json'); e = read(p / 'evaluation.json')
                    complete = read(p / 'complete.json')
                    identity = {'eeg': eeg, 'method': method, 'seed': seed, 'fold': fold}
                    assert m['identity'] == e['identity'] == identity
                    assert complete['best'] == e['checkpoint']
                    best = min(trajectory, key=lambda r: r['val_nll'])
                    assert best['epoch'] == m['best_epoch'] and len(trajectory) == m['epochs']
                    assert trajectory[-1]['cumulative_updates'] == m['updates']
                    assert 60 <= len(trajectory) <= 120
                    assert [r['epoch'] for r in trajectory] == list(range(1, len(trajectory)+1))
                    st = cache_meta[('ST', seed, fold)]
                    ordered_modalities = ['S', 'T'] if eeg == 'ST' else [eeg, 'S', 'T']
                    cs = [st] if eeg == 'ST' else [cache_meta[(eeg, seed, fold)], st]
                    means = np.concatenate([c['mean'] for c in cs], axis=0)
                    stds = np.concatenate([c['std'] for c in cs], axis=0)
                    binding = m['input_binding']
                    assert e['input_binding'] == binding
                    assert binding['subject_ids'] == st['ids'] and binding['labels'] == st['labels'] and binding['splits'] == st['splits']
                    assert np.array_equal(means, np.asarray(binding['normalization']['mean']))
                    assert np.array_equal(np.maximum(stds, 1e-6), np.asarray(binding['normalization']['std']))
                    for b in binding['sources']:
                        bound(b)
                    tail = avg(r['val_nll'] for r in trajectory[-10:])-avg(r['val_nll'] for r in trajectory[-20:-10])
                    fit = {**identity, 'epochs': m['epochs'], 'updates': m['updates'], 'best_epoch': m['best_epoch'],
                           'stopping_reason': m['stopping_reason'], 'final_patience_count': trajectory[-1]['stale'],
                           'best_val_nll': best['val_nll'], 'final_val_nll': trajectory[-1]['val_nll'],
                           'last10_minus_previous10_val_nll': tail,
                           'first10_train_objective': avg(r['loss'] for r in trajectory[:10]),
                           'last10_train_objective': avg(r['loss'] for r in trajectory[-10:]),
                           'last_components': trajectory[-1]['components'], 'parameters': m['parameters'],
                           'capped_best_ge110_and_tail_improving': m['epochs'] == 120 and m['best_epoch'] >= 110 and tail < -1e-4,
                           'checkpoint': e['checkpoint'], 'trajectory': desc(p / 'trajectory.json'),
                           'qmf_source_status': 'Q0_old_local_variant' if method == 'QMF' else 'unchanged_v2_method',
                           'val_input_abs_max_n0': max(bundle[(mod, seed, fold)]['val_max_abs_n0'] for mod in ordered_modalities),
                           'test_input_abs_max_n0': max(bundle[(mod, seed, fold)]['test_max_abs_n0'] for mod in ordered_modalities)}
                    for role in ['train', 'val', 'test']:
                        for metric in ['macro_f1', 'nll']:
                            fit[f'selected_{role}_{metric}'] = e['metrics'][role][metric]
                    fits.append(fit)
                    assert len(e['rows']) == 37 and len({r['subject_id'] for r in e['rows']}) == 37
                    evaluation_count += len(e['rows'])
                    for r in e['rows']:
                        old = baseline[(seed, fold, r['subject_id'])]
                        assert r['role'] == old['role'] and r['label'] == old['label']
                        assert r['prediction'] == int(r['p_mdd'] > .5)
                        if r['role'] == 'test':
                            testrows.append(r)
                            key = (eeg, method, seed, fold, r['subject_id'])
                            assert key not in lookup
                            lookup[key] = r
    assert len(fits) == 1200 and len(testrows) == 8880 and evaluation_count == 44400
    fitmap = {(r['eeg'], r['method'], r['seed'], r['fold']): r for r in fits}
    baseline_ledger = [{**r, **flags(r['p'], r['label']), 'baseline_unit': 'original_E_A'} for r in base_test]
    ledger = []
    paired = defaultdict(list)
    for r in testrows:
        eeg, method, seed, fold, sid = (r[k] for k in ['eeg', 'method', 'seed', 'fold', 'subject_id'])
        y = r['label']; p = r['p_mdd']; b = baseline[(seed, fold, sid)]
        f = fitmap[(eeg, method, seed, fold)]
        q = {**r, **flags(p, y), 'baseline_probability': b['p']}
        q.update({'baseline_'+k: v for k, v in flags(b['p'], y).items()})
        mods = ['S', 'T'] if eeg == 'ST' else [eeg, 'S', 'T']
        for name, mod in [('E', None if eeg == 'ST' else eeg), ('S', 'S'), ('T', 'T')]:
            prob = mono[(mod, seed, fold, sid)] if mod else None
            q[name+'_probability'] = prob
            q[name+'_correct'] = None if prob is None else int(prob > .5) == y
            q[name+'_prediction'] = None if prob is None else int(prob > .5)
        q['ST_both_original_branches_wrong'] = not q['S_correct'] and not q['T_correct']
        q['E_correct_when_ST_both_wrong'] = None if eeg == 'ST' else q['E_correct'] and q['ST_both_original_branches_wrong']
        q['modalities_disagree'] = len({int(mono[(mod, seed, fold, sid)] > .5) for mod in mods}) > 1
        q['all_original_branches_correct_but_fusion_wrong'] = all(int(mono[(mod, seed, fold, sid)] > .5) == y for mod in mods) and not q['correct']
        q['subject_max_abs_n0'] = max(exposure[(mod, seed, fold, sid)]['max_abs_n0'] for mod in mods)
        q['subject_abs_gt100'] = q['subject_max_abs_n0'] > 100
        q['subject_abs_gt1000'] = q['subject_max_abs_n0'] > 1000
        q['subject_any_nonzero_on_n1_mask'] = any(exposure[(mod, seed, fold, sid)]['nonzero_on_n1_mask_coordinates'] > 0 for mod in mods)
        q.update({k: f[k] for k in ['best_epoch', 'epochs', 'updates', 'final_patience_count', 'stopping_reason', 'val_input_abs_max_n0', 'last10_minus_previous10_val_nll']})
        q['quality_valid_windows'] = None; q['speech_recording_coverage'] = None; q['missing_quality_metadata'] = True
        for ref in ['baseline', 'same_method_ST', 'same_method_E00']:
            refp = b['p'] if ref == 'baseline' else lookup[('ST' if ref == 'same_method_ST' else 'E00', method, seed, fold, sid)]['p_mdd']
            rf = flags(refp, y)
            q[ref+'_probability'] = refp
            q[ref+'_corrected'] = q['correct'] and not rf['correct']
            q[ref+'_new_error'] = not q['correct'] and rf['correct']
            item = {'subject_id': sid, 'seed': seed, 'fold': fold, 'label': y,
                    'corrected': q[ref+'_corrected'], 'new_error': q[ref+'_new_error']}
            categories = {'all': True, 'MDD': y == 1, 'HC': y == 0,
                          'reference_mdd_missed': rf['mdd_missed'], 'reference_hc_false_alarm': rf['hc_false_alarm'],
                          'reference_boundary': rf['boundary'], 'reference_high_confidence_error': rf['high_confidence_error'],
                          'candidate_mdd_missed': q['mdd_missed'], 'candidate_hc_false_alarm': q['hc_false_alarm'],
                          'candidate_boundary': q['boundary'], 'candidate_high_confidence_error': q['high_confidence_error'],
                          'modalities_disagree': q['modalities_disagree'], 'ST_both_original_branches_wrong': q['ST_both_original_branches_wrong'],
                          'subject_abs_gt100': q['subject_abs_gt100'], 'subject_abs_le100': not q['subject_abs_gt100'],
                          'subject_any_nonzero_on_n1_mask': q['subject_any_nonzero_on_n1_mask']}
            for category, active in categories.items():
                if active:
                    paired[(eeg, method, ref, category)].append(item)
        ledger.append(q)
    paired_table = []
    for key, rs in sorted(paired.items()):
        corrected = [r for r in rs if r['corrected']]; added = [r for r in rs if r['new_error']]
        paired_table.append(dict(zip(['eeg', 'method', 'reference', 'error_type'], key),
                                 paired_seed_person_records=len(rs), unique_people=len({r['subject_id'] for r in rs}),
                                 corrected_records=len(corrected), new_error_records=len(added), net_corrected_records=len(corrected)-len(added),
                                 corrected_unique_people=len({r['subject_id'] for r in corrected}),
                                 new_error_unique_people=len({r['subject_id'] for r in added}),
                                 corrected_keys=[f"{r['seed']}/{r['fold']}/{r['subject_id']}" for r in corrected],
                                 new_error_keys=[f"{r['seed']}/{r['fold']}/{r['subject_id']}" for r in added]))
    people = []
    for sid in sorted({r['subject_id'] for r in base_test}):
        br = [r for r in baseline_ledger if r['subject_id'] == sid]
        rs = [r for r in ledger if r['subject_id'] == sid]
        assert len(br) == 3 and len(rs) == 240
        people.append({'subject_id': sid, 'label': br[0]['label'], 'baseline_seed_probabilities': {str(r['seed']): r['p'] for r in br},
                       'baseline_error_seeds': [r['seed'] for r in br if not r['correct']],
                       'baseline_boundary_seeds': [r['seed'] for r in br if r['boundary']],
                       'baseline_high_confidence_error_seeds': [r['seed'] for r in br if r['high_confidence_error']],
                       'v2_condition_seed_records': len(rs), 'v2_wrong_records': sum(not r['correct'] for r in rs),
                       'v2_corrected_vs_baseline_records': sum(r['baseline_corrected'] for r in rs),
                       'v2_new_error_vs_baseline_records': sum(r['baseline_new_error'] for r in rs),
                       'source_exposure_unique_modality_seed_records_gt100': sum(r['max_abs_n0'] > 100 for r in exposures if r['subject_id'] == sid and r['role'] == 'test'),
                       'quality_valid_windows': None, 'speech_recording_coverage': None,
                       'interpretation': '37 unique people; 240 condition/seed records are repeated model observations, not independent participants.'})
    by_method = {}
    for method in METHODS:
        rs = [r for r in fits if r['method'] == method]
        by_method[method] = {'fits': len(rs), 'best_epoch_median': float(np.median([r['best_epoch'] for r in rs])),
                             'best_epoch_120': sum(r['best_epoch'] == 120 for r in rs),
                             'stopping_reasons': dict(Counter(r['stopping_reason'] for r in rs)),
                             'capped_best_ge110_and_tail_improving': sum(r['capped_best_ge110_and_tail_improving'] for r in rs),
                             **{f'mean_selected_{role}_{metric}': avg(r[f'selected_{role}_{metric}'] for r in rs)
                                for role in ['train', 'val', 'test'] for metric in ['macro_f1', 'nll']}}
    associations = []
    for eeg in ['ST', *EEG]:
        for method in METHODS:
            rs = [r for r in ledger if r['eeg'] == eeg and r['method'] == method]
            for exposed in [False, True]:
                selected = [r for r in rs if r['subject_abs_gt100'] == exposed]
                associations.append({'eeg': eeg, 'method': method, 'subject_abs_gt100': exposed,
                                     'records': len(selected), 'unique_people': len({r['subject_id'] for r in selected}),
                                     'errors': sum(not r['correct'] for r in selected),
                                     'error_rate': avg(not r['correct'] for r in selected),
                                     'baseline_corrected': sum(r['baseline_corrected'] for r in selected),
                                     'baseline_new_errors': sum(r['baseline_new_error'] for r in selected),
                                     'high_confidence_errors': sum(r['high_confidence_error'] for r in selected),
                                     'interpretation': 'Descriptive co-occurrence in reused participants; no causal or significance claim. >100 was inherited from the prior diagnostic, not tuned to outcomes.'})
    hypotheses = [
        {'观察事实': '255个模态/seed/fold中共有2781个train零标准差维；未见输入的N0最大绝对值由缓存直接复算。',
         '可能机制': '训练常量坐标在未见人中非零，经1e-6分母放大并影响融合层；仅有限性检查不足。',
         '隔离实验': 'B中其余四法N0对N1及QMF完整N0Q0/N1Q0/N0Q1/N1Q1；先固定lr=1e-3与120轮，仅按train定义掩码。',
         '否证条件': 'N1虽消除掩码维尺度异常，但配对NLL/F1及纠错不改善，则不能归因主性能差距于此缺陷；不排除其他分布差异。'},
        {'观察事实': '逐人账本记录原分支正确而融合失败；原所选融合头训练/验证/测试差距见training_selection_summary.json。',
         '可能机制': '仅26/27人上重新学习原分类器已学判别方向，产生二阶段泛化差距；尚非因果结论。',
         '隔离实验': '本轮既定接口定位：原128维分布、原分类层概率、融合概率逐人配对；结合EST相对共享ST纠错与新增错。不得自动增加新结构训练。',
         '否证条件': '若融合损失未学习或多数错误已由原分支共同犯下，则“重学头丢失原有信息”不足解释；即使接口丢失存在也不直接证明正则化修复有效。'},
        {'观察事实': f"旧CCML有{by_method['CCML']['capped_best_ge110_and_tail_improving']}/240、ECML有{by_method['ECML']['capped_best_ge110_and_tail_improving']}/240达到120且best≥110、末10轮NLL比前10轮下降超过1e-4。",
         '可能机制': '120轮上限可能截断部分验证改善；val仅3人，验证改善不保证测试F1上升。',
         '隔离实验': 'D对CCML/ECML全部480身份按已锁定基础lr接续，合格latest继承耐心，121–240固定0.1η；早停身份保留等价端点。',
         '否证条件': '修正调参后没有合格晚期轨迹或追加未改善已登记指标；不能用此理由复活早停、延长其他三法或无限扩轮。'},
        {'观察事实': 'QMF旧诊断确认非零历史排序损失和梯度与已绑定RGBD作者提交不同；旧240身份属于Q0变体。',
         '可能机制': '排序目标差异改变质量学习，但是否造成较差F1尚未隔离。',
         '隔离实验': 'B锁定作者Q1先核值/梯度与恢复，再用四格隔离排序、标准化及其交互。',
         '否证条件': '即使Q1来源验收通过，若配对指标无改善，则不能声称旧来源差异解释低分；仍保留正确实现与阴性结果。'}]
    # Cross-check the sealed aggregate without using any new-candidate results.
    for summary in old_summary['summary']:
        rs = [r for r in fits if r['eeg'] == summary['eeg'] and r['method'] == summary['method']]
        assert len(rs) == 15 and abs(avg(r['selected_test_macro_f1'] for r in rs)-summary['macro_f1']) < 1e-14
    counts = {'old_fits': len(fits), 'old_prediction_rows_checked_all_roles': evaluation_count,
              'oof_ledger_rows': len(ledger), 'conditions': 80, 'unique_people': len(people),
              'baseline_oof_rows': 111, 'cache_npz_files': 240, 'unique_modality_seed_fold_bundles': len(bundles),
              'source_feature_person_rows_all_roles': len(exposures), 'train_zero_std_dimensions': 2781,
              'new_fits': 0, 'new_model_forwards': 0, 'optimizer_updates': 0}
    table('baseline_error_ledger.csv', baseline_ledger)
    table('oof_error_ledger.csv', ledger)
    table('subject_summary_37.csv', people)
    table('error_type_corrections.csv', paired_table)
    table('input_exposure_by_subject.csv', exposures)
    table('input_error_associations.csv', associations)
    table('training_selection_by_fit.csv', fits)
    table('mechanism_hypotheses.csv', hypotheses)
    save('input_bundle_audit.json', {'bundles': bundles, 'quality_metadata_note': 'Unknown per-person valid-window quality, recording coverage and missingness are null, never inferred from classification error or from the fixed 60 selected windows.'})
    save('training_selection_summary.json', {'by_method': by_method,
         'limitations': ['The 240 fits per method are not independent people.', 'Objectives differ across methods and annealing stages.', 'Old selected checkpoints only; no new test inference or checkpoint selection.']})
    maxima = {role: max(r['max_abs_n0'] for r in exposures if r['role'] == role) for role in ['train', 'val', 'test']}
    summary = {'schema': 'fusion_eeg_grid_v3_phase_a_v1', 'scope': 'post_hoc_read_only_sealed_v2',
               'counts': counts, 'baseline': {'unit': 'original_E_A', 'macro_f1_recomputed': primary,
                 'probabilities': 'Direct archived baseline_predictions.jsonl values, no weighted recomputation.',
                 'warning': 'The different historical_EST unit has macro_f1=0.7241798941798943 and is not the locked baseline.'},
               'n0_max_abs_by_role': maxima,
               'fits_exposed_abs_gt100': {role: sum(r[role+'_input_abs_max_n0'] > 100 for r in fits) for role in ['val', 'test']},
               'baseline_errors_seed_records': sum(not r['correct'] for r in baseline_ledger),
               'baseline_error_unique_people': len({r['subject_id'] for r in baseline_ledger if not r['correct']}),
               'phase_b_c_d_results_read': False, 'hypotheses_are_causal_findings': False,
               'quality_metadata_status': 'Per-subject quality-valid window counts, speech coverage and true missingness were not available in inspected bound feature/prediction artifacts; null retained.',
               'limitations': ['A uses previously observed outer-fold outcomes and cannot regain blinded-test status.',
                              '111 seed-person predictions represent 37 people; 15 EEG branches are not independent datasets.',
                              'Feature exposure is an association, not a causal attribution or broken-model count.',
                              'This phase validates supplied prediction/cache/trajectory bindings, not a new model reload or complete upstream checkpoint re-audit.',
                              'Selected-window count is not a count of artifact-free or clinically valid EEG windows.',
                              'Known old QMF parity failure remains open until phase B Q1; no old certificate is reused as repair acceptance.']}
    save('summary.json', summary)
    report = [
        '阶段A：旧结果错误账本和机制假说（只读诊断）', '',
        f'已核查1,200份旧拟合、44,400条已存train/val/test预测；完整外折账本8,880条，80条件各111条，覆盖37名唯一受试者。',
        f'主基线直接读取原已存 original_E_A 概率，独立复算五折再三种子 Macro-F1={primary:.16f}；未运行固定权重融合。',
        '名称 historical_EST 指向另一份72.418%历史条件，不能误用为72.550%基线。',
        f'240份缓存拆分为255个去重模态/seed/fold；train零标准差维共2781。N0绝对值最大：{maxima}。',
        '输入暴露和错误共现只作描述，不据此证明N1或Q1一定提高F1。原质量元数据缺失处为null，不能把60个选中窗口当作60个质量有效窗。',
        '纠错表同时配对历史基线、同方法共享ST与E00；错误标签允许重叠，各类计数不能相加作总人数。',
        '已登记四条机制假说和否证条件，见mechanism_hypotheses.csv。隔离试验仅指v3已经登记的后续阶段，没有新增模型、推理或训练。',
        '本阶段不读取任何v3新候选测试预测。没有模型forward、参数更新、重新选检查点、搜索阈值或融合权重。',
        '训练轨迹、选模、尺度来源和逐人配对均有SHA来源表。该诊断不能替代B至E独立验收，不能恢复盲测身份。',
        '', '产物：', *[r['path'].split('/')[-1] for r in OUTPUTS], '']
    write('阶段A诊断报告.txt', '\n'.join(report).encode('utf-8'))
    # Rehash every consumed source after computation to detect concurrent mutation.
    for record in list(SOURCES.values()):
        check_pause()
        assert sha(ROOT / record['path']) == record['sha256'], f'Source changed while diagnosing: {record["path"]}'
    save('source_manifest.json', {'sources': sorted(SOURCES.values(), key=lambda r: r['path']),
                                'consumed_source_files': len(SOURCES), 'all_sources_rehashed_unchanged': True})
    outputs = list(OUTPUTS)
    done = {'schema': 'fusion_eeg_grid_v3_phase_a_complete_v1', 'passed': True,
            'at': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'counts': counts,
            'script': desc(__file__), 'plan': desc(PLAN), 'outputs': outputs,
            'wall_seconds': time.perf_counter()-started, 'training_seconds': 0,
            'independent_review_pending': True,
            'acceptance': {'source_sha_checks': True, 'cohort_and_role_alignment': True,
                           'all_1200_v2_normalization_bindings_recomputed': True,
                           'all_1200_trajectory_selected_epochs_checked': True,
                           'all_80_old_condition_primary_f1_matches_saved_report': True,
                           'baseline_primary_recomputed': True, 'old_sources_unchanged': True,
                           'no_new_candidate_test_reads': True, 'no_model_calls': True}}
    save('complete.json', done)
    print(json.dumps({'complete': str(OUT / 'complete.json'), 'counts': counts,
                      'source_files': len(SOURCES), 'seconds': done['wall_seconds']}, ensure_ascii=False))
    return done


if __name__ == '__main__':
    main()
