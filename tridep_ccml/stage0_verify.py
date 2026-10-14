"""Independently audit the frozen stage-0 evidence without fitting or inference."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import sklearn
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             recall_score, roc_auc_score)

WORK = Path(__file__).resolve().parent.parent
S00 = WORK / 'S00'
OUT = WORK / 'output/analysis/tridep_stage0_20261004'
SCRIPT = Path(__file__).resolve()
MANIFEST = OUT / 'supplementary_source_manifest.json'
RESULT = OUT / 'independent_verification.json'
REPORT = OUT / '阶段0补充来源与独立核验.md'
CONDITIONS = ('ST-val-weighted', 'EST-original-val-weighted', 'ST-equal',
              'EST-equal', 'eeg', 'speech', 'text')
KEYS = ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')
SEEDS = (11, 22, 33)
SPLITS = ('train', 'val', 'test')
TOLERANCE = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(WORK).as_posix()


def binding(path):
    return {'path': relative(path), 'sha256': digest(path), 'bytes': Path(path).stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class Sources:
    def __init__(self):
        self.items = {}

    def add(self, path, role, expected=None):
        value = binding(path)
        if expected is not None:
            require(value['sha256'] == expected, 'Pre-existing source hash mismatch: ' + value['path'])
        item = self.items.setdefault(value['path'], {**value, 'roles': [], 'expected_sha256': expected})
        require(item['sha256'] == value['sha256'], 'Source changed during audit: ' + value['path'])
        if expected is not None:
            item['expected_sha256'] = expected
        if role not in item['roles']:
            item['roles'].append(role)
        return value

    def ordered(self):
        return [self.items[key] for key in sorted(self.items)]


def calculate_metrics(selected):
    y = np.asarray([row['label'] for row in selected], dtype=np.int64)
    p = np.asarray([row['p_final'] for row in selected], dtype=np.float64)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probability')
    prediction = (p > 0.5).astype(np.int64)
    return {'macro_f1': float(f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'mdd_recall': float(recall_score(y, prediction, pos_label=1, zero_division=0)),
            'auc': float(roc_auc_score(y, p)),
            'nll': float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
            'brier': float(brier_score_loss(y, p)),
            'confusion_hc_mdd': confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def audit_metrics(folds, subjects):
    stored = read(OUT / 'metrics.json')
    controls = rows(OUT / 'control_predictions.jsonl')
    branches = rows(OUT / 'branch_probabilities.jsonl')
    combined = list(controls)
    for row in branches:
        for modality in ('eeg', 'speech', 'text'):
            combined.append({**row, 'condition': modality, 'p_final': row['p_' + modality]})
    result = {}
    maximum = 0.0
    comparisons = 0
    groups = 0

    def compare(actual, expected):
        nonlocal maximum, comparisons
        for key in KEYS:
            delta = abs(actual[key] - expected[key])
            maximum = max(maximum, delta)
            comparisons += 1
            require(delta <= TOLERANCE, f'Independent metric mismatch {key}: {delta}')
        if 'confusion_hc_mdd' in actual and 'confusion_hc_mdd' in expected:
            require(actual['confusion_hc_mdd'] == expected['confusion_hc_mdd'], 'Confusion matrix mismatch')

    for condition in CONDITIONS:
        selected = [row for row in combined if row['condition'] == condition]
        fold_values = []
        means = {}
        pooled = {}
        stored_by_group = {(r['seed'], r['fold'], r['split']): r for r in stored[condition]['fold_metrics']}
        for seed in SEEDS:
            for fold in range(1, 6):
                for split in SPLITS:
                    group = [row for row in selected if (row['seed'], row['outer_fold'], row['split']) == (seed, fold, split)]
                    expected_ids = set(folds[f'fold_{fold}'][split])
                    require(len(group) == len(expected_ids) and {r['subject_id'] for r in group} == expected_ids,
                            'Independent split coverage mismatch')
                    require(all(r['label'] == subjects[r['subject_id']]['label'] for r in group), 'Prediction label mismatch')
                    value = {'seed': seed, 'fold': fold, 'split': split, **calculate_metrics(group)}
                    compare(value, stored_by_group[(seed, fold, split)])
                    fold_values.append(value)
                    groups += 1
            means[str(seed)] = {split: {key: float(np.mean([r[key] for r in fold_values if r['seed'] == seed and r['split'] == split]))
                                      for key in KEYS} for split in SPLITS}
            for split in SPLITS:
                compare(means[str(seed)][split], stored[condition]['train_val_test_seed_means'][str(seed)][split])
            group = [row for row in selected if row['seed'] == seed and row['split'] == 'test']
            require(len(group) == 37 and len({r['subject_id'] for r in group}) == 37, 'OOF subject coverage mismatch')
            pooled[str(seed)] = calculate_metrics(group)
            compare(pooled[str(seed)], stored[condition]['pooled_oof_per_seed'][str(seed)])
        main_test = {key: float(np.mean([means[str(seed)]['test'][key] for seed in SEEDS])) for key in KEYS}
        compare(main_test, stored[condition]['main_test'])
        result[condition] = {'fold_metrics': fold_values, 'train_val_test_seed_means': means,
                             'pooled_oof_per_seed': pooled, 'main_test': main_test}
    require(groups == 315, 'Expected 315 independent metric groups')
    return {'passed': True, 'library': 'scikit-learn', 'sklearn_version': sklearn.__version__,
            'decision_rule': 'MDD iff p > 0.5; p == 0.5 is HC', 'tolerance': TOLERANCE,
            'conditions': list(CONDITIONS), 'metric_groups_verified': groups,
            'scalar_comparisons': comparisons, 'max_absolute_error': maximum,
            'aggregation': 'Within each seed, arithmetic mean of 5 outer folds; then mean of 3 seed means. Pooled OOF is separately reported.',
            'independent_metrics': result}


def audit_asr_and_sources(sources):
    sources.add(SCRIPT, 'independent_verification_script')
    asset_path = S00 / 'manifests/asr_support_assets.json'
    sources.add(asset_path, 'pre_existing_ASR_asset_lock')
    assets = []
    for name, expected in read(asset_path).items():
        item = sources.add(S00 / name, 'ASR_runtime_asset_verified_against_pre_existing_hash', expected['sha256'])
        require(item['bytes'] == expected['bytes'], 'ASR runtime asset byte count mismatch')
        assets.append(item)
    require(len(assets) == 2, 'Expected VAD and ffmpeg assets')
    recovery_path = S00 / 'transcripts/B0-37/first_segment_code/02010036/22.json'
    recovered = read(recovery_path)
    sources.add(recovery_path, 'recovered_ASR_parent')
    original_path = S00 / 'archive/asr_original_empty/02010036/22.json'
    evidence_path = S00 / 'reports/asr_empty_retry_02010036_22.json'
    original = sources.add(original_path, 'archived_original_empty_ASR', recovered['recovery']['original_transcript_sha256'])
    evidence = sources.add(evidence_path, 'ASR_recovery_evidence', recovered['recovery']['evidence_sha256'])
    old = read(original_path)
    retry = read(evidence_path)
    require(old['empty'] and old['segments'] == [] and old['answer_text'] == '', 'Archived ASR should be empty')
    require(old['source_sha256'] == recovered['source_sha256'] == retry['source_sha256'], 'ASR audio source mismatch')
    require(retry['source_unchanged'] and recovered['recovery']['raw_waveform_unchanged'], 'ASR raw source should be unchanged')
    successful = next(candidate for candidate in sorted(retry['candidates'], key=lambda x: x['gain']) if candidate['segments'])
    require(successful['gain'] == recovered['recovery']['gain'] == 10 and successful['segments'] == recovered['segments'],
            'Recovered ASR must use smallest successful gain')
    requirements = [sources.add(path, 'S00_environment_requirements') for path in sorted(S00.glob('requirements*.txt'))]
    require(len(requirements) == 4, 'Expected 4 S00 requirements files')
    git_result = subprocess.run(['git', '-C', str(S00 / 'upstream-dev'), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=False)
    commit = git_result.stdout.strip() if git_result.returncode == 0 else None
    require(commit is None or re.fullmatch('[0-9a-f]{40,64}', commit) is not None, 'Invalid upstream Git commit')
    return {'passed': True, 'runtime_assets': assets, 'original_empty_ASR': original,
            'recovery_evidence': evidence, 'smallest_successful_gain': 10,
            'requirements': requirements, 'upstream_git_HEAD': commit,
            'upstream_git_read_only': True,
            'upstream_git_error': git_result.stderr.strip() if commit is None else None}


def workbook_rows(path):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        result = {}
        for row in sheet.findall('m:sheetData/m:row', ns):
            cells = {}
            for cell in row.findall('m:c', ns):
                column = re.match('[A-Z]+', cell.attrib['r']).group(0)
                value = cell.find('m:v', ns)
                if cell.attrib.get('t') == 's':
                    text = shared[int(value.text)]
                elif cell.attrib.get('t') == 'inlineStr':
                    text = ''.join(cell.find('m:is', ns).itertext())
                else:
                    text = '' if value is None else value.text
                cells[column] = text
            if int(row.attrib['r']) == 1:
                require(cells.get('A') == 'subject id' and cells.get('B') == 'type', 'Unexpected diagnosis workbook header')
                continue
            if not cells.get('A'):
                continue
            subject_id = str(int(float(cells['A']))).zfill(8)
            require(cells.get('B') in ('HC', 'MDD') and subject_id not in result, 'Invalid or repeated diagnosis row')
            result[subject_id] = {'label': int(cells['B'] == 'MDD'), 'category': cells['B'],
                                  'row': int(row.attrib['r']), 'sheet': 'Sheet1'}
        return result


def audit_labels(sources, subjects):
    raw = read(S00 / 'manifests/modma_raw_files.json')
    labels = read(S00 / 'manifests/modma_labels.json')
    sources.add(S00 / 'manifests/modma_raw_files.json', 'raw_source_mapping')
    sources.add(S00 / 'manifests/modma_labels.json', 'cached_diagnosis_mapping')
    sources.add(S00 / 'manifests/subjects.json', 'active_subject_diagnoses')
    verified = []
    for name, table in labels.items():
        matches = [r for r in raw if r['archive'] == name and r['member'] == table['archive_member']]
        require(len(matches) == 1, 'Diagnosis workbook provenance must be unique')
        item = matches[0]
        sources.add(WORK / item['path'], 'real_diagnosis_workbook_XML_reparsed', item['sha256'])
        observed = workbook_rows(WORK / item['path'])
        require(observed == table['labels'], 'Real diagnosis workbook and cached labels differ')
        require(all(observed[sid]['label'] == s['label'] for sid, s in subjects.items()), 'Active subject diagnosis mismatch')
        verified.append({'archive': name, 'workbook': relative(WORK / item['path']),
                         'full_table_records': len(observed), 'active_subjects_verified': len(subjects)})
    require(sorted(r['full_table_records'] for r in verified) == [52, 53], 'Unexpected real diagnosis table sizes')
    return {'passed': True, 'parser': 'stdlib zipfile + XML; no cached-label-only comparison', 'workbooks': verified}


def audit_nested(sources, folds):
    path = WORK / 'tridep_ccml/experiments/CCML-v2-S-37-fulltext/nested_splits.json'
    sources.add(path, 'legacy_nested_split_person_overlap_evidence')
    nested = read(path)
    require(nested['folds_sha256'] == digest(S00 / read(S00 / 'configs/cohort.json')['fold_file']), 'Legacy outer split mismatch')
    checkpoint_sources = read(OUT / 'checkpoint_provenance.json')
    values = []
    for fold in range(1, 6):
        outer = folds[f'fold_{fold}']
        outer_train, outer_val, outer_test = (set(outer[key]) for key in SPLITS)
        checkpoints = [r for r in checkpoint_sources if r['fold'] == fold and r['modality'] in ('speech', 'text')]
        require(len(checkpoints) == 6, 'Expected 3 seeds x 2 ST checkpoints per fold')
        require(all(set(c['train_ids']) == outer_train for c in checkpoints), 'Fixed ST training provenance mismatch')
        inners = nested['folds'][str(fold)]
        require(len(inners) == 3, 'Expected 3 legacy inner folds')
        val_coverage = []
        for inner in inners:
            train, val = set(inner['train_ids']), set(inner['val_ids'])
            require(not train & val and train | val == outer_train, 'Legacy inner split differs from outer train')
            require(val <= outer_train and not val & outer_val and not val & outer_test, 'Legacy inner validation isolation mismatch')
            require(all(val <= set(c['train_ids']) for c in checkpoints), 'Legacy inner val should have been seen by fixed ST')
            val_coverage.extend(inner['val_ids'])
            values.append({'outer_fold': fold, 'inner_fold': inner['inner_fold'],
                           'inner_val_ids': inner['val_ids'], 'intersection_with_original_ST_train_ids': sorted(val & outer_train),
                           'intersection_with_original_outer_val_ids': sorted(val & outer_val),
                           'intersection_with_original_outer_test_ids': sorted(val & outer_test),
                           'fixed_ST_training_checkpoints_verified': len(checkpoints),
                           'all_inner_val_seen_by_fixed_ST_supervised_fit': True})
        require(len(val_coverage) == len(outer_train) and set(val_coverage) == outer_train, 'Legacy inner validation coverage mismatch')
    return {'passed': True, 'inner_folds_verified': len(values),
            'finding': 'Legacy inner validation subjects are in the corresponding outer-train cohort already used to fit fixed original S/T. These inner validations cannot establish unseen-subject neural generalization for frozen S/T.',
            'fold_evidence': values}


def audit_status_and_segments(status, subjects):
    require(status['passed'] and status['status'] == 'complete' and status['phase'] == 0, 'Stage 0 should be complete')
    for key in ('new_neural_fits', 'new_inner_fits', 'new_calibrator_fits', 'optimizer_updates'):
        require(status[key] == 0, 'Stage 0 must have no new fit or update')
    require(status['next_phase_started'] is False, 'Next phase must not have started')
    replay_expected = {'checkpoints': 45, 'frozen_state_unchanged': 45, 'repeat_input_equal': 45,
                       'classification_changes': 0, 'last_linear_replay_max_error': 0.0, 'max_probability_error': 0.0}
    for key, value in replay_expected.items():
        require(status['frozen_replay'][key] == value, 'Frozen replay status mismatch: ' + key)
    data = read(OUT / 'data_and_split_audit.json')
    stored_zeros = data['speech_zero_segment_records_across_seeds']
    zero_questions = {(r['subject_id'], r['question']) for r in stored_zeros}
    require(len(stored_zeros) == 288 and len(zero_questions) == 96, 'Zero-question audit counts mismatch')
    require(len({(r['subject_id'], r['question'], r['seed']) for r in stored_zeros}) == 288, 'Repeated zero audit record')
    require(all({r['seed'] for r in stored_zeros if (r['subject_id'], r['question']) == key} == set(SEEDS)
                for key in zero_questions), 'Each zero question must appear at all 3 seeds')
    segment_count = 0
    duplicate_count = 0
    independently_empty = set()
    for sid in subjects:
        for question in range(1, 30):
            metadata = read(S00 / f'features/B0-37/hubert_segments/{sid}/{question:02d}.json')
            segment_count += metadata['segment_count']
            duplicate_count += metadata['duplicate_windows']
            if metadata['segment_count'] == 0:
                independently_empty.add((sid, f'{question:02d}'))
    require(independently_empty == zero_questions, 'Independent segment metadata zero-question mismatch')
    require(segment_count == data['hubert_segment_count'] == 6369, 'Segment count mismatch')
    require(duplicate_count == data['duplicate_speech_windows'] == 3, 'Duplicate window count mismatch')
    return {'passed': True, 'frozen_replay_status_verified': replay_expected,
            'new_fits_and_updates': 0, 'unique_zero_questions': 96, 'zero_question_seed_records': 288,
            'hubert_segments': segment_count, 'duplicate_speech_windows': duplicate_count}


def verify_bindings(value):
    for item in value:
        require(digest(WORK / item['path']) == item['sha256'], 'Supplementary binding mismatch: ' + item['path'])
        require((WORK / item['path']).stat().st_size == item['bytes'], 'Supplementary byte count mismatch')


def run(verify_only=False):
    # The main verification checks all locked raw/source/output hashes and frozen evidence.
    # It writes its own previously-unlocked verification.json only; it does not train or infer.
    sys.path.insert(0, str(SCRIPT.parent))
    import stage0
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        stage0.verify()
    main_result = read(OUT / 'verification.json')
    require(main_result['passed'], 'Main stage-0 verification failed')
    status = read(OUT / 'status.json')
    base_bindings = [binding(OUT / name) for name in ('status.json', 'source_manifest.json', 'output_manifest.json')]
    if verify_only:
        old = read(RESULT)
        verify_bindings(old['artifact_bindings'])
        verify_bindings(read(MANIFEST)['sources'])
        require(base_bindings == old['base_artifact_bindings'], 'Main completion artifacts changed')
    sources = Sources()
    cohort = read(S00 / 'configs/cohort.json')
    folds = read(S00 / cohort['fold_file'])
    subjects = {r['subject_id']: r for r in read(S00 / 'manifests/subjects.json')}
    metrics = audit_metrics(folds, subjects)
    asr = audit_asr_and_sources(sources)
    labels = audit_labels(sources, subjects)
    nested = audit_nested(sources, folds)
    frozen = audit_status_and_segments(status, subjects)
    archive_manifest = S00 / 'manifests/modma_archives.json'
    sources.add(archive_manifest, 'historical_archive_manifest_link_only')
    archives = read(archive_manifest)
    source_rows = sources.ordered()
    verify_bindings(source_rows)
    evidence = {'passed': True, 'phase': 0,
                'completed_at_Asia_Shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'main_verification': main_result, 'base_artifact_bindings': base_bindings,
                'independent_metrics': metrics, 'ASR_and_environment': asr,
                'real_workbook_diagnoses': labels, 'legacy_inner_split_overlap': nested,
                'frozen_status_and_segment_audit': frozen, 'supplementary_sources_verified': len(source_rows),
                'historical_ZIP_boundary': {'archives': archives, 'historical_manifest_link_verified': True,
                                            'full_ZIP_rehashed_in_this_run': False,
                                            'active_extracted_raw_files_rehashed_by_main_verification': True},
                'no_new_fitting_or_inference': True}
    if verify_only:
        require(source_rows == read(MANIFEST)['sources'], 'Supplementary source inventory changed')
        for key in ('independent_metrics', 'ASR_and_environment', 'real_workbook_diagnoses',
                    'legacy_inner_split_overlap', 'frozen_status_and_segment_audit'):
            require(evidence[key] == old[key], 'Independent evidence changed: ' + key)
        print(json.dumps({'passed': True, 'supplementary_sources_verified': len(source_rows),
                          'metric_groups_verified': metrics['metric_groups_verified'],
                          'max_absolute_error': metrics['max_absolute_error'], 'verification_mode': 'read_only_supplement'}, ensure_ascii=False))
        return
    require(not MANIFEST.exists() and not RESULT.exists() and not REPORT.exists(),
            'Supplementary artifacts already exist; use --verify to preserve them')
    write(MANIFEST, {'passed': True, 'sources_count': len(source_rows), 'sources': source_rows})
    report = '\n'.join([
        '# 阶段0补充来源与独立核验', '',
        '**阶段0来源锁定与核验全部通过。**', '',
        '本报告补充[阶段0主完成报告](TriDep_阶段0_对照与来源锁定完成报告.md)。主执行脚本、主status、主source manifest、主output manifest及现有已锁产物保持原样；本次未进行训练、推理、校准或优化器更新。', '',
        f'先执行主 `stage0.verify()`：{main_result["source_hashes_verified"]} 个主来源哈希、{main_result["output_hashes_verified"]} 个主产物哈希全部通过，15个缓存包及原验证集权重重新核对。', '',
        f'使用 scikit-learn {sklearn.__version__} 独立核对四个融合控制及E/S/T单分支，共7条件×3种子×5外折×train/val/test＝315组，逐组F1、BA、MDD recall、AUC、NLL、Brier全部通过；最大绝对误差 `{metrics["max_absolute_error"]:.17g}`，容差 `1e-12`。分类严格使用 `p > 0.5`，等于0.5判HC。每种子先平均5外折，再平均3种子；合并OOF另行核验。', '',
        '冻结状态核验：45个检查点的state未改、45次重复输入相同、分类改变0、最后线性层回放误差0、最大概率误差0。逐题元数据独立重计：6369片段、3重复窗、96个唯一零片段题、288个种子记录。', '',
        f'补充来源清单锁定 **{len(source_rows)} 个文件**：实际VAD与ffmpeg文件对照既存预期SHA-256/字节数、02010036/22原空转写与重试证据分别对照recovery中既存哈希、4份S00 requirements、独立核验脚本、原始标签工作簿与映射以及legacy nested split。两份实际XLSX的53/52行全表经ZIP/XML重新解析，与manifest全字段及活动37人标签一致。', '',
        '语音转写恢复遵守gain 1→10→100的既定尝试顺序，确认gain10为首个非空结果；原空文本及证据保持可追溯。', '',
        f'只读记录 upstream-dev Git HEAD：`{asr["upstream_git_HEAD"] or "不可读取；vendor/source文件仍受主来源锁定"}`。未锁定或改动.git内部状态。', '',
        'legacy nested_splits的15个inner折全部复核：inner val属于同一外折原ST训练人员，与原outer val/test交集均空；相应3种子×S/T共6个检查点的train_ids逐一证明这些inner val已用于原固定S/T监督训练。因此该inner val不能用作冻结S/T的未见人员神经模型泛化证据。', '',
        '**归档核验边界：** 两个历史原ZIP合计约12.85GB（约13GB），本次仅通过既存modma_archives manifest链接历史SHA-256与来源记录，未重新对完整ZIP做全量哈希；活动人员对应的解压raw数据已由主程序重新逐文件全量哈希核验。', '',
        '证据链：[补充来源清单](supplementary_source_manifest.json)、[独立最终核验](independent_verification.json)。independent_verification将本报告、补充来源清单、独立脚本和主完成状态/清单通过SHA-256绑定；补充清单逐文件锁定来源。', '',
        '可重跑入口：`S00/.venv/Scripts/python.exe tridep_ccml/stage0_verify.py --verify`。该入口先运行主核验，再重复独立指标和来源审计，保持已保存的补充证据文件不变。', '',
    ])
    REPORT.write_text(report, encoding='utf-8')
    evidence['artifact_bindings'] = [binding(SCRIPT), binding(MANIFEST), binding(REPORT)]
    write(RESULT, evidence)
    verify_bindings(evidence['artifact_bindings'])
    verify_bindings(base_bindings)
    print(json.dumps({'passed': True, 'supplementary_sources_verified': len(source_rows),
                      'metric_groups_verified': metrics['metric_groups_verified'],
                      'max_absolute_error': metrics['max_absolute_error'],
                      'independent_verification': relative(RESULT)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify existing supplement without rewriting it')
    run(parser.parse_args().verify)
