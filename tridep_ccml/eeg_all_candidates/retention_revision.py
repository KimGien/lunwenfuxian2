"""Apply the user's 2026-10-13 retention rule to sealed EEG results.

This standard-library-only entry point does not import training/evaluation
runners. Historical reports, scores, checkpoints and their seals stay intact;
the current decision is a separately versioned, post-test development decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean

WORK = Path(__file__).resolve().parents[2]
REPORT = Path('output/analysis/eeg_all_candidates_v1_20261008')
EVALUATION = Path('tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation')
BASELINE = Path('S00/reports/b0_training_summary.json')
BASELINE_SHA256 = '60cef962c8685492c0fa1ba924b8e29b0f6eff0d7346befe958a96b24da9f83e'
POLICY_ID = 's00_original_cbramod_cnn_macro_f1_gt_20261013'
POLICY_FILE = 'retention_policy_revision_20261013.json'
TABLE_FILE = 'unified_comparison_original_tridep_baseline_20261013.csv'
MODELS_FILE = 'retained_models_original_tridep_baseline_20261013.json'
REPORT_FILE = '九候选统一完成报告_原TRIDEP基线修订版_20261013.md'
METHODS = ('CSBrain', 'REVE', 'EEGPT', 'LUNA', 'CodeBrain', 'EEGConformer', 'BIOT', 'EEG2Rep', 'EEGMamba')
SEEDS = (11, 22, 33)
FOLDS = (1, 2, 3, 4, 5)
METRICS = ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def descriptor(path, root):
    return {'path': path.relative_to(root).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(desc, root):
    path = (root / desc['path']).resolve()
    require(path.is_relative_to(root.resolve()), 'Source outside workspace')
    require(descriptor(path, root)['sha256'] == desc['sha256'], 'Source changed: ' + desc['path'])
    return path


def keep(score, baseline):
    require(all(math.isfinite(x) and 0 <= x <= 1 for x in (score, baseline)), 'Invalid Macro-F1')
    return score > baseline  # No rounding, minimum delta, seed count, BA or recall gate.


def macro_f1(rows):
    scores = []
    for label in (0, 1):
        tp = sum(r['label'] == label and int(r['p_eeg'] > .5) == label for r in rows)
        fp = sum(r['label'] != label and int(r['p_eeg'] > .5) == label for r in rows)
        fn = sum(r['label'] == label and int(r['p_eeg'] > .5) != label for r in rows)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.)
    return mean(scores)


def baseline_evidence(root):
    source = root / BASELINE
    checked({'path': BASELINE.as_posix(), 'sha256': BASELINE_SHA256}, root)
    summary = read_json(source)
    require(summary['passed'] and summary['experiment'] == 'B0-37', 'Original S00 baseline required')
    require(summary['seeds'] == list(SEEDS) and summary['folds_per_seed'] == 5, 'Baseline aggregation mismatch')
    fold_source = root / 'S00/manifests/fold_assignments.json'
    fold_sha = descriptor(fold_source, root)['sha256']
    sources, seed_means, test_ids, labels = [descriptor(source, root), descriptor(fold_source, root)], {}, {}, {}
    for seed in SEEDS:
        path = root / f'S00/runs/B0-37/seed_{seed}/probabilities.jsonl'
        records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        rows = [r for r in records if r['split'] == 'test']
        require(len(rows) == 37 and len({r['subject_id'] for r in rows}) == 37, 'Baseline OOF coverage mismatch')
        for row in rows:
            require(row['experiment'] == 'B0-37' and row['seed'] == seed and row['folds_sha256'] == fold_sha, 'Baseline identity mismatch')
            require(row['label'] in (0, 1) and math.isfinite(row['p_eeg']) and 0 <= row['p_eeg'] <= 1, 'Invalid baseline prediction')
            require(row['subject_id'] not in labels or labels[row['subject_id']] == row['label'], 'Inconsistent labels')
            labels[row['subject_id']] = row['label']
        scores = []
        for fold in FOLDS:
            group = [r for r in rows if r['outer_fold'] == fold]
            require(len(group) in (7, 8), 'Baseline fold size mismatch')
            score = macro_f1(group)
            reported = summary['per_seed'][str(seed)]['fold_metrics'][f'fold_{fold}']['eeg']['macro_f1']
            require(abs(score - reported) < 1e-12, 'Baseline fold score mismatch')
            scores.append(score)
            test_ids[(seed, fold)] = {r['subject_id'] for r in group}
        seed_means[str(seed)] = mean(scores)
        sources.append(descriptor(path, root))
    score = summary['aggregate']['eeg']['macro_f1']['mean_of_seed_fold_means']
    require(abs(mean(seed_means.values()) - score) < 1e-12, 'Baseline mean mismatch')
    require(len(labels) == 37 and sum(labels.values()) == 16, 'Baseline cohort mismatch')
    return {'condition': 'S00-B0-37-CBraMod-Mumtaz-originalCNN-60ep', 'macro_f1': score,
            'experiment': 'B0-37', 'source_json_path': 'aggregate.eeg.macro_f1.mean_of_seed_fold_means',
            'aggregation': 'mean5folds then mean3seeds', 'seed_fold_means': seed_means,
            'prediction_recalculation_passed': True, 'distinct_people': 37,
            'sources': sources}, test_ids


def classify(rows, evaluation, baseline):
    require(len(rows) == 23 and len({r['condition'] for r in rows}) == 23, 'Expected 23 unique conditions')
    require(set(r['condition'] for r in rows) == set(evaluation['conditions']), 'Condition inventory mismatch')
    result = []
    historical = ('matched_control', 'macro_f1_delta', 'retention_passed', 'probability_benefit',
                  'best_selection_and_retention_unchanged')
    for row in rows:
        summary = evaluation['conditions'][row['condition']]
        require(str(row['primary_snapshot']) == summary['primary_snapshot'], 'Primary snapshot changed')
        primary = summary['variants'][summary['primary_snapshot']]['primary']
        require(all(abs(float(row[key]) - primary[key]) < 1e-12 for key in METRICS), 'Saved metric disagreement')
        score = primary['macro_f1']
        updated = {('historical_' + key if key in historical else key): value for key, value in row.items()}
        updated.update(retention_policy=POLICY_ID, retention_baseline=baseline['condition'],
                       baseline_macro_f1=baseline['macro_f1'], macro_f1_delta=score - baseline['macro_f1'],
                       macro_f1_delta_percentage_points=100 * (score - baseline['macro_f1']),
                       retention_passed=keep(score, baseline['macro_f1']),
                       retention_status='保留' if keep(score, baseline['macro_f1']) else '不保留',
                       checkpoint_selection_unchanged=True, post_test_development_decision=True)
        result.append(updated)
    return result


def retained_artifacts(rows, evaluation, test_ids, root):
    retained = {r['condition']: str(r['primary_snapshot']) for r in rows if r['retention_passed']}
    found, models = set(), []
    for desc in evaluation['probability_batch_artifacts']:
        path = Path(desc['path'])
        condition, seed_part, fold_part, snapshot_part = path.parts[-5:-1]
        if condition not in retained:
            continue
        expected_snapshot = 'best' if retained[condition] == 'best' else 'epoch_' + retained[condition]
        if snapshot_part != expected_snapshot:
            continue
        completion = read_json(checked(desc, root))
        binding = completion['binding']
        seed, fold = int(seed_part.removeprefix('seed_')), int(fold_part.removeprefix('fold_'))
        key = condition, seed, fold
        require(completion['passed'] and key not in found, 'Invalid or duplicate retained completion')
        require((binding['condition'], binding['seed'], binding['fold']) == key, 'Retained binding mismatch')
        require(str(binding['snapshot']) == retained[condition], 'Retained snapshot mismatch')
        require(set(binding['roles']['test']) == test_ids[(seed, fold)], 'Retained test cohort mismatch')
        checkpoint = (root / binding['checkpoint']['path']).resolve()
        require(checkpoint.is_relative_to(root.resolve()) and checkpoint.is_file(), 'Retained checkpoint missing')
        found.add(key)
        models.append({'condition': condition, 'seed': seed, 'fold': fold, 'snapshot': binding['snapshot'],
                       'evaluation_completion': desc, 'checkpoint': binding['checkpoint'],
                       'training_metadata': binding['metadata'], 'training_completion': binding['completion'],
                       'checkpoint_selection': binding['checkpoint_selection'],
                       'selected_checkpoint_evidence': binding.get('selected_checkpoint_evidence')})
    require(found == {(c, s, f) for c in retained for s in SEEDS for f in FOLDS}, 'Retained model coverage mismatch')
    return sorted(models, key=lambda m: (m['condition'], m['seed'], m['fold']))


def render_report(rows, families, baseline):
    kept = [r for r in rows if r['retention_passed']]
    lines = ['# 九篇 EEG 单模态统一完成报告：原 TRI-DEP 基线修订版', '',
             '修订日期：2026-10-13。当前保留判定按用户本次指令更新；原始训练、分数、预测、选模和旧验收记录保持不变。', '',
             f"基线为本地原 S00/B0-37 的 CBraMod-Mumtaz＋原CNN，测试 Macro-F1 = **{baseline['macro_f1']:.16f}（{100*baseline['macro_f1']:.8f}%）**。",
             '这是原60轮EEG分支，不是修正输入后48.26%的CBraMod控制，也不是本轮44.04%的新小头。', '',
             '**当前规则：每条已登记路线的主要快照 Macro-F1 严格高于上述精确基线即保留；相等或低于不保留。**',
             '不再以超过LaBraM、至少提高1个百分点、两个种子改善、BA或MDD召回作为本次保留的附加门槛。其余指标和历史匹配比较仍保留供查阅。', '',
             '主口径保持每种子五折平均、再平均三个种子；阈值保持 p(MDD)>0.5。原CNN诊断只使用预定第300轮，其他路线仍使用原验证损失选出的best；没有按测试分数重新选快照。', '',
             f"23条监督路线中保留 **{len(kept)}条**，对应 **{len(kept)*15}个**原种子/折模型。九篇候选保留 **{sum(f['retained'] for f in families.values())}篇、{sum(r['method'] in METHODS for r in kept)}条路线**；LaBraM五条对照也纳入保留。", '',
             '| 九篇候选 | 当前结论 | 保留路线 |', '|---|---|---|']
    for method, family in families.items():
        lines.append(f"| {method} | {'保留' if family['retained'] else '不保留'} | {'；'.join(family['retained_conditions']) or '无'} |")
    lines += ['', '所有路线的结果如下，差值单位为百分点；实际判断使用未舍入数值。', '',
              '| 条件 | 测试Macro-F1 | 相对原基线 | 当前结论 |', '|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['condition']} | {100*float(row['macro_f1']):.4f}% | {row['macro_f1_delta_percentage_points']:+.4f} | {row['retention_status']} |")
    lines += ['', '保留按路线决定：CodeBrain两条保留；LUNA仅冻结小头保留；BIOT仅末层微调保留，不能将同家族未过线的路线一起标为通过。', '',
              '这是在已有测试结果公开后按用户要求进行的开发保留重判，不是原预登记规则，也不是统计显著性或独立泛化结论。原S00与新路线的输入处理、窗长或训练方式存在差异，该差值作为用户指定的历史基线参照，不单独归因于骨干。',
              '保留表示保留候选路线及已保存的折内模型；本次不新增训练、模型推理、融合选权或部署，也不自动重启已取消的融合计划。不保留路线的实验记录和权重同样不删除。',
              '360次正式训练已完成：345次监督拟合＋15次EEG2Rep自监督拟合。EEGMamba沿用作者公开实现适配实验的范围说明。', '',
              f'- [当前比较表]({TABLE_FILE})', f'- [当前策略及逐路线判定]({POLICY_FILE})',
              f'- [保留模型与来源清单]({MODELS_FILE})',
              '- [原预登记结论（历史）](九候选统一完成报告.md)',
              '- [原S00基线结果](../../../S00/reports/b0_training_summary.json)', '']
    return '\n'.join(lines)


def build(root=WORK):
    baseline, test_ids = baseline_evidence(root)
    complete_path = root / EVALUATION / 'complete.json'
    complete = read_json(complete_path)
    require(complete['passed'] and complete['formal_fits_verified'] == 360, 'Formal completion required')
    source_keys = ('evaluation_report', 'comparison_table', 'oof_probabilities', 'nine_paper_report', 'evaluator_source')
    expected_paths = {'evaluation_report': EVALUATION / 'evaluation_report.json',
                      'comparison_table': REPORT / 'unified_comparison.csv',
                      'oof_probabilities': REPORT / 'oof_probabilities.csv',
                      'nine_paper_report': REPORT / '九候选统一完成报告.md'}
    for key, expected in expected_paths.items():
        require(complete[key]['path'] == expected.as_posix(), 'Historical source identity mismatch: ' + key)
    for key in source_keys:
        checked(complete[key], root)
    evaluation = read_json(root / EVALUATION / 'evaluation_report.json')
    require(evaluation['passed'] and evaluation['primary_aggregation'] == baseline['aggregation'], 'Evaluation aggregation mismatch')
    require(evaluation['distinct_people'] == 37 and evaluation['threshold'] == .5 and evaluation['tie'] == 'HC', 'Evaluation protocol mismatch')
    rows = list(csv.DictReader((root / REPORT / 'unified_comparison.csv').read_text(encoding='utf-8-sig').splitlines()))
    revised = classify(rows, evaluation, baseline)
    models = retained_artifacts(revised, evaluation, test_ids, root)
    families = {}
    for method in METHODS:
        conditions = [r for r in revised if r['method'] == method]
        retained = [r['condition'] for r in conditions if r['retention_passed']]
        families[method] = {'retained': bool(retained), 'retained_conditions': retained,
                            'conditions': [r['condition'] for r in conditions], 'all_prespecified_routes_completed': True}
    sources = [descriptor(complete_path, root), *(complete[key] for key in source_keys), *baseline['sources']]
    policy = {'schema': 'eeg_retention_policy_revision_v1', 'policy_id': POLICY_ID, 'revision_date': '2026-10-13',
              'user_instruction': '将TRI-DEP原论文中使用的CBraMod+CNN在本地的结果作为基线，高于这个结果的都要保留',
              'baseline': baseline, 'rule': {'metric': 'macro_f1', 'operator': '>', 'minimum_delta': 0,
              'decision_scope': 'existing_primary_snapshot_per_condition', 'family_rule': 'any_condition_passes',
              'additional_seed_ba_recall_or_matched_control_gates': False},
              'post_test_development_reclassification': True, 'not_a_significance_test': True,
              'checkpoint_selection_changed': False, 'classification_threshold_changed': False,
              'new_training_fits': 0, 'new_model_forward_calls': 0, 'fusion_or_deployment_changed': False,
              'historical_preregistered_result': complete['evaluation_report'], 'source_artifacts': sources,
              'conditions': {r['condition']: {'method': r['method'], 'primary_snapshot': r['primary_snapshot'],
                  'macro_f1': float(r['macro_f1']), 'macro_f1_delta': r['macro_f1_delta'],
                  'retained': r['retention_passed'], 'historical_matched_control': r['historical_matched_control'],
                  'historical_engineering_retention': evaluation['conditions'][r['condition']]['engineering_retention']}
                  for r in revised}, 'nine_papers': families,
              'retained_conditions': [r['condition'] for r in revised if r['retention_passed']],
              'retained_model_count': len(models)}
    manifest = {'policy_id': POLICY_ID, 'baseline_macro_f1': baseline['macro_f1'],
                'condition_count': len(policy['retained_conditions']), 'model_count': len(models),
                'checkpoint_files_exist': True, 'evaluation_completion_hashes_rechecked': True,
                'checkpoint_hashes_reused_from_sealed_evaluation': True, 'checkpoint_bytes_rehashed': False,
                'models': models}
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(revised[0]))
    writer.writeheader()
    writer.writerows(revised)
    json_bytes = lambda value: (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    products = {POLICY_FILE: json_bytes(policy), TABLE_FILE: buffer.getvalue().encode('utf-8-sig'),
                MODELS_FILE: json_bytes(manifest), REPORT_FILE: render_report(revised, families, baseline).encode('utf-8')}
    current = {'policy_id': POLICY_ID, 'status': 'active', 'retained_condition_count': len(policy['retained_conditions']),
               'retained_candidate_family_count': sum(f['retained'] for f in families.values()),
               'baseline_macro_f1': baseline['macro_f1'], 'generator_source': descriptor(Path(__file__).resolve(), root),
               'historical_sources_unchanged': sources,
               'artifacts': {name: {'path': (REPORT / name).as_posix(), 'sha256': hashlib.sha256(data).hexdigest()}
                             for name, data in products.items()}}
    products['current_retention.json'] = json_bytes(current)
    return products, current


def run(*, check=False, root=WORK):
    products, current = build(root)
    for name, data in products.items():
        path = root / REPORT / name
        if check:
            require(path.is_file() and path.read_bytes() == data, 'Current revision missing or stale: ' + name)
        elif not path.exists() or path.read_bytes() != data:
            temp = path.with_name(path.name + '.tmp')
            temp.write_bytes(data)
            temp.replace(path)
    for desc in current['historical_sources_unchanged']:
        checked(desc, root)
    return {'passed': True, 'mode': 'verify' if check else 'write',
            'baseline_macro_f1': current['baseline_macro_f1'], 'retained_conditions': current['retained_condition_count'],
            'retained_candidate_families': current['retained_candidate_family_count'], 'historical_sources_unchanged': True,
            'new_training_fits': 0, 'new_model_forward_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify existing revision without writing')
    args = parser.parse_args()
    print(json.dumps(run(check=args.check), ensure_ascii=False))
