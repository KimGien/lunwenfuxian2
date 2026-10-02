"""Verify every trained TC-01 artifact and publish its descriptive results."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from .ccml import ThreeViewHeads
from .metrics import summarize, binary_metrics
from .records import ROOT, MODALITIES, RecordContext, digest, read_json, require, validate_prediction_records
from .train import load_features, make_initial_state, predict_records, run_directory, state_digest, validate_config

EXPERIMENT = 'TC-01-37-fulltext'
OUTPUT = ROOT / 'reports/tc01'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Report regeneration is safe only if the existing content is identical.
    if path.exists():
        require(read_json(path) == value, f'Refusing to replace different report: {path}')
        return
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify(context):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = validate_config(context)
    rows, audits = [], []
    for seed in config['seeds']:
        expected_initial = state_digest(make_initial_state(seed))
        for fold in config['folds']:
            directory = run_directory(ROOT, EXPERIMENT, seed, fold)
            run = read_json(directory / 'run.json')
            require(run['status'] == 'complete' and run['training_performed'], 'Incomplete TC01 run')
            require(not (directory / 'failure.json').exists(), 'Failure record present')
            checkpoint = directory / 'checkpoint.pt'
            checksum = digest(checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            sidecar = read_json(checkpoint.with_suffix('.json'))
            metadata = saved['metadata']
            require(metadata == sidecar['metadata'] == run['source_metadata'], 'Checkpoint metadata differs')
            require(checksum == sidecar['checkpoint_sha256'] == run['checkpoint_sha256'], 'Checkpoint hash differs')
            expected = {
                'experiment': EXPERIMENT, 'seed': seed, 'fold': fold,
                'completed_epochs': 60, 'selection': 'last_epoch', 'trainable_parameters': 774,
                'optimizer': 'Adamax', 'class_weights': None, 'resampling': False,
                'upstream_mode': 'frozen_eval_cached_features', 'initialization_sha256': expected_initial,
                'training_config_sha256': context.config_sha256,
                'runner_sha256': digest(ROOT / 'train.py'), 'ccml_sha256': digest(ROOT / 'ccml.py'),
                'records_sha256': digest(ROOT / 'records.py'),
            }
            require(all(metadata.get(k) == v for k, v in expected.items()), 'Training protocol differs')
            train_ids = context.folds[f'fold_{fold}']['train']
            require(metadata['train_ids'] == train_ids, 'Fitting IDs differ')
            generator = torch.Generator().manual_seed(seed)
            batch_hasher = hashlib.sha256()
            for _ in range(config['epochs']):
                order = torch.randperm(len(train_ids), generator=generator)
                batch_hasher.update(order.numpy().astype('<i8').tobytes())
            require(metadata['batch_order_sha256'] == batch_hasher.hexdigest(), 'Batch ordering differs')
            updates = 60 * math.ceil(len(train_ids) / 8)
            require(metadata['optimizer_updates'] == updates, 'Optimization budget differs')
            history = read_json(directory / 'train_history.json')
            require([r['epoch'] for r in history] == list(range(1, 61)), 'Training history incomplete')
            require(all(math.isfinite(r['train_loss']) and r['train_loss'] >= 0 for r in history), 'Nonfinite training loss')
            data = load_features(context, seed, fold)
            model = ThreeViewHeads().eval()
            model.load_state_dict(saved['state_dict'], strict=True)
            recorded = read_rows(directory / 'predictions.jsonl')
            require(len(recorded) == 37 and {r['subject_id'] for r in recorded} == set(context.ids), 'Incomplete fold predictions')
            reproduced = predict_records(context, data, model, EXPERIMENT, seed, fold,
                                         checkpoint, metadata, torch.device('cpu'))
            require(recorded == reproduced, 'Reloaded predictions differ from recorded predictions')
            validate_prediction_records(recorded, EXPERIMENT, context=context, require_complete=False)
            rows.extend(recorded)
            audits.append({'seed': seed, 'fold': fold, 'checkpoint_sha256': checksum,
                           'predictions_sha256': digest(directory / 'predictions.jsonl'),
                           'history_sha256': digest(directory / 'train_history.json'),
                           'initialization_sha256': expected_initial,
                           'batch_order_sha256': batch_hasher.hexdigest(),
                           'epochs': 60, 'optimizer_updates': updates,
                           'all_37_prediction_records_reproduced_exactly': True,
                           'first_train_loss': history[0]['train_loss'], 'last_train_loss': history[-1]['train_loss']})
    validation = validate_prediction_records(rows, EXPERIMENT, context=context)
    require(len(audits) == 15 and len(rows) == 555, 'Missing planned jobs')
    summary = summarize(rows)
    # Independently compare metric calculations to the existing sklearn package.
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, log_loss, brier_score_loss
    for seed in config['seeds']:
        for fold in config['folds']:
            part = [r for r in rows if r['seed'] == seed and r['outer_fold'] == fold and r['split'] == 'test']
            y = [r['label'] for r in part]
            p = [r['p_final'][1] for r in part]
            pred = [int(v > .5) for v in p]
            computed = binary_metrics(y, p)
            independent = {'macro_f1': f1_score(y, pred, average='macro'), 'mdd_recall': recall_score(y, pred),
                           'balanced_accuracy': balanced_accuracy_score(y, pred), 'nll': log_loss(y, p, labels=[0, 1]),
                           'brier': brier_score_loss(y, p)}
            require(all(abs(computed[k] - v) < 1e-12 for k, v in independent.items()), 'Metric cross-check failed')
    return rows, audits, validation, summary


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (seed, values) in zip(axes, summary['seed_reports'].items()):
        values = values['pooled_oof']
        bins = [item for item in values['bins'] if item['n']]
        ax.plot([0, 1], [0, 1], '--', color='#89939c', linewidth=1)
        ax.plot([b['mean_p_mdd'] for b in bins], [b['observed_mdd_rate'] for b in bins], 'o-', color='#1e628c')
        for b in bins:
            ax.annotate(f"n={b['n']}", (b['mean_p_mdd'], b['observed_mdd_rate']), xytext=(3, 7), textcoords='offset points', fontsize=8)
        ax.set(xlim=(-.05, 1.05), ylim=(-.05, 1.10), xlabel='Mean predicted P(MDD)',
               ylabel='Observed MDD rate', title=f"Seed {seed} | ECE5 = {values['ece5']:.4f}")
        ax.grid(alpha=.18)
    fig.suptitle('TC-01 | 37 pooled OOF subjects per seed | 5 equal-width bins')
    fig.savefig(OUTPUT / 'reliability.png', dpi=160)
    plt.close(fig)


def completion_report(summary, branches, baseline, audits):
    primary = summary['primary_seed_mean_of_fold_means']
    lines = ['# TC-01 实验完成报告', '',
             'TC-01-37-fulltext 已完成 3 个种子 × 5 折，共 15 组正式训练。每组采用冻结的父表征，仅训练 774 个新末层参数；三个 CE/softmax 分类头的概率等权平均。全部权重与预测复核通过，没有当前阻塞。', '',
             '## 固定协议与验收', '',
             '- 人群：MODMA-37-v1，37 人，16 MDD、21 HC；同一人群和五折锁。',
             '- 种子：11、22、33；每折 batch=8、60 epochs、Adamax lr=0.001、weight decay=0、梯度范数上限 1。',
             '- 上游表征冻结；只在该折 train 上拟合，无类别加权、重采样、选 epoch、调阈值或验证集调参。',
             '- 使用 CPU 完成小型分类头训练；15 个权重、60 轮训练日志及配对初始化/批序哈希核验通过。每折 240 次更新，共 3,600 次。',
             '- 共 555 行预测（含 train/val/test）；其中测试 111 行，每个种子恰好覆盖 37 人一次。',
             '- 重载 15 个最终权重，全部 555 行预测记录完全一致；逐折指标与 sklearn 独立复算一致。', '',
             '## 主要结果', '',
             '本表为每种子五折均值，再对三个种子取均值。NLL 与 Brier 越低越好；各折人数略有差别，因此与合并测试预测的指标分别报告。', '',
             '| 输出 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |',
             '|---|---:|---:|---:|---:|---:|']
    for label, report in [('TC-01 EEG', branches['eeg']), ('TC-01 语音', branches['speech']), ('TC-01 文本', branches['text']), ('TC-01 等权融合', summary),
                          ('TC-00 原概率等权融合', baseline['TC-00-mean-37-fulltext']), ('TC-00 原验证集选权融合', baseline['TC-00-weighted-37-fulltext'])]:
        m = report['primary_seed_mean_of_fold_means']
        lines.append(f"| {label} | {m['macro_f1']:.6f} | {m['mdd_recall']:.6f} | {m['balanced_accuracy']:.6f} | {m['nll']:.6f} | {m['brier']:.6f} |")
    lines.extend(['', '## 种子结果与合并预测', '',
                  '| 种子 | 五折均值 Macro-F1 | 合并 37 人 Macro-F1 | 合并 NLL | 合并 Brier | 合并 ECE5 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for seed, values in summary['seed_reports'].items():
        p = values['pooled_oof']
        lines.append(f"| {seed} | {values['unweighted_fold_mean']['macro_f1']:.6f} | {p['macro_f1']:.6f} | {p['nll']:.6f} | {p['brier']:.6f} | {p['ece5']:.6f} |")
    lines.extend(['', '![TC-01 五箱可靠性图](reliability.png)', '',
                  'ECE 为 5 个等宽箱的 MDD 概率与实际 MDD 比例偏差；图中标注每箱人数。每个种子只有 37 名独立受试者，图表仅作描述。TC-01 是普通概率模型，不定义 Dirichlet u，因此不生成 u 分布或基于 u 的风险—覆盖率曲线。', '',
                  '## 对照解释', ''])
    reference = baseline['TC-00-mean-37-fulltext']['primary_seed_mean_of_fold_means']
    delta = primary['macro_f1'] - reference['macro_f1']
    lines.extend([f"TC-01 相对 TC-00 等权融合的 Macro-F1 变化为 {delta:+.6f}（{delta*100:+.2f} 个百分点）。两者均采用等权概率融合，但 TC-01 重新初始化并训练末层，旧 TC-00 各分支训练预算不同；此差值不能解释为 CCML 收益。", '',
                  'TC-01 是后续 TC-02、TC-04 的同预算 CE 对照。按照计划保留全部种子和折的结果，不因当前分数调整训练设置。本轮使用此前已查看测试结果的固定划分，属于开发性比较。种子与折不构成新增独立受试者。', '',
                  '## 产物与下一阶段', '',
                  '- `../../runs/TC-01-37-fulltext/seed_<seed>/fold_<fold>/`：权重、训练曲线、逐人预测及运行元数据。',
                  '- `predictions.jsonl`：三种子五折全部预测；`summary.json`：融合、单模态及 TC-00 对照指标。',
                  '- `verification.json`：覆盖、哈希、重载一致性及更新预算检查；`reliability.png`：合并测试预测可靠性图。',
                  '- TC-02 及后续阶段尚未启动。后续阶段保持同一冻结表征、初始化和批次顺序。', ''])
    (OUTPUT / 'TC-01实验完成报告.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    context = RecordContext()
    rows, audits, validation, summary = verify(context)
    branches = {}
    for modality in MODALITIES:
        adapted = deepcopy(rows)
        for row in adapted:
            row['p_final'] = row['views'][modality]['p']
            row['prediction'] = int(row['p_final'][1] > .5)
        branches[modality] = summarize(adapted)
        branches[modality]['output'] = modality
    baseline_rows = read_rows(ROOT / 'reports/tc00_records.jsonl')
    baseline = {}
    for experiment in ('TC-00-weighted-37-fulltext', 'TC-00-mean-37-fulltext'):
        selected = [row for row in baseline_rows if row['experiment'] == experiment]
        validate_prediction_records(selected, experiment, context=context)
        baseline[experiment] = summarize(selected)
    write(OUTPUT / 'verification.json', {'passed': True, 'validation': validation, 'checkpoints_verified': 15,
           'prediction_records_reproduced_exactly': 555, 'maximum_probability_error': 0.,
           'independent_metrics_verifier': 'sklearn', 'config_sha256': context.config_sha256,
           'completion_script_sha256': digest(Path(__file__)), 'folds': audits})
    write(OUTPUT / 'summary.json', {'experiment': EXPERIMENT, 'fusion': summary,
                                  'branches': branches, 'baseline_comparison': baseline})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'predictions.jsonl'
    text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
    if path.exists():
        require(path.read_text(encoding='utf-8') == text, 'Existing combined predictions differ')
    else:
        path.write_text(text, encoding='utf-8')
    figures(summary)
    completion_report(summary, branches, baseline, audits)
    status = {'utc': datetime.now(timezone.utc).isoformat(), 'active_blockers': [],
              'TC-00': {'status': 'complete', 'type': 'parent_recheck'},
              'TC-01': {'status': 'complete', 'experiment': EXPERIMENT, 'completed_jobs': 15,
                        'completed_epochs_per_job': 60, 'verified': True,
                        'report': 'reports/tc01/TC-01实验完成报告.md',
                        'primary_metrics': summary['primary_seed_mean_of_fold_means']},
              'TC-02': {'status': 'not_started'}, 'TC-03': {'status': 'not_started'},
              'TC-04': {'status': 'not_started'}, 'TC-05': {'status': 'not_started'},
              'TC-06': {'status': 'deferred_second_phase'}}
    (ROOT / 'reports/experiment_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'completed_jobs': 15,
                      'primary_metrics': summary['primary_seed_mean_of_fold_means'],
                      'per_seed_macro_f1': {s: v['unweighted_fold_mean']['macro_f1'] for s, v in summary['seed_reports'].items()},
                      'report': str(OUTPUT / 'TC-01实验完成报告.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
