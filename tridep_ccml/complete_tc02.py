"""Verify TC-02, its paired CE control, and fixed-output diagnostics."""
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
import torch
from .complete_tc01 import read_rows, write
from .metrics import binary_metrics, summarize
from .records import ROOT, MODALITIES, RecordContext, digest, read_json, require, validate_prediction_records
from .train import load_features, load_tc02, predict_records, run_directory, validate_config

EXPERIMENT = 'TC-02-37-fulltext'
CONTROL = 'TC-01-37-fulltext'
OUTPUT = ROOT / 'reports/tc02'


def verify(context):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = validate_config(context)
    rows, control_rows, audit = [], [], []
    for seed in config['seeds']:
        for fold in config['folds']:
            data = load_features(context, seed, fold)
            model, checkpoint, metadata = load_tc02(context, data, seed, fold, torch.device('cpu'))
            directory = checkpoint.parent
            run = read_json(directory / 'run.json')
            require(run['status'] == 'complete' and run['training_performed'], 'Incomplete TC02 run')
            require(not (directory / 'failure.json').exists(), 'A TC02 failure record exists')
            require(run['source_metadata'] == metadata and run['checkpoint_sha256'] == digest(checkpoint), 'Run provenance mismatch')
            history = read_json(directory / 'train_history.json')
            require([r['epoch'] for r in history] == list(range(1, 61)), 'Incomplete epochs')
            require(all(math.isfinite(r['train_loss']) and r['train_loss'] >= 0 for r in history), 'Invalid training loss')
            require(metadata['effective_loss_parameters']['anneal_epochs'] == 50, 'Annealing schedule changed')
            train_ids = context.folds[f'fold_{fold}']['train']
            generator = torch.Generator().manual_seed(seed)
            batch_hash = hashlib.sha256()
            for _ in range(60):
                batch_hash.update(torch.randperm(len(train_ids), generator=generator).numpy().astype('<i8').tobytes())
            require(metadata['batch_order_sha256'] == batch_hash.hexdigest(), 'Batch order mismatch')
            require(metadata['optimizer_updates'] == 60 * math.ceil(len(train_ids) / 8), 'Budget mismatch')
            control_dir = run_directory(ROOT, CONTROL, seed, fold)
            control_run = read_json(control_dir / 'run.json')
            control_meta = control_run['source_metadata']
            require(control_run['status'] == 'complete', 'CE control incomplete')
            require(digest(control_dir / 'checkpoint.pt') == control_run['checkpoint_sha256'], 'CE checkpoint changed')
            paired_fields = ('seed', 'fold', 'cohort_id', 'cohort_sha256', 'folds_sha256', 'train_ids',
                'feature_manifest_sha256', 'feature_npz_sha256', 'parent_checkpoints',
                'training_config_sha256', 'initialization_sha256', 'batch_order_sha256',
                'completed_epochs', 'optimizer_updates', 'selection', 'trainable_parameters',
                'upstream_mode', 'optimizer', 'class_weights', 'resampling', 'device',
                'torch_version', 'runner_sha256', 'ccml_sha256', 'records_sha256')
            require(all(metadata[k] == control_meta[k] for k in paired_fields), 'TC01/TC02 paired protocol differs')
            recorded = read_rows(directory / 'predictions.jsonl')
            require(len(recorded) == 37 and {r['subject_id'] for r in recorded} == set(context.ids), 'Fold coverage incomplete')
            reloaded = predict_records(context, data, model, EXPERIMENT, seed, fold, checkpoint, metadata, torch.device('cpu'))
            require(recorded == reloaded, 'Reloaded evidence/probability records differ')
            validate_prediction_records(recorded, EXPERIMENT, context=context, require_complete=False)
            rows.extend(recorded)
            control_rows.extend(read_rows(control_dir / 'predictions.jsonl'))
            audit.append({'seed': seed, 'fold': fold, 'checkpoint_sha256': digest(checkpoint),
                          'control_checkpoint_sha256': control_run['checkpoint_sha256'],
                          'predictions_sha256': digest(directory / 'predictions.jsonl'),
                          'history_sha256': digest(directory / 'train_history.json'),
                          'paired_fields_checked': list(paired_fields), 'paired_protocol_equal': True,
                          'initialization_sha256': metadata['initialization_sha256'],
                          'batch_order_sha256': metadata['batch_order_sha256'],
                          'epochs': 60, 'optimizer_updates': metadata['optimizer_updates'],
                          'all_37_records_reproduced_exactly': True,
                          'first_train_loss': history[0]['train_loss'], 'last_train_loss': history[-1]['train_loss']})
    validation = validate_prediction_records(rows, EXPERIMENT, context=context)
    validate_prediction_records(control_rows, CONTROL, context=context)
    require(len(rows) == len(control_rows) == 555 and len(audit) == 15, 'Planned runs incomplete')
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, log_loss, brier_score_loss
    for seed in config['seeds']:
        for fold in config['folds']:
            selected = [r for r in rows if r['seed'] == seed and r['outer_fold'] == fold and r['split'] == 'test']
            y, p = [r['label'] for r in selected], [r['p_final'][1] for r in selected]
            pred = [int(v > .5) for v in p]
            expected = {'macro_f1': f1_score(y, pred, average='macro'), 'mdd_recall': recall_score(y, pred),
                        'balanced_accuracy': balanced_accuracy_score(y, pred), 'nll': log_loss(y, p, labels=[0, 1]),
                        'brier': brier_score_loss(y, p)}
            observed = binary_metrics(y, p)
            require(all(abs(observed[k] - v) < 1e-12 for k, v in expected.items()), 'Independent metric check failed')
    return rows, control_rows, audit, validation


def diagnostics(rows):
    branches = {}
    for modality in MODALITIES:
        branch_rows = deepcopy(rows)
        for row in branch_rows:
            row['p_final'] = row['views'][modality]['p']
            row['prediction'] = int(row['p_final'][1] > .5)
            row['u_final'] = row['views'][modality]['u']
        # Derived metric views only; not written as registered fusion predictions.
        branches[modality] = summarize(branch_rows)
        branches[modality]['output'] = modality
    subsets = {}
    for name, modalities in [('S+T', ('speech', 'text')), ('E+S', ('eeg', 'speech')),
                             ('E+T', ('eeg', 'text')), ('E+S+T', MODALITIES)]:
        subset_rows = deepcopy(rows)
        for row in subset_rows:
            probabilities = torch.tensor([row['views'][m]['p'] for m in modalities], dtype=torch.float32)
            row['p_final'] = probabilities.mean(0).tolist()
            row['prediction'] = int(row['p_final'][1] > .5)
            row['u_final'] = None
        subsets[name] = summarize(subset_rows)
        subsets[name].update(modalities=list(modalities), views_count=len(modalities),
            fusion_method='mean_probability', status='fixed_TC02_output_subset_diagnostic_no_retraining')
    require(subsets['E+S+T']['primary_seed_mean_of_fold_means'] == summarize(rows)['primary_seed_mean_of_fold_means'], 'Full subset does not reproduce TC02')
    return branches, subsets


def figures(summary, branches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (seed, values) in zip(axes, summary['seed_reports'].items()):
        pooled = values['pooled_oof']
        bins = [b for b in pooled['bins'] if b['n']]
        ax.plot([0, 1], [0, 1], '--', color='#8b969f', linewidth=1)
        ax.plot([b['mean_p_mdd'] for b in bins], [b['observed_mdd_rate'] for b in bins], 'o-', color='#277054')
        for b in bins:
            ax.annotate(f"n={b['n']}", (b['mean_p_mdd'], b['observed_mdd_rate']), xytext=(4, 8), textcoords='offset points', fontsize=8)
        ax.set(xlim=(-.05, 1.05), ylim=(-.05, 1.10), xlabel='Mean predicted P(MDD)', ylabel='Observed MDD rate',
               title=f"Seed {seed} | ECE5 = {pooled['ece5']:.4f}")
        ax.grid(alpha=.18)
    fig.suptitle('TC-02 probability mean | 37 pooled OOF subjects per seed | 5 equal-width bins')
    fig.savefig(OUTPUT / 'reliability.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    colors = ['#355d8c', '#277054', '#b46b32']
    for col, seed in enumerate(summary['seed_reports']):
        for index, modality in enumerate(MODALITIES):
            uncertainty = branches[modality]['seed_reports'][seed]['uncertainty']
            for outcome, shift, marker in [('correct_u', -.15, 'o'), ('incorrect_u', .15, 'x')]:
                values = uncertainty[outcome]
                # Deterministic horizontal spread makes all subject observations visible.
                offsets = [0.] if len(values) == 1 else [(.1 * i / max(1, len(values)-1) - .05) for i in range(len(values))]
                axes[0, col].scatter([index + shift + offset for offset in offsets], values,
                    color=colors[index], marker=marker, s=24, alpha=.72,
                    label=('Correct' if outcome == 'correct_u' else 'Incorrect') if index == 0 else None)
            curve = uncertainty['risk_coverage']
            axes[1, col].step([p['coverage'] for p in curve], [p['risk'] for p in curve], where='post',
                             color=colors[index], label=modality.upper())
        axes[0, col].set(xticks=[0, 1, 2], xticklabels=['EEG', 'Speech', 'Text'], ylim=(0, 1.03),
                        ylabel='Single-view u', title=f'Seed {seed} | branch uncertainty')
        axes[0, col].legend(fontsize=8, loc='upper left')
        axes[1, col].set(xlim=(0, 1.02), ylim=(0, 1.03), xlabel='Coverage', ylabel='Classification error rate',
                        title='Retain low-u subjects; tied u retained together')
        axes[1, col].legend(fontsize=8)
        for ax in axes[:, col]:
            ax.grid(alpha=.16)
    fig.suptitle('TC-02 branch diagnostics | probability-mean fusion has no Dirichlet u')
    fig.savefig(OUTPUT / 'branch_uncertainty.png', dpi=160)
    plt.close(fig)


def report(summary, control, baselines, branches, subsets):
    metrics = ('macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier')
    primary = summary['primary_seed_mean_of_fold_means']
    ce = control['primary_seed_mean_of_fold_means']
    delta = {m: primary[m] - ce[m] for m in metrics}
    per_seed_delta = {seed: item['unweighted_fold_mean']['macro_f1'] - control['seed_reports'][seed]['unweighted_fold_mean']['macro_f1']
                      for seed, item in summary['seed_reports'].items()}
    comparison = {'reference': CONTROL, 'primary_metric_differences_TC02_minus_TC01': delta,
                  'per_seed_macro_f1_differences': per_seed_delta,
                  'seeds_with_higher_macro_f1': sum(v > 0 for v in per_seed_delta.values())}
    lines = ['# TC-02 实验完成报告', '',
        'TC-02-37-fulltext 已完成 3 个种子 × 5 折，共 15 组正式训练。三个 Softplus 证据头仅采用单视图 D+R 损失，最终将三个 Dirichlet 均值概率等权平均。全部权重、证据和预测重载核验通过，没有当前阻塞。', '',
        '## 训练协议与配对验收', '',
        '- MODMA-37-v1；37 人，16 MDD、21 HC；沿用种子 11、22、33 和原五折。',
        '- 三个独立 128→2 新头，共 774 个参数；父表征冻结。每折 60 轮、batch 8、Adamax lr=0.001、weight decay=0、梯度范数上限 1。',
        '- 单视图损失先按视图求和，再按受试者平均；KL 退火为 min(1, epoch/50)。只用各折 train，不选 epoch、不调阈值、不增加类别权重。',
        '- 与 TC-01 的初始化、批次顺序、输入缓存、训练名单、设备、优化器和更新预算逐折匹配。每折 240 次更新，合计 3,600 次。',
        '- 15 个权重均已重载，555 行完整记录的 e/alpha/p/u 及最终概率完全一致；其中测试 111 行，每个种子 37 人各一次。',
        '- 主分类指标、NLL 和 Brier 均逐折与 sklearn 独立复算一致。TC-01 权重没有被续训或覆盖。', '',
        '## 分类与校准结果', '',
        '以下均采用每种子五折均值再取三种子均值。NLL 和 Brier 越低越好。', '',
        '| 输出 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |', '|---|---:|---:|---:|---:|---:|']
    for label, result in [('TC-02 EEG', branches['eeg']), ('TC-02 语音', branches['speech']), ('TC-02 文本', branches['text']),
                          ('TC-02 等权融合', summary), ('TC-01 CE 等权融合', control),
                          ('TC-00 原概率等权融合', baselines['TC-00-mean-37-fulltext']),
                          ('TC-00 原验证集选权融合', baselines['TC-00-weighted-37-fulltext'])]:
        values = result['primary_seed_mean_of_fold_means']
        lines.append('| ' + label + ' | ' + ' | '.join(f'{values[m]:.6f}' for m in metrics) + ' |')
    lines.extend(['', f"相对同预算 TC-01，Macro-F1 变化 {delta['macro_f1']:+.6f}（{delta['macro_f1']*100:+.2f} 个百分点），MDD recall 变化 {delta['mdd_recall']:+.6f}，NLL 变化 {delta['nll']:+.6f}，Brier 变化 {delta['brier']:+.6f}。三个种子中有 {comparison['seeds_with_higher_macro_f1']} 个种子的五折平均 Macro-F1 上升。", '',
        'TC-02 衡量证据头及单视图损失相对普通 CE 的效果，尚未使用 CCML 一致/互补证据融合或完整联合目标。保留全部结果，不根据本轮测试分数修改已固定的训练参数。', '',
        '## 种子与合并测试预测', '',
        '| 种子 | TC-02 五折均值 F1 | 相对 TC-01 变化 | 合并 37 人 F1 | 合并 NLL | 合并 Brier | ECE5 |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for seed, item in summary['seed_reports'].items():
        pooled = item['pooled_oof']
        lines.append(f"| {seed} | {item['unweighted_fold_mean']['macro_f1']:.6f} | {per_seed_delta[seed]:+.6f} | {pooled['macro_f1']:.6f} | {pooled['nll']:.6f} | {pooled['brier']:.6f} | {pooled['ece5']:.6f} |")
    lines.extend(['', '![TC-02 可靠性图](reliability.png)', '',
        'ECE 使用 MDD 概率与实际 MDD 比例的 5 个等宽箱。合并 37 人的指标与五折平均分开报告；图中标注每箱人数。', '',
        '## 不确定性与模态子集诊断', '',
        '每个证据头均有 u=2/sum(alpha)。正确/错误预测的 u 分布和低 u 优先保留的风险—覆盖率曲线按单模态、单种子报告。最终概率等权平均没有唯一的 Dirichlet 总证据，e_final/u_final 留空，不用平均分支 u 冒充融合不确定性。', '',
        '![TC-02 单模态不确定性](branch_uncertainty.png)', '',
        '下表仅对同一组 TC-02 固定输出更换参与概率平均的模态，按实际模态数量除法；没有重训，也没有执行 TC-03 的证据均值、证据求和或 CCML 推理。完整三模态主结果始终保留。', '',
        '| 子集 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |', '|---|---:|---:|---:|---:|---:|'])
    for name, result in subsets.items():
        values = result['primary_seed_mean_of_fold_means']
        lines.append('| ' + name + ' | ' + ' | '.join(f'{values[m]:.6f}' for m in metrics) + ' |')
    lines.extend(['', '## 解释边界与产物', '',
        '本轮是此前已查看过测试结果的固定划分开发比较，独立受试者仍为 37 人。种子或折不能视为新增受试者；小样本校准图和不确定性曲线仅作描述，u 不是出错概率或临床风险。父实验的语音随机编码器、EEG 尺度及通道对应限制继续保留。', '',
        '- `../../runs/TC-02-37-fulltext/seed_<seed>/fold_<fold>/`：15 个权重、训练曲线和逐人证据记录。',
        '- `predictions.jsonl`、`summary.json`、`verification.json`：合并记录、指标和配对验收。',
        '- `subset_diagnostics.json`：固定输出的概率平均子集诊断；单模态 u 分布与风险覆盖率数值在 `summary.json` 中。',
        '- TC-03 及后续阶段尚未执行。TC-03 可直接读取本次 TC-02 权重，无需再次训练。', ''])
    (OUTPUT / 'TC-02实验完成报告.md').write_text('\n'.join(lines), encoding='utf-8')
    return comparison


def main():
    context = RecordContext()
    rows, control_rows, audit, validation = verify(context)
    summary, control = summarize(rows), summarize(control_rows)
    branches, subsets = diagnostics(rows)
    baseline_rows = read_rows(ROOT / 'reports/tc00_records.jsonl')
    baselines = {}
    for experiment in ('TC-00-weighted-37-fulltext', 'TC-00-mean-37-fulltext'):
        selected = [row for row in baseline_rows if row['experiment'] == experiment]
        validate_prediction_records(selected, experiment, context=context)
        baselines[experiment] = summarize(selected)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures(summary, branches)
    comparison = report(summary, control, baselines, branches, subsets)
    write(OUTPUT / 'verification.json', {'passed': True, 'validation': validation, 'checkpoints_verified': 15,
        'prediction_records_reproduced_exactly': 555, 'maximum_probability_error': 0., 'paired_TC01_jobs': 15,
        'independent_metrics_verifier': 'sklearn', 'config_sha256': context.config_sha256,
        'completion_script_sha256': digest(Path(__file__)), 'shared_helpers_sha256': digest(ROOT / 'complete_tc01.py'), 'folds': audit})
    write(OUTPUT / 'summary.json', {'experiment': EXPERIMENT, 'fusion': summary, 'branches': branches,
        'TC01_control': control, 'comparison': comparison, 'historical_baselines': baselines})
    write(OUTPUT / 'subset_diagnostics.json', subsets)
    path = OUTPUT / 'predictions.jsonl'
    text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
    if path.exists():
        require(path.read_text(encoding='utf-8') == text, 'Existing predictions differ')
    else:
        path.write_text(text, encoding='utf-8')
    status = read_json(ROOT / 'reports/experiment_status.json')
    status['utc'] = datetime.now(timezone.utc).isoformat()
    status['TC-02'] = {'status': 'complete', 'experiment': EXPERIMENT, 'completed_jobs': 15,
        'completed_epochs_per_job': 60, 'verified': True, 'paired_TC01_verified': True,
        'report': 'reports/tc02/TC-02实验完成报告.md', 'primary_metrics': summary['primary_seed_mean_of_fold_means']}
    (ROOT / 'reports/experiment_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'completed_jobs': 15, 'primary_metrics': summary['primary_seed_mean_of_fold_means'],
                     'comparison': comparison, 'report': str(OUTPUT / 'TC-02实验完成报告.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
