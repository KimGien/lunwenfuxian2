"""Audit independently trained full CCML and publish the fixed TC00--04 matrix."""
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
from .complete_tc01 import read_rows, write
from .complete_tc03 import metric_crosscheck
from .metrics import summarize
from .records import ROOT, MODALITIES, RecordContext, digest, read_json, require, validate_prediction_records
from .train import load_features, make_initial_state, predict_records, run_directory, state_digest, validate_config

EXPERIMENT = 'TC-04-37-fulltext'
OUTPUT = ROOT / 'reports/tc04'
METRICS = ('macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier')
PAIRED_FIELDS = ('seed', 'fold', 'cohort_id', 'cohort_sha256', 'folds_sha256', 'train_ids',
    'feature_manifest_sha256', 'feature_npz_sha256', 'parent_checkpoints', 'training_config_sha256',
    'initialization_sha256', 'batch_order_sha256', 'completed_epochs', 'optimizer_updates', 'selection',
    'trainable_parameters', 'upstream_mode', 'optimizer', 'class_weights', 'resampling', 'device',
    'torch_version', 'runner_sha256', 'ccml_sha256', 'records_sha256')


def verify(context):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = validate_config(context)
    earlier = {}
    for stage in ('TC-01', 'TC-02'):
        previous = read_json(ROOT / f'reports/{stage.lower().replace("-", "")}/verification.json')
        require(previous['passed'], 'Earlier stage verification failed')
        earlier[stage] = {(item['seed'], item['fold']): item for item in previous['folds']}
    rows, audits = [], []
    controls = {stage: [] for stage in earlier}
    for seed in config['seeds']:
        initial = state_digest(make_initial_state(seed))
        for fold in config['folds']:
            data = load_features(context, seed, fold)
            directory = run_directory(ROOT, EXPERIMENT, seed, fold)
            checkpoint = directory / 'checkpoint.pt'
            checksum = digest(checkpoint)
            run = read_json(directory / 'run.json')
            require(run['status'] == 'complete' and run['training_performed'], 'Incomplete TC04 run')
            require((run['experiment'], run['seed'], run['fold']) == (EXPERIMENT, seed, fold), 'Run identity mismatch')
            require(Path(run['checkpoint_path']).resolve() == checkpoint.resolve(), 'Wrong checkpoint path')
            require(not (directory / 'failure.json').exists(), 'TC04 failure record present')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            metadata = saved['metadata']
            sidecar = read_json(checkpoint.with_suffix('.json'))
            require(metadata == sidecar['metadata'] == run['source_metadata'], 'Checkpoint metadata differs')
            require(checksum == sidecar['checkpoint_sha256'] == run['checkpoint_sha256'], 'Checkpoint digest differs')
            train_ids = context.folds[f'fold_{fold}']['train']
            generator = torch.Generator().manual_seed(seed)
            batch_hash = hashlib.sha256()
            for _ in range(config['epochs']):
                batch_hash.update(torch.randperm(len(train_ids), generator=generator).numpy().astype('<i8').tobytes())
            expected = {'experiment': EXPERIMENT, 'seed': seed, 'fold': fold,
                'cohort_id': context.lock['cohort_id'], 'cohort_sha256': context.cohort_sha256,
                'folds_sha256': context.folds_sha256, 'train_ids': train_ids,
                'feature_manifest_sha256': data['manifest_hash'], 'feature_npz_sha256': data['manifest']['npz_sha256'],
                'parent_checkpoints': data['manifest']['parent_checkpoints'],
                'training_config_sha256': context.config_sha256, 'initialization_sha256': initial,
                'batch_order_sha256': batch_hash.hexdigest(), 'completed_epochs': 60,
                'optimizer_updates': 60 * math.ceil(len(train_ids) / 8), 'selection': 'last_epoch',
                'trainable_parameters': 774, 'optimizer': 'Adamax', 'class_weights': None, 'resampling': False,
                'upstream_mode': 'frozen_eval_cached_features', 'device': 'cpu', 'torch_version': str(torch.__version__),
                'effective_loss_parameters': {'rho': 1.25, 'delta': 1., 'gamma': 1., 'anneal_epochs': 50},
                'runner_sha256': digest(ROOT / 'train.py'), 'ccml_sha256': digest(ROOT / 'ccml.py'),
                'records_sha256': digest(ROOT / 'records.py')}
            for key, value in expected.items():
                require(metadata.get(key) == value, f'TC04 protocol mismatch: {key}')
            history = read_json(directory / 'train_history.json')
            require([item['epoch'] for item in history] == list(range(1, 61)), 'Training epochs incomplete')
            require(all(math.isfinite(item['train_loss']) and item['train_loss'] >= 0 for item in history), 'Invalid training loss')
            paired = {}
            for stage in controls:
                control_dir = run_directory(ROOT, f'{stage}-37-fulltext', seed, fold)
                control_run = read_json(control_dir / 'run.json')
                require(control_run['status'] == 'complete' and control_run['training_performed'], 'Control incomplete')
                control_meta = control_run['source_metadata']
                require(all(metadata[k] == control_meta[k] for k in PAIRED_FIELDS), f'{stage} paired protocol differs')
                for filename, field in (('checkpoint.pt', 'checkpoint_sha256'), ('predictions.jsonl', 'predictions_sha256'),
                                        ('train_history.json', 'history_sha256')):
                    require(digest(control_dir / filename) == earlier[stage][seed, fold][field], f'{stage} artifact changed: {filename}')
                control_saved = torch.load(control_dir / 'checkpoint.pt', map_location='cpu', weights_only=True)
                require(state_digest(saved['state_dict']) != state_digest(control_saved['state_dict']), 'TC04 equals a trained control state')
                controls[stage].extend(read_rows(control_dir / 'predictions.jsonl'))
                paired[stage] = {'protocol_equal': True, 'artifacts_match_previous_verification': True,
                                 'checkpoint_sha256': control_run['checkpoint_sha256']}
            require(state_digest(saved['state_dict']) != initial, 'Weights never changed from initialization')
            model = ThreeViewHeads().eval()
            model.load_state_dict(saved['state_dict'], strict=True)
            recorded = read_rows(directory / 'predictions.jsonl')
            require(len(recorded) == 37 and {r['subject_id'] for r in recorded} == set(context.ids), 'Incomplete fold coverage')
            reproduced = predict_records(context, data, model, EXPERIMENT, seed, fold, checkpoint, metadata, torch.device('cpu'))
            require(recorded == reproduced, 'Reloaded evidence/prediction records differ')
            validate_prediction_records(recorded, EXPERIMENT, context=context, require_complete=False)
            rows.extend(recorded)
            audits.append({'seed': seed, 'fold': fold, 'checkpoint_sha256': checksum,
                'predictions_sha256': digest(directory / 'predictions.jsonl'), 'history_sha256': digest(directory / 'train_history.json'),
                'initialization_sha256': initial, 'batch_order_sha256': batch_hash.hexdigest(),
                'epochs': 60, 'optimizer_updates': metadata['optimizer_updates'], 'paired_controls': paired,
                'all_37_records_reproduced_exactly': True, 'final_state_differs_from_initial_and_controls': True,
                'first_train_loss': history[0]['train_loss'], 'last_train_loss': history[-1]['train_loss']})
    require(len(rows) == 555 and len(audits) == 15, 'Planned TC04 jobs incomplete')
    validation = validate_prediction_records(rows, EXPERIMENT, context=context)
    metric_crosscheck(rows)
    matrix = {}
    baseline_rows = read_rows(ROOT / 'reports/tc00_records.jsonl')
    for name in ('weighted', 'mean'):
        exp = f'TC-00-{name}-37-fulltext'
        part = [r for r in baseline_rows if r['experiment'] == exp]
        validate_prediction_records(part, exp, context=context)
        matrix[f'TC-00-{name}'] = summarize(part)
    for stage, part in controls.items():
        validate_prediction_records(part, f'{stage}-37-fulltext', context=context)
        matrix[stage] = summarize(part)
    previous_tc03 = read_json(ROOT / 'reports/tc03/verification.json')
    require(previous_tc03['passed'], 'TC03 verification failed')
    for name in ('mean', 'sum', 'ccml'):
        part = []
        exp = f'TC-03-{name}-37-fulltext'
        for item in previous_tc03['jobs']:
            if item['experiment'] != exp:
                continue
            path = run_directory(ROOT, exp, item['seed'], item['fold']) / 'predictions.jsonl'
            require(digest(path) == item['predictions_sha256'], 'TC03 predictions changed')
            part.extend(read_rows(path))
        validate_prediction_records(part, exp, context=context)
        matrix[f'TC-03-{name}'] = summarize(part)
    matrix['TC-04'] = summarize(rows)
    verification = {'passed': True, 'completed_training_jobs': 15, 'completed_epochs_per_job': 60,
        'total_optimizer_updates': sum(a['optimizer_updates'] for a in audits), 'validation': validation,
        'checkpoints_verified': 15, 'prediction_records_reproduced_exactly': 555, 'maximum_probability_error': 0.,
        'paired_TC01_jobs': 15, 'paired_TC02_jobs': 15, 'paired_fields_checked': list(PAIRED_FIELDS),
        'TC01_TC02_checkpoint_prediction_history_files_unchanged': 90, 'TC03_prediction_files_unchanged': 45,
        'independent_metrics_verifier': 'sklearn', 'config_sha256': context.config_sha256,
        'completion_script_sha256': digest(__file__),
        'helper_sha256': {name: digest(ROOT / name) for name in ('complete_tc01.py', 'complete_tc03.py', 'metrics.py')},
        'folds': audits}
    return rows, matrix, verification


def branch_summaries(rows):
    branches = {}
    for modality in MODALITIES:
        derived = deepcopy(rows)
        for row in derived:
            row['p_final'] = row['views'][modality]['p']
            row['prediction'] = int(row['p_final'][1] > .5)
            row['u_final'] = row['views'][modality]['u']
        metric_crosscheck(derived)
        branches[modality] = summarize(derived)
        branches[modality].update(output=modality, interpretation='single branch of jointly trained TC04; no separate retraining')
    return branches


def comparisons(matrix):
    main = matrix['TC-04']
    differences = {}
    for name in ('TC-01', 'TC-02', 'TC-03-ccml'):
        differences[name] = {
            'primary_metric_differences_TC04_minus_reference': {
                key: main['primary_seed_mean_of_fold_means'][key] - matrix[name]['primary_seed_mean_of_fold_means'][key]
                for key in METRICS},
            'per_seed_macro_f1_differences': {
                seed: data['unweighted_fold_mean']['macro_f1'] - matrix[name]['seed_reports'][seed]['unweighted_fold_mean']['macro_f1']
                for seed, data in main['seed_reports'].items()}}
    ce = differences['TC-01']
    delta = ce['primary_metric_differences_TC04_minus_reference']
    positive_seeds = sum(value > 0 for value in ce['per_seed_macro_f1_differences'].values())
    checks = {'mean_macro_f1_increase_at_least_0_01': delta['macro_f1'] >= .01,
              'at_least_2_of_3_seeds_increase': positive_seeds >= 2,
              'mdd_recall_decrease_no_more_than_0_05': delta['mdd_recall'] >= -.05}
    return {'references': differences, 'predeclared_engineering_reference': {
        'reference': 'TC-01', 'checks': checks, 'met': all(checks.values()),
        'seeds_with_higher_macro_f1': positive_seeds, 'nll_worsened': delta['nll'] > 0,
        'brier_worsened': delta['brier'] > 0, 'statistical_significance_claim': False}}


def figures(matrix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = matrix['TC-04']
    colors = {'TC-03-ccml': '#79859b', 'TC-04': '#247764'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (seed, values) in zip(axes, summary['seed_reports'].items()):
        pooled = values['pooled_oof']
        bins = [b for b in pooled['bins'] if b['n']]
        ax.plot([0, 1], [0, 1], '--', color='#8b969f', linewidth=1)
        ax.plot([b['mean_p_mdd'] for b in bins], [b['observed_mdd_rate'] for b in bins], 'o-', color=colors['TC-04'])
        for b in bins:
            ax.annotate(f"n={b['n']}", (b['mean_p_mdd'], b['observed_mdd_rate']), xytext=(4, 7), textcoords='offset points', fontsize=8)
        ax.set(xlim=(-.05, 1.05), ylim=(-.05, 1.10), xlabel='Mean predicted P(MDD)', ylabel='Observed MDD rate',
               title=f"Seed {seed} | ECE5 = {pooled['ece5']:.4f}")
        ax.grid(alpha=.18)
    fig.suptitle('TC-04 full CCML | 37 pooled OOF subjects per seed | 5 equal-width bins')
    fig.savefig(OUTPUT / 'reliability.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, seed in enumerate(summary['seed_reports']):
        for index, name in enumerate(colors):
            uncertainty = matrix[name]['seed_reports'][seed]['uncertainty']
            for outcome, shift, marker in [('correct_u', -.12, 'o'), ('incorrect_u', .12, 'x')]:
                values = uncertainty[outcome]
                offsets = np.linspace(-.05, .05, len(values)) if len(values) > 1 else np.zeros(len(values))
                axes[0, col].scatter(index + shift + offsets, values, color=colors[name], marker=marker, s=26, alpha=.75,
                    label=('Correct' if outcome == 'correct_u' else 'Incorrect') if index == 1 else None)
            curve = uncertainty['risk_coverage']
            axes[1, col].step([p['coverage'] for p in curve], [p['risk'] for p in curve], where='post',
                             color=colors[name], label=name)
        axes[0, col].set(xticks=[0, 1], xticklabels=['TC-03 CCML', 'TC-04 full CCML'], xlim=(-.4, 1.4), ylim=(0, 1.03),
                        ylabel='Fused evidence uncertainty u', title=f'Seed {seed} | uncertainty')
        axes[0, col].legend(fontsize=8, loc='upper right')
        axes[1, col].set(xlim=(0, 1.02), ylim=(0, 1.03), xlabel='Coverage', ylabel='Classification error rate',
                        title='Retain low-u subjects; ties retained together')
        axes[1, col].legend(fontsize=8)
        for ax in axes[:, col]:
            ax.grid(alpha=.16)
    fig.suptitle('Fixed-split development diagnostics | 37 independent subjects | u is not error probability')
    fig.savefig(OUTPUT / 'uncertainty.png', dpi=160)
    plt.close(fig)


def write_report(matrix, branches, comparison):
    summary = matrix['TC-04']
    delta_ce = comparison['references']['TC-01']['primary_metric_differences_TC04_minus_reference']
    delta_ccml = comparison['references']['TC-03-ccml']['primary_metric_differences_TC04_minus_reference']
    engineering = comparison['predeclared_engineering_reference']
    lines = ['# TC-04 实验完成报告', '',
        'TC-04-37-fulltext 已完成 3 个种子 × 5 折，共 15 组完整 CCML 训练。权重重载、证据计算、逐人预测、配对训练预算和独立指标复算全部通过，没有当前阻塞。', '',
        f"主口径 Macro-F1 为 {summary['primary_seed_mean_of_fold_means']['macro_f1']:.6f}，相对同预算 TC-01 变化 {delta_ce['macro_f1']:+.6f}（{delta_ce['macro_f1']*100:+.2f} 个百分点）。本轮完整候选没有改善直接 CE 对照，未达到计划预设的继续投入工程参考。结果按原设置保留，未追加调参或挑选种子。", '',
        '## 训练协议与验收', '',
        '- MODMA-37-v1：37 人，16 MDD、21 HC；种子 11、22、33，原五折。只有对应折的 train 标签参与优化。',
        '- 使用同折同种子父模型的冻结表征；独立初始化三个 128→2 证据头，共 774 个参数，Softplus 输出非负证据。',
        '- 与 TC-01、TC-02 采用相同初始权重和 batch 顺序独立训练。TC-04 从共同初始状态开始，没有在已训练 TC-02 上续训。',
        '- 每折 60 轮、batch 8；Adamax lr=0.001、weight decay=0、梯度范数上限 1；CPU 单线程。每折 240 次更新，合计 3,600 次。',
        '- 固定 rho=1.25、delta=1、gamma=1；KL 系数 lambda=min(1,epoch/50)。只保存最后一轮；阈值 0.5，平局为 HC；无验证集搜索、早停、类别权重、重采样或校准拟合。', '',
        '完整目标为：`sum_v[D(alpha_v)+R(alpha_v)] + D(alpha_con) + delta*R(alpha_con) + gamma*D(alpha_cmp)`，其中 R 是乘以 lambda 的屏蔽真类证据 KL。没有互补项 KL，也没有额外最终融合 CE。', '',
        '推理先按类别取各模态证据的最小值 m；互补证据使用 mean(e_v−m)，公共证据按 rho 增强并保持总量后乘实际视图数 V=3，两者相加得到 e_final。最终 p=alpha/sum(alpha)，u=2/sum(alpha)。', '',
        '- 15 个权重、60 轮有限训练损失及来源哈希全部通过核验；初始化、缓存、训练名单、代码、优化器、更新预算与 TC-01/02 逐折一致。',
        '- 重载后 555 行完整证据与预测记录完全复现；测试共 111 行，每个种子 37 人各测试一次，独立受试者数仍为 37。',
        '- Macro-F1、MDD recall、balanced accuracy、NLL、Brier 均逐折与 sklearn 复算一致；单分支指标也已交叉核验。',
        '- TC-01/02 的 90 个权重、预测和训练曲线文件，以及 TC-03 的 45 个预测文件，与此前验收哈希一致。', '',
        '## TC-00 至 TC-04 完整结果矩阵', '',
        '主口径为每种子的五折指标均值，再取三个种子均值。NLL、Brier 越低越好。TC-00 原验证集选权融合是历史基线，TC-01 才是同新增训练预算的直接对照。', '',
        '| 方法 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |',
        '|---|---:|---:|---:|---:|---:|']
    titles = {'TC-00-weighted': 'TC-00 原验证集选权', 'TC-00-mean': 'TC-00 原概率等权',
              'TC-01': 'TC-01 CE 概率等权', 'TC-02': 'TC-02 证据头概率等权',
              'TC-03-mean': 'TC-03 证据均值', 'TC-03-sum': 'TC-03 证据求和',
              'TC-03-ccml': 'TC-03 CCML（复用 TC-02）', 'TC-04': 'TC-04 完整 CCML'}
    for name, value in matrix.items():
        values = value['primary_seed_mean_of_fold_means']
        lines.append('| ' + titles[name] + ' | ' + ' | '.join(f'{values[k]:.6f}' for k in METRICS) + ' |')
    lines.extend(['', 'TC-04 减去对应对照的变化：', '',
        '| 对照 | Macro-F1 变化 | MDD recall 变化 | Balanced accuracy 变化 | NLL 变化 | Brier 变化 |',
        '|---|---:|---:|---:|---:|---:|'])
    for name, value in comparison['references'].items():
        d = value['primary_metric_differences_TC04_minus_reference']
        lines.append('| ' + name + ' | ' + ' | '.join(f'{d[k]:+.6f}' for k in METRICS) + ' |')
    lines.extend(['',
        f"相对仅改变推理规则的 TC-03 CCML，完整训练的 Macro-F1 变化 {delta_ccml['macro_f1']:+.6f}，但 NLL 变化 {delta_ccml['nll']:+.6f}、Brier 变化 {delta_ccml['brier']:+.6f}。这一小幅分类变化未伴随概率质量改善，也未超过 TC-01 或 TC-02。", '',
        '## 预设工程参考', '',
        '| 条件（相对 TC-01） | 本次结果 | 是否满足 |', '|---|---|---|',
        f"| 平均 Macro-F1 至少提高 0.01 | {delta_ce['macro_f1']:+.6f} | {'是' if engineering['checks']['mean_macro_f1_increase_at_least_0_01'] else '否'} |",
        f"| 至少 2/3 种子 Macro-F1 同向提高 | {engineering['seeds_with_higher_macro_f1']}/3 个种子提高 | {'是' if engineering['checks']['at_least_2_of_3_seeds_increase'] else '否'} |",
        f"| MDD recall 平均下降不超过 0.05 | {delta_ce['mdd_recall']:+.6f} | {'是' if engineering['checks']['mdd_recall_decrease_no_more_than_0_05'] else '否'} |",
        f"| 披露 NLL / Brier 是否恶化 | NLL {delta_ce['nll']:+.6f}；Brier {delta_ce['brier']:+.6f}，均恶化 | 已披露 |", '',
        '预设参考未满足。该参考是开发阶段的工程判断，不是统计显著性检验；本轮不支持将完整 CCML 描述为已验证的改进。后续 TC-05 可用于机制核验，但本次没有启动。', '',
        '## 单模态与完整融合', '',
        '单模态值来自同一批完整 CCML 模型的各分支输出，并非另行训练的单模态模型。', '',
        '| 输出 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |', '|---|---:|---:|---:|---:|---:|'])
    for name, value in list(branches.items()) + [('完整 E+S+T', summary)]:
        values = value['primary_seed_mean_of_fold_means']
        lines.append('| ' + name + ' | ' + ' | '.join(f'{values[k]:.6f}' for k in METRICS) + ' |')
    lines.extend(['',
        '计划要求的 TC-02 固定输出模态子集、实际 V=2/3 解耦和预设 EEG 证据倍数扰动已经在 [TC-03 报告](../tc03/TC-03实验完成报告.md) 中完整报告。本次保留三模态候选，不依据测试结果更换输入组合。', '',
        '## 分种子与合并测试预测', '',
        '| 种子 | TC-04 五折均值 F1 | 相对 TC-01 | 相对 TC-03 CCML | 合并 37 人 F1 | 合并 NLL | 合并 Brier | ECE5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for seed, item in summary['seed_reports'].items():
        p = item['pooled_oof']
        d1 = comparison['references']['TC-01']['per_seed_macro_f1_differences'][seed]
        d3 = comparison['references']['TC-03-ccml']['per_seed_macro_f1_differences'][seed]
        lines.append(f"| {seed} | {item['unweighted_fold_mean']['macro_f1']:.6f} | {d1:+.6f} | {d3:+.6f} | {p['macro_f1']:.6f} | {p['nll']:.6f} | {p['brier']:.6f} | {p['ece5']:.6f} |")
    lines.extend(['', '合并 37 人指标与五折均值分开报告，不用合并 F1 替换主口径。', '',
        '![TC-04 可靠性图](reliability.png)', '',
        'ECE 使用 MDD 概率与实际 MDD 比例的 5 个等宽箱；每箱人数在图中标出。该 ECE 不是最高类别置信度的 ECE。', '',
        '## 不确定性与风险—覆盖率', '', '![TC-04 不确定性诊断](uncertainty.png)', '',
        '图中按种子展示 TC-04 与 TC-03 CCML 的正确/错误预测 u 分布，以及优先保留低 u 受试者时的分类错误率；相同 u 的记录一起保留。对应数值及单分支诊断保存在 summary.json。较小 u 本身不是性能改善，u 也不是出错概率或临床风险。', '',
        '## 每折主结果', '',
        '| 种子 | 折 | 测试人数 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |',
        '|---|---|---:|---:|---:|---:|---:|---:|'])
    for seed, item in summary['seed_reports'].items():
        for fold, values in item['fold_metrics'].items():
            lines.append(f"| {seed} | {fold} | {values['n_subjects']} | " + ' | '.join(f'{values[k]:.6f}' for k in METRICS) + ' |')
    lines.extend(['', '## 产物与解释边界', '',
        '- `../../runs/TC-04-37-fulltext/seed_<seed>/fold_<fold>/`：15 个权重、训练记录、证据及预测。',
        '- `predictions.jsonl`：555 行合并完整记录；`summary.json`：主结果、单分支、完整矩阵、比较和工程参考。',
        '- `verification.json`：15 折来源与重载核验，以及 TC-01/02 的配对一致性与既有产物保留检查。',
        '- 上一级 `tc04_training.log`：本次 15 组执行记录。`tc04_launch_preflight.json` 仅为训练前快照，不能代替完成状态。',
        '- TC-00 至 TC-04 均已完成；TC-05 尚未开始，TC-06 仍为第二阶段保留项。', '',
        '本轮沿用已查看过测试结果的固定划分，属于开发性比较；37 人以外没有新增独立人群。校准、不确定性和种子差异仅作描述，不宣称独立确认或临床适用性。父实验的语音随机冻结编码器、单位置注意力，以及 EEG 尺度和通道对应限制继续保留，不能通过 CCML 结果消除这些限制。', ''])
    (OUTPUT / 'TC-04实验完成报告.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    context = RecordContext()
    rows, matrix, verification = verify(context)
    branches = branch_summaries(rows)
    comparison = comparisons(matrix)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures(matrix)
    write_report(matrix, branches, comparison)
    write(OUTPUT / 'verification.json', verification)
    write(OUTPUT / 'summary.json', {'experiment': EXPERIMENT, 'fusion': matrix['TC-04'],
        'branches': branches, 'TC00_to_TC04_matrix': matrix, 'comparison': comparison})
    path = OUTPUT / 'predictions.jsonl'
    content = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content, 'Existing combined predictions differ')
    else:
        path.write_text(content, encoding='utf-8')
    status = read_json(ROOT / 'reports/experiment_status.json')
    status['utc'] = datetime.now(timezone.utc).isoformat()
    status['TC-04'] = {'status': 'complete', 'experiment': EXPERIMENT, 'completed_jobs': 15,
        'completed_epochs_per_job': 60, 'verified': True, 'paired_TC01_TC02_verified': True,
        'report': 'reports/tc04/TC-04实验完成报告.md',
        'primary_metrics': matrix['TC-04']['primary_seed_mean_of_fold_means'],
        'predeclared_engineering_reference_met': comparison['predeclared_engineering_reference']['met']}
    (ROOT / 'reports/experiment_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'TC04': status['TC-04'], 'comparison': comparison}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
