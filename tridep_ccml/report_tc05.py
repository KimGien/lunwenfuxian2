"""Render the predeclared TC-05 ablations from already verified summaries.

This module performs no training, inference, metric fitting, or status mutation.
"""
from __future__ import annotations

import json

import numpy as np

from .records import ROOT


OUTPUT = ROOT / 'reports/tc05'
METRICS = ('macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier')
VARIANTS = ('a', 'b', 'c')
NAMES = {'a': 'TC-05a（rho=1）', 'b': 'TC-05b（delta=0）', 'c': 'TC-05c（gamma=0）'}
COLORS = {'TC-04': '#58677f', 'a': '#287d71', 'b': '#bd722a', 'c': '#8f55a5'}


def _numbers(values, signed=False):
    return ' | '.join(format(values[key], '+.6f' if signed else '.6f') for key in METRICS)


def _direction(value, lower_is_better=False):
    if abs(value) < 1e-12:
        return '持平'
    improved = value < 0 if lower_is_better else value > 0
    return '改善' if improved else '下降' if not lower_is_better else '变差'


def _figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    seeds = sorted(summary['variants']['a']['seed_reports'], key=int)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, variant in enumerate(VARIANTS):
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            pooled = summary['variants'][variant]['seed_reports'][seed]['pooled_oof']
            bins = [item for item in pooled['bins'] if item['n']]
            ax.plot([0, 1], [0, 1], '--', color='#9ba4af', linewidth=1)
            ax.plot([item['mean_p_mdd'] for item in bins],
                    [item['observed_mdd_rate'] for item in bins], 'o-', color=COLORS[variant])
            for item in bins:
                ax.annotate(f"n={item['n']}", (item['mean_p_mdd'], item['observed_mdd_rate']),
                            xytext=(4, 7), textcoords='offset points', fontsize=8)
            ax.set(xlim=(-.05, 1.05), ylim=(-.05, 1.12),
                   title=f"TC-05{variant} | seed {seed} | ECE5={pooled['ece5']:.4f}",
                   xlabel='Mean predicted P(MDD)', ylabel='Observed MDD rate')
            ax.grid(alpha=.17)
    fig.suptitle('TC-05 fixed ablations | 37 pooled OOF subjects per seed | 5 equal-width bins', fontsize=13)
    fig.savefig(OUTPUT / 'reliability.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    groups = [('TC-04', summary['controls']['TC-04'])] + [(variant, summary['variants'][variant]) for variant in VARIANTS]
    for col, seed in enumerate(seeds):
        for index, (name, group) in enumerate(groups):
            uncertainty = group['seed_reports'][seed]['uncertainty']
            for outcome, shift, marker in [('correct_u', -.12, 'o'), ('incorrect_u', .12, 'x')]:
                values = uncertainty[outcome]
                offsets = np.linspace(-.045, .045, len(values)) if len(values) > 1 else np.zeros(len(values))
                axes[0, col].scatter(index + shift + offsets, values, color=COLORS[name], marker=marker,
                                     s=25, alpha=.72)
            curve = uncertainty['risk_coverage']
            axes[1, col].step([item['coverage'] for item in curve], [item['risk'] for item in curve],
                              where='post', color=COLORS[name], label=name if name == 'TC-04' else f'TC-05{name}')
        axes[0, col].set(xticks=range(4), xticklabels=['TC-04', '05a', '05b', '05c'],
                        xlim=(-.45, 3.45), ylim=(0, 1.03), ylabel='Fused uncertainty u',
                        title=f'Seed {seed} | correct / incorrect predictions')
        axes[0, col].legend(handles=[
            Line2D([], [], color='#555555', marker='o', linestyle='None', label='Correct'),
            Line2D([], [], color='#555555', marker='x', linestyle='None', label='Incorrect')],
            loc='upper right', fontsize=8)
        axes[1, col].set(xlim=(0, 1.02), ylim=(0, 1.03), xlabel='Coverage', ylabel='Classification error rate',
                        title='Retain low-u subjects; ties retained together')
        axes[1, col].legend(fontsize=8, ncol=2)
        for ax in axes[:, col]:
            ax.grid(alpha=.17)
    fig.suptitle('TC-04 vs TC-05 ablations | 37 independent subjects | u is not error probability', fontsize=13)
    fig.savefig(OUTPUT / 'uncertainty.png', dpi=160)
    plt.close(fig)


def _report(summary, verification):
    variants, controls = summary['variants'], summary['controls']
    comparisons = summary['comparisons']
    lines = ['# TC-05 实验完成报告', '',
        f"TC-05a、TC-05b、TC-05c 已各完成 3 个种子 × 5 折，共 {verification['completed_training_jobs']} 组独立训练，每组 60 轮。权重来源、训练预算、逐人预测和指标核验通过；当前无阻塞。", '',
        '本轮按原计划核验分离增强、一致 KL 和互补监督的作用。所有消融保留原始结果，不根据测试分数选择种子、阈值或额外参数。TC-06 尚未启动。', '',
        '## 三项消融与训练协议', '',
        '| 实验 | rho | delta | gamma | 相对 TC-04 的唯一改动 |',
        '|---|---:|---:|---:|---|',
        '| TC-04 对照 | 1.25 | 1 | 1 | 完整 CCML |',
        '| TC-05a | 1 | 1 | 1 | 公共证据不做分离增强 |',
        '| TC-05b | 1.25 | 0 | 1 | 删除一致证据的 KL 项 |',
        '| TC-05c | 1.25 | 1 | 0 | 删除互补证据的监督项 |', '',
        '**TC-05c 推理仍包含互补证据。** `gamma=0` 仅改变训练目标，不能解释为删除互补分支。TC-05a 的 rho 同时用于本组训练和推理。', '',
        '完整训练目标为 `sum_v[D(alpha_v)+R(alpha_v)] + D(alpha_con) + delta*R(alpha_con) + gamma*D(alpha_cmp)`。R 为乘以 `lambda=min(1,epoch/50)` 的屏蔽真类证据 KL；没有互补 KL，也没有额外最终融合 CE。各消融只改表中一项。', '',
        '删除损失项会改变训练目标的组成和数值尺度，各消融及 TC-04 的训练 loss 不能直接横向比较。较低的训练 loss 不等于泛化更好；性能比较使用下文相同口径的测试指标。', '',
        '- MODMA-37-v1，37 人（16 MDD、21 HC），原五折，种子 11、22、33。只有相应折的训练标签参与优化。',
        '- 使用同折同种子父模型的冻结 128 维表征，三路独立 128→2 证据头，合计 774 个可训练参数，Softplus 输出证据。',
        '- 三项消融均从与 TC-01、TC-02、TC-04 相同的初始权重独立训练，采用相同 batch 顺序；没有从 TC-04 成品权重接着训练。',
        '- 每组 60 轮，batch=8；Adamax lr=0.001、weight decay=0、梯度范数上限 1；CPU 单线程。每组 240 次更新，每项消融 3,600 次更新。',
        f"- 45 组总更新数为 {verification['total_optimizer_updates']:,}；固定最后一轮；阈值 0.5，平局为 HC；无验证集搜索、早停、类别权重、重采样或校准拟合。", '',
        '## 主结果与对照', '',
        '主口径为每个种子先对五折指标等权平均，再对三个种子等权平均。Macro-F1、MDD recall、balanced accuracy 越高越好，NLL、Brier 越低越好。', '',
        '| 实验 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ('TC-01', 'TC-02', 'TC-04'):
        lines.append('| ' + key + ' | ' + _numbers(controls[key]['primary_seed_mean_of_fold_means']) + ' |')
    for variant in VARIANTS:
        lines.append('| ' + NAMES[variant] + ' | ' + _numbers(variants[variant]['primary_seed_mean_of_fold_means']) + ' |')
    lines.extend(['', '下表为消融减去对照；NLL/Brier 为负值表示改善。', '',
        '| 消融 | 对照 | Δ Macro-F1 | Δ MDD recall | Δ Balanced accuracy | Δ NLL | Δ Brier |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for variant in VARIANTS:
        for reference in ('TC-04', 'TC-01'):
            values = comparisons[variant][reference]['primary_metric_differences']
            lines.append(f'| TC-05{variant} | {reference} | ' + _numbers(values, signed=True) + ' |')
    lines.extend(['', '## 机制核验解读', ''])
    mechanisms = {'a': '取消分离增强', 'b': '删除一致 KL', 'c': '删除互补监督'}
    for variant in VARIANTS:
        d4 = comparisons[variant]['TC-04']['primary_metric_differences']
        d1 = comparisons[variant]['TC-01']['primary_metric_differences']
        positive = comparisons[variant]['TC-04']['seeds_with_higher_macro_f1']
        f1_effect = _direction(d4['macro_f1'])
        lines.append(f"- **TC-05{variant}，{mechanisms[variant]}：**相对 TC-04，Macro-F1 {f1_effect}，变化 {d4['macro_f1']:+.6f}（{100*d4['macro_f1']:+.2f} 个百分点），{positive}/3 个种子提高；MDD recall 变化 {d4['mdd_recall']:+.6f}，NLL {_direction(d4['nll'], True)}（{d4['nll']:+.6f}），Brier {_direction(d4['brier'], True)}（{d4['brier']:+.6f}）。相对直接 CE 对照 TC-01 的 Macro-F1 变化为 {d1['macro_f1']:+.6f}。")
    b_seed_differences = comparisons['b']['TC-04']['per_seed_macro_f1_differences']
    b_positive = [seed for seed, difference in b_seed_differences.items() if difference > 1e-12]
    b_tied = [seed for seed, difference in b_seed_differences.items() if abs(difference) <= 1e-12]
    if len(b_positive) == 1 and len(b_tied) == 2:
        lines.extend(['', f"TC-05b 的平均 F1 提升仅来自种子 {b_positive[0]}；种子 {'、'.join(b_tied)} 与 TC-04 持平。不能将这一结果解释为删除一致 KL 带来了稳定的机制收益。"])
    lines.extend(['',
        '上述变化描述在当前冻结表征和固定划分中，单项设置改变后的观测结果；每项都重新训练，不能将差异仅归因于固定输出上的推理变化。单项消融不估计各组件之间的交互，也不保证同时删除多个组件会有相同效果。', '',
        '下表沿用计划相对 TC-01 的工程参考：平均 F1 提高至少 0.01、至少 2/3 种子提高、MDD recall 下降不超过 0.05。它是开发阶段参考，不是显著性检验。', '',
        '| 消融 | Δ F1 ≥ 0.01 | 提高种子数 ≥ 2 | Δ recall ≥ −0.05 | 三项全部满足 |',
        '|---|---|---|---|---|'])
    passing = []
    for variant in VARIANTS:
        comp = comparisons[variant]['TC-01']
        delta = comp['primary_metric_differences']
        checks = (delta['macro_f1'] >= .01, comp['seeds_with_higher_macro_f1'] >= 2, delta['mdd_recall'] >= -.05)
        if all(checks):
            passing.append('TC-05' + variant)
        lines.append(f"| TC-05{variant} | {delta['macro_f1']:+.6f}（{'是' if checks[0] else '否'}） | {comp['seeds_with_higher_macro_f1']}/3（{'是' if checks[1] else '否'}） | {delta['mdd_recall']:+.6f}（{'是' if checks[2] else '否'}） | {'是' if all(checks) else '否'} |")
    lines.extend(['',
        ('满足上述三项工程参考的消融为 ' + '、'.join(passing) + '；仍需结合概率质量和独立验证判断，不能据此宣称显著改进。') if passing else '三项消融均未同时满足上述工程参考；本轮不能据此宣称 CCML 已优于直接 CE 对照。', '',
        '本次完成预设机制核验，不在已查看的测试结果上继续追加调参。后续若比较表征修正或联合训练，应建立独立版本和等预算 CE 对照；本次没有启动 TC-06。', '',
        '## 分种子与合并 OOF', '',
        '每个种子的测试预测覆盖 37 人、每人一次。合并 OOF 指标与五折均值是不同统计口径，下表同时展示但不相互替代。', '',
        '| 消融 | 种子 | 五折均值 F1 | 相对 TC-04 | 相对 TC-01 | 合并 37 人 F1 | 合并 NLL | 合并 Brier | ECE5 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for variant in VARIANTS:
        for seed, item in variants[variant]['seed_reports'].items():
            pooled = item['pooled_oof']
            d4 = comparisons[variant]['TC-04']['per_seed_macro_f1_differences'][seed]
            d1 = comparisons[variant]['TC-01']['per_seed_macro_f1_differences'][seed]
            lines.append(f"| TC-05{variant} | {seed} | {item['unweighted_fold_mean']['macro_f1']:.6f} | {d4:+.6f} | {d1:+.6f} | {pooled['macro_f1']:.6f} | {pooled['nll']:.6f} | {pooled['brier']:.6f} | {pooled['ece5']:.6f} |")
    lines.extend(['', '三个种子的合并 OOF 指标均值（单列，不作主结果）：', '',
        '| 消融 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |',
        '|---|---:|---:|---:|---:|---:|'])
    for variant in VARIANTS:
        lines.append(f'| TC-05{variant} | ' + _numbers(variants[variant]['separate_seed_mean_of_pooled_oof']) + ' |')
    lines.extend(['', '## 单模态与完整融合', '',
        '以下单模态值是共同训练的三模态模型中各分支的输出，不是另行重训的单模态模型。', '',
        '| 消融 | 输出 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for variant in VARIANTS:
        for modality in ('eeg', 'speech', 'text'):
            values = summary['branches'][variant][modality]['primary_seed_mean_of_fold_means']
            lines.append(f'| TC-05{variant} | {modality} | ' + _numbers(values) + ' |')
        lines.append(f'| TC-05{variant} | 完整 E+S+T | ' + _numbers(variants[variant]['primary_seed_mean_of_fold_means']) + ' |')
    lines.extend(['', '## 概率校准与不确定性', '',
        '![TC-05 三项消融的可靠性图](reliability.png)', '',
        '按消融和种子分别画出 37 人 OOF 的可靠性图。ECE5 采用 MDD 概率与实际 MDD 比例的 5 个等宽箱；空箱不绘制，图中标出非空箱人数。这不是最高类别置信度的 ECE。', '',
        '![TC-04 与 TC-05 的不确定性和风险覆盖率](uncertainty.png)', '',
        '上排分别展示正确和错误预测的融合 u；下排从低 u 到高 u 保留受试者，报告保留人群中的分类错误率，同值 u 一起纳入。完整数值保存在 summary.json。u 是证据不足程度，不是出错概率或临床风险；不同模型的 u 更小本身不表示预测更好。', '',
        '## 每折主结果', '',
        '| 消融 | 种子 | 折 | 测试人数 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for variant in VARIANTS:
        for seed, item in variants[variant]['seed_reports'].items():
            for fold, values in item['fold_metrics'].items():
                lines.append(f"| TC-05{variant} | {seed} | {fold} | {values['n_subjects']} | " + _numbers(values) + ' |')
    lines.extend(['', '## 验收与文件', '',
        f"- 完成 {verification['completed_training_jobs']} 组训练并核验权重、训练来源及预算；重载后完整复现 {verification['prediction_records_reproduced_exactly']:,} 行证据与预测。",
        '- 每项消融含 555 行全部划分记录，其中测试 111 行；三项共 1,665 行，测试 333 行。不同种子与消融重复使用同一队列，独立受试者仍为 37 人。',
        '- 每折主指标及单分支指标使用独立指标实现交叉核验。验收细节、权重与预测哈希以及配对协议记录见 `verification.json`。',
        '- `../../runs/TC-05a-37-fulltext/`、`../../runs/TC-05b-37-fulltext/`、`../../runs/TC-05c-37-fulltext/`：各自 15 组权重、训练历史、预测和完成记录。',
        '- `predictions_a.jsonl`、`predictions_b.jsonl`、`predictions_c.jsonl`：三项消融各自的完整合并预测。',
        '- `summary.json`：主口径、单分支、各种子/折、合并 OOF、对照差值和不确定性统计。',
        '- `verification.json`：核验结果；`reliability.png`、`uncertainty.png`：描述性诊断图。',
        f"- 旧产物保留检查：{verification.get('protected_files_unchanged', '详见 verification.json')} 个受保护文件与执行前一致。清单见上一级 `tc05_existing_artifacts_before.json` 和本目录 `existing_artifacts_after.json`。",
        '- 训练前检查快照不代表完成状态；以各折 `run.json` 和本次验收记录为准。', '',
        '## 解释边界', '',
        '本轮沿用已经查看过测试结果的固定划分，属于开发性比较；种子、折和消融数量不增加独立样本量。不宣称统计显著、独立人群确认或临床适用性。', '',
        '父实验的语音随机冻结编码器、单位置注意力，以及 EEG 尺度和通道对应限制继续保留。此处的消融结果不能消除这些限制，也不能把某一项消融的分数最高解释为适用于新数据的最佳配置。TC-06 尚未启动。', ''])
    path = OUTPUT / 'TC-05实验完成报告.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def generate(summary, verification):
    """Write TC-05 report and two figures; inputs must already be verified."""
    if not verification.get('passed'):
        raise ValueError('TC-05 report requires successful verification')
    if (verification.get('completed_training_jobs') != 45 or
            verification.get('prediction_records_reproduced_exactly') != 1665 or
            verification.get('total_optimizer_updates') != 10800):
        raise ValueError('TC-05 report requires all 45 fixed-budget jobs')
    if set(summary['variants']) != set(VARIANTS):
        raise ValueError('Expected all three predeclared ablations')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _figures(summary)
    return _report(summary, verification)


if __name__ == '__main__':
    summary = json.loads((OUTPUT / 'summary.json').read_text(encoding='utf-8'))
    verification = json.loads((OUTPUT / 'verification.json').read_text(encoding='utf-8'))
    print(generate(summary, verification))
