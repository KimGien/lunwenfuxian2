"""Verify TC-03 inference reuse and report every preset fusion comparison."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from .ccml import fuse_evidence
from .complete_tc01 import read_rows, write
from .metrics import binary_metrics, summarize
from .records import ROOT, MODALITIES, RecordContext, digest, read_json, require, validate_prediction_records
from .train import load_features, load_tc02, predict_records, run_directory, validate_config

VARIANTS = {'mean': 'mean_evidence', 'sum': 'sum_evidence', 'ccml': 'ccml'}
OUTPUT = ROOT / 'reports/tc03'


def experiment(name):
    return f'TC-03-{name}-37-fulltext'


def metric_crosscheck(rows):
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, log_loss, brier_score_loss
    for seed in (11, 22, 33):
        for fold in range(1, 6):
            part = [r for r in rows if r['seed'] == seed and r['outer_fold'] == fold and r['split'] == 'test']
            y, p = [r['label'] for r in part], [r['p_final'][1] for r in part]
            pred = [int(v > .5) for v in p]
            expected = {'macro_f1': f1_score(y, pred, average='macro'), 'mdd_recall': recall_score(y, pred),
                        'balanced_accuracy': balanced_accuracy_score(y, pred), 'nll': log_loss(y, p, labels=[0, 1]),
                        'brier': brier_score_loss(y, p)}
            actual = binary_metrics(y, p)
            require(all(abs(actual[k] - v) < 1e-12 for k, v in expected.items()), 'Independent metric check failed')


def verify(context):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    validate_config(context)
    guard = read_json(ROOT / 'reports/tc03_execution_guard.json')
    require(guard['passed'] and guard['optimizer_updates'] == 0 and not guard['training_performed'], 'No-training guard failed')
    require(guard['TC02_files_before'] == guard['TC02_files_after'], 'Inherited artifacts changed')
    for path, checksum in guard['TC02_files_after'].items():
        require(digest(ROOT / path) == checksum, 'TC02 artifact changed since inference')
    rows = {name: [] for name in VARIANTS}
    inherited_rows, audits = [], []
    for seed in context.config['seeds']:
        for fold in context.config['folds']:
            data = load_features(context, seed, fold)
            model, checkpoint, metadata = load_tc02(context, data, seed, fold, torch.device('cpu'))
            source_rows = read_rows(checkpoint.parent / 'predictions.jsonl')
            validate_prediction_records(source_rows, 'TC-02-37-fulltext', context=context, require_complete=False)
            parent_reproduced = predict_records(context, data, model, 'TC-02-37-fulltext', seed, fold, checkpoint, metadata, torch.device('cpu'))
            require(source_rows == parent_reproduced, 'TC02 source evidence no longer reproduces')
            inherited_rows.extend(source_rows)
            indexed = {r['subject_id']: r for r in source_rows}
            for name in VARIANTS:
                exp = experiment(name)
                directory = run_directory(ROOT, exp, seed, fold)
                run = read_json(directory / 'run.json')
                require(run['status'] == 'complete' and not run['training_performed'], 'Unexpected training or incomplete run')
                require((run['experiment'], run['seed'], run['fold']) == (exp, seed, fold), 'Run identity mismatch')
                require(Path(run['checkpoint_path']).resolve() == checkpoint.resolve(), 'Wrong source checkpoint path')
                require(run['checkpoint_sha256'] == digest(checkpoint) and run['source_metadata'] == metadata, 'Source provenance differs')
                require(not any((directory / n).exists() for n in ('checkpoint.pt', 'checkpoint.json', 'train_history.json', 'failure.json')), 'TC03 contains training or failure artifacts')
                saved = read_rows(directory / 'predictions.jsonl')
                require(len(saved) == 37 and {r['subject_id'] for r in saved} == set(context.ids), 'Fold coverage incomplete')
                reproduced = predict_records(context, data, model, exp, seed, fold, checkpoint, metadata, torch.device('cpu'))
                require(saved == reproduced, 'Recomputed inference differs from saved results')
                require(all(row['views'] == indexed[row['subject_id']]['views'] for row in saved), 'TC02 evidence changed')
                validate_prediction_records(saved, exp, context=context, require_complete=False)
                rows[name].extend(saved)
                audits.append({'experiment': exp, 'seed': seed, 'fold': fold,
                    'inherited_checkpoint_sha256': digest(checkpoint),
                    'predictions_sha256': digest(directory / 'predictions.jsonl'),
                    'training_performed': False, 'new_optimizer_updates': 0,
                    'all_37_records_reproduced_exactly': True, 'TC02_view_records_unchanged': True})
    validations = {}
    for name in VARIANTS:
        require(len(rows[name]) == 555, 'Incomplete three-seed inference')
        validations[name] = validate_prediction_records(rows[name], experiment(name), context=context)
        metric_crosscheck(rows[name])
    validate_prediction_records(inherited_rows, 'TC-02-37-fulltext', context=context)
    for mean, summed in zip(rows['mean'], rows['sum']):
        require((mean['seed'], mean['outer_fold'], mean['subject_id']) ==
                (summed['seed'], summed['outer_fold'], summed['subject_id']), 'Mean/sum alignment differs')
        require(mean['prediction'] == summed['prediction'], 'Mean/sum class predictions differ')
        require(summed['u_final'] < mean['u_final'], 'Positive evidence sum should have smaller u')
        require(np.allclose(np.array(mean['e_final']) * 3, summed['e_final'], rtol=2e-6, atol=2e-6), 'Mean/sum evidence relation differs')
        require(np.isclose(summed['u_final'], mean['u_final'] / (3-2*mean['u_final']), rtol=2e-6), 'Mean/sum uncertainty relation differs')
    verification = {'passed': True, 'completed_inference_jobs': 45, 'unique_inherited_checkpoints': 15,
        'new_training_jobs': 0, 'new_optimizer_updates': 0, 'new_checkpoints': 0,
        'prediction_records_reproduced_exactly': 1665, 'maximum_probability_error': 0.,
        'TC02_source_records_reproduced': 555, 'mean_sum_identical_decisions': 555,
        'mean_sum_uncertainty_relation_checked': 555, 'TC02_files_unchanged': len(guard['TC02_files_after']),
        'validation': validations, 'independent_metrics_verifier': 'sklearn',
        'config_sha256': context.config_sha256, 'completion_script_sha256': digest(__file__),
        'execution_guard_sha256': digest(ROOT / 'reports/tc03_execution_guard.json'), 'jobs': audits}
    return rows, inherited_rows, verification


def derived_evidence_rows(source, method, modalities, rho, eeg_factor=1.):
    derived = deepcopy(source)
    with torch.no_grad():
        for row in derived:
            evidence = torch.tensor([[row['views'][m]['e'] for m in modalities]], dtype=torch.float32)
            if 'eeg' in modalities:
                evidence[:, modalities.index('eeg')] *= eeg_factor
            fused = fuse_evidence(evidence, method=method, rho=rho)
            row['p_final'] = fused['p'][0].tolist()
            row['u_final'] = float(fused['u'][0])
            row['prediction'] = int(row['p_final'][1] > .5)
    return derived


def diagnostics(inherited, main_summaries, config):
    subsets = {}
    for method_name, method in VARIANTS.items():
        subsets[method_name] = {}
        for name, modalities in [('S+T', ('speech', 'text')), ('E+S', ('eeg', 'speech')),
                                 ('E+T', ('eeg', 'text')), ('E+S+T', MODALITIES)]:
            derived = derived_evidence_rows(inherited, method, modalities, config['rho'])
            result = summarize(derived)
            # Derived diagnostic summaries are not registered main-experiment rows.
            result['experiment'] = f'TC03-diagnostic-{method_name}-{name}'
            result.update(source_experiment='TC-02-37-fulltext', modalities=list(modalities), views_count=len(modalities),
                          fusion_method=method, rho=config['rho'] if method == 'ccml' else None,
                          training_performed=False, interpretation='fixed-output subset diagnostic, not a separately trained model')
            subsets[method_name][name] = result
            if len(modalities) == 3:
                require(result['primary_seed_mean_of_fold_means'] == main_summaries[method_name]['primary_seed_mean_of_fold_means'], 'Full subset differs from main experiment')
    perturbation = {}
    policy = config['synthetic_perturbation']
    require(policy['modality'] == 'eeg' and policy['operation'] == 'multiply_evidence', 'Unexpected perturbation policy')
    for factor in policy['factors']:
        result = summarize(derived_evidence_rows(inherited, 'ccml', MODALITIES, config['rho'], eeg_factor=factor))
        result['experiment'] = f'TC03-diagnostic-ccml-EEG-times-{factor}'
        result.update(source_experiment='TC-02-37-fulltext', evidence_multiplier=factor,
                      training_performed=False, clinical_or_ood_claim=False)
        perturbation[str(factor)] = result
    return subsets, perturbation


def figures(summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titles = {'mean': 'Mean evidence', 'sum': 'Sum evidence', 'ccml': 'CCML'}
    colors = {'mean': '#355d8c', 'sum': '#b86d32', 'ccml': '#277054'}
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for i, (name, report) in enumerate(summaries.items()):
        for j, (seed, item) in enumerate(report['seed_reports'].items()):
            pooled = item['pooled_oof']
            bins = [b for b in pooled['bins'] if b['n']]
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], '--', color='#8b969f', linewidth=1)
            ax.plot([b['mean_p_mdd'] for b in bins], [b['observed_mdd_rate'] for b in bins], 'o-', color=colors[name])
            for b in bins:
                ax.annotate(f"n={b['n']}", (b['mean_p_mdd'], b['observed_mdd_rate']), xytext=(3, 6), textcoords='offset points', fontsize=7)
            ax.set(xlim=(-.05, 1.05), ylim=(-.05, 1.1), xlabel='Mean P(MDD)', ylabel='Observed MDD rate')
            ax.set_title(f"{titles[name]} | seed {seed} | ECE5={pooled['ece5']:.4f}", fontsize=10)
            ax.grid(alpha=.18)
    fig.suptitle('TC-03 | same TC-02 weights | 37 pooled OOF subjects per seed | 5 bins')
    fig.savefig(OUTPUT / 'reliability.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, seed in enumerate(('11', '22', '33')):
        for i, (name, report) in enumerate(summaries.items()):
            uncertainty = report['seed_reports'][seed]['uncertainty']
            for category, shift, marker in [('correct_u', -.15, 'o'), ('incorrect_u', .15, 'x')]:
                values = uncertainty[category]
                offsets = np.linspace(-.05, .05, len(values)) if len(values) > 1 else np.zeros(len(values))
                axes[0, col].scatter(i+shift+offsets, values, s=22, marker=marker, color=colors[name], alpha=.7,
                    label=('Correct' if category == 'correct_u' else 'Incorrect') if i == 0 else None)
            curve = uncertainty['risk_coverage']
            axes[1, col].step([p['coverage'] for p in curve], [p['risk'] for p in curve], where='post',
                color=colors[name], linestyle='--' if name == 'sum' else '-', label=titles[name])
        axes[0, col].set(xticks=[0, 1, 2], xticklabels=['Mean', 'Sum', 'CCML'], ylim=(0, 1.02), ylabel='Fused u', title=f'Seed {seed} | uncertainty')
        axes[0, col].legend(fontsize=8)
        axes[1, col].set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel='Coverage', ylabel='Classification error rate', title='Retain low-u subjects; ties kept together')
        axes[1, col].legend(fontsize=8)
        for ax in axes[:, col]:
            ax.grid(alpha=.16)
    fig.suptitle('TC-03 uncertainty diagnostics | smaller u alone is not a success criterion')
    fig.savefig(OUTPUT / 'uncertainty.png', dpi=150)
    plt.close(fig)


def write_report(summaries, controls, subsets, perturbation):
    keys = ('macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier')
    titles = {'mean': 'TC-03 证据均值', 'sum': 'TC-03 证据求和', 'ccml': 'TC-03 CCML'}
    reference = controls['TC-02']['primary_seed_mean_of_fold_means']
    comparisons = {name: {key: result['primary_seed_mean_of_fold_means'][key] - reference[key] for key in keys}
                   for name, result in summaries.items()}
    lines = ['# TC-03 实验完成报告', '',
        'TC-03 的证据均值、证据求和和 CCML 三种推理规则均已完成。每种覆盖 3 个种子 × 5 折，共 45 组推理，全部复用 TC-02 的 15 个权重。新增训练和优化器更新均为零；没有当前阻塞。', '',
        '## 复用和数值验收', '',
        '- 运行时禁止调用训练函数、Adamax 构造器和反向传播；TC-02 的 75 个权重及配套文件在执行前后哈希不变。',
        '- 三种方法的每个分支 e/alpha/p/u 与 TC-02 完全一致，仅最终融合规则改变；45 组输出没有新权重或训练曲线。',
        '- 重载原权重后，三种方法共 1,665 行预测及证据记录完全复现，每种方法 555 行，其中测试 111 行，每个种子 37 人恰好测试一次。',
        '- 标签、分折、源特征、训练名单、代码和配置哈希均已核验；分类指标、NLL、Brier 逐折与 sklearn 复算一致。',
        '- 证据均值与求和的全部 555 个分类决策相同；证据和 u 的缩放关系检查通过。', '',
        '## 主结果', '',
        '以下采用每个种子的五折指标均值，再取三个种子均值。NLL、Brier 越低越好。所有预设规则均报告，没有挑选最优规则替换主结果。', '',
        '| 方法 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |', '|---|---:|---:|---:|---:|---:|']
    table = [(titles[name], value) for name, value in summaries.items()] + [('TC-02 概率等权平均', controls['TC-02']), ('TC-01 CE 概率等权平均', controls['TC-01'])]
    for title, value in table:
        metrics = value['primary_seed_mean_of_fold_means']
        lines.append('| ' + title + ' | ' + ' | '.join(f'{metrics[k]:.6f}' for k in keys) + ' |')
    lines.extend(['', '相对 TC-02，同一组证据改用三种规则后的变化：', '',
                  '| 规则 | Macro-F1 变化 | MDD recall 变化 | NLL 变化 | Brier 变化 |', '|---|---:|---:|---:|---:|'])
    for name, delta in comparisons.items():
        lines.append('| ' + titles[name] + ' | ' + ' | '.join(f'{delta[k]:+.6f}' for k in ('macro_f1', 'mdd_recall', 'nll', 'brier')) + ' |')
    lines.extend(['', '本轮三种推理规则的整体结果均低于 TC-02 的概率等权平均。CCML 的 Macro-F1 高于证据均值/求和，但其 NLL 和 Brier 略差于证据均值；三种规则均未带来相对 TC-02 的改善。这里没有训练一致/互补目标，不能据此认定完整 TC-04 联合目标已经验证成功或失败。', '',
        '证据均值与求和只差整体倍数，分类排序一致；其概率、总强度和 u 不同。证据求和使 u 更小，本轮 NLL/Brier 却更差，因此不能以较小 u 作为有效性的证据。证据均值也不同于先对每个分支归一化再平均概率，分支证据强度会影响结果。', '',
        '## 分种子结果和可靠性', '',
        '| 规则 | 种子 | 五折均值 F1 | 合并 37 人 F1 | 合并 NLL | 合并 Brier | ECE5 |', '|---|---|---:|---:|---:|---:|---:|'])
    for name, result in summaries.items():
        for seed, data in result['seed_reports'].items():
            p = data['pooled_oof']
            lines.append(f"| {titles[name]} | {seed} | {data['unweighted_fold_mean']['macro_f1']:.6f} | {p['macro_f1']:.6f} | {p['nll']:.6f} | {p['brier']:.6f} | {p['ece5']:.6f} |")
    lines.extend(['', '![TC-03 可靠性图](reliability.png)', '',
        'ECE 使用 5 个等宽箱比较预测 MDD 概率与实际 MDD 比例，箱内人数显示于图中；合并 37 人的指标与五折均值分开列示。', '',
        '## 不确定性诊断', '', '![TC-03 不确定性与风险覆盖率](uncertainty.png)', '',
        '上图按方法、种子展示正确/错误预测的 u；下图按低 u 优先保留受试者，并同时保留相同 u 的并列记录。证据均值和求和的风险曲线可能重合，因为其 u 的单调变换不改变保留顺序。每个种子仍只有 37 人，u 不是出错概率或临床风险。', '',
        '## 固定输出的模态子集诊断', '',
        '使用相同 TC-02 输出进行 S+T、E+S、E+T、E+S+T 比较，解耦器使用实际 V=2 或 3；没有为子集重训，也不将诊断结果替换三模态主结果。下表列 Macro-F1，完整指标及不确定性数据见 `subset_diagnostics.json`。', '',
        '| 子集 | 证据均值 F1 | 证据求和 F1 | CCML F1 | CCML MDD recall |', '|---|---:|---:|---:|---:|'])
    for subset in ('S+T', 'E+S', 'E+T', 'E+S+T'):
        values = [subsets[name][subset]['primary_seed_mean_of_fold_means']['macro_f1'] for name in VARIANTS]
        recall = subsets['ccml'][subset]['primary_seed_mean_of_fold_means']['mdd_recall']
        lines.append('| ' + subset + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + f' | {recall:.6f} |')
    lines.extend(['', '## 预设扰动检查', '',
        '按固定准备配置将 EEG 输出证据分别乘以 0、0.5、1、2，再进行三模态 CCML 推理。其余分支保持原值。这是对模型输出的人工敏感性检查，不是临床数据增强、真实分布外验证或新的调参依据。', '',
        '| EEG 证据倍数 | Macro-F1 | MDD recall | NLL | Brier |', '|---|---:|---:|---:|---:|'])
    for factor, result in perturbation.items():
        metrics = result['primary_seed_mean_of_fold_means']
        lines.append('| ' + factor + ' | ' + ' | '.join(f'{metrics[k]:.6f}' for k in ('macro_f1', 'mdd_recall', 'nll', 'brier')) + ' |')
    lines.extend(['', '## 产物和解释边界', '',
        '- `../../runs/TC-03-{mean,sum,ccml}-37-fulltext/seed_<seed>/fold_<fold>/`：各折预测与复用记录。',
        '- `predictions_mean.jsonl`、`predictions_sum.jsonl`、`predictions_ccml.jsonl`：三种完整预测；`summary.json`：汇总及对照。',
        '- `verification.json`、上一级 `tc03_execution_guard.json`：重载核验、覆盖及无训练证据。',
        '- `subset_diagnostics.json`、`perturbation_diagnostics.json`：固定输出的诊断。',
        '- 已查看过测试结果的固定划分用于开发性比较。新增融合规则、种子和折均不增加独立受试者数量，原有语音与 EEG 特征限制继续披露。',
        '- TC-04 尚未执行；它需从共同初始化独立训练完整 CCML 目标，不能把 TC-03 的推理结果当作 TC-04 结果。', ''])
    (OUTPUT / 'TC-03实验完成报告.md').write_text('\n'.join(lines), encoding='utf-8')
    return comparisons


def main():
    context = RecordContext()
    rows, inherited, verification = verify(context)
    summaries = {name: summarize(value) for name, value in rows.items()}
    control_rows = []
    for seed in context.config['seeds']:
        for fold in context.config['folds']:
            control_rows.extend(read_rows(run_directory(ROOT, 'TC-01-37-fulltext', seed, fold) / 'predictions.jsonl'))
    validate_prediction_records(control_rows, 'TC-01-37-fulltext', context=context)
    controls = {'TC-01': summarize(control_rows), 'TC-02': summarize(inherited)}
    subsets, perturbation = diagnostics(inherited, summaries, context.config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures(summaries)
    comparison = write_report(summaries, controls, subsets, perturbation)
    write(OUTPUT / 'verification.json', verification)
    write(OUTPUT / 'summary.json', {'variants': summaries, 'controls': controls, 'differences_from_TC02': comparison})
    write(OUTPUT / 'subset_diagnostics.json', subsets)
    write(OUTPUT / 'perturbation_diagnostics.json', perturbation)
    for name, values in rows.items():
        path = OUTPUT / f'predictions_{name}.jsonl'
        text = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in values)
        if path.exists():
            require(path.read_text(encoding='utf-8') == text, 'Existing combined records differ')
        else:
            path.write_text(text, encoding='utf-8')
    status = read_json(ROOT / 'reports/experiment_status.json')
    status['utc'] = datetime.now(timezone.utc).isoformat()
    status['TC-03'] = {'status': 'complete', 'completed_inference_jobs': 45, 'new_training_jobs': 0,
        'new_optimizer_updates': 0, 'verified': True, 'TC02_artifacts_unchanged': True,
        'report': 'reports/tc03/TC-03实验完成报告.md',
        'variants': {name: {'experiment': experiment(name), 'jobs': 15,
                           'primary_metrics': value['primary_seed_mean_of_fold_means']} for name, value in summaries.items()}}
    (ROOT / 'reports/experiment_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'inference_jobs': 45, 'new_training_jobs': 0,
                      'results': status['TC-03']['variants'], 'report': str(OUTPUT / 'TC-03实验完成报告.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
