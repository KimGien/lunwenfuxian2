"""Audit original TRI-DEP with corrected EEG and report paired macro-F1."""
from __future__ import annotations

from collections import Counter
import math

from .tc00_eeg import (OUT, CONFIG, PLAN, EXPERIMENT, S00, configuration, setup, load_inputs,
                      parent_bundle, runtime, models, probabilities, fuse, protect_history)
from .v2.common import (RecordContext, read_json, write_new, descriptor, verify_descriptor, require)
from .metrics import summarize, binary_metrics
from .train import state_digest
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, log_loss, brier_score_loss

METRICS = ['macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier']
MODES = ['weighted', 'mean', 'eeg', 'speech', 'text']


def normalized(rows, mode, experiment):
    result = []
    for r in rows:
        p = (r['p_final'] if mode == 'weighted' else
             sum((1/3)*r['p_'+m] for m in ['eeg', 'speech', 'text']) if mode == 'mean' else r['p_'+mode])
        require(math.isfinite(p) and 0 <= p <= 1, 'Invalid probability')
        result.append({**r, 'experiment': experiment+'-'+mode, 'p_final': [1-p, p],
                       'prediction': int(p > .5), 'u_final': None})
    return result


def audit():
    context, cfg = setup()
    require(read_json(CONFIG) == configuration(context), 'Configuration changed')
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG) and len(pre['checks']) == 15 and
            pre['parent_checkpoints_verified'] == 45 and pre['branch_probabilities_verified'] == 1665,
            'Preflight incomplete')
    data = load_inputs(context)
    all_rows, old_rows, checks = [], [], []
    for value in cfg['seeds']:
        for fold in cfg['folds']:
            runtime(value)
            folder = OUT/f'runs/seed_{value}/fold_{fold}'
            for d in read_json(folder/'complete.json').values():
                verify_descriptor(d)
            rows = read_json(folder/'predictions.json')
            parent, checkpoints, _ = parent_bundle(context, value, fold)
            cp = folder/'checkpoint.pt'
            saved = torch.load(cp, map_location='cpu', weights_only=True)
            meta = saved['metadata']
            require(meta == read_json(folder/'checkpoint.json')['metadata'], 'Checkpoint sidecar mismatch')
            require(meta['experiment'] == EXPERIMENT and meta['configuration'] == descriptor(CONFIG) and
                    meta['seed'] == value and meta['fold'] == fold and
                    meta['train_ids'] == context.folds[f'fold_{fold}']['train'], 'New run identity/train mismatch')
            require(meta['cohort_sha256'] == context.cohort_sha256 and meta['folds_sha256'] == context.folds_sha256,
                    'Cohort changed')
            require(meta['branch_configuration'] == cfg['eeg_branch'] and
                    meta['objective'] == 'original_TRI_DEP_cross_entropy' and meta['updates'] == 120 and
                    meta['epochs'] == 60 and meta['trainable_parameters'] == 498434, 'Original protocol changed')
            require(meta['corrected_features'] == data['corrected_source'] and
                    meta['original_parent_checkpoints'] == checkpoints, 'Input/checkpoint provenance changed')
            verify_descriptor(meta['original_probability_file'])
            old = next(c for c in pre['checks'] if c['seed'] == value and c['fold'] == fold)
            for k in ['initial_state_sha256', 'batch_order_sha256', 'final_cpu_rng_sha256', 'final_cuda_rng_sha256',
                      'train_ids', 'updates', 'epochs', 'trainable_parameters']:
                require(meta[k] == old[k], 'Initial state/batches/RNG/budget pairing mismatch')
            model = models.EEGConvPoolClassifier(**cfg['eeg_branch']['model_kwargs']).to('cuda')
            require(state_digest(model.state_dict()) == meta['initial_state_sha256'], 'Initial state cannot be reconstructed')
            model.load_state_dict(saved['state_dict'], strict=True)
            p = probabilities(model, data['corrected'], context.ids)
            reproduced, selection = fuse(parent, p)
            require(selection == read_json(folder/'fusion_selection.json'), 'Val-only selection mismatch')
            require(len(rows) == 37 and [r['subject_id'] for r in rows] == context.ids, 'New row coverage mismatch')
            for r, reproduced_row, parent_row in zip(rows, reproduced, parent):
                require(all(r[k] == v for k, v in reproduced_row.items()), 'New probabilities do not exactly reload')
                require(r['configuration'] == descriptor(CONFIG) and r['eeg_checkpoint'] == descriptor(cp) and
                        r['corrected_eeg_feature'] == data['corrected_source'][r['subject_id']] and
                        r['unchanged_branch_checkpoints'] == {m: checkpoints[m] for m in ['speech', 'text']}, 'Row source changed')
                require(all(r['p_'+m] == parent_row['p_'+m] for m in ['speech', 'text']), 'Unchanged branch probability changed')
            history = read_json(folder/'train_history.json')
            require([r['epoch'] for r in history] == list(range(1, 61)) and
                    all(math.isfinite(r['train_cross_entropy']) for r in history), 'Missing/nonfinite training history')
            require(state_digest(saved['state_dict']) != meta['initial_state_sha256'], 'EEG model did not learn')
            all_rows += rows
            old_rows += parent
            checks.append({'seed': value, 'fold': fold, 'checkpoint': descriptor(cp), 'exact_reproduced_eeg_probabilities': 37,
                           'speech_text_probabilities_unchanged': True, 'initial_batch_rng_pairing': True,
                           'fusion_selection_only_val': True, 'updates': 120})
            del model, saved
    for rows in [all_rows, old_rows]:
        require(len(rows) == 555 and Counter((r['seed'], r['subject_id']) for r in rows if r['split'] == 'test') ==
                Counter((v, s) for v in cfg['seeds'] for s in context.ids), 'Incomplete OOF coverage')
    sets = {name: {mode: normalized(rows, mode, experiment) for mode in MODES}
            for name, rows, experiment in [('original', old_rows, 'TC-00'), ('corrected', all_rows, EXPERIMENT)]}
    summaries = {name: {mode: summarize(rows) for mode, rows in modes.items()} for name, modes in sets.items()}
    historical = read_json(S00.parent/'tridep_ccml/reports/tc00_summary.json')['mean_over_seed_means_of_five_folds']
    for mode in ['weighted', 'mean']:
        for k, expected in historical[mode].items():
            require(math.isclose(summaries['original'][mode]['primary_seed_mean_of_fold_means'][k], expected,
                                 rel_tol=0., abs_tol=1e-14), 'Historical TC-00 summary mismatch')
    crosschecks = []
    for name, modes in sets.items():
        for mode, rows in modes.items():
            for value in cfg['seeds']:
                for fold in cfg['folds']:
                    test = [r for r in rows if r['seed'] == value and r['outer_fold'] == fold and r['split'] == 'test']
                    labels, p = [r['label'] for r in test], [r['p_final'][1] for r in test]
                    predictions = [int(x > .5) for x in p]
                    score = binary_metrics(labels, p)
                    expected = dict(macro_f1=f1_score(labels, predictions, average='macro', labels=[0, 1], zero_division=0),
                                    mdd_recall=recall_score(labels, predictions, pos_label=1, zero_division=0),
                                    balanced_accuracy=balanced_accuracy_score(labels, predictions),
                                    nll=log_loss(labels, np.clip(p, 1e-15, 1-1e-15), labels=[0, 1]),
                                    brier=brier_score_loss(labels, p))
                    require(all(math.isclose(score[k], expected[k], rel_tol=1e-10, abs_tol=1e-10) for k in METRICS), 'Metric crosscheck failed')
                    crosschecks.append({'family': name, 'mode': mode, 'seed': value, 'fold': fold, 'passed': True})
    changes = {}
    for mode in MODES:
        a = {(r['seed'], r['subject_id']): r for r in sets['original'][mode] if r['split'] == 'test'}
        changed = []
        for r in sets['corrected'][mode]:
            if r['split'] != 'test':
                continue
            old = a[r['seed'], r['subject_id']]
            if old['prediction'] != r['prediction']:
                changed.append({'seed': r['seed'], 'subject_id': r['subject_id'], 'label': r['label'],
                                'before': old['prediction'], 'after': r['prediction'],
                                'direction': 'wrong_to_correct' if r['prediction'] == r['label'] else 'correct_to_wrong'})
        changes[mode] = {'changed_count': len(changed), 'directions': dict(Counter(r['direction'] for r in changed)), 'records': changed}
    result = {'passed': True, 'date': '2026-09-23', 'new_eeg_runs': 15, 'new_updates': 1800,
              'old_training_replay_runs': 15, 'replay_updates': 1800, 'independent_subjects': 37,
              'summaries': summaries, 'changes': changes, 'checks': checks, 'metric_crosschecks': crosschecks,
              'configuration': descriptor(CONFIG), 'tests': descriptor(OUT/'test_results.json'),
              'preflight': descriptor(OUT/'preflight.json'), 'reporter': descriptor(__file__),
              'delta': {mode: {k: summaries['corrected'][mode]['primary_seed_mean_of_fold_means'][k]-
                              summaries['original'][mode]['primary_seed_mean_of_fold_means'][k] for k in METRICS} for mode in MODES},
              'split_scores': {name: {mode: {split: {k: sum(binary_metrics(
                  [r['label'] for r in rows if r['seed'] == v and r['outer_fold'] == f and r['split'] == split],
                  [r['p_final'][1] for r in rows if r['seed'] == v and r['outer_fold'] == f and r['split'] == split])[k]
                  for v in cfg['seeds'] for f in cfg['folds'])/15 for k in METRICS} for split in ['train', 'val', 'test']}
                  for mode, rows in modes.items()} for name, modes in sets.items()},
              'protected_files_unchanged': protect_history()}
    write_new(OUT/'predictions_all.json', sets['corrected'])
    write_new(OUT/'completion_audit.json', result)
    return result


def report(result):
    names = {'weighted': '原验证集选权融合', 'mean': '等权概率融合', 'eeg': 'EEG 单分支', 'speech': '语音单分支', 'text': '文本单分支'}
    lines = ['# TC-00-E：修订 EEG 输入的原始 TRI-DEP 结果', '',
             '日期：2026-09-23。**15 组新 EEG 训练已完成，共 1,800 次更新；另完成 15 组旧 EEG 训练重放核验。**', '',
             '## F1 结果', '',
             '用户所称 T-00 对应本项目 TC-00 / B0-37-fulltext。只采用 CCML-v2-E 中按正确电极标签导出的 60×200 EEG 骨干特征，使用原 TRI-DEP 分类器、CE 训练和概率融合。没有采用 E 阶段经 CCML 适配后的 CNN/投影或证据头。', '',
             '主口径：每种子五折 Macro-F1 均值，再平均种子 11/22/33。', '',
             '| 路径 | 原 TC-00 F1 | 修订 EEG 后 F1 | 变化（百分点） |', '|---|---:|---:|---:|']
    for mode in MODES:
        a, b = [result['summaries'][name][mode]['primary_seed_mean_of_fold_means']['macro_f1'] for name in ['original', 'corrected']]
        lines.append(f'| {names[mode]} | {a:.6f} | {b:.6f} | {(b-a)*100:+.2f} |')
    lines += ['', '| 种子 | 原选权融合 F1 | 新选权融合 F1 | 原等权融合 F1 | 新等权融合 F1 |', '|---|---:|---:|---:|---:|']
    for seed in ['11', '22', '33']:
        values = [result['summaries'][name][mode]['seed_reports'][seed]['unweighted_fold_mean']['macro_f1']
                  for mode in ['weighted', 'mean'] for name in ['original', 'corrected']]
        lines.append('| '+seed+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    lines += ['', '| 修订后路径 | MDD recall | BA | NLL | Brier |', '|---|---:|---:|---:|---:|']
    for mode in ['weighted', 'mean', 'eeg']:
        score = result['summaries']['corrected'][mode]['primary_seed_mean_of_fold_means']
        lines.append('| '+names[mode]+' | '+' | '.join(f'{score[k]:.6f}' for k in METRICS[1:])+' |')
    lines += ['', '选权融合沿用原策略：每折仅使用 3 位验证受试者，步长 0.05，按原字典序处理平局；修改 EEG 后按相同规则重新选择权重。因此选权结果包含融合权重对新分支的响应。等权结果无需选权，可另看固定融合下的变化。没有用测试标签选择权重。', '',
              '## 配对训练与验收', '',
              '- 原 EEGConvPoolClassifier，498,434 参数；60 轮，batch 16，Adamax lr=0.005、weight_decay=0；原交叉熵、无梯度裁剪，训练 Dropout=0.4 开启，推理 eval。随机初始化，最后一轮，阈值 0.5，平局为 HC。',
              '- 复用 E 阶段 37 份标签修订后的冻结 CBraMod 特征；native 尺度、滤波、截取和参考保持原设置。没有新增规范化、CCML 或其他模型模块。',
              '- 语音和全文文本直接复用 TC-00 的同种子同折父权重与概率，避免重复拟合未变化的独立分支；新旧逐条概率相同。',
              '- CUDA 环境、原确定性种子、matmul TF32=False、cuDNN TF32=True。15 组旧 EEG 重放的最终参数与每轮 loss 均精确复现；新旧初始状态、批序、训练结束随机状态一致。',
              '- 45 个旧父检查点及 1,665 条分支概率复核通过；15 个新 EEG 检查点和 555 条 EEG 概率精确重载；两种融合合计 1,110 条分折记录，外层测试每人每种子恰好一次。',
              f'- {read_json(OUT/"test_results.json")["tests_run"]} 项相关测试通过；新旧两版三分支和两种融合共 {len(result["metric_crosschecks"])} 项折级指标交叉核验通过；{result["protected_files_unchanged"]} 项历史文件哈希不变。', '',
              '## 分类变化与解释', '']
    for mode in ['weighted', 'mean', 'eeg']:
        c = result['changes'][mode]
        lines.append(f'- {names[mode]}：111 条重复种子测试记录中 {c["changed_count"]} 条分类改变，错变对 {c["directions"].get("wrong_to_correct",0)}，对变错 {c["directions"].get("correct_to_wrong",0)}。')
    weighted = result['summaries']['corrected']['weighted']['primary_seed_mean_of_fold_means']['macro_f1']
    mean = result['summaries']['corrected']['mean']['primary_seed_mean_of_fold_means']['macro_f1']
    lines += ['', f'原 TRI-DEP 选权融合在本轮 EEG 修订后的主口径 F1 为 **{weighted:.6f}**；等权融合 F1 为 **{mean:.6f}**。本实验提供通道修订在原 TRI-DEP 协议中的配对结果；不把此前 E 的 CCML 适配结果混入本轮。',
              '这里的原始模型指本地已登记的 B0-37-fulltext：37 人、全文文本及前 300 秒 EEG 等本地协议保持不变，不能称为精确复现论文 38 人结果。旧语音随机冻结编码器、短音频零向量、EEG 检查点尺度和原始坐标限制仍在。',
              '这 37 人的测试结果此前已被查看，仍属于开发比较；三个种子不增加独立样本数。完整逐折指标、合并 OOF、各折融合权重及 train/val/test 分数在 completion_audit.json 和 runs 下保存。', '']
    path = OUT/'TC-00-E原始TRIDEP实验完成报告.md'
    content = '\n'.join(lines)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content, 'Existing report differs')
    else:
        path.write_text(content, encoding='utf-8')
    write_new(OUT/'status.json', {'status': 'complete', 'date': '2026-09-23', 'new_eeg_runs': 15,
              'new_updates': 1800, 'old_replay_runs': 15, 'replay_updates': 1800,
              'configuration': descriptor(CONFIG), 'plan': descriptor(PLAN), 'report': descriptor(path),
              'audit': descriptor(OUT/'completion_audit.json'), 'execution_blockers': [],
              'weighted_macro_f1': weighted, 'mean_macro_f1': mean})
    print({'weighted_macro_f1': weighted, 'mean_macro_f1': mean, 'delta': result['delta']})


if __name__ == '__main__':
    report(audit())
