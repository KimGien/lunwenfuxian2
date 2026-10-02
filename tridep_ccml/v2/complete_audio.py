"""Completion audit for the fixed speech-source revision and all matched controls."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

from .audio import (OUT, CONFIG, PLAN, EEG, OLD_CONTROL, NR, VARIANTS, NP_OUT, STAGES,
                    configuration, protect_history, pool_question, pool_subject, load_data,
                    make_model, forward_all, predictions)
from .common import (WORK, RecordContext, read_json, descriptor, verify_descriptor,
                     require, write_new, set_seed)
from .complete_nr import diagnostics
from .complete_p import gate
from .complete_eeg import crosscheck_metrics, METRICS
from .rebuild_eeg import validate_row
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.metrics import summarize
import numpy as np
import torch


def audit_pool(context):
    manifest = read_json(OUT/'speech_pool.json')
    require(manifest['configuration'] == descriptor(CONFIG) and manifest['labels_used'] is False and
            manifest['fit_subject_ids'] == [], 'Pool fitted or configuration changed')
    with np.load(verify_descriptor(manifest['cache']), allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    require(arrays['subject_ids'].tolist() == context.ids, 'Pooled subject order mismatch')
    expected = {(s, q) for s in context.ids for q in range(1, 30)}
    require(len(manifest['sources']) == 1073 and
            {(r['subject_id'], r['question']) for r in manifest['sources']} == expected, 'Question coverage mismatch')
    lookup = {s: i for i, s in enumerate(context.ids)}
    for row in manifest['sources']:
        cache = verify_descriptor(row['cache'])
        metadata = read_json(verify_descriptor(row['metadata']))
        verify_descriptor(row['raw_audio'])
        vectors = np.load(cache, allow_pickle=False)
        i, q = lookup[row['subject_id']], row['question']-1
        require(np.array_equal(pool_question(vectors, metadata['windows_samples']), arrays['question_means'][i, q]), 'Question mean mismatch')
        require(arrays['segment_counts'][i, q] == row['segment_count'] == len(vectors), 'Segment count mismatch')
        require(row['empty'] == (len(vectors) == 0), 'Empty flag mismatch')
    for i in range(37):
        require(np.array_equal(pool_subject(arrays['question_means'][i]), arrays['pooled'][i]), 'Pooled mean mismatch')
    require(int((arrays['segment_counts'] == 0).sum()) == manifest['empty_questions'] == 96, 'Missing short questions')
    require(int((arrays['segment_counts'] == 0).any(axis=1).sum()) == manifest['affected_subjects'] == 29, 'Affected subject count changed')
    return {'questions_reconstructed': 1073, 'pooled_subjects_reconstructed': 37,
            'empty_questions_retained': 96, 'affected_subjects': 29, 'labels_used': False}


def audit():
    context = RecordContext()
    cfg = read_json(CONFIG)
    require(cfg == configuration(context), 'Registered audio configuration changed')
    protected = protect_history()
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG) and
            pre['exact_comparator_records'] == 1110, 'Preflight incomplete')
    pool_check = audit_pool(context)
    tests = read_json(OUT/'test_results.json')
    require(tests['passed'] and tests['failures'] == 0 and tests['errors'] == 0, 'Unit tests incomplete')
    for item in tests['sources'].values():
        verify_descriptor(item)
    all_rows = {s: [] for s in ['adapt', 'control', 'N', 'old_C', 'old_N']}
    checks, changes = [], []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            old_metadata = {}
            for name, folder in [('old_C', EEG/'runs'/OLD_CONTROL/f'seed_{seed}/fold_{fold}'),
                                 ('old_N', NR/'runs'/VARIANTS['N']['experiment']/f'seed_{seed}/fold_{fold}')]:
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                old_metadata[name] = read_json(folder/'checkpoint.json')['metadata']
                all_rows[name].extend(read_json(folder/'predictions.json'))
            initial_head = state_digest(make_initial_state(seed))
            require(old_metadata['old_C']['head_initialization_sha256'] == initial_head ==
                    old_metadata['old_N']['initialization_sha256'], 'Old head pairing mismatch')
            gen, batch = torch.Generator().manual_seed(seed), hashlib.sha256()
            for _ in range(60):
                batch.update(torch.randperm(len(context.folds[f'fold_{fold}']['train']), generator=gen).numpy().astype('<i8').tobytes())
            require(all(m['batch_order_sha256'] == batch.hexdigest() for m in old_metadata.values()), 'Old batch mismatch')
            cache_sources = []
            for stage in STAGES:
                set_seed(seed)
                data = load_data(context, seed, fold, stage)
                model = make_model(seed, stage)
                initial_sha = state_digest(model.state_dict())
                initial_proj = {k: v.clone() for k, v in model.speech_projection.state_dict().items()} if stage == 'adapt' else None
                folder = OUT/'runs'/STAGES[stage]/f'seed_{seed}/fold_{fold}'
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                cp = folder/'checkpoint.pt'
                saved = torch.load(cp, map_location='cpu', weights_only=True)
                meta = saved['metadata']
                require(meta == read_json(cp.with_suffix('.json'))['metadata'], 'Metadata mismatch')
                require(meta['experiment'] == STAGES[stage] and meta['stage'] == stage and
                        meta['seed'] == seed and meta['fold'] == fold, 'Run identity mismatch')
                require(meta['configuration'] == descriptor(CONFIG) and meta['source'] == data['source'], 'Source binding mismatch')
                require(meta['cohort_sha256'] == context.cohort_sha256 and meta['folds_sha256'] == context.folds_sha256 and
                        meta['train_ids'] == context.folds[f'fold_{fold}']['train'], 'Cohort/train provenance mismatch')
                require(meta['initial_state_sha256'] == initial_sha and meta['head_initialization_sha256'] == initial_head and
                        meta['batch_order_sha256'] == batch.hexdigest(), 'Initialization/batch pairing mismatch')
                require(meta['objective'] == 'full_CCML_only' and meta['epochs'] == 60 and meta['updates'] == 240 and
                        meta['selection'] == 'last_epoch', 'Objective/budget/selection mismatch')
                require(meta['head_lr'] == 1e-3 and meta['weight_decay'] == 0. and
                        meta['projection_lr'] == (1e-4 if stage == 'adapt' else None), 'Learning-rate/regularization mismatch')
                count = 99206 if stage == 'adapt' else 774
                require(meta['trainable_parameters'] == count == sum(p.numel() for p in model.parameters()) and
                        meta['normalization'] == cfg['normalization'][stage], 'Trainable scope or normalization mismatch')
                history = read_json(folder/'train_history.json')
                require([r['epoch'] for r in history] == list(range(1, 61)), 'Missing epoch')
                for row in history:
                    require(math.isfinite(row['train_loss']) and math.isclose(row['train_loss'], sum(row['loss_terms'].values()),
                            rel_tol=1e-5, abs_tol=1e-5), 'Loss decomposition mismatch')
                    require(all(math.isfinite(v) and v >= 0 for v in row['head_grad_norm_mean'].values()), 'Invalid head gradients')
                model.load_state_dict(saved['state_dict'], strict=True)
                if stage == 'adapt':
                    require(meta['initial_projection_sha256'] == state_digest(initial_proj), 'Projection initialization mismatch')
                    require(any(r['speech_projection_grad_norm_mean'] > 0 for r in history) and
                            all(math.isfinite(r['speech_projection_grad_norm_mean']) for r in history), 'Missing speech gradient')
                    final_proj = model.speech_projection.state_dict()
                    before = torch.cat([v.flatten() for v in initial_proj.values()])
                    after = torch.cat([v.flatten() for v in final_proj.values()])
                    require(not torch.equal(before, after), 'No speech projection update')
                    changes.append({'seed': seed, 'fold': fold, 'l2_change': float(torch.linalg.vector_norm(after-before)),
                                    'relative_l2_change': float(torch.linalg.vector_norm(after-before)/torch.linalg.vector_norm(before))})
                    exported = load_data(context, seed, fold, 'control')
                    rebuilt = forward_all(model, data, stage, representations=True).numpy()
                    require(np.array_equal(rebuilt, exported['features']), 'New parent representations do not reproduce')
                    require(np.array_equal(rebuilt[:, [0, 2]], data['features'][:, [0, 2]]), 'EEG/text changed')
                    require(exported['source']['parent_checkpoint'] == descriptor(cp), 'Wrong adaptation parent')
                else:
                    require(meta['initial_projection_sha256'] is None, 'Unexpected trainable projection')
                    require(all(r['speech_projection_grad_norm_mean'] is None for r in history), 'Projection gradients in frozen control')
                    cache_sources.append(data['source'])
                rows = read_json(folder/'predictions.json')
                require(rows == predictions(model, data, context, seed, fold, stage, cp), 'New predictions do not reproduce exactly')
                require(len(rows) == 37 and [r['subject_id'] for r in rows] == context.ids, 'Prediction coverage mismatch')
                for row in rows:
                    validate_row(row)
                all_rows[stage].extend(rows)
                checks.append({'stage': stage, 'seed': seed, 'fold': fold, 'checkpoint': descriptor(cp),
                               'exact_reproduced_rows': 37, 'paired_initialization_batches': True, 'updates': 240})
            require(cache_sources[0] == cache_sources[1], 'A-control/A-N use different parents')
    for name, rows in all_rows.items():
        require(len(rows) == 555 and Counter((r['seed'], r['subject_id']) for r in rows if r['split'] == 'test') ==
                Counter((s, i) for s in cfg['seeds'] for i in context.ids), f'Incomplete OOF: {name}')
    summaries = {name: summarize(rows) for name, rows in all_rows.items()}
    old_nr = read_json(NR/'completion_audit.json')
    require(summaries['old_C'] == old_nr['baseline'] and summaries['old_N'] == old_nr['variants']['N']['summary'], 'Historical summary drift')
    comparisons = {'A_N_vs_old_N': gate(summaries['N'], summaries['old_N']),
                   'A_N_vs_A_control': gate(summaries['N'], summaries['control']),
                   'A_control_vs_E_control': gate(summaries['control'], summaries['old_C'])}
    result = {'date': '2026-09-22', 'passed': True, 'formal_runs': 45, 'updates': 10800,
              'subject_fold_rows': 1665, 'independent_subjects': 37, 'summaries': summaries,
              'NP_reference': read_json(NP_OUT/'completion_audit.json')['summaries']['NP'],
              'comparisons': comparisons, 'primary_comparison': 'A_N_vs_old_N',
              'diagnostics': {stage: diagnostics(all_rows[stage], all_rows['old_N' if stage == 'N' else 'old_C']) for stage in STAGES},
              'same_version_normalization_diagnostics': diagnostics(all_rows['N'], all_rows['control']),
              'projection_changes': changes, 'run_checks': checks, 'pool_audit': pool_check,
              'sklearn_fold_crosschecks': {s: crosscheck_metrics(all_rows[s]) for s in STAGES},
              'configuration': descriptor(CONFIG), 'preflight': descriptor(OUT/'preflight.json'),
              'tests': descriptor(OUT/'test_results.json'), 'reporter': descriptor(__file__),
              'protected_files_unchanged': protect_history(),
              'scope': 'speech pooling/projection source revision; previously viewed 37 subjects; all fixed stages reported'}
    require(result['protected_files_unchanged'] == protected, 'History changed during audit')
    write_new(OUT/'completion_audit.json', result)
    return result


def report(result):
    summaries = result['summaries']
    primary = result['comparisons']['A_N_vs_old_N']
    passed = primary['gate']['development_criteria_passed']
    worse = primary['gate']['nll_worsened'] or primary['gate']['brier_worsened']
    lines = ['# CCML-v2-A 语音汇聚修订完成报告', '',
             '日期：2026-09-22。**语音适配、同版本控制、规范化候选各 15 组全部完成，共 45 组、10,800 次更新，验收通过。**', '',
             '## 固定阶段结果', '',
             '主口径为每种子五折均值，再平均三个种子。A-adapt 用于产生新父表征，A-control 与 A-N 在该表征上重新训练独立新头。旧 N 是主要路径比较对象；NP 仅作不同训练范围的历史开发参照。', '',
             '| 实验 | Macro-F1 | MDD recall | BA | NLL | Brier |', '|---|---:|---:|---:|---:|---:|']
    labels = {'old_C': '旧 E-control', 'old_N': '旧 N', 'adapt': 'A-adapt：语音投影适配',
              'control': 'A-control：新父版本控制', 'N': 'A-N：新父版本规范化'}
    for name in ['old_C', 'old_N', 'adapt', 'control', 'N']:
        m = summaries[name]['primary_seed_mean_of_fold_means']
        lines.append('| '+labels[name]+' | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    m = result['NP_reference']['primary_seed_mean_of_fold_means']
    lines.append('| NP：历史开发参照 | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    lines += ['', '| 固定比较 | F1 变化（百分点） | 召回变化（百分点） | F1 改善种子数 | 工程门槛 | NLL 变化 | Brier 变化 |',
              '|---|---:|---:|---:|---|---:|---:|']
    names = {'A_N_vs_old_N': 'A-N−旧 N（主比较）', 'A_N_vs_A_control': 'A-N−A-control',
             'A_control_vs_E_control': 'A-control−E-control'}
    for name, comp in result['comparisons'].items():
        d, g = comp['delta'], comp['gate']
        positive = sum(v > 0 for v in comp['seed_f1_deltas'].values())
        decision = '通过' if g['development_criteria_passed'] else '未通过'
        lines.append(f'| {names[name]} | {100*d["macro_f1"]:+.2f} | {100*d["mdd_recall"]:+.2f} | {positive}/3 | {decision} | {d["nll"]:+.6f} | {d["brier"]:+.6f} |')
    lines += ['', '固定工程参考：平均 F1 至少 +0.01，至少 2/3 种子严格提高，召回下降不超过 0.05；NLL/Brier 越低越好，概率质量权衡另列。该参考不是统计显著性或独立确认。', '',
              '| 种子 | 旧 N F1 | A-adapt F1 | A-control F1 | A-N F1 | A-N−旧 N |',
              '|---|---:|---:|---:|---:|---:|']
    for s in ['11', '22', '33']:
        values = [summaries[n]['seed_reports'][s]['unweighted_fold_mean']['macro_f1'] for n in ['old_N', 'adapt', 'control', 'N']]
        lines.append('| '+s+' | '+' | '.join(f'{v:.6f}' for v in values)+f' | {values[3]-values[0]:+.6f} |')
    lines += ['', '## 修改范围与来源', '',
              '- 固定 HuBERT 片段向量先按题等权取均值，再对 29 题等权均值，得到每人 768 维输入；新语音投影为 Linear(768,128)+ReLU。均值不学习时间顺序或题目重要性。',
              '- 移除了语音路径中的随机冻结 CNN-BiGRU/单位置注意力及旧语音卷积分类器；EEG 与文本沿用 E 父版本同折 128 维表征，逐位不变。',
              '- 语音投影 98,432 参数，适配时加三个头共 99,206；适配后两组控制分别只训练新头 774 参数。没有使用适配后的头作为控制初始化，也没有继承 NP 权重。',
              '- 适配投影 lr=1e-4、头 lr=1e-3；两组控制头 lr=1e-3。统一 60 轮、batch 8、Adamax、weight_decay=0、梯度上限 1、最后一轮、阈值 0.5。',
              '- 三阶段均为完整 CCML，rho=1.25、delta=gamma=1、KL 退火 50 轮；仅 A-N 添加无仿射 LayerNorm，eps=1e-5。没有调参、早停或新 CE 训练。',
              '- 全部 1,073 题缓存、音频来源、骨干文件与窗口关系通过核验；均值重建逐位一致，无跨受试者统计或标签拟合。原有 96 题空片段（29 人）继续作为零题目向量并计入分母，未补提取短音频；重复尾窗继续保留。',
              '- 跨版本差异包含固定均值汇聚、新投影初始化与 CCML 适配的共同变化，不能将结果全部归因于移除随机编码器。A-N 对 A-control 才是同表征版本的规范化比较。', '',
              '## 验收', '',
              '- 正式训练前，旧 E-control/N 共 1,110 条概率精确复现；新头初始化与每折批序全部配对。',
              '- 45 个检查点受限读取成功，1,665 条全人群预测逐条精确复现，45 个测试折指标与 scikit-learn 交叉核验。',
              '- 15 组新父表征从适配检查点精确重建，EEG/文本逐位不变；语音投影均收到梯度且权重更新。A-control/A-N 使用同一个父缓存和独立新头。',
              f'- {read_json(OUT/"test_results.json")["tests_run"]} 项测试通过，{result["protected_files_unchanged"]} 项历史文件哈希不变。', '',
              '## 泛化与证据诊断', '',
              '| 阶段 | train 融合 F1 | val 融合 F1 | test 融合 F1 | EEG S | 语音 S | 文本 S |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        d = result['diagnostics'][stage]
        values = [d['train_val_test'][split]['final']['macro_f1'] for split in ['train', 'val', 'test']]
        values += [d['mean_strength'][m] for m in ['eeg', 'speech', 'text']]
        lines.append('| '+labels[stage]+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    d = result['diagnostics']['N']
    c = d['changes_by_direction']
    lines += ['', f'A-N 相对旧 N 的 111 条重复种子测试记录中，{d["changed_count"]} 条分类改变：错变对 {c.get("wrong_to_correct",0)}，对变错 {c.get("correct_to_wrong",0)}。它们仍只来自 37 位独立受试者。', '',
              '| A-N 模态 | train F1 | val F1 | test F1 |', '|---|---:|---:|---:|']
    for m in ['eeg', 'speech', 'text']:
        lines.append('| '+m+' | '+' | '.join(f'{d["train_val_test"][split][m]["macro_f1"]:.6f}' for split in ['train', 'val', 'test'])+' |')
    lines += ['', '全部阶段的逐种子合并 OOF、NLL/Brier/ECE5、正确/错误不确定性、风险覆盖率、模态指标、逐类 min 来源、公共/互补贡献、损失和梯度诊断详见 `completion_audit.json`。证据强度不等于可靠性。', '',
              '## 阶段结论与下一项', '']
    if passed and not worse:
        lines.append('A-N 通过相对旧 N 的固定工程门槛，NLL/Brier 未恶化，可保留为新的开发候选；仍不构成独立确认。')
    elif passed:
        lines.append('A-N 的分类指标通过相对旧 N 的门槛，但存在概率质量权衡，不自动替换旧开发参照。')
    else:
        lines.append('A-N 未通过相对旧 N 的固定工程门槛，本轮语音汇聚修订未获得足够的开发推进依据。保留旧 N/NP 比较结果，不追加学习率、epoch 或汇聚搜索。')
    lines += ['下一项可独立固定短音频覆盖修订：只补提取本轮 96 题缺失片段，保持 A 的汇聚/投影及训练设置，并与 A 的三个阶段配对比较。该项本轮尚未启动，不能将现阶段结果解释为短音频修复后的表现。',
              '均值汇聚丢失时序/题目结构，短音频与重复尾窗限制仍存在；EEG 尺度/坐标模板仍未完全闭环。EEG/文本父表征仍有历史监督血缘。结果基于已多次查看的 37 人固定划分，新种子/阶段不增加独立样本数。', '',
              '当前状态以本目录 `status.json` 为准，旧阶段的状态和原计划正文均保留历史快照。', '']
    path = OUT/'CCML-v2-A实验完成报告.md'
    content = '\n'.join(lines)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content, 'Completed audio report changed')
    else:
        path.write_text(content, encoding='utf-8')
    write_new(OUT/'status.json', {'status': 'complete', 'date': '2026-09-22',
              'runs': {'adapt': 15, 'control': 15, 'N': 15}, 'updates': 10800,
              'configuration': descriptor(CONFIG), 'plan': descriptor(PLAN),
              'report': descriptor(path), 'audit': descriptor(OUT/'completion_audit.json'),
              'primary_comparison': 'A-N versus old N', 'development_criteria_passed': passed,
              'probability_quality_worsened': worse, 'execution_blockers': [],
              'short_audio_revision_started': False,
              'next_action': 'Separately register short-audio coverage revision with the same A architecture and matched stages'})
    print(summaries['N']['primary_seed_mean_of_fold_means'])
    print(primary)
    print(f'Audio revision complete: 45 runs, 10800 updates, {result["protected_files_unchanged"]} protected files unchanged.')


if __name__ == '__main__':
    report(audit())
