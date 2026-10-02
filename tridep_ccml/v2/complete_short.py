"""Paired completion audit of short-audio coverage against the fixed A stages."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

from . import audio as parent
from . import short_train as train
from .short_audio import (OUT, CONFIG, PLAN, STAGES, configuration, protect_history,
                          verify_short_reextraction, replace_short_means)
from .common import (RecordContext, read_json, descriptor, verify_descriptor, require,
                     write_new, set_seed)
from .complete_nr import diagnostics
from .complete_p import gate
from .complete_eeg import crosscheck_metrics, METRICS
from .rebuild_eeg import validate_row
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.metrics import summarize
import numpy as np
import torch


def audit_pool(context):
    info = read_json(OUT/'speech_pool.json')
    old_info = read_json(verify_descriptor(info['parent_manifest']))
    require(info['configuration'] == descriptor(CONFIG) and info['subject_ids'] == context.ids and
            info['fit_subject_ids'] == [] and info['labels_used'] is False, 'AS pool provenance changed')
    with np.load(verify_descriptor(info['cache']), allow_pickle=False) as a:
        new = {k: a[k].copy() for k in a.files}
    with np.load(verify_descriptor(old_info['cache']), allow_pickle=False) as a:
        old = {k: a[k].copy() for k in a.files}
    require(new['subject_ids'].tolist() == old['subject_ids'].tolist() == context.ids, 'Pool identities changed')
    lookup = {s: i for i, s in enumerate(context.ids)}
    old_sources = {(r['subject_id'], r['question']): r for r in old_info['sources']}
    replacements = {}
    require(len(info['repaired']) == 96, 'Incorrect replacement count')
    for row in info['repaired']:
        key = (row['subject_id'], row['question'])
        source = old_sources[key]
        require(source['empty'], 'Nonempty question replaced')
        meta = read_json(verify_descriptor(row['metadata']))
        require(meta['previous_source'] == source and meta['feature'] == row['feature'] and
                meta['configuration'] == descriptor(CONFIG), 'Replacement source binding changed')
        require(meta['fit_subject_ids'] == [] and meta['labels_used'] is False and
                meta['padding_samples'] == 0 and meta['window_samples'] == [0, source['processed_samples']], 'Unexpected short policy')
        replacements[lookup[key[0]], key[1]-1] = np.load(verify_descriptor(row['feature']), allow_pickle=False)
    questions, counts = replace_short_means(old['question_means'], old['segment_counts'], replacements)
    pooled = np.stack([parent.pool_subject(q) for q in questions])
    require(np.array_equal(questions, new['question_means']) and np.array_equal(counts, new['segment_counts']) and
            np.array_equal(pooled, new['pooled']), 'Repaired pool does not reconstruct exactly')
    empty = old['segment_counts'] == 0
    affected = empty.any(axis=1)
    require(int(empty.sum()) == 96 and int((~empty).sum()) == 977 and (counts > 0).all(), 'Coverage mismatch')
    require(int(affected.sum()) == 29 and np.array_equal(pooled[~affected], old['pooled'][~affected]), 'Unaffected input changed')
    require(info['affected_subject_ids'] == [s for i, s in enumerate(context.ids) if affected[i]] and
            info['unchanged_subject_ids'] == [s for i, s in enumerate(context.ids) if not affected[i]], 'Affected subgroup mismatch')
    return {'questions_covered': 1073, 'repaired': 96, 'unchanged_question_means_exact': 977,
            'unaffected_subject_inputs_exact': 8, 'affected_subject_ids': info['affected_subject_ids'],
            'unchanged_subject_ids': info['unchanged_subject_ids'], 'pool_reconstruction_exact': True}


def fusion_counts(rows):
    result = Counter()
    for row in rows:
        speech_correct = int(row['views']['speech']['p'][1] > .5) == row['label']
        final_correct = row['prediction'] == row['label']
        result['records'] += 1
        result['speech_correct'] += speech_correct
        result['final_correct'] += final_correct
        result['speech_correct_final_wrong'] += speech_correct and not final_correct
        result['speech_wrong_final_correct'] += not speech_correct and final_correct
    return dict(result)


def subgroup_comparison(new_rows, old_rows, ids):
    new = [r for r in new_rows if r['split'] == 'test' and r['subject_id'] in ids]
    old = [r for r in old_rows if r['split'] == 'test' and r['subject_id'] in ids]
    before = {(r['seed'], r['subject_id']): r for r in old}
    counter = Counter()
    for r in new:
        previous = before[r['seed'], r['subject_id']]
        require(previous['label'] == r['label'], 'Paired label mismatch')
        if r['prediction'] != previous['prediction']:
            counter['wrong_to_correct' if r['prediction'] == r['label'] else 'correct_to_wrong'] += 1
    return {'independent_subjects': len(ids), 'test_records': len(new), 'A': fusion_counts(old),
            'AS': fusion_counts(new), 'changed_directions': dict(counter)}


def audit():
    context = RecordContext()
    cfg = read_json(CONFIG)
    require(cfg == configuration(context), 'AS configuration changed')
    old_cfg = read_json(parent.CONFIG)
    allowed = {'family', 'plan', 'sources', 'stages', 'speech', 'comparisons', 'prior_status'}
    require(all(cfg[k] == v for k, v in old_cfg.items() if k not in allowed), 'A training settings changed')
    require(all(cfg['speech'][k] == v for k, v in old_cfg['speech'].items() if k != 'short_question'), 'Speech architecture/pooling changed')
    protected = protect_history()
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG) and pre['exact_A_records'] == 1665, 'AS preflight incomplete')
    for name in ['pool', 'extraction']:
        verify_descriptor(pre[name])
    extraction = verify_short_reextraction()
    pool = audit_pool(context)
    tests = read_json(OUT/'test_results.json')
    require(tests['passed'] and tests['failures'] == 0 and tests['errors'] == 0, 'Tests incomplete')
    for item in tests['sources'].values():
        verify_descriptor(item)
    rows = {v: {s: [] for s in STAGES} for v in ['A', 'AS']}
    checks, changes = [], []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            initial_head = state_digest(make_initial_state(seed))
            gen, batch = torch.Generator().manual_seed(seed), hashlib.sha256()
            for _ in range(60):
                batch.update(torch.randperm(len(context.folds[f'fold_{fold}']['train']), generator=gen).numpy().astype('<i8').tobytes())
            new_parent_sources = []
            for stage in STAGES:
                set_seed(seed)
                data = train.load_data(context, seed, fold, stage)
                old_data = parent.load_data(context, seed, fold, stage)
                require(np.array_equal(data['features'][:, [0, 2]], old_data['features'][:, [0, 2]]), 'A/AS EEG-text differ')
                if stage == 'adapt':
                    require(data['source']['frozen_parent'] == old_data['source']['frozen_parent'], 'Frozen parent changed')
                model = train.make_model(seed, stage)
                initial_hash = state_digest(model.state_dict())
                initial_proj = {k: v.clone() for k, v in model.speech_projection.state_dict().items()} if stage == 'adapt' else None
                previous_dir = parent.OUT/'runs'/parent.STAGES[stage]/f'seed_{seed}/fold_{fold}'
                for item in read_json(previous_dir/'complete.json').values():
                    verify_descriptor(item)
                old_meta = read_json(previous_dir/'checkpoint.json')['metadata']
                rows['A'][stage].extend(read_json(previous_dir/'predictions.json'))
                folder = OUT/'runs'/STAGES[stage]/f'seed_{seed}/fold_{fold}'
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                cp = folder/'checkpoint.pt'
                saved = torch.load(cp, map_location='cpu', weights_only=True)
                meta = saved['metadata']
                require(meta == read_json(cp.with_suffix('.json'))['metadata'], 'Checkpoint metadata mismatch')
                require(meta['experiment'] == STAGES[stage] and meta['stage'] == stage and
                        meta['seed'] == seed and meta['fold'] == fold, 'Run identity mismatch')
                require(meta['configuration'] == descriptor(CONFIG) and meta['source'] == data['source'], 'Source/config mismatch')
                require(meta['train_ids'] == old_meta['train_ids'] == context.folds[f'fold_{fold}']['train'] and
                        meta['cohort_sha256'] == context.cohort_sha256 and meta['folds_sha256'] == context.folds_sha256, 'Training/cohort mismatch')
                require(meta['initial_state_sha256'] == old_meta['initial_state_sha256'] == initial_hash and
                        meta['head_initialization_sha256'] == old_meta['head_initialization_sha256'] == initial_head and
                        meta['batch_order_sha256'] == old_meta['batch_order_sha256'] == batch.hexdigest(), 'A/AS initialization/batch mismatch')
                for key in ['objective', 'epochs', 'updates', 'selection', 'head_lr', 'weight_decay', 'projection_lr',
                            'trainable_parameters', 'normalization', 'initial_projection_sha256', 'device']:
                    require(meta[key] == old_meta[key], f'A/AS training factor changed: {key}')
                require(meta['objective'] == 'full_CCML_only' and meta['epochs'] == 60 and meta['updates'] == 240, 'Budget/objective changed')
                require(meta['trainable_parameters'] == sum(p.numel() for p in model.parameters()) ==
                        (99206 if stage == 'adapt' else 774), 'Wrong parameter count')
                history = read_json(folder/'train_history.json')
                require([r['epoch'] for r in history] == list(range(1, 61)), 'Missing training epoch')
                for h in history:
                    require(math.isfinite(h['train_loss']) and math.isclose(h['train_loss'], sum(h['loss_terms'].values()),
                            rel_tol=1e-5, abs_tol=1e-5), 'Loss components disagree')
                    require(all(math.isfinite(v) and v >= 0 for v in h['head_grad_norm_mean'].values()), 'Invalid gradients')
                model.load_state_dict(saved['state_dict'], strict=True)
                if stage == 'adapt':
                    require(meta['initial_projection_sha256'] == state_digest(initial_proj), 'Projection initialization changed')
                    require(any(h['speech_projection_grad_norm_mean'] > 0 for h in history) and
                            all(math.isfinite(h['speech_projection_grad_norm_mean']) for h in history), 'Projection gradients missing')
                    before = torch.cat([v.flatten() for v in initial_proj.values()])
                    after = torch.cat([v.flatten() for v in model.speech_projection.state_dict().values()])
                    require(not torch.equal(before, after), 'Projection never updated')
                    changes.append({'seed': seed, 'fold': fold, 'l2_change': float(torch.linalg.vector_norm(after-before)),
                                    'relative_l2_change': float(torch.linalg.vector_norm(after-before)/torch.linalg.vector_norm(before))})
                    exported = train.load_data(context, seed, fold, 'control')
                    rebuilt = train.forward_all(model, data, stage, representations=True).numpy()
                    require(np.array_equal(rebuilt, exported['features']) and
                            np.array_equal(rebuilt[:, [0, 2]], data['features'][:, [0, 2]]), 'New representations changed')
                    require(exported['source']['parent_checkpoint'] == descriptor(cp), 'New parent checkpoint differs')
                else:
                    require(all(h['speech_projection_grad_norm_mean'] is None for h in history), 'Frozen projection received gradient')
                    new_parent_sources.append(data['source'])
                observed = read_json(folder/'predictions.json')
                require(observed == train.predictions(model, data, context, seed, fold, stage, cp), 'AS predictions do not reproduce exactly')
                require(len(observed) == 37 and [r['subject_id'] for r in observed] == context.ids, 'Prediction subject coverage mismatch')
                for r in observed:
                    validate_row(r)
                rows['AS'][stage].extend(observed)
                checks.append({'seed': seed, 'fold': fold, 'stage': stage, 'checkpoint': descriptor(cp),
                               'exact_replayed_rows': 37, 'A_pairing_verified': True, 'updates': 240})
            require(new_parent_sources[0] == new_parent_sources[1], 'AS control and N parent differs')
    for version in rows.values():
        for records in version.values():
            require(len(records) == 555 and Counter((r['seed'], r['subject_id']) for r in records if r['split'] == 'test') ==
                    Counter((s, i) for s in cfg['seeds'] for i in context.ids), 'Incomplete test coverage')
    summaries = {v: {s: summarize(r) for s, r in stages.items()} for v, stages in rows.items()}
    old_audit = read_json(parent.OUT/'completion_audit.json')
    require(all(summaries['A'][s] == old_audit['summaries'][s] for s in STAGES), 'A summary drift')
    comparisons = {f'AS_{s}_vs_A_{s}': gate(summaries['AS'][s], summaries['A'][s]) for s in STAGES}
    comparisons['AS_N_vs_AS_control'] = gate(summaries['AS']['N'], summaries['AS']['control'])
    result = {'date': '2026-09-22', 'passed': True, 'formal_runs': 45, 'updates': 10800,
              'subject_fold_rows': 1665, 'independent_subjects': 37, 'summaries': summaries,
              'references': {'old_N': old_audit['summaries']['old_N'], 'NP': old_audit['NP_reference']},
              'comparisons': comparisons, 'primary_comparison': 'AS_N_vs_A_N',
              'diagnostics': {s: diagnostics(rows['AS'][s], rows['A'][s]) for s in STAGES},
              'same_version_N_diagnostics': diagnostics(rows['AS']['N'], rows['AS']['control']),
              'speech_fusion_counts': {v: {s: fusion_counts([r for r in rr if r['split'] == 'test']) for s, rr in vv.items()} for v, vv in rows.items()},
              'input_subgroups': {group: {'all_seeds': subgroup_comparison(rows['AS']['N'], rows['A']['N'], pool[key]),
                    'by_seed': {str(s): subgroup_comparison([r for r in rows['AS']['N'] if r['seed'] == s],
                    [r for r in rows['A']['N'] if r['seed'] == s], pool[key]) for s in cfg['seeds']}}
                    for group, key in [('affected', 'affected_subject_ids'), ('unchanged', 'unchanged_subject_ids')]},
              'pool_audit': pool, 'raw_short_reextraction': extraction,
              'run_checks': checks, 'projection_changes': changes,
              'sklearn_fold_crosschecks': {s: crosscheck_metrics(rows['AS'][s]) for s in STAGES},
              'configuration': descriptor(CONFIG), 'preflight': descriptor(OUT/'preflight.json'),
              'tests': descriptor(OUT/'test_results.json'), 'reporter': descriptor(__file__),
              'protected_files_unchanged': protect_history(),
              'scope': 'only 96 short-question inputs changed from A; same fixed architecture/optimization; viewed developmental cohort'}
    require(result['protected_files_unchanged'] == protected, 'Historical files changed during audit')
    write_new(OUT/'completion_audit.json', result)
    return result


def report(result):
    primary = result['comparisons']['AS_N_vs_A_N']
    passed = primary['gate']['development_criteria_passed']
    worse = primary['gate']['nll_worsened'] or primary['gate']['brier_worsened']
    score = result['summaries']['AS']['N']['primary_seed_mean_of_fold_means']
    old_n = result['references']['old_N']['primary_seed_mean_of_fold_means']
    old_np = result['references']['NP']['primary_seed_mean_of_fold_means']
    lines = ['# CCML-v2-AS 短音频覆盖修订完成报告', '',
             '日期：2026-09-22。**96 题短音频已补提取；适配、控制、规范化各 15 组完成，共 45 组、10,800 次更新，验收通过。**', '',
             '## 配对结果', '',
             '主要判断为 AS-N 对 A-N。主口径是每种子五折均值，再取三种子均值；A 与 AS 的投影/头初始化、批序、架构、优化与预算全部配对，唯一输入变化为原 96 题零向量替换为实际短音频 HuBERT 向量。', '',
             '| 实验 | Macro-F1 | MDD recall | BA | NLL | Brier |', '|---|---:|---:|---:|---:|---:|']
    for stage in STAGES:
        for version in ['A', 'AS']:
            values = result['summaries'][version][stage]['primary_seed_mean_of_fold_means']
            lines.append('| '+version+'-'+stage+' | '+' | '.join(f'{values[k]:.6f}' for k in METRICS)+' |')
    for name, values in [('旧 N 开发参照', old_n), ('NP 开发参照', old_np)]:
        lines.append('| '+name+' | '+' | '.join(f'{values[k]:.6f}' for k in METRICS)+' |')
    lines += ['', '| 比较 | F1 变化（百分点） | 召回变化（百分点） | F1 改善种子数 | 门槛 | NLL 变化 | Brier 变化 |',
              '|---|---:|---:|---:|---|---:|---:|']
    for name, comp in result['comparisons'].items():
        d, g = comp['delta'], comp['gate']
        label = name.replace('_vs_', ' − ').replace('_', '-')
        positives = sum(v > 0 for v in comp['seed_f1_deltas'].values())
        decision = '通过' if g['development_criteria_passed'] else '未通过'
        lines.append(f'| {label} | {100*d["macro_f1"]:+.2f} | {100*d["mdd_recall"]:+.2f} | {positives}/3 | {decision} | {d["nll"]:+.6f} | {d["brier"]:+.6f} |')
    lines += ['', '工程门槛训练前固定为 F1 至少 +0.01、至少 2/3 种子严格提高、MDD recall 下降不超过 0.05；NLL/Brier 越低越好，权衡完整报告。该门槛不是显著性检验。', '',
              '| 种子 | A-N F1 | AS-N F1 | 差值 |', '|---|---:|---:|---:|']
    for s in ['11', '22', '33']:
        a, b = [result['summaries'][v]['N']['seed_reports'][s]['unweighted_fold_mean']['macro_f1'] for v in ['A', 'AS']]
        lines.append(f'| {s} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    ext = read_json(OUT/'extraction_audit.json')
    lines += ['', '## 覆盖与数值验收', '',
              '- 96 题原空片段全部补齐，涉及 29 人；1,073 题现在均有实际 HuBERT 向量。其余 977 题题目均值逐位复用 A，原尾窗重复策略保留。',
              '- 短音频预处理长度 14,848–79,872 点，按 16 kHz 实际长度直接推理，不补零至 5 秒、不复制波形。预处理、PCM16 往返和冻结 HuBERT 均沿用原实现。',
              f'- 每位受试者复算一个已有长片段，共 37 个；固定 rtol=atol=2e-5 内一致，最大绝对误差 {ext["long_control_max_abs_error"]:.9g}。没有改动长题缓存。',
              f'- HuBERT 的 {len(ext["runtime"]["retained_weight_mapping"])} 项保留状态均与锁定权重一致，全模型冻结 eval，提取前后状态哈希相同。全部 96 题重复推理、训练后从原音频再次提取均逐位一致。',
              '- 8 位没有原空题的受试者，投影前 pooled 向量逐位不变。由于投影重新训练，这些人的后续预测仍可改变。', '',
              '## 训练与检查', '',
              '- 保持 A 的片段均值→29 题等权均值→Linear(768,128)+ReLU；适配训练投影及三个头共 99,206 参数，控制与规范化各 774 参数。均值仍不建模时间顺序或题目结构。',
              '- 投影 lr=1e-4、头 lr=1e-3；60 轮、batch 8、Adamax、weight_decay=0、梯度上限 1；完整 CCML，rho=1.25、delta=gamma=1、KL 退火 50 轮；最后一轮、阈值 0.5。仅 N 阶段添加无仿射 LayerNorm，eps=1e-5。',
              '- AS 从与 A 相同的初始投影及新头开始，未继承 A 的已训练参数。三个阶段固定预算全部报告，没有搜索、早停或新 CE 训练。',
              '- 训练前 A 的 1,665 条预测精确重放；45 个 AS 检查点及 1,665 条新证据预测精确重载，45 个测试折指标经 scikit-learn 交叉核验。',
              '- 15 组新父特征从适配检查点精确重建；EEG/文本与 A 逐位不变，语音投影确有梯度和参数更新，两个控制使用同一新父缓存及各自独立初始化的新头。',
              f'- {read_json(OUT/"test_results.json")["tests_run"]} 项测试通过，{result["protected_files_unchanged"]} 项历史文件哈希保持不变。', '',
              '## 分支与融合', '',
              '| 指标 | A-N | AS-N |', '|---|---:|---:|']
    old_diag = read_json(parent.OUT/'completion_audit.json')['diagnostics']['N']
    new_diag = result['diagnostics']['N']
    for label, field in [('语音单分支 test F1', 'speech'), ('最终融合 test F1', 'final')]:
        lines.append('| '+label+' | '+' | '.join(f'{d["train_val_test"]["test"][field]["macro_f1"]:.6f}' for d in [old_diag, new_diag])+' |')
    for key, label in [('speech_correct', '语音正确记录'), ('final_correct', '融合正确记录'),
                       ('speech_correct_final_wrong', '语音正确、融合错误'), ('speech_wrong_final_correct', '语音错误、融合正确')]:
        lines.append('| '+label+' | '+' | '.join(str(result['speech_fusion_counts'][v]['N'][key]) for v in ['A', 'AS'])+' |')
    lines += ['', '计数来自三个种子的 111 条 OOF 记录，仍是 37 位独立受试者；正确条数不是五折均值 F1。', '',
              '| AS-N 模态 | train F1 | val F1 | test F1 | test 平均 S |', '|---|---:|---:|---:|---:|']
    for m in ['final', 'eeg', 'speech', 'text']:
        values = [new_diag['train_val_test'][split][m]['macro_f1'] for split in ['train', 'val', 'test']]
        strength = new_diag['mean_strength'].get(m)
        lines.append('| '+m+' | '+' | '.join(f'{v:.6f}' for v in values)+' | '+(f'{strength:.6f}' if strength is not None else '—')+' |')
    lines += ['', '## 输入覆盖分组（描述性）', '',
              '| 分组 | 独立人数 | 重复种子记录 | A-N 正确 | AS-N 正确 | 错变对 | 对变错 |', '|---|---:|---:|---:|---:|---:|---:|']
    for group, label in [('affected', '原有短题者'), ('unchanged', '原无短题者')]:
        g = result['input_subgroups'][group]['all_seeds']
        changes = g['changed_directions']
        lines.append(f'| {label} | {g["independent_subjects"]} | {g["test_records"]} | {g["A"]["final_correct"]} | {g["AS"]["final_correct"]} | {changes.get("wrong_to_correct",0)} | {changes.get("correct_to_wrong",0)} |')
    lines += ['', '分组只描述输入覆盖变化，不用于选择参数；训练数据变化会使模型影响所有测试人，不能把分组差异当作个体治疗或因果效应。',
              '完整逐种子 OOF、ECE5、不确定性/风险覆盖率、各模态指标、min 来源、公共/互补证据、梯度与参数位移见 `completion_audit.json`。', '',
              '## 决策与后续', '']
    if passed:
        lines.append('AS-N 通过相对 A-N 的预定分类门槛；'+('NLL/Brier 有恶化，保留概率质量权衡。' if worse else 'NLL/Brier 未恶化。'))
    else:
        lines.append('AS-N 未通过相对 A-N 的预定工程门槛。覆盖修复完成，但没有获得足够的融合性能推进依据。')
    lines += [f'AS-N 相对旧 N 和 NP 的 F1 差值分别为 {score["macro_f1"]-old_n["macro_f1"]:+.6f}、{score["macro_f1"]-old_np["macro_f1"]:+.6f}。相对 A 的改善不能自动视为超过旧主线，不自动替换 N/NP。',
              '保留 AS 作为覆盖完整且可复现的语音输入版本。下一阶段应依据分支与融合分歧另立固定方案；本轮不追加汇聚/学习率/阈值搜索，不修改 CCML 公式或按测试结果手工缩放证据。',
              '所有结果仍是多次查看过的 37 人固定划分上的开发比较，新增阶段和种子不增加独立样本。EEG 尺度/坐标模板及均值汇聚的结构损失等限制仍在，EEG/文本父表征仍有历史监督血缘。独立确认需要新受试者数据。', '',
              '当前状态以本目录 `status.json` 为准；此前 A/NP 等状态保留为历史快照。', '']
    path = OUT/'CCML-v2-AS实验完成报告.md'
    content = '\n'.join(lines)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content, 'Completed AS report changed')
    else:
        path.write_text(content, encoding='utf-8')
    write_new(OUT/'status.json', {'status': 'complete', 'date': '2026-09-22', 'repaired_questions': 96,
              'questions_covered': 1073, 'runs': {'adapt': 15, 'control': 15, 'N': 15}, 'updates': 10800,
              'configuration': descriptor(CONFIG), 'plan': descriptor(PLAN), 'report': descriptor(path),
              'audit': descriptor(OUT/'completion_audit.json'), 'primary_comparison': 'AS-N versus A-N',
              'development_criteria_passed': passed, 'probability_quality_worsened': worse,
              'automatically_replaces_N_or_NP': False, 'execution_blockers': [],
              'next_action': 'Review paired speech/fusion diagnostics and separately register the next unchanged-CCML candidate; no implicit scaling or search'})
    print(score)
    print(primary)
    print(f'AS complete: 96 repaired questions, 45 runs, 10800 updates, {result["protected_files_unchanged"]} historical files unchanged.')


if __name__ == '__main__':
    report(audit())
