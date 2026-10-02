"""Audit normalized projection adaptation against the same AS representation parent."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

from .as_projection import (OUT, CONFIG, PLAN, EXPERIMENT, PARENT, CONTROL, as_parent,
                           MODALITIES, PROJECTION_COUNT, TOTAL_COUNT, configuration,
                           protect_history, load_data, verify_initial_parents, predictions,
                           NormalizedProjectionHeads)
from .common import RecordContext, read_json, descriptor, verify_descriptor, require, write_new, set_seed
from .complete_nr import diagnostics
from .complete_p import gate
from .complete_eeg import METRICS, crosscheck_metrics
from .complete_short import fusion_counts
from .rebuild_eeg import validate_row
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.metrics import summarize
import torch


def audit():
    context = RecordContext()
    cfg = read_json(CONFIG)
    require(cfg == configuration(context), 'Registered AS-NP configuration changed')
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG) and
            pre['source_bundles'] == 15 and pre['reconstructed_control_records'] == 1110, 'Preflight incomplete')
    tests = read_json(OUT/'test_results.json')
    require(tests['passed'] and tests['failures'] == 0 and tests['errors'] == 0, 'Tests incomplete')
    for item in tests['sources'].values():
        verify_descriptor(item)
    protected = protect_history()
    rows = {'candidate': [], 'N': [], 'control': []}
    checks, changes = [], []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed)
            data = load_data(context, seed, fold)
            verify_initial_parents(context, seed, fold, data)
            initial = NormalizedProjectionHeads(data['initial'], seed)
            initial_hash = state_digest(initial.state_dict())
            matching = [c for c in pre['checks'] if c['seed'] == seed and c['fold'] == fold]
            require(len(matching) == 1 and matching[0]['source'] == data['source'] and
                    matching[0]['initial_projection_sha256'] == state_digest(data['initial']), 'Preflight source binding mismatch')
            base_meta = {}
            for name in ['control', 'N']:
                folder = PARENT/'runs'/as_parent.STAGES[name]/f'seed_{seed}/fold_{fold}'
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                base_meta[name] = read_json(folder/'checkpoint.json')['metadata']
                rows[name].extend(read_json(folder/'predictions.json'))
            folder = OUT/f'runs/seed_{seed}/fold_{fold}'
            for item in read_json(folder/'complete.json').values():
                verify_descriptor(item)
            cp = folder/'checkpoint.pt'
            saved = torch.load(cp, map_location='cpu', weights_only=True)
            meta = saved['metadata']
            require(meta == read_json(cp.with_suffix('.json'))['metadata'], 'Checkpoint metadata mismatch')
            require(meta['experiment'] == EXPERIMENT and meta['seed'] == seed and meta['fold'] == fold, 'Wrong run identity')
            require(meta['source'] == data['source'] and meta['configuration'] == descriptor(CONFIG), 'Source/config changed')
            require(meta['cohort_sha256'] == context.cohort_sha256 and meta['folds_sha256'] == context.folds_sha256 and
                    meta['train_ids'] == context.folds[f'fold_{fold}']['train'], 'Cohort/train provenance mismatch')
            require(meta['initial_state_sha256'] == initial_hash and meta['initial_projection_sha256'] ==
                    state_digest(data['initial']), 'Initial weights changed')
            require(meta['trainable_parameters'] == TOTAL_COUNT == sum(p.numel() for p in initial.parameters()) and
                    meta['projection_parameters'] == PROJECTION_COUNT, 'Parameter count mismatch')
            require(meta['normalization'] == cfg['normalization'] and meta['eps'] == 1e-5 and
                    meta['projection_lr'] == 1e-4 and meta['head_lr'] == 1e-3 and meta['weight_decay'] == 0., 'Training factors changed')
            require(meta['objective'] == 'full_CCML_only' and meta['epochs'] == 60 and meta['updates'] == 240 and
                    meta['selection'] == 'last_epoch' and meta['dropout'] == 'eval', 'Budget/objective/mode mismatch')
            initial_head = state_digest(make_initial_state(seed))
            require(meta['head_initialization_sha256'] == initial_head and
                    all(m['head_initialization_sha256'] == initial_head for m in base_meta.values()), 'Head pairing mismatch')
            gen, batch = torch.Generator().manual_seed(seed), hashlib.sha256()
            for _ in range(60):
                batch.update(torch.randperm(len(meta['train_ids']), generator=gen).numpy().astype('<i8').tobytes())
            require(meta['batch_order_sha256'] == batch.hexdigest() and
                    all(m['batch_order_sha256'] == batch.hexdigest() for m in base_meta.values()), 'Batch pairing mismatch')
            history = read_json(folder/'train_history.json')
            require([r['epoch'] for r in history] == list(range(1, 61)), 'Missing epoch')
            for h in history:
                require(math.isfinite(h['train_loss']) and math.isclose(h['train_loss'], sum(h['loss_terms'].values()),
                        rel_tol=1e-5, abs_tol=1e-5), 'Loss component mismatch')
                require(all(math.isfinite(v) and v >= 0 for group in ['projection_grad_norm_mean', 'head_grad_norm_mean']
                            for v in h[group].values()), 'Invalid gradient diagnostics')
            model = NormalizedProjectionHeads(data['initial'], seed)
            model.load_state_dict(saved['state_dict'], strict=True)
            require(all(not m.training for m in model.modules()), 'Stochastic training mode enabled')
            require(not any(isinstance(m, (torch.nn.Conv1d, torch.nn.LSTM)) for m in model.modules()), 'Upstream unfreezing introduced')
            drift = {'seed': seed, 'fold': fold, 'projections': {}}
            for modality in MODALITIES:
                require(any(h['projection_grad_norm_mean'][modality] > 0 for h in history), 'Projection received no gradient')
                before = torch.cat([data['initial'][f'{modality}.0.{k}'].flatten() for k in ['weight', 'bias']])
                after = torch.cat([model.projections[modality][0].state_dict()[k].flatten() for k in ['weight', 'bias']])
                require(not torch.equal(before, after), 'Projection did not update')
                drift['projections'][modality] = {'l2_change': float(torch.linalg.vector_norm(after-before)),
                    'relative_l2_change': float(torch.linalg.vector_norm(after-before)/torch.linalg.vector_norm(before))}
            changes.append(drift)
            observed = read_json(folder/'predictions.json')
            require(observed == predictions(model, data, context, seed, fold, cp), 'Predictions do not exactly reproduce')
            require(len(observed) == 37 and [r['subject_id'] for r in observed] == context.ids, 'Subject coverage mismatch')
            for row in observed:
                validate_row(row)
            rows['candidate'].extend(observed)
            checks.append({'seed': seed, 'fold': fold, 'checkpoint': descriptor(cp), 'exact_reproduced_rows': 37,
                           'all_projection_parents_verified': True, 'all_three_projections_updated': True,
                           'head_and_batch_pairing_verified': True, 'updates': 240})
    for values in rows.values():
        require(len(values) == 555 and Counter((r['seed'], r['subject_id']) for r in values if r['split'] == 'test') ==
                Counter((s, i) for s in cfg['seeds'] for i in context.ids), 'OOF coverage incomplete')
    summaries = {name: summarize(values) for name, values in rows.items()}
    parent_audit = read_json(PARENT/'completion_audit.json')
    require(all(summaries[name] == parent_audit['summaries']['AS'][name] for name in ['N', 'control']), 'AS summary drift')
    result = {'date': '2026-09-23', 'passed': True, 'formal_runs': 15, 'updates': 3600,
              'subject_fold_rows': 555, 'independent_subjects': 37, 'summaries': summaries,
              'references': parent_audit['references'], 'primary_comparison': 'candidate_vs_AS_N',
              'comparisons': {name: gate(summaries['candidate'], summaries[name]) for name in ['N', 'control']},
              'diagnostics': {name: diagnostics(rows['candidate'], rows[name]) for name in ['N', 'control']},
              'speech_fusion_counts': {name: fusion_counts([r for r in values if r['split'] == 'test']) for name, values in rows.items()},
              'projection_changes': changes, 'run_checks': checks,
              'sklearn_fold_crosschecks': crosscheck_metrics(rows['candidate']),
              'configuration': descriptor(CONFIG), 'preflight': descriptor(OUT/'preflight.json'),
              'tests': descriptor(OUT/'test_results.json'), 'reporter': descriptor(__file__),
              'protected_files_unchanged': protect_history(),
              'scope': 'AS same-parent normalization plus limited projection adaptation; no CCML formula/loss changes'}
    require(result['protected_files_unchanged'] == protected, 'Historical artifacts changed during audit')
    write_new(OUT/'completion_audit.json', result)
    return result


def report(result):
    comp = result['comparisons']['N']
    passed = comp['gate']['development_criteria_passed']
    worse = comp['gate']['nll_worsened'] or comp['gate']['brier_worsened']
    score = result['summaries']['candidate']['primary_seed_mean_of_fold_means']
    lines = ['# CCML-v2-AS-NP 同版本有限投影适配完成报告', '',
             '日期：2026-09-23。**三个种子×五折的 15 组训练完成，共 3,600 次更新，验收通过。**', '',
             '## 主要结果', '',
             '直接比较为 AS-NP 对 AS-N：相同覆盖完整语音来源、相同初始父投影、新头及批序，增加三路末端投影在规范化后完整 CCML 训练中的可更新性。主口径为每种子五折均值，再平均三个种子。', '',
             '| 实验 | Macro-F1 | MDD recall | BA | NLL | Brier |', '|---|---:|---:|---:|---:|---:|']
    entries = [('AS-control', result['summaries']['control']), ('AS-N：直接对照', result['summaries']['N']),
               ('AS-NP：有限投影适配', result['summaries']['candidate']),
               ('旧 N：历史开发参照', result['references']['old_N']), ('旧 NP：历史开发参照', result['references']['NP'])]
    for label, summary in entries:
        m = summary['primary_seed_mean_of_fold_means']
        lines.append('| '+label+' | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    lines += ['', '| 配对比较 | F1 变化（百分点） | 召回变化（百分点） | 改善种子数 | 门槛 | NLL 变化 | Brier 变化 |',
              '|---|---:|---:|---:|---|---:|---:|']
    for name, comparison in result['comparisons'].items():
        d = comparison['delta']
        positive = sum(v > 0 for v in comparison['seed_f1_deltas'].values())
        decision = '通过' if comparison['gate']['development_criteria_passed'] else '未通过'
        lines.append(f'| AS-NP−AS-{name} | {100*d["macro_f1"]:+.2f} | {100*d["mdd_recall"]:+.2f} | {positive}/3 | {decision} | {d["nll"]:+.6f} | {d["brier"]:+.6f} |')
    lines += ['', '主门槛预先固定为相对 AS-N：平均 F1 至少 +0.01，至少 2/3 种子严格改善，召回下降不超过 0.05；完整报告 NLL/Brier 权衡。该门槛用于开发，不是显著性或独立确认。', '',
              '| 种子 | AS-N F1 | AS-NP F1 | 差值 |', '|---|---:|---:|---:|']
    for seed in ['11', '22', '33']:
        a, b = [result['summaries'][name]['seed_reports'][seed]['unweighted_fold_mean']['macro_f1'] for name in ['N', 'candidate']]
        lines.append(f'| {seed} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines += ['', '## 参数、来源与验收', '',
              '- EEG 1920→128、语音 768→128、文本 1024→128 三路末端投影共 475,520 参数，加三个新证据头 774，共 **476,294**。无仿射 LayerNorm，eps=1e-5；EEG/text Dropout保持 eval，语音路径没有 Dropout。',
              '- EEG/text 使用 P 导出时保存的合法初始父投影及投影前缓存，未使用 P/NP 训练后投影；语音使用同折 AS-adapt 父投影及 AS 完整 pooled 向量。三路初始参数与实际父检查点逐项核验，拟合名单一致。',
              '- 语音全部 1,073 题有效特征沿用 AS；CNN/LSTM及骨干冻结在缓存上游。新头独立共同初始化，未继承 AS-N/AS-adapt/旧 NP 头或优化器状态。',
              '- 固定投影 lr=1e-4、头 lr=1e-3、Adamax、weight_decay=0、梯度上限 1；60 轮、batch 8、最后一轮、阈值 0.5。完整 CCML，rho=1.25、delta=gamma=1、KL 退火 50 轮。没有调参、早停或新 CE 训练。',
              '- 初始路径重建 AS 父表征通过固定容差；复算 1,110 条 AS-control/N 记录，概率满足固定容差且所有分类一致。新头与两个对照初始化、批序相同。',
              '- 15 个新检查点受限重载成功，555 条全人群预测逐条精确复现；三路投影每折均收到梯度且实际更新，训练名单、损失分项、融合算术与15个测试折指标交叉核验通过。',
              f'- {read_json(OUT/"test_results.json")["tests_run"]} 项测试通过，{result["protected_files_unchanged"]} 项历史文件哈希不变。', '',
              '## 分支与融合诊断', '',
              '| AS-NP 模态 | train F1 | val F1 | test F1 | test 平均 S |', '|---|---:|---:|---:|---:|']
    diag = result['diagnostics']['N']
    for modality in ['final', 'eeg', 'speech', 'text']:
        values = [diag['train_val_test'][split][modality]['macro_f1'] for split in ['train', 'val', 'test']]
        strength = diag['mean_strength'].get(modality)
        lines.append('| '+modality+' | '+' | '.join(f'{v:.6f}' for v in values)+' | '+(f'{strength:.6f}' if strength is not None else '—')+' |')
    old_diag = read_json(PARENT/'completion_audit.json')['diagnostics']['N']
    lines += ['', '| 指标 | AS-N | AS-NP |', '|---|---:|---:|',
              '| 语音单分支 test F1 | '+f'{old_diag["train_val_test"]["test"]["speech"]["macro_f1"]:.6f} | {diag["train_val_test"]["test"]["speech"]["macro_f1"]:.6f} |']
    for key, label in [('speech_correct', '语音正确记录'), ('final_correct', '融合正确记录'),
                       ('speech_correct_final_wrong', '语音正确、融合错误'), ('speech_wrong_final_correct', '语音错误、融合正确')]:
        lines.append('| '+label+' | '+' | '.join(str(result['speech_fusion_counts'][name][key]) for name in ['N', 'candidate'])+' |')
    changes = diag['changes_by_direction']
    lines += ['', f'相比 AS-N，111 条重复种子测试记录中 {diag["changed_count"]} 条分类改变：错变对 {changes.get("wrong_to_correct",0)}，对变错 {changes.get("correct_to_wrong",0)}。这些记录仍来自 37 位独立受试者。',
              '逐种子合并 OOF、NLL/Brier/ECE5、不确定性/风险覆盖率、min 来源、公共/互补贡献及逐路参数位移均保存在 `completion_audit.json`。证据强度本身不等于可靠性。', '',
              '## 阶段结论', '']
    if passed:
        lines.append('AS-NP 通过相对 AS-N 的固定分类门槛；'+('存在 NLL/Brier 恶化，需要保留概率质量权衡，不自动升级默认。' if worse else 'NLL/Brier 未恶化，可保留为同版本开发候选。'))
    else:
        lines.append('AS-NP 未通过相对 AS-N 的固定工程门槛，不将投影解冻自动作为下一阶段默认，不追加轮数或网格搜索。')
    old_np = result['references']['NP']['primary_seed_mean_of_fold_means']
    old_n = result['references']['old_N']['primary_seed_mean_of_fold_means']
    lines += [f'相对旧 N 与旧 NP 的 F1 差值分别为 {score["macro_f1"]-old_n["macro_f1"]:+.6f}、{score["macro_f1"]-old_np["macro_f1"]:+.6f}。这些是不同语音表征及参数范围的开发参照，不能归因于单个模块。',
              '下一阶段先围绕本轮分支与融合分歧形成单独固定方案。当前没有引入证据缩放、动态门控或新增融合损失；这些若改变 CCML，需要明确扩展边界，不能混入本轮结果。',
              '全部结果基于已多次查看的 37 人固定划分；新增阶段与种子不增加独立样本。EEG 尺度/坐标模板、均值汇聚的结构损失及上游历史监督血缘等限制仍在。独立确认仍需新受试者。', '',
              '当前状态以本目录 `status.json` 为准，旧阶段产物均保留。', '']
    path = OUT/'CCML-v2-AS-NP实验完成报告.md'
    text = '\n'.join(lines)
    if path.exists():
        require(path.read_text(encoding='utf-8') == text, 'Completed report changed')
    else:
        path.write_text(text, encoding='utf-8')
    write_new(OUT/'status.json', {'status': 'complete', 'date': '2026-09-23', 'runs': 15, 'updates': 3600,
              'trainable_parameters': TOTAL_COUNT, 'configuration': descriptor(CONFIG), 'plan': descriptor(PLAN),
              'report': descriptor(path), 'audit': descriptor(OUT/'completion_audit.json'),
              'primary_comparison': 'AS-NP versus AS-N', 'development_criteria_passed': passed,
              'probability_quality_worsened': worse, 'execution_blockers': [],
              'next_action': 'Review same-version branch/fusion diagnostics and register a separate fixed next hypothesis; no implicit CCML extension'})
    print(score)
    print(comp)
    print(f'AS-NP complete: 15 runs, 3600 updates, {result["protected_files_unchanged"]} historical files unchanged.')


if __name__ == '__main__':
    report(audit())
