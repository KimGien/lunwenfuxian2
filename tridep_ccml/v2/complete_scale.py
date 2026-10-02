"""Independent CCML-S completion audit, nested selection replay, and report."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import os

from .scale import (OUT, CONFIG, PLAN, STAGES, MODALITIES, METRICS, LAMBDAS, ScaleModel,
                    load_setup, predictions, run_folder, select_candidate, protect_history,
                    build_inputs, nested_splits)
from .common import WORK, TC, descriptor, verify_descriptor, read_json, write_new, require, set_seed
from .rebuild_eeg import validate_row
from .complete_eeg import mean_scores, crosscheck_metrics
from .complete_p import gate
from tridep_ccml.metrics import summarize
from tridep_ccml.train import state_digest
import numpy as np
import torch


def inspect_run(context, data, cfg, seed, fold, inner, kind, strength, train_ids, split_map):
    folder = run_folder(seed, fold, inner, kind, strength)
    for d in read_json(folder/'complete.json').values():
        verify_descriptor(d)
    side = read_json(folder/'checkpoint.json')
    saved = torch.load(verify_descriptor(side['checkpoint']), map_location='cpu', weights_only=True)
    meta = saved['metadata']
    require(meta == side['metadata'], 'Checkpoint sidecar differs')
    require(meta['experiment'] == STAGES[kind] and meta['seed'] == seed and meta['fold'] == fold and
            meta['inner_fold'] == inner and meta['kind'] == kind and meta['lambda'] == strength, 'Wrong run identity')
    require(meta['train_ids'] == train_ids and meta['split_map'] == split_map and
            meta['evaluation_ids'] == [s for s in context.ids if s in split_map], 'Illegal fitting/evaluation identities')
    require(meta['configuration'] == descriptor(CONFIG) and meta['inputs'] == descriptor(OUT/'input_manifest.json') and
            meta['cohort_sha256'] == context.cohort_sha256 and meta['folds_sha256'] == context.folds_sha256 and
            meta['supervised_parent'] is None, 'Source binding mismatch')
    if inner is None:
        require(meta['selection'] == descriptor(OUT/f'selections/seed_{seed}/fold_{fold}.json'), 'Wrong selection binding')
    else:
        require(meta['selection'] is None and set(train_ids) <= set(context.folds[f'fold_{fold}']['train']), 'Inner fitting leakage')
    require(meta['epochs'] == 60 and meta['updates'] == 60*math.ceil(len(train_ids)/8), 'Wrong budget')
    set_seed(seed)
    model = ScaleModel(seed, kind == 'S')
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    require(state_digest(initial) == meta['initial_state_sha256'], 'Wrong initialization')
    require(sum(p.numel() for p in model.parameters()) == meta['trainable_parameters'] == cfg['parameters'][kind], 'Parameter scope differs')
    generator, h = torch.Generator().manual_seed(seed), hashlib.sha256()
    for _ in range(60):
        h.update(torch.randperm(len(train_ids), generator=generator).numpy().astype('<i8').tobytes())
    require(h.hexdigest() == meta['batch_order_sha256'], 'Wrong batch order')
    model.load_state_dict(saved['state_dict'], strict=True)
    require(model.scales().detach().tolist() == meta['final_scales'], 'Saved scale mismatch')
    if kind == 'control':
        require(torch.equal(model.scales(), torch.ones(3)), 'Control scales not identity')
    for m in MODALITIES:
        prefix = f'projections.{m}.'
        require(any(not torch.equal(initial[k], saved['state_dict'][k]) for k in initial if k.startswith(prefix)), 'Projection never updated')
    history = read_json(folder/'train_history.json')
    require([r['epoch'] for r in history] == list(range(1, 61)), 'Missing training epochs')
    for row in history:
        expected = sum(row['ccml_loss_terms'].values())+row['scale_shrinkage']
        require(math.isclose(expected, row['train_objective'], rel_tol=2e-5, abs_tol=2e-5), 'Loss decomposition mismatch')
        require(all(.5 <= s <= 2. and math.isfinite(s) for s in row['scales']), 'Invalid scale')
        require(math.isfinite(row['preclip_gradient_mean']) and row['preclip_gradient_mean'] > 0, 'Invalid gradient')
    if kind == 'S':
        require(any(r['scale_gradient_mean'] > 0 for r in history), 'Scale never received a gradient')
    rows = read_json(folder/'predictions.json')
    require(rows == predictions(model, data, context, meta, folder/'checkpoint.pt', split_map), 'Predictions not exactly reproduced')
    for row in rows:
        validate_row(row)
        require(row['label'] == context.labels[row['subject_id']] and row['train_ids'] == train_ids, 'Prediction labels or ancestry differ')
        for j, m in enumerate(MODALITIES):
            require(np.allclose(np.array(row['views'][m]['e_raw'])*row['scales'][j], row['views'][m]['e'], atol=2e-6, rtol=2e-5), 'Scaling equation mismatch')
    return rows, meta


def diagnostics(rows):
    test = [r for r in rows if r['split'] == 'test']
    result = {'test_records': len(test), 'independent_subjects': len({r['subject_id'] for r in test}),
              'final_correct': sum(r['prediction'] == r['label'] for r in test),
              'same_model_unit_scale_test_class_changes': sum(r['prediction'] != r['same_trained_model_unit_scale_diagnostic']['prediction'] for r in test),
              'mean_common_evidence': np.mean([r['e_con'] for r in test], axis=0).tolist(),
              'mean_complementary_evidence': np.mean([r['e_cmp'] for r in test], axis=0).tolist(),
              'views': {}}
    for m in MODALITIES:
        correct = lambda r: int(r['views'][m]['p'][1] > .5) == r['label']
        result['views'][m] = {'mean_raw_strength': float(np.mean([sum(r['views'][m]['e_raw'])+2 for r in test])),
                             'mean_scaled_strength': float(np.mean([sum(r['views'][m]['e'])+2 for r in test])),
                             'branch_correct': sum(correct(r) for r in test),
                             'branch_correct_final_wrong': sum(correct(r) and r['prediction'] != r['label'] for r in test),
                             'branch_wrong_final_correct': sum(not correct(r) and r['prediction'] == r['label'] for r in test),
                             'min_source_counts_HC_MDD': [sum(m in r['min_sources'][k] for r in test) for k in [0, 1]]}
    return result


def audit():
    context, cfg, data = load_setup()
    build_inputs(context)  # Recompute stateless pools from their verified source features.
    nested = read_json(OUT/'nested_splits.json')
    require(nested == nested_splits(context), 'Nested partitions changed')
    lock = read_json(OUT/'all_selections_locked.json')
    require(len(lock['selections']) == 15, 'Missing selection locks')
    for d in lock['selections']:
        verify_descriptor(d)
    outer_rows = {k: [] for k in STAGES}
    checks, selections, scale_rows = [], [], []
    replayed, inner_updates, outer_updates = 0, 0, 0
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            candidates = {}
            for strength in LAMBDAS:
                candidates[strength] = []
                for part in nested['folds'][str(fold)]:
                    split_map = {**dict.fromkeys(part['train_ids'], 'train'), **dict.fromkeys(part['val_ids'], 'inner_val')}
                    rows, meta = inspect_run(context, data, cfg, seed, fold, part['inner_fold'], 'S', strength, part['train_ids'], split_map)
                    candidates[strength] += [r for r in rows if r['split'] == 'inner_val']
                    checks.append({'seed': seed, 'outer_fold': fold, 'inner_fold': part['inner_fold'], 'kind': 'S',
                                   'lambda': strength, 'replayed_rows': len(rows), 'updates': meta['updates']})
                    replayed += len(rows)
                    inner_updates += meta['updates']
            choice = select_candidate(candidates, context.folds[f'fold_{fold}']['train'])
            saved_choice = read_json(OUT/f'selections/seed_{seed}/fold_{fold}.json')
            require(all(saved_choice[k] == v for k, v in choice.items()) and
                    not saved_choice['outer_val_used'] and not saved_choice['outer_test_used'], 'Selection mismatch')
            for d in saved_choice['sources']:
                verify_descriptor(d)
            selections.append(saved_choice)
            outer = context.folds[f'fold_{fold}']
            split_map = {s: split for split, ids in outer.items() for s in ids}
            paired = {}
            for kind, strength in [('control', 0.), ('S', choice['selected_lambda'])]:
                rows, meta = inspect_run(context, data, cfg, seed, fold, None, kind, strength, outer['train'], split_map)
                outer_rows[kind] += rows
                paired[kind] = meta
                checks.append({'seed': seed, 'outer_fold': fold, 'inner_fold': None, 'kind': kind,
                               'lambda': strength, 'replayed_rows': len(rows), 'updates': meta['updates']})
                replayed += len(rows)
                outer_updates += meta['updates']
            require(paired['control']['initial_state_sha256'] == paired['S']['initial_state_sha256'] and
                    paired['control']['batch_order_sha256'] == paired['S']['batch_order_sha256'], 'Outer control/S pairing mismatch')
            scale_rows.append({'seed': seed, 'outer_fold': fold, 'lambda': choice['selected_lambda'],
                               **dict(zip(MODALITIES, paired['S']['final_scales']))})
            print(f'S audit replay seed={seed} outer={fold}: 6 inner + 2 outer checkpoints exact.', flush=True)
    require(len(checks) == 120 and inner_updates == 16200 and outer_updates == 7200, 'Total budget mismatch')
    for rows in outer_rows.values():
        require(len(rows) == 555 and Counter((r['seed'], r['subject_id']) for r in rows if r['split'] == 'test') ==
                Counter((s, i) for s in cfg['seeds'] for i in context.ids), 'Outer OOF coverage mismatch')
    summaries = {k: summarize(v) for k, v in outer_rows.items()}
    split_scores = {k: {split: {m: mean_scores(rows, split, None if m == 'final' else m)
                               for m in ['final', *MODALITIES]} for split in ['train', 'val', 'test']}
                    for k, rows in outer_rows.items()}
    test_map = lambda rows: {(r['seed'], r['subject_id']): r for r in rows if r['split'] == 'test'}
    old, new = test_map(outer_rows['control']), test_map(outer_rows['S'])
    changes = [{'seed': s, 'subject_id': sid, 'old_prediction': old[s, sid]['prediction'],
                'new_prediction': new[s, sid]['prediction'], 'label': new[s, sid]['label'],
                'direction': 'wrong_to_correct' if new[s, sid]['prediction'] == new[s, sid]['label'] else 'correct_to_wrong'}
               for s, sid in old if old[s, sid]['prediction'] != new[s, sid]['prediction']]
    for kind, rows in outer_rows.items():
        write_new(OUT/f'outer_predictions_{kind}.json', rows)
    result = {'passed': True, 'configuration': descriptor(CONFIG), 'completed_runs': len(checks),
              'budget': cfg['budget'], 'exact_replayed_records': replayed, 'run_checks': checks,
              'independent_metric_crosschecks': sum(crosscheck_metrics(rows) for rows in outer_rows.values()),
              'selections': selections, 'selection_counts': dict(Counter(str(s['selected_lambda']) for s in selections)),
              'final_scales': scale_rows, 'scale_summary': {m: {'min': min(s[m] for s in scale_rows),
                           'max': max(s[m] for s in scale_rows), 'mean': float(np.mean([s[m] for s in scale_rows]))} for m in MODALITIES},
              'summaries': summaries, 'comparison': gate(summaries['S'], summaries['control']),
              'split_scores': split_scores, 'diagnostics': {k: diagnostics(v) for k, v in outer_rows.items()},
              'classification_changes': changes, 'change_counts': dict(Counter(r['direction'] for r in changes)),
              'protected_files_unchanged': protect_history(), 'new_supervised_parent_weights_loaded': False,
              'independent_validation': False}
    write_new(OUT/'completion_audit.json', result)
    return result


def report(result):
    scores = {k: v['primary_seed_mean_of_fold_means'] for k, v in result['summaries'].items()}
    comp = result['comparison']
    passed = comp['gate']['development_criteria_passed']
    lines = ['# CCML-v2-S 模态证据尺度校准完成报告', '', '日期：2026-09-24。固定扩展阶段已完成；旧 v2 收尾快照保留。', '',
             '## 主结论', '',
             f'S 相对同版本控制的 Macro-F1 变化为 {100*comp["delta"]["macro_f1"]:+.3f} 个百分点，'
             f'{sum(v>0 for v in comp["seed_f1_deltas"].values())}/3 种子严格改善，'
             f'MDD recall 变化为 {100*comp["delta"]["mdd_recall"]:+.3f} 个百分点。'
             +('通过' if passed else '未通过')+'预先固定的开发分类门槛。', '',
             '| 条件 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |', '|---|---:|---:|---:|---:|---:|']
    for k, title in [('control', 'S-control：尺度固定为 1'), ('S', 'S：约束尺度＋训练内选择')]:
        lines.append('| '+title+' | '+' | '.join(f'{scores[k][m]:.6f}' for m in METRICS)+' |')
    lines += ['', '主口径为每种子五折均值，再取三个种子均值。NLL/Brier 越低越好。'
              f'本次 NLL 变化 {comp["delta"]["nll"]:+.6f}，Brier 变化 {comp["delta"]["brier"]:+.6f}。'
              '概率指标与分类门槛分别解释，不因单项改善自动升级模型。', '',
              '| 种子 | 控制 F1 | S F1 | 差值 |', '|---|---:|---:|---:|']
    for seed in ['11', '22', '33']:
        pair = [result['summaries'][k]['seed_reports'][seed]['unweighted_fold_mean']['macro_f1'] for k in ['control', 'S']]
        lines.append(f'| {seed} | {pair[0]:.6f} | {pair[1]:.6f} | {pair[1]-pair[0]:+.6f} |')
    lines += ['', '## 设计与合法验证', '',
              '这是自定义 CCML 扩展：解耦前 e′ᵥ=sᵥeᵥ，sᵥ=exp(log(2)·tanh(aᵥ))，取值 [0.5,2]，初始为 1。'
              '三个标量与投影/证据头联合训练；目标为原完整 CCML 损失＋λ·mean(log(s)²)。'
              '保持 rho=1.25、delta=gamma=1、50 轮 KL 退火；没有新增最终融合监督或动态门控。', '',
              '现有 AS-NP 的监督父表征不能直接拆为合法内层验证。本轮按原计划建立 S-backbone-mean-v1：'
              'EEG 使用 E 修订的冻结 CBraMod 60×200 时间均值，语音使用 AS 完整 29 题 HuBERT 均值，文本使用全文 MacBERT 29 题 CLS 均值。'
              '三路均接新建 Linear→ReLU→无仿射 LayerNorm128→证据头，无旧 MODMA 监督权重。'
              '控制 223,366 参数，S 223,369 参数；所有监督参数均在对应内层训练人群重新拟合。', '',
              '每个外层 train 进行固定三折受试者分层，比较 λ=0.01/0.1；按三折 inner-val 融合 NLL 均值最小选择，'
              '1e-12 内平票选 0.1。原外层 val 和 test 不进入选择；15 份选择全部锁定后才运行外层模型。'
              '外层 S 和控制从共同初始状态独立拟合，并共享批序。', '',
              '**本轮的直接对照是 S-control，不是 AS-NP。** 汇聚结构、输入维度与监督初始化均已变化，'
              '新结果不能解释为“在 AS-NP 上仅加三个尺度”的效果。旧 NP F1=0.680895、AS-NP F1=0.658326 只作历史参照，不重排为同因素实验。', '',
              '## 选择结果与模态尺度', '',
              'λ 选择次数：'+', '.join(f'{k} → {v}/15' for k, v in sorted(result['selection_counts'].items()))+'。未追加搜索。', '',
              '| 种子 | 外折 | λ | EEG s | 语音 s | 文本 s |', '|---|---:|---:|---:|---:|---:|']
    for r in result['final_scales']:
        lines.append(f'| {r["seed"]} | {r["outer_fold"]} | {r["lambda"]} | {r["eeg"]:.6f} | {r["speech"]:.6f} | {r["text"]:.6f} |')
    lines += ['', '尺度取值在限制内且受向 1 收缩约束。所有模态的共同缩放不能改变当前融合 argmax；'
              '模态间相对缩放会改变 min、公共/互补证据和 u。联合学习的尺度也可能被证据头补偿，不能解释成模态可靠性的因果排名。', '',
              '## 泛化与融合诊断', '',
              '| 条件 | Train F1 | Val F1 | Test F1 | Train−Test |', '|---|---:|---:|---:|---:|']
    for k in ['control', 'S']:
        v = [result['split_scores'][k][p]['final']['macro_f1'] for p in ['train', 'val', 'test']]
        lines.append(f'| {k} | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} | {v[0]-v[2]:.6f} |')
    lines += ['', '| 输出 | 控制 Test F1 | S Test F1 | 控制：分支对但融合错 | S：分支对但融合错 |', '|---|---:|---:|---:|---:|']
    for m in MODALITIES:
        v = [result['split_scores'][k]['test'][m]['macro_f1'] for k in ['control', 'S']]
        counts = [result['diagnostics'][k]['views'][m]['branch_correct_final_wrong'] for k in ['control', 'S']]
        lines.append(f'| {m} | {v[0]:.6f} | {v[1]:.6f} | {counts[0]} | {counts[1]} |')
    changes = result['change_counts']
    lines += ['', f'相对控制，111 条重复种子测试记录中有 {len(result["classification_changes"])} 条分类改变：'
              f'错变对 {changes.get("wrong_to_correct",0)}，对变错 {changes.get("correct_to_wrong",0)}。'
              f'将同一个训练后 S 模型临时恢复单位尺度时，另有 {result["diagnostics"]["S"]["same_model_unit_scale_test_class_changes"]} 条分类与其正常缩放推理不同；'
              '这一项仅为固定模型推理诊断，不替代从头独立训练的控制。', '',
              '逐种子合并 OOF、ECE5、正确/错误 u、风险覆盖、各路证据强度、min 来源、公共和互补量在 completion_audit.json 中完整保留。'
              'u 仍表示证据不足程度，不声称是已校准的错误概率。', '',
              '## 预算与验收', '',
              '| 工作 | 训练组数 | 更新数 |', '|---|---:|---:|',
              '| 两个强度 × 三内折 × 三种子 × 五外折 | 90 | 16,200 |',
              '| 外层完整 CCML 控制 | 15 | 3,600 |', '| 外层 S 候选 | 15 | 3,600 |', '| 合计 | 120 | 23,400 |', '',
              '全部采用 60 轮、batch 8、Adamax、投影 lr=1e-4、头/尺度 lr=1e-3、weight_decay=0、梯度裁剪 1、CPU 单线程确定性模式。'
              '固定末轮与 0.5 阈值，不选择种子/epoch/阈值。本阶段预算单列，不改写旧 v2 的 210 组历史计数。', '',
              f'验收通过：{read_json(OUT/"test_results.json")["tests_run"]} 项测试；120 个检查点、'
              f'{result["exact_replayed_records"]:,} 条预测精确重载；15 份内层选择复算；'
              f'{result["independent_metric_crosschecks"]} 个外层测试折的指标独立交叉核验；'
              f'{result["protected_files_unchanged"]:,} 项历史文件哈希不变。', '',
              '## 收尾决定与边界', '',
              ('通过预定分类开发门槛，可保留为新均值汇聚版本的开发候选；仍需保留概率指标权衡。' if passed else
               '未通过预定分类开发门槛，不升级为默认模型，不追加尺度/收缩强度搜索。'),
              '旧 AS-NP、NP 和 N 的收尾角色保持。本次合法内层拟合与选择未使用外层测试标签，'
              '但研究方案仍来自此前反复查看的 37 人开发过程；不能追溯性消除已有选择偏差，也没有获得新的独立验证人群。'
              'EEG 幅值与坐标来源的已知限制仍然保留。', '',
              f'- [固定计划]({os.path.relpath(PLAN, OUT).replace(chr(92), "/")})',
              '- [固定配置](configuration.json)', '- [无 MODMA 监督父权重的输入来源](input_manifest.json)',
              '- [训练内三折名单](nested_splits.json)', '- [全部内层选择锁](all_selections_locked.json)',
              '- [完整验收与诊断](completion_audit.json)', '- [控制预测](outer_predictions_control.json)',
              '- [S 预测](outer_predictions_S.json)', '- [完成状态](status.json)', '']
    path = OUT/'CCML-v2-S实验完成报告.md'
    text = '\n'.join(lines)
    if path.exists():
        require(path.read_text(encoding='utf-8') == text, 'Report changed')
    else:
        path.write_text(text, encoding='utf-8')
    write_new(OUT/'status.json', {'status': 'complete', 'date': '2026-09-24', 'runs': 120, 'updates': 23400,
              'configuration': descriptor(CONFIG), 'plan': descriptor(PLAN), 'report': descriptor(path),
              'audit': descriptor(OUT/'completion_audit.json'), 'primary_comparison': 'S versus S-control on S-backbone-mean-v1',
              'development_criteria_passed': passed, 'independent_validation': False,
              'existing_v2_closeout_preserved': True, 'additional_search_started': False,
              'execution_blockers': [], 'next_action': 'No additional training registered'})
    print(scores, flush=True)
    print(comp, flush=True)
    print('S complete: 120 runs / 23400 updates; exact reload and nested selection audits passed.', flush=True)


if __name__ == '__main__':
    report(audit())
