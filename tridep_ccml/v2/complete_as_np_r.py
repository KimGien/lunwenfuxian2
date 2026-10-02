"""Audit the fixed AS-NP-R regularization treatment against paired AS-NP."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

from .as_np_r import (OUT, CONFIG, PLAN, EXPERIMENT, PARENT, baseline,
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
    require(cfg == configuration(context), 'Registered AS-NP-R configuration changed')
    pre = read_json(OUT/'preflight.json')
    require(pre['passed'] and pre['configuration'] == descriptor(CONFIG) and
            len(pre['checks']) == 15 and pre['reconstructed_control_records'] == 555, 'Preflight incomplete')
    tests = read_json(OUT/'test_results.json')
    require(tests['passed'] and tests['failures'] == 0 and tests['errors'] == 0, 'Tests incomplete')
    for item in tests['sources'].values():
        verify_descriptor(item)
    protected = protect_history()
    rows = {'candidate': [], 'baseline': []}
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
                    matching[0]['initial_state_sha256'] == initial_hash, 'Preflight source binding mismatch')
            base_meta = {}
            for name in ['baseline']:
                folder = PARENT/f'runs/seed_{seed}/fold_{fold}'
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                base_meta[name] = read_json(folder/'checkpoint.json')['metadata']
                rows[name].extend(read_json(folder/'predictions.json'))
                require(base_meta[name]['initial_state_sha256'] == initial_hash and base_meta[name]['source'] == data['source'] and base_meta[name]['weight_decay'] == 0., 'Baseline factor mismatch')
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
                    meta['projection_lr'] == 1e-4 and meta['head_lr'] == 1e-3 and meta['weight_decay'] == cfg['weight_decay'] == 1e-4, 'Training factors changed')
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
            baseline_state = torch.load(PARENT/f'runs/seed_{seed}/fold_{fold}/checkpoint.pt',
                                        map_location='cpu', weights_only=True)['state_dict']
            require(all(not m.training for m in model.modules()), 'Stochastic training mode enabled')
            require(not any(isinstance(m, (torch.nn.Conv1d, torch.nn.LSTM)) for m in model.modules()), 'Upstream unfreezing introduced')
            drift = {'seed': seed, 'fold': fold, 'projections': {}}
            for modality in MODALITIES:
                require(any(h['projection_grad_norm_mean'][modality] > 0 for h in history), 'Projection received no gradient')
                before = torch.cat([data['initial'][f'{modality}.0.{k}'].flatten() for k in ['weight', 'bias']])
                after = torch.cat([model.projections[modality][0].state_dict()[k].flatten() for k in ['weight', 'bias']])
                paired = torch.cat([baseline_state[f'projections.{modality}.0.{k}'].flatten() for k in ['weight', 'bias']])
                require(not torch.equal(before, after), 'Projection did not update')
                drift['projections'][modality] = {'l2_change': float(torch.linalg.vector_norm(after-before)),
                    'relative_l2_change': float(torch.linalg.vector_norm(after-before)/torch.linalg.vector_norm(before)),
                    'initial_l2_norm': float(torch.linalg.vector_norm(before)), 'final_l2_norm': float(torch.linalg.vector_norm(after)),
                    'baseline_final_l2_norm': float(torch.linalg.vector_norm(paired)),
                    'l2_distance_to_baseline': float(torch.linalg.vector_norm(after-paired)),
                    'max_abs_distance_to_baseline': float((after-paired).abs().max())}
            head_names = [k for k in saved['state_dict'] if k.startswith('heads.')]
            head_now = torch.cat([saved['state_dict'][k].flatten() for k in head_names])
            head_base = torch.cat([baseline_state[k].flatten() for k in head_names])
            drift['heads'] = {'final_l2_norm': float(torch.linalg.vector_norm(head_now)),
                              'baseline_final_l2_norm': float(torch.linalg.vector_norm(head_base)),
                              'l2_distance_to_baseline': float(torch.linalg.vector_norm(head_now-head_base))}
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
    require(summaries['baseline'] == parent_audit['summaries']['candidate'], 'AS-NP summary drift')
    result = {'date': '2026-09-23', 'passed': True, 'formal_runs': 15, 'updates': 3600,
              'subject_fold_rows': 555, 'independent_subjects': 37, 'summaries': summaries,
              'references': parent_audit['references'], 'primary_comparison': 'candidate_vs_AS_NP',
              'comparisons': {name: gate(summaries['candidate'], summaries[name]) for name in ['baseline']},
              'diagnostics': {name: diagnostics(rows['candidate'], rows[name]) for name in ['baseline']},
              'speech_fusion_counts': {name: fusion_counts([r for r in values if r['split'] == 'test']) for name, values in rows.items()},
              'projection_changes': changes, 'baseline_projection_changes': parent_audit['projection_changes'], 'run_checks': checks,
              'sklearn_fold_crosschecks': crosscheck_metrics(rows['candidate']),
              'configuration': descriptor(CONFIG), 'preflight': descriptor(OUT/'preflight.json'),
              'tests': descriptor(OUT/'test_results.json'), 'reporter': descriptor(__file__),
              'protected_files_unchanged': protect_history(),
              'scope': 'Only Adamax weight_decay 0 to 1e-4; full CCML data loss unchanged',
              'baseline_diagnostics': diagnostics(rows['baseline'], rows['baseline'])}
    require(result['protected_files_unchanged'] == protected, 'Historical artifacts changed during audit')
    write_new(OUT/'completion_audit.json', result)
    return result


def report(result):
    comp = result['comparisons']['baseline']
    passed = comp['gate']['development_criteria_passed']
    worse = comp['gate']['nll_worsened'] or comp['gate']['brier_worsened']
    score = result['summaries']['candidate']['primary_seed_mean_of_fold_means']
    delta = comp['delta']
    lines = ['# CCML-v2-AS-NP-R 固定正则对照完成报告', '',
             '日期：2026-09-23。**15 组训练、3,600 次更新已完成，来源与数值验收通过。**', '',
             '## 主比较', '',
             '直接对照为 AS-NP。唯一训练因素是 Adamax weight_decay 从 0 改为 1e-4，作用于所有三路投影及证据头参数（含 bias）。模型、完整 CCML 数据损失、初始参数、来源、批序和预算配对。主指标为每种子五折均值，再取三种子均值。', '',
             '| 实验 | Macro-F1 | MDD recall | BA | NLL | Brier |', '|---|---:|---:|---:|---:|---:|']
    for label, summary in [('AS-NP：直接对照', result['summaries']['baseline']),
                           ('AS-NP-R：固定正则', result['summaries']['candidate']),
                           ('旧 NP：历史参照', result['references']['NP'])]:
        m = summary['primary_seed_mean_of_fold_means']
        lines.append('| '+label+' | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    positive = sum(v > 0 for v in comp['seed_f1_deltas'].values())
    lines += ['', f'相对 AS-NP：F1 {100*delta["macro_f1"]:+.2f} 个百分点，召回 {100*delta["mdd_recall"]:+.2f} 个百分点；NLL {delta["nll"]:+.8f}，Brier {delta["brier"]:+.8f}。三个种子中 {positive} 个 F1 严格改善。', '',
              '| 种子 | AS-NP F1 | AS-NP-R F1 | 差值 |', '|---|---:|---:|---:|']
    for seed in ['11', '22', '33']:
        a, b = [result['summaries'][name]['seed_reports'][seed]['unweighted_fold_mean']['macro_f1'] for name in ['baseline', 'candidate']]
        lines.append(f'| {seed} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines += ['', '固定开发门槛：相对 AS-NP，F1 至少 +0.01、至少 2/3 种子改善、召回下降不超过 0.05；另列 NLL/Brier 权衡。判定：**'+('通过' if passed else '未通过')+'**。这不是统计显著性或独立确认。', '',
              '## 分支、泛化与融合诊断', '',
              '| 模态 | AS-NP train | AS-NP test | AS-NP-R train | AS-NP-R val | AS-NP-R test |', '|---|---:|---:|---:|---:|---:|']
    old, new = result['baseline_diagnostics'], result['diagnostics']['baseline']
    for modality in ['final', 'eeg', 'speech', 'text']:
        values = [old['train_val_test'][s][modality]['macro_f1'] for s in ['train', 'test']]
        values += [new['train_val_test'][s][modality]['macro_f1'] for s in ['train', 'val', 'test']]
        lines.append('| '+modality+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    gap = lambda d: d['train_val_test']['train']['final']['macro_f1']-d['train_val_test']['test']['final']['macro_f1']
    lines += ['', f'融合 train−test F1 差距：AS-NP {gap(old):.6f}，AS-NP-R {gap(new):.6f}；这是描述性诊断，不独立作为选模依据。', '',
              '| 重复种子测试记录 | AS-NP | AS-NP-R |', '|---|---:|---:|']
    for key, label in [('speech_correct', '语音正确'), ('final_correct', '融合正确'),
                       ('speech_correct_final_wrong', '语音正确、融合错误'), ('speech_wrong_final_correct', '语音错误、融合正确')]:
        lines.append('| '+label+' | '+' | '.join(str(result['speech_fusion_counts'][name][key]) for name in ['baseline', 'candidate'])+' |')
    changes = new['changes_by_direction']
    lines += ['', f'111 条重复种子测试记录中 {new["changed_count"]} 条分类改变：错变对 {changes.get("wrong_to_correct",0)}，对变错 {changes.get("correct_to_wrong",0)}。独立受试者仍只有 37 位。', '',
              '| 模态 | AS-NP 平均 S | AS-NP-R 平均 S | 平均投影参数范数：AS-NP | AS-NP-R |', '|---|---:|---:|---:|---:|']
    for modality in MODALITIES:
        norm = lambda key: sum(c['projections'][modality][key] for c in result['projection_changes'])/15
        lines.append(f'| {modality} | {old["mean_strength"][modality]:.6f} | {new["mean_strength"][modality]:.6f} | {norm("baseline_final_l2_norm"):.6f} | {norm("final_l2_norm"):.6f} |')
    lines += ['', '参数范数下降或证据变弱本身不等于泛化改善。逐种子 OOF、ECE5、不确定性与风险覆盖率、min 来源、公共/互补贡献、各折参数相对基线位移保存在 completion_audit.json。', '',
              '## 配对、训练与验收', '',
              '- 使用同折 AS 父投影和新头，从与 AS-NP 相同的初始状态独立训练；没有继承 AS-NP 训练后权重或优化器。语音覆盖保持 1,073 题。',
              '- 三投影 475,520 参数、新头 774，共 476,294。无仿射 LayerNorm eps=1e-5；EEG/text Dropout 保持 eval，语音无 Dropout；骨干和其他上游网络冻结。',
              '- 固定种子 11/22/33、五折、60 轮、batch 8、投影 lr=1e-4、头 lr=1e-3、CPU 单线程；最后一轮、阈值 0.5。没有额外轮数或参数搜索。',
              '- 完整 CCML 数据损失保持 rho=1.25、delta=gamma=1、KL 退火 50 轮。先裁剪 CCML 梯度至范数 1，再由 Adamax 加耦合权重衰减；报告的 train_loss 及分项不包含优化器正则项。',
              '- 训练循环与 AS-NP 的对比检查只允许已登记的正则及产物路径变化；优化器测试覆盖全部权重与 bias，并核对零数据梯度下正则实际更新。',
              '- 训练前 15 组来源、初始参数及 555 条 AS-NP 预测精确重放通过；完成后 15 个新检查点重载、555 条新预测精确重现，三投影均收到梯度并更新，15 个测试折指标交叉核验通过。',
              f'- {read_json(OUT/"test_results.json")["tests_run"]} 项测试通过；{result["protected_files_unchanged"]} 项历史文件哈希不变。', '',
              '## 阶段结论', '']
    if passed:
        lines.append('AS-NP-R 通过预定开发门槛。'+('NLL/Brier 存在恶化，保留权衡，不自动替换。' if worse else 'NLL/Brier 未恶化，可保留为该固定版本的开发候选。'))
    else:
        lines.append('AS-NP-R 未达到预定开发门槛，保留 AS-NP 作为本次对照，不自动把正则组合升级为默认，也不追加更强正则搜索。')
    lines += ['本轮只检验在 N+P 组合上新增固定正则的条件效果，不能替代独立 AS-R/AS-P 对照，也不宣称完整因素矩阵已经完成。',
              '全部结果仍为多次查看的 37 人上的开发比较，历史父表征含 CE 监督血缘；新种子或新阶段不能代替独立人群验证。',
              '本轮工作到固定实验及报告为止，后续实验另行确定。', '']
    path = OUT/'CCML-v2-AS-NP-R实验完成报告.md'
    content = '\n'.join(lines)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content, 'Completed report changed')
    else:
        path.write_text(content, encoding='utf-8')
    write_new(OUT/'status.json', {'status': 'complete', 'date': '2026-09-23', 'runs': 15, 'updates': 3600,
              'configuration': descriptor(CONFIG), 'plan': descriptor(PLAN), 'report': descriptor(path),
              'audit': descriptor(OUT/'completion_audit.json'), 'primary_comparison': 'AS-NP-R versus AS-NP',
              'development_criteria_passed': passed, 'probability_quality_worsened': worse,
              'execution_blockers': [], 'next_action': 'No additional training registered'})
    print(score)
    print(comp)
    print(f'AS-NP-R complete: 15 runs, 3600 updates, {result["protected_files_unchanged"]} historical files unchanged.')


if __name__ == '__main__':
    report(audit())

