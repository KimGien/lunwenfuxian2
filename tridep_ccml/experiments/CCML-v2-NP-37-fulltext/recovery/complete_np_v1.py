"""Audit the fixed N+P combination and the complete two-factor comparison."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

from .np_combo import (OUT, CONFIG, EXPERIMENT, PLAN, PARENT, CONTROL, NR, VARIANTS,
                       MODALITIES, NormalizedProjectionHeads, projection, configuration,
                       load_data, predictions, protect_history)
from .common import (WORK, RecordContext, descriptor, verify_descriptor, read_json,
                     require, write_new, set_seed)
from .complete_nr import diagnostics
from .complete_p import gate
from .complete_eeg import crosscheck_metrics, METRICS
from .rebuild_eeg import validate_row
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.metrics import summarize
import torch


def audit():
    context = RecordContext()
    cfg = read_json(CONFIG)
    require(cfg == configuration(context), 'N+P registered configuration changed')
    preflight = read_json(OUT/'preflight.json')
    require(preflight['passed'] and preflight['cache_bundles'] == 15 and
            preflight['configuration'] == descriptor(CONFIG), 'Preflight missing or stale')
    for check in preflight['checks']:
        verify_descriptor(check['manifest'])
    tests = read_json(OUT/'test_results.json')
    require(tests['passed'] and tests['failures'] == 0 and tests['errors'] == 0, 'Tests not passed')
    for item in tests['sources'].values():
        verify_descriptor(item)
    protected = protect_history()
    all_rows = {name: [] for name in ['C', 'N', 'P', 'NP']}
    run_checks, drift = [], []
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed)
            data = load_data(context, seed, fold)
            model = NormalizedProjectionHeads(data['initial'], seed)
            initial_hash = state_digest(model.state_dict())
            require(sum(p.numel() for p in model.parameters()) == 836742, 'Wrong trainable scope')
            require(sum(p.numel() for p in model.normalizer.parameters()) == 0, 'Learned normalization introduced')
            folders = {'C': PARENT/'runs'/CONTROL/f'seed_{seed}/fold_{fold}',
                       'N': NR/'runs'/VARIANTS['N']['experiment']/f'seed_{seed}/fold_{fold}',
                       'P': projection.OUT/f'runs/seed_{seed}/fold_{fold}'}
            base_meta = {}
            for name, folder in folders.items():
                for item in read_json(folder/'complete.json').values():
                    verify_descriptor(item)
                base_meta[name] = read_json(folder/'checkpoint.json')['metadata']
                rows = read_json(folder/'predictions.json')
                require(len(rows) == 37 and [r['subject_id'] for r in rows] == context.ids, 'Comparator coverage mismatch')
                all_rows[name].extend(rows)
            require(initial_hash == base_meta['P']['initial_state_sha256'], 'P/NP initial parameters differ')
            require(state_digest(data['initial']) == base_meta['P']['initial_projection_sha256'], 'Projection ancestry mismatch')
            require(base_meta['P']['source'] == data['source'], 'P/NP cache differs')
            folder = OUT/f'runs/seed_{seed}/fold_{fold}'
            for item in read_json(folder/'complete.json').values():
                verify_descriptor(item)
            cp = folder/'checkpoint.pt'
            side = read_json(cp.with_suffix('.json'))
            saved = torch.load(cp, map_location='cpu', weights_only=True)
            meta = saved['metadata']
            require(meta == side['metadata'], 'Checkpoint metadata mismatch')
            require(meta['experiment'] == EXPERIMENT and meta['seed'] == seed and meta['fold'] == fold, 'Run identity mismatch')
            require(meta['source'] == data['source'] and meta['configuration'] == descriptor(CONFIG), 'Source/config mismatch')
            require(meta['train_ids'] == context.folds[f'fold_{fold}']['train'], 'Training subjects mismatch')
            require(meta['cohort_sha256'] == context.cohort_sha256 and meta['folds_sha256'] == context.folds_sha256, 'Cohort mismatch')
            require(meta['initial_state_sha256'] == initial_hash and
                    meta['initial_projection_sha256'] == state_digest(data['initial']), 'Initial parameters mismatch')
            require(meta['objective'] == 'full_CCML_only' and meta['epochs'] == 60 and meta['updates'] == 240, 'Objective/budget mismatch')
            require(meta['projection_parameters'] == 835968 and meta['trainable_parameters'] == 836742, 'Parameter count mismatch')
            require(meta['normalization'] == cfg['normalization'] and meta['eps'] == 1e-5, 'Normalization changed')
            require(meta['projection_lr'] == 1e-4 and meta['head_lr'] == 1e-3 and
                    meta['weight_decay'] == 0 and meta['dropout'] == 'eval' and meta['selection'] == 'last_epoch', 'Training factors changed')
            initial_head = state_digest(make_initial_state(seed))
            require(meta['head_initialization_sha256'] == initial_head == base_meta['C']['head_initialization_sha256']
                    == base_meta['N']['initialization_sha256'] == base_meta['P']['head_initialization_sha256'], 'Head pairing mismatch')
            generator = torch.Generator().manual_seed(seed)
            batch = hashlib.sha256()
            for _ in range(60):
                batch.update(torch.randperm(len(meta['train_ids']), generator=generator).numpy().astype('<i8').tobytes())
            require(all(m['batch_order_sha256'] == batch.hexdigest() for m in [meta, *base_meta.values()]), 'Batch pairing mismatch')
            history = read_json(folder/'train_history.json')
            require([r['epoch'] for r in history] == list(range(1, 61)), 'Missing epochs')
            for row in history:
                require(math.isfinite(row['train_loss']) and math.isclose(row['train_loss'], sum(row['loss_terms'].values()),
                        rel_tol=1e-5, abs_tol=1e-5), 'Loss diagnostics mismatch')
                require(all(math.isfinite(v) and v >= 0 for group in
                        ['projection_grad_norm_mean', 'head_grad_norm_mean'] for v in row[group].values()), 'Invalid gradients')
            model.load_state_dict(saved['state_dict'], strict=True)
            require(all(not m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)), 'Dropout enabled')
            require(not any(isinstance(m, (torch.nn.Conv1d, torch.nn.LSTM)) for m in model.modules()), 'Upstream training introduced')
            changes = {'seed': seed, 'fold': fold, 'projections': {}}
            for m in MODALITIES:
                require(any(r['projection_grad_norm_mean'][m] > 0 for r in history), f'No gradient: {m}')
                before = torch.cat([data['initial'][f'{m}.0.{k}'].flatten() for k in ['weight', 'bias']])
                after = torch.cat([model.projections[m][0].state_dict()[k].flatten() for k in ['weight', 'bias']])
                require(not torch.equal(before, after), f'Projection did not update: {m}')
                changes['projections'][m] = {'l2_change': float(torch.linalg.vector_norm(after-before)),
                    'relative_l2_change': float(torch.linalg.vector_norm(after-before)/torch.linalg.vector_norm(before))}
            drift.append(changes)
            rows = read_json(folder/'predictions.json')
            require(rows == predictions(model, data, context, seed, fold, cp), 'Predictions do not exactly reproduce')
            require(len(rows) == 37 and [r['subject_id'] for r in rows] == context.ids, 'Coverage mismatch')
            for row in rows:
                validate_row(row)
            all_rows['NP'].extend(rows)
            run_checks.append({'seed': seed, 'fold': fold, 'checkpoint': descriptor(cp),
                               'exact_reproduced_rows': 37, 'paired_initialization_and_batches': True,
                               'all_projections_updated': True, 'updates': 240})
    for name, rows in all_rows.items():
        require(len(rows) == 555 and Counter((r['seed'], r['subject_id']) for r in rows if r['split'] == 'test') ==
                Counter((s, i) for s in cfg['seeds'] for i in context.ids), f'Incomplete OOF: {name}')
    summaries = {name: summarize(rows) for name, rows in all_rows.items()}
    old_nr = read_json(NR/'completion_audit.json')
    old_p = read_json(projection.OUT/'completion_audit.json')
    for name, expected in [('C', old_nr['baseline']), ('N', old_nr['variants']['N']['summary']), ('P', old_p['summary'])]:
        require(summaries[name] == expected, f'Historical summary changed: {name}')
    metric = {name: summary['primary_seed_mean_of_fold_means'] for name, summary in summaries.items()}
    interaction = {k: metric['NP'][k]-metric['N'][k]-metric['P'][k]+metric['C'][k] for k in METRICS}
    seed_interaction = {s: sum(sign*summaries[name]['seed_reports'][s]['unweighted_fold_mean']['macro_f1']
                        for name, sign in [('NP', 1), ('N', -1), ('P', -1), ('C', 1)]) for s in ['11', '22', '33']}
    result = {'date': '2026-09-22', 'passed': True, 'formal_runs': 15, 'updates': 3600,
              'subject_fold_rows': 555, 'independent_subjects': 37, 'summaries': summaries,
              'comparisons': {name: gate(summaries['NP'], summaries[name]) for name in ['N', 'P', 'C']},
              'primary_comparator': 'N', 'interaction_NP_minus_N_minus_P_plus_C': interaction,
              'seed_f1_interaction': seed_interaction,
              'diagnostics': {name: diagnostics(all_rows['NP'], all_rows[name]) for name in ['N', 'P', 'C']},
              'projection_changes': drift, 'run_checks': run_checks,
              'sklearn_fold_crosschecks': crosscheck_metrics(all_rows['NP']),
              'preflight': descriptor(OUT/'preflight.json'), 'tests': descriptor(OUT/'test_results.json'),
              'protected_files_unchanged': protect_history(), 'configuration': descriptor(CONFIG),
              'reporter': descriptor(__file__),
              'scope': 'fixed combination on previously viewed developmental subjects; interaction is descriptive'}
    require(result['protected_files_unchanged'] == protected, 'History changed during audit')
    write_new(OUT/'completion_audit.json', result)
    return result


def report(result):
    summaries = result['summaries']
    comparison = result['comparisons']['N']
    passed = comparison['gate']['development_criteria_passed']
    probability_worsened = comparison['gate']['nll_worsened'] or comparison['gate']['brier_worsened']
    promoted = passed and not probability_worsened
    lines = ['# CCML-v2-NP 固定组合实验完成报告', '',
             '日期：2026-09-22。**N+P 已完成三个种子×五折的 15 组训练，共 3,600 次更新，验收通过。**', '',
             '## 四组配对结果', '',
             'N+P 从同折初始父投影与新证据头独立训练，在投影输出后加入无仿射 LayerNorm。主比较为 NP−N；同时报告 P 和同版本 EEG 修订控制。以下均为每种子五折均值，再平均三种子。', '',
             '| 实验 | Macro-F1 | MDD recall | BA | NLL | Brier |',
             '|---|---:|---:|---:|---:|---:|']
    for name, label in [('C', 'E-control：同版本控制'), ('N', 'N：规范化'), ('P', 'P：投影训练'), ('NP', 'NP：规范化＋投影训练')]:
        m = summaries[name]['primary_seed_mean_of_fold_means']
        lines.append('| '+label+' | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    lines += ['', '| 对照 | F1 变化（百分点） | 召回变化（百分点） | F1 提高种子数 | 工程门槛 | NLL 变化 | Brier 变化 |',
              '|---|---:|---:|---:|---|---:|---:|']
    for name in ['N', 'P', 'C']:
        comp = result['comparisons'][name]
        d = comp['delta']
        decision = '通过' if comp['gate']['development_criteria_passed'] else '未通过'
        positives = sum(v > 0 for v in comp['seed_f1_deltas'].values())
        lines.append(f'| NP−{name} | {d["macro_f1"]*100:+.2f} | {d["mdd_recall"]*100:+.2f} | {positives}/3 | {decision} | {d["nll"]:+.6f} | {d["brier"]:+.6f} |')
    lines += ['', '主门槛在训练前固定为相对 N：F1 至少 +0.01、至少 2/3 种子严格改善、MDD recall 下降不超过 0.05；NLL/Brier 越低越好，权衡单列。该门槛用于开发决策，不是显著性检验。', '',
              '| 种子 | E-control F1 | N F1 | P F1 | NP F1 | NP−N |',
              '|---|---:|---:|---:|---:|---:|']
    for s in ['11', '22', '33']:
        values = [summaries[name]['seed_reports'][s]['unweighted_fold_mean']['macro_f1'] for name in ['C', 'N', 'P', 'NP']]
        lines.append('| '+s+' | '+' | '.join(f'{v:.6f}' for v in values)+f' | {values[3]-values[1]:+.6f} |')
    interaction = result['interaction_NP_minus_N_minus_P_plus_C']
    lines += ['', '## 交互效应', '',
              '描述性交互量定义为 `NP−N−P+E-control`，即加规范化后投影适配收益与未规范化时投影适配收益的差。', '',
              '| Macro-F1 | MDD recall | BA | NLL | Brier |', '|---:|---:|---:|---:|---:|',
              '| '+' | '.join(f'{interaction[k]:+.6f}' for k in METRICS)+' |', '',
              '逐种子 F1 交互量：'+'；'.join(f'{s}: {v:+.6f}' for s, v in result['seed_f1_interaction'].items())+'。',
              '交互量只描述这些固定算法和划分下的结果，不提供统计显著性或机制因果证明，不能将组合收益全部归因于某一个因素。', '',
              '## 来源、预算与验收', '',
              '- 复用 P 已验收的 15 组投影前缓存（EEG/语音/文本维度 1920/3584/1024），初始投影与 P 逐项同源；新头与 C/N/P 共同初始化，批序逐折一致，未从 N/P 已训练权重续训。',
              '- 三路投影 835,968 个参数及头 774 个参数，共 836,742。无仿射 LayerNorm，eps=1e-5；CNN/LSTM 和骨干冻结在缓存上游，Dropout eval。',
              '- 投影 lr=1e-4、头 lr=1e-3，Adamax、weight_decay=0、梯度上限 1；60 轮、batch 8、最后一轮、阈值 0.5。完整 CCML，rho=1.25、delta=gamma=1、KL 退火 50 轮。',
              '- 训练前复算 555 条控制与 555 条 N 记录，固定容差内概率一致、所有分类一致；555 条 P 记录精确复现。未调整容差、epoch、阈值或超参数。',
              '- 15 个新检查点用 weights_only=True 重载，555 条完整证据预测逐条精确复现；所有投影均收到梯度并实际更新，损失分项、参数位移、名单及覆盖验收通过；15 个测试折指标由 scikit-learn 交叉核验。',
              f'- {read_json(OUT/"test_results.json")["tests_run"]} 项测试通过，{result["protected_files_unchanged"]} 项历史文件哈希不变。', '',
              '## 泛化与证据诊断', '',
              '| 模态 | train F1 | val F1 | test F1 | test 平均 S |', '|---|---:|---:|---:|---:|']
    d = result['diagnostics']['N']
    for m in ['final', 'eeg', 'speech', 'text']:
        values = [d['train_val_test'][split][m]['macro_f1'] for split in ['train', 'val', 'test']]
        strength = d['mean_strength'].get(m)
        lines.append('| '+m+' | '+' | '.join(f'{v:.6f}' for v in values)+' | '+(f'{strength:.4f}' if strength is not None else '—')+' |')
    for name in ['N', 'P', 'C']:
        diag = result['diagnostics'][name]
        changes = diag['changes_by_direction']
        lines.append(f'\n相对 {name}，111 条重复种子测试记录中有 {diag["changed_count"]} 条分类改变：错变对 {changes.get("wrong_to_correct",0)}、对变错 {changes.get("correct_to_wrong",0)}。')
    lines += ['', '逐种子合并 OOF、ECE5、正确/错误不确定性、风险覆盖率、逐类 min 来源、公共/互补贡献及各路权重位移详见 `completion_audit.json`。证据强度不等于可靠性。', '',
              '## 阶段决策与后续', '']
    if promoted:
        lines.append('NP 通过相对 N 的预定工程门槛，NLL/Brier 也未恶化，可保留为后续开发候选；不称为独立验证的最佳模型。')
    elif passed:
        lines.append('NP 的分类指标通过相对 N 的工程门槛，但概率质量存在恶化，保留完整权衡，不自动替换 N。')
    else:
        lines.append('NP 未通过相对 N 的预定工程门槛，保留 N 为当前开发参照。本轮固定组合已完整收尾，不追加网格搜索、训练轮数或全网络解冻。')
    lines += ['后续优先独立登记语音来源修订：从冻结 HuBERT 片段向量构建简单顺序汇聚及小型可训练投影，再建立同表征版本的 CCML 控制；输入修订与本轮组合结果分开解释。该语音候选本轮尚未训练。',
              '全部结果仍是已查看 37 人上的开发比较，三个种子不是 111 位独立受试者。EEG 检查点尺度/坐标模板及语音随机冻结 CNN-BiGRU、单位置注意力、短音频零向量限制继续存在。历史父模型有 CE 血缘，所有新增训练只使用 CCML。', '',
              '本轮方案为 `output/plans/CCML_v2_NP_固定组合实验计划.md`。旧 N/R/P 的状态是历史快照，当前完成状态以本目录 `status.json` 为准。', '']
    path = OUT/'CCML-v2-NP实验完成报告.md'
    content = '\n'.join(lines)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content, 'Completed report changed')
    else:
        path.write_text(content, encoding='utf-8')
    write_new(OUT/'status.json', {'status': 'complete', 'date': '2026-09-22', 'NP_runs': 15,
              'updates': 3600, 'configuration': descriptor(CONFIG), 'plan': descriptor(PLAN),
              'report': descriptor(path), 'audit': descriptor(OUT/'completion_audit.json'),
              'primary_comparator': 'N', 'development_criteria_passed_vs_N': passed,
              'probability_quality_worsened_vs_N': probability_worsened,
              'next_development_reference': 'NP' if promoted else 'N', 'execution_blockers': [],
              'speech_revision_started': False,
              'next_action': 'Separately register frozen HuBERT pooling/projection speech source revision and matched CCML controls'})
    print(summaries['NP']['primary_seed_mean_of_fold_means'])
    print(comparison)
    print(f'N+P complete: 15 runs, 3600 updates, {result["protected_files_unchanged"]} protected files unchanged.')


if __name__ == '__main__':
    report(audit())
