"""Recompute the complete v2 report, retaining the earlier closeout snapshots."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path

from .metrics import binary_metrics, summarize
from .records import RecordContext, digest, read_json, require

WORK = Path(__file__).resolve().parents[1]
TC = WORK / 'tridep_ccml'
OLD = TC / 'reports/ccml_v2_closeout'
OPTIONAL = TC / 'reports/ccml_v2_optional_closeout'
OUT = TC / 'reports/ccml_v2_full_closeout'
REPORT = OUT / 'CCML-v2总体总结与收尾报告.md'
METRICS = ['macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier']
MODALITIES = ['final', 'eeg', 'speech', 'text']
GROUPS = [
    ('修订 EEG＋原语音路径', ['E-adapt', 'E-control', 'N', 'R', 'P', 'NP']),
    ('修订 EEG＋A 语音汇聚', ['A-adapt', 'A-control', 'A-N']),
    ('修订 EEG＋AS 完整语音覆盖', ['AS-adapt', 'AS-control', 'AS-N', 'AS-NP', 'AS-NP-R', 'AS-R', 'AS-P']),
    ('S 骨干均值汇聚版本', ['S-control', 'S', 'G', 'F']),
]
ORDER = [label for _, labels in GROUPS for label in labels]


def descriptor(path):
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest(path)}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def gate(candidate, control):
    a, b = [x['primary_seed_mean_of_fold_means'] for x in (candidate, control)]
    delta = {key: a[key] - b[key] for key in METRICS}
    seeds = {str(s): candidate['seed_reports'][str(s)]['unweighted_fold_mean']['macro_f1'] -
             control['seed_reports'][str(s)]['unweighted_fold_mean']['macro_f1'] for s in [11, 22, 33]}
    return {'delta': delta, 'seed_f1_deltas': seeds,
            'gate': {'development_criteria_passed': delta['macro_f1'] >= .01 and
                     sum(v > 0 for v in seeds.values()) >= 2 and delta['mdd_recall'] >= -.05}}


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    context = RecordContext()
    checked = {}

    def verify(item):
        name, expected = item['path'], item['sha256']
        if name not in checked:
            require(digest(WORK/name) == expected, 'Source hash differs: ' + name)
            checked[name] = expected
        require(checked[name] == expected, 'Conflicting source: ' + name)
        return WORK/name

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                verify(value)
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    protected = {p.relative_to(WORK).as_posix(): digest(p)
                 for folder in [OLD, OPTIONAL] for p in folder.iterdir() if p.is_file()}
    old_status = read_json(OLD/'status.json')
    walk(old_status)
    old = read_json(verify(old_status['results']))
    op_status = read_json(OPTIONAL/'status.json')
    walk(op_status)
    op = read_json(verify(op_status['results']))
    require(old_status['status'] == op_status['status'] == 'complete', 'Prior report incomplete')
    source_statuses = {}
    audits = {}
    configs = {}
    for family in [*old['families'], 'S', 'AS-RP', 'GF']:
        folder = TC/f'experiments/CCML-v2-{family}-37-fulltext'
        status = read_json(folder/'status.json')
        require(status['status'] == 'complete', 'Incomplete family ' + family)
        source_statuses[family] = descriptor(folder/'status.json')
        walk(source_statuses[family])
        walk(status)
        audits[family] = read_json(verify(status['audit']))
        require(audits[family]['passed'], 'Failed completion audit ' + family)
        if family in ['S', 'AS-RP', 'GF']:
            configs[family] = read_json(verify(status['configuration']))
            walk(configs[family])
            tests = read_json(folder/'test_results.json')
            require(tests['passed'], 'Recorded stage tests failed')
            walk(tests)
            walk(descriptor(folder/'test_results.json'))
    expected = {k: v['summary'] for k, v in old['conditions'].items()}
    expected.update({'S-control': audits['S']['summaries']['control'], 'S': audits['S']['summaries']['S'],
                     'AS-R': audits['AS-RP']['summaries']['R'], 'AS-P': audits['AS-RP']['summaries']['P'],
                     'G': audits['GF']['summaries']['G'], 'F': audits['GF']['summaries']['F']})
    grouped, runs = defaultdict(list), defaultdict(list)
    for label, condition in old['conditions'].items():
        for run in condition['runs']:
            walk(run)
            grouped[label] += read_json(verify(run['predictions']))
            runs[label].append(run)
    family_budgets = {}
    inner_runs = []
    for family in ['S', 'AS-RP', 'GF']:
        folder = TC/f'experiments/CCML-v2-{family}-37-fulltext'
        budget = {'outer_runs': 0, 'inner_runs': 0, 'outer_updates': 0, 'inner_updates': 0,
                  'outer_prediction_records': 0, 'inner_prediction_records': 0}
        for pred in sorted((folder/'runs').rglob('predictions.json')):
            complete = read_json(pred.parent/'complete.json')
            walk(complete)
            walk(descriptor(pred.parent/'complete.json'))
            meta = read_json(verify(complete['checkpoint.json']))['metadata']
            rows = read_json(verify(complete['predictions.json']))
            inner = meta.get('inner_fold') is not None
            prefix = 'inner' if inner else 'outer'
            budget[prefix+'_runs'] += 1
            budget[prefix+'_updates'] += meta['updates']
            budget[prefix+'_prediction_records'] += len(rows)
            require(meta['updates'] == 60 * math.ceil(len(meta['train_ids'])/8), 'Unexpected update budget')
            require(set(meta['train_ids']) <= set(context.folds[f"fold_{meta['fold']}"]['train']), 'Invalid training membership')
            require(all(r['experiment'] == meta['experiment'] and r['seed'] == meta['seed'] and
                        r['outer_fold'] == meta['fold'] for r in rows), 'Run identity mismatch')
            label = meta['experiment'].removeprefix('CCML-v2-').removesuffix('-37-fulltext')
            run = {'seed': meta['seed'], 'fold': meta['fold'], 'inner_fold': meta.get('inner_fold'),
                   'updates': meta['updates'], 'checkpoint': complete['checkpoint.pt'],
                   'metadata': complete['checkpoint.json'], 'predictions': complete['predictions.json']}
            if inner:
                require(all(r['subject_id'] in context.folds[f"fold_{meta['fold']}"]['train'] and
                            r['split'] in ['train', 'inner_val'] for r in rows), 'Inner split escaped outer train')
                inner_runs.append({'family': family, 'condition': label, **run})
            else:
                grouped[label] += rows
                runs[label].append(run)
        family_budgets[family] = budget
        print(f'Verified {family}: {budget["outer_runs"]} outer / {budget["inner_runs"]} inner runs.', flush=True)

    conditions = {}
    for label in ORDER:
        rows = grouped[label]
        require(len(rows) == 555 and len(runs[label]) == 15, 'Outer coverage differs: ' + label)
        require({(r['seed'], r['fold']) for r in runs[label]} == {(s, f) for s in [11, 22, 33] for f in range(1, 6)}, 'Missing outer run')
        require(len({(r['seed'], r['outer_fold'], r['subject_id']) for r in rows}) == 555, 'Duplicate prediction')
        for row in rows:
            require(row['subject_id'] in context.folds[f"fold_{row['outer_fold']}"][row['split']], 'Wrong subject split')
            require(row['label'] == context.labels[row['subject_id']], 'Wrong subject label')
            require(row['cohort_sha256'] == context.cohort_sha256 and row['folds_sha256'] == context.folds_sha256, 'Cohort differs')
            require(row['prediction'] == int(row['p_final'][1] > .5), 'Prediction/threshold disagreement')
        test = [r for r in rows if r['split'] == 'test']
        require(Counter((r['seed'], r['subject_id']) for r in test) ==
                Counter((s, sid) for s in [11, 22, 33] for sid in context.ids), 'OOF coverage differs')
        summary = summarize(rows)
        for field in ['primary_seed_mean_of_fold_means', 'separate_seed_mean_of_pooled_oof']:
            for key in METRICS:
                require(math.isclose(summary[field][key], expected[label][field][key], abs_tol=1e-12, rel_tol=1e-12),
                        'Metric differs: ' + label + '/' + key)
        splits = {}
        for split in ['train', 'val', 'test']:
            splits[split] = {}
            for modality in MODALITIES:
                scores = []
                for seed in [11, 22, 33]:
                    for fold in range(1, 6):
                        part = [r for r in rows if r['seed'] == seed and r['outer_fold'] == fold and r['split'] == split]
                        scores.append(binary_metrics([r['label'] for r in part],
                            [(r['p_final'] if modality == 'final' else r['views'][modality]['p'])[1] for r in part]))
                splits[split][modality] = {key: sum(v[key] for v in scores)/15 for key in METRICS}
        parameters = (old['conditions'][label]['parameters'] if label in old['conditions'] else
                      {'S-control': 223366, 'S': 223369, 'G': configs['GF']['G']['parameters'],
                       'F': configs['GF']['F']['parameters'], 'AS-R': configs['AS-RP']['variants']['R']['parameters'],
                       'AS-P': configs['AS-RP']['variants']['P']['parameters']}[label])
        speech_correct = lambda r: int(r['views']['speech']['p'][1] > .5) == r['label']
        conditions[label] = {'summary': summary, 'splits': splits, 'parameters': parameters, 'runs': runs[label],
                            'test_final_correct': sum(r['prediction'] == r['label'] for r in test),
                            'test_speech_correct': sum(speech_correct(r) for r in test),
                            'test_speech_correct_final_wrong': sum(speech_correct(r) and r['prediction'] != r['label'] for r in test)}
    comparisons = []
    pairs = [(x['candidate'], x['comparator'], x) for x in old['comparisons']]
    pairs += [('S', 'S-control', audits['S']['comparison']),
              ('AS-R', 'AS-control', audits['AS-RP']['comparisons']['R']),
              ('AS-P', 'AS-control', audits['AS-RP']['comparisons']['P']),
              ('G', 'S-control', audits['GF']['comparisons']['G']),
              ('F', 'S-control', audits['GF']['comparisons']['F'])]
    for candidate, control, saved in pairs:
        result = gate(conditions[candidate]['summary'], conditions[control]['summary'])
        require(result['gate']['development_criteria_passed'] == saved['gate']['development_criteria_passed'], 'Gate differs')
        for key in METRICS:
            require(math.isclose(result['delta'][key], saved['delta'][key], abs_tol=1e-12), 'Paired delta differs')
        comparisons.append({'candidate': candidate, 'comparator': control, **result})
    changes = {}
    for candidate, control, _ in pairs[-5:]:
        base = {(r['seed'], r['subject_id']): r for r in grouped[control] if r['split'] == 'test'}
        changes[candidate] = sum(r['prediction'] != base[(r['seed'], r['subject_id'])]['prediction']
                                 for r in grouped[candidate] if r['split'] == 'test')
    totals = {'outer_conditions': len(conditions), 'outer_runs': sum(len(v) for v in runs.values()),
              'inner_runs': len(inner_runs), 'training_runs': sum(len(v) for v in runs.values()) + len(inner_runs),
              'optimizer_updates': sum(r['updates'] for v in runs.values() for r in v) + sum(r['updates'] for r in inner_runs),
              'outer_prediction_records': sum(len(v) for v in grouped.values()), 'outer_test_records': len(conditions)*111,
              'inner_prediction_records': sum(v['inner_prediction_records'] for v in family_budgets.values()),
              'new_report_training_runs': 0}
    require(totals['training_runs'] == op['cumulative_v2_budget']['total_runs'] == 570 and
            totals['optimizer_updates'] == op['cumulative_v2_budget']['total_updates'] == 120600 and
            totals['outer_runs'] == 300 and totals['inner_runs'] == 270, 'Cumulative budget differs')
    require(family_budgets['S']['inner_updates'] + family_budgets['S']['outer_updates'] == 23400 and
            family_budgets['GF']['inner_updates'] + family_budgets['GF']['outer_updates'] == 39600 and
            family_budgets['AS-RP']['outer_updates'] == 7200, 'Optional family budget differs')
    for path, sha in protected.items():
        require(digest(WORK/path) == sha, 'Historical closeout changed')
    result = {'date': '2026-09-24', 'scope': 'all_completed_fixed_and_registered_optional_v2_experiments',
              'cohort': old['cohort'], 'conditions': conditions, 'comparisons': comparisons,
              'optional_classification_changes_vs_control': changes, 'families': old['families'],
              'optional_family_budgets': family_budgets, 'inner_runs': inner_runs, 'totals': totals,
              'source_statuses': source_statuses, 'supplement': old['supplement'],
              'previous_fixed_closeout': descriptor(OLD/'status.json'),
              'previous_optional_closeout': descriptor(OPTIONAL/'status.json'),
              'selection_counts': {'S': audits['S']['selection_counts'], **audits['GF']['selection_counts']},
              'verification': {'passed': True, 'recomputed_outer_conditions': 20, 'recomputed_comparisons': 15,
                               'outer_records_checked': 11100, 'run_budgets_checked': 570,
                               'verified_source_files': len(checked), 'previous_closeout_files_unchanged': len(protected),
                               'new_checkpoint_inference': False, 'new_training': False,
                               'note': 'Rehashed run artifacts and recomputed stored outer predictions; exact model replay and inner selection recomputation remain the recorded stage audits.'}}
    write_json(OUT/'verified_sources.json', checked)
    write_json(OUT/'previous_closeouts_snapshot.json', protected)
    write_json(OUT/'results.json', result)
    return result


def plot(result):
    os.environ.setdefault('MPLCONFIGDIR', str(WORK/'tmp/ccml_v2_full_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (title, labels) in zip(axes.flat, GROUPS):
        for i, label in enumerate(labels):
            values = result['conditions'][label]['splits']
            train, test = [values[s]['final']['macro_f1'] for s in ['train', 'test']]
            ax.plot([test, train], [i, i], color='#cbd5df', linewidth=3)
            ax.scatter(test, i, color='#167d8d', s=48, zorder=3)
            ax.scatter(train, i, color='#c78629', s=35, marker='D', zorder=3)
            ax.text(test-.017, i, f'{test:.3f}', ha='right', va='center', fontsize=9)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels)-.5, -.6)
        ax.set_xlim(.35, 1.03)
        ax.set_title(title, fontproperties=font, pad=12)
        ax.set_xlabel('Macro-F1')
        ax.grid(axis='x', color='#e5e9ef')
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color='#167d8d', label='测试集'),
               plt.Line2D([], [], marker='D', linestyle='', color='#c78629', label='训练集')]
    fig.legend(handles=handles, loc='upper center', ncol=2, prop=font, bbox_to_anchor=(.5, .96), frameon=False)
    fig.suptitle('CCML v2 全部 20 个条件：训练与测试泛化差距', fontproperties=font, fontsize=17, y=.99)
    fig.text(.5, .015, '按输入版本分组；跨组分数差异不能归因于单一模块。主口径：三种子的五折均值再平均。',
             ha='center', fontproperties=font, fontsize=10)
    fig.tight_layout(rect=[0, .04, 1, .925], h_pad=3)
    path = OUT/'训练与测试泛化差距_含可选实验.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return path


def build_report(result, figure):
    legacy = (OLD/REPORT.name).read_text(encoding='utf8')
    conditions = result['conditions']
    score = lambda label: conditions[label]['summary']['primary_seed_mean_of_fold_means']
    lines = ['# CCML v2 总体总结与收尾报告（含全部已登记可选实验）', '',
        '更新日期：2026-09-24。范围：Audit 至 AS-NP-R 的原固定阶段、S、AS-R/AS-P、G/F，以及单列的原始 TRI-DEP EEG 修订补充对照。', '',
        '## 1. 总体结论与收尾决定', '',
        '**v2 原固定阶段与全部已登记可选实验均已完成并验收。累计 20 个外层条件、570 组训练（300 组外层＋270 组内层）、120,600 次优化更新。本次总结更新新增训练为 0。**', '',
        '**现有 v2 开发结果中的最高 Macro-F1 为 0.790859，由 S-control、S、G、F 并列取得，MDD recall 均为 0.666667。** '
        '这是 S 骨干均值汇聚版本的成绩。S 静态尺度、G 动态门控、F 额外最终融合监督相对同版本控制均未改变 111 条测试分类，均未通过预先固定的分类提升门槛；不能将较高分数归功于这些扩展。', '',
        'AS-R 的 F1 为 0.472030，与 AS-control 相同；AS-P 为 0.465902，低于 AS-control。二者也均未通过开发门槛。'
        '**本次五项可选候选 S、AS-R、AS-P、G、F 均已执行，但没有一项达到预定分类提升门槛。** G/F 的 NLL、Brier 略有改善，只作为概率指标结果保留。', '',
        '旧 NP（0.680895）仍是旧语音路径中的开发 F1 最高参考；AS-NP（0.658326）仍是旧 AS 监督父表征版本的参考。'
        '旧报告中“NP 为全部 v2 最高”仅描述 S 开始前的范围，已由本报告更新。S-control 保留为新均值版本的开发比较基准；未选出部署模型。', '',
        '**泛化差距尚未消除，也不能据此认定 CCML 已独立验证优于原始 TRI-DEP。** 新版本 train−test F1 差距为 0.188669，'
        '较旧 AS-NP 的 0.341674 小，但结构、输入维度与监督初始化同时变化；这不构成单一改动效果或独立人群验证。', '',
        '## 2. 评价口径与完整完成清单', '',
        '主指标保持原约定：先求每个种子的五折指标算术均值，再求三个种子的均值。F1 均指二分类 Macro-F1；NLL/Brier 越低越好。'
        '合并 OOF 指标另存于 results.json，不替代主口径。固定种子 11/22/33、五个外折、60 轮、batch 8、末轮检查点和 0.5 阈值；不选择最优种子或测试阈值。', '',
        '独立受试者仍为 **37 人（HC 21、MDD 16）**。各折 train 为 27/26/26/27/27 人，val 均为 3 人，test 为 7/8/8/7/7 人。'
        '每个外层条件包含 555 条全划分预测和 111 条重复种子测试记录；全部 20 个条件合计 11,100 条外层预测、2,220 条外层测试记录。'
        '270 组内层训练的 7,182 条预测另列，不能计为额外独立受试者或外层测试记录。', '',
        '| 阶段 | 外层条件 | 外层训练 | 内层训练 | 训练合计 | 更新合计 | 完成报告 |',
        '|---|---|---:|---:|---:|---:|---|']
    family_labels = {'E':'E-adapt / E-control', 'NR':'N / R', 'P':'P', 'NP':'NP', 'A':'A-adapt / A-control / A-N',
                     'AS':'AS-adapt / AS-control / AS-N', 'AS-NP':'AS-NP', 'AS-NP-R':'AS-NP-R'}
    for name, data in result['families'].items():
        rel = Path(os.path.relpath(WORK/data['report']['path'], OUT)).as_posix()
        lines.append(f'| {name} | {family_labels[name]} | {data["runs"]} | 0 | {data["runs"]} | {data["updates"]:,} | [已完成]({rel}) |')
    for family, label in [('S', 'S-control / S'), ('AS-RP', 'AS-R / AS-P'), ('GF', 'G / F')]:
        b = result['optional_family_budgets'][family]
        lines.append(f'| {family} | {label} | {b["outer_runs"]} | {b["inner_runs"]} | {b["outer_runs"]+b["inner_runs"]} | '
                     f'{b["outer_updates"]+b["inner_updates"]:,} | [已完成](../../experiments/CCML-v2-{family}-37-fulltext/CCML-v2-{family}实验完成报告.md) |')
    lines += ['| **v2 累计** | **20 个条件** | **300** | **270** | **570** | **120,600** | **全部完成** |', '',
        'Audit 输入核查不涉及训练。原固定阶段为 210 组/50,400 次更新；S 为 120 组/23,400 次；后续 AS-R/P＋G/F 为 240 组/46,800 次。'
        'AS-R/P 复用的 AS-control 与 G/F 复用的 S-control 均只计一次。外层每组 240 次更新，内层每组 180 次更新。'
        'TC-00-E 的新 EEG CE 训练及旧训练重放在第 6 节单列，不计入 v2。', '', '## 3. 全部 v2 主结果与配对比较', '']
    for title, labels in GROUPS:
        lines += ['### '+title, '', '| 条件 | Macro-F1 | MDD recall | NLL | Brier | 本阶段可训练参数 |', '|---|---:|---:|---:|---:|---:|']
        for label in labels:
            s = score(label)
            lines.append('| '+label+' | '+' | '.join(f'{s[k]:.6f}' for k in ['macro_f1','mdd_recall','nll','brier'])+f' | {conditions[label]["parameters"]:,} |')
        lines += ['']
    lines += ['参数只统计本阶段更新范围；旧版本冻结父表征已包含同折训练集的监督学习。新均值版本无旧 MODMA 监督父权重。'
              '**表格按输入/表征版本分组，跨组不能视为单因素对照。**', '',
              '### 历史参照与比较边界', '',
              'TC-01（原表征＋新 CE 头）的 F1 为 0.699762，TC-04（原表征＋完整 CCML）为 0.611614；原始 TRI-DEP 验证集选权融合为 0.724180。'
              '新均值版本 0.790859 在数值上较 0.724180 高 6.668 个百分点，但输入、结构、训练范围与融合策略不同，且人群已被反复用于开发，'
              '因此不能作为 CCML 损失本身优于 CE 或原始 TRI-DEP 的独立证据。', '',
              '### 固定开发门槛', '',
              '门槛为 F1 至少提高 0.01、至少 2/3 种子严格改善、MDD recall 下降不超过 0.05，同时披露 NLL/Brier。'
              '差值均为候选减对照；F1/召回用百分点，概率损失用原单位。“通过”仅指开发门槛，不代表统计显著。', '',
              '| 候选 / 对照 | ΔF1（百分点） | 改善种子 | Δ召回（百分点） | ΔNLL | ΔBrier | 开发门槛 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for item in result['comparisons']:
        d = item['delta']
        lines.append(f'| {item["candidate"]} / {item["comparator"]} | {100*d["macro_f1"]:+.3f} | '
                     f'{sum(v>0 for v in item["seed_f1_deltas"].values())}/3 | {100*d["mdd_recall"]:+.3f} | '
                     f'{d["nll"]:+.6f} | {d["brier"]:+.6f} | '+('通过' if item['gate']['development_criteria_passed'] else '未通过')+' |')
    lines += ['', '原固定阶段对规范化 N、同版本 NP 及 AS-NP 的改善已有开发证据；单独 P 和固定强度正则未稳定改善。'
              'AS-P 补充对照表明：在旧 AS 输入下，只增加投影适配、没有 N 时，分类不升反降，不能把 AS-NP 的改善直接归为 P 的独立效果。'
              'A-N/N、AS-N/A-N 涉及输入或表征修订；旧 NP 与 AS-NP 也不是单因素对照。', '',
              'S/G/F 均与各自同版本 S-control 比较。三者每个种子的 F1 及测试分类不变，不能把新版本共同达到 0.790859 解释成三种扩展各自有效。', '']
    input_section = legacy.split('## 4. 输入修订与模型变化：已经证实什么\n', 1)[1].split('## 5.', 1)[0]
    lines += ['## 4. 输入修订、可选方法与验证设计', '', input_section.rstrip(), '',
        '### 已完成的五项可选候选', '',
        '- **AS-R / AS-P：**与旧 AS-control 使用相同父版本、初始化和批序。AS-R 仅增加证据头 weight_decay=1e-4；AS-P 仅开放三路末端投影更新、无 LayerNorm。完整 CCML 保留，两项均未通过分类门槛。',
        '- **S：**为合法训练内验证建立 S-backbone-mean-v1。EEG 为修订 CBraMod 时间均值，语音为 AS 完整 29 题 HuBERT 均值，文本为全文 MacBERT 29 题 CLS 均值；三路新建 Linear→ReLU→无仿射 LayerNorm128→证据头。无旧 MODMA 监督初始化。S 在解耦前学习三个 [0.5,2] 的模态尺度，并向 1 收缩。',
        '- **G：**在同一均值版本上增加 9 个逐人门控参数，根据各模态本人的原始不确定性 u 和类别概率差生成 [0.5,2] 尺度；特征停止梯度，尺度向 1 收缩。',
        '- **F：**在同一均值版本上给最终融合 Dirichlet 输出增加期望交叉熵监督，无额外最终 KL、无缩放、无动态门控。G/F 保留完整 CCML，不叠加 S 或彼此。', '',
        'S/G/F 每个外层 train 内固定三折分层，所有监督参数仅在对应 inner-train 重新拟合。S/G 从 0.01/0.1 两个收缩强度中选择，'
        'F 从 0.1/1.0 两个监督权重中选择；准则为三折 inner-val 融合 NLL。S/G 平票取较大收缩，F 平票取较小监督权重（容差 1e-12）。'
        'S 的 15 份选择与后续 G/F 的 30 份选择分别全部锁定后才运行相应外层模型，外层 val/test 未用于本次强度选择。', '',
        '选择次数：'+ '；'.join(f'{name}：'+ '、'.join(f'{key} → {value}/15' for key,value in values.items()) for name,values in result['selection_counts'].items())+'。未追加候选或强度搜索。', '',
        '**合法内层选择改进了本次参数选择流程，但没有消除此前在同一 37 人上反复开发所带来的选择偏差。**', '',
        '## 5. 过拟合与融合诊断的更新', '',
        '旧 14 个条件中 6 个全部训练分类正确，其余平均训练 F1 均不低于 0.9898。新增 AS-R/P 的 train−test 差距仍约 0.52。'
        'S-control/S/G/F 的训练 F1 降至 0.979528、测试 F1 为 0.790859，差距缩小到 0.188669，但仍明显存在。'
        'S/G/F 相对直接控制的训练、验证、测试分类指标均相同，未显示额外缓解泛化差距。', '',
        '| 条件 | Train F1 | Val F1 | Test F1 | Train−Test |', '|---|---:|---:|---:|---:|']
    for label in ORDER:
        vals = [conditions[label]['splits'][s]['final']['macro_f1'] for s in ['train', 'val', 'test']]
        lines.append(f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {vals[0]-vals[2]:.6f} |')
    lines += ['', f'![全部条件训练与测试泛化差距]({figure.name})', '',
        '每折仅 26–27 人训练，固定 60 轮且无早停；旧适配阶段的投影 dropout 沿用 eval 模式，新均值版本没有 dropout。'
        '旧父表征具有监督拟合历史，新均值版本虽减少更新参数并重建监督部分，仍约 22.3 万参数。样本量、模型容量、训练协议和反复开发均是需要考虑的因素；'
        '这些配置与差距不是已由单一对照证明的唯一原因。LayerNorm、CCML 证据约束及一次固定强度正则均不能保证泛化。', '',
        '训练分数为末轮推理模式下的重代入成绩；每折验证集只有 3 人，较高 Val F1 也不能证明过拟合已消除。原始 TRI-DEP 等权融合同样出现 Train F1=1。', '',
        '### 语音分支与融合的版本差异', '',
        '| 条件 | 语音 Test F1 | 融合 Test F1 | 语音正确 / 111 | 融合正确 / 111 | 语音正确但融合错误 |',
        '|---|---:|---:|---:|---:|---:|']
    for label in ['N', 'A-N', 'AS-N', 'AS-NP', 'AS-NP-R', 'AS-R', 'AS-P', 'S-control', 'S', 'G', 'F']:
        v = conditions[label]
        lines.append(f'| {label} | {v["splits"]["test"]["speech"]["macro_f1"]:.6f} | {score(label)["macro_f1"]:.6f} | '
                     f'{v["test_speech_correct"]} | {v["test_final_correct"]} | {v["test_speech_correct_final_wrong"]} |')
    lines += ['', '旧 AS-NP 将“语音正确但融合错误”从 AS-N 的 32 条降至 15 条，但融合仍低于其语音分支。'
              '新均值版本控制的语音 F1 为 0.777540，融合为 0.790859，“语音正确但融合错误”为 9 条；'
              '因此旧报告中“融合始终弱于语音”的诊断不能直接推广到新版本。该变化涉及多处结构/输入差异，仍不能归因于 S/G/F。'
              '所有 111 条均为重复种子诊断记录，不用于按测试标签手调模态权重。', '']
    supplement = legacy.split('## 6. 独立补充：原始 TRI-DEP 使用修订 EEG\n', 1)[1].split('## 7.', 1)[0]
    supplement = supplement.replace('不与 v2 的 210 组混加', '不与 v2 累计的 570 组混加')
    lines += ['## 6. 独立补充：原始 TRI-DEP 使用修订 EEG', '', supplement.rstrip(), '',
        '## 7. 保留结果、完成状态与下一阶段边界', '',
        '| 结果 | 当前角色 | 理由与限制 |', '|---|---|---|',
        '| S-control | 新骨干均值版本的开发比较基准 | F1=0.790859；S/G/F 未带来分类增益，控制较简洁。该角色不等于部署选模。 |',
        '| S / G / F | 已完成的固定方法扩展档案 | 与控制分类相同；保留概率指标和选择记录，不升级为默认。 |',
        '| AS-NP | 旧 AS 监督父表征版本的开发参考 | 同版本相对 AS-N 改善；F1=0.658326，不与新均值版本混作单因素比较。 |',
        '| 旧 NP | 旧语音路径的历史开发参考 | F1=0.680895；已不代表含 S/G/F 后的全部 v2 最高分。 |',
        '| N | 旧固定表征规范化对照 | 保留低更新参数控制及其监督父表征血缘。 |',
        '| AS-R / AS-P 与其余固定条件 | 完整对照档案 | 全部已完成；未通过门槛者不自动升级。 |', '',
        '各条件保留 15 组外层折/种子检查点与预测；S/G/F 另保留全部 270 组内层检查点与选择记录。'
        '检查点索引分别见 results.json 的 conditions → runs 和 inner_runs。未在全部人群上另训最终权重，未选出可部署的单一模型。', '',
        '| 事项 | 当前状态与边界 |', '|---|---|',
        '| 原固定阶段、S、AS-R、AS-P、G、F | 全部已完成；本轮已登记可选实验没有待执行项。 |',
        '| EEG 输入幅值与坐标模板来源闭环 | 仍缺原采集/预训练处理依据；不能凭开发分数反推正确单位或模板。 |',
        '| 独立泛化验证 | 尚缺未反复用于开发的新标注人群；更多种子或原 37 人重新划分不能替代。 |',
        '| 原 TC-06 CE/CCML 配对解冻 | 属于 v2 之外的原计划保留项，未执行，不标记完成。 |',
        '| G+F、soft-min、模态丢弃及额外强度搜索 | 未登记为本轮固定候选，不计为可选实验欠项，也未启动。 |', '',
        '**本次全部已登记实验完成；研究中的外部证据缺口与独立验证缺口仍保留。当前没有后续已登记训练或自动搜索。**', '',
        '## 8. 本次核验与可追溯产物', '',
        f'本次总结更新重新校验 {result["verification"]["verified_source_files"]:,} 个来源文件哈希，核对 570 组训练的预算及运行索引，'
        '检查 11,100 条外层预测的人员划分、标签、阈值分类与覆盖，从保存预测重算全部 20 个条件的主指标、合并 OOF 与 train/val/test 指标，'
        '并重算 15 个配对门槛。结果与已验收阶段一致；复用控制未重复计数。', '',
        '本次未重新训练、未重新运行检查点推理、未重复执行模型测试或内层强度选择。以下为各阶段完成时已有验收证据，不能写作本次新增测试：', '',
        '| 阶段验收 | 已记录证据 |', '|---|---|',
        '| 原固定 v2 收尾 | 210 组、7,770 条预测、14 个条件、10 个比较；当时重核 4,860 个来源文件，末阶段累计 89 项测试通过。 |',
        '| S | 41 项测试；120 个检查点和 3,504 条预测精确重载；15 份内层选择复算，30 个外层测试折指标交叉核验。 |',
        '| AS-R/P | 33 项测试；30 个检查点和 1,110 条新预测精确重载；30 个新增外层测试折指标交叉核验。 |',
        '| G/F | 39 项测试；210 个检查点和 5,898 条新预测精确重载；30 份内层选择复算，30 个新增外层测试折指标交叉核验。 |',
        '| TC-00-E（单列） | 19 项测试；修订 EEG 训练和旧 EEG 精确重放均完成。 |', '',
        '各阶段测试含重叠项目，不相加宣称独立测试覆盖；历史哈希保护数字也不叠加。旧收尾目录、可选总结、阶段配置、报告、预测和权重保持原样。'
        '本次新增完整总结目录及其汇总程序，并更新两份导航 README。', '',
        '- [当前完整完成状态](status.json)',
        '- [全部重算指标、配对差值和外层/内层检查点索引](results.json)',
        '- [本次核验摘要](verification.json)',
        '- [本次来源哈希清单](verified_sources.json)',
        '- [此前两份收尾的保护快照](previous_closeouts_snapshot.json)',
        '- [原固定阶段收尾（历史快照）](../ccml_v2_closeout/CCML-v2总体总结与收尾报告.md)',
        '- [剩余可选实验总结（阶段快照）](../ccml_v2_optional_closeout/CCML-v2剩余可选实验总结.md)',
        '- [计划总索引](../../../output/plans/README.md)', '',
        '状态优先级：本报告及同目录 status.json 描述当前完整 v2 状态；旧阶段中的“可选尚未执行”“旧 NP 最高”等文字仅描述各自生成时点。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf8')


def finish(result, figure):
    verification = {**result['verification'], 'date': result['date'], 'totals': result['totals'],
                    'sources': descriptor(OUT/'verified_sources.json'),
                    'previous_closeouts_snapshot': descriptor(OUT/'previous_closeouts_snapshot.json'),
                    'report_code': descriptor(Path(__file__).resolve())}
    write_json(OUT/'verification.json', verification)
    write_json(OUT/'status.json', {'status': 'complete', 'updated_date': result['date'], 'scope': result['scope'],
        'fixed_experiments_complete': True, 'registered_optional_experiments_complete': True,
        'completed_optional_candidates': ['S', 'AS-R', 'AS-P', 'G', 'F'], 'optional_classification_gates_passed': [],
        'totals': result['totals'], 'highest_observed_development_f1': .7908585858585858,
        'highest_observed_conditions': ['S-control', 'S', 'G', 'F'],
        'retained_development_references': {'S-control': 'backbone mean version comparison baseline',
            'AS-NP': 'historical AS supervised-parent version', 'NP': 'historical speech-path version', 'N': 'historical normalization control'},
        'independent_validation_complete': False, 'research_limitations_resolved': False,
        'unresolved_external_items': ['EEG scale and coordinate provenance', 'independent generalization validation'],
        'outside_scope': ['original TC-06 CE/CCML paired unfreezing'], 'next_registered_training': None,
        'new_training_runs_this_update': 0, 'TC00_E_counted_in_v2': False,
        'report': descriptor(REPORT), 'results': descriptor(OUT/'results.json'),
        'figure': descriptor(figure), 'verification': descriptor(OUT/'verification.json')})
    print(json.dumps({'status': 'complete', 'report': str(REPORT), 'totals': result['totals']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    result = collect()
    figure = plot(result)
    build_report(result, figure)
    finish(result, figure)
