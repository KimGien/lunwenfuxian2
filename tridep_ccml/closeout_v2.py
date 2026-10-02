"""Read-only verification of finished experiments and a separate v2 closeout snapshot."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path

from .v2.common import WORK, TC, RecordContext, descriptor, digest, read_json, write_new, require
from .v2.rebuild_eeg import validate_row
from .v2.complete_p import gate
from .metrics import summarize, binary_metrics

OUT = TC/'reports/ccml_v2_closeout'
REPORT = OUT/'CCML-v2总体总结与收尾报告.md'
SNAPSHOT_DATE = '2026-09-23'
DATE = '2026-09-24'
FAMILIES = ['E', 'NR', 'P', 'NP', 'A', 'AS', 'AS-NP', 'AS-NP-R']
METRICS = ['macro_f1', 'mdd_recall', 'balanced_accuracy', 'nll', 'brier']
MODALITIES = ['final', 'eeg', 'speech', 'text']
ORDER = ['E-adapt', 'E-control', 'N', 'R', 'P', 'NP', 'A-adapt', 'A-control', 'A-N',
         'AS-adapt', 'AS-control', 'AS-N', 'AS-NP', 'AS-NP-R']
GROUPS = [('修订 EEG＋原语音路径', ORDER[:6]), ('修订 EEG＋A 语音汇聚', ORDER[6:9]),
          ('修订 EEG＋AS 完整语音覆盖', ORDER[9:])]


def exp(label):
    return f'CCML-v2-{label}-37-fulltext'


def get_expected(audits):
    result = {label: audits['E']['summary'][exp(label)] for label in ['E-adapt', 'E-control']}
    result.update({label: audits['NR']['variants'][label]['summary'] for label in ['N', 'R']})
    result['P'], result['NP'] = audits['P']['summary'], audits['NP']['summaries']['NP']
    for label in ['adapt', 'control', 'N']:
        result['A-'+label] = audits['A']['summaries'][label]
        result['AS-'+label] = audits['AS']['summaries']['AS'][label]
    result['AS-NP'] = audits['AS-NP']['summaries']['candidate']
    result['AS-NP-R'] = audits['AS-NP-R']['summaries']['candidate']
    return result


def link(path):
    return Path(os.path.relpath(path, OUT)).as_posix()


def verify_and_summarize():
    context = RecordContext()
    OUT.mkdir(parents=True, exist_ok=True)
    verified = {}

    def check(item):
        name, expected = item['path'], item['sha256']
        if name in verified:
            require(verified[name] == expected, 'Conflicting historical hash: '+name)
        else:
            require(digest(WORK/name) == expected, 'Historical file changed: '+name)
            verified[name] = expected
        return WORK/name

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                check(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    supplement_dir = TC/'experiments/TC-00-E-37-fulltext'
    frozen = read_json(supplement_dir/'history_before.json')
    for p in supplement_dir.rglob('*'):
        if p.is_file():
            frozen[p.relative_to(WORK).as_posix()] = digest(p)
    for p in [TC/'tc00_eeg.py', TC/'complete_tc00_eeg.py', TC/'test_tc00_eeg.py',
              WORK/'output/plans/TC00_EEG修订_原始TRIDEP固定对照计划.md',
              *[(TC/'experiments')/f'TC-00-E-{s}.log' for s in ['preflight', 'train', 'completion']]]:
        frozen[p.relative_to(WORK).as_posix()] = digest(p)
    for name, sha in frozen.items():
        check({'path': name, 'sha256': sha})
    require('output/plans/README.md' not in frozen and 'tridep_ccml/README.md' not in frozen,
            'Mutable navigation unexpectedly protected')
    audits, statuses, family_info = {}, {}, {}
    grouped, run_info = defaultdict(list), defaultdict(list)
    for family in FAMILIES:
        folder = TC/f'experiments/CCML-v2-{family}-37-fulltext'
        status = read_json(folder/'status.json')
        require(status['status'] == 'complete', 'Unfinished family '+family)
        walk(status)
        audit = read_json(check(status['audit']))
        require(audit['passed'], 'Failed stage audit '+family)
        config = status.get('configuration', descriptor(folder/'configuration.json'))
        walk(read_json(check(config)))
        statuses[family], audits[family] = status, audit
        count = 0
        for prediction in sorted((folder/'runs').rglob('predictions.json')):
            run = prediction.parent
            complete = read_json(run/'complete.json')
            walk(complete)
            sidecar = read_json(run/'checkpoint.json')
            walk(sidecar)
            meta = sidecar['metadata']
            rows = read_json(prediction)
            require(len(rows) == 37 and [r['subject_id'] for r in rows] == context.ids, 'Prediction identities changed')
            condition = rows[0]['experiment'].removeprefix('CCML-v2-').removesuffix('-37-fulltext')
            require(condition in ORDER, 'Unexpected experiment')
            seed, fold = rows[0]['seed'], rows[0]['outer_fold']
            require(meta['experiment'] == exp(condition) and meta['seed'] == seed and meta['fold'] == fold and
                    meta['train_ids'] == context.folds[f'fold_{fold}']['train'], 'Run metadata differs')
            require(meta['cohort_sha256'] == context.cohort_sha256 and meta['folds_sha256'] == context.folds_sha256,
                    'Run cohort changed')
            updates = meta.get('updates', meta.get('optimizer_updates'))
            require(updates == 240, 'Run update budget mismatch')
            for r in rows:
                require(r['experiment'] == exp(condition) and r['seed'] == seed and r['outer_fold'] == fold and
                        r['subject_id'] in context.folds[f'fold_{fold}'][r['split']] and
                        r['label'] == context.labels[r['subject_id']] and r['train_ids'] == meta['train_ids'],
                        'Prediction labels/split/provenance mismatch')
                validate_row(r)
            grouped[condition] += rows
            run_info[condition].append({'seed': seed, 'fold': fold, 'updates': updates,
                                       'trainable_parameters': meta['trainable_parameters'],
                                       'checkpoint': descriptor(run/'checkpoint.pt'), 'predictions': descriptor(prediction)})
            count += 1
        require(sum(c['updates'] for label in run_info for c in run_info[label]
                    if c['checkpoint']['path'].startswith(folder.relative_to(WORK).as_posix()+'/')) == status['updates'],
                'Family total updates differ')
        family_info[family] = {'runs': count, 'updates': status['updates'],
                               'status': descriptor(folder/'status.json'), 'audit': status['audit'],
                               'report': status['report'], 'configuration': config}
    expected = get_expected(audits)
    conditions = {}
    for label in ORDER:
        rows = grouped[label]
        require(len(run_info[label]) == 15 and {(r['seed'], r['fold']) for r in run_info[label]} ==
                {(s, f) for s in [11, 22, 33] for f in range(1, 6)}, 'Missing seed/fold run')
        require(len(rows) == 555 and Counter((r['seed'], r['subject_id']) for r in rows if r['split'] == 'test') ==
                Counter((s, i) for s in [11, 22, 33] for i in context.ids), 'Incomplete OOF coverage')
        summary = summarize(rows)
        for field in ['primary_seed_mean_of_fold_means', 'separate_seed_mean_of_pooled_oof']:
            require(all(math.isclose(summary[field][k], expected[label][field][k], abs_tol=1e-12, rel_tol=1e-12)
                        for k in METRICS), 'Stage summary mismatch '+label)
        splits = {}
        for split in ['train', 'val', 'test']:
            splits[split] = {}
            for modality in MODALITIES:
                values = []
                for s in [11, 22, 33]:
                    for f in range(1, 6):
                        part = [r for r in rows if r['seed'] == s and r['outer_fold'] == f and r['split'] == split]
                        values.append(binary_metrics([r['label'] for r in part],
                            [(r['p_final'] if modality == 'final' else r['views'][modality]['p'])[1] for r in part]))
                splits[split][modality] = {k: sum(v[k] for v in values)/15 for k in METRICS}
        parameters = {r['trainable_parameters'] for r in run_info[label]}
        require(len(parameters) == 1, 'Parameters vary by run')
        test = [r for r in rows if r['split'] == 'test']
        speech_correct = lambda r: int(r['views']['speech']['p'][1] > .5) == r['label']
        conditions[label] = {'experiment': exp(label), 'summary': summary, 'splits': splits,
                             'parameters': parameters.pop(), 'runs': run_info[label],
                             'test_speech_correct_final_wrong': sum(speech_correct(r) and r['prediction'] != r['label'] for r in test),
                             'test_final_correct': sum(r['prediction'] == r['label'] for r in test),
                             'test_speech_correct': sum(speech_correct(r) for r in test),
                             'train_correct_records': sum(r['prediction'] == r['label'] for r in rows if r['split'] == 'train'),
                             'train_records': sum(r['split'] == 'train' for r in rows)}
    pairs = [('N','E-control'), ('R','E-control'), ('P','E-control'), ('NP','N'),
             ('A-N','N'), ('A-N','A-control'), ('AS-N','A-N'), ('AS-N','AS-control'),
             ('AS-NP','AS-N'), ('AS-NP-R','AS-NP')]
    stored = [audits['NR']['variants']['N']['stage_gate'], audits['NR']['variants']['R']['stage_gate'],
              audits['P']['paired_control_comparison']['gate'], audits['NP']['comparisons']['N']['gate'],
              audits['A']['comparisons']['A_N_vs_old_N']['gate'], audits['A']['comparisons']['A_N_vs_A_control']['gate'],
              audits['AS']['comparisons']['AS_N_vs_A_N']['gate'], audits['AS']['comparisons']['AS_N_vs_AS_control']['gate'],
              audits['AS-NP']['comparisons']['N']['gate'], audits['AS-NP-R']['comparisons']['baseline']['gate']]
    comparisons = []
    for (new, old), previous in zip(pairs, stored):
        comparison = gate(conditions[new]['summary'], conditions[old]['summary'])
        require(comparison['gate']['development_criteria_passed'] == previous['development_criteria_passed'],
                'Stage gate disagreement')
        comparisons.append({'candidate': new, 'comparator': old, **comparison,
                            'scope': 'source revision comparison' if (new, old) in [('A-N','N'), ('AS-N','A-N')] else 'same-version comparison'})
    supplement_status = read_json(supplement_dir/'status.json')
    walk(supplement_status)
    supplement = read_json(check(supplement_status['audit']))
    require(supplement_status['status'] == 'complete' and supplement['passed'], 'Supplement incomplete')
    historical = {}
    for stage, file in [('TC-00', TC/'reports/tc00_metrics.json'), ('TC-01', TC/'reports/tc01/summary.json'),
                        ('TC-04', TC/'reports/tc04/summary.json')]:
        if file.exists():
            historical[stage] = descriptor(file)
    total_runs = sum(f['runs'] for f in family_info.values())
    total_updates = sum(f['updates'] for f in family_info.values())
    require(total_runs == 210 and total_updates == 50400, 'Unexpected v2 totals')
    result = {'date': SNAPSHOT_DATE, 'scope': 'Completed fixed v2 experiments; no new training',
              'cohort': {'independent_subjects': 37, 'HC': 21, 'MDD': 16,
                         'train_per_fold': [len(context.folds[f'fold_{f}']['train']) for f in range(1,6)],
                         'val_per_fold': 3, 'test_per_fold': [len(context.folds[f'fold_{f}']['test']) for f in range(1,6)],
                         'cohort_sha256': context.cohort_sha256, 'folds_sha256': context.folds_sha256},
              'families': family_info, 'conditions': conditions, 'comparisons': comparisons,
              'totals': {'conditions': 14, 'training_runs': total_runs, 'optimizer_updates': total_updates,
                         'prediction_records': sum(len(r) for r in grouped.values()), 'test_records': 14*111,
                         'new_closeout_training_runs': 0},
              'supplement': {'status': descriptor(supplement_dir/'status.json'), 'report': supplement_status['report'],
                             'summaries': supplement['summaries'], 'split_scores': supplement['split_scores'],
                             'new_eeg_runs': 15, 'new_updates': 1800, 'replay_runs': 15, 'replay_updates': 1800},
              'historical_reference_files': historical,
              'verification': {'passed': True, 'recomputed_conditions': 14, 'recomputed_comparisons': 10,
                               'prediction_arithmetic_checked': 7770, 'verified_distinct_files': len(verified),
                               'new_checkpoint_inference': False,
                               'note': 'Closeout rehashes artifacts and recomputes stored predictions; exact checkpoint replay remains the completed stage audit.'}}
    write_new(OUT/'historical_snapshot.json', frozen)
    write_new(OUT/'verified_sources.json', verified)
    write_new(OUT/'results.json', result)
    return result


def write_text_new(path, value):
    if path.exists():
        require(path.read_text(encoding='utf-8') == value, 'Refusing changed closeout artifact: '+str(path))
    else:
        path.write_text(value, encoding='utf-8')


def plot_results(result):
    os.environ.setdefault('MPLCONFIGDIR', str(WORK/'tmp/ccml_v2_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharex=True,
                             gridspec_kw={'width_ratios': [1.12, 1, 1.12]})
    colors = ['#3274A1', '#AF7624', '#287F6C']
    for ax, (title, labels), color in zip(axes, GROUPS, colors):
        for i, label in enumerate(labels):
            c = result['conditions'][label]
            train = c['splits']['train']['final']['macro_f1']
            test = c['splits']['test']['final']['macro_f1']
            ax.plot([test, train], [i, i], color='#CBD2D9', linewidth=2, zorder=1)
            ax.scatter(train, i, s=42, color='#858D96', marker='s', zorder=3)
            ax.scatter(test, i, s=62, color=color, zorder=4)
            ax.annotate(f'{test:.3f}', (test, i), xytext=(0, -17), textcoords='offset points',
                        ha='center', color=color, fontsize=10)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels)-.35, -.65)
        ax.set_xlim(.30, 1.045)
        ax.set_xticks([.4, .6, .8, 1.0])
        ax.set_title(title, fontproperties=font, fontsize=12, pad=16)
        ax.set_xlabel('Macro-F1')
        ax.grid(axis='x', color='#E8EAED')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='s', color='#858D96', linestyle='', label='训练集'),
                        Line2D([], [], marker='o', color='#3274A1', linestyle='', label='测试集（颜色区分输入版本）')],
               loc='upper center', bbox_to_anchor=(.5, .915), ncol=2, frameon=False, prop=font)
    fig.suptitle('CCML v2：训练与测试的泛化差距', fontproperties=font, fontsize=19, y=.985)
    fig.text(.5, .025, '每个条件为 3 种子 × 5 折的均值；全部结果来自同一 37 人。分组间含输入与父表征变化，不能视作单因素排名。',
             ha='center', fontproperties=font, fontsize=10, color='#58616B')
    fig.subplots_adjust(left=.075, right=.985, top=.76, bottom=.13, wspace=.48)
    path = OUT/'训练与测试泛化差距.png'
    if not path.exists():
        fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return path


def build_report(result, figure):
    c = result['conditions']
    score = lambda name: c[name]['summary']['primary_seed_mean_of_fold_means']
    supplement = result['supplement']
    primary = lambda family, mode: supplement['summaries'][family][mode]['primary_seed_mean_of_fold_means']
    all_train_perfect = [name for name in ORDER if c[name]['train_correct_records'] == c[name]['train_records']]
    lines = [
        '# CCML v2 总体总结与收尾报告', '',
        f'收尾日期：{DATE}。范围：既定 v2 固定实验、已完成的原始 TRI-DEP EEG 补充对照及其解释边界。', '',
        '## 1. 总体结论与收尾决定', '',
        '**v2 既定固定实验执行完毕，现予收尾。本次仅核验和汇总，新增训练为 0。** '
        'Audit、E、N/R、P、NP、A、AS、AS-NP、AS-NP-R 均已完成。8 个训练阶段包含 14 个条件、210 组训练、50,400 次优化更新。'
        '每个条件均覆盖种子 11/22/33 × 5 折。原始 TRI-DEP 的 TC-00-E 作为独立补充对照列示，不计入 v2 的 CCML 训练预算。', '',
        f'旧语音路径的 **NP 是已执行 v2 条件中开发 Macro-F1 最高者：{score("NP")["macro_f1"]:.6f}**；'
        f'覆盖完整语音的 **AS-NP 为该版本的开发参考：{score("AS-NP")["macro_f1"]:.6f}**。'
        '两者不合并、不互相覆盖，也不宣称为独立验证的最优模型。AS-NP-R 未带来分类改善，不替换 AS-NP。', '',
        '**当前不能认定已解决过拟合，也不能认定 CCML 已优于原始 TRI-DEP。** '
        '输入修订、复现验收与泛化性能是三个不同结论：前两项已形成可追溯产物，泛化问题仍待后续独立设计验证。'
        '可选的 S 扩展、独立人群验证及原 TC-06 不标记为完成。', '',
        '## 2. 评价口径与完成清单', '',
        '全部主指标沿用预先约定口径：先求每个种子的五折指标算术均值，再求三个种子的均值。'
        '下文 F1 均指二分类 Macro-F1；NLL 和 Brier 越低越好。合并 OOF 指标另存于 results.json，不替代主口径。'
        '训练集、验证集分数也按同一折均值口径汇总。', '',
        '独立受试者始终为 **37 人（HC 21、MDD 16）**。各折 train 为 27/26/26/27/27 人，val 均为 3 人，test 为 7/8/8/7/7 人。'
        '每个条件有 555 条全划分预测、111 条重复种子测试记录；全 v2 共 7,770 条预测、1,554 条测试记录，均不能当作新增独立样本。', '',
        '| 阶段 | 固定条件 | 训练组数 | 优化更新数 | 完成记录 |',
        '|---|---|---:|---:|---|',
    ]
    names = {'E': 'E-adapt / E-control', 'NR': 'N / R', 'P': 'P', 'NP': 'NP',
             'A': 'A-adapt / A-control / A-N', 'AS': 'AS-adapt / AS-control / AS-N',
             'AS-NP': 'AS-NP', 'AS-NP-R': 'AS-NP-R'}
    for f, info in result['families'].items():
        lines.append(f'| {f} | {names[f]} | {info["runs"]} | {info["updates"]:,} | [阶段报告]({link(WORK/info["report"]["path"])}) |')
    lines += ['| **v2 合计** | **14 个条件** | **210** | **50,400** | **全部完成** |', '',
              '前置 [Audit 输入核查报告](../ccml_v2_audit/CCML-v2-Audit输入核查报告.md) 不涉及训练。'
              '以上均为每组 60 轮、batch 8、240 次更新、固定末轮检查点；未根据测试结果改阈值或提前停止。', '',
              '## 3. 全部 v2 主结果', '']
    for title, labels in GROUPS:
        lines += [f'### {title}', '', '| 条件 | Macro-F1 | MDD recall | NLL | Brier | 本阶段可训练参数 |',
                  '|---|---:|---:|---:|---:|---:|']
        for label in labels:
            s = score(label)
            lines.append(f'| {label} | {s["macro_f1"]:.6f} | {s["mdd_recall"]:.6f} | {s["nll"]:.6f} | {s["brier"]:.6f} | {c[label]["parameters"]:,} |')
        lines.append('')
    lines += ['参数列只计算该阶段实际更新参数，不代表从头训练的整个系统容量；冻结的父表征已包含同折训练集的监督学习。', '',
              '### 历史参照（不计入 v2 训练预算）', '',
              '| 历史条件 | Macro-F1 | MDD recall | NLL | Brier |', '|---|---:|---:|---:|---:|']
    for stage, description in [('tc01', 'TC-01：原表征＋新 CE 头'), ('tc04', 'TC-04：原表征＋完整 CCML')]:
        historical_score = read_json(TC/f'reports/{stage}/summary.json')['fusion']['primary_seed_mean_of_fold_means']
        title = f'[{description}](../{stage}/{stage.upper().replace("TC", "TC-")}实验完成报告.md)'
        lines.append('| '+title+' | '+' | '.join(f'{historical_score[k]:.6f}' for k in ['macro_f1', 'mdd_recall', 'nll', 'brier'])+' |')
    lines += ['', '原始 TC-00 的选权与等权融合见第 6 节。不同输入、父表征及训练范围之间的成绩只作开发参照，不能据此单独归因于 CCML 损失。', '',
              '### 配对比较与固定开发门槛', '',
              '固定门槛为 F1 至少提高 0.01、至少 2/3 种子严格改善、召回下降不超过 0.05；同时披露 NLL/Brier。'
              '“通过”仅指开发门槛，不代表统计显著或独立验证。下表差值均为“候选减对照”；F1/召回用百分点，概率损失用原单位。', '',
              '| 候选 / 对照 | ΔF1（百分点） | 改善种子 | Δ召回（百分点） | ΔNLL | ΔBrier | 开发门槛 |',
              '|---|---:|---:|---:|---:|---:|---|']
    for pair in result['comparisons']:
        d = pair['delta']
        positive = sum(x > 0 for x in pair['seed_f1_deltas'].values())
        passed = '通过' if pair['gate']['development_criteria_passed'] else '未通过'
        lines.append(f'| {pair["candidate"]} / {pair["comparator"]} | {100*d["macro_f1"]:+.3f} | {positive}/3 | {100*d["mdd_recall"]:+.3f} | {d["nll"]:+.6f} | {d["brier"]:+.6f} | {passed} |')
    lines += ['',
        'N 的规范化效果在同版本固定对照中得到支持；单独 P 的提升仅一个种子为正。旧 NP 相对 N 的 F1 改善较一致，但 NLL 略有恶化。'
        'A-N 相对旧 N 下降，不能用 A 版本内部规范化的改善掩盖这一结果。AS 完整覆盖改善概率指标和语音分支，但当时融合分类没有变化。'
        'AS-NP 相对 AS-N 改善了融合；AS-NP-R 的三个种子 F1 及全部 111 条测试分类均未改变，NLL/Brier 反而略差。', '',
        'A-N / N 与 AS-N / A-N 为输入或表征版本修订比较，不能当作同一输入下的纯 N/P/R 主效应。'
        '旧 NP 与 AS-NP 也不构成单因素对照，跨版本差异不能直接解释为某一个模块的贡献。', '',
        '## 4. 输入修订与模型变化：已经证实什么', '',
        '- **EEG：**Audit 确认原 19 电极按索引选取存在偏移。E 改为按标签匹配，重新导出 37 人的 60×200 冻结骨干特征；'
        '原数值、滤波及参考策略保持不变。随后又进行了 CCML EEG CNN 与投影适配，因此 E 与旧 TC-04 的差异同时含输入修订和适配，不能单独归因于通道修订。',
        '- **A 语音汇聚：**替换原冻结随机 CNN-BiGRU 汇聚路径及旧语音分类投影，采用片段均值、29 题等权均值的 768 维输入及可训练 768→128 投影。'
        '该阶段仍保留 96 个短音频题目的零向量，由下一阶段处理。',
        '- **AS 短语音：**96 题按实际长度提取 HuBERT 特征，题目覆盖达到 1,073/1,073；977 题旧均值和 8 位原无空题受试者的汇聚向量不变。'
        '已完成长片段及短音频重提取精确复核。这是覆盖修复成功，不意味着融合 F1 必然上升。',
        '- **AS-NP：**在同一 AS 输入下保留无仿射 LayerNorm，允许三路末端投影及新证据头更新，共 476,294 参数；'
        '沿用合法同折父初始化，未接着旧 NP 的训练后权重继续训练。',
        '- **AS-NP-R：**仅将 Adamax weight_decay 从 0 改为 1e-4，作用于全部投影和证据头参数（含 bias）。'
        '该正则确实缩小参数范数，但没有解决本次测试分类与泛化差距；单一强度失败不等于所有正则方法无效。', '',
        'EEG 电极标签已修订，但 GSN129/HydroCel 坐标模板对应关系和 Mumtaz 预训练输入幅值尺度的来源闭环仍未完成。'
        '本轮没有凭测试结果任意缩放输入，也不能将性能下降定论为幅值错误。', '',
        '## 5. 过拟合与融合失效的收尾诊断', '',
        f'重新汇总保存预测后，**14 个条件中 {len(all_train_perfect)} 个的全部训练预测均正确**，其余条件的平均训练 F1 也均不低于 0.9898。'
        '训练 F1=1 本身不足以判定严重程度；与下表持续存在的 train−test 差距一起，构成明显的泛化问题。'
        '这些是末轮、推理模式下的重代入训练分数，不是未见样本成绩。', '',
        '| 条件 | Train F1 | Val F1 | Test F1 | Train−Test |', '|---|---:|---:|---:|---:|']
    for label in ORDER:
        vals = [c[label]['splits'][part]['final']['macro_f1'] for part in ['train', 'val', 'test']]
        lines.append(f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {vals[0]-vals[2]:.6f} |')
    lines += ['', f'![训练与测试泛化差距]({figure.name})', '',
        '主要原因与证据边界如下：', '',
        '1. **每折仅 26–27 人训练。**即使只有 774 个新头参数，输入也是已在同一训练折上监督拟合的父表征，不能把整条学习链理解成只有 774 个参数。'
        'AS-NP 又更新了 47.6 万参数，样本量与可拟合容量的差距更大。',
        '2. **协议有利于训练分数饱和。**固定训练 60 轮，未使用早停；大多数条件 weight_decay=0。'
        '适配阶段的投影 dropout 沿用 eval 模式，因此相关随机失活没有在梯度更新时发挥正则作用。'
        '这些是复核到的配置，不是已经通过对照证明的唯一原因。',
        '3. **CCML 与规范化不保证泛化。**CCML 的正确类证据掩码 KL 项并不直接禁止模型把训练样本拟合正确；'
        'LayerNorm 也不限制可记忆容量。F1 达到 1 只要求分类方向正确，不表示训练损失已经为 0。',
        '4. **本次正则强度不足以改变观测结果。**AS-NP-R 与 AS-NP 都是 399/399 条训练记录分类正确，测试 F1 都为 0.658326。'
        '不能因有 weight_decay 就宣布过拟合已解决。',
        '5. **开发集的反复查看带来选择偏差。**所有阶段反复使用同一 37 人与既有外层折；三个种子不是三批新受试者。'
        '已完成的名单与父权重审计未发现已检查环节的 train/test 人员混用，但这不能替代对所有潜在泄漏或自适应选择偏差的检验。', '',
        '验证集每折只有 3 人，Val F1 波动很大，不能凭一个较高验证分数认定过拟合消失。'
        '原始 TRI-DEP 等权融合也出现 Train F1=1，因此该现象不能完全归咎于 v2 或 CCML。', '',
        '### 语音分支改善为何不等于融合改善', '',
        '| 条件 | 语音 Test F1 | 融合 Test F1 | 语音正确 / 111 | 融合正确 / 111 | 语音正确但融合错误 |',
        '|---|---:|---:|---:|---:|---:|']
    for label in ['N', 'A-N', 'AS-N', 'AS-NP', 'AS-NP-R']:
        value = c[label]
        lines.append(f'| {label} | {value["splits"]["test"]["speech"]["macro_f1"]:.6f} | {score(label)["macro_f1"]:.6f} | {value["test_speech_correct"]} | {value["test_final_correct"]} | {value["test_speech_correct_final_wrong"]} |')
    lines += ['',
        'AS-NP 将“语音正确但融合错误”从 32 条降至 15 条，表明投影适配改善了部分模态共同作用；'
        '但语音单分支 F1 同时下降，且融合仍低于该分支。上述重复种子记录只用于诊断，不能据此按测试标签手调证据权重。', '',
        '## 6. 独立补充：原始 TRI-DEP 使用修订 EEG', '',
        '按用户后续明确要求，将 E 阶段通道标签修订后的 60×200 EEG 特征接入原始 TC-00 / B0-37-fulltext。'
        '该对照恢复原始 EEGConvPoolClassifier 的 CE 训练、原语音/文本权重和原概率融合；未使用 E 阶段经 CCML 训练的 EEG 聚合器或投影。'
        '因此它是用户单独指定的 CE 补充对照，v2 主线 CCML-only 的历史范围不变。', '',
        '15 组新 EEG 训练、1,800 次更新完成；另有 15 组旧 EEG 训练、1,800 次更新用于精确重放原结果。'
        '新增实验与旧训练复现分开计数，不与 v2 的 210 组混加。', '',
        '| 原始 TRI-DEP 输出 | 原 EEG F1 | 修订 EEG F1 | 差值（百分点） |', '|---|---:|---:|---:|']
    for mode, title in [('weighted', '验证集选权融合'), ('mean', '等权融合'), ('eeg', 'EEG'), ('speech', '语音'), ('text', '文本')]:
        old, new = primary('original', mode)['macro_f1'], primary('corrected', mode)['macro_f1']
        lines.append(f'| {title} | {old:.6f} | {new:.6f} | {100*(new-old):+.3f} |')
    lines += ['',
        '选权融合的 111 条测试分类全部相同；15 组中有 14 组的验证集选权给 EEG 权重为 0。'
        '所以融合 F1 不变不能证明 EEG 修订没有影响。等权融合有 5 条分类改变（1 条错变对、4 条对变错），'
        'EEG 分支有 15 条改变（6 条错变对、9 条对变错），两者平均 F1 下降。', '',
        '这支持“标签修订并未在本次固定原始 TRI-DEP 协议下提高 EEG 分类性能”，不支持“应恢复已知错位的电极选择”。'
        '输入正确性与特定小样本协议的开发分数需要分别讨论。', '',
        f'详细核验见 [TC-00-E 原始 TRI-DEP 完成报告]({link(WORK/supplement["report"]["path"])}).', '',
        '## 7. 保留结果、未完成事项与下一阶段边界', '',
        '| 结果 | 收尾角色 | 保留理由与限制 |', '|---|---|---|',
        '| AS-NP | 完整语音覆盖版本的开发参考 | 同版本相对 AS-N 改善；仍有明显泛化差距，不作为部署定版。 |',
        '| 旧 NP | v2 历史开发 F1 最高参考 | 保留旧语音路径和 NLL 权衡，不能称为总体独立最优。 |',
        '| N | 固定表征规范化对照 | 作为旧输入版本的低更新参数控制；冻结父表征仍有监督血缘。 |',
        '| 其余全部条件 | 完整对照档案 | 保留权重、配置、历史报告和预测；未通过门槛者不自动升级。 |', '',
        '“保留参考”均指既有 15 组折/种子检查点及其预测集合；本轮没有重新全人群训练一个最终权重，也没有选出可部署的单一模型。'
        '每个检查点的路径和哈希见 results.json 的 conditions → runs。', '',
        '| 未完成事项 | 收尾处理 |', '|---|---|',
        '| 可选 CCML-v2-S 证据缩放、动态门控或新融合损失 | 尚未执行，涉及方法扩展；不因现有测试分数自动启动。 |',
        '| 独立 AS-R / AS-P 主效应对照 | 尚未执行；属于可追加的因子分解，不伪称本轮已经验证。 |',
        '| EEG 输入幅值与坐标模板来源闭环 | 仍待原数据说明、上游预训练处理依据核实；不凭性能反推正确单位。 |',
        '| 泛化确认 | 尚缺未反复用于开发的独立验证人群；现有分数不能当作最终泛化证明。 |',
        '| 原 TC-06 CE/CCML 配对解冻 | 原计划保留项，不属于此次 v2 完成范围，不标记完成。 |', '',
        '若另立下一阶段，应先固定主要问题、合法训练内验证方案和比较预算，再评估正则/容量/融合变化。'
        '同一数据上的嵌套验证有助于规范未来选择，但不能追溯性消除已经发生的反复开发；最终仍需独立数据确认。'
        '**当前没有自动续跑或新增超参数搜索任务。**', '',
        '## 8. 收尾核验与可追溯产物', '',
        f'本轮重新校验 {result["verification"]["verified_distinct_files"]:,} 个不同来源文件的哈希，核对 210 组运行身份、划分和更新预算，'
        '检查 7,770 条 CCML 预测的数值关系，从保存预测重新计算全部 14 个条件的主指标与合并 OOF 指标，并复核 10 个配对开发门槛；结果与阶段验收一致。', '',
        '本轮没有重复检查点推理或重新训练。阶段完成时已保存的精确预测重载、独立指标交叉核验和测试结果继续作为其原始验收证据：'
        '最后 v2 阶段累计 89 项测试通过，TC-00-E 补充对照 19 项测试通过；不把各阶段重叠测试数相加为新的独立覆盖。'
        '历史配置、代码、计划正文、预测与检查点保持原样；只新增本收尾目录与汇总程序，并更新两份导航 README。', '',
        '- [机器可读收尾状态](status.json)',
        '- [完整重算结果、配对差值、划分指标与检查点索引](results.json)',
        '- [本轮核验摘要](verification.json)',
        '- [来源文件哈希清单](verified_sources.json)',
        '- [收尾前历史保护快照](historical_snapshot.json)',
        f'- [训练与测试对照图]({figure.name})',
        f'- [原始 v2 计划（历史快照）]({link(WORK/"output/plans/CCML_v2_性能优化与后续计划.md")})',
        f'- [计划总索引]({link(WORK/"output/plans/README.md")})', '',
        '状态优先级：本收尾状态描述当前 v2 总体状态；各阶段旧报告中的“下一步”“尚未启动”等文字仅描述当时快照。'
        '既有 TC 阶段 experiment_status.json 继续保留其历史含义。', '',
    ]
    write_text_new(REPORT, '\n'.join(lines))


def finish(result, figure):
    verify = {**result['verification'], 'date': DATE,
              'totals': result['totals'], 'source_manifest': descriptor(OUT/'verified_sources.json'),
              'historical_snapshot': descriptor(OUT/'historical_snapshot.json'),
              'closeout_code': descriptor(Path(__file__).resolve())}
    write_new(OUT/'verification.json', verify)
    status = {'status': 'complete', 'completed_date': DATE, 'scope': 'fixed_experimental_v2_closeout',
              'fixed_experiments_complete': True, 'independent_validation_complete': False,
              'research_limitations_resolved': False, 'automatic_followup_training': False,
              'totals': result['totals'], 'supplement_counted_in_v2': False,
              'retained_development_references': {
                  'AS-NP': 'complete speech coverage development reference; not deployment validation',
                  'NP': 'highest observed v2 development macro-F1; historical speech path',
                  'N': 'fixed representation normalization control'},
              'not_completed': ['optional S or fusion method extension', 'independent AS-R/AS-P factorial controls',
                                'EEG scale and coordinate provenance closure', 'independent generalization validation',
                                'original TC-06 (outside this closeout scope)'],
              'report': descriptor(REPORT), 'results': descriptor(OUT/'results.json'),
              'figure': descriptor(figure), 'verification': descriptor(OUT/'verification.json')}
    write_new(OUT/'status.json', status)
    print(json.dumps({'status': 'complete', 'report': status['report']['path'],
                      'totals': result['totals'], 'verified_files': verify['verified_distinct_files']}, ensure_ascii=False))


if __name__ == '__main__':
    result = verify_and_summarize()
    figure = plot_results(result)
    build_report(result, figure)
    finish(result, figure)
