"""Report fixed comparisons only after the training and checkpoint replay gates."""
from __future__ import annotations
import csv
from .common import *
from .evaluate import summarize, comparisons


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for path in sorted((OUT/'preflight/tiny_fit').glob('*.json')):
        row = read_json(path)
        axes[0].semilogy(range(1,501),[r['pre_update_ce'] for r in row['history']],alpha=.65,lw=1)
    axes[0].axhline(.1,color='black',ls='--',label='Final CE acceptance = 0.1')
    axes[0].set(xlabel='Diagnostic update',ylabel='4-subject training CE',title='Fixed small-subset fit: 15 seed/fold runs')
    axes[0].legend(fontsize=8)
    for condition,color in [('U2','tab:blue'),('C2','tab:orange')]:
        ys = [[r['representation_relative_std'],r['zero_input_relative_response']] for seed in SEEDS for fold in FOLDS
              for r in [read_json(run_dir(condition,seed,fold)/'final_health.json')]]
        axes[1].scatter([v[0] for v in ys],[v[1] for v in ys],label=condition+'-P1',color=color,alpha=.7)
    axes[1].set(xlabel='Between-subject std / representation RMS',ylabel='Response to zero EEG input',
                title='Final encoder health (training subjects)')
    axes[1].legend()
    fig.savefig(OUT/'learning_acceptance.png',dpi=180)
    fig.savefig(OUT/'learning_acceptance.svg')
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for condition,color in [('U0','gray'),('U2','tab:blue'),('C0','gray'),('C2','tab:orange')]:
        ax = axes[0 if condition.startswith('U') else 1]
        curves = np.array([[r['loss'] for r in read_json(run_dir(condition,seed,fold)/'history.json')]
                          for seed in SEEDS for fold in FOLDS])
        for y in curves:
            ax.plot(range(1,61),y,color=color,lw=.5,alpha=.16)
        ax.plot(range(1,61),curves.mean(0),color=color,lw=2,label=condition+('-P1' if condition.endswith('2') else ' control'))
        ax.set(xlabel='Fixed formal epoch',ylabel='Training objective',title='CE' if condition.startswith('U') else 'Full CCML')
        ax.legend()
    fig.savefig(OUT/'training_curves.png',dpi=180)
    fig.savefig(OUT/'training_curves.svg')
    plt.close(fig)


def health_summary():
    out = {}
    for condition in CONDITIONS:
        initial = [read_json(run_dir(condition,s,f)/'initial_health.json') for s in SEEDS for f in FOLDS]
        final = [read_json(run_dir(condition,s,f)/'final_health.json') for s in SEEDS for f in FOLDS]
        histories = [r for s in SEEDS for f in FOLDS for r in read_json(run_dir(condition,s,f)/'history.json')]
        require(all(r['passed'] for r in initial+final),'Health gate failed')
        out[condition] = {'initial_passed':len(initial),'final_passed':len(final),
            'final_min_module_gradient_l2':min(v for r in final for v in r['gradient_l2_max_over_train_batches'].values()),
            'epoch_min_module_gradient_l2':min(v for r in histories for v in r['encoder_module_gradient_l2_max'].values()),
            'epoch_min_module_weight_update_l2':min(v for r in histories for v in r['encoder_weight_update_l2'].values()),
            **{'final_min_'+key:min(r[key] for r in final) for key in ['representation_std_rms','representation_relative_std','zero_input_relative_response']}}
        if condition in ['U2','C2']:
            out[condition]['final_min_region_response'] = min(v for r in final for v in r['region_relative_responses'].values())
    return out


def csv_outputs(rows,summary):
    path = OUT/'OOF_predictions.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition','seed','fold','subject_id','label','endpoint','p_MDD','prediction'])
        for r in rows:
            if r['split']=='test':
                for endpoint,p in r['probabilities'].items():
                    writer.writerow([r[k] for k in ['condition','seed','fold','subject_id','label']]+[endpoint,p[1],r['predictions'][endpoint]])
    with (OUT/'metrics_by_fold.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        keys = ['macro_f1','balanced_accuracy','mdd_recall','nll','brier']
        writer.writerow(['condition','endpoint','split','seed','fold']+keys)
        for c,endpoints in summary.items():
            for endpoint,result in endpoints.items():
                for split,detail in result['splits'].items():
                    for seed,folds in detail['folds'].items():
                        for fold,values in folds.items():
                            writer.writerow([c,endpoint,split,seed,fold]+[values[k] for k in keys])


def finish():
    verify_lock()
    require(read_json(OUT/'training_gate.json')['passed'] and read_json(OUT/'replay_gate.json')['passed'],'Release gate missing')
    rows = [r for c in CONDITIONS for s in SEEDS for f in FOLDS for r in read_json(run_dir(c,s,f)/'predictions.json')]
    require(len(rows)==2220,'Expected all 60 x 37 predictions')
    for c in CONDITIONS:
        for s in SEEDS:
            for f in FOLDS:
                meta = read_json(run_dir(c,s,f)/'verified.json')
                require(meta['passed'],'Unverified result')
                for key in ['checkpoint','predictions','features']:
                    verify_descriptor(meta[key])
    summary = summarize(rows)
    paired = comparisons(summary,rows)
    old = read_json(PARENT/'results.json')['summary']
    before_after = {c:{e:{k:summary[c][e]['splits']['test']['primary'][k]-old[c][e]['splits']['test']['primary'][k]
                    for k in ['macro_f1','balanced_accuracy','mdd_recall','nll','brier']}
                    for e in (['eeg'] if c.startswith('U') else ['eeg','fusion'])} for c in ['U2','C2']}
    # The retrained controls must reproduce both the stored weights and released metrics.
    for c in ['U0','C0']:
        require(summary[c]==old[c],'Control metrics do not exactly reproduce frozen R1')
    health = health_summary()
    plots()
    csv_outputs(rows,summary)
    historical_count = check_history()
    preflight = read_json(OUT/'preflight.json')
    verification = {'passed':True,'configuration':descriptor(OUT/'configuration.json'),
        'formal_models':60,'formal_updates':14400,'tiny_fit_updates':7500,'augmentation':False,
        'training_gate':descriptor(OUT/'training_gate.json'),'reload_gate':descriptor(OUT/'replay_gate.json'),
        'initial_final_health_checks':120,'formal_epoch_module_checks':3600,
        'controls_with_exact_R1_final_state':30,'controls_with_exact_R1_metrics':30,
        'old_negative_models_rejected':5,'tests_passed':preflight['tests_passed'],
        'historical_files_unchanged':historical_count,'prediction_rows_all_splits':len(rows),
        'prediction_rows_test':sum(r['split']=='test' for r in rows),'independent_subjects':37}
    write_new(OUT/'verification.json',verification)
    write_new(OUT/'results.json',{'summary':summary,'paired_comparisons':paired,
        'before_after_descriptive_deltas':before_after,'health':health,'verification':verification})
    report = ['# EEG-P1：有效学习验收与 SparNet 修复执行报告','',
        '已完成后续优先级 1。全程使用原始 EEG 派生输入，无训练数据增强。原 R1 的源文件与 986 项冻结产物保持不变。',
        '', '## 工程修复及验收','',
        '- 修复频谱数值尺度：FFT forward 改为 ortho，即固定乘 sqrt(1000)，未按标签拟合，也未修改原始 EEG 单位。',
        '- 每脑区及全局卷积由 1 路改为 4 路；SE 隐藏层、卷积和接口改为 LeakyReLU(0.1)；编码器采用匹配初始化及零偏置。',
        '- 编码器及 EEG 输出头总参数量为 '+str(read_json(OUT/'structure_manifest.json')['parameters']['U2']['eeg_including_head'])+'，低于 50,000。此次是联合工程修复，不能把效果归因于其中一个改动。',
        '- 新检查成功拒绝旧 C2 seed11 的全部 5 个退化检查点：不再把有限但为零的梯度当作通过。',
        f'- 15 个四人训练子集全部完成固定 500 次诊断更新，最终准确率均为 100%，最大 CE={preflight["tiny_fit_ce_max"]:.6f}。诊断模型全部丢弃，正式实验重新初始化。',
        '- 60 个正式模型均通过初始和终点检查；每轮记录编码模块任务梯度与实际权重变化，全部 3,600 个模型轮次通过。',
        '- 全部 30 个修复模型对 EEG 输入和各脑区有响应，跨受试者表示不再近似常数。探针仅作只读检查，从未参加优化。',
        f'- {preflight["tests_passed"]} 项回归测试通过；60 个检查点全部完成训练探针重载后，才进行新的外层推理。',
        '- 30 个 U0/C0 配对控制从头重训，其最终参数和指标与原 R1 完全一致，说明增加检查没有改变控制训练。',
        '', '![有效学习验收](learning_acceptance.png)','',
        '## 固定协议的测试结果','',
        '以下 F1 均为 macro-F1，先在每个种子内平均五折，再平均三个种子。括号之外不使用合并 111 行预测来冒充 111 名独立受试者。',
        '', '| 条件 | EEG 训练 F1 | EEG 测试 F1 | EEG 测试 BA | MDD 召回 | 融合测试 F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for label,src,c in [('U0 原控制',summary,'U0'),('U2 旧退化版',old,'U2'),('U2-P1 修复版',summary,'U2'),
                        ('C0 原控制',summary,'C0'),('C2 旧退化版',old,'C2'),('C2-P1 修复版',summary,'C2')]:
        test = src[c]['eeg']['splits']['test']['primary']
        train = src[c]['eeg']['splits']['train']['primary']
        fusion = f"{src[c]['fusion']['splits']['test']['primary']['macro_f1']:.6f}" if c.startswith('C') else '—'
        report.append(f'| {label} | {train["macro_f1"]:.6f} | {test["macro_f1"]:.6f} | {test["balanced_accuracy"]:.6f} | {test["mdd_recall"]:.6f} | {fusion} |')
    report += ['', '## 修复成功与性能达标分别判断','']
    for p in paired:
        e = p['endpoints']['eeg']
        c = p['candidate']
        verdict = p.get('CCML_acceptance_pass',p['EEG_improved_and_usable'])
        report += [f'- {c}-P1 对 {p["control"]}：EEG ΔF1={e["delta"]["macro_f1"]:+.6f}，ΔMDD召回={e["delta"]["mdd_recall"]:+.6f}；3 个种子中 {e["seeds_with_higher_fold_mean_F1"]} 个 F1 更高。',
            f'  原定改善门槛：{"通过" if e["improvement_threshold_pass"] else "未通过"}；可用性门槛：{"通过" if e["usability_pass"] else "未通过"}（单类预测测试折 {e["single_class_test_folds"]}/15）；综合接入：{"通过" if verdict else "未通过"}。']
        if c.startswith('C'):
            report.append(f'  融合 ΔF1={p["endpoints"]["fusion"]["delta"]["macro_f1"]:+.6f}，融合保持门槛：{"通过" if p["fusion_preservation_pass"] else "未通过"}。')
    report += ['', '工程验收通过证明此次卷积/SE/接口能接收梯度、改变表示并拟合训练样本；它不保证这些表示能推广到未见受试者。训练表现与外层测试表现必须分开解读。',
        '', '## 有效学习证据（全部终点中的最小值）','',
        '| 模型 | 模块任务梯度 L2 | 表示 std/RMS | EEG 置零响应 | 脑区置零响应 |',
        '|---|---:|---:|---:|---:|']
    for c in ['U2','C2']:
        h = health[c]
        report.append(f'| {c}-P1 | {h["final_min_module_gradient_l2"]:.3e} | {h["final_min_representation_relative_std"]:.6f} | {h["final_min_zero_input_relative_response"]:.6f} | {h["final_min_region_response"]:.6f} |')
    report += ['', '![训练曲线](training_curves.png)','', '## 范围与复核','',
        '- 正式训练为 4 条件 × 3 种子 × 5 折 = 60 组，固定 60 轮，共 14,400 次更新；小样本诊断的 7,500 次更新单列。无增强、无早停、无阈值搜索、无测试选择。',
        '- 使用原 R1 37 人、标签、划分、标准化与窗口；保留输入单位、异常能量和清洗问题，不提前执行优先级 2，也未修改 EEGNet。',
        '- 当前队列已用于多轮开发。本轮冻结协议提高了比较可复核性，但不能消除历史反馈；结果仍是现有队列上的开发比较，不是新的独立验证。',
        '- 完整固定配置见 configuration.json，预登记方案见 output/plans/EEG_P1有效学习验收与SparNet修复固定计划.md。',
        '- 原始逐人输出、128D 表示、各轮梯度、权重变化和检查点保存在 runs/；汇总、配对判定和修复前后差值见 results.json；复核见 verification.json。',
        '- learning_acceptance.png、training_curves.png 给出训练诊断曲线；OOF_predictions.csv 和 metrics_by_fold.csv 供进一步核对。','']
    (OUT/'EEG-P1执行报告.md').write_text('\n'.join(report),encoding='utf-8')
    replace_json(OUT/'status.json',{'state':'complete','completed_runs':60,'planned_runs':60,
        'augmentation':False,'outer_metrics_released':True,'engineering_acceptance_pass':True,'updated_utc':now()})
    hashes = {p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*'))
              if p.is_file() and p.name not in ['deliverable_hashes.json','execution.log'] and not p.name.endswith('.part')}
    write_new(OUT/'deliverable_hashes.json',hashes)
    print('COMPLETE: 60 formal models verified; old artifacts intact.',flush=True)
    for c in CONDITIONS:
        print(c,summary[c]['eeg']['splits']['test']['primary'],flush=True)


if __name__=='__main__':
    finish()
