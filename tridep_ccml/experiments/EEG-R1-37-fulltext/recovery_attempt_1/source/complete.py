"""Verify all 120 fits, then publish descriptive comparisons and artifacts."""
from __future__ import annotations
import csv
from .common import *
from .evaluate import evaluate_all, summarize, comparisons, verify_probability_row, METRICS


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    context = RecordContext()
    rows, runs = [], []
    for c in CONDITIONS:
        for seed in SEEDS:
            for fold in FOLDS:
                folder = run_dir(c,seed,fold)
                verified = read_json(folder/'verified.json')
                require(verified['passed'],'Unverified run')
                for k in ['predictions','features','checkpoint']:
                    verify_descriptor(verified[k])
                predictions = read_json(folder/'predictions.json')
                require(len(predictions)==37 and [r['subject_id'] for r in predictions]==context.ids,'Incomplete subject predictions')
                for r in predictions:
                    require(r['condition']==c and r['seed']==seed and r['fold']==fold and
                            r['label']==context.labels[r['subject_id']] and
                            r['subject_id'] in context.folds[f'fold_{fold}'][r['split']] and
                            r['train_ids']==context.folds[f'fold_{fold}']['train'], 'Prediction identity mismatch')
                    verify_probability_row(r)
                rows.extend(predictions)
                runs.append(read_json(folder/'trained.json'))
    return rows,runs


def tables(summary,rows,paired):
    folds,seeds,oof,changes = [],[],[],[]
    for c,endpoints in summary.items():
        for endpoint,s in endpoints.items():
            for split,v in s['splits'].items():
                for seed,fs in v['folds'].items():
                    for fold,m in fs.items():
                        folds.append({'condition':c,'endpoint':endpoint,'split':split,'seed':seed,'fold':fold,
                            **{k:m[k] for k in METRICS},'n_subjects':m['n_subjects'],'predicted_HC':m['predicted_HC'],
                            'predicted_MDD':m['predicted_MDD'],'single_class_predictions':m['single_class_predictions']})
                for seed,m in v['seed_fold_means'].items():
                    seeds.append({'condition':c,'endpoint':endpoint,'split':split,'seed':seed,**m})
    for r in rows:
        if r['split']=='test':
            for endpoint,p in r['probabilities'].items():
                oof.append({'condition':r['condition'],'endpoint':endpoint,'seed':r['seed'],'fold':r['fold'],
                    'subject_id':r['subject_id'],'label':r['label'],'p_HC':p[0],'p_MDD':p[1],
                    'prediction':r['predictions'][endpoint]})
    for pair in paired:
        for endpoint,value in pair['endpoints'].items():
            for row in value['classification_changes']:
                changes.append({'candidate':pair['candidate'],'control':pair['control'],'endpoint':endpoint,**row})
    for filename,items in [('metrics_by_fold.csv',folds),('metrics_by_seed.csv',seeds),
                           ('OOF_predictions.csv',oof),('classification_changes.csv',changes)]:
        if items:
            write_csv(OUT/filename,items)


def curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    for c,ax in zip(CONDITIONS,axes.flat):
        lines = np.array([[r['loss'] for r in read_json(run_dir(c,s,f)/'history.json')] for s in SEEDS for f in FOLDS])
        x = np.arange(1,61)
        ax.plot(x,lines.mean(0),color='#2563a6',linewidth=2)
        ax.fill_between(x,lines.min(0),lines.max(0),color='#2563a6',alpha=.15,label='range over 15 runs')
        ax.set_title(c+' / '+('EEG-only CE' if c.startswith('U') else 'full CCML'))
        ax.set_xlabel('Epoch'); ax.set_ylabel('Subject loss'); ax.grid(alpha=.2)
    fig.suptitle('EEG-R1: original EEG, no data augmentation; fixed epoch 60')
    fig.savefig(OUT/'training_curves.png',dpi=160)
    fig.savefig(OUT/'training_curves.svg')
    plt.close(fig)


def report(summary,paired,runs,verification,selected=None):
    selected = selected or CONDITIONS
    names = {'U0':'CBraMod + 原 CNN，独立 CE','U1':'EEGNet 改造，独立 CE','U2':'SparNet-inspired，独立 CE',
             'C0':'S-control，完整 CCML','C1':'EEGNet 改造，完整 CCML','C2':'SparNet-inspired，完整 CCML',
             'C3':'CBraMod token 均值，完整 CCML','C4':'CBraMod+MHAP，完整 CCML'}
    success_u = [p['candidate'] for p in paired if p['direct_predeclared_test'] and p['candidate'].startswith('U') and p['EEG_improved_and_usable']]
    success_c = [p['candidate'] for p in paired if p['direct_predeclared_test'] and p['candidate'].startswith('C') and p.get('CCML_acceptance_pass')]
    lines = ['# EEG-R1 原始 EEG 分支替换执行报告','',
        '已完成 8 条件 × 3 种子 × 5 折，共 120 组正式训练、28,800 次优化更新；全部固定第 60 轮。',
        '**未使用 EEG 数据增强。** 使用原始 staged MAT，经既定通道修订、重采样、滤波和分窗；没有抖动、重采样受试者、合成窗口或增强缓存。', '',
        f'按预登记改善与可用性门槛，U 线通过：{", ".join(success_u) or "无"}；完整 CCML 接入通过：{", ".join(success_c) or "无"}。',
        '这里是固定开发比较；37 人已多次用于开发，不能视为新的独立验证队列。','',
        '## 测试集主统计','',
        '每个种子先取五折指标均值，再对三个种子取均值。U 的 EEG 是独立 CE；C 的 EEG 是联合训练分支输出。', '',
        '| 条件 | 路径 | EEG Macro-F1 | EEG BA | EEG MDD recall | 融合 Macro-F1 | 融合 BA | 融合 MDD recall |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in selected:
        a = summary[c]['eeg']['splits']['test']['primary']
        f = summary[c].get('fusion',{}).get('splits',{}).get('test',{}).get('primary')
        vals = [f'{a[k]:.6f}' for k in ['macro_f1','balanced_accuracy','mdd_recall']]
        vals += [f'{f[k]:.6f}' for k in ['macro_f1','balanced_accuracy','mdd_recall']] if f else ['—']*3
        lines.append('| '+ ' | '.join([c,names[c]]+vals)+' |')
    lines += ['', '## 配对差值与判定','',
        'EEG 改善要求 ΔF1≥0.03、至少 2/3 种子提高、MDD recall 降幅≤0.05；融合容许 ΔF1≥−0.01、Δrecall≥−0.05。可用性要求 BA>0.5、recall≥0.5，并保守地要求无单类预测测试折。该塌缩解释在训练前写入配置，未按结果改变。','',
        '| 候选/控制 | 终点 | ΔF1 | ΔMDD recall | ΔNLL | ΔBrier | F1 提高种子 | 单类测试折/15 | 改善阈值 | 可用性 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for p in paired:
        if p['candidate'] not in selected:
            continue
        for endpoint,v in p['endpoints'].items():
            d = v['delta']
            lines.append(f'| {p["candidate"]}/{p["control"]} | {endpoint} | {d["macro_f1"]:+.6f} | {d["mdd_recall"]:+.6f} | {d["nll"]:+.6f} | {d["brier"]:+.6f} | {v["seeds_with_higher_fold_mean_F1"]}/3 | {v["single_class_test_folds"]} | {"通过" if v["improvement_threshold_pass"] else "未通过"} | {"通过" if v["usability_pass"] else "未通过"} |')
    lines += ['', 'NLL/Brier 越低越好；融合行的“改善阈值”仅是附加描述，接入判断使用上文融合保留门槛。C4 的注意力直接对照为 C3。C3/C0、C4/C0 和 U2/U1、C2/C1 仅作额外完整比较。','',
        '## 每种子五折均值与合并 OOF','',
        '| 条件/终点 | seed 11 F1 | seed 22 F1 | seed 33 F1 | OOF 每种子指标均值 F1 | 训练−测试 F1 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in selected:
        for endpoint in ['eeg']+(['fusion'] if c.startswith('C') else []):
            v = summary[c][endpoint]
            s = v['splits']['test']['seed_fold_means']
            lines.append(f'| {c}/{endpoint} | '+ ' | '.join(f'{s[str(seed)]["macro_f1"]:.6f}' for seed in SEEDS)+f' | {v["OOF_mean_over_seed_metrics"]["macro_f1"]:.6f} | {v["train_minus_test"]["macro_f1"]:+.6f} |')
    lines += ['', '逐折、val 描述、语音/文本分支、各类预测数、混淆矩阵与全部指标保存在 results.json 和 CSV。111 条 OOF 预测仅为三个种子的重复记录，独立受试者仍是 37 人。未计算伪独立行 bootstrap。','',
        '## 数据、结构与实现边界','',
        '- 原始派生缓存为每人 60×19×1000 float32；token 为 60×19×5×200。37 人均通过标签、来源哈希、形状、有限值、展平重建与 token 均值复现检查（rtol=atol=1e-4）。',
        '- EEGNet/SparNet 按训练折全部等长窗口计算每导均值/标准差；只在训练折拟合，再应用到 val/test。标准化属于新增处理，收益不能全部归因于网络结构。CBraMod 路线保留 native 尺度。',
        '- EEGNet 为 F1=4、D=2、F2=8，保持 60 窗完整批次与正常 train/eval。BN 不使用测试信息；最大范数在优化更新后施加。',
        '- SparNet 使用 1–40 Hz、0.2 Hz 分辨率的单边 FFT 振幅，五脑区 SE/卷积。以论文层表的池化 stride=2 为准；19 导分组、5 秒窗、47→128 接口均为本地改造。FCz/CPz 明确归入 central 过渡组；没有冒充 Fz/Cz，也没有使用存在冲突的 GSN129 坐标。厂家对应与区域边界均为近似。',
        '- C3/C4 共享 200→32 token 投影及 32→128 接口初始化。C4 的两个 16 维头只在每窗 95 个位置上注意力汇聚，随后与均值按 0.5/0.5 合并，再平均 60 窗；不宣称完整 STAMP 复现。',
        '- C 线复用完整 CCML 数学实现，保持 rho=1.25、delta=gamma=1、KL 退火 50 轮，无最终融合额外监督、门控或尺度搜索。语音与全文文本沿用 S-control 来源、投影及证据头；配对初始化和受试者批次逐项核对。',
        '- 全部监督 128 维特征按条件/seed/fold 单独保存；不跨折共享。CBraMod/HuBERT/MacBERT 冻结为无 MODMA 标签拟合的缓存；历史监督模型未被加载。',
        '- 首轮与后续候选全部在外层分数释放前冻结。本次进一步推迟到全部 120 组训练结束才进行外层推理和汇总，未由测试成绩决定候选、阈值或追加种子。','',
        '## 验证与资源','',
        f'- {verification["contract_tests"]} 项工程测试通过；120 个检查点均按保存前的训练探针重新加载复现；所有推理保持权重及 BN 统计不变。',
        f'- 4,440 条逐人记录（含 train/val/test）均通过身份、划分、独立概率/证据复算；五项指标独立复算误差≤1e-12。',
        f'- 历史文件核验：{verification["protected_history_files"]} 个文件未变。',
        f'- 正式训练累计 {sum(r["seconds"] for r in runs)/60:.2f} 分钟；实测最大已分配显存 {max(r["peak_allocated_bytes"] for r in runs)/2**20:.1f} MiB。导出、烟雾检查、评估和核验另计；无正式训练失败重跑。',
        '- 设备为 RTX 4080 Laptop GPU；float32、确定性算法，未开启混合精度或 TF32。每组 240 次更新。',
        '- 工程阶段初版测量与补充检查点探针后的复验均留档；这些随机小型检查模型未进入正式结果。','',
        '## 交付文件','',
        '- configuration.json / structure_manifest.json：固定协议、真实参数量、脑区/频谱/注意力清单、来源哈希。',
        '- data_manifest.json / cache/standardizers/：37 人原始输入与五折标准化来源。',
        '- runs/：120 组检查点、逐轮损失、逐人批次顺序、逐人概率/证据、128 维特征和重载核验。',
        '- results.json、metrics_by_fold.csv、metrics_by_seed.csv、OOF_predictions.csv、classification_changes.csv：完整统计及分类变化。',
        '- training_curves.png / training_curves.svg：15 次运行训练损失均值与范围。',
        '- verification.json：最终完成检查。','',
        '## 历史参考与结论限制','',
        '历史原 EEG Macro-F1=0.449401、修订 EEG=0.411825、S-control 融合=0.790859 只按 R1 计划作为历史参考；本次直接对照是重新训练的 U0/C0，不能混作同协议成绩。',
        '本轮不因外层最高分自动选择模型。若无候选通过完整门槛，结论为本轮未找到满足预定要求的可靠替代分支；即使部分指标上升，也需分别解释 U、C 和融合。冻结 CBraMod 的 Mumtaz 原始输入尺度来源尚未完全闭环。','',
        '结构依据：[SparNet 原文](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.914823/full)、[EGI 厂家位置对应表](https://www.egi.com/images/HydroCelGSN_10-10.pdf)，以及本地 R1 计划和通道审计。','']
    return '\n'.join(lines)


def complete():
    cfg = verify_lock()
    evaluate_all()
    rows,runs = collect()
    summary = summarize(rows)
    paired = comparisons(summary,rows)
    history = read_json(OUT/'history_before.json')
    for name,sha in history.items():
        require(digest(WORK/name)==sha,'Historical artifact changed: '+name)
    verification = {'passed':True,'formal_runs':len(runs),'updates':sum(r['updates'] for r in runs),
        'prediction_rows_all_splits':len(rows),'OOF_rows':sum(r['split']=='test' for r in rows),
        'independent_subjects':37,'checkpoint_reloads':120,'independent_metrics':True,
        'metric_recompute_atol':1e-12,'augmentation':False,'protected_history_files':len(history),
        'contract_tests':read_json(OUT/'test_results.json')['tests'],
        'configuration':descriptor(OUT/'configuration.json'),'data_manifest':descriptor(OUT/'data_manifest.json'),
        'formal_retries':0,'all_planned_candidates_finished':True}
    require(verification['formal_runs']==120 and verification['updates']==28800 and len(rows)==4440,'Completion budget mismatch')
    write_new(OUT/'results.json',{'summary':summary,'paired_comparisons':paired,'verification':verification})
    tables(summary,rows,paired)
    curves()
    (OUT/'EEG-R1执行报告.md').write_text(report(summary,paired,runs,verification),encoding='utf-8')
    for i,stage in enumerate(STAGES,1):
        (OUT/f'阶段{i}报告.md').write_text(report(summary,paired,runs,verification,stage),encoding='utf-8')
    write_new(OUT/'verification.json',verification)
    outputs = {p.relative_to(OUT).as_posix():digest(p) for p in OUT.glob('*') if p.is_file() and p.name not in ['status.json','deliverable_hashes.json']}
    replace_json(OUT/'deliverable_hashes.json',outputs)
    replace_json(OUT/'status.json', {'state':'complete','completed_runs':120,'planned_runs':120,'updates':28800,
        'augmentation':False,'outer_metrics_released':True,'verification':descriptor(OUT/'verification.json'),'updated_utc':now()})
    for c in CONDITIONS:
        eeg = summary[c]['eeg']['splits']['test']['primary']
        fused = summary[c].get('fusion',{}).get('splits',{}).get('test',{}).get('primary',{})
        print(f'RESULT {c}: EEG_F1={eeg["macro_f1"]:.6f} EEG_recall={eeg["mdd_recall"]:.6f} fusion_F1={fused.get("macro_f1")}',flush=True)
    print('COMPLETE 120/120; original EEG, no augmentation; checkpoint/metric/history checks passed.',flush=True)


if __name__ == '__main__':
    complete()
