from .core import *

def comparison(results,candidate,control,deterministic=False):
    a,b=results[candidate]['test'],results[control]['test']
    delta={k:a['mean'][k]-b['mean'][k] for k in METRICS}
    if deterministic:
        units=[a['seeds']['None']['folds'][str(f)]['macro_f1']-b['seeds']['None']['folds'][str(f)]['macro_f1'] for f in FOLDS]
        stable=sum(v>0 for v in units)>=3
    else:
        units=[a['seeds'][str(s)]['mean']['macro_f1']-b['seeds'][str(s)]['mean']['macro_f1'] for s in SEEDS]
        stable=sum(v>0 for v in units)>=2
    criteria={'f1_delta_ge_0.01':delta['macro_f1']>=.01,'stability':stable,'BA_non_decreasing':delta['balanced_accuracy']>=0,
        'recall_drop_le_0.05':delta['mdd_recall']>=-.05}
    if deterministic:criteria['candidate_BA_above_chance']=a['mean']['balanced_accuracy']>.5
    return {'candidate':candidate,'control':control,'delta':delta,'paired_unit_f1_deltas':units,
        'improved_units':sum(v>0 for v in units),'unit':'fold' if deterministic else 'seed','criteria':criteria,
        'classification_pass':all(criteria.values()),'probability_pass':delta['nll']<0 and delta['brier']<0}

def diagnostics():
    result={}
    for condition in CONDITIONS:
        accum={};clip_count=0;count=0;max_constraint=0;epoch_losses=[]
        for seed in SEEDS:
            for fold in FOLDS:
                stem=run_dir(condition,seed,fold)
                epoch_losses.append([h['loss'] for h in read(stem/'history.json')])
                for r in read(stem/'updates.json'):
                    count+=1;clip_count+=int(r['global_pre']>1)
                    for g,v in r['modules'].items():
                        a=accum.setdefault(g,{k:[] for k in v})
                        for k,val in v.items():a[k].append(val)
                        max_constraint=max(max_constraint,v['constraint_l2'])
        result[condition]={'steps':count,'pre_norm_above_1_fraction':clip_count/count,
            'max_constraint_displacement':max_constraint,
            'modules':{g:{k:{'mean':float(np.mean(v)),'median':float(np.median(v)),'max':max(v)} for k,v in a.items()} for g,a in accum.items()},
            'mean_epoch_loss':np.mean(epoch_losses,axis=0).tolist()}
    return result

def figures(results,diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True,exist_ok=True)
    names=list(results);labels=[n.replace('M1-','') for n in names]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    x=np.arange(len(names))
    for ax,key,title in zip(axes,['macro_f1','balanced_accuracy'],['Macro-F1 (fold means)','Balanced accuracy (fold means)']):
        ax.bar(x,[results[n]['test']['mean'][key] for n in names],color=['#64748b','#2563eb','#64748b','#2563eb','#64748b','#0d9488','#64748b','#9333ea'])
        ax.set_xticks(x,labels,rotation=40,ha='right');ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.savefig(REPORT/'classification.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for n,d in diag.items():axes[0].plot(range(1,61),d['mean_epoch_loss'],label=n.replace('M1-',''))
    axes[0].set(xlabel='Epoch',ylabel='Mean training loss');axes[0].legend(fontsize=7)
    for condition in CONDITIONS[:4]:
        values=[diag[condition]['modules'][m+'_projection']['relative_update']['mean'] for m in ['speech','text']]
        axes[1].plot(['Speech projection','Text projection'],values,marker='o',label=condition.replace('M1-',''))
    axes[1].set(ylabel='Mean ||Adamax update|| / ||parameter||');axes[1].legend(fontsize=7)
    fig.savefig(REPORT/'training_diagnostics.png',dpi=170);plt.close(fig)

def main():
    lock_check();results=read(OUT/'metrics.json');verification=read(OUT/'evaluation_complete.json')
    pairs=[('clip_ST','M1-branch-ST','M1-global-ST'),('clip_EST','M1-branch-EST','M1-global-EST'),
        ('speech_statistics','M1-stat-ST','M1-mean-ST'),('EEG_global','M1-global-EST','M1-global-ST'),
        ('EEG_branch','M1-branch-EST','M1-branch-ST')]
    comps={k:comparison(results,a,b) for k,a,b in pairs}
    comps['covariance']=comparison(results,'M1-covariance','M1-bandpower',True)
    diag=diagnostics();save(OUT/'comparisons.json',comps,immutable=True);save(OUT/'diagnostics.json',diag,immutable=True)
    figures(results,diag)
    combo=comps['clip_ST']['classification_pass'] and comps['speech_statistics']['classification_pass']
    decisions={'combination_registration_triggered':combo,'EEG_fusion_registration_supported':comps['covariance']['classification_pass'],
        'loss_balance':'not automatically triggered; separate mechanism hypothesis needed',
        'backbone_finetuning':'not automatically triggered; separate mechanism hypothesis needed',
        'independent_population':'requires new data','additional_search':False}
    save(OUT/'decisions.json',decisions,immutable=True)
    lines=['# 模型优化第一批完成报告','',
        '已按用户要求跳过数据缺陷排查，以数据无缺陷为本轮工作假设，完成后续计划 v1 的第一批三个模型候选。该假设没有改记为数据审计结论。原输入、历史代码及冻结实验保留。','',
        '## 执行与核验','',
        f'90 组神经训练、21,600 次更新和 10 次确定性线性拟合全部完成。每组使用原 train 人员，无验证选轮、超参搜索或阈值搜索。全部拟合锁定后统一评价。{verification["reloads_exact"]} 个模型精确重载通过，{verification["rows"]} 条逐人模型输出保存，300 组 train/val/test 指标与独立实现核对通过。30 个 global 控制末权重与冻结 G1 逐位一致。',
        '','## 同版本结果','',
        '| 条件 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():lines.append('| '+name+' | '+' | '.join(f'{r["test"]["mean"][k]:.6f}' for k in METRICS)+' |')
    lines+=['','神经方案主口径为每种子五折均值再三种子均值；两条 EEG 线性基线只有五折，不虚构多种子。完整每折、每种子与合并 OOF 结果见 metrics.json。','',
        '| 配对比较 | F1变化（百分点） | BA变化 | 召回变化 | 改善单位 | 分类门槛 | 概率门槛 |',
        '|---|---:|---:|---:|---|---|---|']
    for name,c in comps.items():lines.append(f'| {name} | {100*c["delta"]["macro_f1"]:+.4f} | {c["delta"]["balanced_accuracy"]:+.6f} | {c["delta"]["mdd_recall"]:+.6f} | {c["improved_units"]}/{len(c["paired_unit_f1_deltas"])} {c["unit"]} | '+('通过' if c['classification_pass'] else '未通过')+' | '+('通过' if c['probability_pass'] else '未通过')+' |')
    lines+=['','![分类表现](classification.png)','','## 实际更新诊断','',
        '所有条件都保留逐步的裁剪前后梯度、Adamax 更新、相对更新及约束位移。下表是语音和文本投影的平均相对更新，不将梯度裁剪比例等同于 Adamax 实际学习率。','',
        '| 条件 | 裁剪前总范数大于1比例 | 语音相对更新 | 文本相对更新 | 最大约束位移 |','|---|---:|---:|---:|---:|']
    for name,d in diag.items():lines.append(f'| {name} | {d["pre_norm_above_1_fraction"]:.4f} | {d["modules"]["speech_projection"]["relative_update"]["mean"]:.8f} | {d["modules"]["text_projection"]["relative_update"]["mean"]:.8f} | {d["max_constraint_displacement"]:.8f} |')
    lines+=['','![训练和参数更新](training_diagnostics.png)','','## 执行结论与边界','',
        '语音统计比较保持形式参数总数相同，但零填充控制的附加列无数据梯度；有效自由度差异仍在。协方差与频带功率比较的是两种完整表示，不是严格只添加某个通道关系的消融。',
        '',('ST 裁剪与统计汇聚均通过，触发另行登记同版本四条件组合。' if combo else 'ST 裁剪与统计汇聚没有同时通过，不触发组合训练。'),
        ('EEG 协方差通过独立开发门槛，可作为后续同版本 ST/EST 登记依据；本批没有证明融合增量。' if decisions['EEG_fusion_registration_supported'] else 'EEG 协方差未通过预定独立开发门槛，不进入该路线的复杂融合。'),
        '损失块平衡和有限骨干微调仍为另行登记的条件项，不因本批未提高自动追加。当前 37 人均已反复参与开发，不将这些结果宣称为独立人群泛化验证。','',
        '## 文件入口','',
        f'- [固定执行附表](<{APPENDIX.as_posix()}>)',
        f'- [各折和各种子指标](<{(OUT/"metrics.json").as_posix()}>)',
        f'- [门槛逐项判定](<{(OUT/"comparisons.json").as_posix()}>)',
        f'- [全部逐人输出](<{(OUT/"all_predictions.json").as_posix()}>)',
        f'- [机制诊断汇总](<{(OUT/"diagnostics.json").as_posix()}>)']
    target=REPORT/'模型优化第一批完成报告.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'status.json',{'stage':'complete','completed_neural':90,'completed_linear':10,'completed_updates':21600,
        'outer_results_released':True,'data_defect_audit':'skipped_by_user','report':desc(target),
        'classification_passes':[k for k,c in comps.items() if c['classification_pass']],
        'decisions':decisions,'updated_utc':now()})
    save(OUT/'deliverable_hashes.json',{p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*'))
         if p.is_file() and p.name!='deliverable_hashes.json'},immutable=True)
    print({k: {'f1_delta':v['delta']['macro_f1'],'pass':v['classification_pass']} for k,v in comps.items()},flush=True)

if __name__=='__main__':main()
