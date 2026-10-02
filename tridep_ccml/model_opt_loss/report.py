from .core import *
from tridep_ccml.model_opt_v1.report import comparison

def mechanism_summary():
    rows=[]
    for seed in SEEDS:
        for fold in FOLDS:
            for mods in ['ST','EST']:
                base=read(OUT/f'mechanism/{mods}/seed_{seed}/fold_{fold}/baseline.json')
                new=read(stem('M2-mean-'+mods,seed,fold)/'mechanism_final.json')
                for stage,condition,p in [('initial','M2-sum-'+mods,base['initial']),
                    ('final','M2-sum-'+mods,base['sum_final']),('final','M2-mean-'+mods,new)]:
                    key='single_mean_to_fusion' if '-mean-' in condition else 'single_sum_to_fusion'
                    for module,d in p['modules'].items():
                        rows.append({'stage':stage,'condition':condition,'seed':seed,'fold':fold,'module':module,
                            'single_fusion_norm_ratio':d[key],'cosine':d['single_fusion_cosine']})
    summary={}
    for stage in ['initial','final']:
        for condition in CONDITIONS:
            selected=[r for r in rows if r['stage']==stage and r['condition']==condition]
            if not selected:continue
            result={}
            for module in sorted({r['module'] for r in selected}):
                rr=[r for r in selected if r['module']==module]
                ratios=[r['single_fusion_norm_ratio'] for r in rr if r['single_fusion_norm_ratio'] is not None]
                cos=[r['cosine'] for r in rr if r['cosine'] is not None]
                result[module]={'n':len(rr),'median_single_fusion_norm_ratio':float(np.median(ratios)) if ratios else None,
                    'median_cosine':float(np.median(cos)) if cos else None,'negative_cosine_fraction':sum(v<0 for v in cos)/len(cos) if cos else None}
            summary[stage+'/'+condition]=result
    return {'probes':90,'same_module_loss_gradients':True,'training_only':True,'optimizer_updates':0,'summary':summary,'rows':rows}

def pairing_check():
    count=0
    for seed in SEEDS:
        for fold in FOLDS:
            for mods in ['ST','EST']:
                a=read(stem('M2-sum-'+mods,seed,fold)/'complete.json');b=read(stem('M2-mean-'+mods,seed,fold)/'complete.json')
                assert a['initial_state_sha256']==b['initial_state_sha256']
                assert a['fit_ids']==b['fit_ids'] and a['parameter_count']==b['parameter_count']
                assert a['updates']==b['updates']==240 and b['reload_exact'] and b['paired_rng_exact']
                count+=1
    return {'passed':True,'pairs':count,'new_fits':30,'new_updates':7200,'reused_controls':30,
        'initialization_rng_parameters_fit_ids_budget_paired':True}

def figures(metrics,mechanism):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    labels=['Sum ST','Mean ST','Sum EST','Mean EST']
    for ax,k,title in zip(axes,['macro_f1','balanced_accuracy','nll'],['Macro-F1','Balanced accuracy','NLL (lower is better)']):
        ys=[metrics[n]['test']['mean'][k] for n in CONDITIONS]
        ax.bar(labels,ys,color=['#64748b','#2563eb','#64748b','#0d9488'])
        for i,y in enumerate(ys):ax.text(i,y+.012,f'{y:.3f}',ha='center',fontsize=9)
        ax.set_ylim(0,1 if k!='nll' else max(ys)*1.25);ax.set_title(title);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    fig.savefig(REPORT/'loss_balance_results.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for condition in CONDITIONS:
        values=[mechanism['summary']['final/'+condition][m]['median_cosine'] for m in ['speech_projection','text_projection']]
        ax.plot(['Speech projection','Text projection'],values,marker='o',label=condition.replace('M2-',''))
    ax.axhline(0,color='#888',linestyle='--');ax.set_ylim(-1,1);ax.set_ylabel('Median cosine: single-view vs fusion-block gradient')
    ax.set_title('Same module; fixed training batch; eval mode');ax.legend()
    fig.savefig(REPORT/'loss_block_gradients.png',dpi=180);plt.close(fig)

def main():
    lock_check();v=read(OUT/'evaluation_complete.json');assert v['passed'];m=read(OUT/'metrics.json')
    pairs={'loss_ST':('M2-mean-ST','M2-sum-ST'),'loss_EST':('M2-mean-EST','M2-sum-EST'),
        'EEG_sum':('M2-sum-EST','M2-sum-ST'),'EEG_mean':('M2-mean-EST','M2-mean-ST')}
    comparisons={k:comparison(m,a,b) for k,(a,b) in pairs.items()}
    interaction={k:comparisons['EEG_mean']['delta'][k]-comparisons['EEG_sum']['delta'][k] for k in METRICS}
    mechanism=mechanism_summary();pairing=pairing_check()
    save(OUT/'comparisons.json',comparisons,immutable=True);save(OUT/'interaction.json',interaction,immutable=True)
    save(OUT/'mechanism_summary.json',mechanism,immutable=True);save(OUT/'pairing_verification.json',pairing,immutable=True)
    allrows=read(OUT/'all_predictions.json');changes={}
    for name,(a,b) in pairs.items():
        ar={(r['seed'],r['subject_id']):r for r in allrows if r['condition']==a and r['split']=='test'}
        br={(r['seed'],r['subject_id']):r for r in allrows if r['condition']==b and r['split']=='test'}
        assert ar.keys()==br.keys() and len(ar)==111
        rows=[]
        for key in sorted(ar):
            aa,bb=ar[key],br[key];ca=aa['prediction']==aa['label'];cb=bb['prediction']==bb['label']
            rows.append({'seed':key[0],'subject_id':key[1],'label':aa['label'],'candidate_p':aa['p'],'control_p':bb['p'],
                'wrong_to_right':bool(ca and not cb),'right_to_wrong':bool(cb and not ca)})
        changes[name]={'rows':rows,'wrong_to_right':sum(r['wrong_to_right'] for r in rows),'right_to_wrong':sum(r['right_to_wrong'] for r in rows),
            'unique_subjects':37,'records_are_repeated_seeds':True}
    save(OUT/'paired_subject_changes.json',changes,immutable=True)
    figures(m,mechanism)
    passes=[k for k in ['loss_ST','loss_EST'] if comparisons[k]['classification_pass']]
    lines=['# CCML损失块平衡阶段完成报告','',
        ('本阶段完成。损失块均值方案通过分类开发门槛的范围为：'+ '、'.join(passes)+'。' if passes else '本阶段完成。单分支损失由求和改为均值，在 ST 与 EST 上均未通过预设分类改善门槛，保留原求和目标。'),
        '继续跳过数据缺陷排查，以数据无缺陷为工作假设。只调整单分支损失块的模态聚合，一致与互补损失、模型、输入、优化器、裁剪和推理均与控制一致。','',
        '## 执行与验收','',
        '完成 30 组新训练、7,200 次更新，复用首批 30 个完全同配置的冻结控制。另完成 90 个训练内损失块梯度探针，优化器更新为零。30 对初值、随机状态、参数量、训练名单及预算一致；60 个模型精确重载，2,220 条逐人输出保存，1,110 条控制输出与首批精确一致，180 组折内指标独立复算通过。',
        '所有新增模型完成并锁定后才统一生成本轮新测试结果。没有使用 val 选轮、没有超参数或阈值搜索。','',
        '## 同版本结果','',
        '| 条件 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |','|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:lines.append('| '+c+' | '+' | '.join(f'{m[c]["test"]["mean"][k]:.6f}' for k in METRICS)+' |')
    lines+=['','主口径为每种子五折均值，再对 3 种子取均值；每折和合并 OOF 指标单独保存在 metrics.json。','',
        '| 比较 | F1变化（百分点） | 改善种子 | BA变化 | 召回变化 | 分类门槛 | 概率门槛 |',
        '|---|---:|---:|---:|---:|---|---|']
    for k,c in comparisons.items():lines.append(f'| {k} | {100*c["delta"]["macro_f1"]:+.4f} | {c["improved_units"]}/3 | {c["delta"]["balanced_accuracy"]:+.6f} | {c["delta"]["mdd_recall"]:+.6f} | '+('通过' if c['classification_pass'] else '未通过')+' | '+('通过' if c['probability_pass'] else '未通过')+' |')
    lines+=['',f'EEG 增量的差中差为 {100*interaction["macro_f1"]:+.4f} 个百分点；这是描述量，不等于显著性或独立因果证明。',
        '',f'![分类及概率结果](<{(REPORT/"loss_balance_results.png").as_posix()}>)','',
        '## 损失块机制诊断','',
        '下表比较各末轮模型在固定 train 小批次上的梯度。范数比为该模型实际使用的单分支损失块梯度与一致加互补梯度之比；余弦在同一个模块参数上计算，负值仅说明该次局部方向相反。eval 模式不含 dropout 随机性，不能代表完整训练轨迹或直接证明泛化原因。','',
        '| 条件 | 模块 | 梯度范数比中位数 | 余弦中位数 | 负余弦比例 |','|---|---|---:|---:|---:|']
    for c in CONDITIONS:
        for module in ['speech_projection','text_projection']:
            d=mechanism['summary']['final/'+c][module]
            lines.append(f'| {c} | {module} | {d["median_single_fusion_norm_ratio"]:.6f} | {d["median_cosine"]:.6f} | {d["negative_cosine_fraction"]:.4f} |')
    lines+=['',f'![同模块梯度关系](<{(REPORT/"loss_block_gradients.png").as_posix()}>)','',
        '## 停止规则与后续范围','',
        '分类通过须 F1 提高至少 1 个百分点、至少 2/3 种子改善、BA 不降、MDD召回下降不超过 0.05。概率改善单列，不以它替代分类门槛。',
        '本轮不追加损失系数搜索；第一批没有通过门槛的改动，因此不组合。有限骨干微调未执行，仍需单独登记分支、可训练范围、预算与折内特征生成规则。',
        '全部证据仍来自已反复用于开发的 37 人；多个种子不是新增受试者，也不是独立人群验证。','',
        '## 产物入口','',
        f'- [固定执行附表](<{APPENDIX.as_posix()}>)',
        f'- [完整指标](<{(OUT/"metrics.json").as_posix()}>)',
        f'- [逐项门槛判断](<{(OUT/"comparisons.json").as_posix()}>)',
        f'- [逐人配对变化](<{(OUT/"paired_subject_changes.json").as_posix()}>)',
        f'- [训练内梯度诊断](<{(OUT/"mechanism_summary.json").as_posix()}>)']
    target=REPORT/'CCML损失块平衡阶段完成报告.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'status.json',{'stage':'complete','new_models_completed':30,'new_updates_completed':7200,
        'reused_controls':30,'mechanism_probes':90,'new_outer_results_released':True,'data_defect_audit':'skipped_by_user',
        'classification_passes':passes,'finetuning_executed':False,'report':desc(target),'updated_utc':now()})
    save(OUT/'deliverable_hashes.json',{p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name!='deliverable_hashes.json'},immutable=True)
    print({k:{'f1_delta':c['delta']['macro_f1'],'classification_pass':c['classification_pass'],'probability_pass':c['probability_pass']} for k,c in comparisons.items()},flush=True)

if __name__=='__main__':main()
