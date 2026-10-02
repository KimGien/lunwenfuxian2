from .core import *
from tridep_ccml.model_opt_v1.report import comparison

def pairing():
    pairs=[]
    for s in SEEDS:
        for f in FOLDS:
            a,b=[read(stem(c,s,f)/'complete.json') for c in CONDITIONS]
            assert a['initial_state_sha256']==b['initial_state_sha256'] and a['fit_ids']==b['fit_ids']
            assert a['frozen_layer_sha256']==b['frozen_layer_sha256']
            assert a['updates']==b['updates']==240 and a['reload_exact'] and b['reload_exact']
            assert a['lora_delta_l2']==0 and b['lora_delta_l2']>0
            ra,rb=[torch.load(stem(c,s,f)/'rng_initial.pt',weights_only=True) for c in CONDITIONS]
            assert all(torch.equal(ra[k],rb[k]) for k in ra)
            pa,pb=[torch.load(stem(c,s,f)/'train_probe.pt',weights_only=True)['before'] for c in CONDITIONS]
            assert all(torch.equal(pa[k],pb[k]) for k in ['logits','features','normalized','question_cls'])
            pairs.append({'seed':s,'fold':f,'initial_state_sha256':a['initial_state_sha256'],
                'frozen_layer_sha256':a['frozen_layer_sha256'],'candidate_lora_delta_l2':b['lora_delta_l2']})
    return {'passed':True,'pairs':pairs,'paired_initial_state_rng_outputs_training_ids_budget':True,
        'all_pretrained_weights_unchanged':True,'all_candidate_adapters_changed':True}

def diagnostics():
    result={}
    for c in CONDITIONS:
        accum={};losses=[];count=0;clipped=0;elapsed=0;deltas=[]
        for s in SEEDS:
            for f in FOLDS:
                path=stem(c,s,f);d=read(path/'complete.json');elapsed+=d['elapsed_seconds'];deltas.append(d['lora_delta_l2'])
                losses.append([r['loss'] for r in read(path/'history.json')])
                for r in read(path/'updates.json'):
                    count+=1;clipped+=int(r['global_grad_pre']>1)
                    for g,v in r['modules'].items():
                        a=accum.setdefault(g,{k:[] for k in v})
                        for k,x in v.items():a[k].append(x)
        assert count==3600
        result[c]={'updates':count,'clip_fraction':clipped/count,'training_seconds':elapsed,
            'adapter_delta_l2':{'min':min(deltas),'median':float(np.median(deltas)),'max':max(deltas)},
            'mean_epoch_loss':np.mean(losses,axis=0).tolist(),
            'modules':{g:{k:{'mean':float(np.mean(v)),'median':float(np.median(v)),'max':max(v)} for k,v in a.items()} for g,a in accum.items()}}
    return result

def figures(m,d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,k,title in zip(axes,['macro_f1','balanced_accuracy','mdd_recall'],['Macro-F1','Balanced accuracy','MDD recall']):
        ys=[m[c]['test']['mean'][k] for c in CONDITIONS]
        ax.bar(['Frozen','LoRA'],ys,color=['#64748b','#2563eb']);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        for i,y in enumerate(ys):ax.text(i,y+.02,f'{y:.4f}',ha='center')
    fig.savefig(REPORT/'lora_classification.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for c,label in zip(CONDITIONS,['Frozen','LoRA']):axes[0].plot(range(1,61),d[c]['mean_epoch_loss'],label=label)
    axes[0].set(xlabel='Epoch',ylabel='Mean training loss');axes[0].legend()
    for ax,k,title in zip(axes[1:],['nll','brier'],['NLL (lower is better)','Brier (lower is better)']):
        ys=[m[c]['test']['mean'][k] for c in CONDITIONS]
        ax.bar(['Frozen','LoRA'],ys,color=['#64748b','#2563eb']);ax.set_ylim(0,max(ys)*1.25);ax.set_title(title)
        for i,y in enumerate(ys):ax.text(i,y+max(ys)*.025,f'{y:.4f}',ha='center')
    fig.savefig(REPORT/'lora_training_probability.png',dpi=180);plt.close(fig)

def main():
    lock_check();ev=read(OUT/'evaluation_complete.json');assert ev['passed']
    m=read(OUT/'metrics.json');b=read(OUT/'branch_metrics.json');co=comparison(m,CONDITIONS[1],CONDITIONS[0])
    pair=pairing();diag=diagnostics()
    save(OUT/'comparisons.json',{'text_lora':co},immutable=True);save(OUT/'pairing_verification.json',pair,immutable=True)
    save(OUT/'training_diagnostics.json',diag,immutable=True)
    rows=read(OUT/'all_predictions.json');maps=[{(r['seed'],r['subject_id']):r for r in rows if r['condition']==c and r['split']=='test'} for c in CONDITIONS]
    assert maps[0].keys()==maps[1].keys() and len(maps[0])==111
    changes=[]
    for key in sorted(maps[0]):
        a,bb=maps[0][key],maps[1][key];ca=a['prediction']==a['label'];cb=bb['prediction']==bb['label']
        changes.append({'seed':key[0],'subject_id':key[1],'label':a['label'],'control_p':a['p'],'candidate_p':bb['p'],
            'wrong_to_right':bool(cb and not ca),'right_to_wrong':bool(ca and not cb)})
    change={'rows':changes,'wrong_to_right':sum(r['wrong_to_right'] for r in changes),
        'right_to_wrong':sum(r['right_to_wrong'] for r in changes),'independent_subjects':37,'records_are_repeated_seeds':True}
    save(OUT/'paired_subject_changes.json',change,immutable=True);figures(m,diag)
    accepted=co['classification_pass'];candidate='M3-lora-ST' if accepted else 'M2-mean-ST'
    conclusion=('有限文本微调通过预设分类开发门槛，保留为新的 ST 开发候选。' if accepted else
        '有限文本微调未通过预设分类开发门槛，保留第二批 ST 均值损失候选，不替换。')
    lines=['# MacBERT有限微调阶段完成报告','',conclusion,
        '继续按用户指示跳过数据缺陷排查，以数据无缺陷为工作假设。本轮仅适配文本 MacBERT 最后一层的 query/value；语音输入与 ST 均值损失沿用第二批。','',
        '## 执行与验收','',
        '冻结控制与 LoRA 候选各 15 组，共完成 30 个新模型、7,200 次更新。每组固定 60 轮、batch 8；rank=4、alpha=4、LoRA lr=1e-5、无 dropout，新增 12,288 个可训练参数。原 MacBERT 参数完全冻结。',
        '4 项训练前工程测试通过；15 对初值、随机状态、初始输出、训练名单和预算匹配。所有候选的 LoRA 均实际更新，所有原末层权重保持原哈希。30 个模型与 1,110 条逐人输出精确重载，融合 90 组折内指标及全部端点 360 组折内指标独立复算通过。',
        '全部拟合锁定后统一生成结果，并独立导出 30 套绑定检查点和 train 名单的文本题向量及 pooled 特征。共享缓存仅为无本队列标签监督的前 11 层输出；监督适配特征没有跨折共享。无验证选轮、参数或阈值搜索。','',
        '## 同版本配对结果','',
        '| 条件 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |','|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:lines.append('| '+c+' | '+' | '.join(f'{m[c]["test"]["mean"][k]:.6f}' for k in METRICS)+' |')
    lines+=['','主口径为每种子五折均值，再对 3 种子取均值。每折、每种子及合并 OOF 指标保存在完整指标中。',
        f'LoRA 相对同版本冻结控制 F1 变化 {100*co["delta"]["macro_f1"]:+.4f} 个百分点，{co["improved_units"]}/3 种子改善；BA 变化 {100*co["delta"]["balanced_accuracy"]:+.4f} 个百分点，召回变化 {100*co["delta"]["mdd_recall"]:+.4f} 个百分点。概率门槛'+('通过。' if co['probability_pass'] else '未通过。'),'',
        '| 种子 | 冻结 F1 | LoRA F1 | 变化（百分点） |','|---|---:|---:|---:|']
    for s,delta in zip(SEEDS,co['paired_unit_f1_deltas']):lines.append(f'| {s} | {m[CONDITIONS[0]]["test"]["seeds"][str(s)]["mean"]["macro_f1"]:.6f} | {m[CONDITIONS[1]]["test"]["seeds"][str(s)]["mean"]["macro_f1"]:.6f} | {100*delta:+.4f} |')
    lines+=['',f'![分类结果](<{(REPORT/"lora_classification.png").as_posix()}>)','',
        '## 分支、训练与数值记录','',
        '| 条件 | train F1 | val F1 | test F1 | 语音分支 test F1 | 文本分支 test F1 |','|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:lines.append('| '+c+' | '+' | '.join(f'{x:.6f}' for x in [m[c][sp]['mean']['macro_f1'] for sp in ['train','val','test']]+[b[c][k]['test']['mean']['macro_f1'] for k in ['speech','text']])+' |')
    ld=diag[CONDITIONS[1]]
    lines+=['',f'15 个候选的 LoRA 相对初值位移 L2 中位数为 {ld["adapter_delta_l2"]["median"]:.6f}，范围 [{ld["adapter_delta_l2"]["min"]:.6f}, {ld["adapter_delta_l2"]["max"]:.6f}]。候选全局梯度裁剪比例 {ld["clip_fraction"]:.4%}。这是更新确实发生的机制记录，不能直接证明泛化原因。',
        f'111 条重复种子测试记录中，错变对 {change["wrong_to_right"]}、对变错 {change["right_to_wrong"]}；这些记录来自同一 37 人。',
        f'缓存重放相对原完整模型题 CLS 最大绝对误差 {read(OUT/"cache_replay.json")["max_abs_difference"]:.8g}，低于登记容差；同一计算路径重复和重载逐位一致。由于批次长度和 float32 计算路径不同，以本轮新冻结控制判定 LoRA 效应，第二批 0.781599 仅作历史开发参考。','',
        f'![训练与概率结果](<{(REPORT/"lora_training_probability.png").as_posix()}>)','',
        '## 判定与停止','',
        '分类通过须 F1 提高至少 1 个百分点、至少 2/3 种子改善、BA 不降、召回下降不超过 0.05；NLL/Brier 同时降低才记概率改善。各项判定见 comparisons.json。',
        '本轮预算已经结束，不自动增加 rank、学习率、微调层数或改换 HuBERT；没有新增 EEG 或组合训练。后续计划 v1 的固定候选筛查已完成，只有通过门槛的开发候选保留。',
        '三批累计 150 个新神经模型、36,000 次更新，以及 10 次线性拟合；复用控制、零更新探针单列。37 人已反复用于开发，不能据此宣称独立人群泛化或临床验证。','',
        '## 产物入口','',f'- [固定执行附表](<{APPENDIX.as_posix()}>)',
        f'- [完整融合指标](<{(OUT/"metrics.json").as_posix()}>)',f'- [分支指标](<{(OUT/"branch_metrics.json").as_posix()}>)',
        f'- [门槛判定](<{(OUT/"comparisons.json").as_posix()}>)',f'- [特征与训练名单](<{(OUT/"feature_manifest.json").as_posix()}>)',
        f'- [逐人配对变化](<{(OUT/"paired_subject_changes.json").as_posix()}>)',f'- [训练更新诊断](<{(OUT/"training_diagnostics.json").as_posix()}>)']
    target=REPORT/'MacBERT有限微调阶段完成报告.md';target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(REPORT/'下一阶段交接.json',{'stage_completed':'MacBERT limited LoRA','current_development_candidate':candidate,
        'classification_pass':accepted,'probability_pass':co['probability_pass'],'fixed_v1_candidates_finished':True,
        'new_models':30,'new_updates':7200,'v1_total_neural_fits':150,'v1_total_linear_fits':10,'v1_total_updates':36000,
        'data_defect_audit':'skipped_by_user','independent_validation':False,'registered_remaining_training':0,
        'report':desc(target),'created_utc':now()},immutable=True)
    save(OUT/'status.json',{'stage':'complete','models_completed':30,'updates_completed':7200,'finetuning_executed':True,
        'classification_pass':accepted,'probability_pass':co['probability_pass'],'current_development_candidate':candidate,
        'outer_results_released':True,'data_defect_audit':'skipped_by_user','report':desc(target),'updated_utc':now()})
    save(OUT/'deliverable_hashes.json',{p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name!='deliverable_hashes.json'},immutable=True)
    print({'text_lora':co,'current_development_candidate':candidate},flush=True)

if __name__=='__main__':main()
