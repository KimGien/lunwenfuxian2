from .common import *

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(v) for v in r)+' |' for r in rows])

def main():
    obj=read(REPORT/'results.json');verification=read(REPORT/'verification.json');assert verification['passed']
    lookup={(r['condition'],r['endpoint'],r['split']):r for r in obj['results']}
    pairs=read(REPORT/'subject_paired_diagnostics.json');evidence=read(REPORT/'evidence_diagnostics.json')
    def score(c,e='formal',s='test'):return lookup[(c,e,s)]['mean']
    def fmt(x):return f'{x:.6f}'
    metrics=['macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier']
    decision=obj['decision'];primary=[c for c in obj['comparisons'] if c['new']=='G-CCML-EST' and c['control'] in ['G-CE-EST','G-CCML-ST']]
    lines=['# FUSION-G1 整体改进主线完成报告','',
        '## 1. 结论与完成范围','',
        'G0—G3已完成。CE、EDL、CCML各自的ST（语音＋文本）与EST（EEG＋语音＋文本）共六条件，三种子×五折，共90个新模型、21,600次更新；每个模型60轮，全部按末轮评价。无EEG数据增强。G4阈值/校准、完整内层选模与DFMU未计入本轮，也未根据本轮成绩追加。','',
        '本轮使用EEG-E2v2通过工程验收的原记录处理输入。独立EEG筛查仍未通过；本轮回答的是其系统增量与融合方法差异。所有分数属于已反复用于开发的同一37人，三个种子的111条测试记录不代表111位独立受试者。','']
    for c in primary:
        lines += [f"- **{c['new']} 相对 {c['control']}：** Macro-F1差值 {c['delta']['macro_f1']:+.6f}，{c['F1_seeds_improved']}/3种子严格提高，MDD召回差值 {c['delta']['mdd_recall']:+.6f}；分类门槛{'通过' if c['classification_gate_passed'] else '未通过'}，NLL/Brier同时改善{'成立' if c['probability_quality_improved'] else '不成立'}。"]
    lines += ['', '**阶段决策：** '+('三族均未达到EEG增量门槛，停止本输入与架构版本的EEG融合扩展，保留ST作为后续比较参考。' if decision['stop_this_EEG_fusion_version'] else
        '至少一族达到EEG增量门槛，可保留对应开发互补信号；这不改变EEG独立分支不足的结论。')+
        (' CCML-EST未达到同输入CE-EST收益门槛，停止在这一CCML三模态版本继续叠加S/G/F。' if decision['stop_CCML_EST_SGF_extensions'] else
        ' CCML-EST达到同输入CE-EST的预定分类收益门槛。')+
        (' 预定CCML-EST具有EEG开发互补信号，可作为后续独立评价候选。' if decision['predeclared_CCML_EST_research_candidate'] else
        ' 预定CCML-EST未获得满足门槛的EEG增量支持。'),'',
        '## 2. 正式结果','',
        '主口径：每种子五折指标等权均值，再取三种子均值。阈值固定p(MDD)>0.5，等于0.5判HC；NLL与Brier越低越好。每种子合并37人的OOF指标已另存results.json，不能用它替换下表的预登记口径。','',
        table(['条件','Macro-F1','BA','MDD召回','AUC','NLL','Brier'],[[c]+[fmt(score(c)[k]) for k in metrics] for c in CONDITIONS]),'',
        '新模型从共同随机初值训练，语音/文本投影没有继承旧监督权重；因此与旧C0、P3、S-control的历史高分不是单因素对照。正式主方法仍是预先指定的CCML-EST，不按六条件排行榜重新指定赢家。','',
        '### 逐种子F1与训练/验证差距','',
        table(['条件','种子11','种子22','种子33','训练F1','验证F1','测试F1'],[
            [c]+[fmt(s['macro_f1']) for s in lookup[(c,'formal','test')]['per_seed']]+[fmt(score(c,s=s)['macro_f1']) for s in ['train','val','test']] for c in CONDITIONS]),'',
        '验证数据没有用于选轮、选超参或定阈值；表中训练、验证成绩均在90模型锁定后统一计算。','',
        '## 3. 配对比较与预定门槛','',
        '分类门槛：F1增量≥0.01，至少2/3种子严格改善，MDD召回降幅≤0.05。BA和概率指标完整并列；概率改善要求NLL与Brier均下降。EDL−CE为辅助比较，不另立正式获胜方法。','',
        table(['候选−控制','ΔF1','ΔBA','Δ召回','ΔNLL','ΔBrier','改善种子','分类门槛','概率改善'],[
            [c['new']+' − '+c['control']]+[f"{c['delta'][k]:+.6f}" for k in ['macro_f1','balanced_accuracy','mdd_recall','nll','brier']]+
            [f"{c['F1_seeds_improved']}/3",'通过' if c['classification_gate_passed'] else '未通过','是' if c['probability_quality_improved'] else '否'] for c in obj['comparisons']]),'',
        'EST−ST包含加入EEG后对整个系统训练的影响，不能全部归因于推理时多一条概率。三族的损失均按模态求和、按人平均；CCML另有一致与互补监督项，属于完整方法包比较，不能把CCML−CE解释为某一个公式项的独立因果效应。','',
        '## 4. 分支表现与固定输出诊断','',
        table(['条件','分支','F1','BA','MDD召回','AUC','NLL','Brier'],[[c,m]+[fmt(score(c,m)[k]) for k in metrics]
            for c in CONDITIONS for m in (['eeg','speech','text'] if c.endswith('-EST') else ['speech','text'])]),'',
        '上述分支来自各自联合训练后的模型，不等于另训的单模态系统。G线使用非仿射LayerNorm、分组学习率及无输出头max-norm，与E线独立EEG训练设置不同，不能把两表差值直接归于损失改变。','',
        '### 固定输出推理诊断（不替换正式成绩）','',
        table(['模型','推理方式','F1','BA','召回','AUC','NLL','Brier'],[
            [c,e]+[fmt(score(c,e)[k]) for k in metrics] for c in CONDITIONS
            for e in (['probability_mean']+(['evidence_mean','ccml'] if '-EDL-' in c else ['ccml'] if '-CCML-' in c else [])+
                      (['fixed_ST'] if c.endswith('-EST') else []))]),'',
        'probability_mean为概率均值；evidence_mean为证据均值；ccml按实际模态数计算；fixed_ST只从已训练的三模态模型去掉EEG输出，再按V=2使用该族正式规则。CE正式为softmax概率均值；EDL正式为Dirichlet概率均值；CCML正式为CCML。固定去EEG与重新训练ST回答不同问题，不互相替代。','',
        '### 逐人分类变化','',
        '下面每格为“EST新增正确 / EST新增错误”的重复种子记录数。retrained_ST是重新训练的两模态模型，fixed_ST是三模态训练后只取S/T。逐人身份、三个种子的方向和概率均保存在subject_paired_diagnostics.json。','',
        table(['训练族','对照','种子11','种子22','种子33','合计（111记录/37人）'],[
            [p['family'],p['comparator']]+[f"{s['gains']} / {s['losses']}" for s in p['per_seed']]+
            [f"{sum(s['gains'] for s in p['per_seed'])} / {sum(s['losses'] for s in p['per_seed'])}"] for p in pairs]),'',
        '### 证据与不确定性','',
        table(['模型','分支','平均证据强度S','平均u=2/S'],[[r['condition'],r['modality'],fmt(r['mean_strength']),fmt(r['mean_u'])]
            for r in evidence if r.get('group')=='all']),'',
        'CE没有Dirichlet证据或u，未将CE logits临时转换成证据。完整证据诊断另按分支预测正确/错误分组。较小u表示模型给出较强证据，不自动代表正确或校准良好，需结合上面的NLL/Brier与错误记录判断。','',
        '## 5. 训练是否实际发生及评价核验','',
        table(['条件','首轮损失','末轮损失','全局裁剪率','各分支平均裁剪前梯度范数'],[
            [r['condition'],fmt(r['mean_epoch_1_loss']),fmt(r['mean_epoch_60_loss']),f"{r['mean_clip_rate']:.2%}",
             '; '.join(f'{k}={v:.4f}' for k,v in r['branch_preclip_gradient_mean'].items())] for r in obj['training']]),'',
        '损失只用于判断各自训练过程，不能跨目标项不同的训练族直接比较数值大小。每分支参数状态均发生变化，输出随人员变化；所有更新的损失/梯度有限。所有头均无额外max-norm，只约束EEG空间卷积。全局裁剪会将所有分支按同一比例缩放，因此增加EEG即使在分支损失相加的CE/EDL中也可能改变S/T的优化轨迹；这是一种机制解释，不能仅靠梯度统计断言它造成全部性能差异。','',
        f"训练前26项分组检查与28项数学测试通过。90/90末轮模型训练内探针重载精确一致；评价后再独立重载90个模型，{verification['exact_subject_model_outputs']}份逐人模型输出逐位一致。NumPy独立复算所有推理规则，{verification['independent_fold_metric_vectors']}组逐折指标用独立混淆矩阵、成对AUC和概率公式核验通过。{verification['historical_entries_checked']}项历史清单条目复核不变。",'',
        '90个正式模型全部锁定后才释放验证与测试预测。保存了原始logits、分支特征、证据、逐人概率、训练批序、初始/最终权重、优化器和随机状态、梯度统计及完整哈希。冻结计划明确不执行G4，故没有测试后调阈值。','',
        '## 6. 后续处理','',
        '依据第1节门槛结论执行停止或保留候选，而不是继续搜索到分数提高为止。若后续重新改变EEG表征、融合目标或研究DFMU新输入版本，应独立登记假设和公平控制；现有DFMU v1旧输入路线由其自己的阶段记录管理，不与本轮新EEG结果混用。','',
        '下一项共同需要是冻结拟评价的流程，并在预先确定、未参与开发的新受试者上检验。现有37人的再划分或更多种子不能将它们恢复为独立验证人群。没有新队列时，本轮保留为完整的开发比较，不宣称未来人群性能已确认。','',
        '## 7. 复核入口','',
        '- 执行附表：`output/plans/FUSION_G1六条件执行附表.md`。',
        '- 登记与输入：`tridep_ccml/experiments/FUSION-G1-37-fulltext/registration.json`、`input_manifest.json`。',
        '- 发布门：实验目录`release_gate.json`；训练与预测：`runs/`、`predictions/`。',
        '- 完整指标及门槛：本目录`results.json`；逐人配对：`subject_paired_diagnostics.json`。',
        '- 独立核验：本目录`verification.json`；证据分组：`evidence_diagnostics.json`。',
        '- 所有阶段程序位于`tridep_ccml/fusion_g1/`。prepare/preflight/train/evaluate/summarize/verify/report分别对应登记、训练前验收、训练、预测、指标、独立复核和报告。已冻结输出不会被这些入口静默重训或覆盖。','']
    (REPORT/'FUSION-G1整体改进主线完成报告.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    for j,(family,color) in enumerate([('CE','#3366cc'),('EDL','#dd8844'),('CCML','#228866')]):
        values=[score(f'G-{family}-{mods}')['macro_f1'] for mods in ['ST','EST']]
        x=np.array([j*3,j*3+1]);ax.plot(x,values,color=color,marker='o',linewidth=2,label=family)
        for k,mods in enumerate(['ST','EST']):
            for offset,s in zip([-.12,0,.12],lookup[(f'G-{family}-{mods}','formal','test')]['per_seed']):
                ax.scatter(x[k]+offset,s['macro_f1'],s=22,color=color,alpha=.5)
            ax.annotate(f'{values[k]:.3f}',(x[k],values[k]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks([0,1,3,4,6,7],['CE ST','CE EST','EDL ST','EDL EST','CCML ST','CCML EST'])
    ax.set_ylim(0,1);ax.set_ylabel('Macro-F1 (five-fold mean)');ax.set_title('FUSION-G1: paired ST vs EST; 37 development subjects')
    ax.grid(axis='y',alpha=.2);ax.legend();fig.savefig(REPORT/'formal_f1.png',dpi=180);plt.close(fig)
    print(REPORT/'FUSION-G1整体改进主线完成报告.md',flush=True)

if __name__=='__main__':main()
