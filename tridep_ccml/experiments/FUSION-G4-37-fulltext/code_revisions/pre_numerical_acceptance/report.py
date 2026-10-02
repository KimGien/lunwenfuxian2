from .common import *

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(str(v) for v in r)+' |' for r in rows])

def main():
    result=read(REPORT/'results.json');v=read(REPORT/'verification.json');assert v['passed']
    lookup={(r['condition'],r['variant'],r['split']):r for r in result['results']}
    parameters=read(REPORT/'postprocessor_parameters.json');changes=read(REPORT/'subject_changes.json')['summary']
    def score(c,variant,split='test'):return lookup[(c,variant,split)]['mean']
    def fmt(x):return f'{x:.6f}'
    primary=[c for c in result['comparisons'] if c['kind']=='postprocessing_vs_raw']
    supported=[c for c in primary if c['classification_reference_gate_passed']]
    better_probability=[c for c in primary if c['probability_quality_improved']]
    lines=['# FUSION G4 后处理开发实验完成报告','',
        '## 完成范围与结论','',
        'G4已完成：针对CCML的语音加文本ST和三模态EST，新增90个完整内层模型、16,200次更新，拟合30套阈值和30个正斜率sigmoid校准器，复用30个G1外层模型。全程无EEG数据增强、无外层重训。每个内层模型都从未监督训练的共同初值开始，训练和校准人员严格分开。','',
        '本次由用户在G1结果公开后要求执行，属于新的开发扩展。原G1登记明确未执行G4，原始结果与G3判断保持不变。虽然每个外折后处理参数仅使用该折train的OOF预测拟合，整个方法开发过程已经知晓这些37人的G1表现，因此本报告不声称独立确认或追溯预注册。','',
        ('分类方面，达到本扩展预定开发参考门槛的比较为：'+ '、'.join(c['candidate'][0]+' '+c['candidate'][1] for c in supported)+'。' if supported else
         '分类方面，四项后处理对原始输出的比较均未达到本扩展预定开发参考门槛。')+
        (' 概率方面，NLL和Brier同时改善的比较为：'+'、'.join(c['candidate'][0]+' '+c['candidate'][1] for c in better_probability)+'。' if better_probability else
         ' 概率方面，没有后处理同时改善NLL与Brier。'),'',
        '## 六组正式开发结果','',
        '主口径为每种子五折指标等权均值，再对三种子等权平均。Macro-F1、BA、MDD召回和AUC越高越好，NLL/Brier越低越好。raw为G1原始CCML概率和0.5阈值；threshold仅修改阈值；sigmoid修改最终概率后以0.5分类。','',
        table(['条件','处理','Macro-F1','BA','MDD召回','AUC','NLL','Brier'],[
            [c,var]+[fmt(score(c,var)[k]) for k in METRICS] for c in CONDITIONS for var in VARIANTS]),'',
        'threshold和raw使用完全相同的概率，因此AUC、NLL与Brier不变；阈值变化只影响分类指标。sigmoid的正斜率保持单模型内的概率排序，但不同外折的映射不同，合并37人的OOF AUC可能变化。完整逐折、逐种子与合并OOF结果见results.json。','',
        '## 配对增量与门槛','',
        '本扩展登记的分类参考为F1增加至少0.01、至少2/3种子严格提高、MDD召回降幅不超过0.05；概率改善要求NLL和Brier均下降。二者分开判断。三种匹配后处理的EST对ST比较属于辅助诊断，不能改写G1原始输出下的EEG判定，也不据此追加搜索。','',
        table(['候选对控制','ΔF1','ΔBA','ΔMDD召回','ΔNLL','ΔBrier','改善种子','分类门槛','概率改善'],[
            [' '.join(c['candidate'])+' 对 '+' '.join(c['control'])]+[f"{c['delta'][k]:+.6f}" for k in ['macro_f1','balanced_accuracy','mdd_recall','nll','brier']]+
            [f"{c['F1_seeds_improved']}/3",'通过' if c['classification_reference_gate_passed'] else '未通过','是' if c['probability_quality_improved'] else '否'] for c in result['comparisons']]),'',
        '### 每个种子的分类结果','',
        table(['条件','处理','种子11 F1','种子22 F1','种子33 F1','训练F1','验证F1'],[
            [c,var]+[fmt(s['macro_f1']) for s in lookup[(c,var,'test')]['per_seed']]+
            [fmt(score(c,var,split)['macro_f1']) for split in ['train','val']] for c in CONDITIONS for var in VARIANTS]),'',
        '训练与验证成绩仅作诊断，未用于后处理选择。用于拟合阈值的OOF最大F1也只是拟合分数，不能作为外层性能或独立验证结果。','',
        '## 阈值与校准器的拟合情况','',
        '每个外折仅对其train人员进行三折分层交叉拟合。每人有且仅有一条由未使用本人训练的模型生成的OOF预测。阈值最大化该组OOF的Macro-F1，并列时依次选离0.5最近、再较小的值。分类使用p>阈值，相等判HC。sigmoid用OOF的平均NLL加0.005×[(a−1)²+b²]拟合，a在[0.01,10]、b在[−10,10]，初值(1,0)。完整算法继承P3，没有阈值与sigmoid串联。','',
        table(['条件','种子','外折','阈值','斜率a','截距b','sigmoid对应原概率分界','参数达边界'],[
            [p['condition'],p['seed'],p['fold'],fmt(p['threshold']),fmt(p['a']),fmt(p['b']),fmt(p['sigmoid_identity_equivalent_raw_threshold']),
             '是' if p['slope_on_bound'] or p['intercept_on_bound'] else '否'] for p in parameters]),'',
        'sigmoid对应原概率分界为sigmoid(−b/a)，用于解释分类边界如何移动，不是另一套拟合阈值。校准后的概率不解释为新的Dirichlet证据或不确定性u。','',
        '## 逐人分类变化','',
        '下表每格为后处理相对raw的错变对与对变错记录数。每个种子37人；合计111条记录来自相同37人。subject_changes.json保存每人的三个种子变化，不把重复记录当作独立样本。','',
        table(['条件','处理','种子11 得到与失去','种子22 得到与失去','种子33 得到与失去','HC 得到与失去','MDD 得到与失去'],[
            [c['condition'],c['variant']]+[f"{s['gain']} / {s['loss']}" for s in c['per_seed']]+[f"{s['gain']} / {s['loss']}" for s in c['per_label']] for c in changes]),'',
        '## 核验与解释限制','',
        f"90个内层模型训练探针和OOF输出精确重载，{v['inner_OOF_predictions_exact']}条内层OOF预测复核通过；30个冻结外层模型再次精确重载。30套阈值通过独立候选枚举和混淆矩阵F1核验，30个sigmoid参数重新拟合一致，目标值与有界最优性条件独立检查通过。{v['independent_fold_metric_vectors']}组逐折指标用独立公式复算一致。G1原始成绩逐折一致，threshold的概率指标完全不变。",'',
        f"{v['historical_entries_checked']}项历史哈希清单条目保持不变。全部内层模型锁定、全部校准器锁定后才计算G4变换后的外层成绩；该发布顺序不改变G1成绩此前已经被开发者查看的事实。",'',
        '这些后处理只调整已有分数的决策边界或概率刻度，不重新学习EEG表征。即使F1提高，也不能据此认定EEG信号表示或未来人群泛化已经改善。内层模型拟合17或18人，外层模型拟合26或27人，二者的分数分布可能不同，这是OOF后处理在小样本下的实际限制；本轮未使用外层成绩修正规则。','',
        '本次完整保留raw、threshold和sigmoid，未按外层排行榜选择默认方法。若后续固定某个变体，应明确其来自这些开发结果的选择，并在新的预先确定人群上评价。G4完成后不自动增加校准器、阈值搜索、模型或DFMU实验。','',
        '## 文件入口与累计预算','',
        '- 固定规则：output/plans/FUSION_G4后处理开发实验执行附表.md。',
        '- 登记、内折身份和来源：实验目录registration.json、input_manifest.json、prior_knowledge.json。',
        '- 内层模型与OOF：实验目录runs；30套参数：calibrators；外层派生记录：predictions。',
        '- 指标与配对：本报告目录results.json、subject_changes.json、postprocessor_parameters.json。',
        '- 独立核验：本报告目录verification.json。','',
        'G4新增90个神经拟合、16,200次更新；原两条核心线150个神经模型加10次线性拟合已完成。合计240个新神经模型加10次线性拟合，即250次模型或线性拟合、49,500次神经更新。G4的30个轻量sigmoid拟合与30次阈值搜索单列，复用30个外层模型不重复计数，DFMU独立路线不计入。','']
    (REPORT/'FUSION-G4后处理开发实验完成报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(REPORT/'FUSION-G4后处理开发实验完成报告.md',flush=True)

if __name__=='__main__':main()
