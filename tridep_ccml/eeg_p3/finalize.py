"""Compose the fixed-study report after numerical and artifact verification."""
from __future__ import annotations
import re
from .common import *

def finalize():
    verification=read_json(REPORT/'verification.json')
    require(verification['passed'],'Verification not complete')
    records=read_json(REPORT/'metrics.json')
    lookup={(r['condition'],r['arm'],r['endpoint'],r['postprocess'],r['split']):r for r in records}
    paired=read_json(REPORT/'paired_comparisons.json')
    attention=read_json(REPORT/'attention_summary.json')
    calibration=read_json(REPORT/'calibrator_summary.json')
    def get(c,a='B',e='eeg',p='raw',s='test'):return lookup[(c,a,e,p,s)]['mean']
    def names(row):return f"{row['condition']}/{row['arm'] if row['postprocess']=='raw' else row['postprocess']}/{row['endpoint']}"
    passed=[r for r in paired if r['development_F1_signal_gate']]
    headline='、'.join(names(r) for r in passed) if passed else '无'
    lines=['# EEG-P3：泛化、BN 与训练内部校准完成报告','',
        '日期：2026-09-29。优先级 3 已完成。37 人、原五折、种子 11/22/33；原始 EEG、无增强、无删人删窗。', '',
        '**本轮找到部分 F1 改善，但跨人泛化问题尚未解决。** 逐受试者标准化改善了独立 EEGNet 的平均 F1；BN 重估的效果依赖路线；token 的概率校准没有改善 EEG 分类，融合 F1 的提高还伴随概率误差增大。这只描述反复使用的开发队列，不能写成独立验证收益，也没有自动替换现有模型。','',
        '本轮完成 60 个 EEGNet 外层新模型、90 个 token 内层校准模型、60 个只改 BN 统计量的派生状态。全部训练、BN 参数及校准参数固定并锁定哈希后，才统一计算外层结果。原 B 对照逐一重放通过。','',
        '## 1. EEGNet：泛化与 BN 分开比较','',
        'B=原 R1；BR=原权重只用 train 人重估 BN；G=只改 GroupNorm 后从相同参数重新训练；S=只改为每人每导联独立标准化后重新训练；SR=S 再重估 train BN。每个方案均保留所有 60 个窗口。','',
        '| 路线 | 方案 | train EEG F1 | test EEG F1 | Δtest F1 | test BA | test MDD recall | test AUC |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in ['U1','C1']:
        for a in ['B','BR','G','S','SR']:
            m=get(c,a);t=get(c,a,s='train');d=m['macro_f1']-get(c)['macro_f1']
            lines.append(f'| {c} | {a} | {t["macro_f1"]:.6f} | {m["macro_f1"]:.6f} | {d:+.6f} | {m["balanced_accuracy"]:.6f} | {m["mdd_recall"]:.6f} | {m["auc"]:.6f} |')
    lines+=['','所有数字先平均五折、再平均三个种子。训练测试差距缩小有时来自训练能力下降，不能单独当成泛化改善；应同时看测试 F1、BA、AUC 和逐种子差值。','',
        f'具体看 U1-S，F1 从 {get("U1")["macro_f1"]:.6f} 提升到 {get("U1","S")["macro_f1"]:.6f}，但 BA 仅 {get("U1","S")["balanced_accuracy"]:.6f}、AUC 为 {get("U1","S")["auc"]:.6f}，三个种子仍有一个下降。其 train−test F1 差从 {get("U1",s="train")["macro_f1"]-get("U1")["macro_f1"]:.6f} 扩大到 {get("U1","S",s="train")["macro_f1"]-get("U1","S")["macro_f1"]:.6f}；不能把它写成已经缩小了泛化落差。', '',
        f'C1-SR 的 EEG F1 为 {get("C1","SR")["macro_f1"]:.6f}，高于原 C1 的 {get("C1")["macro_f1"]:.6f}；但其 AUC 仍为 {get("C1","SR")["auc"]:.6f}，且 seed11 下降、其余两个种子上升。当前小样本结果不足以支持稳定的跨人排序能力。','',
        f'独立 EEG 的 BN 重估把测试 F1 从 {get("U1")["macro_f1"]:.6f} 变成 {get("U1","BR")["macro_f1"]:.6f}；共同训练 EEG 从 {get("C1")["macro_f1"]:.6f} 变成 {get("C1","BR")["macro_f1"]:.6f}。BR 不改变学习参数，能直接检验原运行统计量是否影响该模型的判别；它不能证明跨受试者问题完全由 BN 造成。','',
        'BN 重估依次计算各层全部训练输入的总体均值/方差，关闭 dropout，更新前层后再计算后层。它不是在测试批次上重新计算统计量。G 则同时改变归一化的统计轴，虽不依赖其他人的同批数据，但不能把 G 的全部差异归因于移除了运行统计量。 [PyTorch GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)','',
        'BR 是预先固定的一种推理统计方案。训练时的批次分布及 dropout 与推理不同，重估全训练总体统计量本身不保证最优。S 使用该人完整 300 秒的无标签 mean/std，是事先约定的离线变换，不适用于逐点实时场景。它会丢弃绝对幅值，并可能相对放大低质量通道；不能把 S 的变化解释成已清除伪迹。','',
        '| C1 方案 | 融合 F1 | Δ融合 F1 | 融合 BA | 融合 NLL |', '|---|---:|---:|---:|---:|']
    for a in ['B','BR','G','S','SR']:
        m=get('C1',a,'fusion');base=get('C1','B','fusion')
        lines.append(f'| {a} | {m["macro_f1"]:.6f} | {m["macro_f1"]-base["macro_f1"]:+.6f} | {m["balanced_accuracy"]:.6f} | {m["nll"]:.6f} |')
    lines+=['','共同训练中的 EEG 变化可能通过损失改变语音/文本参数；融合变化不能单独归功于 EEG。BR/SR 则保持这些学习参数不变，只改变 EEG 的 BN 推理统计量。','',
        f'![EEGNet 训练与外层开发折比较]({(REPORT/"EEGNet_generalization.png").as_posix()})','',
        '图中橙色点分别表示三个种子的五折均值；它们是重复评价，并非独立队列。','',
        '## 2. C4 注意力：有学习活动是否等于有效选择','',
        f'15 个原 C4 模型仅使用各自 train 人：归一化熵平均为 {attention["normalized_entropy_mean"]["mean"]:.8f}，平均最大权重 {attention["maximum_weight_mean"]["mean"]:.8f}，均匀权重为 1/95≈0.01052632。逆平均权重集中度对应约 {attention["effective_positions_mean"]["mean"]:.3f}/95 个有效位置。','',
        f'query 参数相对初始化的 L2 改变量均值为 {attention["query_change_norm"]["mean"]:.6f}，train 梯度非零；但把同一模型的注意力设成均匀权重后，128 维 EEG 表示相对 L2 改变量平均仅 {100*attention["uniform_ablation_representation_relative_l2"]["mean"]:.3f}%，EEG 概率平均绝对变化为 {attention["uniform_ablation_eeg_probability_mean_abs_change"]["mean"]:.6f}，分类翻转共 {int(attention["uniform_ablation_eeg_prediction_flips"]["sum"])} 条。','',
        '翻转计数的分母是 399 条“模型×训练受试者”记录，重复涉及同一批人；不是 399 位受试者。这说明原注意力有参数更新，但其选择作用有限，不能用“梯度非零”代替“注意力形成了有用选择”，也不能据此否定所有注意力结构。该检查没有修改原模型。','',
        '## 3. Token 路线：概率校准与 F1 阈值分开','',
        '每个外层 train 集做 3 折分层交叉拟合，仅用未参加对应内层模型训练的人的预测拟合参数。每组只有 26–27 人；外层 val/test 不进入拟合。原始方案固定 0.5，sigmoid 只优化正则 NLL、阈值仍为 0.5，threshold 方案仅在训练 OOF 上选择 Macro-F1 阈值。 [交叉拟合校准的训练隔离原则](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)','',
        '| 路线 | 端点 | 方案 | test F1 | ΔF1 | BA | AUC | NLL | Brier |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for c in ['C3','C4']:
        for e in ['eeg','fusion']:
            for p in ['raw','sigmoid','threshold']:
                m=get(c,'B',e,p);base=get(c,'B',e)
                lines.append(f'| {c} | {e} | {p} | {m["macro_f1"]:.6f} | {m["macro_f1"]-base["macro_f1"]:+.6f} | '+
                    ' | '.join(f'{m[k]:.6f}' for k in ['balanced_accuracy','auc','nll','brier'])+' |')
    lines+=['','| 路线/端点 | 内部 OOF AUC 均值 | sigmoid 斜率范围 | 训练 OOF 阈值范围 |',
            '|---|---:|---:|---:|']
    for r in calibration:
        lines.append(f'| {r["condition"]}/{r["endpoint"]} | {np.mean(r["inner_OOF_AUC"]):.6f} | '+
            f'{min(r["sigmoid_a"]):.4f}–{max(r["sigmoid_a"]):.4f} | '+
            f'{min(r["thresholds"]):.4f}–{max(r["thresholds"]):.4f} |')
    lines+=['','内部 OOF AUC 描述用于校准的交叉拟合分数，不是校准器的外层性能，也不同于每个模型内的 AUC。斜率小于 1 表示收缩 log-odds 差异；若这些内部留出预测本身没有稳定排序，校准只能调整现有概率，不能补出新的判别信息。','',
        '本轮 EEG sigmoid 的斜率全部小于 1，其中 C3、C4 各有 10/15 组到达预设下限 0.01；截距均为负。外层预测因此全部判 HC，MDD recall=0，五折平均 F1 降到 0.361772。其每折 AUC 没有改变，体现了“仍有一定排序”与“0.5 阈值下有效分类”的区别。','',
        '训练 OOF 阈值方案将 C3/C4 的 EEG F1 都提高到 0.470522，但 C3 只有一个种子提高，未通过本轮稳定性门槛；C4 有两个种子提高，也不能证明原注意力形成了有效选择。阈值更改利用的是既有分数，不是表示学习的新增收益。','',
        '融合 sigmoid 的 C3/C4 F1 均由 0.788504 增至 0.809747；但 C3 的 NLL 由 0.494292 升至 0.539959、Brier 由 0.157975 升至 0.167942，C4 同样变差。因此可以报告开发集分类边界收益，不能报告整体概率质量改善。这也是最终门槛表只命名为 F1 信号、没有自动选用该方案的原因。','',
        '正斜率 sigmoid 保持每个模型内的排序，五折平均 AUC 已验证不变；它改变概率刻度和 0.5 对应的分类边界，不产生新的 EEG 信息。threshold 不改变概率，因此 NLL/Brier/AUC 与 raw 相同；F1 上升不能称作概率校准改善。不同折各用自己的校准器，合并 OOF 后的跨折排序可能变化，不能与平均折 AUC 混淆。','',
        'EEG 与最终融合概率分别后处理，没有把校准后的概率重新伪装成 Dirichlet 证据，也没有把单模态校准结果塞回证据融合。内层模型训练人数较少，可能与完整 train 模型存在概率分布差异；训练 OOF 内拟合分数不是校准器的无偏性能估计。','',
        f'![Token 校准结果]({(REPORT/"Token_calibration.png").as_posix()})','',
        '## 4. 固定改善门槛与解释边界','',
        '门槛在运行前固定：平均测试 Macro-F1 增量≥0.02、至少两个种子增量为正、平均 BA 不下降。这里是相对同一路线原模型的 P3 诊断门槛，不替代 R1 相对 U0 的≥0.03及完整融合接入要求。逐种子结果全部保留；未按外层成绩选择最终方案、重新调参或追加试验。','',
        '| 通过门槛的组合 | 平均 ΔF1 | seed11 ΔF1 | seed22 ΔF1 | seed33 ΔF1 |','|---|---:|---:|---:|---:|']
    for r in passed:
        lines.append(f'| {names(r)} | {r["delta_mean"]["macro_f1"]:+.6f} | '+
                     ' | '.join(f'{s["macro_f1"]:+.6f}' for s in r['delta_per_seed'])+' |')
    if not passed:lines.append('| 无 | — | — | — | — |')
    lines+=['','本轮保留了 P2 发现的滤波边界瞬态和 CBraMod 单位来源未闭合这两个限制，因此不能宣布输入质量已经修复。BN/标准化/阈值研究也没有改变只有 37 位独立受试者的事实。三个种子的 111 条测试预测是重复评价，不是新增 111 人。','',
        '下一步应先单独版本化验证滤波边界处理，再在训练内部固定处理与模型方案；涉及最终性能的结论仍需要预先确定、未参与开发的独立评价。不能根据本表删除高幅受试者或倒推单位。','',
        '## 5. 验证和文件','',
        f'- 6 项契约测试、两种 EEGNet 的 4 人小样本拟合通过；150 个新模型共 {verification["health_checks"]} 次初末学习检查通过。',
        '- 60 个 BN 派生状态均逐项核实：只改 running_mean/running_var，学习参数不变。',
        '- 30 组校准参数的 OOF 身份、正斜率、拟合目标和最优性通过独立检查；无外层校准拟合。',
        '- 180 份预测的检查点重载、推理状态不变、单人/批次一致性通过；60 个原模型对照重放通过。',
        '- R1 986 项、P1 667 项、P2 134 项冻结产物核验不变；源缓存仍由原哈希绑定。',
        f'- [完整指标与逐种子差值]({(REPORT/"P3全量指标与逐种子差值.md").as_posix()})：含 train/test、NLL、Brier 和全部配对方案。',
        f'- [验证结果]({(REPORT/"verification.json").as_posix()})；metrics.json 为逐折/逐种子及合并 OOF 指标，predictions_long.json 为全部逐人预测。',
        f'- [固定计划]({PLAN.as_posix()})；模型、内层划分、训练记录和校准参数位于 [{OUT.name}]({OUT.as_posix()})。','']
    report=REPORT/'EEG-P3泛化BN与校准报告.md'
    require(not report.exists(),'Report already exists')
    content=re.sub(r'\]\((C:/[^)\n]+)\)',r'](<\1>)','\n'.join(lines))
    report.write_text(content,encoding='utf-8')
    completion={'complete':True,'priority':3,'augmentation':False,'subjects_deleted':0,
        'independent_validation':False,'report':descriptor(report),'verification':descriptor(REPORT/'verification.json'),
        'configuration':descriptor(OUT/'configuration.json'),'sources':{p.relative_to(WORK).as_posix():digest(p) for p in CODE.glob('*.py')},
        'finished_utc':now(),'development_signal_combinations':[names(r) for r in passed]}
    write_new(REPORT/'completion.json',completion)
    replace_json(OUT/'status.json',{'state':'complete','priority':3,'updated_utc':now(),'report':descriptor(report)})
    for root in [OUT,REPORT]:
        hashes={p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='deliverable_hashes.json'}
        write_new(root/'deliverable_hashes.json',hashes)
        for name,sha in hashes.items():require(digest(root/name)==sha,'Final deliverable hash failed')
    print('P3 COMPLETE: '+str(report))

if __name__=='__main__':finalize()
