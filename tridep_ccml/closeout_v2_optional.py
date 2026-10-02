"""Close the four requested remaining optional comparisons without changing history."""
from __future__ import annotations
import os
from pathlib import Path
from .v2.common import WORK,TC,read_json,write_new,descriptor,verify_descriptor,require

OUT=TC/'reports/ccml_v2_optional_closeout'
REPORT=OUT/'CCML-v2剩余可选实验总结.md'


def rel(path):return Path(os.path.relpath(path,OUT)).as_posix()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    families={'AS-RP':TC/'experiments/CCML-v2-AS-RP-37-fulltext',
              'GF':TC/'experiments/CCML-v2-GF-37-fulltext','S':TC/'experiments/CCML-v2-S-37-fulltext'}
    statuses={};audits={}
    for key,folder in families.items():
        status=read_json(folder/'status.json');require(status['status']=='complete','Unfinished optional stage')
        for field in ['configuration','plan','report','audit']:verify_descriptor(status[field])
        statuses[key]=status;audits[key]=read_json(verify_descriptor(status['audit']))
        require(audits[key]['passed'],'Failed completion audit')
    rp,gf=audits['AS-RP'],audits['GF']
    require(rp['runs']==30 and rp['updates']==7200 and gf['budget']['new_training_runs']==210 and gf['budget']['new_updates']==39600,'Budget differs')
    comparisons={**{'AS-'+k:v for k,v in rp['comparisons'].items()},**gf['comparisons']}
    passed=[k for k,v in comparisons.items() if v['gate']['development_criteria_passed']]
    result={'date':'2026-09-24','completed_optional_conditions':['AS-R','AS-P','G','F'],
            'stage_statuses':{k:descriptor(p/'status.json') for k,p in families.items()},
            'comparisons':comparisons,'development_gates_passed':passed,
            'new_budget':{'AS_RP_runs':30,'GF_runs':210,'total_runs':240,'total_updates':46800,
                          'outer_runs':60,'inner_runs':180,'reused_control_runs_not_counted':30},
            'cumulative_v2_budget':{'initial_fixed_runs':210,'S_runs':120,'remaining_optional_runs':240,
                    'total_runs':570,'outer_runs':300,'inner_runs':270,'total_updates':120600,
                    'TC00_E_supplement_excluded':True},
            'independent_subjects':37,'independent_validation':False,
            'replayed_new_records':rp['exact_replayed_records']+gf['exact_new_records'],
            'external_items_unresolved':['EEG input scale / coordinate provenance','new independent cohort'],
            'outside_v2_scope':['original TC-06 CE/CCML paired unfreezing'],
            'unregistered_combinations_not_started':['G+F','soft-min','modality dropout','additional strength search'],
            'no_further_training_registered':True,'closeout_code':descriptor(Path(__file__).resolve())}
    write_new(OUT/'results.json',result)
    lines=['# CCML v2 剩余可选实验总结','','日期：2026-09-24。范围：AS-R、AS-P、动态门控 G、额外最终融合监督 F。','',
           '**本轮四项固定可选实验已全部执行并验收。** 新增 240 组训练、46,800 次更新；已有控制组复用，不重复计算训练预算。原始实验、S 结果和之前的收尾快照完整保留。','',
           '## 两个输入版本分别比较','','### 旧 AS 完整语音覆盖版本','',
           '| 条件 | Macro-F1 | MDD recall | NLL | Brier |','|---|---:|---:|---:|---:|']
    for k,title in [('control','AS-control'),('R','AS-R'),('P','AS-P')]:
        score=rp['summaries'][k]['primary_seed_mean_of_fold_means']
        lines.append('| '+title+' | '+' | '.join(f'{score[m]:.6f}' for m in ['macro_f1','mdd_recall','nll','brier'])+' |')
    lines+=['','AS-R 只增加证据头固定正则（weight_decay=1e-4）；AS-P 只允许三路末端投影更新，无 LayerNorm。两项与 AS-control 同父版本、同初始化和批序，完整 CCML 保留。','',
            '### S 骨干均值版本与合法内层验证','',
            '| 条件 | Macro-F1 | MDD recall | NLL | Brier |','|---|---:|---:|---:|---:|']
    for family,key,title in [('GF','control','S-control'),('S','S','S（已完成静态尺度）'),('GF','G','G：逐人模态门控'),('GF','F','F：额外最终融合监督')]:
        score=audits[family]['summaries'][key]['primary_seed_mean_of_fold_means']
        lines.append('| '+title+' | '+' | '.join(f'{score[m]:.6f}' for m in ['macro_f1','mdd_recall','nll','brier'])+' |')
    lines+=['','G 共有 9 个门控参数（每路 2 个权重＋1 个偏置），依据本人的原始 u/类别概率差产生 [0.5,2] 的尺度并向 1 收缩；F 无门控或缩放，只额外监督最终融合 Dirichlet 输出。两者不叠加 S、不叠加彼此。',
            '所有 G/F 内层监督模型均从无 MODMA 标签拟合的冻结骨干特征出发重新训练；分别通过三折 inner-val NLL 选择两个固定强度。30 份选择锁定后才运行新外层模型。外层 val/test 未用于本次参数选择。',
            '**两张表的输入、汇聚结构和监督初始化不同，不能将跨表分数差异归因于某个单独机制。** 旧 NP/AS-NP 保留历史角色，不与新均值版本混作单因素比较。','',
            '## 固定门槛与收尾决定','',
            '| 候选 | 直接对照 | ΔF1（百分点） | 改善种子 | Δ召回（百分点） | 开发门槛 |','|---|---|---:|---:|---:|---|']
    for key,c in comparisons.items():
        d=c['delta'];positive=sum(v>0 for v in c['seed_f1_deltas'].values())
        lines.append(f'| {key} | '+('AS-control' if key.startswith('AS-') else 'S-control')+f' | {100*d["macro_f1"]:+.3f} | {positive}/3 | {100*d["mdd_recall"]:+.3f} | '+('通过' if c['gate']['development_criteria_passed'] else '未通过')+' |')
    lines+=['','开发门槛：F1≥+0.01、至少两个种子严格改善、召回降幅≤0.05，同时保留 NLL/Brier 权衡。通过者：'+('、'.join(passed) if passed else '无')+'。',
            '未通过者不升级为默认；通过者也仅作为开发候选保留。本轮没有额外组合、强度搜索或部署选模。','',
            '## 泛化差距','',
            '| 条件 | Train F1 | Val F1 | Test F1 | Train−Test |','|---|---:|---:|---:|---:|']
    diag={'AS-control':rp['control_diagnostics'], 'AS-R':rp['diagnostics']['R'],'AS-P':rp['diagnostics']['P'],
          'S-control':gf['diagnostics']['control'],'G':gf['diagnostics']['G'],'F':gf['diagnostics']['F']}
    for key,d in diag.items():
        values=[d['train_val_test'][p]['final']['macro_f1'] for p in ['train','val','test']]
        lines.append(f'| {key} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {values[0]-values[2]:.6f} |')
    lines+=['','37 位受试者仍是唯一独立样本单位，三个种子不是三批人。内层规范化选择没有消除既往反复查看人群的开发偏差；较高分数或验收通过均不能等同于独立泛化验证。','',
            '## 预算和核验','',
            '| 范围 | 外层训练 | 内层训练 | 训练合计 | 更新合计 |','|---|---:|---:|---:|---:|',
            '| 原 v2 固定阶段（Audit 至 AS-NP-R） | 210 | 0 | 210 | 50,400 |',
            '| S | 30 | 90 | 120 | 23,400 |',
            '| 本轮 AS-R/P | 30 | 0 | 30 | 7,200 |',
            '| 本轮 G/F | 30 | 180 | 210 | 39,600 |',
            '| v2 累计 | 300 | 270 | 570 | 120,600 |','',
            '原始 TRI-DEP 的 TC-00-E 及旧训练重放单列，不计入这张 v2 训练表。旧 AS-control、S-control 的复用也不计为新增训练。',
            f'本轮 240 个新检查点和 {result["replayed_new_records"]:,} 条新预测精确重载，30 份 G/F 内层选择复算，60 个新增外层测试折指标独立交叉核验通过。AS-R/P 测试 33 项、G/F 测试 39 项（含重叠核心测试，不能相加宣称独立覆盖）；G/F 最终核验 {gf["protected_files_unchanged"]:,} 项历史文件不变。','',
            '## 仍需外部条件的事项','',
            '| 事项 | 状态与边界 |','|---|---|',
            '| EEG 幅值与坐标模板来源 | 缺少能闭环确认的原采集/预训练处理依据；本轮未凭测试成绩选择单位或通道。 |',
            '| 独立人群验证 | 当前没有新的、未用于开发的同模态标注队列，不能用更多种子或原 37 人重新划分替代。 |',
            '| 原 TC-06 | 原始 CE/CCML 配对解冻计划，属于 v2 之外的保留项，没有标为完成。 |',
            '| soft-min、模态丢弃、G+F 及新增组合 | 原计划未登记为本轮固定候选；不将“剩余可选”无限扩展成额外搜索。 |','',
            '**本轮登记的四项可选实验已完成；研究中的外部证据与独立验证缺口仍然存在。**','',
            f'- [AS-R/P 完成报告]({rel(families["AS-RP"]/"CCML-v2-AS-RP实验完成报告.md")})',
            f'- [G/F 完成报告]({rel(families["GF"]/"CCML-v2-GF实验完成报告.md")})',
            f'- [S 完成报告]({rel(families["S"]/"CCML-v2-S实验完成报告.md")})',
            '- [汇总结果与来源](results.json)','- [当前完成状态](status.json)','']
    content='\n'.join(lines)
    if REPORT.exists():require(REPORT.read_text(encoding='utf8')==content,'Closeout changed')
    else:REPORT.write_text(content,encoding='utf8')
    write_new(OUT/'status.json',{'status':'complete','date':'2026-09-24','scope':'four_registered_remaining_optional_experiments',
              'completed':['AS-R','AS-P','G','F'],'new_training_runs':240,'new_updates':46800,'cumulative_v2':result['cumulative_v2_budget'],
              'independent_validation_complete':False,'external_provenance_resolved':False,'next_registered_training':None,
              'report':descriptor(REPORT),'results':descriptor(OUT/'results.json')})
    print('Remaining optional closeout complete: 240 new runs / 46800 updates.',flush=True)


if __name__=='__main__':main()
