"""Verify all fixed N/R candidates together and report against matched control."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

from .nr import (OUT, PARENT, CONTROL, VARIANTS, CandidateHeads, load_data, logits_all,
                 predictions, configuration, protect_history)
from .common import WORK, TC, RecordContext, descriptor, verify_descriptor, read_json, require, write_new, set_seed
from .rebuild_eeg import validate_row
from .complete_eeg import mean_scores, crosscheck_metrics, METRICS
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.metrics import summarize
import numpy as np
import torch


def diagnostics(rows,baseline):
    base={(r['seed'],r['subject_id']):r for r in baseline if r['split']=='test'}
    tested=[r for r in rows if r['split']=='test']
    changed=[];min_counts=np.zeros((2,3));strength={};contributions={}
    for r in tested:
        old=base[r['seed'],r['subject_id']]
        if r['prediction']!=old['prediction']:
            changed.append({'seed':r['seed'],'subject_id':r['subject_id'],'label':r['label'],
                            'control_prediction':old['prediction'],'candidate_prediction':r['prediction'],
                            'control_p':old['p_final'][1],'candidate_p':r['p_final'][1],
                            'direction':'wrong_to_correct' if r['prediction']==r['label'] else 'correct_to_wrong'})
        e=np.array([r['views'][m]['e'] for m in ['eeg','speech','text']])
        for c in range(2):
            ties=e[:,c]==e[:,c].min();min_counts[c]+=ties/ties.sum()
    for m in ['eeg','speech','text']:
        strength[m]=float(np.mean([sum(r['views'][m]['alpha']) for r in tested]))
    for k in ['e_con','e_cmp','e_final']:
        contributions[k]={'mean_class_evidence':np.mean([r[k] for r in tested],axis=0).tolist(),
                          'mean_evidence_sum':float(np.mean([sum(r[k]) for r in tested]))}
    return {'changed_decisions':changed,'changed_count':len(changed),
            'changes_by_direction':dict(Counter(r['direction'] for r in changed)),
            'mean_strength':strength,'minimum_source_counts_class_by_view':min_counts.tolist(),
            'view_order':['eeg','speech','text'],'evidence_contributions':contributions,
            'train_val_test':{s:{m or 'final':mean_scores(rows,s,m) for m in [None,'eeg','speech','text']} for s in ['train','val','test']}}


def validate_complete():
    context=RecordContext();cfg=read_json(OUT/'configuration.json')
    require(cfg==configuration(context),'Locked N/R configuration changed')
    require(read_json(OUT/'preflight.json')['passed'],'Missing preflight')
    protected=protect_history();all_rows={v:[] for v in VARIANTS};baseline=[];checks=[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed);data=load_data(context,seed,fold)
            parent_dir=PARENT/'runs'/CONTROL/f'seed_{seed}'/f'fold_{fold}'
            base_meta=read_json(parent_dir/'checkpoint.json')['metadata']
            base_rows=read_json(parent_dir/'predictions.json');baseline+=base_rows
            for v,spec in VARIANTS.items():
                folder=OUT/'runs'/spec['experiment']/f'seed_{seed}'/f'fold_{fold}'
                for entry in read_json(folder/'complete.json').values():verify_descriptor(entry)
                cp=folder/'checkpoint.pt';side=read_json(cp.with_suffix('.json'))
                require(verify_descriptor(side['checkpoint'])==cp,'Checkpoint path mismatch')
                saved=torch.load(cp,map_location='cpu',weights_only=True);meta=saved['metadata']
                require(meta==side['metadata'],'Checkpoint metadata mismatch')
                require(meta['experiment']==spec['experiment'] and meta['variant']==v and meta['seed']==seed and meta['fold']==fold,'Wrong experiment identity')
                require(meta['train_ids']==context.folds[f'fold_{fold}']['train'] and meta['source']==data['source'],'Parent/train provenance mismatch')
                require(meta['configuration']==descriptor(OUT/'configuration.json') and meta['hyperparameters']==spec,'Hyperparameters mismatch')
                require(meta['cohort_sha256']==context.cohort_sha256 and meta['folds_sha256']==context.folds_sha256,'Cohort mismatch')
                require(meta['epochs']==60 and meta['updates']==240 and meta['objective']=='full_CCML_only','Training budget/objective mismatch')
                require(meta['trainable_parameters']==774,'Parameter count mismatch')
                require(meta['initialization_sha256']==base_meta['head_initialization_sha256']==state_digest(make_initial_state(seed)),'Initial state pairing failed')
                require(meta['batch_order_sha256']==base_meta['batch_order_sha256'],'Control batch pairing failed')
                gen=torch.Generator().manual_seed(seed);hasher=hashlib.sha256()
                for _ in range(60):hasher.update(torch.randperm(len(meta['train_ids']),generator=gen).numpy().astype('<i8').tobytes())
                require(meta['batch_order_sha256']==hasher.hexdigest(),'Batch regeneration failed')
                history=read_json(folder/'train_history.json')
                require([r['epoch'] for r in history]==list(range(1,61)),'Missing epochs')
                require(all(math.isfinite(r['train_loss']) and math.isclose(sum(r['loss_terms'].values()),r['train_loss'],abs_tol=1e-5,rel_tol=1e-5) for r in history),'Invalid loss diagnostics')
                model=CandidateHeads(v);model.load_state_dict(saved['state_dict'],strict=True)
                require(sum(p.numel() for p in model.parameters())==774,'Unexpected learned normalization')
                observed=read_json(folder/'predictions.json');replayed=predictions(model,data,context,seed,fold,v,cp)
                require(observed==replayed,'Checkpoint predictions do not reproduce exactly')
                require(len(observed)==37 and [r['subject_id'] for r in observed]==context.ids,'Subject coverage mismatch')
                for row in observed:validate_row(row)
                all_rows[v]+=observed
                checks.append({'variant':v,'seed':seed,'fold':fold,'checkpoint':descriptor(cp),
                               'predictions':descriptor(folder/'predictions.json'),'exact_reproduced_rows':37,
                               'batch_and_initialization_match_control':True,'updates':240})
    base_summary=summarize(baseline)
    expected=read_json(PARENT/'completion_audit.json')['summary'][CONTROL]
    require(base_summary['primary_seed_mean_of_fold_means']==expected['primary_seed_mean_of_fold_means'],'Parent control summary mismatch')
    result={'date':'2026-09-22','passed':True,'formal_runs':30,'updates':7200,'independent_subjects':37,
            'parent':PARENT.name,'baseline':base_summary,'baseline_diagnostics':diagnostics(baseline,baseline),
            'variants':{},'runs':checks,'protected_files_unchanged':protected,'nr_configuration':descriptor(OUT/'configuration.json'),
            'reporter':descriptor(__file__),'unit_tests':{'historical':47,'v2_eeg':6,'nr':5},
            'interpretation':'fixed developmental comparison; seeds are repeated observations, not independent subjects'}
    for v,rows in all_rows.items():
        require(len(rows)==555,'Incomplete predictions')
        require(Counter((r['seed'],r['subject_id']) for r in rows if r['split']=='test')==Counter((s,i) for s in cfg['seeds'] for i in context.ids),'OOF subject coverage mismatch')
        summary=summarize(rows);score=summary['primary_seed_mean_of_fold_means'];base=base_summary['primary_seed_mean_of_fold_means']
        delta={k:score[k]-base[k] for k in METRICS}
        seed_delta={s:summary['seed_reports'][s]['unweighted_fold_mean']['macro_f1']-base_summary['seed_reports'][s]['unweighted_fold_mean']['macro_f1'] for s in ['11','22','33']}
        gate={'f1_at_least_plus_0_01':delta['macro_f1']>=.01,'positive_seeds_at_least_2':sum(x>0 for x in seed_delta.values())>=2,
              'recall_drop_at_most_0_05':delta['mdd_recall']>=-.05,'nll_worsened':delta['nll']>0,'brier_worsened':delta['brier']>0}
        gate['development_criteria_passed']=all(gate[k] for k in ['f1_at_least_plus_0_01','positive_seeds_at_least_2','recall_drop_at_most_0_05'])
        result['variants'][v]={'summary':summary,'delta_vs_same_parent_control':delta,'seed_f1_deltas':seed_delta,
                               'stage_gate':gate,'diagnostics':diagnostics(rows,baseline),'sklearn_fold_crosschecks':crosscheck_metrics(rows)}
    require(protect_history()==protected,'History count changed')
    write_new(OUT/'completion_audit.json',result)
    return result


def report(result):
    base=result['baseline']['primary_seed_mean_of_fold_means']
    lines=['# CCML-v2-N/R 固定对照实验完成报告','','日期：2026-09-22。**N 与 R 各 15 组完成，共 30 组、7,200 次更新，无当前执行阻塞。**','','## 主要结果','',
           '直接控制为 EEG 修订后的 CCML-v2-E-control，使用完全相同的 15 组父表征。主口径为各种子五折均值，再取三个种子均值。旧 TC-04 仅是跨版本历史参照。','',
           '| 实验 | Macro-F1 | MDD recall | BA | NLL | Brier |','|---|---:|---:|---:|---:|---:|',
           '| 同版本 CCML 控制 | '+' | '.join(f'{base[k]:.6f}' for k in METRICS)+' |']
    for v,entry in result['variants'].items():
        m=entry['summary']['primary_seed_mean_of_fold_means']
        lines.append('| '+('N：逐人逐模态 LayerNorm' if v=='N' else 'R：weight_decay=1e-4')+' | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    lines+=['','| 候选 | F1 变化（百分点） | 召回变化（百分点） | F1 提高种子数 | 预定工程门槛 |','|---|---:|---:|---:|---|']
    for v,e in result['variants'].items():
        d=e['delta_vs_same_parent_control'];gate=e['stage_gate']
        lines.append(f'| {v} | {100*d["macro_f1"]:+.2f} | {100*d["mdd_recall"]:+.2f} | {sum(x>0 for x in e["seed_f1_deltas"].values())}/3 | '+('通过' if gate['development_criteria_passed'] else '未通过')+' |')
    lines+=['','工程门槛为平均 F1 至少增加 0.01、至少两个种子同向提高、MDD recall 下降不超过 0.05；另外完整披露 NLL/Brier。它不是显著性检验，也不是独立人群确认。','',
            '| 种子 | 同版本控制 F1 | N F1 | R F1 |','|---|---:|---:|---:|']
    for s in ['11','22','33']:
        values=[result['baseline']['seed_reports'][s]['unweighted_fold_mean']['macro_f1']]+[result['variants'][v]['summary']['seed_reports'][s]['unweighted_fold_mean']['macro_f1'] for v in ['N','R']]
        lines.append('| '+s+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    lines+=['','## 实验约束与验收','',
            '- N：在每个 128 维模态向量进入分类头前执行无仿射 LayerNorm，eps=1e-5；weight_decay=0。归一化只用该人该模态的维度，不拟合跨人统计。',
            '- R：保持输入不变，仅把 Adamax weight_decay 改为 1e-4；不叠加 LayerNorm。该系数是固定起点，未做搜索。',
            '- 两项均从共同种子的新头初始化，参数 774 个；与控制逐折核对初始化和批序；60 轮、batch 8、lr=0.001、梯度上限 1。rho=1.25、delta=gamma=1，KL 退火 50 轮，使用最后一轮、阈值 0.5。',
            '- 只在该折 train 优化，父表征全部冻结；没有新的 CE 训练、证据缩放、阈值选择或 N+R 组合。已有父表征仍包含历史 CE 训练来源。',
            '- 正式训练前以无改动路径精确重放 15 组控制、555 条记录；之后 30 个新检查点重载并精确复现 1,110 条预测，30 个测试折指标与 scikit-learn 交叉核验。',
            f'- 47 项原测试、6 项 EEG 路径测试和 5 项 N/R 测试通过；{result["protected_files_unchanged"]} 项历史文件哈希不变。首轮测试曾因不同矩阵批大小的浮点累加误差触发逐位比较失败，改为对归一化结果逐位校验、对分类输出使用 1e-6 相对/1e-7 绝对容差后通过；同批次内改变其他人/模态的测试仍要求逐位一致。',
            '- 本轮新检查点将版本元数据存为普通字符串，可直接用 weights_only=True 读取。','',
            '## 泛化与证据诊断','',
            '| 实验 | train 融合 F1 | val 融合 F1 | test 融合 F1 | EEG 平均 S | 语音平均 S | 文本平均 S |','|---|---:|---:|---:|---:|---:|---:|']
    for v,d in [('控制',result['baseline_diagnostics'])]+[(v,e['diagnostics']) for v,e in result['variants'].items()]:
        values=[d['train_val_test'][s]['final']['macro_f1'] for s in ['train','val','test']]+[d['mean_strength'][m] for m in ['eeg','speech','text']]
        lines.append('| '+v+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    for v,e in result['variants'].items():
        d=e['diagnostics'];c=d['changes_by_direction']
        lines.append(f'\n{v} 在三个种子的 111 条测试记录中，较控制改变 {d["changed_count"]} 条分类：错变对 {c.get("wrong_to_correct",0)}，对变错 {c.get("correct_to_wrong",0)}。重复种子记录不是 111 个独立人。')
    lines+=['','单模态 train/val/test 指标、逐类 min 来源、公共/互补证据、逐种子合并 OOF、ECE5、正确/错误不确定性和风险覆盖曲线均在 `completion_audit.json`。证据强度变化只作诊断，不能直接解释为可靠性提高。','',
            '## 下一步','']
    passing=[v for v,e in result['variants'].items() if e['stage_gate']['development_criteria_passed']]
    if passing:
        lines.append('达到工程门槛的候选：'+ '、'.join(passing)+'。可保留为后续开发候选；NLL/Brier 的变化仍需与分类收益一起评估，不将其称为独立验证的最优设置。')
    else:
        lines.append('N/R 均未达到预定工程门槛，不把任何一项自动升级为默认设置，也不基于本轮测试成绩追加搜索系数。')
    lines+=['',
            '计划中的下一阶段为 P：从同折的新父权重独立开始，仅解冻三路最后投影及新证据头，使用完整 CCML；它独立于 N/R，不直接继承某个候选的头。本次尚未启动 P，需要先导出投影前缓存、核实投影形状/参数量及训练名单，再做 15 组。',
            '本轮沿用已查看结果的 37 人固定划分，属于开发性比较。EEG 检查点尺度、坐标模板和语音路径限制仍保留；不同种子不增加独立样本数。上阶段及原 TC 系列的状态文件保留为各阶段历史记录，当前 N/R 状态以本目录 `status.json` 为准。','']
    path=OUT/'CCML-v2-NR实验完成报告.md';content='\n'.join(lines)
    if path.exists():require(path.read_text(encoding='utf-8')==content,'Completed report changed')
    else:path.write_text(content,encoding='utf-8')
    write_new(OUT/'status.json',{'status':'complete','date':'2026-09-22','N_runs':15,'R_runs':15,'updates':7200,
                              'control':CONTROL,'parent':PARENT.name,'report':descriptor(path),'audit':descriptor(OUT/'completion_audit.json'),
                              'development_criteria_passed':passing,'execution_blockers':[],'P_started':False,
                              'next_action':'Prepare same-fold pre-projection caches and independent CCML-v2-P; no automatic N/R combination'})
    print({v:e['summary']['primary_seed_mean_of_fold_means'] for v,e in result['variants'].items()})
    print({v:e['stage_gate'] for v,e in result['variants'].items()})
    print(f'Completed audit: 30 runs, 7200 updates, {result["protected_files_unchanged"]} protected files unchanged.')


if __name__=='__main__':report(validate_complete())
