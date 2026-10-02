"""Validate projection adaptation, paired control, and fixed-stage report."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

from .p import (OUT, CONFIG, PARENT, CONTROL, NR, EXPERIMENT, DIMENSIONS, MODALITIES,
                PROJECTION_COUNT, ProjectionHeads, configuration, load_data, predictions,
                project_all, protect_history)
from .common import WORK, TC, RecordContext, descriptor, verify_descriptor, read_json, require, write_new, set_seed
from .resume_eeg import load_checkpoint as load_eeg_checkpoint
from .rebuild_eeg import validate_row
from .complete_nr import diagnostics
from .complete_eeg import crosscheck_metrics, METRICS
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.metrics import summarize
import numpy as np
import torch


def gate(summary,control):
    score=summary['primary_seed_mean_of_fold_means'];base=control['primary_seed_mean_of_fold_means']
    delta={k:score[k]-base[k] for k in METRICS}
    seed_delta={s:summary['seed_reports'][s]['unweighted_fold_mean']['macro_f1']-control['seed_reports'][s]['unweighted_fold_mean']['macro_f1'] for s in ['11','22','33']}
    checks={'mean_f1_at_least_plus_0_01':delta['macro_f1']>=.01,'positive_seeds_at_least_2':sum(x>0 for x in seed_delta.values())>=2,
            'recall_drop_at_most_0_05':delta['mdd_recall']>=-.05,'nll_worsened':delta['nll']>0,'brier_worsened':delta['brier']>0}
    checks['development_criteria_passed']=all(checks[k] for k in ['mean_f1_at_least_plus_0_01','positive_seeds_at_least_2','recall_drop_at_most_0_05'])
    return {'delta':delta,'seed_f1_deltas':seed_delta,'gate':checks}


def audit():
    context=RecordContext();cfg=read_json(CONFIG)
    require(cfg==configuration(context),'P configuration changed')
    preflight=read_json(OUT/'preflight.json')
    require(preflight['passed'] and preflight['cache_bundles']==15,'P preflight incomplete')
    for item in preflight['manifests']:verify_descriptor(item)
    protected=protect_history();all_rows=[];baseline=[];checks=[];drift=[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed);data=load_data(context,seed,fold)
            cache_manifest=read_json(verify_descriptor(data['source']['manifest']))
            initial_model=ProjectionHeads(data['initial'],seed)
            initial_hash=state_digest(initial_model.state_dict())
            require(sum(p.numel() for p in initial_model.parameters())==836742,'Initial trainable count mismatch')
            for m in MODALITIES:
                source=verify_descriptor(cache_manifest['sources'][m]['checkpoint'])
                if m=='eeg':
                    parent=load_eeg_checkpoint(source);prefix='eeg.classifier.2.';meta=parent['metadata']
                else:
                    parent=torch.load(source,map_location='cpu',weights_only=True);prefix='classifier.'+('2.' if m=='speech' else '0.');meta=parent
                require(meta['seed']==seed and meta['fold']==fold and meta['train_ids']==context.folds[f'fold_{fold}']['train'],'Projection initialization from wrong outer fold')
                for k in ['weight','bias']:
                    require(torch.equal(data['initial'][f'{m}.0.{k}'],parent['state_dict'][prefix+k]),'Initial projection differs from its legal parent')
                del parent
            frozen_path=verify_descriptor(cache_manifest['source_frozen_bundle']['features'])
            with np.load(frozen_path,allow_pickle=False) as archive:parent_reps=archive['features']
            initial_reps=project_all(initial_model,data).numpy()
            require(np.allclose(initial_reps,parent_reps,rtol=cfg['preflight_representation_rtol'],atol=cfg['preflight_representation_atol']),'Initial P representations differ from parent')
            folder=OUT/f'runs/seed_{seed}/fold_{fold}'
            complete=read_json(folder/'complete.json')
            for item in complete.values():verify_descriptor(item)
            cp=folder/'checkpoint.pt';side=read_json(cp.with_suffix('.json'));verify_descriptor(side['checkpoint'])
            saved=torch.load(cp,map_location='cpu',weights_only=True);meta=saved['metadata']
            require(meta==side['metadata'],'P checkpoint metadata mismatch')
            require(meta['experiment']==EXPERIMENT and meta['seed']==seed and meta['fold']==fold,'Wrong run identity')
            require(meta['source']==data['source'] and meta['configuration']==descriptor(CONFIG),'P source/config binding mismatch')
            require(meta['cohort_sha256']==context.cohort_sha256 and meta['folds_sha256']==context.folds_sha256,'P cohort mismatch')
            require(meta['train_ids']==context.folds[f'fold_{fold}']['train'],'P training subjects mismatch')
            require(meta['objective']=='full_CCML_only' and meta['epochs']==60 and meta['updates']==240,'Objective/budget mismatch')
            require(meta['trainable_parameters']==836742 and meta['projection_parameters']==PROJECTION_COUNT,'Trainable count mismatch')
            require(meta['projection_lr']==1e-4 and meta['head_lr']==1e-3 and meta['weight_decay']==0 and meta['dropout']=='eval','P factors changed')
            require(meta['initial_state_sha256']==initial_hash and meta['initial_projection_sha256']==state_digest(data['initial']),'Initial P state mismatch')
            parent_dir=PARENT/'runs'/CONTROL/f'seed_{seed}'/f'fold_{fold}'
            base_meta=read_json(parent_dir/'checkpoint.json')['metadata'];baseline+=read_json(parent_dir/'predictions.json')
            require(meta['head_initialization_sha256']==base_meta['head_initialization_sha256']==state_digest(make_initial_state(seed)),'New heads not independently initialized')
            gen=torch.Generator().manual_seed(seed);h=hashlib.sha256()
            for _ in range(60):h.update(torch.randperm(len(meta['train_ids']),generator=gen).numpy().astype('<i8').tobytes())
            require(meta['batch_order_sha256']==base_meta['batch_order_sha256']==h.hexdigest(),'P/control batch order mismatch')
            history=read_json(folder/'train_history.json')
            require([r['epoch'] for r in history]==list(range(1,61)),'Missing epochs')
            for row in history:
                require(math.isfinite(row['train_loss']) and math.isclose(row['train_loss'],sum(row['loss_terms'].values()),rel_tol=1e-5,abs_tol=1e-5),'Loss diagnostics mismatch')
                require(all(math.isfinite(v) and v>=0 for group in ['projection_grad_norm_mean','head_grad_norm_mean'] for v in row[group].values()),'Invalid gradient diagnostics')
            for m in MODALITIES:
                require(any(r['projection_grad_norm_mean'][m]>0 for r in history),f'Projection never received a gradient: {m}')
            model=ProjectionHeads(data['initial'],seed);model.load_state_dict(saved['state_dict'],strict=True)
            require(all(not m.training for m in model.modules() if isinstance(m,torch.nn.Dropout)),'Dropout activated')
            require(not any(isinstance(m,(torch.nn.Conv1d,torch.nn.LSTM)) for m in model.modules()),'Frozen upstream modules entered training model')
            row_drift={'seed':seed,'fold':fold,'projections':{}}
            for m in MODALITIES:
                before=torch.cat([data['initial'][f'{m}.0.{k}'].flatten() for k in ['weight','bias']])
                after=torch.cat([model.projections[m][0].state_dict()[k].detach().flatten() for k in ['weight','bias']])
                require(not torch.equal(before,after),f'Projection did not update: {m}')
                row_drift['projections'][m]={'l2_change':float(torch.linalg.vector_norm(after-before)),
                                              'relative_l2_change':float(torch.linalg.vector_norm(after-before)/torch.linalg.vector_norm(before))}
            drift.append(row_drift)
            rows=read_json(folder/'predictions.json');replay=predictions(model,data,context,seed,fold,cp)
            require(rows==replay,'P saved predictions do not reproduce exactly')
            require(len(rows)==37 and [r['subject_id'] for r in rows]==context.ids,'P coverage mismatch')
            for row in rows:validate_row(row)
            all_rows+=rows;checks.append({'seed':seed,'fold':fold,'checkpoint':descriptor(cp),'predictions':descriptor(folder/'predictions.json'),
                                         'exact_reproduced_rows':37,'projection_init_verified_against_parents':True,
                                         'all_three_projections_updated':True,'updates':240})
    require(len(all_rows)==555 and Counter((r['seed'],r['subject_id']) for r in all_rows if r['split']=='test')==Counter((s,i) for s in cfg['seeds'] for i in context.ids),'P OOF coverage incomplete')
    summary=summarize(all_rows);control=summarize(baseline)
    nr=read_json(NR/'completion_audit.json');n_summary=nr['variants']['N']['summary']
    require(control['primary_seed_mean_of_fold_means']==nr['baseline']['primary_seed_mean_of_fold_means'],'Baseline drift')
    result={'date':'2026-09-22','passed':True,'formal_runs':15,'updates':3600,'subject_fold_rows':555,'independent_subjects':37,
            'summary':summary,'control':control,'N_reference':n_summary,'paired_control_comparison':gate(summary,control),
            'P_vs_N_descriptive_only':{k:summary['primary_seed_mean_of_fold_means'][k]-n_summary['primary_seed_mean_of_fold_means'][k] for k in METRICS},
            'diagnostics':diagnostics(all_rows,baseline),'projection_changes':drift,'run_checks':checks,
            'sklearn_fold_crosschecks':crosscheck_metrics(all_rows),'preflight':preflight,
            'protected_files_unchanged':protect_history(),'unit_tests':{'historical':47,'v2_eeg':6,'nr':5,'p':4},
            'configuration':descriptor(CONFIG),'reporter':descriptor(__file__),
            'source_audits':{'E':descriptor(PARENT/'completion_audit.json'),'NR':descriptor(NR/'completion_audit.json')},
            'scope':'P alone, no N/R combination; same parent, initialization and batch order as control; developmental cohort already viewed'}
    require(result['protected_files_unchanged']==protected,'History changed')
    write_new(OUT/'completion_audit.json',result)
    return result


def report(result):
    score=result['summary']['primary_seed_mean_of_fold_means'];comparison=result['paired_control_comparison'];g=comparison['gate'];delta=comparison['delta']
    lines=['# CCML-v2-P 有限投影适配完成报告','','日期：2026-09-22。**P 已完成三个种子×五折的 15 组训练，共 3,600 次更新；当前无执行阻塞。**','','## 结果','',
           'P 是独立候选，仅改变三路最后投影是否可更新；未叠加 N 的 LayerNorm 或 R 的权重衰减。直接控制为同一 EEG 修订父版本的 CCML-v2-E-control，N 单列为已完成的另一候选。主口径为各种子五折均值，再平均三个种子。','',
           '| 实验 | Macro-F1 | MDD recall | BA | NLL | Brier |','|---|---:|---:|---:|---:|---:|']
    for name,summary in [('同版本冻结表征控制',result['control']),('N：规范化（已有候选）',result['N_reference']),('P：仅训练最后投影和新头',result['summary'])]:
        m=summary['primary_seed_mean_of_fold_means'];lines.append('| '+name+' | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    positives=sum(v>0 for v in comparison['seed_f1_deltas'].values())
    lines+=['',f'P 相对直接控制：Macro-F1 **{delta["macro_f1"]*100:+.2f} 个百分点**，MDD recall {delta["mdd_recall"]*100:+.2f} 个百分点；{positives}/3 个种子 F1 提高。预定工程门槛'+('通过。' if g['development_criteria_passed'] else '未通过。'),
            f'NLL 变化 {delta["nll"]:+.6f}，Brier 变化 {delta["brier"]:+.6f}；这两项越低越好。门槛仍为 F1 至少 +0.01、至少两个种子提高、召回下降不超过 0.05，并完整披露概率质量变化。它不是显著性或独立确认。','',
            '| 种子 | 控制 F1 | N F1 | P F1 | P−控制 |','|---|---:|---:|---:|---:|']
    for s in ['11','22','33']:
        values=[result[k]['seed_reports'][s]['unweighted_fold_mean']['macro_f1'] for k in ['control','N_reference','summary']]
        lines.append('| '+s+' | '+' | '.join(f'{v:.6f}' for v in values)+f' | {values[2]-values[0]:+.6f} |')
    lines+=['','## 训练范围与来源','',
            '| 路径 | 最后投影 | 投影参数量 | 初始权重来源 |','|---|---|---:|---|',
            '| EEG | 1920→128 | 245,888 | 同种子、同折 CCML-v2-E 适配父模型 |',
            '| 语音 | 3584→128 | 458,880 | 同种子、同折历史语音父模型 |',
            '| 文本 | 1024→128 | 131,200 | 同种子、同折 fulltext 父模型 |','',
            '三路投影共 835,968 个参数，另有三个新头 774 个参数，总计 **836,742**。CNN/LSTM 与骨干以冻结 eval 模式导出投影前缓存，正式训练模型中不包含这些模块。缓存维度分别是 37×1920、37×3584、37×1024，不能用已有 128 维向量冒充可训练投影。',
            '固定投影 lr=1e-4、头 lr=1e-3；60 轮、batch 8、Adamax、weight_decay=0、全体可训练参数梯度上限 1；投影后的 Dropout 保持 eval。rho=1.25、delta=gamma=1、KL 退火 50 轮。最后一轮、阈值 0.5。没有搜索、早停或选择种子。',
            '新头与控制按相同种子独立初始化，未继承 E 适配头或 N/R 头。各折优化仅使用该折 train；上游监督拟合名单也须匹配该折。所有新增训练使用 CCML；历史语音/文本父权重仍有 CE 来源，不能称整个表征血缘均为 CCML。','',
            '## 数值复现与验收','',
            '- 初次缓存导出在语音路径严格复现检查处停止，当时未训练。诊断发现，历史语音/文本父特征使用 cuDNN TF32；关闭该模式时，首折语音表征最大误差约 3.80×10^-4，恢复原模式后原父输出精确一致。EEG 修订路径继续关闭 TF32。',
            '- 保留初次 `configuration.json` 和 `recovery/p_export_v1.py`，有效设置另存为 `configuration_v2.json`。这次修订仅恢复冻结上游的原数值路径，未修改验收容差、模型参数或训练计划。',
            f'- 15 组缓存均重建原父表征，复算控制概率最大误差为 {result["preflight"]["max_control_probability_error"]:.9g}；555 条控制分类全部一致。CPU 投影与原 GPU 导出允许固定小浮点误差，未按测试性能选数值模式。',
            '- 15 个新检查点受限读取成功，555 条预测逐条精确复现；初始化、每个投影的实际父权重、训练名单、批序和融合算术全部核对。15 个测试折指标与 scikit-learn 交叉验算。',
            '- 三路投影在每一折均收到梯度并发生实际权重更新。每轮损失分项、投影/头梯度范数，以及各路权重位移均已保存。',
            f'- 共 62 项测试通过，{result["protected_files_unchanged"]} 项历史文件哈希不变；没有修改旧实验或重训冻结骨干。','',
            '## 泛化与证据诊断','',
            '| 模态 | train F1 | val F1 | test F1 | test 平均 S |','|---|---:|---:|---:|---:|']
    d=result['diagnostics']
    for m in ['final','eeg','speech','text']:
        values=[d['train_val_test'][s][m]['macro_f1'] for s in ['train','val','test']]
        strength=d['mean_strength'].get(m)
        lines.append('| '+m+' | '+' | '.join(f'{v:.6f}' for v in values)+' | '+(f'{strength:.4f}' if strength is not None else '—')+' |')
    changes=d['changes_by_direction']
    lines+=['',f'三个种子的 111 条测试记录中，P 相对控制改变 {d["changed_count"]} 条分类：错变对 {changes.get("wrong_to_correct",0)}，对变错 {changes.get("correct_to_wrong",0)}。这些重复记录仍来自 37 个独立受试者。',
            '逐种子合并 OOF、ECE5、正确/错误不确定性、风险覆盖曲线、单模态指标、逐类 min 来源、公共/互补证据和权重位移见 `completion_audit.json`。证据强度不等于可靠性。','',
            '## 阶段结论与后续','']
    if g['development_criteria_passed']:
        lines.append('P 达到相对同版本控制的预定工程门槛，可保留为开发候选；仍需结合 N 的表现及 NLL/Brier 判断后续路线，不能宣称已经得到独立验证的最佳模型。')
    else:
        lines.append('P 未达到相对同版本控制的预定工程门槛，不自动升级为默认，也不以增加轮数或解冻整个 CNN/LSTM 作为本轮延伸。')
    p_minus_n=result['P_vs_N_descriptive_only']['macro_f1']
    lines.append(f'P 与已完成 N 的 F1 差值为 {p_minus_n:+.6f}。这是两个独立固定候选之间的开发比较；P 本轮没有叠加 N。')
    lines+=['',
            'Audit、E 控制、N/R、P 的近期固定实验已完成。本次未启动组合、证据校准或新的语音版本。若继续研究 N 与 P 的组合，须另立固定方案，与 N、P 及原控制共同比较，并保留交互效应解释；不能把组合收益全部归因于规范化或投影中的一项。',
            '所有结果基于已查看过的 37 人固定划分；新增种子不是新增独立样本。EEG 检查点尺度、原始坐标模板以及语音随机冻结编码器、单位置注意力、短音频零向量等限制继续披露。','',
            '当前完成状态见本目录 `status.json`。此前 N/R 状态中的 P 未开始描述为该阶段历史快照。','']
    path=OUT/'CCML-v2-P实验完成报告.md';content='\n'.join(lines)
    if path.exists():require(path.read_text(encoding='utf-8')==content,'Completed P report changed')
    else:path.write_text(content,encoding='utf-8')
    write_new(OUT/'status.json',{'status':'complete','date':'2026-09-22','P_runs':15,'updates':3600,
                              'trainable_parameters':836742,'configuration':descriptor(CONFIG),
                              'report':descriptor(path),'audit':descriptor(OUT/'completion_audit.json'),
                              'development_criteria_passed':g['development_criteria_passed'],'execution_blockers':[],
                              'N_P_combination_started':False,'recent_fixed_plan_complete':True,
                              'next_action':'Review N versus P and probability quality before a separately registered combination or input revision'})
    print(score);print(comparison);print(f'P complete: 15 runs, 3600 updates, {result["protected_files_unchanged"]} historical files unchanged.')


if __name__=='__main__':report(audit())
