"""Replay every G/F nested run and selection; report both fixed candidates."""
from __future__ import annotations
from collections import Counter
import hashlib
import math
from . import optional_extensions as x
from .common import TC, read_json,write_new,descriptor,verify_descriptor,require,set_seed
from .rebuild_eeg import validate_row
from .complete_eeg import mean_scores,crosscheck_metrics
from .complete_p import gate
from .complete_nr import diagnostics
from .complete_scale import diagnostics as scale_diagnostics
from tridep_ccml.metrics import summarize
from tridep_ccml.train import state_digest
import numpy as np
import torch


def inspect_run(context,data,cfg,kind,seed,fold,inner,value,train_ids,split_map):
    folder=x.run_folder(kind,seed,fold,inner,value)
    for d in read_json(folder/'complete.json').values():verify_descriptor(d)
    side=read_json(folder/'checkpoint.json');cp=verify_descriptor(side['checkpoint'])
    saved=torch.load(cp,map_location='cpu',weights_only=True);meta=saved['metadata']
    require(meta==side['metadata'] and meta['experiment']==x.STAGES[kind] and meta['kind']==kind and
            meta['seed']==seed and meta['fold']==fold and meta['inner_fold']==inner and meta['value']==value,'Run identity mismatch')
    require(meta['configuration']==descriptor(x.CONFIG) and meta['inputs']==descriptor(x.s.OUT/'input_manifest.json') and
            meta['train_ids']==train_ids and meta['split_map']==split_map and meta['supervised_parent'] is None and
            meta['cohort_sha256']==context.cohort_sha256 and meta['folds_sha256']==context.folds_sha256,'Source/split mismatch')
    require(meta['epochs']==60 and meta['updates']==60*math.ceil(len(train_ids)/8),'Budget mismatch')
    require(set(train_ids)<=set(context.folds[f'fold_{fold}']['train']),'Fit leakage')
    if inner is not None:require(meta['selection'] is None,'Inner run must not use a selection')
    else:require(meta['selection']==descriptor(x.OUT/f'selections/{kind}/seed_{seed}/fold_{fold}.json'),'Selection binding mismatch')
    set_seed(seed);model=x.ExtensionModel(seed,kind)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    require(state_digest(initial)==meta['initial_state_sha256'] and x.base_hash(model)==meta['initial_base_sha256'],'Initial state mismatch')
    require(sum(p.numel() for p in model.parameters())==meta['parameters']==cfg[kind]['parameters'],'Parameter scope mismatch')
    gen=torch.Generator().manual_seed(seed);h=hashlib.sha256()
    for _ in range(60):h.update(torch.randperm(len(train_ids),generator=gen).numpy().astype('<i8').tobytes())
    require(h.hexdigest()==meta['batch_order_sha256'],'Batch order mismatch')
    for m in x.MODALITIES:
        require(any(not torch.equal(initial[k],saved['state_dict'][k]) for k in initial if k.startswith('projections.'+m)), 'Projection never updated')
    model.load_state_dict(saved['state_dict'],strict=True)
    rows=read_json(folder/'predictions.json')
    require(rows==x.predictions(model,data,context,meta,cp,split_map),'Exact prediction replay mismatch')
    history=read_json(folder/'train_history.json')
    require([r['epoch'] for r in history]==list(range(1,61)),'Missing epochs')
    for r in history:
        require(math.isclose(r['train_objective'],sum(r['ccml_loss_terms'].values())+r['extra_loss'],abs_tol=2e-5,rel_tol=2e-5),'Loss decomposition mismatch')
        require(math.isfinite(r['gradient_mean']) and r['gradient_mean']>0 and all(.5<=v<=2. for v in r['fit_scale_min']+r['fit_scale_max']),'Invalid gradient/scale')
    if kind=='G':require(any(r['gate_gradient_mean']>0 for r in history),'Gate never received gradients')
    for r in rows:
        validate_row(r)
        raw=np.array([r['views'][m]['e_raw'] for m in x.MODALITIES])
        scale=np.array(r['scales']);strength=raw.sum(-1)+2
        q=np.stack((2/strength,np.abs(raw[:,1]-raw[:,0])/strength),-1)
        require(np.allclose(q,r['quality_features'],atol=2e-6,rtol=2e-5),'Quality feature mismatch')
        require(np.allclose(raw*scale[:,None],[r['views'][m]['e'] for m in x.MODALITIES],atol=2e-6,rtol=2e-5),'Scale equation mismatch')
        require(np.isfinite(scale).all() and (scale>=.5).all() and (scale<=2.).all(),'Scale bounds failed')
        if kind=='F':require(np.array_equal(scale,np.ones(3)),'F unexpectedly scales evidence')
    return rows,meta


def audit():
    context,cfg,data=x.load_setup();x.s.build_inputs(context)
    splits=read_json(x.s.OUT/'nested_splits.json')
    require(splits==x.s.nested_splits(context),'Nested splits changed')
    lock=read_json(x.OUT/'all_selections_locked.json')
    require(len(lock['selections'])==30 and lock['configuration']==descriptor(x.CONFIG),'Incomplete selection lock')
    for d in lock['selections']:verify_descriptor(d)
    outer={k:[] for k in x.VALUES};control=[];checks=[];selections={k:[] for k in x.VALUES}
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            folder=x.s.run_folder(seed,fold,None,'control',0.)
            for d in read_json(folder/'complete.json').values():verify_descriptor(d)
            control+=read_json(folder/'predictions.json')
    for kind in x.VALUES:
        for seed in cfg['seeds']:
            for fold in cfg['folds']:
                choices={}
                for value in x.VALUES[kind]:
                    choices[value]=[]
                    for part in splits['folds'][str(fold)]:
                        split_map={**dict.fromkeys(part['train_ids'],'train'),**dict.fromkeys(part['val_ids'],'inner_val')}
                        rows,meta=inspect_run(context,data,cfg,kind,seed,fold,part['inner_fold'],value,part['train_ids'],split_map)
                        choices[value]+=[r for r in rows if r['split']=='inner_val']
                        checks.append({'kind':kind,'seed':seed,'outer_fold':fold,'inner_fold':part['inner_fold'],'value':value,'updates':meta['updates'],'exact_records':len(rows)})
                choice=x.select_candidate(kind,choices,context.folds[f'fold_{fold}']['train'])
                saved=read_json(x.OUT/f'selections/{kind}/seed_{seed}/fold_{fold}.json')
                require(all(saved[k]==v for k,v in choice.items()) and not saved['outer_val_used'] and not saved['outer_test_used'],'Selection recomputation mismatch')
                for d in saved['sources']:verify_descriptor(d)
                selections[kind].append(saved)
                outer_split=context.folds[f'fold_{fold}'];split_map={sid:p for p,ids in outer_split.items() for sid in ids}
                rows,meta=inspect_run(context,data,cfg,kind,seed,fold,None,choice['selected_value'],outer_split['train'],split_map)
                base=read_json(x.s.run_folder(seed,fold,None,'control',0.)/'checkpoint.json')['metadata']
                require(meta['train_ids']==base['train_ids'] and meta['initial_base_sha256']==base['initial_state_sha256'] and
                        meta['batch_order_sha256']==base['batch_order_sha256'],'Outer control pairing mismatch')
                outer[kind]+=rows
                checks.append({'kind':kind,'seed':seed,'outer_fold':fold,'inner_fold':None,'value':choice['selected_value'],'updates':meta['updates'],'exact_records':len(rows)})
                print(f'{kind} replay seed={seed} outer={fold}: 6 inner + 1 outer exact; selection recomputed.',flush=True)
    require(len(checks)==210 and sum(r['updates'] for r in checks)==39600,'Total budget mismatch')
    for kind,rows in outer.items():
        require(len(rows)==555 and Counter((r['seed'],r['subject_id']) for r in rows if r['split']=='test')==Counter((s,i) for s in cfg['seeds'] for i in context.ids),'Outer coverage mismatch')
        write_new(x.OUT/f'outer_predictions_{kind}.json',rows)
    all_rows={'control':control,**outer};summaries={k:summarize(v) for k,v in all_rows.items()}
    old=read_json(x.s.OUT/'completion_audit.json')['summaries']['control']
    require(summaries['control']['primary_seed_mean_of_fold_means']==old['primary_seed_mean_of_fold_means'],'Control summary changed')
    result={'passed':True,'configuration':descriptor(x.CONFIG),'budget':cfg['budget'],'checks':checks,
            'exact_new_records':sum(r['exact_records'] for r in checks),'selections':selections,
            'selection_counts':{k:dict(Counter(str(r['selected_value']) for r in v)) for k,v in selections.items()},
            'summaries':summaries,'comparisons':{k:gate(summaries[k],summaries['control']) for k in x.VALUES},
            'diagnostics':{k:diagnostics(v,control) for k,v in all_rows.items()},
            'scale_and_fusion_diagnostics':{k:scale_diagnostics(v) for k,v in all_rows.items()},
            'test_scale_summary':{k:{m:{'min':float(np.min([r['scales'][j] for r in v if r['split']=='test'])),
                    'mean':float(np.mean([r['scales'][j] for r in v if r['split']=='test'])),
                    'max':float(np.max([r['scales'][j] for r in v if r['split']=='test']))} for j,m in enumerate(x.MODALITIES)} for k,v in outer.items()},
            'independent_metric_crosschecks':sum(crosscheck_metrics(v) for v in outer.values()),
            'protected_files_unchanged':x.protect_history(),'independent_validation':False}
    write_new(x.OUT/'completion_audit.json',result)
    return result


def report(result):
    summaries=result['summaries'];comparison=result['comparisons']
    lines=['# CCML-v2-G / F 独立可选扩展完成报告','','日期：2026-09-24。两项固定扩展全部完成，保留全部候选和旧阶段。','',
           '## 主结果','','| 条件 | Macro-F1 | MDD recall | Balanced accuracy | NLL | Brier |','|---|---:|---:|---:|---:|---:|']
    for kind,title in [('control','S-control（复用）'),('G','G：逐人模态门控'),('F','F：额外最终融合监督')]:
        score=summaries[kind]['primary_seed_mean_of_fold_means']
        lines.append('| '+title+' | '+' | '.join(f'{score[m]:.6f}' for m in x.s.METRICS)+' |')
    lines+=['','主口径：每种子五折均值→三个种子均值。G/F 均直接与同版本 S-control 比较，未叠加 S，也未叠加彼此。NLL/Brier 越低越好。','']
    for kind in x.VALUES:
        c=comparison[kind];d=c['delta']
        lines.append(f'- {kind}：F1 {100*d["macro_f1"]:+.3f} 个百分点；{sum(v>0 for v in c["seed_f1_deltas"].values())}/3 种子严格提高；召回 {100*d["mdd_recall"]:+.3f} 个百分点；NLL {d["nll"]:+.6f}、Brier {d["brier"]:+.6f}。'+('通过' if c['gate']['development_criteria_passed'] else '未通过')+'预定开发门槛。')
    lines+=['','| 种子 | S-control F1 | G F1 | F F1 |','|---|---:|---:|---:|']
    for seed in ['11','22','33']:
        lines.append('| '+seed+' | '+' | '.join(f'{summaries[k]["seed_reports"][seed]["unweighted_fold_mean"]["macro_f1"]:.6f}' for k in ['control','G','F'])+' |')
    lines+=['','## 方法与训练内选择','',
            '沿用 S-backbone-mean-v1：E 修订 EEG 的冻结骨干时间均值、AS 覆盖完整语音均值、全文 MacBERT 题目 CLS 均值；各路新投影、无仿射 LayerNorm128 和新证据头。没有 MODMA 监督父权重，所有内层模型仅在自己的 train 拟合。旧 AS-NP/AS-control 不作为本轮直接对照。','',
            '**G**：每路从本人的原始 u 和绝对类别概率差生成两个停止梯度的描述量，独立 2→1 线性门控，共 9 个零初始化参数；s=exp(log(2)·tanh(a)) 限制在 [0.5,2]，解耦前缩放证据。目标为完整 CCML＋λ·mean(log(s)²)，λ=0.01/0.1。共 223,375 参数。','',
            '**F**：尺度固定为 1，保留全部原 CCML 损失，增加 β×最终 Dirichlet 期望交叉熵；不加额外 final KL，β=0.1/1.0。共 223,366 参数。它是 CCML 监督扩展，不是新的普通 CE 对照，也未替换原监督项。','',
            '两者均保持 rho=1.25、delta=gamma=1、KL 退火 50 轮。按三折 inner-val 融合 NLL 均值选择；1e-12 内平票 G 选较强收缩，F 选较弱额外监督。30 份选择全部锁定后，才从共同初始化运行外层重拟合。外层 val/test 没有用于选择。','',
            '参数选择次数：'+ '；'.join(k+' '+', '.join(f'{v} → {n}/15' for v,n in sorted(result['selection_counts'][k].items())) for k in x.VALUES)+'。','',
            '## 泛化与融合诊断','','| 条件 | Train F1 | Val F1 | Test F1 | Train−Test |','|---|---:|---:|---:|---:|']
    for k in ['control','G','F']:
        d=result['diagnostics'][k]['train_val_test'];values=[d[p]['final']['macro_f1'] for p in ['train','val','test']]
        lines.append(f'| {k} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {values[0]-values[2]:.6f} |')
    lines+=['','| 模态 | G 测试尺度最小 | G 平均 | G 最大 |','|---|---:|---:|---:|']
    for m in x.MODALITIES:
        d=result['test_scale_summary']['G'][m];lines.append(f'| {m} | {d["min"]:.6f} | {d["mean"]:.6f} | {d["max"]:.6f} |')
    lines+=['','| 输出 | 控制 Test F1 | G Test F1 | F Test F1 |','|---|---:|---:|---:|']
    for m in x.MODALITIES:
        lines.append('| '+m+' | '+' | '.join(f'{result["diagnostics"][k]["train_val_test"]["test"][m]["macro_f1"]:.6f}' for k in ['control','G','F'])+' |')
    lines+=['']
    for k in x.VALUES:
        d=result['diagnostics'][k];counts=d['changes_by_direction']
        lines.append(f'{k} 相对控制的 111 条重复种子测试记录中，{d["changed_count"]} 条分类改变：错变对 {counts.get("wrong_to_correct",0)}、对变错 {counts.get("correct_to_wrong",0)}。')
    lines+=['','证据原值/缩放值、尺度、门控输入、alpha/p/u、min 来源和公共/互补量均保存；逐种子 OOF、ECE5、u 与风险覆盖、分支融合冲突见完整验收文件。u 只是证据不足程度，门控与证据头可能相互补偿，不能当作可靠性的因果估计。','',
            '## 预算与验收','',
            'G/F 各 90 组内层训练、15 组外层重拟合，共新增 210 组、39,600 次更新。S-control 15 组复用、零新增训练；不能重复计入。全部 60 轮、batch 8、Adamax、投影 lr=1e-4、头/门控 lr=1e-3、weight_decay=0、梯度裁剪 1、固定末轮与 0.5 阈值。不同额外项的总 loss 不直接横比。','',
            f'{read_json(x.OUT/"test_results.json")["tests_run"]} 项测试通过；训练前 555 条控制预测精确重载；完成后 210 个新检查点、{result["exact_new_records"]:,} 条预测精确重现，30 份内层选择复算，30 个外层测试折指标独立交叉核验通过；{result["protected_files_unchanged"]:,} 项历史文件哈希不变。','',
            '## 阶段决定与限制','',
            '两项均按固定预算报告，未通过门槛者不升级；即便通过也仅保留为开发候选，不自动合并 G+F 或追加强度搜索。研究仍来自同一已反复查看的 37 人；合法内层选择不能消除以往开发选择偏差，也未构成独立人群验证。EEG 幅值与坐标来源限制、原 TC-06 的范围均不因本轮完成而改变。','',
            '- [固定计划](../../../output/plans/CCML_v2_G_F_独立扩展固定计划.md)','- [配置](configuration.json)','- [输入和内层名单来源](preflight.json)',
            '- [全部选择锁](all_selections_locked.json)','- [验收与诊断](completion_audit.json)','- [G 外层预测](outer_predictions_G.json)',
            '- [F 外层预测](outer_predictions_F.json)','- [完成状态](status.json)','']
    path=x.OUT/'CCML-v2-GF实验完成报告.md';content='\n'.join(lines)
    if path.exists():require(path.read_text(encoding='utf8')==content,'Completed report changed')
    else:path.write_text(content,encoding='utf8')
    write_new(x.OUT/'status.json',{'status':'complete','date':'2026-09-24','new_runs':210,'updates':39600,'configuration':descriptor(x.CONFIG),
              'plan':descriptor(x.PLAN),'report':descriptor(path),'audit':descriptor(x.OUT/'completion_audit.json'),
              'passed_gates':{k:c['gate']['development_criteria_passed'] for k,c in comparison.items()},
              'independent_validation':False,'next_action':'No further optional experiment or search registered'})
    print({k:v['primary_seed_mean_of_fold_means'] for k,v in summaries.items()},flush=True)
    print({k:c['gate'] for k,c in comparison.items()},flush=True)


if __name__=='__main__':report(audit())
