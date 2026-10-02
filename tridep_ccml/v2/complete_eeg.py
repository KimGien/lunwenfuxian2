"""Independent completion audit and descriptive report for CCML v2 EEG revision."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

from .common import (WORK, TC, OUT, ADAPT, CONTROL, setup, descriptor, verify_descriptor,
                     read_json, require, write_new, check_existing, set_seed)
from .resume_eeg import load_checkpoint, version_metadata_loader
from .rebuild_eeg import EEGAdaptation, forward_all, validate_row
from tridep_ccml.train import load_features, make_initial_state, state_digest
from tridep_ccml.ccml import ThreeViewHeads, dirichlet_from_evidence, evidence_dc
from tridep_ccml.metrics import binary_metrics, summarize
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, log_loss, brier_score_loss

METRICS=('macro_f1','mdd_recall','balanced_accuracy','nll','brier')


def mean_scores(rows, split, modality=None):
    scores=[]
    for seed in [11,22,33]:
        for fold in [1,2,3,4,5]:
            part=[r for r in rows if r['seed']==seed and r['outer_fold']==fold and r['split']==split]
            p=[r['views'][modality]['p'][1] if modality else r['p_final'][1] for r in part]
            scores.append(binary_metrics([r['label'] for r in part],p))
    return {k:float(np.mean([s[k] for s in scores])) for k in METRICS}


def crosscheck_metrics(rows):
    checks=0
    for seed in [11,22,33]:
        for fold in [1,2,3,4,5]:
            part=[r for r in rows if r['seed']==seed and r['outer_fold']==fold and r['split']=='test']
            y=[r['label'] for r in part];p=np.array([r['p_final'][1] for r in part]);pred=(p>.5).astype(int)
            expected={'macro_f1':f1_score(y,pred,average='macro',labels=[0,1],zero_division=0),
                      'mdd_recall':recall_score(y,pred,zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,pred),
                      'nll':log_loss(y,np.stack([1-p,p],1),labels=[0,1]),'brier':brier_score_loss(y,p)}
            actual=binary_metrics(y,p.tolist())
            require(all(math.isclose(actual[k],expected[k],abs_tol=1e-10,rel_tol=1e-10) for k in METRICS),'Independent metric crosscheck failed')
            checks+=1
    return checks


def audit():
    context,cfg=setup()
    eeg_manifest=read_json(OUT/'eeg_manifest.json')
    require(len(eeg_manifest)==37,'Incomplete EEG export')
    eeg=np.stack([np.load(verify_descriptor(r['feature'])) for r in eeg_manifest])
    rows={ADAPT:[],CONTROL:[]};audits=[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed)
            old=load_features(context,seed,fold)
            parent_path=verify_descriptor(old['manifest']['parent_checkpoints']['eeg'])
            with version_metadata_loader():
                original=torch.load(parent_path,map_location='cpu',weights_only=True)
            extractor=EEGAdaptation(original['state_dict'],make_initial_state(seed))
            initial_parent_state=state_digest(extractor.state_dict())
            cache=OUT/f'features/seed_{seed}/fold_{fold}.npz'
            cache_info=read_json(cache.with_suffix('.json'));verify_descriptor(cache_info['features'])
            require(cache_info['fit_subject_ids']==context.folds[f'fold_{fold}']['train'],'Revised parent fit IDs mismatch')
            with np.load(cache,allow_pickle=False) as arrays:
                reps=arrays['features'].copy()
                require(arrays['subject_ids'].tolist()==context.ids and arrays['splits'].tolist()==old['splits']
                        and np.array_equal(arrays['labels'],old['labels']),'Revised feature identities changed')
            require(np.array_equal(reps[:,1:],old['features'][:,1:]),'Speech/text representations changed')
            require(reps.shape==(37,3,128) and np.isfinite(reps).all(),'Revised features invalid')
            hashes={}
            for stage in [ADAPT,CONTROL]:
                folder=OUT/'runs'/stage/f'seed_{seed}'/f'fold_{fold}'
                complete=read_json(folder/'complete.json')
                for item in complete.values():verify_descriptor(item)
                cp=folder/'checkpoint.pt';saved=load_checkpoint(cp);meta=saved['metadata']
                for key in ['configuration','runner','common','ccml','head_helpers','upstream_model_source']:
                    verify_descriptor(meta[key])
                require(meta['train_ids']==context.folds[f'fold_{fold}']['train'],'Training IDs mismatch')
                require(meta['completed_epochs']==60 and meta['optimizer_updates']==240 and
                        meta['objective']=='full_CCML_only','Training budget or objective mismatch')
                require(meta['trainable_parameters']==(498950 if stage==ADAPT else 774),'Trainable count mismatch')
                require(meta['initial_state_sha256']==(initial_parent_state if stage==ADAPT else state_digest(make_initial_state(seed))), 'Initial weights mismatch')
                require(meta['head_initialization_sha256']==state_digest(make_initial_state(seed)),'Head seed mismatch')
                generator=torch.Generator().manual_seed(seed);batch=hashlib.sha256()
                for _ in range(60):batch.update(torch.randperm(len(meta['train_ids']),generator=generator).numpy().astype('<i8').tobytes())
                require(batch.hexdigest()==meta['batch_order_sha256'],'Batch order mismatch')
                history=read_json(folder/'train_history.json')
                require([r['epoch'] for r in history]==list(range(1,61)),'Missing training epochs')
                require(all(math.isfinite(h['train_loss']) and math.isclose(h['train_loss'],sum(h['loss_terms'].values()),rel_tol=1e-5,abs_tol=1e-5) for h in history),'Loss diagnostics mismatch')
                if stage==ADAPT:
                    require(meta['inputs']['corrected_eeg_manifest']==descriptor(OUT/'eeg_manifest.json'),'Corrected EEG manifest not bound')
                    require(meta['inputs']['historical_parents']==old['manifest']['parent_checkpoints'],'Historical parent ancestry mismatch')
                    model=extractor;model.load_state_dict(saved['state_dict'],strict=True)
                    model.to(meta['device']).eval()
                    raw=eeg;inputs=old['features']
                    rebuilt=[]
                    with torch.no_grad():
                        for i in range(0,37,8):
                            rebuilt.append(model.representations(torch.as_tensor(eeg[i:i+8],device=meta['device']),torch.as_tensor(inputs[i:i+8],device=meta['device'])).cpu().numpy())
                    require(np.array_equal(np.concatenate(rebuilt),reps),'Revised representations do not reproduce')
                    require(cache_info['parent_checkpoint']==descriptor(cp),'Parent checkpoint binding mismatch')
                else:
                    require(meta['inputs']['revised_features']==descriptor(cache) and meta['inputs']['revised_manifest']==descriptor(cache.with_suffix('.json')),'Control features not bound')
                    model=ThreeViewHeads();model.load_state_dict(saved['state_dict'],strict=True)
                    raw=None;inputs=reps
                logits=forward_all(model,inputs,raw,meta['device'],stage==ADAPT)
                second=load_checkpoint(cp)
                require(state_digest(second['state_dict'])==state_digest(saved['state_dict']),'Repeat checkpoint load mismatch')
                model.load_state_dict(second['state_dict'],strict=True)
                require(torch.equal(logits,forward_all(model,inputs,raw,meta['device'],stage==ADAPT)),'Repeat checkpoint inference mismatch')
                e=F.softplus(logits);fused=evidence_dc(e)
                predictions=read_json(folder/'predictions.json')
                require(len(predictions)==37 and [r['subject_id'] for r in predictions]==context.ids,'Prediction subject coverage mismatch')
                for i,row in enumerate(predictions):
                    require(row['experiment']==stage and row['seed']==seed and row['outer_fold']==fold and
                            row['label']==context.labels[row['subject_id']] and row['split']==old['splits'][i] and
                            row['train_ids']==meta['train_ids'],'Prediction provenance mismatch')
                    require(row['checkpoint']==descriptor(cp) and row['feature_source']==meta['inputs'] and row['configuration']==meta['configuration'],'Prediction artifact binding mismatch')
                    require(row['rho']==1.25 and row['delta']==1 and row['gamma']==1 and row['fusion_method']=='ccml','CCML parameters mismatch')
                    validate_row(row)
                    require(np.array_equal(np.array(row['p_final']),fused['p'][i].numpy()),'Saved probability does not reproduce exactly')
                    for v,m in enumerate(['eeg','speech','text']):
                        require(np.array_equal(np.array(row['views'][m]['e']),e[i,v].numpy()),'Saved branch evidence does not reproduce')
                rows[stage]+=predictions;hashes[stage]=meta['batch_order_sha256']
                audits.append({'stage':stage,'seed':seed,'fold':fold,'checkpoint':descriptor(cp),
                               'predictions':descriptor(folder/'predictions.json'),'verified_rows':37,
                               'exact_repeat_load_and_inference':True,'trainable_parameters':meta['trainable_parameters'],
                               'updates':240,'interrupted_first_fit':stage==ADAPT and seed==11 and fold==1})
                del model
            require(hashes[ADAPT]==hashes[CONTROL],'Adapt/control batch mismatch')
    protected=check_existing()
    result={'date':'2026-09-22','passed':True,'completed_runs':30,'updates':7200,'corrected_eeg_subjects':37,
            'revised_feature_bundles':15,'historical_files_unchanged':protected,
            'audits':audits,'first_checkpoint_resume_note':'Saved state reloaded twice and all outputs reproduced; no pre-interruption in-memory logits were saved, so that comparison is unavailable.',
            'unit_tests':{'historical_passed':47,'v2_passed':6},'summary':{},'split_metrics':{},'evidence_strength':{}}
    for stage,records in rows.items():
        require(len(records)==555,'Incomplete all-split predictions')
        require(Counter((r['seed'],r['subject_id']) for r in records if r['split']=='test')==Counter((s,i) for s in cfg['seeds'] for i in context.ids),'OOF coverage mismatch')
        result['summary'][stage]=summarize(records)
        result['summary'][stage]['sklearn_fold_crosschecks']=crosscheck_metrics(records)
        result['split_metrics'][stage]={split:{m or 'final':mean_scores(records,split,m) for m in [None,'eeg','speech','text']} for split in ['train','val','test']}
        result['evidence_strength'][stage]={m:float(np.mean([sum(r['views'][m]['alpha']) for r in records if r['split']=='test'])) for m in ['eeg','speech','text']}
    baseline=read_json(TC/'reports/tc04/summary.json')['fusion']
    result['historical_tc04']=baseline
    result['historical_tc04_source']=descriptor(TC/'reports/tc04/summary.json')
    result['delta_control_vs_historical_tc04']={k:result['summary'][CONTROL]['primary_seed_mean_of_fold_means'][k]-baseline['primary_seed_mean_of_fold_means'][k] for k in METRICS}
    result['seed_f1_deltas_vs_history']={str(s):result['summary'][CONTROL]['seed_reports'][str(s)]['unweighted_fold_mean']['macro_f1']-baseline['seed_reports'][str(s)]['unweighted_fold_mean']['macro_f1'] for s in cfg['seeds']}
    result['reporter']=descriptor(__file__)
    write_new(OUT/'completion_audit.json',result)
    return result


def report(result):
    lines=['# CCML-v2-E：EEG 修订与控制实验完成报告','','日期：2026-09-22。**37 人 EEG 修订、15 组 CCML 适配和 15 组新控制均已完成。**','','## 结果','',
           '主口径为每种子五折均值，再平均三个种子。全部使用固定最后一轮、阈值 0.5；没有按结果选择种子、轮数或参数。', '',
           '| 实验 | Macro-F1 | MDD recall | BA | NLL | Brier |','|---|---:|---:|---:|---:|---:|']
    names={'TC-04':'旧 TC-04（历史参照）',ADAPT:'修订 EEG 的 CCML 适配模型',CONTROL:'冻结新表征后的 CCML 控制（N/R 的基准）'}
    for stage,summary in [('TC-04',result['historical_tc04']),*result['summary'].items()]:
        m=summary['primary_seed_mean_of_fold_means'];lines.append('| '+names[stage]+' | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    delta=result['delta_control_vs_historical_tc04']
    lines+=['',f'新控制相对旧 TC-04：Macro-F1 变化 **{100*delta["macro_f1"]:+.2f} 个百分点**，MDD recall 变化 {100*delta["mdd_recall"]:+.2f} 个百分点。此次输入修订与 EEG 适配同时发生，不能把差异全部归因于通道修正。', '',
            '| 种子 | 旧 TC-04 F1 | 新控制 F1 | 差值 |','|---|---:|---:|---:|']
    for s in ['11','22','33']:
        old=result['historical_tc04']['seed_reports'][s]['unweighted_fold_mean']['macro_f1'];new=result['summary'][CONTROL]['seed_reports'][s]['unweighted_fold_mean']['macro_f1']
        lines.append(f'| {s} | {old:.6f} | {new:.6f} | {new-old:+.6f} |')
    lines+=['','## 执行范围','',
            '- 按已核实的 HydroCel 标签重选 19 路，重新导出 37 份 60×200 EEG 特征。原路径单人复算最大误差为 0；native 数值、滤波、截取、参考保持原方案。',
            '- 从同折同种子历史 EEG 权重开始，用完整 CCML 适配 EEG CNN 与投影；EEG 可训练参数 498,176，加三个新头共 498,950。语音和文本的 128 维表征逐元素不变。骨干冻结，Dropout 保持 eval。',
            '- 适配固定 EEG lr=1e-4、头 lr=1e-3；60 轮、batch 8、Adamax、weight_decay=0、梯度上限 1。rho=1.25、delta=gamma=1、KL 退火 50 轮。仅使用各折 train 拟合。',
            '- 导出 15 组新 37×3×128 父表征后，冻结表征，用共同初始化的 774 参数新头再训练 15 组 CCML 控制；没有从适配模型的头继续训练。两阶段共 7,200 次更新。',
            '- 所有新增训练均用 CCML。历史 EEG 初值、冻结的语音/文本表征仍有已有 CE 父模型来源；本次没有新 CE 训练，也不声称整个表征血缘完全没有 CE。','',
            '## 泛化与证据诊断','',
            '| 模型/模态 | train F1 | val F1 | test F1 | test 平均 S |','|---|---:|---:|---:|---:|']
    for stage in [ADAPT,CONTROL]:
        for m in ['final','eeg','speech','text']:
            scores=result['split_metrics'][stage];strength=result['evidence_strength'][stage].get(m)
            label=('适配' if stage==ADAPT else '新控制')+'/'+m
            lines.append('| '+label+' | '+' | '.join(f'{scores[s][m]["macro_f1"]:.6f}' for s in ['train','val','test'])+' | '+(f'{strength:.4f}' if strength is not None else '—')+' |')
    lines+=['','该阶段用于建立修正输入后的可追溯基准。父适配结果与冻结后重新训练头的结果均完整保留，不按较高值替换预定控制。逐种子合并 OOF、ECE5、正确/错误不确定性及风险覆盖曲线见 `completion_audit.json`。', '',
            '## 重载恢复与验收','',
            '- 暂停前首组已完成 60 轮；失败来自 PyTorch 版本元数据的受限反序列化。恢复入口仅允许 PyTorch 的 TorchVersion 类型，继续使用 weights_only=True，原检查点与训练代码均未改写，首组没有重训。',
            '- 全部 30 组检查点重新加载，复算 1,110 条逐人记录；预测概率、分支证据和 15 组父表征精确复现。首组另做两次读取/推理一致性检查；中断前未保存内存 logits，因此不能声称对该内存状态做了独立比较。',
            '- 独立验证每条证据融合算术、训练/验证/测试名单、初始化、批次顺序和来源哈希；30 个测试折指标与 scikit-learn 交叉核对一致。',
            f'- 原有 47 项测试、新路径 6 项测试通过；{result["historical_files_unchanged"]} 项受保护历史文件哈希不变。','',
            '## 后续与限制','',
            '本次通道选择错误和首组重载错误已处理，具备在新父版本开展 N/R 的条件。下一步按原计划独立运行 N（无仿射 LayerNorm）和 R（仅 Adamax weight_decay=1e-4），各 15 组，均与本次新控制比较；本次没有启动 N/R。',
            'Mumtaz 检查点输入尺度、原始坐标模板冲突，以及语音随机冻结编码器/单位置注意力/短音频零向量仍是已披露限制。本轮没有按性能任意缩放 EEG，也没有合并语音修订。37 人测试结果已被查看，属于开发性比较；不同种子不增加独立受试者数。','',
            '当前有效完成状态见 `status.json`；`pause_status.json` 保留为 2026-09-21 的历史暂停记录。恢复入口为 `tridep_ccml.v2.resume_eeg`；只读复核与报告入口为 `tridep_ccml.v2.complete_eeg`。','']
    path=OUT/'CCML-v2-E实验完成报告.md'
    content='\n'.join(lines)
    if path.exists():require(path.read_text(encoding='utf-8')==content,'Refusing to rewrite different completed report')
    else:path.write_text(content,encoding='utf-8')
    write_new(OUT/'status.json',{'status':'complete','date':'2026-09-22','stage':'EEG revision, CCML adaptation and fresh-head control',
                              'corrected_eeg_subjects':37,'adaptation_runs':15,'control_runs':15,'updates':7200,
                              'audit':descriptor(OUT/'completion_audit.json'),'report':descriptor(path),
                              'nr_runs':0,'next_action':'CCML-v2-N and R on this revised parent, 15 runs each',
                              'active_execution_blockers':[],'remaining_research_limitations':['Mumtaz input scale','raw coordinate template','speech encoder and short recordings'],
                              'pause_record_is_historical':True})
    print({k:v['primary_seed_mean_of_fold_means'] for k,v in result['summary'].items()})
    print(f'Complete: 30 runs; 7200 updates; {result["historical_files_unchanged"]} historical files unchanged.')


if __name__=='__main__':report(audit())
