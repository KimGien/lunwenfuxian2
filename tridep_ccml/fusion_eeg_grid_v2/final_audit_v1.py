"""Separate read-only verifier. Outer data stays locked until all1200 and owner release."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from . import common as C

SOURCE=Path(__file__).resolve()
CONTROL=C.OUT/'final_audit_v1'
BASELINE=.7255026455026455
KEYS=('macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier')

def prepare():
    path=CONTROL/'protocol.json'
    value={'schema':'fusion_final_audit_v1','source':C.desc(SOURCE),
      'required_models':1200,'required_conditions':80,'required_ST':75,'required_EST':1125,
      'outer_read_prerequisites':['owner absent and native training tree released','all1200 completions valid',
                                  'all1200 completed fixed-checkpoint evaluation'],
      'metrics_implementation':'independent NumPy confusion counts and rank-sum AUC; no producer metrics import',
      'threshold':'>0.5','metric_atol':1e-12,'normalization':'recompute only each fold train',
      'cost_policy':'separate recorded categories; no sum of overlapping wall envelope and nested durations; unknown null',
      'source_revisions':'only previously declared historical plan revision; no model/input exception',
      'final_deliverables':['76-row baseline/EST primary table','15x5 EST matrix','5-row ST table',
                            'seed/fold metrics','OOF','paired differences','cost and source acceptance'],
      'no_optimizer_or_neural_forward':True}
    C.save(path,value,immutable=True)
    return value

def read_gate():
    prepare()
    C.check_pause()
    C.require(not (C.OUT/'control/owner.json').exists() and not (C.OUT/'control/queue.lock').exists(),
              'Owner must release before final audit')
    active=[r for r in C.inventory() if 'fusion_eeg_grid_v2.run' in r['command'] and '--run' in r['command']]
    C.require(not active,'Native training queue still exists')
    ids=list(C.identities());C.require(len(ids)==1200,'Matrix identity size differs')
    # Only completion files and provenance are read here. No outer probabilities.
    completions=[]
    for identity in ids:
        p=C.fitdir(identity)/'complete.json'
        C.require(p.exists(),'Missing formal fit '+str(identity))
        v=C.read(p)
        C.require(v['identity']==identity and v['passed'] and v['test_forwards']==0,
                  'Invalid formal completion '+str(identity))
        for key in ('best','latest','metadata','trajectory'):C.checked(v[key])
        completions.append(C.desc(p))
    evaluation=C.read(C.OUT/'evaluation_complete.json')
    C.require(evaluation['passed'] and evaluation['models']==evaluation['independent_replay_models']==1200,
              'All1200 independent model replay missing')
    C.save(CONTROL/'outer_read_gate.json',{'passed':True,'models':1200,'completion_sources':completions,
                 'evaluation_complete':C.desc(C.OUT/'evaluation_complete.json')},immutable=True)
    return ids

def independent_metrics(rows, np):
    yy=np.asarray([r['label'] for r in rows],dtype=np.int64)
    pp=np.asarray([r['p_mdd'] for r in rows],dtype=np.float64)
    C.require(len(rows)>0 and np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all(),'Invalid probability rows')
    pred=(pp>.5).astype(np.int64)
    matrix=np.bincount(2*yy+pred,minlength=4).reshape(2,2)
    f1=[]
    for label in (0,1):
        tp=int(matrix[label,label]);denom=int(matrix[label].sum()+matrix[:,label].sum())
        f1.append(2*tp/denom if denom else 0.)
    recall=float(matrix[1,1]/matrix[1].sum());specificity=float(matrix[0,0]/matrix[0].sum())
    # Alternative to producer pairwise AUC: average ranks for tied observations.
    order=np.argsort(pp,kind='stable');ranks=np.empty(len(pp),dtype=np.float64);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and pp[order[j]]==pp[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2
        i=j
    npos=int(yy.sum());nneg=len(yy)-npos
    auc=float((ranks[yy==1].sum()-npos*(npos+1)/2)/(npos*nneg))
    true=np.where(yy==1,pp,1-pp)
    return {'macro_f1':mean(f1),'balanced_accuracy':(recall+specificity)/2,'mdd_recall':recall,
            'auc':auc,'nll':float(-np.log(np.clip(true,1e-15,1)).mean()),
            'brier':float(np.square(pp-yy).mean()),'confusion_matrix':matrix.tolist()}

def verify_sources():
    total=0
    for filename in ('registration.json','source_registration.json','feature_gate.json'):
        value=C.read(C.OUT/filename)
        for source in value['sources']:C.checked(source);total+=1
    source=C.read(C.OUT/'source_registration.json')
    for revision in source['historical_document_revisions']:
        C.checked(revision['current_document'])
        C.require(revision['role']=='plan','Undeclared substantive source exception')
    # Every cache remains bound to the same subject, label, role and checkpoint.
    reg=C.read(C.OUT/'registration.json')
    C.require(reg['formal_fits']==1200 and reg['excluded']=='E14' and reg['EEG']==C.EEG,
              'Registered final scope changed')
    return {'passed':True,'checked_source_descriptors':total,
      'registration':C.desc(C.OUT/'registration.json'),'source_registration':C.desc(C.OUT/'source_registration.json'),
      'feature_gate':C.desc(C.OUT/'feature_gate.json'),'historical_document_revisions':source['historical_document_revisions']}

def verify_fit(identity,np,torch):
    directory=C.fitdir(identity);complete=C.read(directory/'complete.json');metadata=C.read(directory/'metadata.json')
    latest=torch.load(C.checked(complete['latest']),map_location='cpu',weights_only=False)
    best=torch.load(C.checked(complete['best']),map_location='cpu',weights_only=False)
    trajectory=C.read(C.checked(complete['trajectory']))
    binding=metadata['input_binding'];ntrain=len(binding['roles']['train'])
    C.require(metadata['identity']==latest['identity']==best['identity']==identity,'Model identity mismatch')
    C.require(60<=latest['epoch']<=120 and latest['epoch']==metadata['epochs']==len(trajectory), 'Epoch budget mismatch')
    C.require(latest['updates']==metadata['updates']==len(trajectory)*math.ceil(ntrain/8), 'Actual update count mismatch')
    for step in latest['optimizer']['state'].values():
        C.require(int(step['step'])==latest['updates'],'Optimizer state does not evidence every recorded update')
    C.require(C.state_sha(latest['state_dict'])==metadata['terminal_state_sha256']!=metadata['initial_state_sha256'],
              'Terminal state has no real parameter change')
    C.require(C.state_sha(best['state_dict'])==metadata['best_state_sha256'],'Selected state hash differs')
    rates=[];validation=[];patience=None;stale=0
    for epoch,record in enumerate(trajectory,1):
        C.require(record['epoch']==epoch and len(record['train_order'])==ntrain and
                  sorted(record['train_order'])==list(range(ntrain)) and sum(record['batch_sizes'])==ntrain and
                  max(record['batch_sizes'])==8,'Epoch did not preserve all train people/batch8')
        expected=.001*epoch/10 if epoch<=10 else .001*(.1+.9*(1+math.cos(math.pi*(epoch-10)/110))/2)
        C.require(abs(record['lr']-expected)<1e-15,'Warmup/cosine learning rate differs')
        C.require(all(math.isfinite(v) for v in (record['loss'],record['max_gradient_norm'],record['val_nll'],*record['components'].values())),
                  'Nonfinite objective/gradient/validation evidence')
        C.require(record['updates']==math.ceil(ntrain/8),'Missing optimizer update')
        validation.append(record['val_nll']);rates.append(record['lr'])
        if patience is None or record['val_nll']<=patience-1e-4:patience,stale=record['val_nll'],0
        else:stale+=1
        C.require(record['stale']==stale,'Early-stop state reconstruction differs')
        if identity['method']=='CCML':
            parts=('single_data','single_kl','consistent_data','consistent_kl','complement_data')
            C.require(set(parts)<=record['components'].keys() and
                      abs(sum(record['components'][p] for p in parts)-record['loss'])<=2e-5,'Complete CCML objective decomposition missing')
        elif identity['method']=='MMDynamics':
            views=2 if identity['eeg']=='ST' else 3
            C.require(all(f'view{v}_{name}' in record['components'] for v in range(views)
                          for name in ('gate_penalty','single_ce','confidence_mse')),'MMDynamics core auxiliary objectives missing')
        elif identity['method']=='QMF':
            C.require({'classification_sum','ranking'}<=record['components'].keys(),'QMF ranking history objective missing')
        elif identity['method']=='ECML':
            C.require({'edl_average','conflict_consistency'}<=record['components'].keys(),'ECML consistency objective missing')
    chosen=min(range(len(validation)),key=lambda i:validation[i])+1
    C.require(chosen==metadata['best_epoch']==latest['best_epoch']==best['epoch'],'Validation-only earliest best selection differs')
    C.require(latest['epoch']==120 or stale>=60,'Unregistered early stop')
    C.require(latest['scheduler']['epoch']==latest['epoch'] and all(k in latest['rng'] for k in ('python','numpy','torch','cuda')),
              'Incomplete recoverable scheduler/RNG state')
    if identity['method']=='QMF':
        history=latest['quality_history']
        views=2 if identity['eeg']=='ST' else 3
        C.require(history is not None and history['cumulative_ce'].shape==(views,ntrain) and
                  np.isfinite(history['cumulative_ce']).all() and np.isfinite(history['confidence']).all(),
                  'QMF history has wrong train-only shape or invalid values')
    for epoch in (60,120):
        C.require((directory/f'epoch_{epoch}.pt').exists()==(latest['epoch']>=epoch),'Required reached snapshot absent')
    st=C.OUT/'features/ST'/f"seed_{identity['seed']}"/f"fold_{identity['fold']}.npz"
    with np.load(st,allow_pickle=False) as z:
        features=z['features'].copy();ids=z['subject_ids'].tolist();labels=z['labels'].tolist();splits=z['splits'].tolist()
    if identity['eeg']!='ST':
        ep=C.OUT/'features'/identity['eeg']/f"seed_{identity['seed']}"/f"fold_{identity['fold']}.npz"
        with np.load(ep,allow_pickle=False) as z:features=np.concatenate([z['features'][:,None],features],axis=1)
    C.require(binding['subject_ids']==ids and binding['labels']==labels and binding['splits']==splits,'Fitted input identities differ')
    train=[ids.index(sid) for sid in binding['roles']['train']]
    expected_mean=features[train].astype(np.float64).mean(0)
    expected_std=np.maximum(features[train].astype(np.float64).std(0),1e-6)
    C.require(np.array_equal(expected_mean,np.asarray(binding['normalization']['mean'])) and
              np.array_equal(expected_std,np.asarray(binding['normalization']['std'])) and
              binding['normalization']['fit_people']==binding['roles']['train'],'Normalizer not bound to exact fold train')
    evaluation=C.read(directory/'evaluation.json')
    C.require(evaluation['checkpoint']==complete['best'] and evaluation['frozen'] and evaluation['independent_replay_bitwise'],
              'Final probabilities not bound to independently reloaded best')
    C.require(len(evaluation['rows'])==37 and len({r['subject_id'] for r in evaluation['rows']})==37,'Evaluation person coverage differs')
    test=[r for r in evaluation['rows'] if r['role']=='test'];recomputed=independent_metrics(test,np)
    for key in KEYS:C.require(abs(recomputed[key]-evaluation['metrics']['test'][key])<=1e-12,'Independent metric mismatch: '+key)
    C.require(recomputed['confusion_matrix']==evaluation['metrics']['test']['confusion_matrix'],'Confusion matrix mismatch')
    return metadata,evaluation,recomputed

def costs(records):
    def total(key):
        values=[m.get(key) for m,_,_ in records]
        return sum(values) if all(isinstance(v,(int,float)) and math.isfinite(v) for v in values) else None
    feature_records=[C.read(p) for p in (C.OUT/'features').glob('E*/seed_*/fold_*.json')]
    recorded=[v['seconds'] for v in feature_records if 'seconds' in v]
    startup=C.read(C.OUT/'startup_acceptance.json')
    events=[C.read(p) for p in (C.OUT/'control').glob('lost_owner_*.json') if 'record_' not in p.name]
    failures=[C.desc(p) for p in (C.OUT/'control').glob('failure_*.json')]
    return {'schema':'fusion_recorded_cost_v1','formal_fits':1200,'optimizer_updates':sum(m['updates'] for m,_,_ in records),
      'formal_train_recorded_seconds':total('train_seconds'),'validation_recorded_seconds':total('validation_seconds'),
      'checkpoint_writes_recorded_seconds':total('durable_write_seconds'),
      'feature_export_recorded_seconds':sum(recorded),'EEG_export_records_with_timing':len(recorded),
      'E00_ST_source_cache_replay_total_seconds':None,'source_registration_audit_total_seconds':None,
      'selected_model_evaluation_recorded_seconds':sum(e['inference_seconds'] for _,e,_ in records),
      'independent_second_test_replay_total_seconds':None,'engineering_recovery_total_seconds':None,
      'engineering_benchmark':C.desc(C.OUT/'engineering/complete.json'),'failed_startup_attempts':startup['failed_engineering_attempts'],
      'recovery_events':events,'failure_evidence':failures,
      'unknown_uncommitted_update_or_exit_tails':None,'all_process_active_training_seconds':None,
      'combined_total_seconds':None,
      'interpretation':'These are separately recorded elapsed categories, not a complete active compute total. Unmeasured admission, recovery, final replay and failure tails remain null. No overlapping wall envelope is added.',
      'all_OS_low_power_windows_reconstructed':False,
      'OS_scope_note':'Final system event evidence and sleep/pause influence must be checked separately; no blanket uninterrupted-runtime claim.'}

def final_report(summary,cost):
    est=[r for r in summary if r['eeg']!='ST'];st=[r for r in summary if r['eeg']=='ST']
    lookup={(r['eeg'],r['method']):r for r in summary}
    lines=['# EEG十五版本与五种融合方法最终全组合报告','',
      '本轮80个条件、1,200次正式拟合已完成，全部采用固定分支和最终融合输出。主指标先平均每种子的五折，再平均三个种子。',
      '','| 条件 | 方法 | Macro-F1 | 对本地基线差值 | BA | MDD召回 | AUC | NLL | Brier |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|',
      '| 本地 TRI-DEP 复现基线 | 原公布权重 | 72.550% | 0.000 pp | 73.000% | 61.667% | 0.782222 | 0.568573 | 0.186360 |']
    for r in est:
        lines.append(f"| {r['eeg']} {C.EEG[r['eeg']]} | {r['method']} | {100*r['macro_f1']:.3f}% | {r['delta_baseline_pp']:+.3f} pp | {100*r['balanced_accuracy']:.3f}% | {100*r['mdd_recall']:.3f}% | {r['auc']:.6f} | {r['nll']:.6f} | {r['brier']:.6f} |")
    lines.extend(['','| EEG | CCML | LMF | MMDynamics | QMF | ECML |','|---|---:|---:|---:|---:|---:|'])
    for eeg,name in C.EEG.items():
        lines.append('| '+eeg+' '+name+' | '+' | '.join(f"{100*lookup[(eeg,m)]['macro_f1']:.3f}% ({lookup[(eeg,m)]['delta_baseline_pp']:+.3f} pp)" for m in C.METHODS)+' |')
    lines.extend(['','| ST方法 | Macro-F1 | 三种子标准差 |','|---|---:|---:|'])
    for r in st:lines.append(f"| {r['method']} | {100*r['macro_f1']:.3f}% | {100*r['seed_macro_f1_sd']:.3f} pp |")
    ranked=sorted(est,key=lambda r:r['macro_f1'],reverse=True);best=ranked[0]
    lines.extend(['',f"最高开发观测值为 {best['eeg']}＋{best['method']}：Macro-F1 {100*best['macro_f1']:.3f}%，相对本地基线 {best['delta_baseline_pp']:+.3f} 个百分点。完整低分与阴性条件均保留。",
      '','完整逐折、逐种子和对E00/ST/基线配对差值、纠错与新增错误见CSV；逐人概率、混淆矩阵与检查点来源见各fit的evaluation.json。',
      '','这是固定分支上的二阶段开发比较。37人及val3已反复用于开发，111种子预测不是111独立人员。跨EEG的预处理/训练来源差异不能全部归因于架构。数值提高不自动表示显著性、泛化保证或临床效果。',
      '','QMF采用锁定作者RGBD路径和真实logits，初始相等历史使用零排名/边距保护。ECML采用锁定作者RCML的顺序证据平均、完整EDL与冲突一致性目标。微调best处于热身的实际情况及外部预训练参与者重叠排除未知均保留在来源元数据中。',
      '','成本各类别与未测部分见real_cost_v1.json。未测时长为null，未把嵌套、暂停或等待与训练相加。系统低功耗/暂停范围需要补充核验；当前记录不证明总active时长或完整不间断运行。'])
    C.atomic(C.REPORT/'最终全组合实验报告.md',('\n'.join(lines)+'\n').encode())

def run():
    ids=read_gate()
    import numpy as np
    import torch
    start=time.perf_counter();sources=verify_sources();records=[]
    for number,identity in enumerate(ids,1):
        C.check_pause();records.append(verify_fit(identity,np,torch))
        if number%75==0:print(f'Independent fit audit {number}/1200',flush=True)
    results=C.read(C.REPORT/'results.json');summary=results['summary']
    C.require(len(summary)==80 and len({(r['eeg'],r['method']) for r in summary})==80,'Final condition table incomplete')
    by={}
    for m,e,v in records:
        i=m['identity'];by[(i['eeg'],i['method'],i['seed'],i['fold'])]=v
    for r in summary:
        seeds=[]
        for seed in C.SEEDS:
            values=[by[(r['eeg'],r['method'],seed,fold)] for fold in range(1,6)]
            sm={k:mean(v[k] for v in values) for k in KEYS};seeds.append(sm)
            C.require(abs(sm['macro_f1']-r[f'seed_{seed}_macro_f1'])<=1e-12,'Seed mean differs')
            C.require(abs(stdev(v['macro_f1'] for v in values)-r[f'fold_sd_seed{seed}'])<=1e-12,'Fold dispersion differs')
        for key in KEYS:C.require(abs(mean(s[key] for s in seeds)-r[key])<=1e-12,'Primary aggregation differs')
        C.require(abs(stdev(s['macro_f1'] for s in seeds)-r['seed_macro_f1_sd'])<=1e-12,'Seed dispersion differs')
        C.require(abs(100*(r['macro_f1']-BASELINE)-r['delta_baseline_pp'])<=1e-12,'Baseline delta arithmetic differs')
    with (C.REPORT/'OOF.csv').open(encoding='utf-8-sig',newline='') as stream:oof=list(csv.DictReader(stream))
    C.require(len(oof)==8880 and len({tuple(r[k] for k in ('eeg','method','seed','fold','subject_id')) for r in oof})==8880,
              'OOF identity coverage differs')
    cost=costs(records);C.save(C.REPORT/'real_cost_v1.json',cost,immutable=True);final_report(summary,cost)
    acceptance={'passed':True,'models':1200,'conditions':80,'sources':sources,
      'training_and_selection_reconstructed':True,'independent_metrics_recomputed':True,'OOF_rows':8880,
      'audit_seconds':time.perf_counter()-start,'source':C.desc(SOURCE),'protocol':C.desc(CONTROL/'protocol.json'),
      'results':C.desc(C.REPORT/'results.json'),'report':C.desc(C.REPORT/'最终全组合实验报告.md'),
      'cost':C.desc(C.REPORT/'real_cost_v1.json'),'at':C.now(),
      'all_plan_complete':False,'remaining':'separate system-event/pause scope and independent final semantic/cost delivery review'}
    C.save(CONTROL/'complete.json',acceptance,immutable=True)
    print(json.dumps({'audit':'core_complete','models':1200,'all_plan_complete':False}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.run:run()
    else:print(json.dumps(prepare(),ensure_ascii=False))

if __name__=='__main__':main()
