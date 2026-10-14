"""All1200 gate, frozen checkpoint replay, paired metrics and final artifacts."""
from __future__ import annotations
import csv
import io
import json
import math
from statistics import mean,pstdev,stdev
import time
import numpy as np
import torch
from . import common as C, training as T
from .models import Fusion
from tridep_ccml.eeg_tridep_fusion.run import metrics

KEYS=('macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier')

def gate():
    done=[]
    for identity in C.identities():
        complete=T.complete_verified(identity)
        C.require(complete is not None,'All1200 trained fits required before outer evaluation')
        done.append(C.desc(C.fitdir(identity)/'complete.json'))
    C.require(len(done)==1200,'Incorrect formal matrix size')
    value={'passed':True,'formal_fits':1200,'sources':done}
    C.save(C.OUT/'training_gate.json',value,immutable=True)
    return value

def evaluate_fit(identity,registration,device):
    directory=C.fitdir(identity);dest=directory/'evaluation.json'
    complete=T.complete_verified(identity)
    if dest.exists():
        old=C.read(dest)
        C.require(old['checkpoint']==complete['best'] and old['registration']==registration,'Cached evaluation binding differs')
        return old
    C.check_pause();x,y,ix,binding=T.load_inputs(identity,device)
    selected=torch.load(C.checked(complete['best']),map_location='cpu',weights_only=False)
    C.require(selected['identity']==identity and selected['registration']==registration and selected['input_binding']==binding,
              'Best model binding differs')
    model=Fusion(identity['method'],x.shape[1],len(ix['train'])).to(device)
    model.load_state_dict(selected['state_dict'],strict=True);T.restore_history(model,selected['quality_history'])
    # Preserve the training-validation Parameter flags: torch.matmul can choose
    # a different folding kernel when requires_grad is toggled, even inside
    # inference_mode. Gradients are disabled for every actual forward below.
    model.eval();before=C.state_sha(model.state_dict())
    probabilities=[];latency=0.
    with torch.inference_mode():
        # Validation parity is independently repeated with the original val batch.
        vi=torch.as_tensor(ix['val'],device=device)
        valprob=model(x[vi])['p'].cpu()
        C.require(torch.equal(valprob,selected['best_prob']) and not valprob.requires_grad,
                  'Selected validation probability replay differs or gradient tracking is enabled')
        for role in ('train','val','test'):
            idx=torch.as_tensor(ix[role],device=device);start=time.perf_counter()
            p=model(x[idx])['p'].double().cpu().numpy()
            latency+=time.perf_counter()-start
            C.require(np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6),'Final probability invalid')
            for j,i in enumerate(ix[role]):
                probabilities.append(dict(**identity,subject_id=binding['subject_ids'][i],role=role,
                  label=binding['labels'][i],p_mdd=float(p[j,1]),prediction=int(p[j,1]>.5)))
    C.require(before==C.state_sha(model.state_dict()) and all(p.grad is None for p in model.parameters()),'Evaluation modified frozen model')
    # A fresh second instance independently checks test outputs and full state loading.
    fresh=Fusion(identity['method'],x.shape[1],len(ix['train'])).to(device)
    fresh.load_state_dict(torch.load(C.checked(complete['best']),map_location='cpu',weights_only=False)['state_dict'],strict=True)
    fresh.eval()
    with torch.inference_mode():
        ti=torch.as_tensor(ix['test'],device=device);repeat=fresh(x[ti])['p'].double().cpu().numpy()[:,1]
    actual=[r['p_mdd'] for r in probabilities if r['role']=='test']
    C.require(np.array_equal(repeat,np.asarray(actual)),'Independent test checkpoint replay not bitwise identical')
    result={'identity':identity,'checkpoint':complete['best'],'registration':registration,'input_binding':binding,
      'rows':probabilities,'metrics':{role:metrics([dict(label=r['label'],p=r['p_mdd']) for r in probabilities if r['role']==role])
                                   for role in ('train','val','test')},
      'frozen':True,'freeze_scope':'state unchanged; inference_mode disables all gradient tracking; original Parameter flags retained',
      'evaluator_source':C.desc(__file__),'independent_replay_bitwise':True,'inference_seconds':latency,'at':C.now()}
    C.save(dest,result,immutable=True)
    return result

def write_csv(path,rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);C.atomic(path,stream.getvalue().encode('utf-8-sig'))

def aggregate(evaluations,baseline):
    perfit=[];allrows=[];summary=[];lookup={}
    for result in evaluations:
        i=result['identity'];lookup[(i['eeg'],i['method'],i['seed'],i['fold'])]=result
        metadata=C.read(C.fitdir(i)/'metadata.json')
        perfit.append(dict(**i,**{k:result['metrics']['test'][k] for k in KEYS},
          best_epoch=metadata['best_epoch'],epochs=metadata['epochs'],updates=metadata['updates'],
          parameters=metadata['parameters'],device=metadata['device'],train_seconds=metadata['train_seconds'],
          validation_seconds=metadata['validation_seconds'],status='complete'))
        allrows.extend(result['rows'])
    baseline_map={(r['seed'],r.get('fold',r.get('outer_fold')),r['subject_id']):r for r in baseline}
    for eeg in ('ST',*C.EEG):
        for method in C.METHODS:
            ss=[]
            for seed in C.SEEDS:
                values=[lookup[(eeg,method,seed,f)]['metrics']['test'] for f in range(1,6)]
                ss.append({'seed':seed,**{k:mean(v[k] for v in values) for k in KEYS},
                           'fold_macro_f1_sd':stdev(v['macro_f1'] for v in values)})
            row={'eeg':eeg,'method':method,**{k:mean(s[k] for s in ss) for k in KEYS},
              'seed_11_macro_f1':ss[0]['macro_f1'],'seed_22_macro_f1':ss[1]['macro_f1'],'seed_33_macro_f1':ss[2]['macro_f1'],
              'seed_macro_f1_sd':stdev(s['macro_f1'] for s in ss),
              'fold_sd_seed11':ss[0]['fold_macro_f1_sd'],'fold_sd_seed22':ss[1]['fold_macro_f1_sd'],
              'fold_sd_seed33':ss[2]['fold_macro_f1_sd'],
              'delta_baseline_pp':100*(mean(s['macro_f1'] for s in ss)-.7255026455026455),'fits':15,'status':'complete'}
            summary.append(row)
    by={(r['eeg'],r['method']):r for r in summary}
    pairs=[]
    for eeg in C.EEG:
        for method in C.METHODS:
            r=by[(eeg,method)];r['delta_E00_pp']=100*(r['macro_f1']-by[('E00',method)]['macro_f1'])
            r['delta_ST_pp']=100*(r['macro_f1']-by[('ST',method)]['macro_f1'])
            for seed in C.SEEDS:
                for fold in range(1,6):
                    a=lookup[(eeg,method,seed,fold)]
                    for ref in ('baseline','E00','ST'):
                        if ref=='baseline':
                            br=[baseline_map[(seed,fold,x['subject_id'])] for x in a['rows'] if x['role']=='test']
                            refrows={x['subject_id']:dict(label=x['label'],p=x['p_eeg']*.05+x['p_speech']*.35+x['p_text']*.6) for x in br}
                        else:
                            refrows={x['subject_id']:dict(label=x['label'],p=x['p_mdd']) for x in lookup[(ref,method,seed,fold)]['rows'] if x['role']=='test'}
                        rm=metrics(list(refrows.values()));corrected=[];newerrors=[]
                        for x in a['rows']:
                            if x['role']!='test':continue
                            old=refrows[x['subject_id']];C.require(x['label']==old['label'],'Paired label differs')
                            was=int(old['p']>.5)==x['label'];isnow=x['prediction']==x['label']
                            if isnow and not was:corrected.append(x['subject_id'])
                            if was and not isnow:newerrors.append(x['subject_id'])
                        pairs.append(dict(eeg=eeg,method=method,seed=seed,fold=fold,reference=ref,
                          **{k+'_delta':a['metrics']['test'][k]-rm[k] for k in KEYS},
                          corrected=json.dumps(corrected),new_errors=json.dumps(newerrors)))
    write_csv(C.REPORT/'EST_summary.csv',[r for r in summary if r['eeg']!='ST'])
    write_csv(C.REPORT/'ST_summary.csv',[r for r in summary if r['eeg']=='ST'])
    write_csv(C.REPORT/'fit_metrics.csv',perfit);write_csv(C.REPORT/'paired_differences.csv',pairs)
    write_csv(C.REPORT/'OOF.csv',[r for r in allrows if r['role']=='test'])
    C.atomic(C.REPORT/'predictions.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows).encode())
    C.save(C.REPORT/'results.json',{'summary':summary,'fit_metrics':perfit,'baseline_macro_f1':.7255026455026455,
                                  'primary_aggregation':'mean5folds then mean3seeds','seed_sd':'sample SD, ddof1'})
    content=['# EEG十五版本与五种融合方法全组合结果','',
      '本轮1,200正式拟合与独立检查点重载已完成。主基线为本地TRI-DEP复现Macro-F1 72.550%。', '',
      '| EEG | CCML | LMF | MMDynamics | QMF | ECML |','|---|---:|---:|---:|---:|---:|']
    for eeg,name in C.EEG.items():
        content.append('| '+eeg+' '+name+' | '+' | '.join(f"{by[(eeg,m)]['macro_f1']*100:.3f}% ({by[(eeg,m)]['delta_baseline_pp']:+.3f} pp)" for m in C.METHODS)+' |')
    content.extend(['','全部75EST、5ST、逐折/逐种子、六项指标、配对差值、纠错及新增错误见CSV/JSON。',
      '','这是固定分支上的二阶段开发比较；37人和val3已反复开发使用，不能作为独立盲测。111种子预测不是111独立人员。',
      '不同EEG分支的预处理与训练来源亦有差异；已知适配与外部预训练参与者重叠排除未知沿用原来源披露。',
      'QMF采用作者RGBD路径，历史相等时目标和边距定义为零；ECML采用固定作者仓库的顺序证据平均及冲突一致性损失。',
      '所有阴性结果保留。数值正差不自动表示显著性或临床效果。',
      '','真实训练、特征推理、工程、验证、写盘成本及恢复未知尾由实验日志和metadata保存；最终独立成本与语义审查仍须单独验收。'])
    C.atomic(C.REPORT/'全组合结果报告.md',('\n'.join(content)+'\n').encode())
    return summary

def run(registration,engineering):
    gate();values=[]
    for n,i in enumerate(C.identities(),1):
        values.append(evaluate_fit(i,registration,engineering['methods'][i['method']]['device']))
        if n%15==0:print(f'Evaluation {n}/1200',flush=True)
    baseline=C.rows(C.STAGE0/'branch_probabilities.jsonl')
    summary=aggregate(values,baseline)
    C.save(C.OUT/'evaluation_complete.json',{'passed':True,'models':1200,'conditions':80,
       'independent_replay_models':1200,'OOF_rows':8880,'results':C.desc(C.REPORT/'results.json'),
       'at':C.now(),'all_plan_complete':False,
       'remaining':'independent semantic/cost/source final audit and final completion certificate'})
    return summary
