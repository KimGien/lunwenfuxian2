"""Post-fit reporting; cannot train models or select a method."""
from __future__ import annotations
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
from .common import *
from .methods import apply_sigmoid

METRICS=['macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier']

def metric(rows):
    y=np.array([r['label'] for r in rows]); p=np.array([r['p'] for r in rows],dtype=float)
    pred=np.array([int(r['p']>r['threshold']) for r in rows])
    matrix=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=matrix.ravel()
    f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0))
    manual=((2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0)+(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))/2
    require(abs(f1-manual)<1e-12,'Independent F1 arithmetic mismatch')
    return {'macro_f1':f1,'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'mdd_recall':float(tp/(tp+fn)),'auc':float(roc_auc_score(y,p)),
        'nll':float(-np.log(np.clip(np.where(y,p,1-p),1e-15,1)).mean()),
        'brier':float(np.mean((p-y)**2)),'confusion_HC_MDD':matrix.tolist(),
        'n_people':len(rows),'predicted_MDD':int(pred.sum())}

def summarize():
    lock_check()
    gate=read_json(OUT/'release_gate.json')
    require(gate['passed'],'No release gate')
    for name,sha in gate['fitted_artifacts'].items(): require(digest(OUT/name)==sha,'Post-release fit changed: '+name)
    REPORT.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for path in sorted((OUT/'predictions').rglob('fold_*.json')):
        obj=read_json(path)
        for row in obj['rows']:
            for endpoint,p in row['probabilities'].items():
                base={'condition':obj['condition'],'arm':obj['arm'],'seed':obj['seed'],'fold':obj['fold'],
                    'subject_id':row['subject_id'],'label':row['label'],'split':row['split'],
                    'endpoint':endpoint,'p':p,'threshold':.5,'postprocess':'raw'}
                allrows.append(base)
        if obj['condition'] in ['C3','C4']:
            cal=read_json(OUT/f'calibrators/{obj["condition"]}/seed_{obj["seed"]}/fold_{obj["fold"]}.json')
            for row in obj['rows']:
                for endpoint in ['eeg','fusion']:
                    p=row['probabilities'][endpoint];params=cal['parameters'][endpoint]
                    base={'condition':obj['condition'],'arm':'B','seed':obj['seed'],'fold':obj['fold'],
                        'subject_id':row['subject_id'],'label':row['label'],'split':row['split'],'endpoint':endpoint}
                    allrows.append({**base,'p':float(apply_sigmoid(p,params['sigmoid'])),'threshold':.5,'postprocess':'sigmoid'})
                    allrows.append({**base,'p':p,'threshold':params['threshold']['threshold'],'postprocess':'threshold'})
    keys=['condition','arm','endpoint','postprocess','split']
    grouped=defaultdict(list)
    for row in allrows: grouped[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,rows in sorted(grouped.items()):
        fold_metrics=[];seed_metrics=[];oof_metrics=[]
        for seed in SEEDS:
            subset=[r for r in rows if r['seed']==seed]
            for fold in FOLDS:
                fm=metric([r for r in subset if r['fold']==fold])
                fold_metrics.append({'seed':seed,'fold':fold,**fm})
            sm={k:float(np.mean([r[k] for r in fold_metrics if r['seed']==seed])) for k in METRICS}
            seed_metrics.append({'seed':seed,**sm})
            if key[-1]=='test':
                require(len(subset)==37 and len({r['subject_id'] for r in subset})==37,'Outer OOF coverage failed')
                oof_metrics.append({'seed':seed,**metric(subset)})
        mean={k:float(np.mean([s[k] for s in seed_metrics])) for k in METRICS}
        result.append({**dict(zip(keys,key)),'mean':mean,'per_seed':seed_metrics,'per_fold':fold_metrics,
                       'pooled_OOF_per_seed':oof_metrics,'aggregation':'mean of five fold metrics, then three seeds'})
    lookup={(r['condition'],r['arm'],r['endpoint'],r['postprocess'],r['split']):r for r in result}
    paired=[]
    for r in result:
        if r['split']!='test' or r['endpoint'] not in ['eeg','fusion']:continue
        if r['arm']=='B' and r['postprocess']=='raw':continue
        base=lookup[(r['condition'],'B',r['endpoint'],'raw','test')]
        deltas={k:r['mean'][k]-base['mean'][k] for k in METRICS}
        seed_deltas=[{'seed':s['seed'],**{k:s[k]-base['per_seed'][i][k] for k in METRICS}}
                     for i,s in enumerate(r['per_seed'])]
        supported=deltas['macro_f1']>=.02 and sum(s['macro_f1']>0 for s in seed_deltas)>=2 and deltas['balanced_accuracy']>=0
        paired.append({k:r[k] for k in keys[:-1]}|{'delta_mean':deltas,'delta_per_seed':seed_deltas,
            'development_F1_signal_gate':supported,'independent_validation':False})
    write_new(REPORT/'predictions_long.json',allrows)
    write_new(REPORT/'metrics.json',result)
    write_new(REPORT/'paired_comparisons.json',paired)
    for c in ['C3','C4']:
        for endpoint in ['eeg','fusion']:
            raw=lookup[(c,'B',endpoint,'raw','test')]
            sig=lookup[(c,'B',endpoint,'sigmoid','test')]
            require(all(abs(a['auc']-b['auc'])<1e-12 for a,b in zip(raw['per_fold'],sig['per_fold'])),
                    'Positive sigmoid unexpectedly changed within-fold ranking')
    plots(lookup)
    tables(lookup,paired)
    attention=[read_json(p) for p in sorted((OUT/'attention').glob('*.json'))]
    stats={}
    for name in ['query_change_norm','query_gradient_norm','uniform_ablation_representation_relative_l2',
                 'uniform_ablation_eeg_probability_mean_abs_change','uniform_ablation_eeg_prediction_flips']:
        a=np.array([r[name] for r in attention],dtype=float)
        stats[name]={'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),'sum':float(a.sum())}
    for name in ['normalized_entropy_mean','maximum_weight_mean','effective_positions_mean']:
        a=np.array([np.mean([s[name] for s in r['subjects']]) for r in attention])
        stats[name]={'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max())}
    write_new(REPORT/'attention_summary.json',stats)
    calstats=[]
    for c in ['C3','C4']:
        for endpoint in ['eeg','fusion']:
            docs=[read_json(OUT/f'calibrators/{c}/seed_{s}/fold_{f}.json') for s in SEEDS for f in FOLDS]
            cps=[d['parameters'][endpoint] for d in docs]
            calstats.append({'condition':c,'endpoint':endpoint,
                'thresholds':[p['threshold']['threshold'] for p in cps],
                'sigmoid_a':[p['sigmoid']['a'] for p in cps],'sigmoid_b':[p['sigmoid']['b'] for p in cps],
                'inner_OOF_AUC':[float(roc_auc_score([r['label'] for r in d['oof'][endpoint]],
                                  [r['p'] for r in d['oof'][endpoint]])) for d in docs]})
    write_new(REPORT/'calibrator_summary.json',calstats)
    bnrows=[]
    for path in sorted((OUT/'runs').rglob('bn_fit.json')):
        value=read_json(path)
        for layer in value['layers']:
            old=np.array(layer['old_variance']);new=np.array(layer['population_variance'])
            shift=np.abs(np.array(layer['old_mean'])-layer['mean'])/np.sqrt(new+1e-3)
            bnrows.append({'run':path.parent.relative_to(OUT).as_posix(),'layer':layer['layer'],
                'old_to_recomputed_variance_median':float(np.median(old/np.maximum(new,1e-12))),
                'normalized_mean_shift_max':float(shift.max())})
    write_new(REPORT/'bn_statistics_summary.json',bnrows)
    print(json.dumps({'paired':paired,'attention':stats},ensure_ascii=False,indent=2))

def tables(lookup,paired):
    lines=['# P3 全量指标表','', '先平均五折，再平均三个种子；本表仅描述复用开发队列。','',
           '| 路线 | 方案 | 端点 | train F1 | test F1 | ΔF1 | BA | MDD recall | AUC | NLL | Brier |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in ['U1','C1','C3','C4']:
        for arm,post in ([('B','raw'),('BR','raw'),('G','raw'),('S','raw'),('SR','raw')] if c in ['U1','C1']
                         else [('B','raw'),('B','sigmoid'),('B','threshold')]):
            for endpoint in (['eeg'] if c=='U1' else ['eeg','fusion']):
                r=lookup[(c,arm,endpoint,post,'test')]['mean']; t=lookup[(c,arm,endpoint,post,'train')]['mean']
                base=lookup[(c,'B',endpoint,'raw','test')]['mean']['macro_f1']
                lines.append(f'| {c} | {arm if post=="raw" else post} | {endpoint} | {t["macro_f1"]:.6f} | {r["macro_f1"]:.6f} | {r["macro_f1"]-base:+.6f} | '+
                    ' | '.join(f'{r[k]:.6f}' for k in METRICS[1:])+' |')
    lines+=['','## 逐种子配对差','','| 路线/方案/端点 | seed 11 ΔF1 | seed 22 ΔF1 | seed 33 ΔF1 | 固定开发改善门槛 |','|---|---:|---:|---:|---|']
    for p in paired:
        lines.append(f'| {p["condition"]}/{p["arm"]}/{p["postprocess"]}/{p["endpoint"]} | '+
            ' | '.join(f'{s["macro_f1"]:+.6f}' for s in p['delta_per_seed'])+f' | {"通过" if p["development_F1_signal_gate"] else "未通过"} |')
    (REPORT/'P3全量指标与逐种子差值.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def plots(lookup):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,c in zip(axes,['U1','C1']):
        arms=['B','BR','G','S','SR'];x=np.arange(5)
        train=[lookup[(c,a,'eeg','raw','train')]['mean']['macro_f1'] for a in arms]
        test=[lookup[(c,a,'eeg','raw','test')]['mean']['macro_f1'] for a in arms]
        ax.bar(x-.18,train,.36,label='Train',color='#9eabb5')
        ax.bar(x+.18,test,.36,label='Held-out development folds',color='#227991')
        for i,a in enumerate(arms):
            ax.scatter([i+.18]*3,[s['macro_f1'] for s in lookup[(c,a,'eeg','raw','test')]['per_seed']],s=20,color='#f3993f',zorder=4)
        ax.set_xticks(x,arms);ax.set_ylim(0,1.05);ax.set_ylabel('Macro-F1');ax.set_title(c+' / EEG');ax.grid(axis='y',alpha=.15)
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('B: original | BR: refit BN | G: GroupNorm | S: subject scaling | SR: S + refit BN',fontsize=10)
    fig.savefig(REPORT/'EEGNet_generalization.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for col,endpoint in enumerate(['eeg','fusion']):
        for row,m in enumerate(['macro_f1','nll']):
            ax=axes[row,col]
            for j,c in enumerate(['C3','C4']):
                values=[lookup[(c,'B',endpoint,p,'test')]['mean'][m] for p in ['raw','sigmoid','threshold']]
                ax.plot(np.arange(3),values,'o-',label=c,color=['#227991','#e48b32'][j])
            ax.set_xticks(np.arange(3),['Raw / 0.5','Sigmoid / 0.5','Train-OOF threshold'])
            ax.set_title(endpoint.upper()+' / '+m);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Calibration fitted exclusively on inner held-out training subjects',fontsize=12)
    fig.savefig(REPORT/'Token_calibration.png',dpi=170);plt.close(fig)

if __name__=='__main__': summarize()
