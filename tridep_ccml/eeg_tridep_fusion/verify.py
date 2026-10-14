"""Independent CPU verifier: NumPy grid, sklearn metrics, raw source row joins.

Does not import the producer's metric, aggregation, fusion or gate functions.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, roc_auc_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tridep_ccml/experiments/EEG-TriDep-Fusion-v1-37-20261013'
KEYS=('macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier')
SEEDS=(11,22,33)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s]
def check(ok,msg):
    if not ok:raise ValueError(msg)
def hashfile(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def near(a,b):
    if a is None or b is None:check(a is b,'Missing metric mismatch')
    else:check(abs(float(a)-float(b))<1e-12,f'Numerical mismatch {a} != {b}')
def metric(rr):
    y=np.array([r['label'] for r in rr]);p=np.array([r['p'] for r in rr]);yp=(p>.5).astype(int)
    return dict(macro_f1=float(f1_score(y,yp,labels=[0,1],average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,yp)),mdd_recall=float(recall_score(y,yp,zero_division=0)),
        auc=float(roc_auc_score(y,p)),nll=float(-np.log(np.maximum(np.where(y==1,p,1-p),1e-15)).mean()),
        brier=float(np.mean((y-p)**2)),confusion_matrix=confusion_matrix(y,yp,labels=[0,1]).tolist(),subjects=len(y))
def grid(rr,st=False):
    check(all(r['role']=='val' for r in rr),'Non-val grid input')
    ww=np.array([[i/20,j/20,(20-i-j)/20] for i in ([0] if st else range(21)) for j in range(21-i)])
    probabilities=(ww[:,0,None]*np.array([r['p_eeg'] for r in rr])+ww[:,1,None]*np.array([r['p_speech'] for r in rr]))+ww[:,2,None]*np.array([r['p_text'] for r in rr])
    y=np.array([r['label'] for r in rr])[None,:];yp=probabilities>.5
    tp=np.sum(yp & (y==1),axis=1);tn=np.sum((~yp)&(y==0),axis=1)
    fp=np.sum(yp & (y==0),axis=1);fn=np.sum((~yp)&(y==1),axis=1)
    f0=np.divide(2*tn,2*tn+fp+fn,out=np.zeros(len(ww)),where=(2*tn+fp+fn)!=0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(ww)),where=(2*tp+fp+fn)!=0)
    score=(f0+f1)/2;idx=np.where(score==score.max())[0]
    return ww[idx].tolist(),float(score.max())
def aggregate_verify(rr,expected):
    perseed=[]
    role_seed={k:[] for k in ('train','val','test')}
    for s in SEEDS:
        foldrole={k:[] for k in role_seed}
        for f in range(1,6):
            for role in role_seed:
                m=metric([r for r in rr if r['seed']==s and r['fold']==f and r['role']==role])
                original=expected['seeds'][str(s)]['folds'][str(f)][role]
                for k in KEYS:near(m[k],original[k])
                check(m['confusion_matrix']==original['confusion_matrix'],'Confusion matrix mismatch')
                check(m['subjects']==original['subjects'],'Metric group size mismatch')
                foldrole[role].append(m)
        om=metric([r for r in rr if r['seed']==s and r['role']=='test'])
        for k in KEYS:near(om[k],expected['seeds'][str(s)]['oof'][k])
        check(om['subjects']==37,'OOF size')
        avg={k:float(np.mean([m[k] for m in foldrole['test']])) for k in KEYS}
        for k in KEYS:near(avg[k],expected['seeds'][str(s)]['fold_mean'][k])
        perseed.append(avg)
        for role in role_seed:role_seed[role].append({k:float(np.mean([m[k] for m in foldrole[role]])) for k in KEYS})
    for k in KEYS:near(np.mean([m[k] for m in perseed]),expected['primary'][k])
    for role in role_seed:
        for k in KEYS:near(np.mean([m[k] for m in role_seed[role]]),expected['roles'][role][k])

def gate_check(a,b,g):
    d={k:a['primary'][k]-b['primary'][k] for k in KEYS}
    n=sum(a['seeds'][str(s)]['fold_mean']['macro_f1']>b['seeds'][str(s)]['fold_mean']['macro_f1'] for s in SEEDS)
    for k in KEYS:near(d[k],g['deltas'][k])
    ok=d['macro_f1']>=.01 and n>=2 and d['balanced_accuracy']>=0 and d['mdd_recall']>=-.05
    check(g['passed']==ok and g['strictly_improved_seeds']==n,'Gate mismatch')
    check(g['probability_benefit']==(d['nll']<0 and d['brier']<0),'Probability gate mismatch')

def main():
    start=time.perf_counter();active=read(OUT/'active_registration.json');check(hashfile(ROOT/active['path'])==active['sha256'],'Active registration changed');reg=read(ROOT/active['path']);a=ROOT/read(OUT/'execution_pointer.json')['attempt']
    check(read(a/'execution_complete.json')['passed'],'Execution unfinished')
    for d in reg['code']:check(hashfile(ROOT/d['path'])==d['sha256'],'Registered code changed')
    sources=read(a/'sources_before.json')
    for d in sources:check(hashfile(ROOT/d['path'])==d['sha256'],'Source changed '+d['path'])
    original={(r['seed'],r['outer_fold'],r['subject_id']):r for r in rows(ROOT/reg['stage0_branch']['path'])}
    olds={(r['condition'],r['seed'],r['outer_fold'],r['subject_id']):r for r in rows(ROOT/reg['stage0_controls']['path'])}
    data=rows(a/'inputs.jsonl');check(len(data)==13875,'Input count')
    cache={}
    inputmap={}
    groups=defaultdict(list)
    for r in data:
        k=(r['seed'],r['fold'],r['subject_id']);b=original[k]
        for field in ('label','p_speech','p_text'):check(r[field]==b[field],'ST/label changed')
        check(r['role']==b['split'] and r['old_p_eeg']==b['p_eeg'],'Old EEG or split mismatch')
        for field,name in [('old_st','ST-val-weighted'),('old_est','EST-original-val-weighted')]:check(r[field]==olds[(name,*k)]['p_final'],'Historical final changed')
        check(r['old_weights']==olds[('EST-original-val-weighted',*k)]['fusion_weights'],'Old weight changed')
        if r['batch_source'] not in cache:cache[r['batch_source']]={x['subject_id']:x for x in read(ROOT/r['batch_source'])['rows']}
        e=cache[r['batch_source']][r['subject_id']]
        check(e['p_mdd']==r['p_eeg'] and e['label']==r['label'] and e['role']==r['role'],'Imported EEG probability changed')
        ik=(r['unit'],*k);check(ik not in inputmap,'Duplicate input');inputmap[ik]=r
        groups[(r['unit'],r['seed'],r['fold'])].append(r)
    for rr in groups.values():
        check(len(rr)==37 and len({r['subject_id'] for r in rr})==37,'Unit coverage')
        check(sorted(Counter(r['role'] for r in rr).values()) in ([3,7,27],[3,8,26]),'Unit roles')
    sels=read(a/'selections.json');check(len(sels)==375,'Selection count');selected={}
    for s in sels:
        k=(s['unit'],s['seed'],s['fold']);rr=[r for r in groups[k] if r['role']=='val'];w,score=grid(rr)
        check(s['weights']==w[0] and s['tied_best_weights']==w and s['ties']==len(w),'Selection/tie mismatch')
        near(score,s['validation_macro_f1']);check(s['nonzero_eeg_tied_best']==any(x[0]>0 for x in w),'Nonzero tie flag')
        selected[k]=w[0]
    predictions=rows(a/'predictions.jsonl');check(len(predictions)==55500,'Prediction count')
    pg=defaultdict(list)
    for p in predictions:
        r=inputmap[(p['unit'],p['seed'],p['fold'],p['subject_id'])]
        if p['view']=='A':value=.05*r['p_eeg']+.35*r['p_speech']+.60*r['p_text'];w=[.05,.35,.60]
        elif p['view']=='B':w=selected[p['unit'],p['seed'],p['fold']];value=w[0]*r['p_eeg']+w[1]*r['p_speech']+w[2]*r['p_text']
        elif p['view']=='C':w=[1/3]*3;value=(r['p_eeg']+r['p_speech']+r['p_text'])/3
        else:w=r['old_weights'];value=w[0]*r['p_eeg']+w[1]*r['p_speech']+w[2]*r['p_text']
        check(p['p']==value and p['prediction']==int(value>.5) and p['weights']==w,'Fusion formula mismatch')
        near(p['threshold_distance'],value-.5)
        check(p['label']==r['label'] and p['role']==r['role'],'Prediction role changed')
        pg[p['unit'],p['view']].append(p)
    baseline=rows(a/'baseline_predictions.jsonl');bg=defaultdict(list);bmap={}
    for p in baseline:
        k=(p['seed'],p['fold'],p['subject_id']);r=original[k];E,S,T=r['p_eeg'],r['p_speech'],r['p_text']
        st=olds[('ST-val-weighted',*k)]['p_final'];est=olds[('EST-original-val-weighted',*k)]['p_final']
        values={'historical_ST':st,'historical_EST':est,'ST_A':(.35*S+.60*T)/.95,'ST_B':st,'ST_C':(S+T)/2,'ST_D':st,
                'ST_paper_reference':.45*S+.55*T,'original_E_A':.05*E+.35*S+.60*T,
                'original_E_B':est,'original_E_C':(E+S+T)/3,'original_E_D':est}
        check(values[p['unit']]==p['p'] and p['prediction']==int(p['p']>.5),'Baseline formula mismatch')
        bg[p['unit']].append(p);bmap[(p['unit'],*k)]=p
    # Recompute the baseline grids separately, including exact historical ST match.
    for rec in read(a/'baseline_selections.json'):
        rr=[dict(r,role=r['split'],fold=r['outer_fold']) for r in original.values() if r['seed']==rec['seed'] and r['outer_fold']==rec['fold'] and r['split']=='val']
        st=rec['kind']=='ST_B';w,score=grid(rr,st)
        check(rec['weights']==w[0] and rec['tied_best_weights']==w,'Baseline grid mismatch')
        oldname='ST-val-weighted' if st else 'EST-original-val-weighted'
        check(w[0]==olds[(oldname,rec['seed'],rec['fold'],rr[0]['subject_id'])]['fusion_weights'],'Baseline historical weight mismatch')
    result=read(a/'results.json')
    for (u,v),rr in pg.items():aggregate_verify(rr,result['conditions'][u][v])
    for u,rr in bg.items():aggregate_verify(rr,result['baselines'][u])
    check(len(pg)==100 and len(bg)==11,'Result matrix incomplete')
    for u in reg['matrix']:
        for v in ('A','B','C','D'):
            entry=result['comparisons'][u['id']][v];c=entry['comparisons'];aa=result['conditions'][u['id']][v]
            for name,bn in [('original_E','original_E_'+v),('ST','ST_'+v),('historical_ST','historical_ST'),('historical_EST','historical_EST')]:gate_check(aa,result['baselines'][bn],c[name])
            if u['matched_control']:gate_check(aa,result['conditions'][u['matched_control']][v],c['matched_LaBraM'])
            keep=bool(u['candidate'] and all(c[k]['passed'] for k in ('matched_LaBraM','original_E','ST')) and c['historical_ST']['deltas']['macro_f1']>0 and c['historical_EST']['deltas']['macro_f1']>0)
            check(entry['protocol_retention']==keep and entry['primary_retention']==(keep if v=='A' else None),'Retention mismatch')
    for model,views in result['duration'].items():
        for v,g in views.items():gate_check(result['conditions'][model+'@300'][v],result['conditions'][model+'@60'][v],g)
    changes=rows(a/'paired_changes.jsonl');check(len(changes)==11100,'Change row count')
    pmap={(p['unit'],p['view'],p['seed'],p['fold'],p['subject_id']):p for p in predictions}
    cg=defaultdict(list)
    for c in changes:
        p=pmap[c['unit'],c['view'],c['seed'],c['fold'],c['subject_id']];b=bmap[('ST_'+c['view'],c['seed'],c['fold'],c['subject_id'])]
        oldok=b['prediction']==p['label'];newok=p['prediction']==p['label']
        status='corrected' if newok and not oldok else 'new_error' if oldok and not newok else 'probability_only' if p['p']!=b['p'] else 'identical'
        check(c['status']==status and c['p_before']==b['p'] and c['p_after']==p['p'],'Pairwise change mismatch')
        cg[c['unit'],c['view'],status].append(c)
    for u in reg['matrix']:
        for v in ('A','B','C','D'):
            for status in ('corrected','new_error','probability_only','identical'):
                rr=cg[u['id'],v,status];expected=result['changes'][u['id']][v][status]
                check(expected==dict(records=len(rr),distinct_people=len({r['subject_id'] for r in rr}),mdd_records=sum(r['label']==1 for r in rr),hc_records=sum(r['label']==0 for r in rr)),'Change aggregate mismatch')
    flat=list(csv.DictReader((a/'summary.csv').open(encoding='utf-8-sig',newline='')))
    check(len(flat)==100,'Summary row count')
    for r in flat:
        for k in KEYS:near(float(r[k]),result['conditions'][r['id']][r['view']]['primary'][k])
        check((r['retained']=='True')==result['comparisons'][r['id']][r['view']]['protocol_retention'],'CSV retention mismatch')
    certificate=dict(passed=True,verification_seconds=time.perf_counter()-start,source_files_rehashed=len(sources),
        source_rows_rejoined=len(data),fusion_rows_exact=len(predictions),baseline_rows_exact=len(baseline),
        candidate_grids_recomputed=375,baseline_grids_recomputed=30,aggregated_results_recomputed=111,
        changes_recomputed=len(changes),all_gates_recomputed=True,verification='separate implementation; NumPy selection + sklearn metrics',
        sources_manifest_sha256=hashfile(a/'sources_before.json'),results_sha256=hashfile(a/'results.json'),
        predictions_sha256=hashfile(a/'predictions.jsonl'),selection_sha256=hashfile(a/'selections.json'),
        new_model_fits=0,neural_forwards=0)
    save(a/'independent_verification.json',certificate)
    print(json.dumps(certificate,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
