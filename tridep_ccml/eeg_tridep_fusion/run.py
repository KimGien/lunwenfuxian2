"""CPU-only, source-bound execution. No model libraries or training imports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tridep_ccml/experiments/EEG-TriDep-Fusion-v1-37-20261013'
REPORT = ROOT / 'output/analysis/eeg_tridep_fusion_v1_20261013'
OLD = ROOT / 'output/analysis/tridep_stage0_20261004'
EEG = ROOT / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
TABLE = ROOT / 'output/analysis/eeg_all_candidates_v1_20261008/unified_comparison.csv'
PLAN = ROOT / 'output/plans/TriDep_九篇EEG接入融合执行计划_v1_20261013.md'
SEEDS = (11, 22, 33)
ROLES = ('train', 'val', 'test')
KEYS = ('macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier')
VIEWS = ('A', 'B', 'C', 'D')
METHODS = ('EEGPT', 'CSBrain', 'REVE', 'LUNA', 'CodeBrain', 'BIOT', 'EEGMamba')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]

def now():
    return datetime.now(timezone.utc).isoformat()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        require(path.read_text(encoding='utf-8') == data, f'Refusing to replace existing artifact: {path}')
        return
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(data)

def jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def desc(path):
    return {'path': Path(path).relative_to(ROOT).as_posix(), 'sha256': sha(path)}

def file_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f'Source outside workspace: {relative}')
    return path

def metrics(rows, field='p'):
    require(rows, 'Empty metrics group')
    cm = [[0, 0], [0, 0]]
    for r in rows:
        require(r['label'] in (0, 1) and math.isfinite(r[field]) and 0 <= r[field] <= 1, 'Invalid probability/label')
        cm[r['label']][int(r[field] > .5)] += 1
    tn, fp = cm[0]
    fn, tp = cm[1]
    a = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    b = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    pos = [r[field] for r in rows if r['label'] == 1]
    neg = [r[field] for r in rows if r['label'] == 0]
    rec = tp/(tp+fn) if tp+fn else None
    spec = tn/(tn+fp) if tn+fp else None
    return dict(macro_f1=(a+b)/2, balanced_accuracy=(rec+spec)/2 if rec is not None and spec is not None else None,
                mdd_recall=rec, auc=sum((p > n)+.5*(p == n) for p in pos for n in neg)/len(pos)/len(neg) if pos and neg else None,
                nll=mean(-math.log(max(min(r[field] if r['label'] else 1-r[field], 1.), 1e-15)) for r in rows),
                brier=mean((r[field]-r['label'])**2 for r in rows), confusion_matrix=cm, subjects=len(rows))

def average(values):
    return {k: mean(v[k] for v in values) if all(v[k] is not None for v in values) else None for k in KEYS}

def grid_select(rows, st_only=False):
    require(rows and all(r['role'] == 'val' for r in rows), 'Selection accepts validation rows only')
    require(len({(r['seed'], r['fold']) for r in rows}) == 1, 'Do not pool folds/seeds')
    require(len({r['subject_id'] for r in rows}) == len(rows), 'Duplicate selection subjects')
    best = -1.
    winners = []
    for i in ([0] if st_only else range(21)):
        for j in range(21-i):
            w = [i/20, j/20, (20-i-j)/20]
            rr = [dict(label=r['label'], p=sum(a*r[k] for a,k in zip(w, ('p_eeg','p_speech','p_text')))) for r in rows]
            score = metrics(rr)['macro_f1']
            if score > best:
                best, winners = score, [w]
            elif score == best:
                winners.append(w)
    return dict(weights=winners[0], validation_macro_f1=best, tied_best_weights=winners,
                ties=len(winners), nonzero_eeg_tied_best=any(w[0]>0 for w in winners), candidates=21 if st_only else 231)

def matrix():
    rows = list(csv.DictReader(TABLE.open(encoding='utf-8-sig', newline='')))
    require(len(rows) == 23, 'Expected 23 existing conditions')
    byname = {r['condition']: r for r in rows}
    main = [m+'-frozen-smallhead' for m in METHODS] + ['EEGConformer-scratch','EEG2Rep-supervised',
            'LaBraM-native4s-frozen-smallhead','LaBraM-native5s-frozen-smallhead','CBraMod-frozen-smallhead']
    ft = [m+'-lastblock-finetune' for m in METHODS]+['LaBraM-native4s-lastblock-finetune','LaBraM-native5s-lastblock-finetune']
    diag = ['EEGPT-mean-projection4s-originalCNN-300ep','LaBraM-native4s-originalCNN-300ep']
    require(set(main+ft+diag) == set(byname), 'Condition inventory differs')
    result = []
    for group, names in [('main',main), ('finetune',ft), ('duration',diag)]:
        for name in names:
            r = byname[name]
            for snap in (['60','300'] if group == 'duration' else ['best']):
                result.append(dict(id=name+'@'+snap, condition=name, snapshot=snap, group=group, method=r['method'],
                    matched_control=r['matched_control']+'@'+snap if r['matched_control'] else None,
                    candidate=bool(r['matched_control']), input_seconds=int(r['seconds_per_window']), route=r['route']))
    require(len(result) == 25, 'Expected 25 condition snapshots')
    return result

def registration():
    return dict(schema='eeg_tridep_fusion_v1', plan=desc(PLAN), matrix=matrix(), seeds=list(SEEDS), folds=list(range(1,6)),
        views={'A':[.05,.35,.60], 'B':'val Macro-F1 / original 231-grid / first strict maximum',
               'C':'(pE+pS+pT)/3', 'D':'stage0 saved per-fold EST weights'},
        threshold='p>0.5; ties HC', probability_floor=1e-15, aggregation='mean5folds then mean3seeds',
        gate=dict(macro_f1_min_delta=.01, improved_seeds_min=2, ba_min_delta=0., recall_min_delta=-.05),
        stage0_branch=desc(OLD/'branch_probabilities.jsonl'), stage0_controls=desc(OLD/'control_predictions.jsonl'),
        old_metrics=desc(OLD/'metrics.json'), existing_evaluation_complete=desc(EEG/'evaluation/complete.json'),
        condition_table=desc(TABLE), code=[desc(p) for p in sorted(Path(__file__).parent.glob('*.py'))],
        units=375, view_units=1500, selections=375, candidate_scores=86625,
        new_model_fits=0, optimizer_updates=0, asr=0, neural_forwards_planned=0,
        scope='37 previously viewed subjects; development comparison, not independent blind test')

def key(r):
    return (r['seed'], r.get('fold', r.get('outer_fold')), r['subject_id'])

def build_inputs(attempt, reg):
    verified = {}
    def check(d):
        p = file_path(d['path'])
        if d['path'] in verified:
            require(verified[d['path']]['sha256'] == d['sha256'], 'Conflicting source digests '+d['path'])
            return p
        require(p.is_file(), 'Missing source '+str(p))
        st = p.stat()
        actual = sha(p)
        require(actual == d['sha256'], 'Source digest changed '+str(p))
        require((st.st_size,st.st_mtime_ns) == (p.stat().st_size,p.stat().st_mtime_ns), 'Source changed while hashing')
        verified[d['path']] = dict(d, size=st.st_size, mtime_ns=st.st_mtime_ns)
        return p
    for k in ('plan','stage0_branch','stage0_controls','old_metrics','existing_evaluation_complete','condition_table'):
        check(reg[k])
    for d in reg['code']: check(d)
    global_complete = read(check(reg['existing_evaluation_complete']))
    require(global_complete['passed'] and global_complete['formal_fits_verified']==360, 'EEG evaluation not completed')
    evaluation = read(check(global_complete['evaluation_report']))
    require(evaluation['nll_probability_floor']==1e-15, 'Metric floor mismatch')
    completion_sources = {x['path']:x for x in evaluation['probability_batch_artifacts']}
    base = lines(check(reg['stage0_branch']))
    require(len(base)==555 and len({key(r) for r in base})==555, 'Expected 555 unique historical branch rows')
    labels = {}
    for r in base:
        labels.setdefault(r['subject_id'],r['label'])
        require(labels[r['subject_id']]==r['label'], 'Label differs across folds')
        for d in r['parent_checkpoints'].values(): check(d)
    require(len(labels)==37 and sum(labels.values())==16, 'Cohort differs')
    controls = lines(check(reg['stage0_controls']))
    old_controls = {(r['condition'],*key(r)):r for r in controls}
    require(len(old_controls)==len(controls), 'Duplicate old controls')
    byfit = {(s,f):[r for r in base if key(r)[:2]==(s,f)] for s in SEEDS for f in range(1,6)}
    oldweights = {}
    for (s,f), rr in byfit.items():
        require(len(rr)==37 and set(r['subject_id'] for r in rr)==set(labels), 'Old fit cohort mismatch')
        require(sorted(sum(r['split']==role for r in rr) for role in ROLES) in ([3,7,27],[3,8,26]), 'Old role counts')
        for name in ('ST-val-weighted','EST-original-val-weighted'):
            ws = []
            for r in rr:
                o = old_controls[(name,*key(r))]
                require(o['label']==r['label'] and o['split']==r['split'], 'Historical identity mismatch')
                p = sum(a*r[k] for a,k in zip(o['fusion_weights'],('p_eeg','p_speech','p_text')))
                require(p==o['p_final'] and int(p>.5)==o['prediction'], 'Historical probability does not replay')
                ws.append(o['fusion_weights'])
            require(all(w==ws[0] for w in ws), 'Weights changed within fit')
            if name.startswith('EST'): oldweights[(s,f)] = ws[0]
        val = [dict(r,fold=f,role=r['split']) for r in rr if r['split']=='val']
        sel = grid_select(val,st_only=True)
        require(sel['weights']==old_controls[('ST-val-weighted',s,f,rr[0]['subject_id'])]['fusion_weights'], 'ST grid mismatch')
    imported = []
    unit_records = []
    for unit in reg['matrix']:
        for s in SEEDS:
            for f in range(1,6):
                folder = EEG/'evaluation'/unit['condition']/f'seed_{s}'/f'fold_{f}'/('best' if unit['snapshot']=='best' else 'epoch_'+unit['snapshot'])
                rel = (folder/'complete.json').relative_to(ROOT).as_posix()
                require(rel in completion_sources, 'Completion not bound by prior final report '+rel)
                comp = read(check(completion_sources[rel]))
                require(comp['passed'] and comp['optimizer_updates']==comp['ema_updates']==0, 'Invalid completed prediction evidence')
                b = comp['binding']
                require(b['condition']==unit['condition'] and b['seed']==s and b['fold']==f and str(b['snapshot'])==unit['snapshot'], 'EEG binding identity differs')
                require(b['test_selection_or_preprocessing_fit'] is False, 'Test-informed fitting')
                for k in ('test_gate','registration','checkpoint','metadata','completion','condition_source_lock','evaluator_source'):
                    check(b[k])
                for d in b['input_sources']+b['training_sources']: check(d)
                meta = read(check(b['metadata']))
                parent_registration = read(check(b['registration']))
                original = {r['subject_id']:r for r in byfit[(s,f)]}
                require(set(b['ordered_cohort'])==set(labels), 'EEG cohort mismatch')
                require(set().union(*(set(b['roles'][role]) for role in ROLES))==set(labels), 'Incomplete role sets')
                require(sum(len(b['roles'][role]) for role in ROLES)==37, 'Role overlap')
                for role in ROLES:
                    expected = {sid for sid,r in original.items() if r['split']==role}
                    require(set(b['roles'][role])==expected and set(parent_registration['folds'][f'fold_{f}'][role])==expected, 'EEG/ST registered role mismatch')
                    if role+'_ids' in meta:
                        require(set(meta[role+'_ids'])==expected, 'EEG/ST metadata role mismatch')
                    else:
                        require(role=='test' and unit['condition'] in ('EEGConformer-scratch','EEG2Rep-supervised'), 'Unexpected missing metadata role')
                        require(set(labels)-set(meta['train_ids'])-set(meta['val_ids'])==expected, 'Metadata complement differs from registered test people')
                seen = set()
                for d in comp['batches']:
                    batch = read(check(d))
                    require(batch['binding']==b, 'Batch binding mismatch')
                    require(batch['frozen_state_verified'] and batch['no_gradients_verified'], 'Batch not frozen')
                    require(batch['model_state_before']==batch['model_state_after']==b['selected_model_state_sha256'], 'Model state mismatch')
                    for r in batch['rows']:
                        sid = r['subject_id']
                        require(sid not in seen and sid in original, 'Duplicate/unknown EEG person')
                        seen.add(sid)
                        old = original[sid]
                        require(r['label']==old['label'] and r['role']==old['split'], 'EEG identity/role mismatch')
                        p = r['p_mdd']
                        require(math.isfinite(p) and 0<=p<=1 and int(p>.5)==r['prediction'], 'Invalid EEG probability')
                        require(abs(p+r['p_hc']-1)<1e-12, 'Probability sum mismatch')
                        imported.append(dict(unit=unit['id'], seed=s, fold=f, subject_id=sid,role=r['role'],label=r['label'],
                            p_eeg=p,p_speech=old['p_speech'],p_text=old['p_text'],old_p_eeg=old['p_eeg'],
                            old_st=old_controls[('ST-val-weighted',s,f,sid)]['p_final'],
                            old_est=old_controls[('EST-original-val-weighted',s,f,sid)]['p_final'],
                            old_weights=oldweights[(s,f)], batch_source=d['path']))
                require(seen==set(labels), 'Missing EEG people')
                unit_records.append(dict(unit=unit['id'],seed=s,fold=f,completion=completion_sources[rel],checkpoint=b['checkpoint'],
                    selection=b['checkpoint_selection'],best_epoch=meta.get('best_epoch'),
                    selected_checkpoint_phase=meta.get('selected_checkpoint_phase','not_applicable'),
                    selected_encoder_parameter_change_count=meta.get('selected_encoder_parameter_change_count'), rows=len(seen)))
        print('Source audit:',unit['id'],len(unit_records),'/375',flush=True)
    require(len(imported)==25*555 and len(unit_records)==375, 'Incomplete input matrix')
    # Existing EEG metrics must replay, before any new fusion scores are computed.
    for unit in reg['matrix']:
        data=[dict(r,p=r['p_eeg']) for r in imported if r['unit']==unit['id']]
        actual=aggregate(data)
        expected=evaluation['conditions'][unit['condition']]['variants'][unit['snapshot']]['primary']
        for k in KEYS: require(abs(actual['primary'][k]-expected[k])<1e-12, 'Prior EEG metric mismatch '+unit['id']+' '+k)
    manifest=list(verified.values())
    dump(attempt/'sources_before.json',manifest)
    dump(attempt/'unit_sources.json',unit_records)
    jsonl(attempt/'inputs.jsonl',imported)
    dump(attempt/'preflight.json',dict(passed=True,checked_at=now(),units=375,rows=len(imported),
        sources=len(manifest),bytes_hashed=sum(x['size'] for x in manifest),historical_probabilities_exact=True,
        historical_ST_grid_exact=True,old_eeg_metrics_replayed=True,missing_predictions=0,
        model_loading=0,neural_forwards=0,checkpoint_validation='SHA256 of bound checkpoint bytes; existing frozen evaluation provenance reused'))
    return imported

def aggregate(rows):
    seeds={}
    for s in SEEDS:
        folds={str(f):{role:metrics([r for r in rows if r['seed']==s and r['fold']==f and r['role']==role]) for role in ROLES} for f in range(1,6)}
        test=[r for r in rows if r['seed']==s and r['role']=='test']
        require(len(test)==37 and len({r['subject_id'] for r in test})==37, 'OOF coverage mismatch')
        seeds[str(s)]=dict(folds=folds,fold_mean=average([folds[str(f)]['test'] for f in range(1,6)]),oof=metrics(test))
    return dict(primary=average([v['fold_mean'] for v in seeds.values()]),seeds=seeds,
                roles={role:average([average([seeds[str(s)]['folds'][str(f)][role] for f in range(1,6)]) for s in SEEDS]) for role in ROLES})

def gate(a,b):
    delta={k:a['primary'][k]-b['primary'][k] for k in KEYS}
    ns=sum(a['seeds'][str(s)]['fold_mean']['macro_f1']>b['seeds'][str(s)]['fold_mean']['macro_f1'] for s in SEEDS)
    return dict(deltas=delta,strictly_improved_seeds=ns,
        passed=delta['macro_f1']>=.01 and ns>=2 and delta['balanced_accuracy']>=0 and delta['mdd_recall']>=-.05,
        probability_benefit=delta['nll']<0 and delta['brier']<0,not_a_significance_test=True)

def fusion(r,w):
    return sum(a*r[k] for a,k in zip(w,('p_eeg','p_speech','p_text')))

def execute(attempt,reg,data):
    selections=[]
    selected={}
    # Seal every B weight before reading outer test labels for fusion metrics.
    for unit in reg['matrix']:
        for s in SEEDS:
            for f in range(1,6):
                val=[r for r in data if r['unit']==unit['id'] and r['seed']==s and r['fold']==f and r['role']=='val']
                choice=grid_select(val)
                selections.append(dict(unit=unit['id'],seed=s,fold=f,validation_subject_ids=[r['subject_id'] for r in val],**choice))
                selected[unit['id'],s,f]=choice['weights']
    dump(attempt/'selections.json',selections)
    allpred=[]
    for r in data:
        wmap={'A':[.05,.35,.60],'B':selected[r['unit'],r['seed'],r['fold']],'C':[1/3]*3,'D':r['old_weights']}
        for view,w in wmap.items():
            p=(r['p_eeg']+r['p_speech']+r['p_text'])/3 if view=='C' else fusion(r,w)
            allpred.append(dict(unit=r['unit'],view=view,seed=r['seed'],fold=r['fold'],subject_id=r['subject_id'],
                role=r['role'],label=r['label'],p=p,prediction=int(p>.5),weights=w,threshold_distance=p-.5))
    jsonl(attempt/'predictions.jsonl',allpred)
    base=[r for r in data if r['unit']==reg['matrix'][0]['id']]
    baseline=[]
    baseline_sel=[]
    oldB={}
    for s in SEEDS:
        for f in range(1,6):
            val=[dict(r,p_eeg=r['old_p_eeg']) for r in base if r['seed']==s and r['fold']==f and r['role']=='val']
            b=grid_select(val)
            require(b['weights']==val[0]['old_weights'],'Original EST grid no longer replays')
            baseline_sel.append(dict(seed=s,fold=f,kind='original_E_B',**b))
            baseline_sel.append(dict(seed=s,fold=f,kind='ST_B',**grid_select(val,True)))
            oldB[s,f]=b['weights']
    dump(attempt/'baseline_selections.json',baseline_sel)
    for r in base:
        old=dict(r,p_eeg=r['old_p_eeg'])
        pp={'historical_ST':r['old_st'],'historical_EST':r['old_est'],
            'ST_A':(.35*r['p_speech']+.60*r['p_text'])/.95,'ST_B':r['old_st'],
            'ST_C':(r['p_speech']+r['p_text'])/2,'ST_D':r['old_st'],
            'ST_paper_reference':.45*r['p_speech']+.55*r['p_text'],
            'original_E_A':fusion(old,[.05,.35,.60]),'original_E_B':fusion(old,oldB[r['seed'],r['fold']]),
            'original_E_C':(r['old_p_eeg']+r['p_speech']+r['p_text'])/3,'original_E_D':r['old_est']}
        for name,p in pp.items():
            baseline.append(dict(unit=name,seed=r['seed'],fold=r['fold'],subject_id=r['subject_id'],role=r['role'],label=r['label'],p=p,prediction=int(p>.5)))
    jsonl(attempt/'baseline_predictions.jsonl',baseline)
    bm={name:aggregate([r for r in baseline if r['unit']==name]) for name in sorted({r['unit'] for r in baseline})}
    oldmetrics=read(OLD/'metrics.json')
    for name,oldname in [('historical_ST','ST-val-weighted'),('historical_EST','EST-original-val-weighted')]:
        for k in KEYS: require(abs(bm[name]['primary'][k]-oldmetrics[oldname]['main_test'][k])<1e-10,'Historical metric mismatch '+k)
    results={u['id']:{v:aggregate([r for r in allpred if r['unit']==u['id'] and r['view']==v]) for v in VIEWS} for u in reg['matrix']}
    comparisons={}
    summary=[]
    for u in reg['matrix']:
        comparisons[u['id']]={}
        for v in VIEWS:
            a=results[u['id']][v]
            ctrls={'original_E':bm['original_E_'+v],'ST':bm['ST_'+v], 'historical_ST':bm['historical_ST'],'historical_EST':bm['historical_EST']}
            if u['matched_control']: ctrls['matched_LaBraM']=results[u['matched_control']][v]
            comp={k:gate(a,b) for k,b in ctrls.items()}
            qualifies=bool(u['candidate'] and all(comp[k]['passed'] for k in ('matched_LaBraM','original_E','ST')) and
                comp['historical_ST']['deltas']['macro_f1']>0 and comp['historical_EST']['deltas']['macro_f1']>0)
            comparisons[u['id']][v]=dict(comparisons=comp,protocol_retention=qualifies,primary_retention=qualifies if v=='A' else None)
            summary.append(dict(**u,view=v,**a['primary'],delta_matched=comp.get('matched_LaBraM',{}).get('deltas',{}).get('macro_f1'),
                delta_original=comp['original_E']['deltas']['macro_f1'],delta_ST=comp['ST']['deltas']['macro_f1'],
                delta_historical_ST=comp['historical_ST']['deltas']['macro_f1'],retained=qualifies))
    # Subject-level changes versus the protocol's ST, with unique people reported separately.
    bmap={(r['unit'],r['seed'],r['fold'],r['subject_id']):r for r in baseline}
    changes=[]
    change_summary={}
    for r in allpred:
        if r['role']!='test': continue
        b=bmap[('ST_'+r['view'],r['seed'],r['fold'],r['subject_id'])]
        before=b['prediction']==r['label']; after=r['prediction']==r['label']
        status='corrected' if after and not before else 'new_error' if before and not after else 'probability_only' if r['p']!=b['p'] else 'identical'
        changes.append(dict(unit=r['unit'],view=r['view'],seed=r['seed'],fold=r['fold'],subject_id=r['subject_id'],label=r['label'],
            p_before=b['p'],p_after=r['p'],status=status))
    jsonl(attempt/'paired_changes.jsonl',changes)
    for u in reg['matrix']:
        change_summary[u['id']]={}
        for v in VIEWS:
            rr=[r for r in changes if r['unit']==u['id'] and r['view']==v]
            change_summary[u['id']][v]={status:dict(records=sum(r['status']==status for r in rr),
                distinct_people=len({r['subject_id'] for r in rr if r['status']==status}),
                mdd_records=sum(r['status']==status and r['label']==1 for r in rr),hc_records=sum(r['status']==status and r['label']==0 for r in rr))
                for status in ('corrected','new_error','probability_only','identical')}
    duration={}
    for model in ('EEGPT-mean-projection4s-originalCNN-300ep','LaBraM-native4s-originalCNN-300ep'):
        duration[model]={v:gate(results[model+'@300'][v],results[model+'@60'][v]) for v in VIEWS}
    dump(attempt/'results.json',dict(baselines=bm,conditions=results,comparisons=comparisons,duration=duration,changes=change_summary))
    with (attempt/'summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print('All 100 condition/view summaries calculated; ready for independent verification.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'complete.json').exists(),'Already complete; do not repeat')
    reg=registration();regpath=OUT/'registration.json'
    if regpath.exists() and read(regpath)!=reg:
        initial=read(regpath)
        require({k:v for k,v in initial.items() if k!='code'}=={k:v for k,v in reg.items() if k!='code'}, 'Protocol changed during repair')
        archive=OUT/'preflight_schema_repair_original_sources'
        for d in initial['code']:
            require(sha(archive/Path(d['path']).name)==d['sha256'],'Original registered source not preserved')
        reg['preflight_schema_repair']=dict(supersedes=desc(regpath),reason='Conformer/EEG2Rep metadata has train/val IDs only; validate test IDs against original registration and cohort complement',
            protocol_changed=False,new_fusion_results_before_repair=False,original_sources=[desc(p) for p in sorted(archive.glob('*.py'))])
        regpath=OUT/'registration_v2.json'
    dump(regpath,reg)
    dump(OUT/'active_registration.json',desc(regpath))
    attempt=OUT/('attempt_'+str(time.time_ns()));attempt.mkdir()
    started=time.perf_counter()
    dump(attempt/'started.json',dict(started_at=now(),registration=desc(regpath),python=sys.version,argv=sys.argv))
    try:
        data=build_inputs(attempt,reg)
        execute(attempt,reg,data)
        # Full byte-level after-check of all bound sources, not merely stat checks.
        sources=read(attempt/'sources_before.json')
        for d in sources: require(sha(file_path(d['path']))==d['sha256'],'Protected source changed: '+d['path'])
        dump(attempt/'execution_complete.json',dict(passed=True,ended_at=now(),elapsed_seconds=time.perf_counter()-started,
            units=375,views=1500,validation_selections=375,validation_candidate_scores=86625,
            baseline_grid_replays=30,baseline_grid_scores=15*(231+21),new_training=0,optimizer_updates=0,
            model_loads=0,neural_forwards=0,asr=0,source_hashes_after_verified=len(sources),independent_verification_pending=True))
        dump(OUT/'execution_pointer.json',dict(attempt=attempt.relative_to(ROOT).as_posix()))
        print('EXECUTION_COMPLETE',attempt.relative_to(ROOT).as_posix(),flush=True)
    except BaseException as e:
        dump(attempt/'failed.json',dict(ended_at=now(),elapsed_seconds=time.perf_counter()-started,error_type=type(e).__name__,error=str(e)))
        raise

if __name__=='__main__':main()
