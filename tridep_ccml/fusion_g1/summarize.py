from collections import defaultdict
from .common import *
from tridep_ccml.eeg_p3.summarize import metric

def main():
    lock_check();assert read(OUT/'evaluation_complete.json')['models']==90
    grouped=defaultdict(list); originals={}; training=[]
    for path in sorted((OUT/'predictions').rglob('fold_*.json')):
        obj=read(path);c=obj['condition'];seed=obj['seed'];fold=obj['fold']
        for row in obj['rows']:
            originals[(c,seed,fold,row['subject_id'])]=row
            for endpoint,p in row['probabilities'].items():
                grouped[(c,endpoint,row['split'])].append({'seed':seed,'fold':fold,'subject_id':row['subject_id'],
                    'label':row['label'],'p':p,'threshold':.5})
    results=[]
    for (c,ep,split),rows in sorted(grouped.items()):
        folds=[{'seed':s,'fold':f,**metric([r for r in rows if r['seed']==s and r['fold']==f])} for s in SEEDS for f in FOLDS]
        seeds=[{'seed':s,**{k:float(np.mean([f[k] for f in folds if f['seed']==s])) for k in METRICS}} for s in SEEDS]
        oof=[]
        if split=='test':
            for s in SEEDS:
                subset=[r for r in rows if r['seed']==s]
                assert len(subset)==len({r['subject_id'] for r in subset})==37
                oof.append({'seed':s,**metric(subset)})
        results.append({'condition':c,'endpoint':ep,'split':split,'mean':{k:float(np.mean([s[k] for s in seeds])) for k in METRICS},
            'per_seed':seeds,'per_fold':folds,'pooled_OOF_per_seed':oof})
    lookup={(r['condition'],r['endpoint'],r['split']):r for r in results}
    comparisons=[]
    def compare(new,old,kind):
        a=lookup[(new,'formal','test')];b=lookup[(old,'formal','test')]
        delta={k:a['mean'][k]-b['mean'][k] for k in METRICS}
        sd=[{'seed':s['seed'],**{k:s[k]-b['per_seed'][i][k] for k in METRICS}} for i,s in enumerate(a['per_seed'])]
        improved=sum(x['macro_f1']>0 for x in sd)
        gate=delta['macro_f1']>=.01 and improved>=2 and delta['mdd_recall']>=-.05
        comparisons.append({'new':new,'control':old,'kind':kind,'delta':delta,'per_seed_delta':sd,
            'F1_seeds_improved':improved,'classification_gate_passed':gate,
            'probability_quality_improved':delta['nll']<0 and delta['brier']<0})
    for family in ['CE','EDL','CCML']:compare(f'G-{family}-EST',f'G-{family}-ST','EEG_increment')
    for mods in ['ST','EST']:
        compare(f'G-CCML-{mods}',f'G-CE-{mods}','CCML_vs_CE')
        compare(f'G-EDL-{mods}',f'G-CE-{mods}','EDL_vs_CE_auxiliary')
    pairs=[]
    ctx=RecordContext()
    for family in ['CE','EDL','CCML']:
        est=f'G-{family}-EST';st=f'G-{family}-ST'
        for comparator in ['retrained_ST','fixed_ST','speech','text']:
            rows=[]
            for seed in SEEDS:
                for fold in FOLDS:
                    for sid in ctx.folds[f'fold_{fold}']['test']:
                        row=originals[(est,seed,fold,sid)]
                        base=(originals[(st,seed,fold,sid)]['probabilities']['formal'] if comparator=='retrained_ST'
                              else row['probabilities'][comparator])
                        p=row['probabilities']['formal'];y=row['label'];bc=int(base>.5)==y;nc=int(p>.5)==y
                        rows.append({'subject_id':sid,'seed':seed,'fold':fold,'label':y,'base_p':base,'EST_p':p,
                            'base_correct':bc,'EST_correct':nc,'transition':'gain' if nc and not bc else 'loss' if bc and not nc else 'unchanged'})
            byseed=[{'seed':s,'gains':sum(r['transition']=='gain' for r in rows if r['seed']==s),
                'losses':sum(r['transition']=='loss' for r in rows if r['seed']==s)} for s in SEEDS]
            byperson=[{'subject_id':sid,'label':ctx.labels[sid],
                'gain_seeds':[r['seed'] for r in rows if r['subject_id']==sid and r['transition']=='gain'],
                'loss_seeds':[r['seed'] for r in rows if r['subject_id']==sid and r['transition']=='loss']} for sid in ctx.ids]
            pairs.append({'family':family,'comparator':comparator,'per_seed':byseed,'per_person':byperson,'rows':rows,
                'unique_people':37,'repeated_seed_records':111})
    evidence=[]
    for c in CONDITIONS:
        if '-CE-' in c:
            evidence.append({'condition':c,'applicable':False,'reason':'CE produces probabilities, not Dirichlet evidence'});continue
        names=['eeg','speech','text'] if c.endswith('-EST') else ['speech','text']
        for i,m in enumerate(names):
            for correctness in ['all','correct','wrong']:
                subset=[r for (cond,s,f,sid),r in originals.items() if cond==c and r['split']=='test' and
                    (correctness=='all' or (((r['probabilities'][m]>.5)==r['label'])==(correctness=='correct')))]
                evidence.append({'condition':c,'modality':m,'group':correctness,'repeated_records':len(subset),
                    'mean_strength':float(np.mean([r['branch_strength'][i] for r in subset])) if subset else None,
                    'mean_u':float(np.mean([r['branch_u'][i] for r in subset])) if subset else None,'applicable':True})
    for c in CONDITIONS:
        runs=[read(p) for p in sorted((OUT/f'runs/{c}').rglob('complete.json'))]
        histories=[read(p) for p in sorted((OUT/f'runs/{c}').rglob('history.json'))]
        modalities=['eeg','speech','text'] if c.endswith('-EST') else ['speech','text']
        training.append({'condition':c,'models':len(runs),'updates':sum(r['updates'] for r in runs),
            'mean_clip_rate':float(np.mean([r['clip_rate'] for r in runs])),
            'mean_epoch_1_loss':float(np.mean([h[0]['loss'] for h in histories])),
            'mean_epoch_60_loss':float(np.mean([h[-1]['loss'] for h in histories])),
            'branch_preclip_gradient_mean':{m:float(np.mean([ep['branch_gradient_norm_mean'][m] for h in histories for ep in h])) for m in modalities},
            'branch_state_change_l2_mean':{m:float(np.mean([r['branch_state_change_l2'][m] for r in runs])) for m in modalities},
            'all_reload_exact':all(r['reload_exact'] for r in runs)})
    eeg_gates={c['new'].split('-')[1]:c['classification_gate_passed'] for c in comparisons if c['kind']=='EEG_increment'}
    ccml_gates={c['new'].split('-')[2]:c['classification_gate_passed'] for c in comparisons if c['kind']=='CCML_vs_CE'}
    decision={'EEG_increment_by_family':eeg_gates,'CCML_vs_CE_by_modality':ccml_gates,
        'stop_this_EEG_fusion_version':not any(eeg_gates.values()),
        'stop_CCML_EST_SGF_extensions':not ccml_gates['EST'],
        'EEG_engineering_passed':True,'independent_EEG_screen_passed':False,
        'predeclared_CCML_EST_research_candidate':eeg_gates['CCML'],
        'independent_population_validation':False,'leaderboard_winner_selected':False}
    REPORT.mkdir(parents=True,exist_ok=True)
    save(REPORT/'results.json',{'aggregation':'five folds equally weighted per seed, then three seeds equally weighted',
        'results':results,'comparisons':comparisons,'training':training,'decision':decision},immutable=True)
    save(REPORT/'subject_paired_diagnostics.json',pairs,immutable=True)
    save(REPORT/'evidence_diagnostics.json',evidence,immutable=True)
    print('Formal test metrics:',flush=True)
    for c in CONDITIONS:print(c,lookup[(c,'formal','test')]['mean'],flush=True)
    print('Decision',decision,flush=True)

if __name__=='__main__':main()
