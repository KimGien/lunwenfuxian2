from collections import defaultdict
from .common import *
from tridep_ccml.eeg_p3.summarize import metric

def main():
    cfg=config_check();lock=read(OUT/'postprocessors_lock.json');assert lock['passed']
    for p,sha in lock['artifacts'].items():assert digest(OUT/p)==sha,p
    for d in read(OUT/'postprocess_revision_lock.json')['files']:verify(d)
    groups=defaultdict(list);pairs=[];params=[]
    for c in CONDITIONS:
        for seed in SEEDS:
            for fold in FOLDS:
                path=G1/f'predictions/{c}/seed_{seed}/fold_{fold}.json'
                assert desc(path) in cfg['parents']
                old=read(path);calpath=OUT/f'calibrators/{c}/seed_{seed}/fold_{fold}.json';cal=read(calpath)
                threshold=cal['parameters']['threshold']['threshold'];sig=cal['parameters']['sigmoid']
                rows=[]
                for r in old['rows']:
                    raw=r['probabilities']['formal'];q=float(apply_sigmoid(raw,sig))
                    variants={'raw':{'p':raw,'threshold':.5},'threshold':{'p':raw,'threshold':threshold},'sigmoid':{'p':q,'threshold':.5}}
                    row={k:r[k] for k in ['subject_id','label','split']};row['variants']=variants;rows.append(row)
                    for v,pp in variants.items():
                        record={**{k:row[k] for k in ['subject_id','label','split']},'seed':seed,'fold':fold,**pp}
                        groups[(c,v,r['split'])].append(record)
                        if r['split']=='test' and v!='raw':
                            before=int(raw>.5)==r['label'];after=int(pp['p']>pp['threshold'])==r['label']
                            pairs.append({'condition':c,'variant':v,'seed':seed,'fold':fold,'subject_id':r['subject_id'],
                                'label':r['label'],'raw_p':raw,**pp,'raw_correct':before,'postprocess_correct':after,
                                'transition':'gain' if after and not before else 'loss' if before and not after else 'unchanged'})
                save(OUT/f'predictions/{c}/seed_{seed}/fold_{fold}.json',{'condition':c,'seed':seed,'fold':fold,
                    'source_prediction':desc(path),'calibrator':desc(calpath),'rows':rows,
                    'sigmoid_has_dirichlet_evidence':False,'raw_scores_previously_published':True},immutable=True)
                params.append({'condition':c,'seed':seed,'fold':fold,'threshold':threshold,'a':sig['a'],'b':sig['b'],
                    'OOF_optimistic_threshold_fit_f1':cal['parameters']['threshold']['oof_fit_macro_f1'],
                    'sigmoid_identity_equivalent_raw_threshold':float(1/(1+np.exp(sig['b']/sig['a']))),
                    'slope_on_bound':min(abs(sig['a']-.01),abs(sig['a']-10))<1e-8,
                    'intercept_on_bound':min(abs(sig['b']+10),abs(sig['b']-10))<1e-8})
    results=[]
    for (c,v,split),rows in sorted(groups.items()):
        fold_metrics=[{'seed':s,'fold':f,**metric([r for r in rows if r['seed']==s and r['fold']==f])} for s in SEEDS for f in FOLDS]
        seeds=[{'seed':s,**{k:float(np.mean([r[k] for r in fold_metrics if r['seed']==s])) for k in METRICS}} for s in SEEDS]
        oof=[]
        if split=='test':
            for s in SEEDS:
                sr=[r for r in rows if r['seed']==s];assert len(sr)==len({r['subject_id'] for r in sr})==37
                oof.append({'seed':s,**metric(sr)})
        results.append({'condition':c,'variant':v,'split':split,'mean':{k:float(np.mean([r[k] for r in seeds])) for k in METRICS},
            'per_seed':seeds,'per_fold':fold_metrics,'pooled_OOF_per_seed':oof})
    lookup={(r['condition'],r['variant'],r['split']):r for r in results};comparisons=[]
    def compare(new,old,kind):
        a=lookup[(*new,'test')];b=lookup[(*old,'test')];delta={k:a['mean'][k]-b['mean'][k] for k in METRICS}
        sd=[{'seed':s['seed'],**{k:s[k]-b['per_seed'][i][k] for k in METRICS}} for i,s in enumerate(a['per_seed'])]
        improved=sum(s['macro_f1']>0 for s in sd)
        comparisons.append({'candidate':list(new),'control':list(old),'kind':kind,'delta':delta,'per_seed_delta':sd,
            'F1_seeds_improved':improved,'classification_reference_gate_passed':delta['macro_f1']>=.01 and improved>=2 and delta['mdd_recall']>=-.05,
            'probability_quality_improved':delta['nll']<0 and delta['brier']<0})
    for c in CONDITIONS:
        for v in ['threshold','sigmoid']:compare((c,v),(c,'raw'),'postprocessing_vs_raw')
    for v in VARIANTS:compare(('G-CCML-EST',v),('G-CCML-ST',v),'matched_postprocessing_EEG_increment')
    pairsummary=[]
    for c in CONDITIONS:
        for v in ['threshold','sigmoid']:
            rr=[r for r in pairs if r['condition']==c and r['variant']==v]
            pairsummary.append({'condition':c,'variant':v,'unique_people':37,'repeated_seed_records':111,
                'per_seed':[{'seed':s,**{t:sum(r['seed']==s and r['transition']==t for r in rr) for t in ['gain','loss']}} for s in SEEDS],
                'per_label':[{'label':y,**{t:sum(r['label']==y and r['transition']==t for r in rr) for t in ['gain','loss']}} for y in [0,1]],
                'per_person':[{'subject_id':sid,'gain_seeds':[r['seed'] for r in rr if r['subject_id']==sid and r['transition']=='gain'],
                    'loss_seeds':[r['seed'] for r in rr if r['subject_id']==sid and r['transition']=='loss']} for sid in cfg['outer_folds']['fold_1']['train']+cfg['outer_folds']['fold_1']['val']+cfg['outer_folds']['fold_1']['test']]})
    REPORT.mkdir(parents=True,exist_ok=True)
    save(REPORT/'results.json',{'study_type':'post-G1-release development extension','G1_outer_results_already_seen':True,
        'aggregation':'five-fold equally weighted means, then three seeds','results':results,'comparisons':comparisons,
        'best_variant_selected':False,'G1_conclusions_overwritten':False,'independent_population_validation':False},immutable=True)
    save(REPORT/'postprocessor_parameters.json',params,immutable=True)
    save(REPORT/'subject_changes.json',{'summary':pairsummary,'rows':pairs},immutable=True)
    save(OUT/'evaluation_complete.json',{'completed_utc':now(),'outer_models_reused':30,'base_subject_model_records':1110,
        'variant_records':3330,'test_variant_records':666,'unique_people':37,
        'artifacts':{p.relative_to(OUT).as_posix():digest(p) for p in sorted((OUT/'predictions').rglob('*.json'))}},immutable=True)
    for c in CONDITIONS:
        for v in VARIANTS:print(c,v,lookup[(c,v,'test')]['mean'],flush=True)

if __name__=='__main__':main()
