"""Post-lock prediction, paired comparisons and checkpoint reloading."""
from collections import defaultdict
from tridep_ccml.eeg_p3.summarize import metric, METRICS
from .training import *


def gate_check():
    config_check();gate=read(OUT/'release_gate.json');assert gate['passed']
    for name,sha in gate['fitted_artifacts'].items():assert digest(OUT/name)==sha,name


def aggregate(allrows):
    groups=defaultdict(list)
    for row in allrows:groups[(row['condition'],row['split'])].append(row)
    results=[]
    for (name,split),rows in sorted(groups.items()):
        per_fold=[];per_seed=[];pooled=[]
        for seed in sorted(set(r['seed'] for r in rows)):
            sr=[r for r in rows if r['seed']==seed]
            for fold in FOLDS:
                fr=[r for r in sr if r['fold']==fold]
                assert fr
                per_fold.append({'seed':seed,'fold':fold,**metric(fr)})
            per_seed.append({'seed':seed,**{k:float(np.mean([r[k] for r in per_fold if r['seed']==seed])) for k in METRICS}})
            if split=='test':
                assert len(sr)==37 and len(set(r['subject_id'] for r in sr))==37
                pooled.append({'seed':seed,**metric(sr)})
        result={'condition':name,'split':split,'mean':{k:float(np.mean([r[k] for r in per_seed])) for k in METRICS},
            'per_fold':per_fold,'per_seed':per_seed,'pooled_oof':pooled,
            'aggregation':'unweighted five-fold mean, then seed mean; deterministic linear/baselines have one run per fold'}
        if name=='E-HC':
            for obj in [result['mean']]+per_fold+per_seed+pooled:
                for k in ['auc','nll','brier']:obj[k]=None
        results.append(result)
    return results


def run():
    gate_check();data=Data();allrows=[];checks=[]
    for fold in FOLDS:
        data.select(fold);splitmap={s:k for k,ids in data.splits.items() for s in ids}
        for seed in SEEDS:
            th=read(OUT/f'thresholds/seed_{seed}/fold_{fold}.json')['parameters']['threshold']
            for name,kind,old in [('E-N0','old',True),('E-N1','new',False)]:
                data.kind=kind;model=load(seed,fold,old=old);p,z=predict(model,data,data.ids)
                other=load(seed,fold,old=old);rp,rz=predict(other,data,data.ids)
                assert np.array_equal(p,rp) and np.array_equal(z,rz)
                solo,_=predict(model,data,data.ids[:1]);np.testing.assert_allclose(solo,p[:1],rtol=2e-5,atol=2e-6)
                if old:
                    doc=read(P3/f'predictions/U1_S/seed_{seed}/fold_{fold}.json')
                    byid={r['subject_id']:r['probabilities']['eeg'] for r in doc['rows']}
                    historical=np.array([byid[s] for s in data.ids])
                    np.testing.assert_allclose(p,historical,rtol=1e-5,atol=1e-6)
                    assert np.array_equal(p>.5,historical>.5)
                    # Reuse the frozen P3 probabilities after successful replay, as planned.
                    p=historical
                rows=[]
                for sid,value in zip(data.ids,p):
                    row={'condition':name,'seed':seed,'fold':fold,'subject_id':sid,'label':data.labels[sid],
                         'split':splitmap[sid],'p':float(value),'threshold':.5}
                    rows.append(row);allrows.append(row)
                    if not old:allrows.append({**row,'condition':'E-N1-threshold','threshold':th})
                target=OUT/f'predictions/{name}/seed_{seed}/fold_{fold}.json'
                save(target,{'rows':rows,'reload_exact':True,'batch_independence_checked':True},immutable=True)
                np.savez_compressed(target.with_suffix('.npz'),subject_ids=np.array(data.ids),features=z)
                checks.append({'condition':name,'seed':seed,'fold':fold,'reload_exact':True,'old_predictions_replayed':old})
                del model,other
            # All inner checkpoint predictions must reproduce the saved OOF values.
            data.kind='new';doc=read(OUT/f'thresholds/seed_{seed}/fold_{fold}.json')
            for part in config_check()['inner_splits'][str(fold)]:
                model=load(seed,fold,part['inner']);p,_=predict(model,data,part['calibration'])
                stored={r['subject_id']:r['p'] for r in doc['oof']}
                assert np.array_equal(p,np.array([stored[s] for s in part['calibration']]))
                assert doc['parameters']==fit_threshold(doc['oof'],data.train)
                del model
        for kind,name in [('old','E-L0'),('new','E-L1')]:
            x=np.load(OUT/f'linear/{kind}/features.npy');model=joblib.load(OUT/f'linear/{kind}/fold_{fold}.joblib')
            fit=[data.index[s] for s in data.train]
            np.testing.assert_allclose(model[0].mean_,x[fit].mean(axis=0),rtol=0,atol=1e-12)
            p=model.predict_proba(x)[:,1]
            cp=model[-1];independent=1/(1+np.exp(-(model[0].transform(x)@cp.coef_[0]+cp.intercept_[0])))
            np.testing.assert_allclose(p,independent,rtol=0,atol=1e-14)
            for sid,value in zip(data.ids,p):
                allrows.append({'condition':name,'seed':0,'fold':fold,'subject_id':sid,'label':data.labels[sid],
                    'split':splitmap[sid],'p':float(value),'threshold':.5})
        prior=float(np.mean([data.labels[s] for s in data.train]))
        for name,prob in [('E-HC',0.),('E-PRIOR',prior)]:
            for sid in data.ids:
                allrows.append({'condition':name,'seed':0,'fold':fold,'subject_id':sid,'label':data.labels[sid],
                    'split':splitmap[sid],'p':prob,'threshold':.5})
        print('EVALUATED',fold,'/5',flush=True)
    results=aggregate(allrows)
    lookup={(r['condition'],r['split']):r for r in results}
    comparisons=[]
    for new,old,minimum in [('E-N1','E-N0',2),('E-L1','E-L0',3),('E-N1-threshold','E-N1',None)]:
        a,b=lookup[(new,'test')],lookup[(old,'test')]
        delta={k:a['mean'][k]-b['mean'][k] for k in METRICS}
        seed_delta=[{'seed':r['seed'],**{k:r[k]-b['per_seed'][i][k] for k in METRICS}} for i,r in enumerate(a['per_seed'])]
        fold_delta=[{'seed':r['seed'],'fold':r['fold'],**{k:r[k]-b['per_fold'][i][k] for k in METRICS}} for i,r in enumerate(a['per_fold'])]
        improves=sum(r['macro_f1']>0 for r in (fold_delta if new=='E-L1' else seed_delta))
        passed=(delta['macro_f1']>=.03 and improves>=minimum and delta['mdd_recall']>=-.05) if minimum else None
        comparisons.append({'new':new,'old':old,'delta':delta,'per_seed':seed_delta,'per_fold':fold_delta,
                            'improvement_gate':passed,'improved_seed_or_fold_count':improves})
    nn=lookup[('E-N1','test')]
    collapsed=[{'seed':r['seed'],'fold':r['fold'],'predicted_MDD':r['predicted_MDD']} for r in nn['per_fold'] if r['predicted_MDD'] in [0,r['n_people']]]
    usability={'balanced_accuracy_above_chance':nn['mean']['balanced_accuracy']>.5,
        'mdd_recall_at_least_half':nn['mean']['mdd_recall']>=.5,'no_single_class_test_fold':not collapsed,
        'auc_above_chance':nn['mean']['auc']>.5,'collapsed_folds':collapsed}
    usability['passed']=all(usability[k] for k in ['balanced_accuracy_above_chance','mdd_recall_at_least_half','no_single_class_test_fold'])
    save(REPORT/'predictions_long.json',allrows,immutable=True)
    save(REPORT/'metrics.json',results,immutable=True)
    save(REPORT/'comparisons.json',comparisons,immutable=True)
    save(REPORT/'usability.json',usability,immutable=True)
    save(OUT/'evaluation_verification.json',{'passed':True,'old_models_replayed':15,'new_outer_models_reloaded':15,
        'new_inner_models_reloaded':45,'inner_thresholds_recomputed':15,'linear_pipelines_verified':10,
        'all_rows':len(allrows),'checks':checks,'released_after_all_fits':True},immutable=True)
    save(OUT/'status.json',{'state':'evaluation_complete_report_pending','new_neural':60,'new_linear':10,'updated_utc':now()})
    for r in results:
        if r['split']=='test':print('RESULT',r['condition'],r['mean'],flush=True)
    print('COMPARISONS',comparisons,'USABILITY',usability,flush=True)


if __name__=='__main__':run()
