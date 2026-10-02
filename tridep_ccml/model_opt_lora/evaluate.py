from .core import *
from tridep_ccml.fusion_g1.evaluate import outputs
from tridep_ccml.model_opt_v1.evaluate import rows_for,summarize

def main():
    lock_check();gate=read(OUT/'release_gate.json');assert gate['passed'] and gate['models']==30
    for name,sha in gate['fitted_artifacts'].items():assert digest(OUT/name)==sha,name
    assert not (OUT/'evaluation_complete.json').exists()
    set_seed(11);data=Data();all_rows=[];result={};branches={};exports=[];reloads=0
    for c in CONDITIONS:
        collected=[]
        for seed in SEEDS:
            for fold in FOLDS:
                path=stem(c,seed,fold);cp=torch.load(path/'final.pt',weights_only=True,map_location='cpu')
                assert cp['fit_ids']==data.ctx.folds[f'fold_{fold}']['train']
                assert cp['condition']==c and cp['seed']==seed and cp['fold']==fold
                m=Model(c,seed).cuda();m.load(cp['state'])
                assert state_hash(m.state())==cp['final_state_sha256'] and m.frozen_hash()==cp['frozen_layer_sha256']
                p=infer(m,data,data.ids)
                re=Model(c,seed).cuda();re.load(torch.load(path/'final.pt',weights_only=True,map_location='cpu')['state'])
                again=infer(re,data,data.ids)
                assert all(torch.equal(p[k],again[k]) for k in ['logits','features','normalized','question_cls'])
                assert state_hash(m.state())==cp['final_state_sha256'];reloads+=1
                ep,e,b=outputs(p['logits'],'CCML',m.modalities)
                extra=lambda i:{'probabilities':{k:float(v[i]) for k,v in ep.items()},'logits':p['logits'][i].tolist(),
                    'evidence':e[i].tolist(),'branch_u':b['u'][i].tolist()}
                rows=rows_for(data,c,seed,fold,ep['formal'].numpy(),extra)
                target=OUT/f'predictions/{c}/seed_{seed}/fold_{fold}.json';target.parent.mkdir(parents=True,exist_ok=True)
                feature=target.with_suffix('.pt')
                torch.save({'condition':c,'seed':seed,'fold':fold,'fit_ids':cp['fit_ids'],'subject_ids':data.ids,
                    'checkpoint':desc(path/'final.pt'),'label_free_cache':desc(OUT/'cache_manifest.json'),
                    'question_cls':p['question_cls'],'text_pooled':p['question_cls'].mean(1),
                    'projected':p['features'],'normalized':p['normalized'],'logits':p['logits'],
                    'supervised_adapter_scope':'this checkpoint and its training fold only'},feature)
                exports.append({'condition':c,'seed':seed,'fold':fold,'fit_ids':cp['fit_ids'],
                    'checkpoint':desc(path/'final.pt'),'feature':desc(feature)})
                save(target,{'checkpoint':desc(path/'final.pt'),'features':desc(feature),'modalities':m.modalities,'rows':rows},immutable=True)
                collected+=rows;del m,re,cp,p,again
        result[c]=summarize(collected,SEEDS)
        branches[c]={k:summarize([{**r,'p':r['probabilities'][k]} for r in collected],SEEDS)
            for k in ['speech','text','probability_mean']}
        all_rows+=collected;print('EVALUATED',c,flush=True)
    assert reloads==30 and len(all_rows)==1110
    assert len(exports)==len({r['feature']['path'] for r in exports})==len({r['checkpoint']['path'] for r in exports})==30
    save(OUT/'metrics.json',result,immutable=True);save(OUT/'branch_metrics.json',branches,immutable=True)
    save(OUT/'all_predictions.json',all_rows,immutable=True)
    save(OUT/'feature_manifest.json',{'exports':exports,'supervised_features_shared_between_folds':False,
        'exported_after_all_fits_locked':True,'subjects_per_export':37,'questions_per_subject':29},immutable=True)
    save(OUT/'evaluation_complete.json',{'passed':True,'completed_utc':now(),'reloads_exact':reloads,'rows':1110,
        'test_oof_rows':222,'independent_subjects':37,'formal_fold_metric_checks':90,'all_endpoint_fold_metric_checks':360,
        'fold_bound_feature_exports':30,'all_results_released_together':True},immutable=True)
    save(OUT/'status.json',{'stage':'evaluated','models_completed':30,'updates_completed':7200,
        'outer_results_released':True,'updated_utc':now()})
    print('Exact reload, independent metrics, and 30 fold-bound feature exports passed.',flush=True)

if __name__=='__main__':main()
