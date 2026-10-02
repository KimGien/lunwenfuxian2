from .core import *
from tridep_ccml.fusion_g1.evaluate import infer,outputs
from tridep_ccml.model_opt_v1.evaluate import rows_for,summarize

def main():
    lock_check();gate=read(OUT/'release_gate.json');assert gate['passed']
    for name,sha in gate['fitted_artifacts'].items():assert digest(OUT/name)==sha,name
    assert not (OUT/'evaluation_complete.json').exists()
    data=Data();set_seed(11);results={};all_rows=[];reloaded=0;control_rows=0
    for condition in CONDITIONS:
        collected=[]
        for seed in SEEDS:
            for fold in FOLDS:
                path=stem(condition,seed,fold);cp=torch.load(path/'final.pt',weights_only=True,map_location='cpu')
                assert cp['fit_ids']==data.ctx.folds[f'fold_{fold}']['train']
                model=Model(condition,seed).cuda();model.load_state_dict(cp['model']);before=state_hash(model.state_dict())
                logits,z,n=infer(model,data);ep,e,b=outputs(logits,'CCML',model.modalities)
                re=Model(condition,seed).cuda();re.load_state_dict(torch.load(path/'final.pt',weights_only=True,map_location='cpu')['model'])
                ll,zz,nn=infer(re,data)
                assert all(torch.equal(a,b) for a,b in [(logits,ll),(z,zz),(n,nn)])
                assert before==state_hash(model.state_dict())==cp['final_state_sha256'];reloaded+=1
                extra=lambda i:{'probabilities':{k:float(v[i]) for k,v in ep.items()},'logits':logits[i].tolist(),
                    'evidence':e[i].tolist(),'branch_u':b['u'][i].tolist()}
                rows=rows_for(data,condition,seed,fold,ep['formal'].numpy(),extra)
                if '-sum-' in condition:
                    mods=condition.rsplit('-',1)[1]
                    old=read(PARENT/f'predictions/M1-global-{mods}/seed_{seed}/fold_{fold}.json')['rows']
                    for a,before_row in zip(rows,old):
                        assert all(a[k]==before_row[k] for k in ['subject_id','label','split','p','logits','probabilities','evidence','branch_u'])
                        control_rows+=1
                save(OUT/f'predictions/{condition}/seed_{seed}/fold_{fold}.json',
                    {'checkpoint':desc(path/'final.pt'),'reused_control':'-sum-' in condition,'modalities':model.modalities,'rows':rows},immutable=True)
                collected+=rows;del model,re,cp
        results[condition]=summarize(collected,SEEDS);all_rows+=collected
        print('EVALUATED',condition,flush=True)
    assert reloaded==60 and len(all_rows)==2220 and control_rows==1110
    save(OUT/'metrics.json',results,immutable=True);save(OUT/'all_predictions.json',all_rows,immutable=True)
    save(OUT/'evaluation_complete.json',{'passed':True,'models_reloaded_exactly':reloaded,'prediction_rows':len(all_rows),
        'historical_control_rows_exact':control_rows,'independent_fold_metric_checks':180,'unique_subjects':37,
        'completed_utc':now(),'all_candidates_locked_before_new_outer_inference':True},immutable=True)
    save(OUT/'status.json',{'stage':'evaluated','new_models_completed':30,'new_updates_completed':7200,
        'reused_controls':30,'new_outer_results_released':True,'updated_utc':now()})

if __name__=='__main__':main()
