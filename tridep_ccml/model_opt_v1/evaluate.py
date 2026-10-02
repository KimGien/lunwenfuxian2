import joblib
from torch.nn import functional as F
from sklearn.metrics import f1_score,balanced_accuracy_score,recall_score,roc_auc_score,log_loss,brier_score_loss
from .core import *
from tridep_ccml.fusion_g1.evaluate import infer,outputs
from tridep_ccml.metrics import binary_metrics

def metrics(rows):
    y=[r['label'] for r in rows];p=[r['p'] for r in rows]
    result=binary_metrics(y,p);result['auc']=float(roc_auc_score(y,p)) if len(set(y))==2 else None
    independent={'macro_f1':f1_score(y,np.array(p)>.5,labels=[0,1],average='macro',zero_division=0),
        'balanced_accuracy':balanced_accuracy_score(y,np.array(p)>.5),'mdd_recall':recall_score(y,np.array(p)>.5,zero_division=0),
        'nll':log_loss(y,np.c_[1-np.array(p),p],labels=[0,1]),'brier':brier_score_loss(y,p)}
    assert all(abs(result[k]-v)<1e-12 for k,v in independent.items())
    return result

def summarize(rows,seeds):
    result={}
    for split in ['train','val','test']:
        parts=[r for r in rows if r['split']==split];per_seed={}
        for seed in seeds:
            selected=[r for r in parts if r['seed']==seed]
            byfold={str(f):metrics([r for r in selected if r['fold']==f]) for f in FOLDS}
            mean={k:float(np.mean([v[k] for v in byfold.values()])) if all(v[k] is not None for v in byfold.values()) else None for k in METRICS}
            per_seed[str(seed)]={'folds':byfold,'mean':mean,'pooled':metrics(selected)}
            if split=='test':assert len(selected)==37 and len({r['subject_id'] for r in selected})==37
        result[split]={'seeds':per_seed,'mean':{k:float(np.mean([v['mean'][k] for v in per_seed.values()]))
             if all(v['mean'][k] is not None for v in per_seed.values()) else None for k in METRICS}}
    return result

def rows_for(data,condition,seed,fold,p,extra=None):
    splits=data.ctx.folds[f'fold_{fold}'];rows=[]
    for i,sid in enumerate(data.ids):
        row={'condition':condition,'seed':seed,'fold':fold,'subject_id':sid,'label':data.ctx.labels[sid],
            'split':next(s for s in ['train','val','test'] if sid in splits[s]),'p':float(p[i]),'prediction':int(p[i]>.5)}
        if extra:row.update(extra(i))
        rows.append(row)
    return rows

def main():
    lock_check();gate=read(OUT/'release_gate.json');assert gate['passed']
    for name,sha in gate['fitted_artifacts'].items():assert digest(OUT/name)==sha,name
    assert not (OUT/'evaluation_complete.json').exists()
    data=load_data();set_seed(11);all_rows=[];result={};reloads=0
    for condition in CONDITIONS:
        collected=[]
        for seed in SEEDS:
            for fold in FOLDS:
                stem=run_dir(condition,seed,fold);cp=torch.load(stem/'final.pt',weights_only=True,map_location='cpu')
                assert cp['fit_ids']==data.ctx.folds[f'fold_{fold}']['train']
                m=Model(condition,seed).cuda();m.load_state_dict(cp['model']);initial=state_hash(m.state_dict())
                logits,z,n=infer(m,data);ep,e,b=outputs(logits,'CCML',m.modalities)
                re=Model(condition,seed).cuda();re.load_state_dict(torch.load(stem/'final.pt',weights_only=True,map_location='cpu')['model'])
                ll,zz,nn=infer(re,data)
                assert all(torch.equal(a,b) for a,b in [(ll,logits),(zz,z),(nn,n)])
                assert initial==state_hash(m.state_dict())==cp['final_state_sha256'];reloads+=1
                extra=lambda i:{'probabilities':{k:float(v[i]) for k,v in ep.items()},'logits':logits[i].tolist(),
                    'evidence':e[i].tolist(),'branch_u':b['u'][i].tolist()}
                rows=rows_for(data,condition,seed,fold,ep['formal'].numpy(),extra)
                target=OUT/f'predictions/{condition}/seed_{seed}/fold_{fold}.json'
                save(target,{'checkpoint':desc(stem/'final.pt'),'modalities':m.modalities,'rows':rows},immutable=True)
                collected+=rows;del m,re,cp
        result[condition]=summarize(collected,SEEDS);all_rows+=collected
        print('EVALUATED',condition,flush=True)
    covs=np.load(OUT/'linear/covariances.npy');bp=np.load(E2/'linear/new/features.npy')
    for condition in ['M1-bandpower','M1-covariance']:
        collected=[]
        for fold in FOLDS:
            stem=OUT/f'linear/{condition}/fold_{fold}';cp=joblib.load(stem/'model.joblib')
            ids=data.ctx.folds[f'fold_{fold}']['train'];assert cp['fit_ids']==ids
            if condition=='M1-covariance':
                expected=matrix_function(covs[[data.index[s] for s in ids]],np.log).mean(0)
                assert np.array_equal(expected,cp['reference_log']);x=tangent(covs,cp['reference_log'])
            else:x=bp
            assert np.allclose(cp['model'][0].mean_,x[[data.index[s] for s in ids]].mean(0),atol=1e-12,rtol=1e-12)
            p=cp['model'].predict_proba(x)[:,1]
            re=joblib.load(stem/'model.joblib');assert np.array_equal(p,re['model'].predict_proba(x)[:,1]);reloads+=1
            rows=rows_for(data,condition,None,fold,p)
            save(OUT/f'predictions/{condition}/fold_{fold}.json',{'checkpoint':desc(stem/'model.joblib'),'rows':rows},immutable=True)
            collected+=rows
        result[condition]=summarize(collected,[None]);all_rows+=collected
    save(OUT/'metrics.json',result,immutable=True)
    save(OUT/'all_predictions.json',all_rows,immutable=True)
    save(OUT/'evaluation_complete.json',{'completed_utc':now(),'reloads_exact':reloads,'rows':len(all_rows),
        'independent_subjects':37,'independent_metric_checks':100*3,'all_results_released_together':True,
        'reference_and_scaler_train_scope_verified':True},immutable=True)
    save(OUT/'status.json',{'stage':'evaluated','completed_neural':90,'completed_linear':10,'completed_updates':21600,
        'outer_results_released':True,'updated_utc':now()})
    print('Evaluation and independent metric verification complete.',flush=True)

if __name__=='__main__':main()
