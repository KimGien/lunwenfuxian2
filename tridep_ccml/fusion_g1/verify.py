from collections import defaultdict
from .common import *
from .model import Model
from .evaluate import infer

def manual_metric(rows):
    y=np.array([r['label'] for r in rows]);p=np.array([r['p'] for r in rows],dtype=np.float64);pred=p>.5
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    auc=((p[y==1,None]>p[None,y==0]).sum()+.5*(p[y==1,None]==p[None,y==0]).sum())/((y==1).sum()*(y==0).sum())
    return {'macro_f1':((2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0)+(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))/2,
        'balanced_accuracy':(tn/(tn+fp)+tp/(tp+fn))/2,'mdd_recall':tp/(tp+fn),'auc':float(auc),
        'nll':float(np.mean(-y*np.log(np.clip(p,1e-15,1))-(1-y)*np.log(np.clip(1-p,1e-15,1)))),
        'brier':float(((p-y)**2).mean())}

def numpy_ccml(e):
    common=e.min(axis=1);powers=common**1.25
    share=np.divide(powers,powers.sum(-1,keepdims=True),out=np.zeros_like(powers),where=powers.sum(-1,keepdims=True)>0)
    con=e.shape[1]*common.sum(-1,keepdims=True)*share
    final=con+(e-common[:,None]).mean(axis=1)
    return (final+1)/(final+1).sum(-1,keepdims=True)

def main():
    lock_check();gate=read(OUT/'release_gate.json');ev=read(OUT/'evaluation_complete.json')
    for d in read(OUT/'evaluation_lock.json')['files']:verify(d)
    for manifest in [gate['fitted_artifacts'],ev['artifacts']]:
        for p,sha in manifest.items():assert digest(OUT/p)==sha,p
    data=Data();set_seed(11);summary=read(REPORT/'results.json');groups=defaultdict(list)
    models=rows_verified=0;pair_initial={};feature_health=[]
    for path in sorted((OUT/'predictions').rglob('fold_*.json')):
        obj=read(path);c=obj['condition'];seed=obj['seed'];fold=obj['fold']
        cp=torch.load(verify(obj['checkpoint']),weights_only=True,map_location='cpu')
        model=Model(c,seed).cuda();model.load_state_dict(cp['model'])
        init=Model(c,seed).state_dict();assert state_hash(init)==cp['initial_state_sha256']
        assert cp['train_ids']==data.ctx.folds[f'fold_{fold}']['train'] and cp['epochs']==60 and cp['updates']==240
        assert state_hash(cp['model'])==cp['final_state_sha256']
        assert all(torch.isfinite(v).all() for v in cp['model'].values())
        if 'eeg' in model.modalities:
            weight=model.encoders['eeg'].spatial.weight.detach()
            assert float(torch.linalg.vector_norm(weight,dim=(1,2,3)).max())<=1.000001
        logits,z,n=infer(model,data)
        saved=torch.load(verify(obj['feature_file']),weights_only=True)
        assert saved['subject_ids']==data.ids and saved['modalities']==model.modalities
        for a,k in [(logits,'logits'),(z,'features'),(n,'normalized_features')]:assert torch.equal(a,saved[k]),(c,seed,fold,k)
        # Independent numpy recomputation, using recorded logits/evidence and actual V.
        l=np.array([r['logits'] for r in obj['rows']],dtype=np.float64)
        assert np.array_equal(l,logits.double().numpy())
        if obj['family']=='CE':
            ex=np.exp(l-l.max(-1,keepdims=True)); bp=ex/ex.sum(-1,keepdims=True)
            assert all(r['evidence'] is None and r['branch_u'] is None and r['branch_strength'] is None for r in obj['rows'])
        else:
            e=np.array([r['evidence'] for r in obj['rows']],dtype=np.float64)
            assert np.allclose(e,np.logaddexp(0,l),rtol=2e-6,atol=2e-6)
            strength=(e+1).sum(-1);bp=(e+1)/strength[...,None]
            assert np.allclose(strength,np.array([r['branch_strength'] for r in obj['rows']]),atol=2e-5,rtol=2e-6)
            assert np.allclose(2/strength,np.array([r['branch_u'] for r in obj['rows']]),atol=2e-6)
        eps={m:bp[:,i,1] for i,m in enumerate(model.modalities)};eps['probability_mean']=bp.mean(1)[:,1]
        if obj['family']!='CE':
            eps['ccml']=numpy_ccml(e)[:,1]
            if obj['family']=='EDL':
                avg=e.mean(1)+1;eps['evidence_mean']=(avg/avg.sum(-1,keepdims=True))[:,1]
        eps['formal']=eps['ccml' if obj['family']=='CCML' else 'probability_mean']
        if 'eeg' in model.modalities:
            eps['fixed_ST']=(numpy_ccml(e[:,1:])[:,1] if obj['family']=='CCML' else bp[:,1:].mean(1)[:,1])
        for i,row in enumerate(obj['rows']):
            assert row['subject_id']==data.ids[i] and row['label']==data.ctx.labels[data.ids[i]]
            assert row['subject_id'] in data.ctx.folds[f'fold_{fold}'][row['split']]
            assert set(eps)==set(row['probabilities'])
            for endpoint,p in row['probabilities'].items():
                assert abs(p-eps[endpoint][i])<3e-6 and 0<=p<=1,(c,seed,fold,endpoint)
                groups[(c,endpoint,row['split'],seed,fold)].append({'p':p,'label':row['label']})
            rows_verified+=1
        feature_health.append({'condition':c,'seed':seed,'fold':fold,
            'across_person_feature_std_norm':{m:float(z[:,i].std(0).norm()) for i,m in enumerate(model.modalities)},
            'minimum_probability_distance_to_threshold':min(abs(r['probabilities']['formal']-.5) for r in obj['rows'])})
        assert all(v>0 for v in feature_health[-1]['across_person_feature_std_norm'].values())
        models+=1
        if models%15==0:print(f'Exact reload + independent inference verified {models}/90',flush=True)
    checked_metrics=0
    for row in summary['results']:
        for fm in row['per_fold']:
            actual=manual_metric(groups[(row['condition'],row['endpoint'],row['split'],fm['seed'],fm['fold'])])
            for k in METRICS:assert abs(actual[k]-fm[k])<1e-12,(row['condition'],row['endpoint'],k)
            checked_metrics+=1
        for sm in row['per_seed']:
            for k in METRICS:assert abs(sm[k]-np.mean([f[k] for f in row['per_fold'] if f['seed']==sm['seed']]))<1e-12
        for k in METRICS:assert abs(row['mean'][k]-np.mean([s[k] for s in row['per_seed']]))<1e-12
    for seed in SEEDS:
        for fold in FOLDS:
            ids=data.ctx.folds[f'fold_{fold}']['train'];orders_=read(OUT/f'orders/seed_{seed}/fold_{fold}.json')
            assert len(orders_)==60 and all(sorted(r)==sorted(ids) for r in orders_)
    hist=history();assert hist==read(OUT/'historical_before.json')
    save(REPORT/'verification.json',{'passed':True,'verified_utc':now(),'exact_reloaded_models':models,
        'exact_subject_model_outputs':rows_verified,'independent_fold_metric_vectors':checked_metrics,
        'independent_fusion_numpy_tolerance':3e-6,'independent_metric_tolerance':1e-12,
        'CE_no_fabricated_evidence':True,'actual_view_counts_verified':True,'training_identity_and_budget_verified':True,
        'historical_hashes_unchanged':hist,'historical_entries_checked':sum(v['files'] for v in hist.values()),
        'feature_health':feature_health},immutable=True)
    print(f'Verified {models} models, {rows_verified} subject outputs, {checked_metrics} metric vectors, history unchanged.',flush=True)

if __name__=='__main__':main()
