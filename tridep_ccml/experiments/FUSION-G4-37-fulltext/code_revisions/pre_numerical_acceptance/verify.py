from collections import defaultdict
from .common import *
from .train import predict
from tridep_ccml.fusion_g1.evaluate import infer
from tridep_ccml.fusion_g1.verify import numpy_ccml

def manual_metrics(rows):
    y=np.array([r['label'] for r in rows]);p=np.array([r['p'] for r in rows],dtype=np.float64)
    pred=p>np.array([r['threshold'] for r in rows])
    tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
    f1=((2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0)+(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))/2
    auc=((p[y==1,None]>p[None,y==0]).sum()+.5*(p[y==1,None]==p[None,y==0]).sum())/((y==1).sum()*(y==0).sum())
    return {'macro_f1':f1,'balanced_accuracy':(tn/(tn+fp)+tp/(tp+fn))/2,'mdd_recall':tp/(tp+fn),
        'auc':float(auc),'nll':float(-np.log(np.clip(np.where(y,p,1-p),1e-15,1)).mean()),'brier':float(((p-y)**2).mean())}

def main():
    cfg=config_check()
    for d in read(OUT/'postprocess_lock.json')['files']:verify(d)
    for name in ['inner_models_lock.json','postprocessors_lock.json','evaluation_complete.json']:
        for p,sha in read(OUT/name)['artifacts'].items():assert digest(OUT/p)==sha,p
    data=Data();set_seed(11);inner_count=oof_count=outer_count=cal_count=metric_count=0;groups=defaultdict(list);numerical=[]
    for c in CONDITIONS:
        for seed in SEEDS:
            for fold in FOLDS:
                ids=cfg['outer_folds'][f'fold_{fold}']['train'];oof=[]
                for part in cfg['inner_splits'][str(fold)]:
                    stem=folder(c,seed,fold,part['inner']);cp=torch.load(stem/'final.pt',weights_only=True,map_location='cpu')
                    assert cp['fit_ids']==part['fit'] and cp['calibration_ids']==part['calibration']
                    assert cp['updates']==180 and cp['epochs']==60
                    assert cp['registration']==desc(OUT/'registration.json')
                    model=Model(c,seed).cuda();assert state_hash(model.state_dict())==cp['initial_state_sha256']
                    model.load_state_dict(cp['model']);assert state_hash(model.state_dict())==cp['final_state_sha256']
                    opt=cp['optimizer']['param_groups'];assert [g['lr'] for g in opt]==[1e-4,1e-3] and all(g['weight_decay']==0 for g in opt)
                    if 'eeg' in model.modalities:
                        assert all(int(model.encoders['eeg'].__getattr__(b).num_batches_tracked)==180 for b in ['bn1','bn2','bn3'])
                        assert float(torch.linalg.vector_norm(model.encoders['eeg'].spatial.weight.detach(),dim=(1,2,3)).max())<=1.000001
                    saved=torch.load(stem/'probes.pt',weights_only=True)
                    for key,pids in [('after',part['fit'][:8]),('oof',part['calibration'])]:
                        actual=predict(model,data,pids)
                        for k in ['logits','features','p']:assert torch.equal(actual[k],saved[key][k]),(c,seed,fold,part['inner'],k)
                    rows=read(stem/'oof.json');assert [r['subject_id'] for r in rows]==part['calibration']
                    assert np.array_equal([r['p'] for r in rows],saved['oof']['p'].double().numpy())
                    e=np.logaddexp(0,saved['oof']['logits'].double().numpy());expected=numpy_ccml(e)[:,1]
                    assert np.max(np.abs(expected-np.array([r['p'] for r in rows])))<3e-6
                    for r in rows:
                        assert r['label']==data.ctx.labels[r['subject_id']] and r['fit_ids']==part['fit']
                        assert r['subject_id'] not in r['fit_ids'] and set(r['fit_ids'])<set(ids)
                    oof+=rows;oof_count+=len(rows);inner_count+=1
                cal=read(OUT/f'calibrators/{c}/seed_{seed}/fold_{fold}.json')
                assert cal['oof']==oof and cal['train_ids']==ids and not cal['outer_val_test_used'];validate_oof(oof,ids)
                threshold=cal['parameters']['threshold'];sig=cal['parameters']['sigmoid']
                # Independent enumeration and confusion-matrix F1, including deterministic ties.
                u=sorted(set(r['p'] for r in oof));candidates=sorted(set([0.,.5,1.]+[(a+b)/2 for a,b in zip(u,u[1:])]))
                scores=[manual_metrics([{'p':r['p'],'label':r['label'],'threshold':t} for r in oof])['macro_f1'] for t in candidates]
                best=max(scores);chosen=min([t for t,s in zip(candidates,scores) if abs(s-best)<1e-12],key=lambda t:(abs(t-.5),t))
                assert threshold['threshold']==chosen and threshold['candidate_count']==len(candidates)
                assert abs(threshold['oof_fit_macro_f1']-best)<1e-12
                refit=fit_sigmoid(oof,ids)
                for k in ['a','b','objective']:assert abs(refit[k]-sig[k])<1e-12
                pp=np.clip([r['p'] for r in oof],1e-7,1-1e-7);y=np.array([r['label'] for r in oof])
                x=np.log(pp)-np.log1p(-pp);z=sig['a']*x+sig['b'];q=1/(1+np.exp(-z))
                objective=np.mean(np.logaddexp(0,z)-y*z)+.005*((sig['a']-1)**2+sig['b']**2)
                assert abs(objective-sig['objective'])<1e-10
                gradient=[float(((q-y)*x).mean()+.01*(sig['a']-1)),float((q-y).mean()+.01*sig['b'])]
                residuals=[]
                for value,g,lo,hi in zip([sig['a'],sig['b']],gradient,[.01,-10.],[10.,10.]):
                    assert lo<=value<=hi
                    residual=min(g,0) if value<=lo+1e-8 else max(g,0) if value>=hi-1e-8 else g
                    residuals.append(abs(residual));assert abs(residual)<1e-5,'Sigmoid KKT condition failed'
                cal_count+=1
                # Reload the actual frozen outer model, using G1's exact evaluation batches.
                parent=read(G1/f'predictions/{c}/seed_{seed}/fold_{fold}.json')
                cp=torch.load(verify(parent['checkpoint']),weights_only=True,map_location='cpu')
                model=Model(c,seed).cuda();model.load_state_dict(cp['model']);l,z,n=infer(model,data)
                oldfeature=torch.load(verify(parent['feature_file']),weights_only=True)
                for a,k in [(l,'logits'),(z,'features'),(n,'normalized_features')]:assert torch.equal(a,oldfeature[k]),'G1 parent reload differs'
                outer_count+=1
                new=read(OUT/f'predictions/{c}/seed_{seed}/fold_{fold}.json');assert len(new['rows'])==37
                raw_values=[];sig_values=[]
                for old,r in zip(parent['rows'],new['rows']):
                    assert all(r[k]==old[k] for k in ['subject_id','label','split'])
                    raw=old['probabilities']['formal'];v=r['variants'];assert v['raw']=={'p':raw,'threshold':.5}
                    assert v['threshold']=={'p':raw,'threshold':chosen}
                    clipped=min(max(raw,1e-7),1-1e-7)
                    expected=1/(1+np.exp(-(sig['a']*(np.log(clipped)-np.log1p(-clipped))+sig['b'])))
                    assert abs(expected-v['sigmoid']['p'])<1e-12 and v['sigmoid']['threshold']==.5
                    raw_values.append(raw);sig_values.append(v['sigmoid']['p'])
                    for variant,vv in v.items():groups[(c,variant,r['split'],seed,fold)].append({'label':r['label'],**vv})
                a=np.array(raw_values);b=np.array(sig_values);order=np.argsort(a)
                assert np.all(np.diff(b[order])>=0)
                numerical.append({'condition':c,'seed':seed,'fold':fold,'max_projected_gradient':max(residuals),
                    'raw_probabilities_clipped':int(((a<1e-7)|(a>1-1e-7)).sum()),
                    'sigmoid_saturated_outputs':int(((b==0)|(b==1)).sum()),
                    'introduced_adjacent_rank_ties':int(((np.diff(a[order])>0)&(np.diff(b[order])==0)).sum())})
        print(f'{c}: verified inner {inner_count}/90, calibrators {cal_count}/30, outer replays {outer_count}/30',flush=True)
    summary=read(REPORT/'results.json');baseline=read(WORK/'output/analysis/fusion_g1_20260929/results.json')
    lookup={(r['condition'],r['variant'],r['split']):r for r in summary['results']}
    for row in summary['results']:
        for fm in row['per_fold']:
            actual=manual_metrics(groups[(row['condition'],row['variant'],row['split'],fm['seed'],fm['fold'])])
            for k in METRICS:assert abs(actual[k]-fm[k])<1e-12,(row['condition'],row['variant'],k)
            metric_count+=1
        for sm in row['per_seed']:
            for k in METRICS:assert abs(sm[k]-np.mean([f[k] for f in row['per_fold'] if f['seed']==sm['seed']]))<1e-12
        for k in METRICS:assert abs(row['mean'][k]-np.mean([s[k] for s in row['per_seed']]))<1e-12
        if row['variant']=='raw':
            old=next(r for r in baseline['results'] if r['condition']==row['condition'] and r['endpoint']=='formal' and r['split']==row['split'])
            assert row['mean']==old['mean'] and row['per_fold']==old['per_fold']
    for c in CONDITIONS:
        for split in ['train','val','test']:
            raw=lookup[(c,'raw',split)];th=lookup[(c,'threshold',split)];sig=lookup[(c,'sigmoid',split)]
            for r,t,s in zip(raw['per_fold'],th['per_fold'],sig['per_fold']):
                for k in ['auc','nll','brier']:assert t[k]==r[k]
                if not any(n['introduced_adjacent_rank_ties'] for n in numerical if n['condition']==c and n['seed']==r['seed'] and n['fold']==r['fold']):
                    assert s['auc']==r['auc']
    hist=history();assert hist==read(OUT/'historical_before.json')
    assert (inner_count,outer_count,cal_count,metric_count)==(90,30,30,270)
    save(REPORT/'verification.json',{'passed':True,'verified_utc':now(),'new_inner_models_reloaded':inner_count,
        'inner_OOF_predictions_exact':oof_count,'frozen_outer_models_reloaded':outer_count,'threshold_and_sigmoid_sets_refitted':cal_count,
        'independent_fold_metric_vectors':metric_count,'parent_raw_metrics_exact':True,'threshold_probability_metrics_unchanged':True,
        'inner_training_identity_and_budget_verified':True,'numerical_diagnostics':numerical,
        'historical_entries_checked':sum(v['files'] for v in hist.values()),'history':hist},immutable=True)
    print(f'G4 verification passed: {inner_count} new + {outer_count} parent reloads; {oof_count} OOF rows; {metric_count} metric vectors.',flush=True)

if __name__=='__main__':main()
