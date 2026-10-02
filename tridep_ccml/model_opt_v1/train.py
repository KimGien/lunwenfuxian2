import time
import warnings
import joblib
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from .core import *
from tridep_ccml.fusion_g1.train import probe

def linear_train(data):
    path=OUT/'linear';path.mkdir(exist_ok=True)
    covpath=path/'covariances.npy'
    if not covpath.exists():
        byid={r['subject_id']:r['new_cache'] for r in read(E2/'input_manifest.json')['records']}
        covs=[];shrinks=[]
        for sid in data.ids:
            x=np.load(WORK/byid[sid]['path']);window_cov=[];sh=[]
            for w in x:
                lw=LedoitWolf(store_precision=False,assume_centered=False).fit(w.T.astype(np.float64))
                window_cov.append(lw.covariance_);sh.append(float(lw.shrinkage_))
            covs.append(np.mean(window_cov,0));shrinks.append(sh)
        np.save(covpath,np.stack(covs));save(path/'shrinkage.json',{'subject_ids':data.ids,'values':shrinks},immutable=True)
    covs=np.load(covpath);bp=np.load(E2/'linear/new/features.npy')
    for fold in FOLDS:
        ids=data.ctx.folds[f'fold_{fold}']['train'];ix=[data.index[s] for s in ids]
        ref=matrix_function(covs[ix],np.log).mean(0)
        for condition in ['M1-bandpower','M1-covariance']:
            stem=path/condition/f'fold_{fold}';done=stem/'complete.json'
            if done.exists():
                for d in read(done)['artifacts']:assert digest(WORK/d['path'])==d['sha256']
                continue
            assert not stem.exists();stem.mkdir(parents=True)
            x=bp if condition=='M1-bandpower' else tangent(covs,ref)
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,penalty='l2',solver='lbfgs',class_weight=None,max_iter=2000,tol=1e-8))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(x[ix],[data.ctx.labels[s] for s in ids])
            joblib.dump({'model':model,'reference_log':None if condition=='M1-bandpower' else ref,'fit_ids':ids,'condition':condition},stem/'model.joblib')
            re=joblib.load(stem/'model.joblib')
            assert np.array_equal(model.predict_proba(x[ix]),re['model'].predict_proba(x[ix]))
            save(done,{'condition':condition,'fold':fold,'fit_ids':ids,'n_features':x.shape[1],
                'reload_exact':True,'iterations':model[-1].n_iter_.tolist(),'artifacts':[desc(stem/'model.joblib')],
                'completed_utc':now(),'heldout_inference':False},immutable=True)
            print('LINEAR',condition,fold,flush=True)

def neural_train(data):
    done=0
    for seed in SEEDS:
        for fold in FOLDS:
            ids=data.ctx.folds[f'fold_{fold}']['train'];batch_orders=orders(ids,seed)
            for condition in CONDITIONS:
                stem=run_dir(condition,seed,fold)
                if (stem/'complete.json').exists():
                    for d in read(stem/'complete.json')['artifacts']:assert digest(WORK/d['path'])==d['sha256']
                    done+=1;continue
                assert not stem.exists(),f'Inspect incomplete run before resuming: {stem}'
                stem.mkdir(parents=True);start_time=time.monotonic()
                model=Model(condition,seed).cuda();init_hash=state_hash(model.state_dict());set_seed(seed)
                optimizer=model.optimizer();before_probe=probe(model,data,ids[:8]);gg=groups(model)
                torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()},stem/'rng_initial.pt')
                records=[];history=[];updates=0
                for epoch,order in enumerate(batch_orders,1):
                    model.train();losses=[]
                    for start in range(0,len(order),8):
                        batch=order[start:start+8];assert set(batch)<=set(ids)
                        x,y=data.batch(batch);optimizer.zero_grad(set_to_none=True)
                        loss=model.objective(model(x),y,epoch);assert torch.isfinite(loss);loss.backward()
                        snapshots={g:[p.detach().clone() for p in ps] for g,ps in gg.items()}
                        pre={g:l2([p.grad for p in ps]) for g,ps in gg.items()}
                        pn={g:l2(ps) for g,ps in gg.items()}
                        post_total=clip(model);post={g:l2([p.grad for p in ps]) for g,ps in gg.items()}
                        optimizer.step()
                        opt_update={g:l2([p-s for p,s in zip(ps,snapshots[g])]) for g,ps in gg.items()}
                        before_constraint={g:[p.detach().clone() for p in ps] for g,ps in gg.items()}
                        model.constrain()
                        constraint={g:l2([p-s for p,s in zip(ps,before_constraint[g])]) for g,ps in gg.items()}
                        net={g:l2([p-s for p,s in zip(ps,snapshots[g])]) for g,ps in gg.items()}
                        records.append({'update':updates+1,'epoch':epoch,'global_pre':math.sqrt(sum(v*v for v in pre.values())),
                            'global_post':post_total,'modules':{g:{'grad_pre':pre[g],'grad_post':post[g],
                            'parameter_l2':pn[g],'optimizer_update_l2':opt_update[g],
                            'relative_update':opt_update[g]/max(pn[g],1e-30),'constraint_l2':constraint[g],
                            'net_update_l2':net[g]} for g in gg}})
                        losses.append((float(loss.detach()),len(batch)));updates+=1
                    history.append({'epoch':epoch,'loss':sum(v*n for v,n in losses)/len(ids),'updates':len(losses)})
                assert updates==240
                after_probe=probe(model,data,ids[:8]);final={k:v.detach().cpu() for k,v in model.state_dict().items()}
                final_hash=state_hash(final);historical_exact=None
                if '-global-' in condition:
                    old=read(G1/f'runs/G-CCML-{condition.rsplit("-",1)[1]}/seed_{seed}/fold_{fold}/complete.json')
                    assert init_hash==old['initial_state_sha256']
                    historical_exact=final_hash==old['final_state_sha256']
                    assert historical_exact,'Global control no longer reproduces G1'
                torch.save({'model':final,'optimizer':optimizer.state_dict(),'condition':condition,'seed':seed,'fold':fold,
                    'fit_ids':ids,'initial_state_sha256':init_hash,'final_state_sha256':final_hash,
                    'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(),
                    'registration':desc(OUT/'registration.json')},stem/'final.pt')
                replay=Model(condition,seed).cuda();replay.load_state_dict(torch.load(stem/'final.pt',weights_only=True,map_location='cpu')['model'])
                re_probe=probe(replay,data,ids[:8])
                assert all(torch.equal(after_probe[k],re_probe[k]) for k in ['logits','features','normalized'])
                torch.save({'before':before_probe,'after':after_probe},stem/'train_probe.pt')
                save(stem/'history.json',history,immutable=True);save(stem/'updates.json',records,immutable=True)
                save(stem/'complete.json',{'condition':condition,'seed':seed,'fold':fold,'fit_ids':ids,'updates':updates,
                    'initial_state_sha256':init_hash,'final_state_sha256':final_hash,'reload_exact':True,
                    'historical_G1_exact':historical_exact,'parameter_count':sum(p.numel() for p in model.parameters()),
                    'heldout_inference':False,'elapsed_seconds':time.monotonic()-start_time,'completed_utc':now(),
                    'artifacts':[desc(stem/n) for n in ['final.pt','history.json','updates.json','train_probe.pt','rng_initial.pt']]},immutable=True)
                done+=1
                save(OUT/'status.json',{'stage':'training','completed_neural':done,'completed_linear':10,'completed_updates':done*240,
                    'outer_results_released':False,'last':f'{condition}/seed{seed}/fold{fold}','updated_utc':now()})
                print(f'{done}/90 {condition} seed={seed} fold={fold} {time.monotonic()-start_time:.1f}s',flush=True)
                del model,replay,optimizer,snapshots,before_constraint,final,records

def main():
    lock_check();assert read(OUT/'tests.json')['passed'];data=load_data()
    linear_train(data);neural_train(data);lock_check()
    assert len(list((OUT/'runs').rglob('complete.json')))==90
    assert len(list((OUT/'linear').rglob('complete.json')))==10
    files={p.relative_to(OUT).as_posix():digest(p) for root in ['runs','linear'] for p in sorted((OUT/root).rglob('*')) if p.is_file()}
    save(OUT/'release_gate.json',{'passed':True,'neural_fits':90,'linear_fits':10,'updates':21600,
        'locked_utc':now(),'fitted_artifacts':files,'outer_results_released':False},immutable=True)
    save(OUT/'status.json',{'stage':'all_fits_locked','completed_neural':90,'completed_linear':10,'completed_updates':21600,
        'outer_results_released':False,'updated_utc':now()})
    print('All 100 fits locked; ready for evaluation.',flush=True)

if __name__=='__main__':main()
