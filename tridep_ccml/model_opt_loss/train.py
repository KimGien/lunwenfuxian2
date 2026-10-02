import time
from .core import *
from tridep_ccml.fusion_g1.train import probe

def baseline_probes(data):
    for seed in SEEDS:
        for fold in FOLDS:
            ids=data.ctx.folds[f'fold_{fold}']['train'][:8]
            for mods in ['ST','EST']:
                path=OUT/f'mechanism/{mods}/seed_{seed}/fold_{fold}/baseline.json'
                if path.exists():continue
                m=Model('M2-sum-'+mods,seed).cuda()
                initial=block_probe(m,data,ids,1)
                cp=torch.load(stem('M2-sum-'+mods,seed,fold)/'final.pt',weights_only=True,map_location='cpu')
                m.load_state_dict(cp['model']);final=block_probe(m,data,ids,60)
                save(path,{'seed':seed,'fold':fold,'modalities':mods,'initial':initial,'sum_final':final},immutable=True)
                del m,cp
    save(OUT/'baseline_probe_complete.json',{'probes':60,'optimizer_updates':0,'outer_subjects_used':False,'completed_utc':now()},immutable=True)
    print('60 training-only baseline probes completed; zero optimizer updates.',flush=True)

def main():
    lock_check();data=Data();set_seed(11);baseline_probes(data);done=0
    for seed in SEEDS:
        for fold in FOLDS:
            ids=data.ctx.folds[f'fold_{fold}']['train'];orderings=orders(ids,seed)
            for condition in NEW:
                path=stem(condition,seed,fold)
                if (path/'complete.json').exists():
                    for d in read(path/'complete.json')['artifacts']:assert digest(WORK/d['path'])==d['sha256']
                    done+=1;continue
                assert not path.exists(),f'Inspect incomplete run: {path}'
                path.mkdir(parents=True);started=time.monotonic();mods=condition.rsplit('-',1)[1]
                control=read(stem('M2-sum-'+mods,seed,fold)/'complete.json')
                m=Model(condition,seed).cuda();ih=state_hash(m.state_dict());assert ih==control['initial_state_sha256']
                set_seed(seed);opt=m.optimizer();before=probe(m,data,ids[:8]);gg=groups(m)
                rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
                crng=torch.load(stem('M2-sum-'+mods,seed,fold)/'rng_initial.pt',weights_only=True,map_location='cpu')
                assert all(torch.equal(rng[k].cpu(),crng[k]) for k in rng)
                torch.save(rng,path/'rng_initial.pt');hist=[];steps=[];updates=0
                for epoch,order in enumerate(orderings,1):
                    m.train();losses=[]
                    for offset in range(0,len(order),8):
                        batch=order[offset:offset+8];assert set(batch)<=set(ids)
                        x,y=data.batch(batch);opt.zero_grad(set_to_none=True)
                        loss=m.objective(m(x),y,epoch);loss.backward()
                        snapshot={g:[p.detach().clone() for p in ps] for g,ps in gg.items()}
                        pre={g:l2([p.grad for p in ps]) for g,ps in gg.items()};pn={g:l2(ps) for g,ps in gg.items()}
                        norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True))
                        post={g:l2([p.grad for p in ps]) for g,ps in gg.items()};opt.step()
                        update={g:l2([p-s for p,s in zip(ps,snapshot[g])]) for g,ps in gg.items()}
                        bc={g:[p.detach().clone() for p in ps] for g,ps in gg.items()};m.constrain()
                        constraint={g:l2([p-s for p,s in zip(ps,bc[g])]) for g,ps in gg.items()}
                        net={g:l2([p-s for p,s in zip(ps,snapshot[g])]) for g,ps in gg.items()}
                        steps.append({'update':updates+1,'epoch':epoch,'global_grad_pre':norm,'modules':{g:{'grad_pre':pre[g],
                            'grad_post':post[g],'parameter_l2':pn[g],'optimizer_update_l2':update[g],
                            'relative_update':update[g]/max(pn[g],1e-30),'constraint_l2':constraint[g],'net_update_l2':net[g]} for g in gg}})
                        losses.append((float(loss.detach()),len(batch)));updates+=1
                    hist.append({'epoch':epoch,'loss':sum(v*n for v,n in losses)/len(ids),'updates':len(losses)})
                assert updates==240
                after=probe(m,data,ids[:8]);final_probe=block_probe(m,data,ids[:8],60)
                final={k:v.detach().cpu() for k,v in m.state_dict().items()};fh=state_hash(final)
                torch.save({'model':final,'optimizer':opt.state_dict(),'condition':condition,'seed':seed,'fold':fold,
                    'fit_ids':ids,'initial_state_sha256':ih,'final_state_sha256':fh,'epochs':60,'updates':updates,
                    'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(),'registration':desc(OUT/'registration.json')},path/'final.pt')
                re=Model(condition,seed).cuda();re.load_state_dict(torch.load(path/'final.pt',weights_only=True,map_location='cpu')['model'])
                again=probe(re,data,ids[:8]);assert all(torch.equal(after[k],again[k]) for k in ['logits','features','normalized'])
                torch.save({'before':before,'after':after},path/'train_probe.pt')
                save(path/'mechanism_final.json',final_probe,immutable=True);save(path/'history.json',hist,immutable=True)
                save(path/'updates.json',steps,immutable=True)
                save(path/'complete.json',{'condition':condition,'seed':seed,'fold':fold,'fit_ids':ids,'updates':updates,
                    'initial_state_sha256':ih,'final_state_sha256':fh,'reload_exact':True,'paired_rng_exact':True,
                    'control':desc(stem('M2-sum-'+mods,seed,fold)/'complete.json'),'parameter_count':sum(p.numel() for p in m.parameters()),
                    'heldout_inference':False,'elapsed_seconds':time.monotonic()-started,'completed_utc':now(),
                    'artifacts':[desc(path/f) for f in ['final.pt','rng_initial.pt','train_probe.pt','history.json','updates.json','mechanism_final.json']]},immutable=True)
                done+=1
                save(OUT/'status.json',{'stage':'training','new_models_completed':done,'new_updates_completed':done*240,
                    'reused_controls':30,'last':f'{condition}/seed{seed}/fold{fold}','new_outer_results_released':False,'updated_utc':now()})
                print(f'{done}/30 {condition} seed={seed} fold={fold} {time.monotonic()-started:.1f}s',flush=True)
                del m,re,opt,final,steps,snapshot,bc
    lock_check();assert len(list((OUT/'runs').rglob('complete.json')))==30
    save(OUT/'release_gate.json',{'passed':True,'new_models':30,'new_updates':7200,'reused_controls':30,
        'mechanism_probes':90,'probe_optimizer_updates':0,'created_utc':now(),
        'fitted_artifacts':{p.relative_to(OUT).as_posix():digest(p) for root in ['runs','mechanism'] for p in sorted((OUT/root).rglob('*')) if p.is_file()}},immutable=True)
    save(OUT/'status.json',{'stage':'fits_locked','new_models_completed':30,'new_updates_completed':7200,
        'reused_controls':30,'new_outer_results_released':False,'updated_utc':now()})
    print('30 new fits locked; ready for four-condition evaluation.',flush=True)

if __name__=='__main__':main()
