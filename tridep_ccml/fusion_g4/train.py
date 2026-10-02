import time
from torch.nn import functional as F
from tridep_ccml.ccml import evidence_dc
from .common import *

@torch.no_grad()
def predict(model,data,ids):
    model.eval();before=state_hash(model.state_dict());ls=[];zs=[]
    for start in range(0,len(ids),8):
        x,_=data.batch(ids[start:start+8]);l,z,_=model(x,features=True)
        ls.append(l.cpu());zs.append(z.cpu())
    logits=torch.cat(ls);features=torch.cat(zs)
    p=evidence_dc(F.softplus(logits),rho=1.25)['p'][:,1]
    assert before==state_hash(model.state_dict()) and torch.isfinite(p).all()
    return {'subject_ids':ids,'logits':logits,'features':features,'p':p}

def main():
    cfg=config_check();assert read(OUT/'preflight.json')['passed'];data=Data();done=0;t0=time.monotonic()
    for seed in SEEDS:
        initial=torch.load(G1/f'initial/seed_{seed}.pt',map_location='cpu',weights_only=True)
        for fold in FOLDS:
            outer=data.ctx.folds[f'fold_{fold}']
            for part in cfg['inner_splits'][str(fold)]:
                fit=part['fit'];cal=part['calibration'];i=part['inner']
                batch_orders=read(OUT/f'orders/seed_{seed}/fold_{fold}/inner_{i}.json')
                assert batch_orders==orders(fit,seed)
                for c in CONDITIONS:
                    stem=folder(c,seed,fold,i);target=stem/'complete.json'
                    if target.exists():
                        for d in read(target)['artifacts']:verify(d)
                        done+=1;continue
                    assert not stem.exists(),'Inspect incomplete run before resume: '+str(stem)
                    stem.mkdir(parents=True);model=Model(c,seed).cuda()
                    init={k:v for k,v in initial.items() if k in model.state_dict()};model.load_state_dict(init,strict=True)
                    assert state_hash(model.state_dict())==state_hash(Model(c,seed).state_dict())
                    set_seed(seed);optimizer=model.optimizer();before=predict(model,data,fit[:8]);records=[];updates=0
                    torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()},stem/'rng_initial.pt')
                    started=time.monotonic()
                    for epoch,order in enumerate(batch_orders,1):
                        model.train();losses=[];norms=[];clip=0;grads={m:[] for m in model.modalities}
                        for j in range(0,len(order),8):
                            batch=order[j:j+8]
                            assert set(batch)<=set(fit) and set(batch).isdisjoint(cal+outer['val']+outer['test'])
                            x,y=data.batch(batch);optimizer.zero_grad(set_to_none=True)
                            loss=model.objective(model(x),y,epoch);assert torch.isfinite(loss);loss.backward()
                            for m in model.modalities:
                                grads[m].append(gradient_norm(list(model.encoders[m].parameters())+list(model.heads[m].parameters())))
                            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                            optimizer.step();model.constrain();updates+=1;losses.append((float(loss.detach()),len(batch)))
                            norms.append(norm);clip+=int(norm>1)
                        records.append({'epoch':epoch,'loss':sum(v*n for v,n in losses)/len(fit),'updates':len(losses),
                            'global_gradient_mean':float(np.mean(norms)),'clip_count':clip,
                            'branch_gradient_mean':{m:float(np.mean(v)) for m,v in grads.items()}})
                    assert updates==180
                    after=predict(model,data,fit[:8]);oof=predict(model,data,cal)
                    final={k:v.detach().cpu() for k,v in model.state_dict().items()}
                    changed={m:sum(float((final[k]-init[k]).double().square().sum()) for k in final
                        if (k.startswith(f'encoders.{m}.') or k.startswith(f'heads.{m}.')) and final[k].is_floating_point())**.5 for m in model.modalities}
                    assert all(v>0 for v in changed.values())
                    cp={'model':final,'optimizer':optimizer.state_dict(),'condition':c,'seed':seed,'fold':fold,'inner':i,
                        'fit_ids':fit,'calibration_ids':cal,'epochs':60,'updates':updates,'initial_state_sha256':state_hash(init),
                        'final_state_sha256':state_hash(final),'registration':desc(OUT/'registration.json'),
                        'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()}
                    torch.save(cp,stem/'final.pt');torch.save({'before':before,'after':after,'oof':oof},stem/'probes.pt')
                    replay=Model(c,seed).cuda();replay.load_state_dict(torch.load(stem/'final.pt',weights_only=True)['model'])
                    for expected,ids in [(after,fit[:8]),(oof,cal)]:
                        actual=predict(replay,data,ids)
                        for k in ['logits','features','p']:assert torch.equal(actual[k],expected[k]),'Reload differs'
                    rows=[{'subject_id':sid,'label':data.ctx.labels[sid],'p':float(p),'fit_ids':fit,'inner':i,
                        'checkpoint':desc(stem/'final.pt')} for sid,p in zip(cal,oof['p'])]
                    save(stem/'oof.json',rows,immutable=True);save(stem/'history.json',records,immutable=True)
                    save(target,{'condition':c,'seed':seed,'fold':fold,'inner':i,'fit_ids':fit,'calibration_ids':cal,
                        'updates':updates,'clip_rate':sum(r['clip_count'] for r in records)/updates,
                        'branch_state_change_l2':changed,'reload_exact':True,'outer_val_test_used':False,
                        'completed_utc':now(),'seconds':time.monotonic()-started,
                        'artifacts':[desc(stem/n) for n in ['final.pt','probes.pt','rng_initial.pt','history.json','oof.json']]},immutable=True)
                    done+=1
                    save(OUT/'status.json',{'stage':'inner_training','completed_inner_models':done,'target_inner_models':90,
                        'completed_updates':done*180,'G4_outer_transforms_released':False,'last':f'{c}/seed{seed}/fold{fold}/inner{i}','updated_utc':now()})
                    print(f'{done}/90 {c} seed={seed} fold={fold} inner={i} {time.monotonic()-started:.1f}s total={time.monotonic()-t0:.1f}s',flush=True)
                    del model,replay,optimizer,cp
    config_check();assert len(list((OUT/'runs').rglob('complete.json')))==90
    save(OUT/'inner_models_lock.json',{'passed':True,'locked_utc':now(),'models':90,'updates':16200,
        'artifacts':{p.relative_to(OUT).as_posix():digest(p) for p in sorted((OUT/'runs').rglob('*')) if p.is_file()}},immutable=True)
    save(OUT/'status.json',{'stage':'inner_training_complete','completed_inner_models':90,'completed_updates':16200,'G4_outer_transforms_released':False,'updated_utc':now()})

if __name__=='__main__':main()
