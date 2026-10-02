import time
from .core import *

def main():
    lock_check();data=Data();done=0;initials={}
    for seed in SEEDS:
        for fold in FOLDS:
            ids=data.ctx.folds[f'fold_{fold}']['train'];orderings=orders(ids,seed)
            for c in CONDITIONS:
                path=stem(c,seed,fold)
                if (path/'complete.json').exists():
                    d=read(path/'complete.json')
                    for a in d['artifacts']:assert digest(WORK/a['path'])==a['sha256']
                    initials[(seed,fold)]=d['initial_state_sha256'];done+=1;continue
                assert not path.exists(),f'Inspect incomplete run: {path}'
                path.mkdir(parents=True);started=time.monotonic();m=Model(c,seed).cuda();set_seed(seed)
                initial=state_hash(m.state());base=m.frozen_hash();adapter0=[p.detach().clone() for p in m.adapter_parameters()]
                if (seed,fold) in initials:assert initial==initials[(seed,fold)]
                initials[(seed,fold)]=initial
                before=infer(m,data,ids[:8]);rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
                if c==CONDITIONS[1]:
                    crng=torch.load(stem(CONDITIONS[0],seed,fold)/'rng_initial.pt',weights_only=True,map_location='cpu')
                    assert all(torch.equal(rng[k].cpu(),crng[k]) for k in rng)
                    prev=torch.load(stem(CONDITIONS[0],seed,fold)/'train_probe.pt',weights_only=True)['before']
                    assert all(torch.equal(before[k],prev[k]) for k in ['logits','features','normalized','question_cls'])
                torch.save(rng,path/'rng_initial.pt');opt=m.optimizer();gg=module_groups(m);hist=[];steps=[];updates=0
                for epoch,order in enumerate(orderings,1):
                    m.train();losses=[]
                    for start in range(0,len(order),8):
                        batch=order[start:start+8];x,y=data.batch(batch,ids);opt.zero_grad(set_to_none=True)
                        loss=m.objective(m(x),y,epoch);loss.backward()
                        assert all(p.grad is None for n,p in m.layer.named_parameters() if not (n.endswith('.A') or n.endswith('.B')))
                        snapshots={g:[p.detach().clone() for p in ps] for g,ps in gg.items()}
                        grads={g:norm([p.grad for p in ps]) for g,ps in gg.items()};pn={g:norm(ps) for g,ps in gg.items()}
                        gn=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
                        opt.step();changes={g:norm([p-v for p,v in zip(ps,snapshots[g])]) for g,ps in gg.items()}
                        steps.append({'update':updates+1,'epoch':epoch,'global_grad_pre':gn,
                            'modules':{g:{'grad_pre':grads[g],'parameter_l2':pn[g],'update_l2':changes[g],'relative_update':changes[g]/max(pn[g],1e-30)} for g in gg}})
                        losses.append((float(loss.detach()),len(batch)));updates+=1
                    hist.append({'epoch':epoch,'loss':sum(v*n for v,n in losses)/len(ids),'updates':len(losses)})
                assert updates==240 and base==m.frozen_hash()
                adapter_delta=norm([p-v for p,v in zip(m.adapter_parameters(),adapter0)])
                assert (adapter_delta>0) if m.learned else (adapter_delta==0)
                after=infer(m,data,ids[:8]);state={k:v.detach().cpu() for k,v in m.state().items()};final=state_hash(state)
                torch.save({'state':state,'condition':c,'seed':seed,'fold':fold,'fit_ids':ids,'updates':updates,'epochs':60,
                    'initial_state_sha256':initial,'final_state_sha256':final,'frozen_layer_sha256':base,
                    'optimizer':opt.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(),'registration':desc(OUT/'registration.json')},path/'final.pt')
                re=Model(c,seed).cuda();re.load(torch.load(path/'final.pt',weights_only=True,map_location='cpu')['state'])
                again=infer(re,data,ids[:8]);assert all(torch.equal(after[k],again[k]) for k in ['logits','features','normalized','question_cls'])
                torch.save({'before':before,'after':after},path/'train_probe.pt');save(path/'history.json',hist,immutable=True);save(path/'updates.json',steps,immutable=True)
                save(path/'complete.json',{'condition':c,'seed':seed,'fold':fold,'fit_ids':ids,'updates':updates,'reload_exact':True,
                    'initial_state_sha256':initial,'final_state_sha256':final,'frozen_layer_sha256':base,'frozen_parameters_unchanged':True,
                    'lora_delta_l2':adapter_delta,'elapsed_seconds':time.monotonic()-started,'completed_utc':now(),'heldout_inference':False,
                    'artifacts':[desc(path/f) for f in ['final.pt','rng_initial.pt','train_probe.pt','history.json','updates.json']]},immutable=True)
                done+=1;save(OUT/'status.json',{'stage':'training','models_completed':done,'updates_completed':done*240,
                    'last':f'{c}/seed{seed}/fold{fold}','outer_results_released':False,'updated_utc':now()})
                print(f'{done}/30 {c} seed={seed} fold={fold} {time.monotonic()-started:.1f}s',flush=True)
                del m,re,opt,state,steps,snapshots,adapter0
    lock_check();assert len(list((OUT/'runs').rglob('complete.json')))==30
    save(OUT/'release_gate.json',{'passed':True,'models':30,'updates':7200,'created_utc':now(),
        'fitted_artifacts':{p.relative_to(OUT).as_posix():digest(p) for p in sorted((OUT/'runs').rglob('*')) if p.is_file()}},immutable=True)
    save(OUT/'status.json',{'stage':'fits_locked','models_completed':30,'updates_completed':7200,'outer_results_released':False,'updated_utc':now()})
    print('All 30 fits locked; ready for evaluation and fold-bound feature export.',flush=True)

if __name__=='__main__':main()
