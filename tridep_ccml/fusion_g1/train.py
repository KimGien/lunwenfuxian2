import time
from .common import *
from .model import Model,gradient_norm

@torch.no_grad()
def probe(model,data,ids):
    model.eval(); x,_=data.batch(ids)
    logits,z,n=model(x,features=True)
    return {'subject_ids':ids,'logits':logits.cpu(),'features':z.cpu(),'normalized':n.cpu()}

def main():
    lock_check(); assert read(OUT/'preflight.json')['passed']
    data=Data(); done=0; started=time.monotonic()
    for seed in SEEDS:
        initial=torch.load(OUT/f'initial/seed_{seed}.pt',weights_only=True,map_location='cpu')
        for fold in FOLDS:
            ids=data.ctx.folds[f'fold_{fold}']['train']; batch_orders=read(OUT/f'orders/seed_{seed}/fold_{fold}.json')
            assert batch_orders==orders(ids,seed)
            for condition in CONDITIONS:
                stem=OUT/f'runs/{condition}/seed_{seed}/fold_{fold}'
                if (stem/'complete.json').exists():
                    record=read(stem/'complete.json')
                    for d in record['artifacts']:verify(d)
                    done+=1;continue
                assert not stem.exists(),f'Incomplete run must be inspected before resuming: {stem}'
                stem.mkdir(parents=True)
                model=Model(condition,seed).cuda()
                init={k:v for k,v in initial.items() if k in model.state_dict()}
                model.load_state_dict(init,strict=True)
                init_hash=state_hash(model.state_dict())
                set_seed(seed)
                optimizer=model.optimizer(); before=probe(model,data,ids[:8]); hist=[]; updates=0
                torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()},stem/'rng_initial.pt')
                run_start=time.monotonic()
                for epoch,order in enumerate(batch_orders,1):
                    model.train(); losses=[];norms=[];clip=0;grads={m:[] for m in model.modalities}
                    for start in range(0,len(order),8):
                        batch=order[start:start+8]; x,y=data.batch(batch)
                        optimizer.zero_grad(set_to_none=True)
                        loss=model.objective(model(x),y,epoch)
                        assert torch.isfinite(loss);loss.backward()
                        for m in model.modalities:
                            params=list(model.encoders[m].parameters())+list(model.heads[m].parameters())
                            grads[m].append(gradient_norm(params))
                        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                        optimizer.step();model.constrain()
                        losses.append((float(loss.detach()),len(batch)));norms.append(norm);clip+=int(norm>1);updates+=1
                    hist.append({'epoch':epoch,'loss':sum(v*n for v,n in losses)/len(order),
                        'updates':len(losses),'global_grad_norm_mean':float(np.mean(norms)),'global_grad_norm_max':max(norms),
                        'clip_count':clip,'branch_gradient_norm_mean':{m:float(np.mean(v)) for m,v in grads.items()},
                        'branch_gradient_norm_max':{m:max(v) for m,v in grads.items()}})
                assert updates==240
                after=probe(model,data,ids[:8]); final={k:v.detach().cpu() for k,v in model.state_dict().items()}
                changed={m:sum(float((final[k]-init[k]).double().square().sum()) for k in final
                    if (k.startswith(f'encoders.{m}.') or k.startswith(f'heads.{m}.')) and final[k].is_floating_point())**.5 for m in model.modalities}
                assert all(v>0 for v in changed.values()) and torch.isfinite(after['logits']).all()
                cp={'condition':condition,'seed':seed,'fold':fold,'train_ids':ids,'epochs':60,'updates':updates,
                    'initial_state_sha256':init_hash,'final_state_sha256':state_hash(final),'model':final,
                    'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(),
                    'registration':desc(OUT/'registration.json')}
                torch.save(cp,stem/'final.pt');torch.save({'before':before,'after':after},stem/'train_probe.pt')
                replay=Model(condition,seed).cuda();replay.load_state_dict(torch.load(stem/'final.pt',weights_only=True)['model'])
                reprobe=probe(replay,data,ids[:8])
                for k in ['logits','features','normalized']:assert torch.equal(after[k],reprobe[k]),'Reload mismatch'
                save(stem/'history.json',hist,immutable=True)
                save(stem/'complete.json',{'condition':condition,'seed':seed,'fold':fold,'updates':updates,
                    'train_ids':ids,'validation_used_for_selection':False,'heldout_inference':False,
                    'initial_state_sha256':init_hash,'final_state_sha256':state_hash(final),
                    'branch_state_change_l2':changed,'reload_exact':True,'clip_rate':sum(h['clip_count'] for h in hist)/updates,
                    'elapsed_seconds':time.monotonic()-run_start,'completed_utc':now(),
                    'artifacts':[desc(stem/name) for name in ['final.pt','train_probe.pt','rng_initial.pt','history.json']]},immutable=True)
                done+=1
                save(OUT/'status.json',{'stage':'training','completed_models':done,'target_models':90,'completed_updates':done*240,
                    'outer_results_released':False,'last':f'{condition}/seed{seed}/fold{fold}','updated_utc':now()})
                print(f'{done}/90 {condition} seed={seed} fold={fold} {time.monotonic()-run_start:.1f}s elapsed={time.monotonic()-started:.1f}s',flush=True)
                del model,replay,optimizer,cp
    lock_check()
    runs=sorted((OUT/'runs').rglob('complete.json'));assert len(runs)==90
    save(OUT/'release_gate.json',{'passed':True,'created_utc':now(),'completed_models':90,'updates':21600,
        'G4_postprocess_enabled':False,'all_models_locked_before_outer_inference':True,
        'fitted_artifacts':{p.relative_to(OUT).as_posix():digest(p) for p in sorted((OUT/'runs').rglob('*')) if p.is_file()}},immutable=True)
    save(OUT/'status.json',{'stage':'training_complete','completed_models':90,'completed_updates':21600,'outer_results_released':False,'updated_utc':now()})

if __name__=='__main__':main()
