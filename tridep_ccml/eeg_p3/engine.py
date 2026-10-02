from __future__ import annotations
import copy
import time
from torch.nn import functional as F
from .common import *
from .methods import *
from tridep_ccml.eeg_replacement.data import FoldData
from tridep_ccml.eeg_replacement.train import orders,step
from tridep_ccml.ccml import ccml_loss,dirichlet_from_evidence,evidence_dc

class Data:
    def __init__(self,fold):
        self.base=FoldData(fold)
        self.ids=self.base.context.ids
        self.train=self.base.splits['train']
        self.labels=self.base.context.labels
        raw=np.stack([np.load(verify_descriptor(r['raw']),allow_pickle=False)
                      for r in self.base.manifest['raw_and_tokens']])
        normalized,mean,std=subject_standardize(raw)
        self.subject=torch.from_numpy(normalized).cuda()
        target=OUT/f'subject_standardizers/fold_{fold}.json'
        write_new(target,{'subject_ids':self.ids,'mean':mean.tolist(),'std':std.tolist(),
            'fit_scope':'one complete 300-second record independently; no labels or cohort fit'})

    def batch(self,ids,condition,arm):
        x,y=self.base.batch(ids,condition)
        if arm in ['S','SR']:
            x['eeg']=self.subject[[self.base.index[s] for s in ids]]
        return x,y

    def fit_batch(self,ids,condition,arm,allowed):
        require(set(ids)<=set(allowed)<=set(self.train),'Training includes outer/inner held-out identities')
        return self.batch(ids,condition,arm)

@torch.no_grad()
def predict(model,data,ids,condition,arm,batch_size=8):
    model.eval()
    before=state_digest(model.state_dict())
    outputs=[]; features=[]
    for k in range(0,len(ids),batch_size):
        x,_=data.batch(ids[k:k+batch_size],condition,arm)
        outputs.append(model(x).cpu())
        z=model.representations(x)
        features.append((z[:,0] if model.joint else z).cpu())
    out=torch.cat(outputs)
    if model.joint:
        probs={m:dirichlet_from_evidence(out)['p'][:,i,1].numpy() for i,m in enumerate(MODALITIES)}
        probs['fusion']=evidence_dc(out)['p'][:,1].numpy()
    else:
        probs={'eeg':out.softmax(-1)[:,1].numpy()}
    require(before==state_digest(model.state_dict()),'Inference changed model state')
    return probs,torch.cat(features).numpy(),out.numpy()

def health(model,data,c,arm,ids):
    require(set(ids)<=set(data.train),'Health check must use training people')
    model.eval(); before=state_digest(model.state_dict())
    grads={}
    for k in range(0,len(ids),8):
        x,y=data.fit_batch(ids[k:k+8],c,arm,ids)
        model.zero_grad(set_to_none=True)
        out=model(x)
        loss=ccml_loss(out,y,60,**LOSS) if model.joint else F.cross_entropy(out,y)
        loss.backward()
        for name,p in model.eeg.named_parameters():
            require(p.grad is not None and torch.isfinite(p.grad).all(),'Invalid EEG gradient: '+name)
            # Aggregate weight/bias within each trainable module.
            module=name.rsplit('.',1)[0]
            grads[module]=max(grads.get(module,0.),float(p.grad.norm()))
    model.zero_grad(set_to_none=True)
    _,z,_=predict(model,data,ids,c,arm)
    with torch.no_grad():
        x,_=data.batch(ids[:8],c,arm)
        normal=model.representations(x)
        if model.joint: normal=normal[:,0]
        x['eeg']=torch.zeros_like(x['eeg'])
        zero=model.representations(x)
        if model.joint: zero=zero[:,0]
        response=float((normal-zero).norm()/normal.norm().clamp_min(1e-12))
    rms=float(np.sqrt(np.mean(z.astype(float)**2)))
    std=float(np.sqrt(np.mean(np.var(z.astype(float),axis=0))))
    require(state_digest(model.state_dict())==before,'Health changed model')
    failures=[name for name,value in grads.items() if value<=1e-12]
    if std<=1e-6 or std/max(rms,1e-12)<=1e-4: failures.append('near_constant_representation')
    if response<=1e-4: failures.append('input_insensitivity')
    return {'passed':not failures,'failures':failures,'train_ids':ids,'module_max_gradient_l2':grads,
            'representation_std_rms':std,'relative_representation_std':std/max(rms,1e-12),
            'zero_input_relative_response':response,'state_unchanged':True}

def save_checkpoint(path,model,metadata):
    require(not path.exists(),'Checkpoint already exists without completion marker')
    path.parent.mkdir(parents=True,exist_ok=True)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({'model':state,'state_sha256':state_digest(state),**metadata},path)

def train_new(data,c,arm,seed,fold,fit_ids,inner=None):
    path=folder(c,arm,seed,fold,inner)
    if (path/'trained.json').exists():
        meta=read_json(path/'trained.json')
        verify_descriptor(meta['checkpoint']);verify_descriptor(meta['history'])
        require(meta['fit_ids']==fit_ids and meta['configuration']==descriptor(OUT/'configuration.json'),'Stale run')
        return
    set_seed(seed)
    model=make_model(c,seed,arm).cuda()
    initial=state_digest(model.state_dict())
    shared=state_digest(model.shared_state()) if model.joint else None
    paired=make_model(c,seed,'B')
    require(all(torch.equal(p.detach().cpu(),dict(paired.named_parameters())[name])
                for name,p in model.named_parameters()),'Paired initial learned parameters differ')
    del paired
    if inner is None:
        old=read_json(R1/f'runs/{c}/seed_{seed}/fold_{fold}/trained.json')
        if arm=='S': require(initial==old['initial_state_sha256'],'S initial state differs from B')
        require(shared==old['initial_shared_sha256'],'Paired non-EEG initialization differs')
    path.mkdir(parents=True,exist_ok=True)
    initial_health=health(model,data,c,arm,fit_ids)
    write_new(path/'initial_health.json',initial_health)
    require(initial_health['passed'],'Initial learning gate failed: '+str(path))
    epoch_orders=orders(fit_ids,seed)
    if inner is None:
        parent_orders=read_json(R1/f'runs/{c}/seed_{seed}/fold_{fold}/batch_order.json')['epochs']
        require(epoch_orders==parent_orders,'Paired training orders differ')
    write_new(path/'batch_order.json',{'fit_ids':fit_ids,'orders':epoch_orders,'batch_size':8})
    optimizer=model.optimizer()
    history=[]; updates=0; start=time.perf_counter()
    for epoch,order in enumerate(epoch_orders,1):
        model.train(); total=0.
        for k in range(0,len(order),8):
            batch=order[k:k+8]
            x,y=data.fit_batch(batch,c,arm,fit_ids)
            loss,norm=step(model,optimizer,x,y,epoch)
            total+=loss*len(batch);updates+=1
        history.append({'epoch':epoch,'loss':total/len(fit_ids)})
    model.eval()
    final_health=health(model,data,c,arm,fit_ids)
    write_new(path/'final_health.json',final_health)
    require(final_health['passed'],'Final learning gate failed: '+str(path))
    require(updates==60*math.ceil(len(fit_ids)/8),'Wrong update count')
    cpmeta={'condition':c,'arm':arm,'seed':seed,'fold':fold,'inner':inner,'fit_ids':fit_ids,
        'epoch':60,'updates':updates,'initial_state_sha256':initial,'initial_shared_sha256':shared,
        'configuration':descriptor(OUT/'configuration.json'),'augmentation':False}
    save_checkpoint(path/'checkpoint.pt',model,cpmeta)
    write_new(path/'history.json',history)
    write_new(path/'trained.json',{**cpmeta,'checkpoint':descriptor(path/'checkpoint.pt'),
        'history':descriptor(path/'history.json'),'seconds':time.perf_counter()-start,
        'final_state_sha256':state_digest(model.state_dict())})
    print(f'TRAINED {c}-{arm} seed={seed} fold={fold} inner={inner} {time.perf_counter()-start:.1f}s',flush=True)

def load_model(c,arm,seed,fold,inner=None):
    if arm=='B': path=R1/f'runs/{c}/seed_{seed}/fold_{fold}/checkpoint.pt'
    else: path=folder(c,arm,seed,fold,inner)/'checkpoint.pt'
    cp=torch.load(path,map_location='cpu',weights_only=True)
    require(cp.get('final_state_sha256',cp.get('state_sha256'))==state_digest(cp['model']),'Checkpoint digest failed')
    model=make_model(c,seed,arm).cuda().eval()
    model.load_state_dict(cp['model'],strict=True)
    return model,descriptor(path)

def derive_bn(data,c,source,seed,fold):
    arm={'B':'BR','S':'SR'}[source]
    path=folder(c,arm,seed,fold)
    if (path/'derived.json').exists():
        meta=read_json(path/'derived.json');verify_descriptor(meta['checkpoint']);return
    model,origin=load_model(c,source,seed,fold)
    def batches():
        for k in range(0,len(data.train),8):
            ids=data.train[k:k+8]
            yield ids,data.fit_batch(ids,c,arm,data.train)[0]
    report=recompute_bn(model,batches,data.train)
    path.mkdir(parents=True,exist_ok=True)
    write_new(path/'bn_fit.json',report)
    save_checkpoint(path/'checkpoint.pt',model,{'condition':c,'arm':arm,'seed':seed,'fold':fold,
        'fit_ids':data.train,'source':origin,'augmentation':False})
    write_new(path/'derived.json',{'source':origin,'checkpoint':descriptor(path/'checkpoint.pt'),
        'bn_fit':descriptor(path/'bn_fit.json'),'learned_parameters_unchanged':True})

def token_attention(data,seed,fold):
    path=OUT/f'attention/seed_{seed}_fold_{fold}.json'
    if path.exists(): return
    model,source=load_model('C4','B',seed,fold)
    before=state_digest(model.state_dict())
    original=make_model('C4',seed,'B').eeg.query.detach().cuda()
    rows=[]
    with torch.no_grad():
        for sid in data.train:
            x,_=data.fit_batch([sid],'C4','B',data.train)
            z=model.eeg.projection(x['eeg'].flatten(2,3))
            heads=z.reshape(*z.shape[:-1],2,16)
            w=((heads*model.eeg.query).sum(-1)/4).softmax(-2)
            entropy=-(w*w.clamp_min(1e-30).log()).sum(-2)/math.log(95)
            rows.append({'subject_id':sid,'normalized_entropy_mean':float(entropy.mean()),
                'maximum_weight_mean':float(w.max(-2).values.mean()),
                'effective_positions_mean':float(1/w.square().sum(-2).mean())})
    p,z,_=predict(model,data,data.train,'C4','B')
    model.eeg.attention=False
    q,uniform,_=predict(model,data,data.train,'C4','B')
    model.eeg.attention=True
    model.zero_grad(set_to_none=True)
    x,y=data.fit_batch(data.train[:8],'C4','B',data.train)
    ccml_loss(model(x),y,60,**LOSS).backward()
    query_grad=float(model.eeg.query.grad.norm());model.zero_grad(set_to_none=True)
    require(before==state_digest(model.state_dict()),'Attention diagnostic mutated model')
    write_new(path,{'source':source,'train_ids':data.train,'subjects':rows,
        'query_initial_norm':float(original.norm()),'query_final_norm':float(model.eeg.query.norm()),
        'query_change_norm':float((model.eeg.query-original).norm()),'query_gradient_norm':query_grad,
        'uniform_ablation_representation_relative_l2':float(np.linalg.norm(z-uniform)/max(np.linalg.norm(z),1e-12)),
        'uniform_ablation_eeg_probability_mean_abs_change':float(np.mean(np.abs(p['eeg']-q['eeg']))),
        'uniform_ablation_eeg_prediction_flips':int(np.sum((p['eeg']>.5)!=(q['eeg']>.5))),
        'state_unchanged':True,'outer_people_used':False})

def calibrate(data,c,seed,fold,splits):
    target=OUT/f'calibrators/{c}/seed_{seed}/fold_{fold}.json'
    if target.exists(): return
    rows={'eeg':[],'fusion':[]}
    for split in splits:
        i=split['inner'];fit=split['fit'];ids=split['calibration']
        train_new(data,c,'I',seed,fold,fit,inner=i)
        model,source=load_model(c,'I',seed,fold,inner=i)
        p,_,_=predict(model,data,ids,c,'I')
        for j,sid in enumerate(ids):
            for endpoint in rows:
                rows[endpoint].append({'subject_id':sid,'label':data.labels[sid],'p':float(p[endpoint][j]),
                    'fit_ids':fit,'inner':i,'checkpoint':source})
        del model
    params={e:{'sigmoid':fit_sigmoid(r,data.train),'threshold':fit_threshold(r,data.train)} for e,r in rows.items()}
    write_new(target,{'condition':c,'seed':seed,'fold':fold,'train_ids':data.train,'oof':rows,'parameters':params,
        'outer_val_test_used':False,'augmentation':False})
    print(f'CALIBRATED {c} seed={seed} fold={fold}',flush=True)

def evaluate_model(data,c,arm,seed,fold):
    target=OUT/f'predictions/{c}_{arm}/seed_{seed}/fold_{fold}.json'
    if target.exists(): return
    model,source=load_model(c,arm,seed,fold)
    p,z,out=predict(model,data,data.ids,c,arm)
    # Independent single-person inference verifies batch independence for a train person.
    solo,_,_=predict(model,data,data.ids[:1],c,arm,batch_size=1)
    for e in solo: np.testing.assert_allclose(solo[e],p[e][:1],rtol=2e-5,atol=2e-6)
    reload_model,_=load_model(c,arm,seed,fold)
    rp,_,_=predict(reload_model,data,data.train[:8],c,arm)
    for e in rp:
        np.testing.assert_allclose(rp[e],p[e][[data.ids.index(s) for s in data.train[:8]]],rtol=2e-5,atol=2e-6)
    if arm=='B':
        old=read_json(R1/f'runs/{c}/seed_{seed}/fold_{fold}/predictions.json')
        mapping={r['subject_id']:r for r in old}
        for e in p:
            np.testing.assert_allclose(p[e],[mapping[s]['probabilities'][e][1] for s in data.ids],rtol=1e-5,atol=1e-6)
    splitmap={sid:s for s,ids in data.base.splits.items() for sid in ids}
    rows=[]
    for j,sid in enumerate(data.ids):
        rows.append({'subject_id':sid,'label':data.labels[sid],'split':splitmap[sid],
            'probabilities':{e:float(a[j]) for e,a in p.items()}})
    write_new(target,{'condition':c,'arm':arm,'seed':seed,'fold':fold,'source':source,
        'rows':rows,'checkpoint_reload_passed':True,'batch_independence_passed':True,
        'baseline_replay_passed':arm=='B','inference_state_unchanged':True})
    feature=target.with_suffix('.npz')
    np.savez_compressed(feature,features=z,subject_ids=np.array(data.ids),outputs=out)

def small_fit(data,arm):
    ids=[next(s for s in data.train if data.labels[s]==y) for y in [0,1]]
    ids += [next(s for s in data.train if data.labels[s]==y and s not in ids) for y in [0,1]]
    set_seed(11)
    model=make_model('U1',11,arm).cuda().train()
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    x,y=data.fit_batch(ids,'U1',arm,ids)
    for i in range(500):
        model.train();opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x),y);loss.backward();opt.step();model.constrain()
    model.eval()
    # BN is evaluated consistently after train-only reestimation for this engineering probe.
    if arm=='S': recompute_bn(model,lambda:iter([(ids,x)]),ids)
    with torch.no_grad():
        out=model(x); ce=float(F.cross_entropy(out,y)); accuracy=float((out.argmax(-1)==y).float().mean())
    return {'arm':arm,'ids':ids,'steps':500,'ce':ce,'accuracy':accuracy,'passed':ce<.1 and accuracy==1.,
            'probe_not_used_for_model_selection':True}
