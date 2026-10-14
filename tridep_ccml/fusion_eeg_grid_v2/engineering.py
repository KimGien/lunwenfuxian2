"""Synthetic-only author parity, backend timing and exact recovery checks."""
from __future__ import annotations
import copy
import json
import math
import subprocess
import time
import numpy as np
import torch
from . import common as C
from .models import Fusion,load_reference
from . import training as T

TOLERANCE={'backend_probability_atol':1e-4,'backend_loss_atol':1e-4,'backend_gradient_atol':1e-4,
           'backend_rtol':1e-4,'source_EEG_logit_atol':2e-5,'stage0_probability_atol':2e-6}

def fixture(views):
    generator=torch.Generator().manual_seed(8041)
    return torch.randn(27,views,128,generator=generator),torch.arange(27)%2

def author_parity():
    result=[]
    for views in (2,3):
        x,y=fixture(views)
        for method in C.METHODS:
            C.runtime(11)
            model=Fusion(method,views,27).eval()
            output=model(x)
            C.require(output['p'].shape==(27,2) and torch.allclose(output['p'].sum(-1),torch.ones(27)), 'Invalid model probabilities')
            if method=='ECML':
                loss_module=load_reference('ECML','loss_function.py')
                evidence=output['evidence']
                ref=loss_module.get_loss({i:evidence[:,i] for i in range(views)},output['final'],y,23,2,50,1.,'cpu')
                actual,_=model.objective(x,y,23)
                C.require(torch.allclose(actual,ref,atol=2e-6,rtol=2e-6),'ECML author loss mismatch')
                original=load_reference('ECML','model.py').RCML(views,[[128,32]]*views,2).eval()
                for i in range(views):
                    original.EvidenceCollectors[i].net[0].load_state_dict(model.heads[i][0].state_dict())
                    original.EvidenceCollectors[i].net[3].load_state_dict(model.heads[i][3].state_dict())
                _,final=original([x[:,i] for i in range(views)])
                C.require(torch.equal(output['final'],final),'ECML author sequential aggregation differs')
            elif method=='LMF':
                # Independent sum-over-rank evaluation, retaining constant interactions.
                full=[]
                for r in range(2):
                    row=torch.ones((len(x),16))
                    for v in range(views):
                        row=row*(torch.cat([torch.ones(len(x),1),x[:,v]],1)@model.factors[v][r])
                    full.append(row*model.rank_weights[0,r])
                logits=model.classifier(sum(full)+model.bias)
                C.require(torch.allclose(logits,output['logits'],atol=1e-6,rtol=1e-5),'LMF independent rank contraction differs')
            elif method=='QMF':
                logits=output['logits_v'];energy=-torch.logsumexp(logits,dim=2)
                ref=(logits*(-.1*energy).unsqueeze(-1)).sum(1)
                C.require(torch.equal(ref,output['logits']),'QMF original energy fusion mismatch')
                before=model.history.copy();loss,_=model.objective(x[:8],y[:8],1,torch.arange(8))
                C.require(torch.isfinite(loss) and np.any(model.history[:,:8]>0) and
                          np.array_equal(model.history[:,8:],before[:,8:]),'QMF first-batch or train-history boundary failed')
            elif method=='MMDynamics':
                reference=load_reference('MMDynamics','model.py').MMDynamic([128]*views,[32],2,.2).eval()
                reference.load_state_dict(model.network.state_dict(),strict=True)
                loss,_=model.objective(x,y,1)
                expected,logits=reference([x[:,i] for i in range(views)],y)
                C.require(torch.equal(loss,expected) and torch.equal(output['logits'],logits),'MMDynamics author forward/full loss mismatch')
            elif method=='CCML':
                from tridep_ccml import ccml
                e=output['evidence'];common=e.amin(1);powers=common.double().pow(1.25)
                con=(views*common.double().sum(-1,keepdim=True)*powers/powers.sum(-1,keepdim=True)).float()
                fused=con+(e-common[:,None]).mean(1);alpha=fused+1
                C.require(torch.allclose(alpha/alpha.sum(-1,keepdim=True),output['p'],atol=2e-7,rtol=2e-7),'CCML independent decomposition differs')
            result.append({'method':method,'views':views,'passed':True})
    return result

def backend_parity(method,views):
    x,y=fixture(views);records=[]
    for device in ('cpu','cuda'):
        model,opt=T.setup(method,views,27,71,device);model.eval()
        xd,yd=x.to(device),y.to(device)
        p=model(xd)['p'].detach().cpu()
        loss,_=model.objective(xd[:8],yd[:8],3,torch.arange(8,device=device))
        loss.backward()
        grads=torch.cat([param.grad.detach().reshape(-1).cpu() for param in model.parameters() if param.grad is not None])
        records.append((p,float(loss),grads))
    a,b=records
    C.require(torch.allclose(a[0],b[0],atol=1e-4,rtol=1e-4) and abs(a[1]-b[1])<=1e-4+1e-4*abs(a[1])
              and torch.allclose(a[2],b[2],atol=1e-4,rtol=1e-4),'Predeclared cross-backend tolerance failed: '+method)
    return {'views':views,'probability_error':float((a[0]-b[0]).abs().max()),
            'loss_error':abs(a[1]-b[1]),'gradient_error':float((a[2]-b[2]).abs().max()),'passed':True}

def recovery_check(method,views,device):
    x,y=fixture(views);x,y=x.to(device),y.to(device)
    uninterrupted,opt=T.setup(method,views,27,31,device)
    for epoch in range(1,5):
        T.train_epoch(uninterrupted,opt,x,y,epoch)
    reference=C.cpu_state(uninterrupted.state_dict());refopt=C.cpu_state(opt.state_dict())
    refhist=T.history_state(uninterrupted);refrng=C.rng_state()
    resumed,optimizer=T.setup(method,views,27,31,device)
    for epoch in (1,2):
        T.train_epoch(resumed,optimizer,x,y,epoch)
    p=C.OUT/'engineering'/f'{method}_{views}_{device}_resume.pt'
    C.save_torch(p,{'state':C.cpu_state(resumed.state_dict()),'optimizer':C.cpu_state(optimizer.state_dict()),
                   'history':T.history_state(resumed),'rng':C.rng_state(),'epoch':2})
    value=torch.load(p,map_location='cpu',weights_only=False)
    resumed,optimizer=T.setup(method,views,27,31,device)
    resumed.load_state_dict(value['state']);optimizer.load_state_dict(value['optimizer'])
    T.restore_history(resumed,value['history']);C.restore_rng(value['rng'])
    for epoch in (3,4):
        T.train_epoch(resumed,optimizer,x,y,epoch)
    C.require(T.equal_state(reference,C.cpu_state(resumed.state_dict())) and
              T.equal_state(refopt,C.cpu_state(optimizer.state_dict())) and
              T.equal_state(refhist,T.history_state(resumed)) and T.equal_state(refrng,C.rng_state()),
              'Serialized recovery is not bitwise equivalent: '+method)
    return {'views':views,'device':device,'model_optimizer_rng_history_bitwise':True,'checkpoint':C.desc(p)}

def benchmark(method,views,device,trial):
    x,y=fixture(views)
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter()
    model,opt=T.setup(method,views,27,11,device);x,y=x.to(device),y.to(device)
    for epoch in (1,2,3):
        record=T.train_epoch(model,opt,x,y,epoch)
        model.eval()
        with torch.inference_mode():
            p=model(x[:3])['p'];float(T.nll(p,y[:3]))
        C.save_torch(C.OUT/'engineering'/f'benchmark_{method}_{views}_{device}_{trial}.pt',
                     {'state':C.cpu_state(model.state_dict()),'optimizer':C.cpu_state(opt.state_dict()),
                      'rng':C.rng_state(),'history':T.history_state(model)})
        C.save(C.OUT/'engineering'/f'benchmark_{method}_{views}_{device}_{trial}.json',record)
    if device=='cuda':torch.cuda.synchronize()
    return {'trial':trial,'seconds':time.perf_counter()-start,'updates':12,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}

def run():
    path=C.OUT/'engineering/complete.json'
    if path.exists():
        value=C.read(path)
        C.require(value['passed'],'Engineering not accepted')
        for source in value['sources']:C.checked(source)
        return value
    C.save(C.OUT/'engineering/tolerances.json',TOLERANCE,immutable=True)
    parity=author_parity(); methods={}
    for method in C.METHODS:
        C.check_pause()
        cross=[backend_parity(method,views) for views in (2,3)]
        timing={device:[benchmark(method,views,device,trial) for views in (2,3) for trial in range(3)]
                for device in ('cpu','cuda')}
        medians={device:float(np.median([r['seconds'] for r in rr])) for device,rr in timing.items()}
        device=min(medians,key=medians.get)
        recover=[recovery_check(method,v,device) for v in (2,3)]
        methods[method]={'device':device,'precision':'FP32','cross_backend':cross,'timing':timing,
          'median_end_to_end_seconds':medians,'recovery':recover,
          'inference_batch':'original per-person source export; val3 as single batch',
          'inference_batch_optimization_enabled':False}
        print(f"Engineering {method}: backend {device}; medians {medians}",flush=True)
    C.require(lr_last_test(),'Learning-rate or patience contract failed')
    try:
        gpu_sample=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv'],
           capture_output=True,text=True,timeout=20,check=True).stdout
    except Exception as error:
        gpu_sample={'unavailable':repr(error)}
    value={'passed':True,'author_parity':parity,'methods':methods,'tolerances':TOLERANCE,
      'backend_selection_data':'synthetic training only; no validation/test people read',
      'sources':[C.desc(p) for p in sorted(C.CODE.glob('*.py'))]+[C.desc(C.CODE/'upstream_sources.json')],
      'gpu_diagnostic_sample':gpu_sample,'at':C.now(),'formal_fits':0,'benchmark_updates_separate':True,
      'optimization':'device-resident features; epoch diagnostics; original batch preserved; atomic checkpoint',
      'claimed_speedup':None,'source_export_original_batch_retained':True}
    C.save(path,value,immutable=True)
    return value

def lr_last_test():
    return T.lr_for_epoch(1)==1e-4 and T.lr_for_epoch(10)==1e-3 and abs(T.lr_for_epoch(120)-1e-4)<1e-15
