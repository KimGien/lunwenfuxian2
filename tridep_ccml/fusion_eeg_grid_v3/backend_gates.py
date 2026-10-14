"""Synthetic end-to-end backend timing; no candidate test predictions."""
from __future__ import annotations
import gc
import statistics
import time
import numpy as np
import torch
from . import common as C
from . import training as T

TOLERANCE = {'atol': 1e-4, 'rtol': 1e-4, 'selection_data': 'synthetic only; no test metrics'}
def fixture(views):
    g=torch.Generator().manual_seed(8041)
    return torch.randn(30,views,128,generator=g), torch.arange(30)%2

def synchronize(device):
    if device=='cuda': torch.cuda.synchronize()

def parity(method,views,variant):
    x,y=fixture(views); outputs=[]
    for device in ('cpu','cuda'):
        C.check_pause()
        model,opt=T.setup(method,views,27,71,device,qmf_variant=variant)
        if model.history is not None:
            model.history[:]=np.linspace(0.,2.,27)[None,:]
        model.eval()
        xd,yd=x.to(device),y.to(device)
        prob=model(xd)['p'].detach().cpu()
        loss,_=model.objective(xd[:8],yd[:8],23,torch.arange(8,device=device))
        loss.backward()
        gradients=torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None])
        outputs.append((prob,float(loss.detach()),gradients))
        del model,opt
    a,b=outputs
    for aa,bb in ((a[0],b[0]),(torch.tensor(a[1]),torch.tensor(b[1])),(a[2],b[2])):
        C.require(torch.allclose(aa,bb,atol=TOLERANCE['atol'],rtol=TOLERANCE['rtol']), 'Predeclared cross-backend tolerance failed')
    return {'method':method,'views':views,'variant':variant,'passed':True,
            'probability_max_error':float((a[0]-b[0]).abs().max()),
            'loss_error':abs(a[1]-b[1]),'gradient_max_error':float((a[2]-b[2]).abs().max())}

def benchmark(method,views,device,variant,trial,directory):
    C.check_pause()
    x,y=fixture(views)
    synchronize(device); started=time.perf_counter()
    model,opt=T.setup(method,views,27,31+trial,device,qmf_variant=variant)
    x,y=x.to(device),y.to(device)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    rows=[]
    for epoch in range(1,4):
        C.check_pause()
        row=T.train_epoch(model,opt,x[:27],y[:27],epoch)
        model.eval()
        with torch.inference_mode(): v=T.nll(model(x[27:])['p'],y[27:])
        C.require(bool(torch.isfinite(v)), 'Nonfinite benchmark validation')
        C.save_torch(directory/f'{method}_{views}_{device}_{variant}_{trial}.pt',
          {'epoch':epoch,'model':C.cpu_state(model.state_dict()),'optimizer':C.cpu_state(opt.state_dict()),
           'rng':C.rng_state(),'quality_history':T.history_state(model),'validation_nll':float(v),
           'scheduler':T.scheduler_state(epoch,dict(T.FIXED,base_lr=.001,budget=120))})
        rows.append(row)
    synchronize(device)
    elapsed=time.perf_counter()-started
    result={'views':views,'device':device,'variant':variant,'trial':trial,'seconds':elapsed,
            'optimizer_updates':sum(r['updates'] for r in rows),'epochs':3,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'scope':'setup+transfer once+train+validation+durable checkpoint; synthetic'}
    del model,opt,x,y; gc.collect()
    if device=='cuda': torch.cuda.empty_cache()
    return result

def _run():
    C.check_pause()
    output=C.OUT/'backend_gate.json'
    if output.exists():
        proof=C.read(output)
        for s in proof['sources']: C.checked(s)
        C.require(proof['passed'],'Backend gate not passed')
        return proof
    directory=C.REPORT/'backend_gates'/f'run_{time.time_ns()}'
    C.save(directory/'predeclared.json',dict(TOLERANCE,decision='lower median end-to-end time; tie CPU; same method device across N/Q variants'),immutable=True)
    C.require(torch.cuda.is_available(),'Both registered backend tests require CUDA availability')
    methods={}; total_updates=0
    for method in C.METHODS:
        cross=[parity(method,v,q) for v in (2,3) for q in (('Q0','Q1') if method=='QMF' else ('Q1',))]
        timing={}
        for device in ('cpu','cuda'):
            rows=[benchmark(method,v,device,q,t,directory) for v in (2,3)
                  for q in (('Q0','Q1') if method=='QMF' else ('Q1',)) for t in range(3)]
            timing[device]=rows
            total_updates+=sum(r['optimizer_updates'] for r in rows)
        medians={d:statistics.median(r['seconds'] for r in rows) for d,rows in timing.items()}
        selected='cpu' if medians['cpu']<=medians['cuda'] else 'cuda'
        methods[method]={'device':selected,'precision':'FP32','cross_backend':cross,
                         'timing':timing,'median_end_to_end_seconds':medians,
                         'inference_batch_optimization_enabled':False}
        print('Backend locked',method,selected,medians,flush=True)
        C.save(directory/'progress.json',methods)
    result={'passed':True,'methods':methods,'tolerances':TOLERANCE,
            'sources':[C.desc(C.CODE/p) for p in ('common.py','models.py','preprocessing.py','training.py','backend_gates.py','run_b.py')],
            'predeclared':C.desc(directory/'predeclared.json'),'formal_fits':0,
            'synthetic_optimizer_updates':total_updates,'test_forwards':0,'at':C.now(),
            'claimed_new_kernel_speedup':False,'optimization':'resident feature tensors, existing kernels, no new inference batching'}
    C.save(output,result,immutable=True)
    return result

def run():
    C.check_pause()
    if (C.OUT/'backend_gate.json').exists():
        return _run()
    from .run_b import acquire,release
    owner=acquire(module='tridep_ccml.fusion_eeg_grid_v3.backend_gates',formal=False)
    try:
        return _run()
    except BaseException as error:
        C.save(C.REPORT/'backend_gates'/f'failure_{time.time_ns()}.json',
               {'passed':False,'error':repr(error),'owner':owner,'formal_fits':0,'at':C.now(),
                'unknown_engineering_tail_updates':None,'unknown_engineering_tail_seconds':None})
        raise
    finally:
        release(owner)
if __name__=='__main__': run()

