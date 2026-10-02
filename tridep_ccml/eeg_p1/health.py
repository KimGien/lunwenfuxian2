"""Task-loss gradients and input-dependent representations, using train subjects only."""
from __future__ import annotations
import math
from contextlib import contextmanager
from torch import nn
from torch.nn import functional as F
from .common import *
from tridep_ccml.ccml import ccml_loss


@contextmanager
def unchanged(model):
    state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    modes = {name:m.training for name,m in model.named_modules()}
    gradients = {name:None if p.grad is None else p.grad.detach().clone() for name,p in model.named_parameters()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        yield
    finally:
        model.load_state_dict(state,strict=True)
        for name,m in model.named_modules():
            m.training = modes[name]
        for name,p in model.named_parameters():
            p.grad = gradients[name]
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)


def weight_modules(model):
    # Exclude classification heads: changing only the prior is not encoder learning.
    result = {}
    for name,module in model.eeg.named_modules():
        if isinstance(module,(nn.Linear,nn.Conv1d,nn.Conv2d)):
            if model.condition=='U0' and name=='classifier.5':
                continue
            result['eeg.'+name+'.weight'] = module.weight
    require(bool(result),'No encoder weights registered')
    return result


def gradient_norms(model):
    values = {}
    for name,p in weight_modules(model).items():
        require(p.grad is not None and torch.isfinite(p.grad).all().item(), 'Missing/nonfinite encoder gradient: '+name)
        values[name] = float(p.grad.detach().norm())
    return values


def check_train_ids(data,ids):
    require(len(ids)==len(set(ids)) and len(ids)>=2 and set(ids)<=set(data.splits['train']),
            'Health/fit probes must use distinct train subjects only')


def eeg_representation(model,inputs):
    z = model.representations(inputs)
    require(torch.isfinite(z).all().item(),'Nonfinite encoder representation')
    return z[:,0] if model.joint else z


def relative_difference(a,b):
    value = float((a-b).double().norm()/a.double().norm().clamp_min(1e-12))
    require(math.isfinite(value),'Nonfinite input response')
    return value


def audit(model,data,ids=None,epoch=1):
    ids = list(ids or data.splits['train'])
    check_train_ids(data,ids)
    before = state_digest(model.state_dict())
    grads = {name:0. for name in weight_modules(model)}
    with unchanged(model):
        model.train()
        for start in range(0,len(ids),8):
            inputs,labels = data.batch(ids[start:start+8],model.condition)
            model.zero_grad(set_to_none=True)
            output = model(inputs)
            loss = ccml_loss(output,labels,epoch,**LOSS) if model.joint else F.cross_entropy(output,labels)
            loss.backward()
            for name,value in gradient_norms(model).items():
                grads[name] = max(grads[name],value)
        model.eval()
        actual,zeroed,regional = [],[],{}
        with torch.inference_mode():
            for start in range(0,len(ids),8):
                inputs,_ = data.batch(ids[start:start+8],model.condition)
                actual.append(eeg_representation(model,inputs))
                zeroed.append(eeg_representation(model,{**inputs,'eeg':torch.zeros_like(inputs['eeg'])}))
                if hasattr(model.eeg,'region_indices'):
                    for region,indices in enumerate(model.eeg.region_indices):
                        altered = inputs['eeg'].clone()
                        altered[:,:,indices,:] = 0
                        regional.setdefault(str(region),[]).append(eeg_representation(model,{**inputs,'eeg':altered}))
            z,z0 = torch.cat(actual),torch.cat(zeroed)
            std = float(z.double().std(0,unbiased=False).square().mean().sqrt())
            rms = float(z.double().square().mean().sqrt())
            relative_std = std/max(rms,1e-12)
            response = relative_difference(z,z0)
            responses = {k:relative_difference(z,torch.cat(v)) for k,v in regional.items()}
    require(before==state_digest(model.state_dict()),'Health check mutated model/BN')
    failures = ['zero/tiny gradient: '+k for k,v in grads.items() if v<=GATES['gradient_l2_min']]
    if std<=GATES['representation_std_rms_min']:
        failures.append('constant/near-constant subject representation')
    if relative_std<=GATES['representation_relative_std_min']:
        failures.append('tiny relative representation variance')
    if response<=GATES['zero_input_response_min']:
        failures.append('insensitive to zeroed EEG input')
    failures += ['insensitive region: '+k for k,v in responses.items() if v<=GATES['region_response_min']]
    return {'passed':not failures,'failures':failures,'fit_subject_ids':ids,'epoch':epoch,
        'gradient_l2_max_over_train_batches':grads,'representation_std_rms':std,
        'representation_rms':rms,'representation_relative_std':relative_std,
        'zero_input_relative_response':response,'region_relative_responses':responses,
        'model_and_rng_restored':True,'thresholds':GATES}


def require_health(result):
    require(result['passed'],'Effective-learning gate failed: '+'; '.join(result['failures']))
