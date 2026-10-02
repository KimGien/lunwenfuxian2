from __future__ import annotations
import math
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch import nn
from .common import *
from tridep_ccml.eeg_replacement.models import R1Model

def subject_standardize(raw):
    x=np.asarray(raw,dtype=np.float64)
    require(x.ndim==4 and np.isfinite(x).all(),'Invalid per-subject signal')
    mean=x.mean((1,3),keepdims=True)
    std=x.std((1,3),keepdims=True)
    require((std>1e-12).all(),'Constant channel in subject standardization')
    return ((x-mean)/std).astype(np.float32),mean.squeeze((1,3)),std.squeeze((1,3))

def make_model(condition,seed,arm):
    model=R1Model(condition,seed)
    if arm=='G':
        for name,n in [('bn1',4),('bn2',8),('bn3',8)]:
            old=getattr(model.eeg,name)
            layer=nn.GroupNorm(2,n,eps=old.eps,affine=True)
            with torch.no_grad():
                layer.weight.copy_(old.weight)
                layer.bias.copy_(old.bias)
            setattr(model.eeg,name,layer)
    return model

@torch.no_grad()
def recompute_bn(model,batches,allowed_ids):
    """Sequential exact population moments, weighted by activation elements."""
    require(not model.training,'BN recomputation requires dropout disabled')
    before=named_parameters_digest(model)
    reports=[]
    for name,layer in model.named_modules():
        if not isinstance(layer,nn.modules.batchnorm._BatchNorm):
            continue
        sums,squares,count=None,None,0
        def collect(module,args):
            nonlocal sums,squares,count
            x=args[0].double()
            dims=(0,)+tuple(range(2,x.ndim))
            a=x.sum(dims); b=x.square().sum(dims)
            sums=a if sums is None else sums+a
            squares=b if squares is None else squares+b
            count+=x.numel()//x.shape[1]
        handle=layer.register_forward_pre_hook(collect)
        seen=[]
        try:
            for ids,x in batches():
                require(set(ids)<=set(allowed_ids),'Non-train subject in BN fit')
                seen.extend(ids)
                model(x)
        finally:
            handle.remove()
        require(sorted(seen)==sorted(allowed_ids),'Incomplete/repeated BN fit identities')
        mean=sums/count
        variance=squares/count-mean.square()
        require(torch.isfinite(variance).all() and (variance>=-1e-10).all(),'Invalid BN variance')
        variance=variance.clamp_min(0)
        old_mean=layer.running_mean.clone(); old_var=layer.running_var.clone()
        layer.running_mean.copy_(mean)
        layer.running_var.copy_(variance)
        reports.append({'layer':name,'count_per_channel':count,'mean':mean.cpu().tolist(),
            'population_variance':variance.cpu().tolist(),
            'old_mean':old_mean.cpu().tolist(),'old_variance':old_var.cpu().tolist()})
    require(before==named_parameters_digest(model),'BN update changed learned parameters')
    return {'fit_subject_ids':list(allowed_ids),'layers':reports,'learned_parameters_unchanged':True,
            'dropout_disabled':True,'sequential_population_moments':True}

def inner_splits(ids,labels,fold):
    splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=93000+fold)
    result=[]
    for i,(fit,cal) in enumerate(splitter.split(ids,[labels[s] for s in ids]),1):
        result.append({'inner':i,'fit':[ids[j] for j in fit],'calibration':[ids[j] for j in cal]})
    require(sorted(s for row in result for s in row['calibration'])==sorted(ids),'OOF coverage failed')
    return result

def validate_oof(rows,train_ids):
    require(sorted(r['subject_id'] for r in rows)==sorted(train_ids),'Calibration OOF IDs mismatch')
    for r in rows:
        require(r['subject_id'] not in r['fit_ids'] and set(r['fit_ids'])<set(train_ids),
                'Calibration model leakage')
        require(r['label'] in [0,1] and np.isfinite(r['p']) and 0<r['p']<1,'Invalid calibration row')

def fit_sigmoid(rows,train_ids):
    validate_oof(rows,train_ids)
    y=np.array([r['label'] for r in rows],dtype=float)
    x=logit(np.clip([r['p'] for r in rows],1e-7,1-1e-7))
    def objective(v):
        a,b=v; z=a*x+b; error=expit(z)-y
        loss=np.mean(np.logaddexp(0,z)-y*z)+.005*((a-1)**2+b*b)
        jac=np.array([(error*x).mean()+.01*(a-1),error.mean()+.01*b])
        return loss,jac
    result=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.01,10),(-10,10)],
                    options={'ftol':1e-13,'gtol':1e-9,'maxiter':1000})
    require(result.success and np.isfinite(result.fun),'Calibration optimizer failed')
    return {'a':float(result.x[0]),'b':float(result.x[1]),'objective':float(result.fun),
            'fit_subject_ids':list(train_ids),'positive_slope':True,'iterations':int(result.nit)}

def apply_sigmoid(p,params):
    return expit(params['a']*logit(np.clip(p,1e-7,1-1e-7))+params['b'])

def fit_threshold(rows,train_ids):
    validate_oof(rows,train_ids)
    p=np.array([r['p'] for r in rows]); y=np.array([r['label'] for r in rows])
    unique=np.unique(p)
    candidates=np.unique(np.r_[0.,.5,1.,(unique[1:]+unique[:-1])/2])
    scores=[float(f1_score(y,p>t,labels=[0,1],average='macro',zero_division=0)) for t in candidates]
    best=max(scores)
    chosen=min([float(t) for t,s in zip(candidates,scores) if abs(s-best)<1e-12],key=lambda t:(abs(t-.5),t))
    return {'threshold':chosen,'fit_subject_ids':list(train_ids),'oof_fit_macro_f1':best,
            'candidate_count':len(candidates),'optimistic_fit_score_not_performance':True}
