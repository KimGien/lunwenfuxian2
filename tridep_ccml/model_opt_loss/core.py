from __future__ import annotations
from pathlib import Path
from torch.nn import functional as F
from tridep_ccml.model_opt_v1.core import (WORK,read,save,digest,desc,now,np,torch,Data,orders,state_hash,
    SEEDS,FOLDS,G1,G1Model,set_seed,groups,l2,METRICS)
from tridep_ccml.ccml import ccml_loss,evidence_dc,_loss_terms,expected_cross_entropy

CODE=Path(__file__).resolve().parent
OUT=WORK/'tridep_ccml/experiments/MODEL-OPT-v1-LOSS-37-fulltext'
PARENT=WORK/'tridep_ccml/experiments/MODEL-OPT-v1-37-fulltext'
REPORT=WORK/'output/analysis/model_opt_loss_20260930'
APPENDIX=WORK/'output/plans/模型优化第二批_CCML损失块平衡执行附表_v1.md'
CONDITIONS=['M2-sum-ST','M2-mean-ST','M2-sum-EST','M2-mean-EST']
NEW=['M2-mean-ST','M2-mean-EST']

def terms(e,y,epoch):
    f=evidence_dc(e,1.25)
    vd,vr=_loss_terms(e,y,epoch,50)
    cd,cr=_loss_terms(f['e_con'],y,epoch,50)
    cmp=expected_cross_entropy(f['e_cmp']+1,F.one_hot(y,e.shape[-1]).to(e.dtype))
    return vd+vr,cd,cr,cmp

def mean_loss(e,y,epoch):
    v,c,r,p=terms(e,y,epoch)
    loss=(v.mean(1)+c+r+p).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite balanced loss')
    return loss

class Model(G1Model):
    def __init__(self,condition,seed):
        assert condition in CONDITIONS
        super().__init__('G-CCML-'+condition.rsplit('-',1)[1],seed)
        self.condition=condition
    def objective(self,logits,y,epoch):
        e=F.softplus(logits)
        return mean_loss(e,y,epoch) if '-mean-' in self.condition else ccml_loss(e,y,epoch)

def stem(condition,seed,fold):
    if '-sum-' in condition:
        return PARENT/f'runs/M1-global-{condition.rsplit("-",1)[1]}/seed_{seed}/fold_{fold}'
    return OUT/f'runs/{condition}/seed_{seed}/fold_{fold}'

def lock_check():
    cfg=read(OUT/'registration.json')
    for d in cfg['frozen_files']:assert digest(WORK/d['path'])==d['sha256'],d['path']
    return cfg

def block_probe(model,data,ids,epoch):
    assert len(ids)==8
    model.eval();before=state_hash(model.state_dict())
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    x,y=data.batch(ids);e=F.softplus(model(x));v,c,r,p=terms(e,y,epoch)
    objectives={'single':v.sum(1).mean(),'consistent':(c+r).mean(),'complementary':p.mean()}
    params=list(model.parameters());gmap=groups(model);index={id(p):i for i,p in enumerate(params)}
    grads={name:torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) for name,loss in objectives.items()}
    grads={k:[torch.zeros_like(p) if g is None else g for p,g in zip(params,gs)] for k,gs in grads.items()}
    details={};views=len(model.modalities)
    for name,ps in gmap.items():
        vectors={k:torch.cat([gs[index[id(p)]].detach().flatten().double() for p in ps]) for k,gs in grads.items()}
        a=vectors['single'];b=vectors['consistent']+vectors['complementary']
        an,bn=float(a.norm()),float(b.norm());cos=float(torch.dot(a,b)/(a.norm()*b.norm())) if an*bn>1e-30 else None
        details[name]={'single_sum_l2':an,'single_mean_l2':an/views,'consistent_l2':float(vectors['consistent'].norm()),
            'complementary_l2':float(vectors['complementary'].norm()),'fusion_block_l2':bn,
            'single_sum_to_fusion':an/bn if bn>1e-15 else None,
            'single_mean_to_fusion':an/views/bn if bn>1e-15 else None,
            'single_fusion_cosine':cos}
    assert before==state_hash(model.state_dict())
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    return {'subject_ids':ids,'epoch_for_annealing':epoch,'mode':'eval; no update; fixed train batch',
        'state_sha256':before,'state_and_rng_unchanged':True,'views':views,
        'losses':{**{k:float(val.detach()) for k,val in objectives.items()},
            'single_mean':float(v.mean(1).mean().detach()),'total_sum':float(ccml_loss(e,y,epoch).detach()),
            'total_mean':float(mean_loss(e,y,epoch).detach())},'modules':details}
