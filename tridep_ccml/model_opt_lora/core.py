from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer
from tridep_ccml.model_opt_loss.core import (WORK,read,save,digest,desc,now,np,SEEDS,FOLDS,G1,set_seed,state_hash,
    METRICS,orders,Model as LossModel)
from tridep_ccml.records import RecordContext

CODE=Path(__file__).resolve().parent
OUT=WORK/'tridep_ccml/experiments/MODEL-OPT-v1-LORA-37-fulltext'
REPORT=WORK/'output/analysis/model_opt_lora_20261001'
APPENDIX=WORK/'output/plans/模型优化第三批_MacBERT有限微调执行附表_v1.md'
BACKBONE=WORK/'S00/models/macbert'
CONDITIONS=['M3-frozen-ST','M3-lora-ST']

def config():
    c=BertConfig.from_pretrained(BACKBONE,local_files_only=True)
    c._attn_implementation='eager'
    return c

class LoRALinear(nn.Module):
    def __init__(self,base,seed,learned,rank=4):
        super().__init__();self.base=base.requires_grad_(False);self.rank=rank
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.A=nn.Parameter(torch.empty(rank,base.in_features).uniform_(-1/math.sqrt(base.in_features),1/math.sqrt(base.in_features)),requires_grad=learned)
            self.B=nn.Parameter(torch.zeros(base.out_features,rank),requires_grad=learned)
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.A),self.B)

class Model(nn.Module):
    def __init__(self,condition,seed):
        super().__init__();assert condition in CONDITIONS
        self.condition=condition;self.learned=condition=='M3-lora-ST';self.modalities=['speech','text']
        self.top=LossModel('M2-mean-ST',seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.layer=BertLayer(config())
        self.layer.load_state_dict(torch.load(OUT/'cache/last_layer.pt',map_location='cpu',weights_only=True))
        self.layer.requires_grad_(False)
        for name,offset in [('query',5100),('value',5200)]:
            setattr(self.layer.attention.self,name,LoRALinear(getattr(self.layer.attention.self,name),seed+offset,self.learned))
        self.layer.eval()
    def train(self,mode=True):
        super().train(mode);self.layer.eval();return self
    def adapter_parameters(self):return [p for n,p in self.layer.named_parameters() if n.endswith('.A') or n.endswith('.B')]
    def state(self):
        return {'top.'+k:v for k,v in self.top.state_dict().items()}|{'adapter.'+k:v for k,v in self.layer.state_dict().items() if k.endswith('.A') or k.endswith('.B')}
    def load(self,s):
        self.top.load_state_dict({k[4:]:v for k,v in s.items() if k.startswith('top.')},strict=True)
        with torch.no_grad():
            named=dict(self.layer.named_parameters())
            for k,v in s.items():
                if k.startswith('adapter.'):named[k[8:]].copy_(v)
    def frozen_hash(self):return state_hash({k:v for k,v in self.layer.state_dict().items() if not (k.endswith('.A') or k.endswith('.B'))})
    def text(self,questions,recompute=True):
        order=sorted(range(len(questions)),key=lambda i:(len(questions[i]),i));outputs=[]
        for start in range(0,len(order),32):
            ix=order[start:start+32];lengths=[len(questions[i]) for i in ix]
            h=nn.utils.rnn.pad_sequence([questions[i] for i in ix],batch_first=True)
            valid=torch.arange(h.shape[1],device=h.device)[None,:]<torch.tensor(lengths,device=h.device)[:,None]
            mask=(~valid).to(h.dtype)[:,None,None,:]*torch.finfo(h.dtype).min
            def run(a,b):return self.layer(a,attention_mask=b)[0][:,0,:]
            out=checkpoint(run,h,mask,use_reentrant=False,preserve_rng_state=False) if recompute and self.learned and self.training and torch.is_grad_enabled() else run(h,mask)
            outputs.append(out)
        sorted_cls=torch.cat(outputs);restore=torch.argsort(torch.tensor(order,device=sorted_cls.device))
        return sorted_cls[restore].reshape(-1,29,768)
    def forward(self,x,features=False):
        question_cls=self.text(x['questions']);pooled=question_cls.mean(1)
        logits,z,n=self.top({'speech':x['speech'],'text':pooled},features=True)
        return (logits,z,n,question_cls) if features else logits
    def objective(self,logits,y,epoch):return self.top.objective(logits,y,epoch)
    def optimizer(self):
        ps=[{'params':self.top.encoders.parameters(),'lr':1e-4},{'params':self.top.heads.parameters(),'lr':1e-3}]
        if self.learned:ps.append({'params':self.adapter_parameters(),'lr':1e-5})
        return torch.optim.Adamax(ps,weight_decay=0)

class Data:
    def __init__(self):
        self.ctx=RecordContext();self.ids=self.ctx.ids;self.index={s:i for i,s in enumerate(self.ids)}
        path=WORK/read(G1/'input_manifest.json')['arrays']['speech']['path']
        self.speech=torch.from_numpy(np.load(path)).cuda()
        self.questions={s:[v.cuda() for v in torch.load(OUT/f'cache/subjects/{s}.pt',weights_only=True)['hidden']] for s in self.ids}
        self.labels=self.ctx.labels
    def batch(self,ids,fit_ids=None):
        if fit_ids is not None:assert set(ids)<=set(fit_ids)
        return {'speech':self.speech[[self.index[s] for s in ids]],'questions':[q for s in ids for q in self.questions[s]]},torch.tensor([self.labels[s] for s in ids],device='cuda')

def lock_check():
    cfg=read(OUT/'registration.json')
    for d in cfg['frozen_files']:assert digest(WORK/d['path'])==d['sha256'],d['path']
    return cfg

def stem(c,s,f):return OUT/f'runs/{c}/seed_{s}/fold_{f}'

def module_groups(m):
    return {**{n+'_projection':list(m.top.encoders[n].parameters()) for n in m.modalities},
        **{n+'_head':list(m.top.heads[n].parameters()) for n in m.modalities},
        **({'text_lora':m.adapter_parameters()} if m.learned else {})}

def norm(ps):return float(torch.stack([p.detach().norm(dtype=torch.float64) for p in ps]).norm())

@torch.no_grad()
def infer(m,data,ids):
    m.eval();ls=[];zs=[];ns=[];qs=[]
    for start in range(0,len(ids),8):
        x,_=data.batch(ids[start:start+8]);l,z,n,q=m(x,features=True)
        ls.append(l.cpu());zs.append(z.cpu());ns.append(n.cpu());qs.append(q.cpu())
    return {'subject_ids':ids,'logits':torch.cat(ls),'features':torch.cat(zs),'normalized':torch.cat(ns),'question_cls':torch.cat(qs)}
