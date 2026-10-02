from torch import nn
from torch.nn import functional as F
from .common import *
from tridep_ccml.eeg_replacement.models import EEGNet,max_norm_
from tridep_ccml.ccml import ce_heads_loss,single_evidential_loss,ccml_loss

OFFSETS={'eeg':1000,'speech':2000,'text':3000}

class Model(nn.Module):
    def __init__(self,condition,seed):
        super().__init__()
        assert condition in CONDITIONS
        self.condition=condition; self.family=condition.split('-')[1]
        self.modalities=['eeg','speech','text'] if condition.endswith('-EST') else ['speech','text']
        self.encoders=nn.ModuleDict();self.heads=nn.ModuleDict()
        # CPU generators only: construction never changes CUDA RNG or pairing.
        for m in self.modalities:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+OFFSETS[m])
                self.encoders[m]=EEGNet() if m=='eeg' else nn.Sequential(nn.Linear(768,128),nn.ReLU())
                self.heads[m]=nn.Linear(128,2)
        self.norm=nn.LayerNorm(128,eps=1e-5,elementwise_affine=False)
    def forward(self,inputs,features=False):
        z=torch.stack([self.encoders[m](inputs[m]) for m in self.modalities],1)
        n=self.norm(z)
        logits=torch.stack([self.heads[m](n[:,i]) for i,m in enumerate(self.modalities)],1)
        return (logits,z,n) if features else logits
    def optimizer(self):
        return torch.optim.Adamax([{'params':self.encoders.parameters(),'lr':1e-4},
                                  {'params':self.heads.parameters(),'lr':1e-3}],weight_decay=0)
    def constrain(self):
        if 'eeg' in self.encoders: max_norm_(self.encoders['eeg'].spatial.weight,1.)
    def objective(self,logits,y,epoch):
        if self.family=='CE': return ce_heads_loss(logits,y)
        e=F.softplus(logits)
        if self.family=='EDL':return single_evidential_loss(e,y,epoch,anneal_epochs=50)
        return ccml_loss(e,y,epoch,anneal_epochs=50,rho=1.25,delta=1.,gamma=1.)

def gradient_norm(params):
    return float(torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None])))
