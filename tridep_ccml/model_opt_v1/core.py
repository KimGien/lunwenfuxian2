from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tridep_ccml.eeg_e2.common import WORK, read, save, digest, desc, now
from tridep_ccml.fusion_g1.common import Data, orders, state_hash, SEEDS, FOLDS
from tridep_ccml.fusion_g1.model import Model as G1Model
from tridep_ccml.v2.common import set_seed

CODE=Path(__file__).resolve().parent
OUT=WORK/'tridep_ccml/experiments/MODEL-OPT-v1-37-fulltext'
REPORT=WORK/'output/analysis/model_opt_v1_20260930'
APPENDIX=WORK/'output/plans/模型优化第一批执行附表_v1.md'
G1=WORK/'tridep_ccml/experiments/FUSION-G1-37-fulltext'
E2=WORK/'tridep_ccml/experiments/EEG-E2v2-37-fulltext'
CONDITIONS=['M1-global-ST','M1-branch-ST','M1-global-EST','M1-branch-EST','M1-mean-ST','M1-stat-ST']
METRICS=['macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier']

class Model(G1Model):
    def __init__(self,condition,seed):
        assert condition in CONDITIONS
        super().__init__('G-CCML-'+condition.rsplit('-',1)[1],seed)
        self.condition=condition
        if condition in ['M1-mean-ST','M1-stat-ST']:
            old=self.encoders['speech'][0]
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed+4200)
                extra=torch.empty(128,768).uniform_(-1/math.sqrt(768),1/math.sqrt(768))
                layer=nn.Linear(1536,128)
            with torch.no_grad():
                layer.weight.copy_(torch.cat([old.weight,extra],1));layer.bias.copy_(old.bias)
            self.encoders['speech'][0]=layer

    def forward(self,inputs,features=False):
        if self.condition in ['M1-mean-ST','M1-stat-ST']:
            extra=inputs['speech_std'] if self.condition=='M1-stat-ST' else torch.zeros_like(inputs['speech'])
            inputs={**inputs,'speech':torch.cat([inputs['speech'],extra],1)}
        return super().forward(inputs,features)

def groups(model):
    result={}
    for m in model.modalities:
        if m=='eeg':
            result['eeg_encoder']=[p for n,p in model.encoders[m].named_parameters() if not n.startswith('interface.')]
            result['eeg_projection']=list(model.encoders[m].interface.parameters())
        else:result[m+'_projection']=list(model.encoders[m].parameters())
        result[m+'_head']=list(model.heads[m].parameters())
    assert len({id(p) for ps in result.values() for p in ps})==len(list(model.parameters()))
    return result

def l2(tensors):
    # Accumulate diagnostics in float64: float32 reductions over large constant
    # gradients can falsely report a budget violation while the true norm is 1.
    return float(torch.linalg.vector_norm(torch.stack([t.detach().norm(dtype=torch.float64) for t in tensors])))

def clip(model):
    if '-branch-' in model.condition:
        for m in model.modalities:
            ps=list(model.encoders[m].parameters())+list(model.heads[m].parameters())
            torch.nn.utils.clip_grad_norm_(ps,1/math.sqrt(len(model.modalities)),error_if_nonfinite=True)
    else:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    actual=l2([p.grad for p in model.parameters() if p.grad is not None])
    assert actual<=1.00001
    return actual

def matrix_function(a,fn):
    w,v=np.linalg.eigh((a+np.swapaxes(a,-1,-2))/2)
    if fn is not np.exp:assert np.all(w>0), 'Covariance must be SPD'
    return (v*fn(w)[...,None,:])@np.swapaxes(v,-1,-2)

def tangent(covs,reference_log):
    z=matrix_function(covs,np.log)-reference_log
    i,j=np.triu_indices(z.shape[-1])
    return z[:,i,j]*np.where(i==j,1.,math.sqrt(2.))

def lock_check():
    r=read(OUT/'registration.json')
    for d in r['frozen_files']:assert digest(WORK/d['path'])==d['sha256'],d['path']
    return r

def run_dir(condition,seed,fold):return OUT/f'runs/{condition}/seed_{seed}/fold_{fold}'

def load_data():
    d=Data()
    d.inputs['speech_std']=torch.from_numpy(np.load(OUT/'cache/speech_std.npy')).cuda()
    return d
