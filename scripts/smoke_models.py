"""Synthetic forward/backward checks, reproducibility and checkpoint coverage."""
import json
import os
from pathlib import Path
import random
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream-dev'))
import numpy as np
import torch
from lib.models import models
from prepare import save

def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def main():
    cfg=json.loads((ROOT/'configs/b0.json').read_text())
    device='cuda' if torch.cuda.is_available() else 'cpu'
    result={'synthetic_only':True,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':device,'models':{}}
    if device=='cuda': result['gpu']=torch.cuda.get_device_name(0)
    for key,c in cfg['branches'].items():
        seed(11); model=getattr(models,c['model_class'])(**c['model_kwargs']).to(device)
        x=torch.randn(2,*c['shape'],device=device); y=torch.tensor([0,1],device=device)
        model.eval()
        with torch.no_grad(): first=model(x); second=model(x)
        assert first.shape==(2,2) and torch.isfinite(first).all()
        assert torch.equal(first,second)
        seed(11); repeat=getattr(models,c['model_class'])(**c['model_kwargs']).to(device).eval()
        with torch.no_grad(): assert torch.equal(first,repeat(x))
        del repeat
        model.train(); opt=torch.optim.Adamax(model.parameters(),lr=c['lr'])
        loss=torch.nn.functional.cross_entropy(model(x),y); loss.backward(); opt.step()
        assert torch.isfinite(loss)
        result['models'][key]={'forward_backward_passed':True,'repeat_equal':True,'shape':list(first.shape),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        del model,opt,x,y
    # Check actual Mumtaz weight tensor mapping, not just file presence.
    from lib.models.cbramod.cbramod import CBraMod
    backbone=CBraMod(in_dim=200,out_dim=200,d_model=200,dim_feedforward=800,seq_len=30,n_layer=12,nhead=8)
    path=ROOT/'upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth'
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    ckpt=ckpt.get('state_dict',ckpt)
    ckpt={k.replace('backbone.',''):v for k,v in ckpt.items()}
    own=backbone.state_dict()
    matched=[k for k,v in own.items() if k in ckpt and tuple(v.shape)==tuple(ckpt[k].shape)]
    missing=[k for k in own if k not in matched]
    result['cbramod_checkpoint']={'matched_keys':len(matched),'total_keys':len(own),'missing_or_mismatched':missing,'unexpected_keys':[k for k in ckpt if k not in own]}
    backbone.load_state_dict({k:ckpt[k] for k in matched},strict=False)
    backbone.proj_out=torch.nn.Identity(); backbone.eval()
    with torch.no_grad(): out=backbone(torch.randn(1,19,5,200))
    assert torch.isfinite(out).all()
    result['cbramod_checkpoint']['output_shape']=list(out.shape)
    save(ROOT/'reports/model_smoke.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
