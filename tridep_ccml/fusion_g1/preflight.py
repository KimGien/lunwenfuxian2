import unittest
from torch.nn import functional as F
from .common import *
from .model import Model
from tridep_ccml.ccml import evidence_dc,ce_heads_loss,single_evidential_loss,ccml_loss

def main():
    lock_check();set_seed(11)
    checks=[]
    for seed in SEEDS:
        base=torch.load(OUT/f'initial/seed_{seed}.pt',weights_only=True)
        for c in CONDITIONS:
            cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
            m=Model(c,seed)
            assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
            assert all(torch.equal(v,base[k]) for k,v in m.state_dict().items())
            assert not list(m.norm.parameters()) and m.norm.eps==1e-5
            assert len(m.modalities)==(3 if c.endswith('-EST') else 2)
            checks.append(f'{seed}/{c}: paired initialization, actual V, normalization, RNG isolation')
    for views in [2,3]:
        logits=torch.tensor([[[.3,-.7],[.2,.8],[1.,2.]],[[.4,1.],[.8,.5],[.1,.2]]],requires_grad=True)[:,:views]
        y=torch.tensor([0,1]);e=F.softplus(logits)
        manual=sum(F.cross_entropy(logits[:,v],y) for v in range(views))
        assert torch.allclose(ce_heads_loss(logits,y),manual)
        m=e.amin(1);con=views*m.sum(-1,keepdim=True)*m.pow(1.25)/m.pow(1.25).sum(-1,keepdim=True)
        final=con+(e-m[:,None]).mean(1); p=(final+1)/(final+1).sum(-1,keepdim=True)
        assert torch.allclose(evidence_dc(e)['p'],p,atol=1e-6)
        for objective in [single_evidential_loss,ccml_loss]:
            loss=objective(e,y,1);grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
            assert torch.isfinite(grad).all() and grad.abs().sum()>0
        checks.append(f'V={views}: summed CE, independent CCML formula, finite EDL/CCML gradients')
    data=Data();ids=data.ctx.folds['fold_1']['train'][:8];x,y=data.batch(ids)
    for c in CONDITIONS:
        model=Model(c,11).cuda().eval()
        with torch.no_grad():
            out,z,n=model(x,features=True)
            assert out.shape==(8,len(model.modalities),2) and torch.isfinite(out).all()
            assert all(float(z[:,i].std(dim=0).norm())>0 for i in range(len(model.modalities)))
        model.train();model.objective(model(x),y,1).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append(c+': real train-only batch, distinct subject features, all parameter gradients finite')
    suite=unittest.defaultTestLoader.loadTestsFromName('tridep_ccml.tests.test_ccml')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    save(OUT/'preflight.json',{'passed':True,'created_utc':now(),'checks':checks,
        'mathematical_unit_tests':result.testsRun,'extra_training_updates':0,'heldout_inference':False,
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu':torch.cuda.get_device_name()},immutable=True)
    print(f'Preflight passed: {len(checks)} grouped checks + {result.testsRun} mathematical tests.',flush=True)

if __name__=='__main__':main()
