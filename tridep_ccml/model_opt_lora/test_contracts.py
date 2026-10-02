import unittest
from .core import *

class Contracts(unittest.TestCase):
    def test_real_fit_batch_backward(self):
        import time
        data=Data();ids=data.ctx.folds['fold_1']['train'][:8]
        m=Model('M3-lora-ST',11).cuda().train();before=m.frozen_hash()
        x,y=data.batch(ids,data.ctx.folds['fold_1']['train']);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        loss=m.objective(m(x),y,1);loss.backward()
        self.assertTrue(all(p.grad is None for n,p in m.layer.named_parameters() if not (n.endswith('.A') or n.endswith('.B'))))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
        self.assertGreater(sum(float(p.grad.norm()) for n,p in m.layer.named_parameters() if n.endswith('.B')),0)
        self.assertEqual(before,m.frozen_hash())
        save(OUT/'backward_probe.json',{'passed':True,'fit_ids':ids,'outer_fold':1,'optimizer_updates':0,
            'loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-started,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'frozen_parameters_unchanged':True},immutable=True)
    def test_zero_adapter_exact_and_frozen_gradient(self):
        base=nn.Linear(7,5);layer=LoRALinear(base,11,True,rank=2);x=torch.randn(9,7)
        self.assertTrue(torch.equal(layer(x),base(x)))
        layer(x).square().sum().backward()
        self.assertIsNone(base.weight.grad);self.assertIsNone(base.bias.grad)
        self.assertGreater(float(layer.B.grad.norm()),0)
        self.assertEqual(float(layer.A.grad.norm()),0)
    def test_checkpoint_gradient_equivalence(self):
        c=BertConfig(hidden_size=32,intermediate_size=64,num_attention_heads=4,hidden_dropout_prob=0,attention_probs_dropout_prob=0)
        c._attn_implementation='eager';a=BertLayer(c).requires_grad_(False)
        a.attention.self.query=LoRALinear(a.attention.self.query,12,True,rank=2)
        a.attention.self.value=LoRALinear(a.attention.self.value,13,True,rank=2)
        a.eval();x=torch.randn(3,11,32);mask=torch.zeros(3,1,1,11)
        def run(xx,mm):return a(xx,attention_mask=mm)[0][:,0,:]
        ps=[p for p in a.parameters() if p.requires_grad]
        direct=run(x,mask);gd=torch.autograd.grad(direct.square().sum(),ps)
        reco=checkpoint(run,x,mask,use_reentrant=False,preserve_rng_state=False);gc=torch.autograd.grad(reco.square().sum(),ps)
        self.assertTrue(torch.equal(direct,reco))
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(gd,gc)))
    def test_pairing_and_cache_replay(self):
        data=Data();maxdiff=0
        for seed in SEEDS:
            a,b=Model(CONDITIONS[0],seed).cuda(),Model(CONDITIONS[1],seed).cuda()
            self.assertEqual(state_hash(a.state()),state_hash(b.state()))
            self.assertEqual(sum(p.numel() for p in b.adapter_parameters()),12288)
            self.assertTrue(all(not p.requires_grad for p in a.layer.parameters()))
            self.assertTrue(all((n.endswith('.A') or n.endswith('.B'))==p.requires_grad for n,p in b.layer.named_parameters()))
            ids=data.ids[:8];x,_=data.batch(ids)
            with torch.no_grad():
                a.eval();b.eval();la=a(x);lb=b(x);self.assertTrue(torch.equal(la,lb))
            del a,b
        a=Model(CONDITIONS[0],11).cuda()
        out=infer(a,data,data.ids)
        repeat=infer(a,data,data.ids)
        self.assertTrue(all(torch.equal(out[k],repeat[k]) for k in ['logits','features','normalized','question_cls']))
        for i,sid in enumerate(data.ids):
            old=torch.load(OUT/f'cache/subjects/{sid}.pt',weights_only=True)['full_cls']
            maxdiff=max(maxdiff,float((out['question_cls'][i]-old).abs().max()))
            self.assertTrue(torch.allclose(out['question_cls'][i],old,atol=5e-5,rtol=2e-5))
        save(OUT/'cache_replay.json',{'passed':True,'subjects':37,'question_vectors':1073,'max_abs_difference':maxdiff,
            'rtol':2e-5,'atol':5e-5,'repeated_path_bitwise_exact':True,'formal_updates':0},immutable=True)

def main():
    set_seed(11);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    save(OUT/'tests.json',{'passed':r.wasSuccessful(),'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'formal_updates':0,'created_utc':now()},immutable=True)
    if not r.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
