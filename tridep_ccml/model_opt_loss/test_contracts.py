import unittest
from .core import *

class Contracts(unittest.TestCase):
    def test_exact_sum_and_paired_initialization(self):
        for seed in SEEDS:
            for mods in ['ST','EST']:
                a,b=Model('M2-sum-'+mods,seed),Model('M2-mean-'+mods,seed)
                old=G1Model('G-CCML-'+mods,seed)
                self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
                self.assertEqual(state_hash(a.state_dict()),state_hash(old.state_dict()))
                logits=torch.randn(5,len(a.modalities),2);y=torch.tensor([0,1,0,1,0])
                self.assertTrue(torch.equal(a.objective(logits,y,31),old.objective(logits,y,31)))

    def test_formula_and_gradients(self):
        gen=torch.Generator().manual_seed(91)
        for views in [2,3]:
            e=torch.rand(5,views,2,generator=gen,dtype=torch.float64,requires_grad=True)+.02
            y=torch.tensor([0,1,0,1,0])
            for epoch in [1,30,60]:
                v,c,r,p=terms(e,y,epoch)
                total=ccml_loss(e,y,epoch);balanced=mean_loss(e,y,epoch)
                single=v.sum(1).mean();fusion=(c+r+p).mean()
                self.assertTrue(torch.allclose(total,single+fusion,atol=1e-12,rtol=0))
                self.assertTrue(torch.allclose(balanced,single/views+fusion,atol=1e-12,rtol=0))
                self.assertFalse(torch.allclose(balanced,total/views))
                gs=torch.autograd.grad(single,e,retain_graph=True)[0]
                gf=torch.autograd.grad(fusion,e,retain_graph=True)[0]
                gb=torch.autograd.grad(balanced,e,retain_graph=True)[0]
                self.assertTrue(torch.allclose(gb,gs/views+gf,atol=1e-12,rtol=1e-12))

    def test_one_view_identity(self):
        e=torch.tensor([[[.3,.7]],[[1.,2.]]],dtype=torch.float64)
        y=torch.tensor([0,1])
        self.assertTrue(torch.equal(ccml_loss(e,y,60),mean_loss(e,y,60)))

    def test_zero_evidence_and_finite_gradients(self):
        for views in [2,3]:
            e=torch.zeros(4,views,2,requires_grad=True);y=torch.tensor([0,1,1,0])
            loss=mean_loss(e,y,60);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(e.grad).all())

def main():
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    save(OUT/'tests.json',{'passed':r.wasSuccessful(),'tests':r.testsRun,'errors':len(r.errors),'failures':len(r.failures),
        'new_formal_fits':0,'created_utc':now()},immutable=True)
    if not r.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
