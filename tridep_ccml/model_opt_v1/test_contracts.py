import unittest
from .core import *

class Contracts(unittest.TestCase):
    def test_g1_initialization(self):
        for seed in SEEDS:
            for modalities in ['ST','EST']:
                old=G1Model('G-CCML-'+modalities,seed)
                for policy in ['global','branch']:
                    new=Model(f'M1-{policy}-{modalities}',seed)
                    self.assertEqual(state_hash(old.state_dict()),state_hash(new.state_dict()))

    def test_statistics_pairing(self):
        for seed in SEEDS:
            a,b=Model('M1-mean-ST',seed),Model('M1-stat-ST',seed)
            self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
            x={'speech':torch.randn(3,768),'speech_std':torch.zeros(3,768),'text':torch.randn(3,768)}
            self.assertTrue(torch.equal(a(x),b(x)))
            x['speech_std']=torch.ones(3,768)
            self.assertFalse(torch.equal(a(x),b(x)))

    def test_clip_budget_and_module_coverage(self):
        for modalities in ['ST','EST']:
            for policy in ['global','branch']:
                model=Model(f'M1-{policy}-{modalities}',11)
                groups(model)
                for p in model.parameters():p.grad=torch.full_like(p,10.)
                self.assertLessEqual(clip(model),1.000001)
                if policy=='branch':
                    for m in model.modalities:
                        ps=list(model.encoders[m].parameters())+list(model.heads[m].parameters())
                        self.assertAlmostEqual(l2([p.grad for p in ps]),1/math.sqrt(len(model.modalities)),places=6)

    def test_adamax_update_is_not_gradient_scale(self):
        a,b=nn.Parameter(torch.tensor([1.,2.])),nn.Parameter(torch.tensor([1.,2.]))
        oa,ob=torch.optim.Adamax([a],lr=.01),torch.optim.Adamax([b],lr=.01)
        for _ in range(3):
            a.grad=torch.tensor([3.,-4.]);b.grad=torch.tensor([.3,-.4]);oa.step();ob.step()
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=0))

    def test_matrix_log_roundtrip_and_tangent_norm(self):
        rng=np.random.default_rng(3);a=rng.normal(size=(5,4,4));cov=a@a.transpose(0,2,1)+np.eye(4)
        logs=matrix_function(cov,np.log)
        self.assertTrue(np.allclose(matrix_function(logs,np.exp),cov,rtol=1e-12,atol=1e-12))
        ref=logs[:3].mean(0);z=tangent(cov,ref)
        self.assertEqual(z.shape,(5,10))
        self.assertTrue(np.allclose(np.sum(z*z,1),np.sum((logs-ref)**2,(1,2))))
        self.assertTrue(np.allclose(z[:3].mean(0),0,atol=1e-14))
        changed=cov.copy();changed[3:]*=100
        self.assertTrue(np.array_equal(matrix_function(changed[:3],np.log).mean(0),ref))

    def test_subject_pooling(self):
        x=np.array([[[1.,2.],[3.,6.],[5.,10.]]])
        self.assertTrue(np.allclose(x.std(1,ddof=0),np.sqrt([[8/3,32/3]])))
        self.assertTrue(np.allclose(x.std(1,ddof=0),(x+100).std(1,ddof=0)))

def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    save(OUT/'tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'formal_training_updates':0,'created_utc':now()},immutable=True)
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
