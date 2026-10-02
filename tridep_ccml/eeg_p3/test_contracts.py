import copy
import unittest
from .methods import *

class Contracts(unittest.TestCase):
    def test_subject_isolation(self):
        x=np.random.default_rng(8).normal(size=(4,3,2,10))
        a,_,_=subject_standardize(x)
        y=x.copy();y[3]*=99
        b,_,_=subject_standardize(y)
        np.testing.assert_array_equal(a[:3],b[:3])
        c,_,_=subject_standardize(8*x+3)
        np.testing.assert_allclose(a,c,atol=1e-6)
        with self.assertRaises(ValueError): subject_standardize(np.zeros_like(x))

    def test_bn_population_and_partition(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.bn=nn.BatchNorm2d(2);self.drop=nn.Dropout(.5);self.bn2=nn.BatchNorm2d(2)
            def forward(self,x): return self.bn2(self.drop(torch.relu(self.bn(x))))
        x=torch.arange(120,dtype=torch.float32).reshape(5,2,3,4)/9
        model=Tiny().eval();other=copy.deepcopy(model)
        ids=list(range(5))
        recompute_bn(model,lambda:iter([(ids,x)]),ids)
        recompute_bn(other,lambda:iter([(ids[:2],x[:2]),(ids[2:],x[2:])]),ids)
        torch.testing.assert_close(model.bn.running_mean,x.mean((0,2,3)))
        torch.testing.assert_close(model.bn.running_var,x.var((0,2,3),unbiased=False))
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,other.state_dict()[k])
        with self.assertRaises(ValueError): recompute_bn(model,lambda:iter([([99],x[:1])]),ids)

    def test_gn_batch_independence(self):
        layer=nn.GroupNorm(2,4,eps=1e-3)
        x=torch.randn(5,4,3,6)
        torch.testing.assert_close(layer(x)[:1],layer(x[:1]))

    def test_calibration_identity_and_monotonicity(self):
        ids=list('abcdef')
        rows=[{'subject_id':s,'label':i%2,'p':.15+.1*i,'fit_ids':[t for t in ids if t!=s]} for i,s in enumerate(ids)]
        params=fit_sigmoid(rows,ids)
        q=apply_sigmoid(np.linspace(.01,.99,20),params)
        self.assertTrue(np.all(np.diff(q)>0))
        bad=copy.deepcopy(rows);bad[0]['fit_ids'].append(ids[0])
        with self.assertRaises(ValueError): fit_sigmoid(bad,ids)
        with self.assertRaises(ValueError): fit_sigmoid(rows,ids+['heldout'])

    def test_threshold_tie(self):
        ids=list('abcd');ps=[.1,.2,.8,.9]
        rows=[{'subject_id':s,'label':int(i>1),'p':ps[i],'fit_ids':[t for t in ids if t!=s]} for i,s in enumerate(ids)]
        self.assertEqual(fit_threshold(rows,ids)['threshold'],.5)

    def test_inner_oof(self):
        ids=[str(i) for i in range(26)]; labels={s:int(s)%2 for s in ids}
        splits=inner_splits(ids,labels,1)
        self.assertEqual(sorted(s for r in splits for s in r['calibration']),sorted(ids))
        for r in splits: self.assertFalse(set(r['fit'])&set(r['calibration']))

if __name__=='__main__': unittest.main()
