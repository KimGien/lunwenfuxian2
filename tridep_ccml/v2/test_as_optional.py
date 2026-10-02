import unittest
import torch
from torch import nn
from torch.nn import functional as F
from . import as_optional as a
from .common import set_seed, LOSS
from tridep_ccml.ccml import ccml_loss
from tridep_ccml.train import state_digest


class ASOptionalTests(unittest.TestCase):
    def setUp(self):
        set_seed(11)
        self.data={'initial':{f'{m}.0.{k}':v for m,d in a.ap.DIMENSIONS.items() for k,v in nn.Linear(d,128).state_dict().items()}}
        self.x={m:torch.randn(4,d) for m,d in a.ap.DIMENSIONS.items()}

    def test_only_p_opens_projections_and_neither_normalizes(self):
        for v in a.VARIANTS:
            model=a.make_model(self.data,11,v)
            self.assertEqual(sum(p.numel() for p in model.parameters()),a.VARIANTS[v]['parameters'])
            self.assertIsInstance(model.normalizer,nn.Identity)
            self.assertTrue(all(not m.training for m in model.modules()))
        self.assertEqual(state_digest(a.make_model(self.data,11,'P').projections.state_dict()),state_digest(self.data['initial']))

    def test_p_and_frozen_head_start_with_identical_logits(self):
        p=a.make_model(self.data,11,'P');r=a.make_model(self.data,11,'R')
        self.assertTrue(torch.equal(p(self.x),r(p.representations(self.x))))

    def test_decay_applies_all_head_weights_and_bias_only_in_r(self):
        for variant in a.VARIANTS:
            model=a.make_model(self.data,11,variant); opt=a.optimizer_for(model,variant)
            expected=set(map(id,model.parameters()))
            actual={id(p) for group in opt.param_groups for p in group['params']}
            self.assertEqual(expected,actual)
            self.assertTrue(all(g['weight_decay']==a.VARIANTS[variant]['weight_decay'] for g in opt.param_groups))
            before=[p.detach().clone() for p in model.parameters()]
            for p in model.parameters():p.grad=torch.zeros_like(p)
            opt.step()
            for p,old in zip(model.parameters(),before):
                if variant=='R':self.assertLess(float(p.norm()),float(old.norm()))
                else:self.assertTrue(torch.equal(p,old))

    def test_p_has_finite_gradients_and_updates_every_projection(self):
        model=a.make_model(self.data,11,'P');opt=a.optimizer_for(model,'P')
        old={m:state_digest(model.projections[m].state_dict()) for m in a.MODALITIES}
        loss=ccml_loss(F.softplus(model(self.x)),torch.tensor([0,1,0,1]),30,**LOSS)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        for m in a.MODALITIES:self.assertNotEqual(old[m],state_digest(model.projections[m].state_dict()))

    def test_p_does_not_mix_people_before_fusion(self):
        model=a.make_model(self.data,11,'P');before=model(self.x)
        changed={m:v.clone() for m,v in self.x.items()};changed['eeg'][1:]*=100
        after=model(changed)
        self.assertTrue(torch.equal(before[0],after[0]))
        self.assertTrue(torch.equal(before[:,1:],after[:,1:]))


if __name__=='__main__':unittest.main()
