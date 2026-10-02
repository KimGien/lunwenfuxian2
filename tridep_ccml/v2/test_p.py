import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .p import ProjectionHeads, DIMENSIONS, PROJECTION_COUNT, models, MODALITIES
from .common import set_seed, LOSS
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ccml_loss


def synthetic_projection_state():
    return {f'{m}.0.{k}':v for m,d in DIMENSIONS.items() for k,v in nn.Linear(d,128).state_dict().items()}


class ProjectionTests(unittest.TestCase):
    def test_only_last_projections_and_fresh_heads_are_trainable(self):
        set_seed(11);model=ProjectionHeads(synthetic_projection_state(),11)
        self.assertEqual(sum(p.numel() for p in model.projections.parameters()),PROJECTION_COUNT)
        self.assertEqual(sum(p.numel() for p in model.parameters()),836742)
        self.assertEqual(state_digest(model.heads.state_dict()),state_digest(make_initial_state(11)))
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.Dropout)))
        self.assertFalse(any(isinstance(m,(nn.Conv1d,nn.LSTM)) for m in model.modules()))

    def test_preprojection_hook_preserves_full_frozen_eeg_representation(self):
        set_seed(22);parent=models.EEGConvPoolClassifier(input_dim=200,seq_len=60).requires_grad_(False).eval()
        parent.classifier[-1]=nn.Identity();captured=[]
        handle=parent.classifier[2].register_forward_pre_hook(lambda _,args:captured.append(args[0].detach()))
        raw=torch.randn(3,60,200)
        with torch.no_grad():expected=parent(raw)
        handle.remove()
        self.assertEqual(captured[0].shape,(3,1920))
        projected=F.relu(F.linear(captured[0],parent.classifier[2].weight,parent.classifier[2].bias))
        self.assertTrue(torch.equal(expected,projected))

    def test_ccml_updates_all_three_projections_without_changing_cached_inputs(self):
        set_seed(33);model=ProjectionHeads(synthetic_projection_state(),33)
        x={m:torch.randn(4,d) for m,d in DIMENSIONS.items()};copies={m:v.clone() for m,v in x.items()}
        before={m:state_digest(model.projections[m].state_dict()) for m in MODALITIES}
        self.assertTrue(torch.equal(model(x),model(x)))
        optimizer=torch.optim.Adamax([{'params':model.projections.parameters(),'lr':1e-4},{'params':model.heads.parameters(),'lr':1e-3}])
        loss=ccml_loss(F.softplus(model(x)),torch.tensor([0,1,0,1]),1,**LOSS);loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        for m in MODALITIES:
            self.assertNotEqual(before[m],state_digest(model.projections[m].state_dict()))
            self.assertTrue(torch.equal(x[m],copies[m]));self.assertIsNone(x[m].grad)

    def test_postprojection_cache_cannot_masquerade_as_projection_input(self):
        set_seed(11);model=ProjectionHeads(synthetic_projection_state(),11)
        with self.assertRaises(RuntimeError):model({m:torch.zeros(2,128) for m in MODALITIES})


if __name__=='__main__':unittest.main()
