import io
import unittest

import torch
from torch.nn import functional as F

from .nr import CandidateHeads, VARIANTS
from .common import set_seed, LOSS
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ThreeViewHeads, ccml_loss


class NRCandidateTests(unittest.TestCase):
    def test_layernorm_is_subject_and_view_local(self):
        set_seed(11);model=CandidateHeads('N');model.load_state_dict(make_initial_state(11))
        x=torch.randn(4,3,128);before=model(x)[0,0]
        changed=x.clone();changed[1:]=1000;changed[0,1:]=-1000
        self.assertTrue(torch.equal(before,model(changed)[0,0]))
        self.assertTrue(torch.equal(model.normalizer(x[:1]),model.normalizer(x)[:1]))
        # Different matrix batch sizes may change floating-point accumulation.
        torch.testing.assert_close(model(x[:1]),model(x)[:1],rtol=1e-6,atol=1e-7)

    def test_normalization_has_no_trainable_affine_parameters(self):
        for variant in VARIANTS:
            model=CandidateHeads(variant);model.load_state_dict(make_initial_state(22))
            self.assertEqual(sum(p.numel() for p in model.parameters()),774)
            self.assertEqual(state_digest(model.state_dict()),state_digest(make_initial_state(22)))
        model=CandidateHeads('N');x=torch.randn(2,3,128)
        self.assertTrue(torch.equal(model.normalizer(x),F.layer_norm(x,(128,),eps=1e-5)))

    def test_control_step_exactly_matches_historical_head_path(self):
        set_seed(33);a=ThreeViewHeads();b=CandidateHeads('C')
        initial=make_initial_state(33);a.load_state_dict(initial);b.load_state_dict(initial)
        x=torch.randn(8,3,128);y=torch.tensor([0,1,0,1,1,0,1,0])
        for model in [a,b]:
            optimizer=torch.optim.Adamax([{'params':model.parameters(),'lr':.001}],weight_decay=0.)
            loss=ccml_loss(F.softplus(model(x)),y,1,**LOSS);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        self.assertEqual(state_digest(a.state_dict()),state_digest(b.state_dict()))

    def test_regularization_changes_optimizer_only(self):
        set_seed(11);a=CandidateHeads('C');b=CandidateHeads('R')
        initial=make_initial_state(11);a.load_state_dict(initial);b.load_state_dict(initial)
        x=torch.randn(3,3,128);self.assertTrue(torch.equal(a(x),b(x)))
        for model,wd in [(a,0.),(b,VARIANTS['R']['weight_decay'])]:
            optimizer=torch.optim.Adamax(model.parameters(),lr=.001,weight_decay=wd)
            for p in model.parameters():p.grad=torch.zeros_like(p)
            optimizer.step()
        self.assertEqual(state_digest(a.state_dict()),state_digest(initial))
        self.assertNotEqual(state_digest(b.state_dict()),state_digest(initial))

    def test_new_metadata_roundtrips_with_default_restricted_loading(self):
        model=CandidateHeads('N');stream=io.BytesIO()
        torch.save({'state_dict':model.state_dict(),'metadata':{'torch':str(torch.__version__),'variant':'N'}},stream)
        stream.seek(0);saved=torch.load(stream,map_location='cpu',weights_only=True)
        self.assertEqual(state_digest(saved['state_dict']),state_digest(model.state_dict()))


if __name__=='__main__':unittest.main()
