"""Regression tests for prior dead encoders, isolation, pairing and meaningful gradients."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
from torch.nn import functional as F
from .common import *
from .models import P1Model, R1Model, SparNetP1
from .health import audit, unchanged, check_train_ids, weight_modules, eeg_representation
from .train import step
from tridep_ccml.eeg_replacement.models import SparNet


class SyntheticTrain:
    def __init__(self):
        self.splits = {'train':['a','b','c','d'],'val':['v'],'test':['t']}
        self.raw = torch.randn(4,60,19,1000,device='cuda')*torch.tensor([.5,1.,2.,3.],device='cuda')[:,None,None,None]
        self.y = torch.tensor([0,1,0,1],device='cuda')

    def batch(self,ids,condition):
        idx = [self.splits['train'].index(s) for s in ids]
        return {'eeg':self.raw[idx]},self.y[idx]


class Contracts(unittest.TestCase):
    def setUp(self):
        set_seed(11)

    def test_outer_subjects_rejected(self):
        data = SimpleNamespace(splits={'train':['a','b'],'val':['v'],'test':['t']})
        for ids in [['a','v'],['a','t'],['a','a']]:
            with self.assertRaises(ValueError):
                check_train_ids(data,ids)
        check_train_ids(data,['a','b'])

    def test_pairing_and_original_controls(self):
        for seed in SEEDS:
            for condition in ['U0','C0']:
                self.assertEqual(state_digest(P1Model(condition,seed).state_dict()),state_digest(R1Model(condition,seed).state_dict()))
            a,b = P1Model('U2',seed),P1Model('C2',seed)
            self.assertEqual(state_digest(a.eeg.state_dict()),state_digest(b.eeg.state_dict()))
            self.assertEqual(state_digest(a.classifier.state_dict()),state_digest(R1Model('U2',seed).classifier.state_dict()))
            self.assertEqual(state_digest(b.shared_state()),state_digest(R1Model('C0',seed).shared_state()))
            self.assertLess(a.counts()['eeg_including_head'],50000)

    def test_fft_fixed_scale_axes_and_spectrum_shortcut_gradients(self):
        x = torch.randn(2,60,19,1000,device='cuda')
        spectrum = SparNetP1.spectrum(x)
        torch.testing.assert_close(spectrum,SparNet.spectrum(x)*1000**.5,rtol=1e-5,atol=1e-6)
        a = P1Model('U2',11).cuda()
        output = a.eeg(x)
        self.assertEqual(output.shape,(2,128))
        output.square().mean().backward()
        first = {k:p.grad.clone() for k,p in a.eeg.named_parameters()}
        a.zero_grad(set_to_none=True)
        shortcut = a.eeg.from_spectrum(spectrum)
        shortcut.square().mean().backward()
        torch.testing.assert_close(output,shortcut,rtol=0,atol=0)
        for k,p in a.eeg.named_parameters():
            torch.testing.assert_close(p.grad,first[k],rtol=0,atol=0)

    def test_health_restores_model_modes_rng_gradients(self):
        model = P1Model('U2',11).cuda().train()
        model.eeg.regions[0].eval()
        data = SyntheticTrain()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        states = state_digest(model.state_dict())
        modes = [m.training for m in model.modules()]
        cpu,cuda = torch.get_rng_state(),torch.cuda.get_rng_state()
        result = audit(model,data)
        self.assertTrue(result['passed'],result['failures'])
        self.assertEqual(states,state_digest(model.state_dict()))
        self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda,torch.cuda.get_rng_state()))
        self.assertTrue(all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters()))

    def test_dead_encoder_rejected_despite_finite_loss_and_bias_update(self):
        model = R1Model('U2',11).cuda().train()
        with torch.no_grad():
            model.eeg.global_conv[1].weight.zero_()
            model.eeg.global_conv[1].bias.fill_(-1.)
        result = audit(model,SyntheticTrain())
        self.assertFalse(result['passed'])
        self.assertTrue(any('gradient' in s for s in result['failures']))
        self.assertTrue(any('representation' in s for s in result['failures']))
        self.assertEqual(result['zero_input_relative_response'],0.)

    def test_nonfinite_representation_rejected(self):
        model = SimpleNamespace(joint=False,representations=lambda x:torch.full((2,128),float('nan')))
        with self.assertRaises(ValueError):
            eeg_representation(model,{})

    def test_real_task_updates_all_encoder_modules(self):
        model = P1Model('U2',11).cuda().train()
        data = SyntheticTrain()
        before = {k:p.detach().clone() for k,p in weight_modules(model).items()}
        x,y = data.batch(data.splits['train'],'U2')
        loss,norm,grads = step(model,model.optimizer(),x,y,1)
        self.assertTrue(all(v>GATES['gradient_l2_min'] for v in grads.values()))
        self.assertTrue(all(torch.any(p.detach()!=before[k]).item() for k,p in weight_modules(model).items()))
        self.assertEqual(len(grads),19)

    def test_unchanged_restores_on_exception(self):
        model = P1Model('U0',11).cuda()
        before = state_digest(model.state_dict())
        cpu,cuda = torch.get_rng_state(),torch.cuda.get_rng_state()
        with self.assertRaises(RuntimeError):
            with unchanged(model):
                with torch.no_grad():
                    next(model.parameters()).add_(10)
                torch.rand(3,device='cuda')
                torch.rand(3)
                raise RuntimeError('injected')
        self.assertEqual(before,state_digest(model.state_dict()))
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda,torch.cuda.get_rng_state()))


if __name__=='__main__':
    unittest.main()
