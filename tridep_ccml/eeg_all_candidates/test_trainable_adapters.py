"""CPU-only author-core, gradient, SSP and independent EMA evidence."""
from __future__ import annotations

import copy
import random
import unittest

import numpy as np
import torch

from .adapters import eegconformer as A
from .adapters import eeg2rep as E


class TrainableAdapterCPU(unittest.TestCase):
    def setUp(self):
        self.cpu = torch.get_rng_state()
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        torch.set_num_threads(4)

    def tearDown(self):
        torch.set_rng_state(self.cpu); random.setstate(self.python); np.random.set_state(self.numpy)

    def test_conformer_original_cnn_attention_both_have_finite_nonzero_gradients(self):
        torch.manual_seed(13)
        model, audit = A.load_encoder('cpu', trainable=True)
        self.assertEqual(tuple(model.patch.shallownet[1].kernel_size), (19, 1))
        self.assertEqual(len(model.transformer), 6)
        raw = np.random.default_rng(4).normal(size=(2, 19, 1000)).astype(np.float32)
        statistics = {'mean': 0., 'std': 1., 'ddof': 0, 'fit_subject_ids': ['engineering_train']}
        output = A.encode_windows(model, torch.from_numpy(A.adapt_raw(raw, statistics)))
        self.assertEqual(tuple(output.shape), (2, 40))
        loss = output.square().mean()
        loss.backward()
        for prefix in ('patch.', 'transformer.'):
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        self.assertEqual(audit['neural_forward_calls_during_load'], 0)
        with self.assertRaises(ValueError):
            A.load_encoder('cpu', trainable=False)

    def test_eeg2rep_ssp_output_matches_author_and_ema_never_aliases(self):
        torch.manual_seed(4)
        model, audit = E.build_model('cpu', stage='ssl')
        model.eval()
        x = torch.randn(2, 19, 1000)
        random.seed(71)
        author = model.core.pretrain_forward(x)
        mask = E.make_ssp_mask(rng=random.Random(71))
        output = model.pretrain_forward(x, mask=mask)
        for produced, original in zip((output['target_masked'], output['prediction_masked'], output['context'], output['target_full']), author):
            self.assertTrue(torch.equal(produced, original))
        self.assertEqual(len(mask['visible_indices']), 250)
        self.assertTrue(np.array_equal(mask['masked_indices'], np.setdiff1d(np.arange(500), mask['visible_indices'])))
        original_python = random.getstate()
        self.assertEqual(E.make_ssp_mask(rng=random.Random(33)), E.make_ssp_mask(rng=random.Random(33)))
        self.assertEqual(original_python, random.getstate())
        before = [p.detach().clone() for p in model.core.target_encoder.parameters()]
        with torch.no_grad():
            for p in model.core.contex_encoder.parameters():
                p.add_(.5)
        for target, old, online in zip(model.core.target_encoder.parameters(), before, model.core.contex_encoder.parameters()):
            self.assertFalse(target.requires_grad)
            self.assertNotEqual(target.data_ptr(), online.data_ptr())
            self.assertTrue(torch.equal(target, old))
        online_before = [p.detach().clone() for p in model.core.contex_encoder.parameters()]
        model.momentum_update()
        for target, old, online in zip(model.core.target_encoder.parameters(), before, model.core.contex_encoder.parameters()):
            self.assertTrue(torch.allclose(target, old * .99 + online * .01, atol=1e-7, rtol=1e-6))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(online_before, model.core.contex_encoder.parameters())))
        self.assertTrue(audit['requires_fold_local_ssl_training'])

    def test_eeg2rep_loss_equals_author_formula_and_reaches_trainable_ssl_modules(self):
        torch.manual_seed(9)
        predicted = torch.randn(4, 37, 16, requires_grad=True)
        target = torch.randn_like(predicted)
        actual = E.ssl_loss({'prediction_masked': predicted, 'target_masked': target})
        y = torch.nn.AdaptiveAvgPool1d(1)(predicted.transpose(2, 1)).squeeze()
        y = y - y.mean(dim=0)
        std_loss = torch.nn.functional.relu(1. - torch.sqrt(y.var(dim=0) + .0001)).mean()
        cov = y.T @ y / 3
        off = cov.flatten()[:-1].view(15, 17)[:, 1:].flatten()
        expected = torch.nn.functional.mse_loss(predicted, target) + std_loss + off.square().sum() / 16
        self.assertTrue(torch.allclose(actual['loss'], expected, atol=1e-7, rtol=1e-6))
        g1 = torch.autograd.grad(actual['loss'], predicted, retain_graph=True)[0]
        g2 = torch.autograd.grad(expected, predicted)[0]
        self.assertTrue(torch.allclose(g1, g2, atol=1e-7, rtol=1e-6))
        model, _ = E.build_model('cpu', stage='ssl')
        output = model.pretrain_forward(torch.randn(2, 19, 1000), mask=E.make_ssp_mask(rng=random.Random(13)))
        E.ssl_loss(output)['loss'].backward()
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable))
        for prefix in ('core.InputEmbedding.', 'core.contex_encoder.', 'core.Predictor.'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for name, p in trainable if name.startswith(prefix)), 0.)
        self.assertTrue(all(p.grad is None for p in model.core.target_encoder.parameters()))
        with self.assertRaises(ValueError):
            E.ssl_loss({'prediction_masked': predicted[:1], 'target_masked': target[:1]})

    def test_eeg2rep_supervised_single_window_preserves_axis_and_encoder_gradients(self):
        torch.manual_seed(22)
        model, _ = E.build_model('cpu', stage='supervised')
        output = E.encode_windows(model, torch.randn(1, 19, 1000))
        self.assertEqual(tuple(output.shape), (1, 16))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.core.target_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.core.Predictor.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
