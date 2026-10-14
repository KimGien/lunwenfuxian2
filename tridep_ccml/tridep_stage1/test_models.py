"""Axis, gradient, pairing and structure tests; no training or fitting runs."""
from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from .models import CBraModTokenHead, CONDITIONS, TOKEN_SHAPE


def test_graph() -> torch.Tensor:
    # A fixed, nonregular chain; deliberately distinct from the registered
    # electrode graph. It tests ordering and propagation only.
    graph = torch.eye(19)
    for i in range(18):
        graph[i, i + 1] = graph[i + 1, i] = 1
    return graph / graph.sum(1, keepdim=True)


class TokenHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        generator = torch.Generator().manual_seed(1803)
        cls.tokens = torch.randn((2,) + TOKEN_SHAPE, generator=generator)
        cls.graph = test_graph()
        cls.channel_perm = torch.tensor([7, 2, 11, 0, 15, 4, 16, 1, 12, 6, 18, 3, 13, 8, 5, 17, 9, 14, 10])
        cls.time_perm = torch.tensor([0, 3, 1, 4, 2])
        cls.window_perm = torch.randperm(60, generator=generator)

    def model(self, condition, adjacency=None):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(919)
            return CBraModTokenHead(condition, self.graph if adjacency is None else adjacency).eval()

    def test_shape_and_exact_parameter_match(self):
        reference = None
        for condition in CONDITIONS:
            with self.subTest(condition=condition), torch.no_grad():
                model = self.model(condition)
                logits, features = model(self.tokens, return_features=True)
                self.assertEqual(logits.shape, (2, 2))
                self.assertEqual(features.shape, (2, 128))
                windows = model.window_features(self.tokens)
                self.assertEqual(windows.shape, (2, 60, 128))
                self.assertTrue(torch.equal(features, windows.mean(1)))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(model.architecture()["trainable_parameters"], 12098)
                self.assertEqual(model.architecture()["parameters"], 12098)
                state = model.state_dict()
                if reference is None:
                    reference = state
                else:
                    self.assertEqual(set(reference), set(state))
                    self.assertTrue(all(torch.equal(reference[k], v) for k, v in state.items()))

    def test_rejects_axis_errors_and_invalid_graph(self):
        model = self.model("mean")
        for bad in (self.tokens[:, :59], self.tokens.transpose(2, 3), self.tokens[:0], self.tokens.long()):
            with self.subTest(shape=tuple(bad.shape)), self.assertRaises(ValueError):
                model(bad)
        for graph in (torch.eye(18), torch.zeros(19, 19), torch.eye(19) * 2, torch.eye(19).fill_(float("nan"))):
            with self.subTest(graph_shape=tuple(graph.shape)), self.assertRaises(ValueError):
                self.model("mean", graph)
        with self.assertRaises(ValueError):
            self.model("unknown")

    def test_all_parameters_have_finite_nonzero_gradients_and_encoder_is_frozen(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                model = self.model(condition)
                # Positive biases keep this synthetic gradient check away from
                # accidental dead ReLU channels; they are not model defaults.
                with torch.no_grad():
                    for module in model.modules():
                        if hasattr(module, "bias") and module.bias is not None:
                            module.bias.fill_(2.)
                tokens = self.tokens[:1].clone().requires_grad_(True)
                logits = model(tokens)
                F.cross_entropy(logits, torch.tensor([1])).backward()
                for name, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
                    self.assertGreater(float(param.grad.abs().sum()), 0., name)
                self.assertIsNone(tokens.grad)
                self.assertFalse(model.adjacency.requires_grad)
                self.assertIsNone(model.adjacency.grad)

    def test_subject_isolation_window_order_and_repeatability(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition), torch.no_grad():
                model = self.model(condition)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                logits = model(self.tokens)
                self.assertTrue(torch.equal(logits, model(self.tokens)))
                self.assertTrue(torch.allclose(logits.flip(0), model(self.tokens.flip(0)), atol=1e-6, rtol=0))
                self.assertTrue(torch.allclose(logits[:1], model(self.tokens[:1]), atol=1e-6, rtol=0))
                self.assertTrue(torch.allclose(logits, model(self.tokens[:, self.window_perm]), atol=1e-6, rtol=0))
                self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_pooled_axis_permutation_invariance(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition), torch.no_grad():
                model = self.model(condition)
                result = model.encode(self.tokens)
                if not model.use_spatial:
                    self.assertTrue(torch.allclose(result, model.encode(self.tokens[:, :, self.channel_perm]), atol=1e-6, rtol=0))
                if not model.use_temporal:
                    self.assertTrue(torch.allclose(result, model.encode(self.tokens[:, :, :, self.time_perm]), atol=1e-6, rtol=0))

    def test_spatial_order_requires_matching_graph_and_temporal_order_matters(self):
        with torch.no_grad():
            for condition in ("spatial", "spatiotemporal"):
                with self.subTest(condition=condition):
                    model = self.model(condition)
                    result = model.encode(self.tokens)
                    shuffled = self.tokens[:, :, self.channel_perm]
                    self.assertGreater(float((result - model.encode(shuffled)).abs().max()), 1e-6)
                    graph = self.graph[self.channel_perm][:, self.channel_perm]
                    relabeled = self.model(condition, graph)
                    self.assertTrue(torch.allclose(result, relabeled.encode(shuffled), atol=1e-6, rtol=0))
            for condition in ("temporal", "spatiotemporal"):
                with self.subTest(condition=condition):
                    model = self.model(condition)
                    result = model.encode(self.tokens)
                    shuffled = self.tokens[:, :, :, self.time_perm]
                    self.assertGreater(float((result - model.encode(shuffled)).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
