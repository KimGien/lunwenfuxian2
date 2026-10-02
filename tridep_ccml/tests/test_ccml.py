"""Synthetic TC mathematical checks; no MODMA inputs or subject training."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

import torch
from torch.distributions import Dirichlet, kl_divergence
from torch.nn import functional as F

from tridep_ccml.ccml import (
    MODALITIES, ThreeViewHeads, annealing_coefficient, ccml_loss, ce_heads_loss,
    dirichlet_from_evidence, evidence_dc, expected_cross_entropy, fuse_evidence,
    fuse_predictions, kl_to_uniform, predict_binary, single_evidential_loss,
)


REFERENCE = Path(__file__).resolve().parents[1] / "references" / "ccml"


def load_reference(name: str):
    spec = importlib.util.spec_from_file_location(f"tc_reference_{name}", REFERENCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MathematicalChecks(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(7123)

    def random_evidence(self, batch=7, views=3, classes=2):
        return 0.1 + 8 * torch.rand(batch, views, classes, dtype=torch.float64, generator=self.generator)

    def assert_close(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, expected, **kwargs)

    def test_zero_evidence_is_uniform_and_has_unit_uncertainty(self):
        result = evidence_dc(torch.zeros(4, 3, 2, dtype=torch.float64))
        for name in ("m", "e_con", "e_cmp", "e_final"):
            self.assertEqual(result[name].count_nonzero().item(), 0)
        self.assert_close(result["alpha"], torch.ones(4, 2, dtype=torch.float64))
        self.assert_close(result["p"], torch.full((4, 2), 0.5, dtype=torch.float64))
        self.assert_close(result["u"], torch.ones(4, dtype=torch.float64))
        self.assertEqual(predict_binary(result["p"]).sum().item(), 0)

    def test_zero_common_evidence_preserves_residual_mean(self):
        evidence = torch.tensor([[[5., 0.], [0., 7.], [1., 1.]]])
        result = evidence_dc(evidence)
        self.assert_close(result["e_con"], torch.zeros(1, 2))
        self.assert_close(result["e_cmp"], torch.tensor([[2., 8. / 3]]))
        self.assertTrue((result["alpha"] >= 1).all())

    def test_identical_opinions_have_no_complement_and_preserve_common_mass(self):
        opinion = torch.tensor([[2., 5.]], dtype=torch.float64)
        result = evidence_dc(opinion[:, None].expand(1, 3, 2))
        expected = 21 * opinion.pow(1.25) / opinion.pow(1.25).sum(-1, keepdim=True)
        self.assert_close(result["e_con"], expected)
        self.assert_close(result["e_cmp"], torch.zeros_like(opinion))
        self.assert_close(result["e_con"].sum(-1), torch.tensor([21.], dtype=torch.float64))

    def test_equal_class_common_evidence_does_not_create_preference(self):
        result = evidence_dc(torch.full((1, 3, 2), 6.), rho=1.25)
        self.assert_close(result["p"], torch.tensor([[0.5, 0.5]]))

    def test_rho_one_has_exact_expected_decomposition(self):
        evidence = torch.tensor([[[4., 8.], [2., 3.], [5., 1.]]], dtype=torch.float64)
        result = evidence_dc(evidence, rho=1)
        self.assert_close(result["m"], torch.tensor([[2., 1.]], dtype=torch.float64))
        self.assert_close(result["e_con"], torch.tensor([[6., 3.]], dtype=torch.float64))
        self.assert_close(result["e_cmp"], torch.tensor([[5. / 3, 3.]], dtype=torch.float64))

    def test_strong_symmetric_conflict_does_not_choose_a_class(self):
        evidence = torch.tensor([[[100., 0.], [0., 100.]]])
        result = evidence_dc(evidence)
        self.assert_close(result["p"], torch.tensor([[0.5, 0.5]]))
        self.assert_close(result["e_final"], torch.tensor([[50., 50.]]))
        # This explicitly demonstrates that low u is not a conflict detector.
        self.assertLess(result["u"].item(), 0.02)

    def test_copying_opinions_changes_strength_with_actual_view_count(self):
        opinion = torch.tensor([[[1., 4.]]], dtype=torch.float64)
        single = evidence_dc(opinion)
        duplicate = evidence_dc(opinion.repeat(1, 2, 1))
        triple = evidence_dc(opinion.repeat(1, 3, 1))
        self.assert_close(duplicate["e_con"], 2 * single["e_con"])
        self.assert_close(triple["e_con"], 3 * single["e_con"])
        self.assertLess(triple["u"].item(), duplicate["u"].item())
        self.assertEqual(predict_binary(single["p"]), predict_binary(triple["p"]))

    def test_fixed_single_modality_perturbation_is_finite_and_detectable(self):
        # Prespecified synthetic perturbation: halve only view 0 evidence.
        evidence = torch.tensor([[[4., 1.], [3., 2.], [2., 3.]]], dtype=torch.float64)
        perturbed = evidence.clone()
        perturbed[:, 0] *= 0.5
        result = evidence_dc(perturbed)
        self.assert_close(perturbed[:, 1:], evidence[:, 1:])
        self.assertFalse(torch.allclose(result["p"], evidence_dc(evidence)["p"]))
        self.assertTrue(torch.isfinite(result["p"]).all())
        self.assertTrue((result["e_final"] >= 0).all())

    def test_modality_permutation_invariance(self):
        evidence = self.random_evidence()
        result = evidence_dc(evidence)
        permuted = evidence_dc(evidence[:, [2, 0, 1]])
        for key in result:
            self.assert_close(permuted[key], result[key])

    def test_class_permutation_equivariance_and_loss_invariance(self):
        evidence = self.random_evidence()
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 0])
        result = evidence_dc(evidence)
        permuted = evidence_dc(evidence.flip(-1))
        for key in result:
            self.assert_close(permuted[key], result[key] if key == "u" else result[key].flip(-1))
        self.assert_close(ccml_loss(evidence, labels, 12), ccml_loss(evidence.flip(-1), 1 - labels, 12))

    def test_mean_and_sum_evidence_have_same_decisions_different_uncertainty(self):
        evidence = self.random_evidence(batch=20)
        mean = fuse_evidence(evidence, "mean_evidence")
        summed = fuse_evidence(evidence, "sum_evidence")
        self.assert_close(mean["p"].argmax(-1), summed["p"].argmax(-1))
        self.assert_close(predict_binary(mean["p"]), predict_binary(summed["p"]))
        self.assertTrue((summed["u"] < mean["u"]).all())
        self.assertFalse(torch.allclose(mean["p"], summed["p"]))

    def test_dirichlet_mean_is_not_softmax(self):
        evidence = torch.tensor([[0., 3.]])
        result = dirichlet_from_evidence(evidence)
        self.assert_close(result["p"], torch.tensor([[0.2, 0.8]]))
        self.assert_close(result["u"], torch.tensor([0.4]))
        self.assertFalse(torch.allclose(result["p"], evidence.softmax(-1)))

    def test_probability_mean_has_no_fabricated_evidence_uncertainty(self):
        logits = self.random_evidence()
        for evidential in (True, False):
            result = fuse_predictions(logits, evidential=evidential)
            self.assertIsNone(result["u"])
            self.assertIsNone(result["alpha"])
            self.assertIsNone(result["e_final"])
            source = dirichlet_from_evidence(F.softplus(logits))["p"] if evidential else logits.softmax(-1)
            self.assert_close(result["p"], source.mean(1))

    def test_invalid_inputs_fail_instead_of_becoming_legal_dirichlet_parameters(self):
        for value in (-1., float("nan"), float("inf")):
            evidence = torch.ones(1, 3, 2)
            evidence[0, 0, 0] = value
            with self.assertRaises(ValueError):
                evidence_dc(evidence)
        with self.assertRaises(ValueError):
            evidence_dc(torch.ones(1, 3, 2), rho=0.5)
        with self.assertRaises(ValueError):
            evidence_dc(torch.ones(1, 0, 2))

    def test_extreme_scale_decomposition_preserves_mass_and_finite_gradients(self):
        for scale in (1e-45, 1e-38, 1e-30, 1e20):
            evidence = (torch.tensor([[[1., 2.], [3., 4.], [2., 3.]]]) * scale).requires_grad_()
            result = evidence_dc(evidence)
            expected_mass = 3 * evidence.amin(1).sum(-1)
            self.assert_close(result["e_con"].sum(-1), expected_mass)
            result["e_final"].sum().backward()
            self.assertTrue(torch.isfinite(evidence.grad).all())

    def test_subnormal_float32_evidence_has_finite_correct_mass_gradient(self):
        evidence = torch.full((1, 3, 2), 1e-45, requires_grad=True)
        result = evidence_dc(evidence)
        result["e_final"].sum().backward()
        self.assertTrue(torch.isfinite(evidence.grad).all())
        self.assert_close(evidence.grad, torch.ones_like(evidence))

    def test_zero_and_saturated_evidence_have_finite_loss_gradients(self):
        for values in ([0., 0.], [0., 10000.], [1., 1.]):
            evidence = torch.tensor([[values] * 3], dtype=torch.float64, requires_grad=True)
            loss = ccml_loss(evidence, torch.tensor([1]), epoch=50)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(evidence.grad).all())

    def test_full_loss_gradcheck_away_from_minimum_ties(self):
        evidence = self.random_evidence(batch=2).requires_grad_()
        labels = torch.tensor([0, 1])
        self.assertTrue(torch.autograd.gradcheck(lambda x: ccml_loss(x, labels, epoch=17), evidence))

    def test_annealing_uses_epoch_number_and_saturates(self):
        self.assertEqual(annealing_coefficient(0), 0)
        self.assertEqual(annealing_coefficient(1), 0.02)
        self.assertEqual(annealing_coefficient(25), 0.5)
        self.assertEqual(annealing_coefficient(50), 1)
        self.assertEqual(annealing_coefficient(60), 1)

    def test_kl_matches_independent_pytorch_distribution(self):
        alpha = self.random_evidence() + 1
        reference = kl_divergence(Dirichlet(alpha), Dirichlet(torch.ones_like(alpha)))
        self.assert_close(kl_to_uniform(alpha), reference)
        self.assert_close(kl_to_uniform(torch.ones(5, 2)), torch.zeros(5))

    def test_true_class_evidence_is_removed_from_kl(self):
        evidence = torch.tensor([[[1., 2.]], [[100., 2.]]], dtype=torch.float64)
        labels = torch.tensor([0, 0])
        penalties = []
        for i in range(2):
            penalties.append(single_evidential_loss(evidence[i:i+1], labels[i:i+1], 50)
                             - single_evidential_loss(evidence[i:i+1], labels[i:i+1], 0))
        self.assert_close(penalties[0], penalties[1])

    def test_ce_is_unweighted_view_sum_and_subject_mean(self):
        logits = self.random_evidence(batch=2)
        labels = torch.tensor([0, 1])
        expected = sum(F.cross_entropy(logits[:, v], labels) for v in range(3))
        self.assert_close(ce_heads_loss(logits, labels), expected)

    def test_full_loss_has_only_planned_components_and_ablation_weights(self):
        evidence = self.random_evidence(batch=2)
        labels = torch.tensor([0, 1])
        fused = evidence_dc(evidence)
        one_hot = F.one_hot(labels, 2)
        con_data = expected_cross_entropy(fused["e_con"] + 1, one_hot).mean()
        cmp_data = expected_cross_entropy(fused["e_cmp"] + 1, one_hot).mean()
        con_kl = 0.5 * kl_divergence(Dirichlet(1 + (1 - one_hot) * fused["e_con"]),
                                    Dirichlet(torch.ones_like(fused["e_con"]))).mean()
        single = single_evidential_loss(evidence, labels, 25)
        full = ccml_loss(evidence, labels, 25)
        self.assert_close(full, single + con_data + con_kl + cmp_data)
        self.assert_close(full - ccml_loss(evidence, labels, 25, delta=0), con_kl)
        self.assert_close(full - ccml_loss(evidence, labels, 25, gamma=0), cmp_data)

    def test_default_heads_are_independent_and_have_774_trainable_parameters(self):
        torch.manual_seed(11)
        heads = ThreeViewHeads()
        self.assertEqual(sum(p.numel() for p in heads.parameters() if p.requires_grad), 774)
        self.assertEqual(len({layer.weight.data_ptr() for layer in heads.heads.values()}), 3)
        features = torch.randn(4, 3, 128)
        logits = heads(features)
        self.assertEqual(tuple(logits.shape), (4, 3, 2))
        self.assert_close(heads.evidence(features), F.softplus(logits))
        self.assert_close(heads({name: features[:, i] for i, name in enumerate(MODALITIES)}), logits)

    def test_state_save_reload_preserves_output_exactly(self):
        heads = ThreeViewHeads()
        features = torch.randn(3, 3, 128)
        before = fuse_predictions(heads(features), "ccml")
        buffer = io.BytesIO()
        torch.save(heads.state_dict(), buffer)
        buffer.seek(0)
        restored = ThreeViewHeads()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        after = fuse_predictions(restored(features), "ccml")
        for key in before:
            self.assert_close(after[key], before[key], rtol=0, atol=0)

    def test_head_parameters_get_finite_nonzero_full_loss_gradients(self):
        heads = ThreeViewHeads()
        features = torch.randn(4, 3, 128)
        loss = ccml_loss(heads.evidence(features), torch.tensor([0, 1, 0, 1]), 1)
        loss.backward()
        for parameter in heads.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum(), 0)

    def test_fixed_reference_files_match_sha256_and_git_blob(self):
        record = json.loads((REFERENCE / "source.json").read_text(encoding="utf-8"))
        self.assertEqual(record["commit"], "8938ae23bcba5d6ff1508b35fc9a6664eca54f71")
        for item in record["files"]:
            data = (REFERENCE / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual(hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(), item["git_blob_sha"])

    def test_positive_inputs_match_pinned_official_decomposition_and_loss(self):
        upstream_model = load_reference("model")
        upstream_loss = load_reference("loss_function")
        evidence = self.random_evidence()
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 0])
        for views in (1, 2, 3):
            values = evidence[:, :views]
            upstream = upstream_model.CCML(views, [[128]] * views, 2, "cpu")
            inputs = {v: values[:, v] + 1 for v in range(views)}
            for rho in (1., 1.25):
                final, consistent, complementary = upstream.Evidence_DC(inputs, rho)
                result = evidence_dc(values, rho)
                self.assert_close(result["alpha"], final)
                self.assert_close(result["e_con"] + 1, consistent)
                self.assert_close(result["e_cmp"] + 1, complementary)
                reference = upstream_loss.get_loss(
                    {v: values[:, v] for v in range(views)}, final - 1, consistent - 1,
                    complementary - 1, labels, 25, 2, 50, 1., 1., "cpu")
                self.assert_close(ccml_loss(values, labels, 25, rho=rho), reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)
