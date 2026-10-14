"""CPU synthetic tests for the SSL autocast boundary; no models or cohort."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from . import ssl_v2 as V2


class MatmulDtypes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.dtypes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.mm.default:
            self.dtypes.append(result.dtype)
        return result


def fixture():
    generator = torch.Generator().manual_seed(20261008)
    prediction = torch.randn(64, 37, 16, generator=generator, requires_grad=True)
    target = torch.randn(64, 37, 16, generator=generator, requires_grad=True)
    return {'prediction_masked': prediction, 'target_masked': target}


class SSLPrecisionV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_float_cast_alone_does_not_keep_covariance_float32(self):
        output = fixture()
        with MatmulDtypes() as observed, torch.autocast('cpu', dtype=torch.bfloat16):
            old = V2._ORIGINAL_LOSS(output)
        self.assertEqual(observed.dtypes, [torch.bfloat16])
        self.assertEqual(old['offdiagonal_covariance'].dtype, torch.bfloat16)
        self.assertEqual(old['loss'].dtype, torch.float32)

    def test_each_component_and_gradient_exactly_match_fp32_reference(self):
        output = fixture()
        with torch.autocast('cpu', enabled=False):
            reference = V2._ORIGINAL_LOSS(output)
        reference_gradient = torch.autograd.grad(reference['loss'], output['prediction_masked'], retain_graph=True)[0]
        with MatmulDtypes() as observed, torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled('cpu'))
            actual = V2.ssl_loss(output)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertFalse(torch.is_autocast_enabled('cpu'))
        self.assertEqual(observed.dtypes, [torch.float32])
        for key in V2.COMPONENTS:
            self.assertEqual(actual[key].dtype, torch.float32)
            self.assertTrue(torch.equal(actual[key], reference[key]), key)
        gradient, target_gradient = torch.autograd.grad(actual['loss'], (output['prediction_masked'], output['target_masked']), allow_unused=True)
        self.assertTrue(torch.equal(gradient, reference_gradient))
        self.assertIsNone(target_gradient)

    def test_values_match_independent_author_fp32_formula(self):
        output = fixture()
        predicted, target = output['prediction_masked'], output['target_masked'].detach()
        y = torch.nn.AdaptiveAvgPool1d(1)(predicted.transpose(2, 1)).squeeze()
        y = y-y.mean(dim=0)
        std = torch.sqrt(y.var(dim=0) + .0001)
        cov = y.T@y/63
        off = cov.flatten()[:-1].view(15, 17)[:, 1:].flatten()
        components = {'alignment_mse': torch.nn.functional.mse_loss(predicted, target),
                      'variance_hinge': torch.nn.functional.relu(1-std).mean(),
                      'offdiagonal_covariance': off.square().sum()/16}
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = V2.ssl_loss(output)
        for key, expected in components.items():
            torch.testing.assert_close(actual[key], expected, rtol=1e-7, atol=1e-8)

    def test_bf16_rounded_encoder_output_uses_exact_fp32_loss_and_gradient(self):
        values = fixture()
        output = {name: value.detach().to(torch.bfloat16).requires_grad_(True) for name, value in values.items()}
        with torch.autocast('cpu', enabled=False):
            reference = V2._ORIGINAL_LOSS(output)
        reference_grad = torch.autograd.grad(reference['loss'], output['prediction_masked'], retain_graph=True)[0]
        with MatmulDtypes() as observed, torch.autocast('cpu', dtype=torch.bfloat16):
            actual = V2.ssl_loss(output)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(observed.dtypes, [torch.float32])
        for key in V2.COMPONENTS:
            self.assertEqual(actual[key].dtype, torch.float32)
            self.assertTrue(torch.equal(actual[key], reference[key]), key)
        grad, target_grad = torch.autograd.grad(actual['loss'], (output['prediction_masked'], output['target_masked']), allow_unused=True)
        self.assertTrue(torch.equal(grad, reference_grad))
        self.assertIsNone(target_grad)

    def test_loss_exception_restores_outer_autocast(self):
        output = fixture()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with patch.object(V2, '_ORIGINAL_LOSS', side_effect=ValueError('synthetic loss failure')):
                with self.assertRaisesRegex(ValueError, 'synthetic loss failure'):
                    V2.ssl_loss(output)
            self.assertTrue(torch.is_autocast_enabled('cpu'))

    def test_prepare_fit_validation_share_wrapped_loss_and_restore_global_hooks(self):
        original_loss, original_save = V2.E.ssl_loss, V2.S.save
        with patch.object(V2, '_verify_existing_precision_settings'):
            with V2.activate():
                self.assertIs(V2.V1.E.ssl_loss, V2.ssl_loss)
                self.assertIs(V2.V1.S.save, V2._save_with_provenance)
                for function in (V2.V1.prepare, V2.V1.fit_one, V2.V1.validation):
                    self.assertIs(function.__globals__['E'].ssl_loss, V2.ssl_loss)
            self.assertIs(V2.E.ssl_loss, original_loss)
            self.assertIs(V2.S.save, original_save)

    def test_actual_settings_contract_and_metadata_gain_v2_source_provenance(self):
        settings = {'prepared_sources': [], 'loss_precision': 'float32'}
        V2.augment_record('settings', settings)
        repair = settings['ssl_precision_repair']
        self.assertIn(repair['wrapper_source'], settings['prepared_sources'])
        self.assertFalse(settings['loss_autocast_enabled'])
        contract = {'input_contract': copy.deepcopy(settings), 'implementation_sources': []}
        V2.augment_record('contract', contract)
        self.assertIn(repair['wrapper_source'], contract['implementation_sources'])
        metadata = {'condition': 'EEG2Rep-fold-local-ssl', 'settings': {'unchanged': 'identity_descriptor'}}
        V2.augment_record('metadata', metadata)
        self.assertEqual(metadata['settings'], {'unchanged': 'identity_descriptor'})
        self.assertEqual(metadata['ssl_precision_repair'], repair)
        self.assertFalse(metadata['loss_autocast_enabled'])

    def test_real_save_router_augments_before_first_immutable_write(self):
        records = [
            (V2.V1.SETTINGS, {'prepared_sources': [], 'loss_precision': 'float32'}),
            (V2.V1.ENGINEERING_CONTRACT, {'input_contract': {}, 'implementation_sources': []}),
            (V2.S.OUT / 'runs' / V2.V1.CONDITION / 'seed_11/fold_1/metadata.json',
             {'condition': V2.V1.CONDITION, 'settings': {'path': 'synthetic_identity', 'sha256': 'a'*64}}),
        ]
        with patch.object(V2, '_ORIGINAL_SAVE') as saved:
            for path, value in records:
                V2._save_with_provenance(path, value, immutable=True)
                self.assertIn('ssl_precision_repair', saved.call_args.args[1])
                self.assertTrue(saved.call_args.kwargs['immutable'])
        self.assertEqual(records[-1][1]['settings'], {'path': 'synthetic_identity', 'sha256': 'a'*64})


if __name__ == '__main__':
    unittest.main()
