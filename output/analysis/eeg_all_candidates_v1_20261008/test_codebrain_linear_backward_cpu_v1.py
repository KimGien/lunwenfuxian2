"""Synthetic CPU qualification only; this is never a GPU/model certificate."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
import torch
import torch.nn.functional as F
from tridep_ccml.eeg_all_candidates import codebrain_linear_backward_cpu_v1 as M

CASES = []


class CpuNativeDerivativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = torch.are_deterministic_algorithms_enabled()
        cls.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        cls.guard = patch.object(torch.cuda, '_lazy_init',
                                 side_effect=AssertionError('No GPU initialization permitted'))
        cls.guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop()
        torch.use_deterministic_algorithms(cls.saved, warn_only=cls.warn)

    def pair(self, length, factor, channels=3, strided=False):
        generator = torch.Generator(device='cpu').manual_seed(193 + length + factor)
        base = torch.randn(2, channels, length * (2 if strided else 1), generator=generator)
        first = (base[..., ::2] if strided else base).detach().requires_grad_()
        second = first.detach().clone().requires_grad_()
        actual = M.interpolate(first, scale_factor=factor, mode='linear')
        reference = F.interpolate(second, scale_factor=factor, mode='linear', align_corners=False)
        self.assertTrue(torch.equal(actual, reference))
        gradient = torch.randn(actual.shape, generator=generator)
        actual.backward(gradient); reference.backward(gradient)
        self.assertTrue(torch.equal(first.grad, second.grad))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        CASES.append({'length': length, 'factor': factor, 'channels': channels,
                      'strided': strided, 'forward_bitwise_equal_cpu': True,
                      'backward_bitwise_equal_cpu': True})

    def test_native_derivative_all_power_scales_and_boundaries(self):
        for length in (1, 2, 3, 8, 16, 31):
            for factor in (1, 2, 4, 8, 32):
                with self.subTest(length=length, factor=factor):
                    self.pair(length, factor)

    def test_noncontiguous_native_derivative(self):
        for factor in (1, 2, 8):
            self.pair(13, factor, strided=True)

    def test_author_kernel_channels(self):
        for factor in (1, 2, 4, 8, 16):
            self.pair(2, factor, channels=128)

    def test_repeat_determinism(self):
        value = torch.linspace(-1, 1, 24).reshape(2, 3, 4)
        outputs, gradients = [], []
        for _ in range(3):
            x = value.clone().requires_grad_()
            y = M.interpolate(x, scale_factor=8, mode='linear')
            y.square().sum().backward()
            outputs.append(y.detach()); gradients.append(x.grad)
        self.assertTrue(all(torch.equal(outputs[0], x) for x in outputs))
        self.assertTrue(all(torch.equal(gradients[0], x) for x in gradients))

    def test_no_grad_native_output(self):
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).requires_grad_()
        with torch.no_grad():
            y = M.interpolate(x, scale_factor=2, mode='linear')
            self.assertTrue(torch.equal(y, F.interpolate(x, scale_factor=2, mode='linear')))
            self.assertFalse(y.requires_grad)

    def test_private_namespace_does_not_patch_torch_global(self):
        original = F.interpolate
        namespace = M.functional_namespace()
        self.assertIs(F.interpolate, original)
        self.assertIs(namespace.interpolate, M.interpolate)
        self.assertIs(namespace.pad, F.pad)

    def test_unsupported_scope_rejected(self):
        x = torch.zeros(1, 2, 4)
        for kwargs in ({'scale_factor': 3}, {'size': 8}, {'scale_factor': 2, 'align_corners': True},
                       {'scale_factor': 2, 'recompute_scale_factor': True}, {'scale_factor': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                M.interpolate(x, mode='linear', **kwargs)
        with self.assertRaises(ValueError):
            M.interpolate(x.double(), scale_factor=2, mode='linear')

    def test_other_mode_native_passthrough(self):
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        self.assertTrue(torch.equal(M.interpolate(x, scale_factor=2, mode='nearest'),
                                    F.interpolate(x, scale_factor=2, mode='nearest')))


def descriptor(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(WORK).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    stream = io.StringIO(); started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CpuNativeDerivativeTests))
    value = {'schema': 'codebrain_native_forward_cpu_backward_synthetic_tests_v1',
             'passed': result.wasSuccessful(), 'tests': result.testsRun,
             'failures': len(result.failures), 'errors': len(result.errors),
             'observed_test_runner_seconds': time.perf_counter() - started,
             'source': descriptor(M.__file__), 'test_source': descriptor(__file__),
             'exact_cpu_forward_and_derivative_cases': CASES,
             'scope': 'Synthetic CPU FP32 tensors only, no author model, weights, cohort or GPU',
             'model_constructions': 0, 'checkpoint_loads': 0, 'raw_arrays_loaded': 0,
             'GPU_operations': 0, 'optimizer_steps': 0, 'test_probability_reads': 0,
             'production_integration_applied': False, 'GPU_qualification_complete': False,
             'final_plan_certificate': False, 'test_output': stream.getvalue()}
    target = WORK / 'output/analysis/eeg_all_candidates_v1_20261008' / (
        'codebrain_linear_cpu_backward_synthetic_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'passed': value['passed'], 'tests': result.testsRun,
                      'exact_cpu_cases': len(CASES), 'artifact': descriptor(target),
                      'test_output': stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
