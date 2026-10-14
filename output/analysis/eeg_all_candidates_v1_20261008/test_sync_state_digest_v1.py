"""Byte-plan/fake-device tests, with optional synthetic CPU tensor checks.

No model, checkpoint, cohort, CUDA initialization, or live training is used.
CPU tensor tests exercise the candidate's explicit synthetic-only CPU option.
"""
from __future__ import annotations

import ast
from collections import namedtuple, OrderedDict
import hashlib
import importlib.util
from pathlib import Path
import random
import struct
import sys
import unittest

WORK = Path(__file__).resolve().parents[3]
SOURCE = WORK / 'tridep_ccml/eeg_all_candidates/sync_state_digest_v1.py'
spec = importlib.util.spec_from_file_location('_sync_digest_candidate_test', SOURCE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
CPU_TESTS = '--cpu-tensors' in sys.argv
if CPU_TESTS:
    sys.argv.remove('--cpu-tensors')

DeviceBase = namedtuple('DeviceBase', 'type index')


class Device(DeviceBase):
    def __str__(self):
        return self.type if self.index is None else self.type + ':' + str(self.index)


class FakeTensor:
    def __init__(self, data, *, dtype='torch.float32', shape=None, device=None,
                 width=4, log=None, failure=None, layout='strided', conj=False, neg=False):
        self.data, self.dtype = bytes(data), dtype
        self.shape = (len(data) // width,) if shape is None else tuple(shape)
        self.device = device or Device('cuda', 0)
        self.width, self.layout = width, layout
        self.log = [] if log is None else log
        self.failure, self.conj, self.neg = failure, conj, neg
        self.is_quantized = False

    def numel(self):
        return len(self.data) // self.width

    def element_size(self):
        return self.width

    def is_conj(self):
        return self.conj

    def is_neg(self):
        return self.neg

    def detach(self):
        self.log.append('detach')
        return self

    def contiguous(self):
        self.log.append('contiguous')
        return self

    def reshape(self, *shape):
        self.log.append('reshape')
        return self

    def view(self, dtype):
        self.log.append('view_bytes')
        assert dtype == 'torch.uint8'
        return FakeTensor(self.data, dtype=dtype, width=1, device=self.device, log=self.log)

    def cpu(self):
        self.log.append('host_copy')
        if self.failure == 'copy':
            raise RuntimeError('synthetic copy failure')
        return self

    def numpy(self):
        return self.data


class FakeTorch:
    Tensor = FakeTensor
    strided = 'strided'
    for _name in M.NUMPY_DTYPES:
        locals()[_name] = 'torch.' + _name

    def __init__(self, *, failure=None):
        self.log, self.failure = [], failure

    def cat(self, pieces, dim=0):
        self.log.append('cat')
        assert dim == 0
        if self.failure == 'pack':
            raise RuntimeError('synthetic allocation failure')
        return FakeTensor(b''.join(p.data for p in pieces), dtype=self.uint8,
                          width=1, device=pieces[0].device, log=self.log, failure=self.failure)


def reference_fake(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(tensor.dtype.encode())
        h.update(str(tensor.shape).encode()); h.update(tensor.data)
    return h.hexdigest()


class StandardLibraryTests(unittest.TestCase):
    def sample(self):
        return OrderedDict([('z.scalar', FakeTensor(struct.pack('<f', -0.), shape=())),
                            ('a.int', FakeTensor(struct.pack('<qq', 3, -7), dtype='torch.int64', width=8)),
                            ('m.empty', FakeTensor(b'', shape=(0, 2)))])

    def test_import_is_standard_library_only(self):
        if not CPU_TESTS:
            self.assertNotIn('torch', sys.modules)
            self.assertNotIn('numpy', sys.modules)

    def test_sorted_headers_mixed_bytes_and_single_transfer(self):
        t, events, state = FakeTorch(), [], self.sample()
        digest = M.packed_state_digest(state, torch_module=t, observer=events.append)
        self.assertEqual(digest, reference_fake(state))
        self.assertEqual(t.log, ['cat', 'host_copy'])
        self.assertEqual(events[-1]['cuda_to_cpu_copy_calls'], 1)
        self.assertEqual(events[-1]['tensor_count'], 3)
        self.assertEqual([e['event'] for e in events], ['pack_started', 'pack_returned',
                         'host_copy_started', 'host_copy_returned', 'digest_returned'])

    def test_packing_uses_detach_contiguous_byte_views_without_cast(self):
        state = self.sample(); original = [(v.data, v.dtype, v.shape) for v in state.values()]
        M.packed_state_digest(state, torch_module=FakeTorch())
        self.assertEqual(original, [(v.data, v.dtype, v.shape) for v in state.values()])
        for value in state.values():
            self.assertEqual(value.log, ['detach', 'contiguous', 'reshape', 'view_bytes'])

    def test_signed_zero_and_nan_payloads_are_raw_bytes(self):
        patterns = [0, 0x80000000, 0x7fc00001, 0x7fc00002, 0x7f800000]
        values = [M.packed_state_digest({'x': FakeTensor(struct.pack('<I', p))}, torch_module=FakeTorch()) for p in patterns]
        self.assertEqual(len(set(values)), len(patterns))

    def test_empty_state_and_empty_tensors_do_not_copy(self):
        for state in ({}, {'empty': FakeTensor(b'', shape=(0, 2))}):
            t, events = FakeTorch(), []
            self.assertEqual(M.packed_state_digest(state, torch_module=t, observer=events.append), reference_fake(state))
            self.assertEqual(t.log, [])
            self.assertEqual(events[-1]['host_copy_calls'], 0)

    def test_cpu_mixed_device_layout_and_dtype_fall_back_before_work(self):
        bad = [FakeTensor(b'\0'*4, device=Device('cpu', None)),
               FakeTensor(b'\0'*4, dtype='torch.bfloat16'),
               FakeTensor(b'\0'*4, layout='sparse'),
               FakeTensor(b'\0'*4, conj=True), FakeTensor(b'\0'*4, neg=True)]
        states = [{'x': x} for x in bad] + [{'x': FakeTensor(b'\0'*4), 'y': FakeTensor(b'\0'*4, device=Device('cuda', 1))}]
        for state in states:
            t, calls, events = FakeTorch(), [], []
            def original(value): calls.append(value); return 'original-result'
            digest = M.make_state_digest(original, torch_module=t, observer=events.append)
            self.assertEqual(digest(state), 'original-result')
            self.assertEqual(calls, [state]); self.assertEqual(t.log, [])
            self.assertTrue(all(not x.log for x in state.values()))
            self.assertEqual(events[0]['event'], 'original_fallback_started')

    def test_budget_and_invalid_budget_checked_before_copy(self):
        t, state = FakeTorch(), self.sample()
        with self.assertRaises(M.UnsupportedState): M.packed_state_digest(state, torch_module=t, max_pack_bytes=1)
        for budget in (-1, True, 1.5):
            with self.assertRaises(ValueError): M.make_state_digest(reference_fake, torch_module=t, max_pack_bytes=budget)
        self.assertFalse(t.log)

    def test_pack_and_copy_failure_are_not_retried_as_fallback(self):
        for failure in ('pack', 'copy'):
            t, events = FakeTorch(failure=failure), []
            def original(state): self.fail('Post-operation failure cannot fall back')
            digest = M.make_state_digest(original, torch_module=t, observer=events.append)
            with self.assertRaises(RuntimeError): digest(self.sample())
            self.assertNotIn('digest_returned', [x['event'] for x in events])
            self.assertNotIn('original_fallback_started', [x['event'] for x in events])
            if failure == 'copy':
                self.assertEqual(t.log.count('host_copy'), 1)
                self.assertNotIn('host_copy_returned', [x['event'] for x in events])

    def test_cpu_option_is_explicit_and_not_a_cuda_claim(self):
        state = {'x': FakeTensor(b'\0'*4, device=Device('cpu', None))}; events = []
        with self.assertRaises(M.UnsupportedState): M.packed_state_digest(state, torch_module=FakeTorch())
        self.assertEqual(M.packed_state_digest(state, torch_module=FakeTorch(), allow_cpu_for_synthetic_check=True,
                                             observer=events.append), reference_fake(state))
        self.assertEqual(events[-1]['cuda_to_cpu_copy_calls'], 0)

    def test_hash_rejects_gaps_truncation_and_trailing_bytes(self):
        row = {'name': 'x', 'dtype': 'torch.uint8', 'shape': (1,), 'offset': 0, 'nbytes': 1}
        for entries, data in [([{**row, 'offset': 1}], b'01'), ([row], b''), ([row], b'01')]:
            with self.assertRaises(ValueError): M.hash_packed_bytes(entries, data)


@unittest.skipUnless(CPU_TESTS, 'Optional explicit CPU-tensor synthetic checks')
class CpuTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        # Compile only the sealed pure digest function, not model/context code.
        path = WORK / 'tridep_ccml/tridep_stage1/common.py'
        node = next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'state_digest')
        env = {'hashlib': hashlib}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        cls.original = staticmethod(env['state_digest'])

    def check(self, state):
        t = self.torch
        import numpy as np
        snapshots = {name: value.detach().clone() for name, value in state.items()}
        versions = {name: value._version for name, value in state.items()}
        requires_grad = {name: value.requires_grad for name, value in state.items()}
        gradients = {name: None if value.grad is None else value.grad.clone() for name, value in state.items()}
        rng = t.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        result = M.packed_state_digest(state, torch_module=t, allow_cpu_for_synthetic_check=True)
        self.assertEqual(result, self.original(state))
        self.assertTrue(t.equal(rng, t.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], after_numpy[0])
        self.assertTrue(np.array_equal(numpy_rng[1], after_numpy[1]))
        self.assertEqual(numpy_rng[2:], after_numpy[2:])
        for name, value in state.items():
            self.assertEqual(value._version, versions[name])
            self.assertEqual(value.requires_grad, requires_grad[name])
            self.assertEqual(value.detach().contiguous().reshape(-1).view(t.uint8).numpy().tobytes(),
                             snapshots[name].contiguous().reshape(-1).view(t.uint8).numpy().tobytes())
            if gradients[name] is None:
                self.assertIsNone(value.grad)
            else:
                self.assertTrue(t.equal(value.grad, gradients[name]))
        return result

    def test_all_supported_dtypes_scalars_empty_and_requires_grad(self):
        t = self.torch
        state = {name: t.tensor([0, 1], dtype=getattr(t, name)) for name in M.NUMPY_DTYPES}
        state.update(scalar=t.tensor(-0., requires_grad=True), empty=t.empty((0, 2)),
                     grad=t.arange(12., requires_grad=True).reshape(3, 4).detach().requires_grad_())
        state['grad'].grad = t.full_like(state['grad'], 0.25)
        self.check(state)

    def test_noncontiguous_and_channels_last_match_original(self):
        t = self.torch
        self.check({'transpose': t.arange(30.).reshape(5, 6).t(),
                    'stride': t.arange(30.).reshape(5, 6)[:, ::2],
                    'channels_last': t.arange(48.).reshape(2, 3, 2, 4).contiguous(memory_format=t.channels_last)})

    def test_nan_signed_zero_exact_payload_bits(self):
        t = self.torch
        bits = t.tensor([0, -2147483648, 2143289345, 2143289346, 2139095040], dtype=t.int32)
        self.check({'bits': bits.view(t.float32)})

    def test_cpu_wrapper_calls_original_and_original_errors_remain(self):
        t = self.torch; calls = []
        def original(state): calls.append(True); return self.original(state)
        wrapper = M.make_state_digest(original, torch_module=t)
        state = {'x': t.tensor([1., 2.])}
        self.assertEqual(wrapper(state), self.original(state)); self.assertEqual(calls, [True])
        with self.assertRaises(TypeError): wrapper({'unsupported': t.tensor([1.], dtype=t.bfloat16)})


if __name__ == '__main__':
    unittest.main(verbosity=2)
