"""Pure synthetic control/checkpoint checks; no OS process or tensor runtime."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
spec = importlib.util.spec_from_file_location('safe_stop_under_test', Path(__file__).with_name('safe_stop_acceleration_v1.py'))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


class Tensor:
    def __init__(self, values, shape=None, dtype='float32'):
        self.values = list(values)
        self.shape = shape if shape is not None else (len(self.values),)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type='cpu')

    def numel(self):
        return math.prod(self.shape) if self.shape else 1

    def __float__(self):
        if len(self.values) != 1:
            raise ValueError('not scalar')
        return float(self.values[0])


def digest(state):
    value = {name: {'values': tensor.values, 'shape': tensor.shape, 'dtype': tensor.dtype}
             for name, tensor in state.items()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rng_set(value):
    if value != ('synthetic_numpy_rng',):
        raise ValueError('bad numpy state')


def torch_rng_set(value):
    if value.shape != (4,) or value.dtype != 'uint8':
        raise ValueError('bad torch CPU state')


TORCH = types.SimpleNamespace(
    uint8='uint8', is_tensor=lambda value: isinstance(value, Tensor),
    isfinite=lambda value: types.SimpleNamespace(all=lambda: all(math.isfinite(n) for n in value.values)),
    equal=lambda a, b: (a.values, a.shape, a.dtype) == (b.values, b.shape, b.dtype),
    Generator=lambda **kwargs: types.SimpleNamespace(set_state=torch_rng_set))
F = types.SimpleNamespace(state_digest=digest,
    bag_indices=lambda epoch: types.SimpleNamespace(tolist=lambda: [((epoch - 1) % 8 * 8 + i) % 60 for i in range(8)]),
    np=types.SimpleNamespace(random=types.SimpleNamespace(RandomState=lambda: types.SimpleNamespace(set_state=rng_set))))


class CheckpointTests(unittest.TestCase):
    def fixture(self, epoch=11):
        condition, seed, fold = 'CSBrain-lastblock-finetune', 22, 4
        lock, registry = {'path': 'synthetic_lock.json'}, {'path': 'synthetic_registry.json'}
        registration = {'folds': {'fold_4': {'train': ['a', 'b'], 'val': ['c'], 'test': ['d']}},
                        'raw_sources': [{'subject_id': sid, 'raw': {'path': sid + '.npy'}} for sid in 'abcd']}
        contract = {'trainable_parameter_names': ['encoder.selected', 'head.weight'],
                    'frozen_parameter_names': ['encoder.frozen']}
        state = {'encoder.selected': Tensor([1., 2.]), 'head.weight': Tensor([3., 4.]),
                 'encoder.frozen': Tensor([5., 6.]), 'encoder.buffer': Tensor([7.])}
        initial = {'condition': condition, 'seed': seed, 'fold': fold, 'registry': registry, 'condition_source_lock': lock,
            'raw_sources': {sid: {'path': sid + '.npy'} for sid in 'abc'}, 'state_dict': state,
            'epoch': 0, 'optimizer_updates': 0, 'backward_calls': 0, 'history': [], 'subject_order': [],
            'best_selected_state_dict': None, 'best_selected_state_sha256': None,
            'sampler_position': {'epoch_completed': 0, 'next_batch': 0},
            'scheduler_state_dict': {'policy': 'finetune', 'completed_epoch': 0, 'extended': False},
            'initial_state_sha256': digest(state),
            'frozen_initial_sha256': digest({k: v for k, v in state.items() if k not in contract['trainable_parameter_names']}),
            'optimizer_state_dict': {'param_groups': [], 'state': {}},
            'rng': {'python': random.Random(1).getstate(), 'numpy': ('synthetic_numpy_rng',),
                    'torch_cpu': Tensor([1, 2, 3, 4], dtype='uint8'), 'torch_cuda': [Tensor([1, 2], dtype='uint8')]}}
        for name, params, lr in [('head', [0], 3e-4), ('backbone', [1], 0.)]:
            initial['optimizer_state_dict']['param_groups'].append({'group_name': name, 'params': params,
                'lr': lr, 'weight_decay': .01, 'betas': (.9, .999), 'eps': 1e-8})
        saved = copy.deepcopy(initial)
        saved.update(epoch=epoch, optimizer_updates=epoch, backward_calls=2 * epoch,
                     sampler_position={'epoch_completed': epoch, 'next_batch': 0},
                     scheduler_state_dict={'policy': 'finetune', 'completed_epoch': epoch, 'extended': False})
        saved['history'] = [{'epoch': i, 'optimizer_updates': i, 'backward_calls': 2 * i,
                             'window_indices': F.bag_indices(i).tolist(), 'unfrozen': i > 10,
                             'train_bag_ce': 1., 'train_eval_ce': 1., 'val_eval_ce': 1. - i * .001} for i in range(1, epoch + 1)]
        saved['subject_order'] = [['a', 'b'] for _ in range(epoch)]
        saved['best_selected_state_dict'] = {name: copy.deepcopy(state[name]) for name in contract['trainable_parameter_names']}
        saved['best_selected_state_sha256'] = digest(saved['best_selected_state_dict'])
        for pid, group in enumerate(saved['optimizer_state_dict']['param_groups']):
            base = H.P.POLICIES['finetune'].head_lr if pid == 0 else H.P.POLICIES['finetune'].backbone_lr
            group['lr'] = base * H.P.learning_rate_multiplier('finetune', epoch, parameter_group=group['group_name'])
            steps = epoch if pid == 0 else max(0, epoch - 10)
            if steps:
                saved['optimizer_state_dict']['state'][pid] = {'step': Tensor([steps], shape=()),
                    'exp_avg': Tensor([0., 0.]), 'exp_avg_sq': Tensor([0., 0.])}
        kwargs = dict(contract=contract, registration=registration, condition=condition, seed=seed,
                      fold=fold, lock=lock, registry=registry, torch=TORCH, F=F)
        return initial, saved, kwargs

    def verify(self, initial, saved, kwargs):
        return H.validate_checkpoint_payloads(initial, saved, **kwargs)

    def test_valid_terminal_and_head_warmup_partial(self):
        for epoch in (4, 11):
            with self.subTest(epoch=epoch):
                result = self.verify(*self.fixture(epoch))
                self.assertEqual(result['epoch'], epoch)
                self.assertTrue(result['resume_full_state_verified'])

    def test_missing_or_duplicated_group_parameter_is_rejected(self):
        for parameters in ([], [0], [1, 1]):
            initial, saved, args = self.fixture()
            saved['optimizer_state_dict']['param_groups'][1]['params'] = parameters
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.verify(initial, saved, args)

    def test_wrong_moment_shape_or_dtype_rejected(self):
        for moment in (Tensor([0.], shape=(1,)), Tensor([0., 0.], dtype='float64')):
            initial, saved, args = self.fixture()
            saved['optimizer_state_dict']['state'][0]['exp_avg'] = moment
            with self.subTest(moment=moment), self.assertRaises(ValueError):
                self.verify(initial, saved, args)

    def test_nonfinite_best_even_matching_digest_is_rejected(self):
        initial, saved, args = self.fixture()
        saved['best_selected_state_dict']['head.weight'].values[0] = float('nan')
        saved['best_selected_state_sha256'] = digest(saved['best_selected_state_dict'])
        with self.assertRaises(ValueError):
            self.verify(initial, saved, args)

    def test_frozen_buffer_change_rejected(self):
        initial, saved, args = self.fixture()
        saved['state_dict']['encoder.buffer'].values[0] = 9.
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            self.verify(initial, saved, args)

    def test_missing_rng_state_and_cuda_shape_rejected(self):
        for mutation in ('missing', 'cpu_shape', 'cuda_shape', 'numpy'):
            initial, saved, args = self.fixture()
            if mutation == 'missing':
                del saved['rng']['python']
            elif mutation == 'cpu_shape':
                saved['rng']['torch_cpu'] = Tensor([1], dtype='uint8')
            elif mutation == 'cuda_shape':
                saved['rng']['torch_cuda'] = [Tensor([1], dtype='uint8')]
            else:
                saved['rng']['numpy'] = ('bad',)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.verify(initial, saved, args)

    def test_test_raw_descriptor_injection_rejected(self):
        initial, saved, args = self.fixture()
        saved['raw_sources']['d'] = {'path': 'd.npy'}
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.verify(initial, saved, args)

    def test_extra_optimizer_state_rejected(self):
        initial, saved, args = self.fixture()
        saved['optimizer_state_dict']['state'][9] = saved['optimizer_state_dict']['state'][0]
        with self.assertRaisesRegex(ValueError, 'IDs'):
            self.verify(initial, saved, args)

    def test_history_boolean_counter_and_invalid_extension_rejected(self):
        for change in ('counter', 'extension'):
            initial, saved, args = self.fixture()
            if change == 'counter':
                saved['history'][0]['optimizer_updates'] = True
            else:
                saved['scheduler_state_dict']['extended'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify(initial, saved, args)


class Function:
    def __init__(self, call):
        self.call = call
    def __call__(self, *args):
        return self.call(*args)


class ScopedControlTests(unittest.TestCase):
    def setUp(self):
        self.refs = [{'role': role, 'handle': role} for role in ('dispatcher', 'launcher', 'gpu_child')]
        self.calls = []
        self.native = types.SimpleNamespace(kernel=types.SimpleNamespace(
            TerminateProcess=lambda handle, code: self.calls.append(('terminate', handle)) or True,
            WaitForSingleObject=lambda handle, milliseconds: self.calls.append(('wait', handle)) or 0),
            ntdll=types.SimpleNamespace(NtResumeProcess=Function(lambda handle: self.calls.append(('resume', handle)) or 0)))

    def test_partial_termination_forbids_old_tree_resume(self):
        self.native.kernel.TerminateProcess = lambda handle, code: self.calls.append(('terminate', handle)) or handle != 'launcher'
        state = {}
        with patch.object(H.S, 'save'):
            with self.assertRaisesRegex(ValueError, 'termination'):
                H.terminate_held(self.native, self.refs, state, 'synthetic')
        self.assertTrue(state['termination_started'])
        self.assertEqual(state['exit_verified_roles'], ['gpu_child'])
        self.assertEqual(state['termination_request_inflight_role'], 'launcher')
        with self.assertRaises(ValueError):
            H.resume_unterminated_paused(self.native, self.refs, state)
        self.assertFalse(any(kind == 'resume' for kind, role in self.calls))

    def test_timeout_after_first_termination_is_not_healthy_old_training(self):
        self.native.kernel.WaitForSingleObject = lambda *args: 258
        state = {}
        with patch.object(H.S, 'save'), self.assertRaises(ValueError):
            H.terminate_held(self.native, self.refs, state, 'synthetic')
        self.assertTrue(state['termination_started'])
        self.assertEqual(state['termination_requested_roles'], ['gpu_child'])
        self.assertEqual(state['exit_verified_roles'], [])

    def test_first_api_return_failure_still_forbids_automatic_resume(self):
        self.native.kernel.TerminateProcess = lambda *args: False
        state = {}
        with patch.object(H.S, 'save'), self.assertRaises(ValueError):
            H.terminate_held(self.native, self.refs, state, 'synthetic')
        self.assertTrue(state['termination_started'])
        self.assertEqual(state['termination_requested_roles'], [])

    def test_normal_termination_proves_each_exit_child_first(self):
        state = {}
        writes = []
        with patch.object(H.S, 'save', side_effect=lambda path, value: writes.append(copy.deepcopy(value))):
            H.terminate_held(self.native, self.refs, state, 'synthetic')
        self.assertEqual(state['exit_verified_roles'], ['gpu_child', 'launcher', 'dispatcher'])
        self.assertIsNone(state['termination_request_inflight_role'])
        self.assertEqual(writes[0]['termination_requested_roles'], [])

    def test_resume_failure_does_not_resume_launcher_or_dispatcher(self):
        self.native.ntdll.NtResumeProcess = Function(lambda handle: self.calls.append(('resume', handle)) or 1)
        state = {}
        self.assertFalse(H.resume_unterminated_paused(self.native, self.refs, state))
        self.assertEqual(self.calls, [('resume', 'gpu_child')])
        self.assertEqual(state['resume_failed_role'], 'gpu_child')

    def test_pretermination_failure_can_resume_children_before_dispatcher(self):
        state = {}
        self.assertTrue(H.resume_unterminated_paused(self.native, self.refs, state))
        self.assertEqual(state['resume_returned_roles'], ['gpu_child', 'launcher', 'dispatcher'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
