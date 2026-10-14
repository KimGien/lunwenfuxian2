"""No tensor runtime: test host sum order, guard AST, scope and fake diagnostics."""
import ast
from contextlib import nullcontext
import math
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import sync_reduction_v1 as M
from tridep_ccml.eeg_all_candidates import sync_qualification_gate_v1 as G


class Tensor:
    dtype = 'float32'
    transfers = 0
    def __init__(self, value): self.value = value
    def detach(self): return self
    def to(self, **kwargs): return self
    def cpu(self): Tensor.transfers += 1; return self
    def tolist(self): return self.value


class Raw:
    shape, dtype = (60, 19, 1000), 'float32'
    def __init__(self, sid): self.sid = sid
    def __getitem__(self, key): return self.sid


def fake_module():
    def stack(values, dim=0):
        values = [v.value for v in values]
        return Tensor(values if dim == 0 else [list(v) for v in zip(*values)])
    torch = types.SimpleNamespace(inference_mode=nullcontext, stack=stack,
        tensor=lambda value, **kw: value, isfinite=lambda x: Tensor(float(math.isfinite(x.value))),
        nn=types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=lambda logits, target: Tensor(logits))))
    rng = {'value': 10}
    return types.SimpleNamespace(torch=torch, state_digest=lambda state: state['value'], rng_state=lambda device: rng['value'],
        restore_rng=lambda value: rng.update(value=value), rng_equal=lambda a, b: a == b, rng=rng)


class FakeModel:
    window_chunk = 8
    device = 'cuda-fake'
    def __init__(self, F): self.training = True; self.F = F; self.state = {'value': 'unchanged'}; self.calls = []
    def modules(self): return [self]
    def state_dict(self): return self.state
    def eval(self): self.training = False
    def __call__(self, raw):
        self.calls.append(raw); self.F.rng['value'] += 1
        return {'a': 1., 'b': 2., 'c': 3.}[raw]


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.F = fake_module(); self.model = FakeModel(self.F)
        self.split = {'train': ['a', 'b'], 'val': ['c'], 'test': ['d']}
        self.raw = {s: Raw(s) for s in 'abc'}; self.labels = dict(a=0, b=1, c=0, d=1)
        self.observed = []; Tensor.transfers = 0

    def make(self):
        return M._make_diagnostics(self.F, defer_representation=False, observer=self.observed.append)

    def test_two_host_transfers_exact_roles_and_sum(self):
        output = self.make()(self.model, self.raw, self.split, self.labels)
        self.assertEqual(output, {'train_eval_ce': 1.5, 'val_eval_ce': 3.})
        self.assertEqual(self.model.calls, ['a', 'b', 'c'])
        self.assertEqual(Tensor.transfers, 2)
        self.assertTrue(self.model.training); self.assertEqual(self.F.rng['value'], 10)
        self.assertTrue(self.observed[-1]['all_finite_checks_passed_before_return'])

    def test_nonfinite_delayed_until_role_and_never_returned(self):
        self.F.torch.nn.functional.cross_entropy = lambda logits, target: Tensor(float('nan') if logits == 1 else logits)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.make()(self.model, self.raw, self.split, self.labels)
        self.assertEqual(self.model.calls, ['a', 'b'])
        self.assertTrue(self.model.training); self.assertEqual(self.F.rng['value'], 10)
        self.assertFalse(self.observed[-1]['all_finite_checks_passed_before_return'])

    def test_outer_test_raw_or_role_overlap_rejected_before_forward(self):
        self.raw['d'] = Raw('d')
        with self.assertRaises(ValueError): self.make()(self.model, self.raw, self.split, self.labels)
        self.assertEqual(self.model.calls, [])

    def test_state_mutation_rejected(self):
        def changed(logits, target): self.model.state['value'] = 'mutated'; return Tensor(logits)
        self.F.torch.nn.functional.cross_entropy = changed
        with self.assertRaisesRegex(ValueError, 'mutated'):
            self.make()(self.model, self.raw, self.split, self.labels)
        self.assertFalse(self.observed[-1]['state_unchanged_on_exit'])
        self.assertTrue(self.model.training); self.assertEqual(self.F.rng['value'], 10)

    def test_failure_restores_modes_rng_and_does_not_allow_batch_change(self):
        self.model.window_chunk = 16
        with self.assertRaisesRegex(ValueError, 'window_chunk8'):
            self.make()(self.model, self.raw, self.split, self.labels)
        self.assertEqual(self.model.calls, [])

    def test_optional_digest_is_local(self):
        calls = []
        original = self.F.state_digest
        candidate = M._make_diagnostics(self.F, defer_representation=False, state_digest=lambda state: calls.append(state.copy()) or state['value'])
        candidate(self.model, self.raw, self.split, self.labels)
        self.assertIs(self.F.state_digest, original)
        self.assertEqual(len(calls), 2)

    def test_all_targets_copied_once_then_original_person_views(self):
        transfers, targets = [], []
        self.F.torch.tensor = lambda values, **kwargs: transfers.append(list(values)) or list(values)
        self.F.torch.nn.functional.cross_entropy = lambda logits, target: targets.append(list(target)) or Tensor(logits)
        candidate = M._make_diagnostics(self.F, defer_representation=False, coalesce_targets=True)
        self.assertEqual(candidate(self.model, self.raw, self.split, self.labels), {'train_eval_ce': 1.5, 'val_eval_ce': 3.})
        self.assertEqual(transfers, [[0, 1, 0]])
        self.assertEqual(targets, [[0], [1], [0]])


class PureSafetyTests(unittest.TestCase):
    def test_python_sequential_not_fsum_or_gpu_reduce(self):
        values = [1e16, 1., 1., 1.]
        expected = 0.
        for value in values: expected += float(value)
        self.assertEqual(M.sequential_role_ce([[v, 1., 1.] for v in values], 4), expected / 4)
        self.assertNotEqual(expected, math.fsum(values))

    def test_all_finite_flags_checked(self):
        for rows in ([[1., 0., 1.]], [[1., 1., 0.]], [[float('inf'), 1., 1.]], [[float('nan'), 1., 1.]], []):
            with self.assertRaises(ValueError): M.sequential_role_ce(rows, 1)

    def test_forward_ast_changes_only_guard_sink(self):
        original, derived = M.forward_ast()
        changed = [n for n in ast.walk(derived) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == '_sync_capture_representation_finite']
        self.assertEqual(len(changed), 1)
        changed[0].func.id = 'require'
        changed[0].args.append(ast.Constant(value='Encoder returned nonfinite representations'))
        self.assertEqual(ast.dump(original), ast.dump(derived))

    def test_opaque_forward_observer_not_bypassed(self):
        class Base:
            def forward(self, raw): return raw
        F = types.SimpleNamespace(RawSubjectModel=Base)
        model = Base(); model.window_chunk = 8; old = lambda raw: raw; model.forward = old
        with self.assertRaisesRegex(ValueError, 'observer'):
            with M.deferred_representation_guard(model, F, []): self.fail('Must reject')
        self.assertIs(model.forward, old)

    def test_input_transfer_ast_keeps_exact_original_encoder_loop_and_head(self):
        original, derived = M.forward_ast(coalesce_input_transfer=True)
        old_loop = next(n for n in original.body if isinstance(n, ast.For))
        new_loop = next(n for n in derived.body if isinstance(n, ast.For))
        self.assertEqual(ast.dump(old_loop.iter), ast.dump(new_loop.iter))
        self.assertEqual(ast.dump(old_loop.body[1]), ast.dump(new_loop.body[1]))
        self.assertEqual(ast.dump(original.body[-1]), ast.dump(derived.body[-1]))
        assignment = next(n for n in derived.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                          and n.targets[0].id == '_sync_adapted_device')
        self.assertEqual(ast.unparse(assignment), '_sync_adapted_device = torch.as_tensor(adapted, device=self.device)')
        self.assertEqual(ast.unparse(new_loop.body[0]), 'tensor = _sync_adapted_device[start:start + self.window_chunk]')

    def test_deferred_context_restores_on_exception_and_reattaches_observer(self):
        class Base:
            def forward(self, raw): return raw
        F = types.SimpleNamespace(RawSubjectModel=Base)
        model = Base(); model.window_chunk = 8
        dummy_ast = ast.parse('def forward(self, raw):\n return raw').body[0]
        old = model.forward
        wrappers = []
        with patch.object(M, 'forward_ast', return_value=(dummy_ast, dummy_ast)):
            with self.assertRaises(RuntimeError):
                with M.deferred_representation_guard(model, F, [], forward_observer=lambda f: wrappers.append(f) or f):
                    self.assertEqual(model.forward('same'), 'same')
                    raise RuntimeError('failure')
        self.assertEqual(model.forward, old)
        self.assertNotIn('forward', model.__dict__)
        self.assertEqual(len(wrappers), 1)

    def test_no_tensor_import_or_global_tensor_bool_patch(self):
        self.assertNotIn('torch', sys.modules); self.assertNotIn('numpy', sys.modules)
        source = M.SOURCE.read_text(encoding='utf-8')
        self.assertNotIn('Tensor.__bool__', source)
        with self.assertRaises(ValueError): M.make_diagnostics(lambda: None)


class BoundaryTests(unittest.TestCase):
    def fixture(self):
        split = {'train': ['a'], 'val': ['b'], 'test': ['c']}
        original, q4 = {'path': 'original.json', 'sha256': 'a'}, {'path': 'q4.json', 'sha256': 'b'}
        rows = [{'fit_id': f'CSBrain-lastblock-finetune/seed_{seed}/fold_{fold}', 'condition': 'CSBrain-lastblock-finetune'}
                for seed in (11, 22, 33) for fold in range(1, 6)]
        proof = {'schema': 'sync_qualification_safe_boundary_v1', 'passed': True, 'queue': 'continuation_queue_v4',
                 'registration': q4, 'original_training_registration': original, 'gpu_children_exit_verified': True,
                 'held_dispatcher_running_suspended': True, 'gpu_idle': True, 'zero_optimizer_probe_only': True,
                 'completed_CSBrain_fits_verified': 15, 'condition': 'CSBrain-lastblock-finetune', 'seed': 22, 'fold': 5,
                 'registered_fold_ids': split}
        return proof, {'fits': rows}, q4, original, split

    def test_natural_boundary_shape_accepts_only_complete15_correct_hold(self):
        fixture = self.fixture()
        self.assertEqual(len(G.verify_shape(*fixture)), 15)
        for key, bad in [('completed_CSBrain_fits_verified', 14), ('held_dispatcher_running_suspended', False),
                         ('gpu_children_exit_verified', False), ('gpu_idle', False), ('zero_optimizer_probe_only', False),
                         ('seed', 11), ('fold', True), ('queue', 'continuation_queue_v3')]:
            changed = dict(fixture[0]); changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): G.verify_shape(changed, *fixture[1:])

    def test_default_qualifier_does_not_enter_preflight_or_gpu(self):
        path = Path(__file__).with_name('qualify_sync_reduction_v1.py')
        spec = importlib.util.spec_from_file_location('_sync_qualifier_test', path)
        Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
        self.assertTrue(Q.exact_ce({'loss': 1.25}, {'loss': 1.25}))
        self.assertFalse(Q.exact_ce({'loss': 0.0}, {'loss': -0.0}))
        self.assertFalse(Q.exact_ce({'loss': float('nan')}, {'loss': float('nan')}))
        writes = []
        with patch.object(Q, 'save', side_effect=lambda p, v, **kw: writes.append(v)), \
             patch.object(Q, 'desc', return_value={'path': 'fixture', 'sha256': 'a'}), \
             patch.object(Q, 'qualify', side_effect=AssertionError('No actual qualification')):
            self.assertEqual(Q.main([]), 0)
        self.assertEqual(writes[0]['GPU'], 0); self.assertEqual(writes[0]['checkpoint_reads'], 0)
        self.assertNotIn('torch', sys.modules)

    def test_failed_preflight_retains_attempt_before_any_tensor_import(self):
        path = Path(__file__).with_name('qualify_sync_reduction_v1.py')
        spec = importlib.util.spec_from_file_location('_sync_qualifier_failed_test', path)
        Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
        writes = []
        with patch.object(Q, 'save', side_effect=lambda p, v, **kw: writes.append(dict(v))), \
             patch.object(Q, 'desc', return_value={'path': 'fixture', 'sha256': 'a'}), \
             patch.object(Q, 'preflight', side_effect=ValueError('boundary not ready')):
            with self.assertRaisesRegex(ValueError, 'not ready'): Q.qualify({}, {})
        self.assertEqual([row['status'] for row in writes], ['running', 'failed'])
        self.assertFalse(writes[-1]['gpu_stage_entered']); self.assertEqual(writes[-1]['optimizer_updates'], 0)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
