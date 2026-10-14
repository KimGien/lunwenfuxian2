"""Pure stdlib safety/contract tests; no torch, tensors, checkpoint or raw reads."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import validation_batching_v1 as B
SPEC = importlib.util.spec_from_file_location('qualification_test_subject', Path(__file__).with_name('qualify_acceleration_20261009.py'))
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def evidence_fixture():
    """In-memory fake evidence only; never materialize a qualification certificate."""
    def d(path):
        return {'path': str(path), 'sha256': 'a' * 64}
    split = {'train': ['a', 'b'], 'val': ['c'], 'test': ['d']}
    registry = {'folds': {'fold_1': split}, 'raw_sources': [{'subject_id': s, 'raw': d(s + '.npy')} for s in 'abcd']}
    sources = {k: d(p.relative_to(B.WORK).as_posix()) for k, p in {
        'diagnostic_wrapper_source': Path(B.__file__), 'qualifier_source': Path(Q.__file__),
        'original_diagnostics_source': B.F1, 'delegated_f2_source': B.F1.with_name('finetune_v2.py')}.items()}
    sources['original_diagnostics_source']['sha256'] = B.F1_SHA
    sources['delegated_f2_source']['sha256'] = B.F2_SHA
    registration = {'path': B.REGISTRY.relative_to(B.WORK).as_posix(), 'sha256': B.REGISTRY_SHA}
    cp, history, safe_d, trace_d, report_d = [d(v) for v in ('latest.pt', 'history.json', 'safe.json', 'attempt.json', 'report.json')]
    lock_d = d((B.REGISTRY.parent / 'engineering/C.json').relative_to(B.WORK).as_posix())
    contract = {'implementation_sources': list(sources.values())}
    lock = {'condition': 'C', 'registration': registration, 'contract': contract,
            'configuration_sha256': hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()}
    safe = {'schema': 'acceleration_safe_stop_v1', 'mode': 'stopped_training', 'source': d(B.SAFE_BUILDERS['stopped_training']), 'passed': True, 'gpu_idle': True,
            'condition': 'C', 'seed': 11, 'fold': 1, 'registration': registration, 'checkpoints': [cp], 'train_history': history}
    comparison = {'checkpoint': cp, 'passed': True, 'bitwise_logits': True, 'logits_max_abs': 0., 'ce_max_abs': 0.,
                  'policy_decisions_equal': True, 'state_unchanged': True, 'rng_restored': True, 'modes_restored': True}
    measurements = [{'checkpoint': cp, 'chunk': chunk, 'repetition': rep, 'warmup': rep == 0,
                     'synchronized_wall_seconds': seconds, 'gpu_event_interval_ms_not_busy_time': seconds * 1000,
                     'peak_allocated_bytes': 1024, 'state_unchanged': True, 'rng_restored': True, 'modes_restored': True}
                    for chunk, seconds in ((8, 4.), (16, 3.), (32, 2.), (60, 1.)) for rep in range(3)]
    candidates = [{'chunk': c, 'qualified': True, 'equivalence_level': 'bitwise', 'comparisons': [copy.deepcopy(comparison)],
                   'median_wall_seconds': seconds} for c, seconds in ((16, 3.), (32, 2.), (60, 1.))]
    report = {'schema': 'diagnostic_batching_qualification_v1', 'passed': True, 'condition': 'C', 'registration': registration,
              **sources, 'baseline_chunk': 8, 'optimizer_updates': 0, 'backward_calls': 0, 'test_forward_count': 0,
              'seed': 11, 'fold': 1, 'role_ids': split, 'weight_mode': 'stable_formal_checkpoint', 'probe_split': split,
              'raw_sources': {s: d(s + '.npy') for s in 'abc'}, 'condition_source_lock': lock_d,
              'actual_runtime': {'device_type': 'cuda', 'torch': 'fake-test-only'}, 'run_requested': True,
              'qualification_mode': 'bitwise_only', 'safe_stop_proof': safe_d, 'attempt_trace': trace_d,
              'checkpoints': [cp], 'train_history': history, 'selected_chunk': 60, 'candidates': candidates,
              'baseline_median_wall_seconds': 4., 'measurements': measurements, 'measured_repeats': 2,
              'selection_rule': 'fastest_qualified_median_wall_no_validation_performance_ranking'}
    trace = {'schema': 'diagnostic_batching_qualification_attempt_v1', 'status': 'complete', 'optimizer_updates': 0,
             'backward_calls': 0, 'test_forward_count': 0, 'forbidden_backward_attempts': 0, 'forbidden_optimizer_attempts': 0,
             'safe_stop_proof': safe_d, 'source_bindings': sources, 'gpu_stage_entered': True, 'seconds': 100., 'measurements': measurements}
    docs = {v['path']: content for v, content in ((report_d, report), (registration, registry), (lock_d, lock),
                                                  (safe_d, safe), (trace_d, trace), (history, []))}
    descriptors = {v['path']: v for v in [registration, safe['source'], *sources.values()]}
    def desc(path):
        rel = Path(path).resolve().relative_to(B.WORK).as_posix()
        return descriptors[rel]
    return report_d, report, trace, docs, desc


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.model = types.SimpleNamespace(window_chunk=8)
        self.raw = {s: types.SimpleNamespace(shape=(60, 19, 1000), dtype='float32') for s in 'ab'}
        self.split = {'train': ['a'], 'val': ['b'], 'test': ['c']}
        self.labels = {'a': 0, 'b': 1, 'c': 0}

    def test_context_restores_success(self):
        for chunk in B.CHUNKS:
            with B.diagnostic_chunk(self.model, self.raw, self.split, self.labels, chunk):
                self.assertEqual(self.model.window_chunk, chunk)
            self.assertEqual(self.model.window_chunk, 8)

    def test_context_restores_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with B.diagnostic_chunk(self.model, self.raw, self.split, self.labels, 32):
                raise RuntimeError('failure')
        self.assertEqual(self.model.window_chunk, 8)

    def test_bad_roles_shape_and_chunk_rejected_before_change(self):
        variants = []
        raw = dict(self.raw); raw['c'] = raw['a']; variants.append((raw, self.split, 16))
        split = copy.deepcopy(self.split); split['val'] = ['a']; variants.append((self.raw, split, 16))
        raw = dict(self.raw); raw['a'] = types.SimpleNamespace(shape=(8, 19, 1000), dtype='float32'); variants.append((raw, self.split, 16))
        variants.extend((self.raw, self.split, c) for c in (True, 7, 16.0))
        for raw, split, chunk in variants:
            with self.subTest(chunk=chunk, split=split), self.assertRaises(ValueError):
                with B.diagnostic_chunk(self.model, raw, split, self.labels, chunk):
                    self.fail('Entered invalid context')
            self.assertEqual(self.model.window_chunk, 8)

    def test_runtime_wrapper_preserves_arguments_return_and_failure(self):
        seen, boundaries = [], []
        def original(*args):
            seen.append(args); self.assertEqual(args[0].window_chunk, 32); return {'loss': 1.}
        with patch.object(B, 'verify_qualification', return_value={'selected_chunk': 32}):
            wrapper = B.make_diagnostics(original, qualification={}, condition='C', on_boundary=boundaries.append)
            self.assertEqual(wrapper(self.model, self.raw, self.split, self.labels), {'loss': 1.})
            self.assertEqual(seen, [(self.model, self.raw, self.split, self.labels)])
            self.assertEqual(boundaries, ['diagnostics_started', 'diagnostics_returned'])
            failing = B.make_diagnostics(lambda *a: (_ for _ in ()).throw(RuntimeError('failure')), qualification={}, condition='C')
            with self.assertRaises(RuntimeError):
                failing(self.model, self.raw, self.split, self.labels)
            self.assertEqual(self.model.window_chunk, 8)


class EvidenceTests(unittest.TestCase):
    def verify(self, mutate=None):
        binding, report, trace, docs, desc = evidence_fixture()
        if mutate:
            mutate(report, trace)
        with patch.object(B, 'read_bound', side_effect=lambda d: docs[d['path']]), patch.object(B, 'descriptor', side_effect=desc), \
             patch.object(Path, 'exists', return_value=False):
            return B.verify_qualification(binding, 'C')

    def test_valid_in_memory_shape(self):
        self.assertEqual(self.verify()['selected_chunk'], 60)

    def test_rejects_counter_bool_nonzero_failed_trace(self):
        for key in ('optimizer_updates', 'backward_calls', 'test_forward_count', 'forbidden_backward_attempts', 'forbidden_optimizer_attempts'):
            for bad in (False, 1):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    self.verify(lambda r, t: t.update({key: bad}))
        with self.assertRaises(ValueError):
            self.verify(lambda r, t: t.update(status='failed'))

    def test_rejects_nonbitwise_policy_and_state(self):
        for key, bad in [('bitwise_logits', False), ('ce_max_abs', 1e-15), ('logits_max_abs', 1e-15),
                         ('policy_decisions_equal', False), ('state_unchanged', False), ('rng_restored', False), ('modes_restored', False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(lambda r, t: r['candidates'][-1]['comparisons'][0].update({key: bad}))

    def test_rejects_raw_source_and_test_people(self):
        with self.assertRaises(ValueError):
            self.verify(lambda r, t: r['raw_sources'].update(d={'path': 'test.npy', 'sha256': 'a' * 64}))
        with self.assertRaises(ValueError):
            self.verify(lambda r, t: r['raw_sources']['a'].update(sha256='b' * 64))

    def test_rejects_nonfinite_bool_resource_or_missing_duplicate_measurement(self):
        for bad in (float('inf'), float('nan'), True, -1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.verify(lambda r, t: r['measurements'][0].update(synchronized_wall_seconds=bad))
        for mutate in (lambda r, t: r['measurements'].pop(), lambda r, t: r['measurements'].append(copy.deepcopy(r['measurements'][0]))):
            with self.assertRaises(ValueError):
                self.verify(mutate)

    def test_rejects_slower_selection_and_duplicate_checkpoint(self):
        with self.assertRaises(ValueError):
            self.verify(lambda r, t: r.update(selected_chunk=16))
        with self.assertRaises(ValueError):
            self.verify(lambda r, t: r['candidates'][-1]['comparisons'].append(copy.deepcopy(r['candidates'][-1]['comparisons'][0])))

    def test_rejects_tolerance_optin(self):
        with self.assertRaises(ValueError):
            B.verify_qualification({}, 'C', allow_tolerance=True)


class GateTests(unittest.TestCase):
    def test_default_pending_has_no_import_or_real_run(self):
        written = []
        with patch.object(Q, 'atomic_json', side_effect=lambda p, v, **kw: written.append(v)), \
             patch.object(Q, 'descriptor', return_value={'path': 'source.py', 'sha256': 'a' * 64}), \
             patch.object(Q, 'execute_explicit', side_effect=AssertionError('Must not run')):
            self.assertEqual(Q.main([]), 0)
        self.assertEqual(written[0]['torch_imports'], 0)
        self.assertEqual(written[0]['raw_reads'], 0)

    def test_missing_proof_cannot_enter_run(self):
        with patch.object(Q, 'execute_explicit', side_effect=AssertionError('Must not run')):
            with self.assertRaises(ValueError):
                Q.main(['--run'])

    def test_preflight_failure_retains_cost_without_gpu(self):
        written = []
        args = types.SimpleNamespace(stop_proof=Path('fake.json'), repeats=2)
        with patch.object(Q, 'json_read', return_value={'condition': 'C'}), \
             patch.object(Q, 'descriptor', return_value={'path': 'source.py', 'sha256': 'a' * 64}), \
             patch.object(Q, 'atomic_json', side_effect=lambda p, v, **kw: written.append(copy.deepcopy(v))), \
             patch.object(Q, 'preflight', side_effect=ValueError('GPU is occupied')):
            with self.assertRaises(ValueError):
                Q.execute_explicit(args)
        self.assertEqual([v['status'] for v in written], ['running', 'failed'])
        self.assertFalse(written[-1]['gpu_stage_entered'])
        self.assertGreaterEqual(written[-1]['seconds'], 0)
        self.assertEqual(written[-1]['optimizer_updates'], 0)

    def test_pure_modules_do_not_top_level_import_torch(self):
        for path in (Path(B.__file__), Path(Q.__file__)):
            module = ast.parse(path.read_text(encoding='utf-8'))
            for node in module.body:
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith('torch') for alias in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('torch'))
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
