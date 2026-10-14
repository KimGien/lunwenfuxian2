"""Synthetic standard-library tests. No torch/model/raw/checkpoint is loaded."""
import copy
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import finetune_v3 as F3


class BufferedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=WORK / 'output/analysis/eeg_all_candidates_v1_20261008')
        self.out = Path(self.tmp.name)
        self.out_patch = patch.object(F3.S, 'OUT', self.out)
        self.out_patch.start()
        self.execution = {
            'condition': 'synthetic', 'seed': 11, 'fold': 1,
            'source': F3.S.desc(F3.SOURCE),
            'delegated_finetune_implementation': {
                'version': 2, 'wrapper_source': {'path': 'fake_f2.py', 'sha256': 'synthetic'},
                'delegated_runner_source': {'path': 'fake_f1.py', 'sha256': 'synthetic'}},
            'diagnostics': {'window_chunk': 8, 'qualification': None},
        }
        self.saved = []
        self.real_save = F3.S.save

        def writer(path, value, **kwargs):
            self.saved.append((Path(path), copy.deepcopy(value), kwargs))
            return self.real_save(path, value, **kwargs)
        self.save_patch = patch.object(F3.S, 'save', writer)
        self.save_patch.start()

    def tearDown(self):
        self.save_patch.stop()
        self.out_patch.stop()
        self.tmp.cleanup()

    def attempt(self):
        return F3.BufferedAttempt('synthetic', 11, 1, self.execution)

    def fake_dependencies(self, attempt):
        model = types.SimpleNamespace(window_chunk=8)
        observed_chunks = []
        folder = self.out / 'runs/synthetic/seed_11/fold_1'
        v1 = types.SimpleNamespace(R=types.SimpleNamespace(fit_directory=lambda *args: folder))

        def update(model, *args, **kwargs):
            observed_chunks.append(('train', model.window_chunk))
            attempt.observe('forward', lambda: None)()
            attempt.observe('backward', lambda: None)()
            attempt.observe('optimizer_step', lambda: None)()
            return {'unchanged': args, 'micro_subjects': kwargs['micro_subjects']}

        def diagnostics(model, *args, **kwargs):
            observed_chunks.append(('diagnostics', model.window_chunk))
            attempt.observe('forward', lambda: None)()
            return {'train_eval_ce': 1., 'val_eval_ce': 2.}

        v1.subject_update, v1.diagnostics = update, diagnostics
        v2 = types.SimpleNamespace(V1=v1, _ACTIVE=False, _TRACE=None)

        @contextmanager
        def activate():
            original_save = F3.S.save
            v2._ACTIVE = True

            def f2_save(path, value, **kwargs):
                if Path(path).name == 'metadata.json':
                    value['versioned_finetune_implementation'] = copy.deepcopy(
                        self.execution['delegated_finetune_implementation'])
                return original_save(path, value, **kwargs)
            F3.S.save = f2_save
            try:
                yield
            finally:
                F3.S.save = original_save
                v2._ACTIVE = False

        v2.activate = activate
        batching = types.SimpleNamespace()

        def make_diagnostics(original, *, qualification, condition):
            def wrapped(model, *args, **kwargs):
                old = model.window_chunk
                model.window_chunk = 32
                try:
                    return original(model, *args, **kwargs)
                finally:
                    model.window_chunk = old
            return wrapped
        batching.make_diagnostics = make_diagnostics
        return v2, batching, model, observed_chunks, folder

    def test_import_has_no_tensor_dependency(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys; from tridep_ccml.eeg_all_candidates import finetune_v3; '
            'assert "torch" not in sys.modules; assert "numpy" not in sys.modules'],
            cwd=WORK, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_many_calls_are_counted_without_per_call_persistence(self):
        trace = self.attempt()
        forward = trace.observe('forward', lambda value: value + 1)
        for number in range(100):
            self.assertEqual(forward(number), number + 1)
        self.assertEqual(len(self.saved), 1)
        self.assertTrue(trace.dirty)
        on_disk = F3.S.read(trace.path)
        self.assertEqual(on_disk['forward_returned'], 0)
        self.assertTrue(on_disk['unflushed_tail_unknown'])
        self.assertIsNone(on_disk['unflushed_tail_counters'])
        trace.flush('diagnostics_after')
        self.assertEqual(len(self.saved), 2)
        self.assertEqual(F3.S.read(trace.path)['forward_returned'], 100)
        self.assertEqual(trace.record['durable_counter_snapshot']['forward_returned'], 100)
        self.assertFalse(trace.dirty)

    def test_encoder_alias_and_window_counts_do_not_duplicate_calls(self):
        trace = self.attempt()
        result = trace.observe('encoder_forward', lambda model, windows: len(windows))
        self.assertEqual(result(None, list(range(8))), 8)
        self.assertEqual(result(None, list(range(3))), 3)
        self.assertEqual(len(self.saved), 1)
        trace.finish()
        for suffix in ('started', 'returned'):
            self.assertEqual(trace.record['encoder_forward_' + suffix], 2)
            self.assertEqual(trace.record['encode_' + suffix], 2)
            self.assertEqual(trace.record['encoder_windows_' + suffix], 11)
        self.assertEqual(len(self.saved), 2)

    def test_exception_preserves_started_without_claiming_return(self):
        trace = self.attempt()
        error = RuntimeError('synthetic interrupted callable')
        def fail():
            raise error
        with self.assertRaises(RuntimeError):
            trace.observe('optimizer_step', fail)()
        trace.finish(error)
        self.assertEqual(trace.record['status'], 'failed')
        self.assertEqual(trace.record['optimizer_step_started'], 1)
        self.assertEqual(trace.record['optimizer_step_returned'], 0)
        self.assertEqual(trace.record['started_without_observed_return']['optimizer_step'], 1)
        self.assertTrue(trace.record['counter_snapshot_exact_for_process'])
        self.assertFalse(trace.record['unflushed_tail_unknown'])
        self.assertTrue(all(value == 0 for value in trace.record['unflushed_tail_counters'].values()))

    def test_false_complete_after_swallowed_call_exception_is_rejected(self):
        trace = self.attempt()
        def fail():
            raise ValueError('unfinished')
        with self.assertRaises(ValueError):
            trace.observe('forward', fail)()
        with self.assertRaisesRegex(ValueError, 'unreturned'):
            trace.finish()
        self.assertEqual(F3.S.read(trace.path)['status'], 'running')

    def test_storage_failure_never_advances_durable_snapshot(self):
        trace = self.attempt()
        trace.observe('forward', lambda: None)()
        trace._save = lambda *args, **kwargs: (_ for _ in ()).throw(OSError('disk failed'))
        with self.assertRaises(OSError):
            trace.finish()
        self.assertFalse(trace.terminal)
        self.assertEqual(trace.record['flush_sequence'], 1)
        self.assertEqual(F3.S.read(trace.path)['forward_returned'], 0)
        self.assertTrue(F3.S.read(trace.path)['unflushed_tail_unknown'])

    def test_zero_call_successful_recheck_is_exact(self):
        trace = self.attempt()
        trace.flush('fit_before')
        trace.finish()
        self.assertTrue(trace.terminal)
        self.assertEqual(trace.record['status'], 'complete')
        self.assertEqual(trace.record['flush_sequence'], 3)
        self.assertTrue(all(trace.record[key] == 0 for key in F3.COUNTERS))
        with self.assertRaises(ValueError):
            trace.observe('forward', lambda: None)()

    def test_boundaries_wrap_original_calls_without_changing_result(self):
        trace = self.attempt()
        calls = []
        original = lambda *args, **kwargs: calls.append((args, kwargs)) or 'same-result'
        result = F3._boundary(original, trace, 'subject_update')(object(), batch=8)
        self.assertEqual(result, 'same-result')
        self.assertEqual(len(calls), 1)
        self.assertEqual(trace.record['flush_counts'], {'attempt_created': 1,
            'subject_update_before': 1, 'subject_update_after': 1})

    def test_boundary_does_not_claim_return_when_function_throws(self):
        trace = self.attempt()
        def fail():
            raise KeyboardInterrupt('synthetic')
        with self.assertRaises(KeyboardInterrupt):
            F3._boundary(fail, trace, 'diagnostics')()
        self.assertEqual(trace.record['last_flush_reason'], 'diagnostics_before')
        self.assertNotIn('diagnostics_after', trace.record['flush_counts'])

    def test_adapter_binding_is_single_and_restorable(self):
        trace = self.attempt()
        original = lambda model, windows: 'encoded'
        adapter = types.SimpleNamespace(encode_windows=original)
        trace.bind_adapter(adapter)
        trace.bind_adapter(adapter)
        self.assertEqual(adapter.encode_windows(None, [1, 2]), 'encoded')
        self.assertEqual(trace.record['encode_returned'], 1)
        trace.restore()
        self.assertIs(adapter.encode_windows, original)

    def test_context_uses_f2_math_and_restores_all_functions(self):
        trace = self.attempt()
        v2, batching, model, chunks, folder = self.fake_dependencies(trace)
        update, diagnostic = v2.V1.subject_update, v2.V1.diagnostics
        save, save_torch = F3.S.save, F3.S.save_torch
        tensor_saves = []
        with patch.object(F3.S, 'save_torch', lambda *args, **kwargs: tensor_saves.append((args, kwargs))):
            with F3.activate(trace, v2, batching):
                self.assertEqual(v2.V1.subject_update(model, 'bag', micro_subjects=1),
                                 {'unchanged': ('bag',), 'micro_subjects': 1})
                self.assertEqual(v2.V1.diagnostics(model, 'raw', 'split', 'labels')['val_eval_ce'], 2.)
                payload = {'epoch': 1, 'optimizer_updates': 4, 'backward_calls': 27}
                F3.S.save_torch(folder / 'latest.pt', payload)
                self.assertEqual(payload, {'epoch': 1, 'optimizer_updates': 4, 'backward_calls': 27})
                F3.S.save(folder / 'metadata.json', {'condition': 'synthetic'}, immutable=True)
            self.assertIs(v2.V1.subject_update, update)
            self.assertIs(v2.V1.diagnostics, diagnostic)
        self.assertIs(F3.S.save, save)
        self.assertIs(F3.S.save_torch, save_torch)
        self.assertIsNone(v2._TRACE)
        self.assertFalse(v2._ACTIVE)
        self.assertFalse(F3._ACTIVE)
        metadata = F3.S.read(folder / 'metadata.json')
        self.assertEqual(metadata['execution_acceleration'], self.execution)
        self.assertEqual(metadata['versioned_finetune_implementation'],
                         self.execution['delegated_finetune_implementation'])
        self.assertEqual(chunks, [('train', 8), ('diagnostics', 8)])
        self.assertEqual(trace.record['latest_durable_checkpoint']['optimizer_updates'], 4)
        self.assertEqual(trace.record['flush_counts']['latest_checkpoint_committed'], 1)
        self.assertEqual(len(tensor_saves), 1)

    def test_only_qualified_diagnostics_changes_chunk_train_and_reload_stay8(self):
        self.execution['diagnostics'] = {'window_chunk': 32, 'qualification': {'synthetic': True}}
        trace = self.attempt()
        v2, batching, model, chunks, folder = self.fake_dependencies(trace)
        with F3.activate(trace, v2, batching):
            v2.V1.subject_update(model, micro_subjects=1)
            v2.V1.diagnostics(model)
            self.assertEqual(model.window_chunk, 8)  # reload uses this same model
            v2.V1.subject_update(model, micro_subjects=1)
        self.assertEqual(chunks, [('train', 8), ('diagnostics', 32), ('train', 8)])

    def test_failed_checkpoint_save_has_no_committed_checkpoint_boundary(self):
        trace = self.attempt()
        v2, batching, model, chunks, folder = self.fake_dependencies(trace)
        def fail(*args, **kwargs):
            raise OSError('checkpoint did not return')
        with patch.object(F3.S, 'save_torch', fail):
            with self.assertRaises(OSError):
                with F3.activate(trace, v2, batching):
                    F3.S.save_torch(folder / 'latest.pt', {})
        self.assertNotIn('latest_checkpoint_committed', trace.record['flush_counts'])
        self.assertIsNone(trace.record['latest_durable_checkpoint'])
        self.assertIsNone(v2._TRACE)
        self.assertFalse(F3._ACTIVE)

    def test_initial_best_and_other_fit_saves_are_not_latest_commits(self):
        trace = self.attempt()
        v2, batching, model, chunks, folder = self.fake_dependencies(trace)
        with patch.object(F3.S, 'save_torch', lambda *args, **kwargs: None):
            with F3.activate(trace, v2, batching):
                for path in (folder / 'initial.pt', folder / 'best.pt', self.out / 'other/latest.pt'):
                    F3.S.save_torch(path, {})
        self.assertEqual(trace.record['flush_sequence'], 1)

    def test_wrapper_setup_exception_restores_context(self):
        self.execution['diagnostics'] = {'window_chunk': 32, 'qualification': {'synthetic': True}}
        trace = self.attempt()
        v2, batching, model, chunks, folder = self.fake_dependencies(trace)
        original_update, original_diagnostic = v2.V1.subject_update, v2.V1.diagnostics
        save = F3.S.save
        batching.make_diagnostics = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('bad qualification'))
        with self.assertRaises(ValueError):
            with F3.activate(trace, v2, batching):
                self.fail('invalid qualification entered execution')
        self.assertIs(v2.V1.subject_update, original_update)
        self.assertIs(v2.V1.diagnostics, original_diagnostic)
        self.assertIs(F3.S.save, save)
        self.assertIsNone(v2._TRACE)
        self.assertFalse(F3._ACTIVE)

    def test_nested_activation_is_rejected_without_corrupting_owner(self):
        trace = self.attempt()
        v2, batching, model, chunks, folder = self.fake_dependencies(trace)
        with F3.activate(trace, v2, batching):
            with self.assertRaisesRegex(ValueError, 'isolated'):
                with F3.activate(trace, v2, batching):
                    pass
            self.assertIs(v2._TRACE, trace)
        self.assertIsNone(v2._TRACE)

    def test_preflight_rejects_before_tensor_import_or_attempt_creation(self):
        imports = []
        def rejecting(*args, **kwargs):
            raise ValueError('unregistered')
        def loader(name, package=None):
            imports.append(name)
            self.assertEqual(name, '.execution_acceleration_v1')
            return types.SimpleNamespace(load_execution_acceleration=rejecting)
        with patch.object(F3.importlib, 'import_module', loader):
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                F3.fit('synthetic', 11, 1, acceleration_registration={})
        self.assertEqual(imports, ['.execution_acceleration_v1'])
        self.assertEqual(self.saved, [])

    def test_training_micro_and_chunk_are_fixed_strict_integers(self):
        for micro, chunk in ((2, 8), (True, 8), (1, 16), (1, 8.0)):
            with self.subTest(micro=micro, chunk=chunk):
                with self.assertRaises(ValueError):
                    F3.fit('synthetic', 11, 1, acceleration_registration={},
                           micro_subjects=micro, window_chunk=chunk)
        self.assertEqual(self.saved, [])

    def test_terminal_snapshot_cannot_be_mutated_by_caller_execution_mapping(self):
        trace = self.attempt()
        self.execution['diagnostics']['window_chunk'] = 60
        self.assertEqual(trace.record['execution_acceleration']['diagnostics']['window_chunk'], 8)
        trace.finish()
        self.assertEqual(F3.S.read(trace.path)['implementation']['version'], 3)

    def synthetic_fit_dependencies(self, *, fail=False, reuse=False):
        holder = {}
        proxy = types.SimpleNamespace(observe=lambda *args: holder['v2']._TRACE.observe(*args))
        v2, batching, model, chunks, folder = self.fake_dependencies(proxy)
        holder['v2'] = v2
        calls = []
        v2.provenance = lambda: copy.deepcopy(self.execution['delegated_finetune_implementation'])
        v2._check_v2_lock = lambda condition: calls.append(('lock', condition))

        def delegated(condition, seed, fold, **kwargs):
            calls.append(('fit', condition, seed, fold, kwargs))
            if reuse:
                return {'original_f2_metadata_reused': True}
            v2.V1.subject_update(model, 'unchanged-bag', micro_subjects=kwargs['micro_subjects'])
            if fail:
                def failing_forward():
                    raise RuntimeError('synthetic full-fit failure')
                v2._TRACE.observe('forward', failing_forward)()
            v2.V1.diagnostics(model, 'unchanged-raw', 'unchanged-split', 'unchanged-labels')
            F3.S.save_torch(folder / 'latest.pt', {'epoch': 27, 'optimizer_updates': 108, 'backward_calls': 729})
            metadata = {'condition': condition, 'seed': seed, 'fold': fold, 'micro_subjects': 1, 'window_chunk': 8}
            F3.S.save(folder / 'metadata.json', metadata, immutable=True)
            return metadata

        v2._FIT = delegated
        def loader(name, package=None):
            return {'.execution_acceleration_v1': types.SimpleNamespace(
                load_execution_acceleration=lambda *args, **kwargs: copy.deepcopy(self.execution)),
                '.finetune_v2': v2, '.validation_batching_v1': batching}[name]
        return loader, v2, calls, folder

    def test_full_fit_delegates_resume_and_records_process_not_trajectory_counts(self):
        loader, v2, calls, folder = self.synthetic_fit_dependencies()
        with patch.object(F3.importlib, 'import_module', loader), \
                patch.object(F3.S, 'save_torch', lambda *args, **kwargs: None):
            metadata = F3.fit('synthetic', 11, 1, resume=True, acceleration_registration={})
        self.assertEqual(calls, [('lock', 'synthetic'), ('fit', 'synthetic', 11, 1,
            {'resume': True, 'micro_subjects': 1, 'window_chunk': 8})])
        traces = list((self.out / 'finetune_v2_attempts').glob('*/*/attempt.json'))
        self.assertEqual(len(traces), 1)
        record = F3.S.read(traces[0])
        self.assertEqual(record['status'], 'complete')
        self.assertEqual(record['optimizer_step_returned'], 1)
        self.assertEqual(record['latest_durable_checkpoint']['optimizer_updates'], 108)
        self.assertEqual(record['flush_sequence'], 8)
        self.assertEqual(record['flush_counts'], dict.fromkeys((
            'attempt_created', 'fit_before', 'subject_update_before', 'subject_update_after',
            'diagnostics_before', 'diagnostics_after', 'latest_checkpoint_committed', 'fit_returned'), 1))
        self.assertEqual(metadata['execution_acceleration'], self.execution)
        self.assertIsNone(v2._TRACE)

    def test_full_fit_exception_is_flushed_after_hooks_restore(self):
        loader, v2, calls, folder = self.synthetic_fit_dependencies(fail=True)
        with patch.object(F3.importlib, 'import_module', loader):
            with self.assertRaisesRegex(RuntimeError, 'full-fit failure'):
                F3.fit('synthetic', 11, 1, acceleration_registration={})
        record = F3.S.read(next((self.out / 'finetune_v2_attempts').glob('*/*/attempt.json')))
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['last_flush_reason'], 'exception')
        self.assertEqual(record['forward_started'], 2)
        self.assertEqual(record['forward_returned'], 1)
        self.assertEqual(record['optimizer_step_returned'], 1)
        self.assertTrue(record['graceful_terminal_flush'])
        self.assertFalse(record['unflushed_tail_unknown'])
        self.assertIsNone(v2._TRACE)
        self.assertFalse(F3._ACTIVE)
        self.assertFalse((folder / 'metadata.json').exists())

    def test_full_fit_completed_f2_reuse_does_not_rewrite_original_metadata(self):
        loader, v2, calls, folder = self.synthetic_fit_dependencies(reuse=True)
        with patch.object(F3.importlib, 'import_module', loader):
            metadata = F3.fit('synthetic', 11, 1, acceleration_registration={})
        self.assertEqual(metadata, {'original_f2_metadata_reused': True})
        record = F3.S.read(next((self.out / 'finetune_v2_attempts').glob('*/*/attempt.json')))
        self.assertEqual(record['status'], 'complete')
        self.assertTrue(all(record[key] == 0 for key in F3.COUNTERS))
        self.assertFalse((folder / 'metadata.json').exists())

    def test_shared_verifier_accepts_real_snapshot_schema_and_rejects_false_exactness(self):
        from tridep_ccml.eeg_all_candidates import execution_acceleration_v1 as verifier
        trace = self.attempt()
        fit = {'condition': 'synthetic', 'seed': 11, 'fold': 1}
        with patch.object(verifier, 'verify_execution_acceleration', return_value=self.execution):
            result = verifier.verify_observer_record(trace.record, F3.S.desc(trace.path), fit)
            self.assertTrue(result['unflushed_tail_unknown'])
            trace.observe('forward', lambda: None)()
            trace.observe('backward', lambda: None)()
            trace.observe('optimizer_step', lambda: None)()
            trace.finish()
            result = verifier.verify_observer_record(trace.record, F3.S.desc(trace.path), fit)
            self.assertTrue(result['counter_snapshot_exact_for_process'])
            wrong = copy.deepcopy(trace.record)
            wrong['unflushed_tail_unknown'] = True
            with self.assertRaises(ValueError):
                verifier.verify_observer_record(wrong, F3.S.desc(trace.path), fit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
