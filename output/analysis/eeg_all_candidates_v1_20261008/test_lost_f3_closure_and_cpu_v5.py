"""Source/JSON and synthetic observer tests only; no tensors, models or GPU."""
import ast
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


A = module('lost_cpu5', ROOT / 'audit_completed_finetune_checkpoints_v5.py')
C = module('lost_compat_tests', ROOT / 'test_execution_acceleration_compatibility_v1.py')
L = A.L


class CpuCompatibilityTests(C.CompatibilityTests):
    def setUp(self):
        super().setUp()
        self.previous = C.A
        C.A = A
        self.addCleanup(setattr, C, 'A', self.previous)

    def test_v4_math_and_three_cpu_load_functions_unchanged(self):
        def funcs(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef)}
        old, new = funcs(ROOT / 'audit_completed_finetune_checkpoints_v4.py'), funcs(Path(A.__file__))
        for name in ('require', 'integer', 'seconds', 'finetune_matrix', 'load_wide', 'read_trace_bytes',
                     'read_trace_snapshot', 'recheck_trace_inventory', 'legacy_trace_boundary',
                     'optimizer_group_boundary', 'checkpoint_payload', 'cpu_audit'):
            self.assertEqual(old[name], new[name], name)

    def test_lost_f3_adapter_preserves_original_and_null_seconds(self):
        lost = self.buffered(status='running', updates=36, backwards=243)
        complete = self.buffered(updates=self.metadata['optimizer_updates'] - 32,
                                 backwards=self.metadata['backward_calls'] - 216)
        complete['artifact'] = {'path': 'synthetic/new/attempt.json', 'sha256': 'd' * 64}
        closure = {'passed': True, 'closure': {'path': 'synthetic/lost.json', 'sha256': 'e' * 64},
                   'unflushed_tail_unknown': True, 'unflushed_tail_counters': None, 'final_elapsed_seconds': None}
        lost['proven_process_loss'] = closure
        before = deepcopy(lost)
        metadata = {**self.metadata, 'execution_acceleration': self.execution(),
                    'versioned_finetune_implementation': A.X.f2_provenance(), 'window_chunk': 8, 'micro_subjects': 1}
        with patch.object(A.X, 'load_execution_acceleration', return_value=self.execution()), \
                patch.object(L, 'verify_lost_f3_attempt', return_value=closure):
            result = A.trace_boundary(self.fit, metadata, [lost, complete])
        self.assertEqual(lost, before)
        self.assertEqual(result['durable_optimizer_updates_counted_once'], self.metadata['optimizer_updates'])
        self.assertEqual(result['recorded_physical_call_subtotals']['optimizer_step_returned'], self.metadata['optimizer_updates'] + 4)
        self.assertEqual(result['recorded_physical_call_subtotals']['backward_returned'], self.metadata['backward_calls'] + 27)
        row = result['attempts'][0]
        self.assertEqual(row['status'], 'process_loss_observed')
        self.assertIsNone(row['seconds_this_attempt'])
        self.assertFalse(row['graceful_terminal_flush'])
        self.assertTrue(result['unflushed_buffered_tails_unknown'])
        self.assertIsNone(result['physical_optimizer_calls_exact'])

    def test_failed_non_graceful_null_seconds_is_not_old_legacy_terminal(self):
        item = self.buffered(status='running')
        item['record']['status'] = 'failed'
        with patch.object(A.X, 'load_execution_acceleration', return_value=self.execution()):
            self.assertTrue(A.X.verify_observer_record(item['record'], item['artifact'], self.fit)['unflushed_tail_unknown'])
            meta = {**self.metadata, 'execution_acceleration': self.execution(),
                    'versioned_finetune_implementation': A.X.f2_provenance(), 'window_chunk': 8, 'micro_subjects': 1}
            with self.assertRaisesRegex(ValueError, 'elapsed'):
                A.trace_boundary(self.fit, meta, [item])

    def test_unbound_or_changed_closure_is_not_accepted(self):
        lost = self.buffered(status='running')
        lost['proven_process_loss'] = {'passed': True}
        meta = {**self.metadata, 'execution_acceleration': self.execution(),
                'versioned_finetune_implementation': A.X.f2_provenance(), 'window_chunk': 8, 'micro_subjects': 1}
        with patch.object(A.X, 'load_execution_acceleration', return_value=self.execution()), \
                patch.object(L, 'verify_lost_f3_attempt', return_value=None), self.assertRaisesRegex(ValueError, 'closure changed'):
            A.trace_boundary(self.fit, meta, [lost])


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.cpu = json.loads((ROOT / 'recovery_cpu_preflight_csb33f4_1791596275747982600.json').read_text(encoding='utf8'))
        self.pause = json.loads((L.S.WORK / self.cpu['pause_evidence']['path']).read_text(encoding='utf8'))
        self.artifact = self.cpu['buffered_trace_pair']['original']
        self.record = json.loads((L.S.WORK / self.artifact['path']).read_text(encoding='utf8'))
        desc = lambda name: {'path': 'synthetic/' + name + '.json', 'sha256': 'a' * 64}
        self.entry = {'path': 'synthetic/entry.py', 'sha256': 'b' * 64}
        self.absence = {'schema': 'q4_process_absence_observation_v1', 'query_succeeded': True,
                        'all_expected_processes_absent': True, 'queried_pids': L.PIDS[:], 'absent_pids': L.PIDS[:],
                        'expected_identities': self.pause['identities'], 'exit_codes': None, 'actual_exit_time': None, 'reason': None}
        self.q4 = L.descriptor(L.S.OUT / 'continuation_queue_v4/registration.json')
        self.proof = {'schema': 'lost_f3_process_closure_v1', 'status': 'process_absence_observed', 'passed': True,
                      'condition': L.IDENTITY[0], 'seed': 33, 'fold': 4, 'closure_source': self.entry,
                      'recovery_registration': desc('reg'), 'original_attempt': self.artifact,
                      'attempt_snapshot': self.cpu['buffered_trace_pair']['snapshot'], 'pause_proof': self.cpu['pause_evidence'],
                      'process_absence_evidence': desc('absence'),
                      'checkpoint_cpu_resume_proof': L.descriptor(ROOT / 'recovery_cpu_preflight_csb33f4_1791596275747982600.json'),
                      'resume_checkpoint': self.cpu['checkpoint_validation']['original_checkpoint_bindings'][1],
                      'preserved_checkpoint': self.cpu['latest_pause_snapshot'], 'old_owner_snapshot': desc('owner'),
                      'old_queue_attempt_snapshot': desc('attempt'), 'durable': {'epoch': 8, 'optimizer_updates': 32, 'backward_calls': 216},
                      'observed_progress': {'epoch': 9, 'optimizer_updates': 36, 'backward_calls': 243},
                      'known_extra_lower_bounds': {'optimizer_updates': 4, 'backward_calls': 27},
                      'graceful_terminal_flush': False, 'unflushed_tail_counters': None, 'final_elapsed_seconds': None}
        self.manifest = {'pause_proof': self.proof['pause_proof'], 'process_absence_evidence': self.proof['process_absence_evidence']}
        self.values = {self.artifact['path']: self.record, self.proof['attempt_snapshot']['path']: self.record,
                       self.proof['pause_proof']['path']: self.pause, self.proof['process_absence_evidence']['path']: self.absence,
                       self.proof['checkpoint_cpu_resume_proof']['path']: self.cpu,
                       self.proof['old_owner_snapshot']['path']: self.pause['owner_snapshot'],
                       self.proof['old_queue_attempt_snapshot']['path']: {'status': 'running', 'child_pid': 18224,
                            'job': {'id': 'CSBrain-lastblock-finetune-fit'}, 'registration': self.q4}}
        self.fit = {'condition': L.IDENTITY[0], 'seed': 33, 'fold': 4}

    def run_fixture(self):
        with patch.object(L, 'read_json', side_effect=lambda b: deepcopy(self.values[b['path']])), \
                patch.object(L, 'verify_recovery_registration', return_value={'registration': self.proof['recovery_registration'],
                    'source': self.entry, 'value': self.manifest}):
            return L.validate_closure({'path': 'synthetic/closure.json', 'sha256': 'c' * 64}, self.proof, self.artifact, self.record, self.fit)

    def test_original_running_known_extra_and_unknown_tail_preserved(self):
        old = deepcopy(self.record); result = self.run_fixture()
        self.assertEqual(old, self.record); self.assertEqual(result['original_status'], 'running')
        self.assertEqual(result['known_extra_optimizer_returns_outside_checkpoint'], 4)
        self.assertEqual(result['known_extra_backward_returns_outside_checkpoint'], 27)
        self.assertIsNone(result['final_elapsed_seconds']); self.assertIsNone(result['unflushed_tail_counters'])

    def test_no_invented_exit_or_tail_or_time(self):
        for field, value in [('graceful_terminal_flush', True), ('final_elapsed_seconds', 0), ('unflushed_tail_counters', {})]:
            with self.subTest(field=field):
                old = self.proof[field]; self.proof[field] = value
                with self.assertRaises(ValueError): self.run_fixture()
                self.proof[field] = old
        self.absence['exit_codes'] = {str(pid): 0 for pid in L.PIDS}
        with self.assertRaises(ValueError): self.run_fixture()

    def test_wrong_identity_checkpoint_delta_and_pid_fail(self):
        for field, value in [('seed', 22), ('durable', {'epoch': 9, 'optimizer_updates': 36, 'backward_calls': 243}),
                             ('known_extra_lower_bounds', {'optimizer_updates': True, 'backward_calls': 27})]:
            with self.subTest(field=field):
                old = self.proof[field]; self.proof[field] = value
                with self.assertRaises(ValueError): self.run_fixture()
                self.proof[field] = old
        self.absence['absent_pids'] = [40840, 18224, 32880]
        with self.assertRaises(ValueError): self.run_fixture()

    def test_checkpoint_descriptors_never_opened_and_original_trace_not_reclassified(self):
        with patch.object(L, 'descriptor', side_effect=lambda p: self.q4 if Path(p).name == 'registration.json' else (_ for _ in ()).throw(AssertionError('unexpected file read'))):
            self.run_fixture()
        self.record['status'] = 'failed'
        with self.assertRaises(ValueError): self.run_fixture()

    def test_nonmatching_and_duplicate_closure_handling(self):
        with tempfile.TemporaryDirectory(dir=L.S.WORK / 'output', prefix='closure_stdlib_') as folder:
            root = Path(folder)
            with patch.object(L, 'ROOT', root):
                self.assertIsNone(L.verify_lost_f3_attempt(self.artifact, self.record, self.fit))
                for name in ('one', 'two'):
                    (root / ('lost_f3_' + name + '.json')).write_text(json.dumps(self.proof), encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                    L.verify_lost_f3_attempt(self.artifact, self.record, self.fit)

    def test_non_json_binding_rejected_before_read(self):
        with self.assertRaises(ValueError):
            L.read_json({'path': 'synthetic/latest.pt', 'sha256': 'a' * 64})


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (CpuCompatibilityTests, ClosureTests))
    stream = io.StringIO(); start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)
    report = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'seconds': time.perf_counter() - start, 'output': stream.getvalue(),
              'sources': [L.descriptor(Path(__file__)), L.descriptor(Path(A.__file__)), L.descriptor(Path(L.__file__))],
              'scope': 'stdlib source/immutable JSON and synthetic observer fixtures; no actual audit/checkpoint/model/GPU/process execution',
              'actual_checkpoint_loads': 0, 'model_constructions': 0, 'GPU_calls': 0, 'process_operations': 0}
    target = ROOT / ('lost_f3_cpu5_stdlib_tests_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf8') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({'passed': report['passed'], 'tests_run': result.testsRun, 'proof': L.descriptor(target),
                      'output': stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
