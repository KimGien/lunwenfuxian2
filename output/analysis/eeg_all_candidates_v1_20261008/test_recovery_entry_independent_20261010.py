"""Independent stdlib fault tests; source AST + isolated synthetic files only."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

WORK = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
ENTRY = WORK / 'tridep_ccml/eeg_all_candidates/recover_q5_from_pause_v1.py'
ENTRY_SHA = '93142aa08565a87f98de8d139d0830ed4984b02290b1dc61ebccd6dfa7ea019a'
TREE = ast.parse(ENTRY.read_text(encoding='utf-8'))


def function(name):
    return copy.deepcopy(next(n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name))


class SyntheticStorage:
    def __init__(self, root):
        self.WORK = root
        self.OUT = root / 'out'
        self.REPORT = root / 'report'
        self.REPORT.mkdir()
        self.checked_paths = []
        self.saved = []
    def now(self): return 'synthetic-time'
    def read(self, path): return json.loads(Path(path).read_text(encoding='utf-8'))
    def save(self, path, value, *, immutable=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x' if immutable else 'w', encoding='utf-8') as stream:
            json.dump(value, stream)
        self.saved.append((path, copy.deepcopy(value)))
    def desc(self, path): return {'path': str(path), 'sha256': 'synthetic-source-only'}
    def checked(self, value):
        self.checked_paths.append(value['path'])
        return Path(value['path'])


class RecoveryEntryTests(unittest.TestCase):
    def setUp(self):
        if hashlib.sha256(ENTRY.read_bytes()).hexdigest() != ENTRY_SHA:
            raise AssertionError('Reviewed source changed')
        tempbase = WORK / '.rq'
        tempbase.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='independent_', dir=tempbase)
        self.base = Path(self.temp.name).resolve()
        self.addCleanup(self.cleanup)
        self.s = SyntheticStorage(self.base)
        self.root = self.s.OUT / 'queue'
        self.root.mkdir(parents=True)
        self.intent = self.root / 'recovery_intent.json'
        self.manifest = self.root / 'recovery_registration.json'
        self.markers = [self.s.OUT / 'execution_acceleration_v1' / name / 'user_pause.json'
                        for name in ('sync_boundary', 'sync_boundary_v2')]
        for marker in self.markers: self.s.save(marker, {'paused': True})
        self.absence_calls = 0
        self.absence_error = False
    def cleanup(self):
        if not self.base.is_relative_to((WORK / '.rq').resolve()):
            raise AssertionError('Fixture cleanup escaped workspace test root')
        self.temp.cleanup()
    def inspect(self):
        self.absence_calls += 1
        if self.absence_error: raise ValueError('synthetic live identity')
        return {'all_expected_processes_absent': True}
    def preflight(self):
        # Compile the exact real preflight prefix, stopping before archive paths.
        node = function('prepare_recovery')
        stop = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'folder' for t in n.targets))
        node.body = node.body[:stop] + [ast.Return(value=ast.Constant(value='passed'))]
        env = {'ROOT': self.root, 'INTENT': self.intent, 'MANIFEST': self.manifest,
               'SOURCE': ENTRY, 'S': self.s, 'os': os, 'json': json, 'inspect_absence': self.inspect}
        module = ast.Module(body=[function('require'), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(ENTRY), 'exec'), env)
        return env['prepare_recovery']()
    def test_valid_preflight_retains_exclusive_intent(self):
        self.assertEqual(self.preflight(), 'passed')
        self.assertEqual(self.absence_calls, 1)
        self.assertEqual(self.s.read(self.intent)['schema'], 'q5_disk_recovery_exclusive_intent_v1')
        self.assertTrue(all(p.exists() for p in self.markers))
    def test_duplicate_intent_refuses_before_absence_or_manifest(self):
        self.preflight()
        before = self.intent.read_bytes()
        with self.assertRaises(FileExistsError): self.preflight()
        self.assertEqual(self.absence_calls, 1)
        self.assertEqual(before, self.intent.read_bytes())
        self.assertFalse(self.manifest.exists())
    def test_existing_q5_owner_refuses_before_intent(self):
        self.s.save(self.root / 'owner.json', {'pid': 123})
        with self.assertRaisesRegex(ValueError, 'owner already'): self.preflight()
        self.assertFalse(self.intent.exists())
        self.assertEqual(self.absence_calls, 0)
    def test_existing_manifest_refuses(self):
        self.s.save(self.manifest, {'preserve': True})
        before = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, 'manifest requires'): self.preflight()
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual(self.absence_calls, 0)
    def test_existing_closure_refuses(self):
        closure = self.s.OUT / 'process_loss_closures/lost_f3_fixture.json'
        self.s.save(closure, {'preserve': True})
        before = closure.read_bytes()
        with self.assertRaisesRegex(ValueError, 'process-loss closure'): self.preflight()
        self.assertEqual(closure.read_bytes(), before)
        self.assertEqual(self.absence_calls, 0)
    def test_changed_pause_marker_refuses(self):
        self.s.save(self.markers[1], {'paused': False})
        with self.assertRaisesRegex(ValueError, 'Strict paused'): self.preflight()
        self.assertEqual(self.absence_calls, 0)
        self.assertTrue(self.markers[0].exists())
    def test_missing_pause_marker_refuses(self):
        self.markers[1].unlink()
        with self.assertRaises(FileNotFoundError): self.preflight()
        self.assertEqual(self.absence_calls, 0)
        self.assertTrue(self.markers[0].exists())
    def test_live_identity_refuses_without_clearing_markers(self):
        self.absence_error = True
        with self.assertRaisesRegex(ValueError, 'live identity'): self.preflight()
        self.assertTrue(self.intent.exists())
        self.assertTrue(all(p.exists() for p in self.markers))
        self.assertFalse(self.manifest.exists())
    def test_independent_closure_and_second_absence_precede_unlink(self):
        node = function('prepare_recovery')
        def calls(suffix):
            return sorted(n.lineno for n in ast.walk(node) if isinstance(n, ast.Call)
                          and ast.unparse(n.func).endswith(suffix))
        owner_unlink = calls('old_owner.unlink')[0]
        self.assertLess(calls('L.verify_recovery_registration')[0], owner_unlink)
        self.assertLess(calls('L.verify_lost_f3_attempt')[0], owner_unlink)
        absence = calls('inspect_absence')
        self.assertEqual(len(absence), 2)
        self.assertLess(absence[-1], owner_unlink)
        self.assertGreater(calls('marker.unlink')[0], owner_unlink)
        self.assertNotIn('handoff_complete', ast.unparse(TREE))
        self.assertNotIn('torch', ast.unparse(TREE))
    def cpu_gate(self, value, all_complete=True):
        manifest = self.root / 'manifest.json'
        self.s.save(manifest, {'sources': [{'path': 'synthetic_reviewed_source.py', 'sha256': 'fixture'}]})
        self.s.save(self.root / 'registration.json', {'sources': []})
        calls = []
        driver = types.ModuleType('synthetic_recovery_driver')
        driver.S, driver.ROOT, driver.time = self.s, self.root, time
        def locked():
            if not all_complete: raise ValueError('not all360')
            return {'passed': True, 'registration': 'fixture360', 'formal_fits_verified': 360, 'statuses': []}
        driver.R = types.SimpleNamespace(assert_test_unlocked=locked)
        driver.run_child = lambda job, owner: calls.append(('test_child', job['id'])) or 'complete'
        driver.evaluate_tail = lambda owner: (_ for _ in ()).throw(AssertionError('Inherited CPU4 must not run'))
        q1 = types.SimpleNamespace(__file__=str(ENTRY.with_name('continuation_queue.py')),
                                   all_complete=lambda _: all_complete,
                                   P=types.SimpleNamespace(CONDITIONS=[types.SimpleNamespace(condition_id='fixture')]))
        q5 = types.SimpleNamespace(isolated_driver=lambda: driver, Q1=q1, SOURCE=ENTRY.with_name('continuation_queue_v5.py'))
        class Loader:
            def exec_module(self, module):
                module.audit = lambda: calls.append(('cpu5_audit',)) or copy.deepcopy(value)
        def spec(name, path):
            self.assertEqual(Path(path).name, 'audit_completed_finetune_checkpoints_v5.py')
            calls.append(('load_cpu5',))
            return types.SimpleNamespace(loader=Loader())
        fake_import = types.SimpleNamespace(util=types.SimpleNamespace(spec_from_file_location=spec,
                                                module_from_spec=lambda _: types.SimpleNamespace()))
        env = {'S': self.s, 'SOURCE': ENTRY, 'ROOT': self.root, 'Q5': q5,
               'Path': Path, 'time': time, 'importlib': fake_import}
        nodes = [function('require'), function('RecoveryStorage'), function('recovery_driver')]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(ENTRY), 'exec'), env)
        self.calls = calls
        result = env['recovery_driver'](self.s.desc(manifest), {'path': 'synthetic_closure.json', 'sha256': 'fixture'})
        return result.evaluate_tail({})
    def valid_cpu(self):
        return {'all_135_finetune_checkpoints_verified': True, 'errors': [],
                'deferred_completed_fits': [], 'checkpoint_loads': 405}
    def test_incomplete360_skips_cpu_and_test(self):
        self.assertIs(self.cpu_gate(self.valid_cpu(), all_complete=False), False)
        self.assertEqual(self.calls, [])
    def test_complete360_cpu405_routes_original_evaluation(self):
        self.assertIs(self.cpu_gate(self.valid_cpu()), True)
        self.assertEqual(self.calls, [('load_cpu5',), ('cpu5_audit',),
                          ('test_child', 'evaluate-mamba-finetune'), ('test_child', 'evaluate-all')])
        proof = self.s.read(self.root / 'pre_test_finetune_cpu_gate.json')
        self.assertEqual(proof['checkpoint_loads'], 405)
        self.assertEqual(Path(proof['actual_cpu_auditor']['path']).name, 'audit_completed_finetune_checkpoints_v5.py')
    def test_cpu404_refuses_before_test(self):
        value = self.valid_cpu(); value['checkpoint_loads'] = 404
        with self.assertRaises(ValueError): self.cpu_gate(value)
        self.assertEqual(self.calls, [('load_cpu5',), ('cpu5_audit',)])
    def test_cpu_error_refuses_before_test(self):
        value = self.valid_cpu(); value['errors'] = ['synthetic fault']
        with self.assertRaises(ValueError): self.cpu_gate(value)
        self.assertEqual(self.calls, [('load_cpu5',), ('cpu5_audit',)])
    def test_cpu_deferred_refuses_before_test(self):
        value = self.valid_cpu(); value['deferred_completed_fits'] = [{'identity': 'fixture'}]
        with self.assertRaises(ValueError): self.cpu_gate(value)
        self.assertEqual(self.calls, [('load_cpu5',), ('cpu5_audit',)])
    def test_cpu_partial_refuses_before_test(self):
        value = self.valid_cpu(); value['all_135_finetune_checkpoints_verified'] = False
        with self.assertRaises(ValueError): self.cpu_gate(value)
        self.assertEqual(self.calls, [('load_cpu5',), ('cpu5_audit',)])


if __name__ == '__main__':
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryEntryTests))
    proof = {'schema': 'independent_recovery_entry_stdlib_tests_v1', 'passed': result.wasSuccessful(),
             'source': {'path': ENTRY.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(ENTRY.read_bytes()).hexdigest()},
             'test_source': {'path': Path(__file__).relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
             'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
             'seconds': time.perf_counter() - started, 'fixture_scope': 'source AST and isolated synthetic JSON only',
             'real_process_queries': 0, 'real_process_control': 0, 'GPU_operations': 0,
             'model_constructions': 0, 'checkpoint_reads': 0, 'raw_or_test_reads': 0,
             'original_artifacts_modified': False}
    path = REPORT / ('independent_recovery_entry_stdlib_tests_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8') as stream: json.dump(proof, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'report': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'passed': result.wasSuccessful()}))
    sys.exit(0 if result.wasSuccessful() else 1)
