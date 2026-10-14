"""Independent stdlib contract tests; all process/GPU/model paths replaced."""
import ast
import __future__
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import sync_qualification_gate_v3 as G
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v1 as L


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load(REPORT / 'qualify_sync_reduction_v3.py', '_independent_diag3')
T = load(REPORT / 'qualify_training_sync_v3.py', '_independent_train3')


class GateFixture:
    """Synthetic JSON/source map. No real preflight dependency can execute."""
    def __init__(self):
        self.documents = {}
        self.live_calls = self.gpu_calls = 0
        self.live_error = None
        self.live = {'passed': True, 'gpu_children_exit_verified': True, 'held_dispatcher_running_suspended': True}
        self.observation = {'idle': True, 'contexts': []}
        self.registry_binding = self.desc(G.OUT / 'registration.json')
        self.queue_binding = self.desc(G.OUT / 'continuation_queue_v5/registration.json')
        self.recovery_binding = self.desc(G.OUT / 'continuation_queue_v5/recovery_registration.json')
        self.closure_binding = self.desc(G.OUT / 'process_loss_closures/lost_f3_fixture.json')
        self.source_binding = self.desc(G.WORK / 'tridep_ccml/eeg_all_candidates/recover_q5_from_pause_v1.py')
        self.owner_binding = self.desc(G.OUT / 'continuation_queue_v5/owner.json')
        self.control_binding = self.desc(G.CONTROL)
        self.sources = [self.desc(G.WORK / f'synthetic_sources/source_{i}.py') for i in range(80)]
        self.split = {'train': [f'train_{i}' for i in range(26)], 'val': [f'val_{i}' for i in range(3)],
                      'test': [f'test_{i}' for i in range(8)]}
        self.rows = [{'fit_id': f'CSBrain-lastblock-finetune/seed_{seed}/fold_{fold}',
                      'condition': 'CSBrain-lastblock-finetune', 'seed': seed, 'fold': fold, 'stage': '3',
                      **{k + '_ids': list(v) for k, v in self.split.items()}}
                     for seed in (11, 22, 33) for fold in range(1, 6)]
        self.registry = {'fits': self.rows, 'folds': {'fold_5': self.split}}
        required = [G.SOURCE, G.GUARDIAN, REPORT / 'qualify_sync_reduction_v3.py',
                    REPORT / 'qualify_training_sync_v3.py', G.SOURCE.with_name('sync_reduction_v1.py'),
                    G.SOURCE.with_name('sync_subject_update_v1.py'), G.SOURCE.with_name('sync_state_digest_v1.py')]
        self.experimental = [self.desc(p) for p in required]
        self.owner = {'source': self.desc(G.SOURCE.with_name('continuation_queue_v5.py')),
                      'execution_registration': self.queue_binding, 'execution_version': 5,
                      'pid': 101, 'child_pid': 102, 'child_job': 'CSBrain-lastblock-finetune-fit',
                      'recovery_registration': self.recovery_binding, 'recovery_execution_source': self.source_binding,
                      'recovery_closure': self.closure_binding}
        self.control = {'schema': 'recovered_q5_sync_control_registration_v3', 'version': 3,
                        'source': self.desc(G.GUARDIAN), 'registration': self.queue_binding,
                        'recovery_registration': self.recovery_binding, 'recovery_closure': self.closure_binding,
                        'experimental_sources': self.experimental, 'protected_sources': self.sources,
                        'queue_owner': self.owner_binding}
        fixed = G.OUT / 'runs/CSBrain-lastblock-finetune/seed_22/fold_5'
        self.history = [{'epoch': 1}, {'epoch': 2}]
        self.proof = {'schema': 'sync_qualification_safe_boundary_v3', 'passed': True,
                      'queue': 'continuation_queue_v5', 'registration': self.queue_binding,
                      'original_training_registration': self.registry_binding,
                      'source': self.desc(G.GUARDIAN), 'experimental_sources': self.experimental,
                      'recovery_registration': self.recovery_binding, 'recovery_closure': self.closure_binding,
                      'control_registration': self.control_binding, 'queue_owner': self.owner_binding,
                      'gpu_children_exit_verified': True, 'held_dispatcher_running_suspended': True,
                      'gpu_idle': True, 'zero_optimizer_probe_only': True, 'completed_CSBrain_fits_verified': 15,
                      'condition': 'CSBrain-lastblock-finetune', 'seed': 22, 'fold': 5,
                      'registered_fold_ids': self.split, 'checkpoint': self.desc(fixed / 'latest.pt'),
                      'train_history': self.desc(fixed / 'train_history.json')}
        self.proof_binding = self.desc(G.OUT / 'synthetic_boundary.json')
        self.attempt_binding = self.desc(G.OUT / 'synthetic_attempt.json')
        self.lineage = {'base_queue_registration': self.queue_binding, 'registration': self.recovery_binding,
                        'source': self.source_binding, 'value': {'sources': self.sources}}
        self.closure_result = {'passed': True, 'closure': self.closure_binding,
                               'recovery_registration': self.recovery_binding}
        for binding, value in [(self.proof_binding, self.proof), (self.registry_binding, self.registry),
                               (self.control_binding, self.control), (self.owner_binding, self.owner),
                               (self.closure_binding, {'original_attempt': self.attempt_binding}),
                               (self.attempt_binding, {'status': 'running'}), (self.proof['train_history'], self.history)]:
            self.documents[binding['path']] = value
        for row in self.rows:
            self.documents[self.desc(G.OUT / 'runs' / row['fit_id'] / 'status.json')['path']] = {'status': 'training_complete'}
    def desc(self, path):
        path = Path(path).resolve()
        name = path.relative_to(G.WORK).as_posix()
        digest = hashlib.sha256(name.encode()).hexdigest()
        if path == G.OUT / 'registration.json': digest = G.REGISTRY_SHA
        if path == G.GPU_OBSERVER: digest = G.GPU_OBSERVER_SHA
        return {'path': name, 'sha256': digest}
    def read(self, binding):
        if binding != self.desc(G.WORK / binding['path']): raise ValueError('Synthetic evidence hash changed')
        return self.documents[binding['path']]
    def guardian(self, binding):
        self.live_calls += 1
        if self.live_error is not None: raise self.live_error
        return self.live
    def gpu(self):
        self.gpu_calls += 1
        return self.observation
    def spec(self, name, path):
        fixture = self
        class Loader:
            def exec_module(self, module):
                if Path(path) == G.GUARDIAN: module.verify_live_boundary = fixture.guardian
                elif Path(path) == G.GPU_OBSERVER: module.gpu_observation = fixture.gpu
                else: raise AssertionError('Unmocked dynamic import')
        return types.SimpleNamespace(loader=Loader())
    def run(self):
        with patch.object(G, 'descriptor', self.desc), patch.object(G, 'read_bound', self.read), \
             patch.object(L, 'verify_recovery_registration', return_value=self.lineage), \
             patch.object(L, 'verify_lost_f3_attempt', return_value=self.closure_result), \
             patch.object(G.importlib.util, 'spec_from_file_location', self.spec), \
             patch.object(G.importlib.util, 'module_from_spec', return_value=types.SimpleNamespace()):
            return G.preflight(self.proof_binding, self.proof['source'])


class GateContractTests(unittest.TestCase):
    def test_valid_fake_complete_boundary_has_fifteen_statuses(self):
        f = GateFixture(); result = f.run()
        self.assertEqual(len(result[4]['completed_condition_statuses']), 15)
        self.assertEqual((result[2]['seed'], result[2]['fold']), (22, 5))
        self.assertEqual((f.live_calls, f.gpu_calls), (1, 1))
    def test_wrong_queue_schema_and_nonbool_guards_refuse(self):
        for key, value in [('queue', 'continuation_queue_v4'), ('schema', 'sync_qualification_safe_boundary_v2'),
                           ('passed', 1), ('gpu_children_exit_verified', 1), ('gpu_idle', 1),
                           ('held_dispatcher_running_suspended', False), ('zero_optimizer_probe_only', False),
                           ('completed_CSBrain_fits_verified', 15.0), ('completed_CSBrain_fits_verified', 14),
                           ('seed', 11), ('fold', True)]:
            with self.subTest(key=key, value=value):
                f = GateFixture(); f.proof[key] = value
                with self.assertRaises(ValueError): f.run()
                self.assertEqual(f.gpu_calls, 0)
    def test_missing_or_duplicate_formal_identity_refuses(self):
        for duplicate in (False, True):
            f = GateFixture()
            if duplicate: f.rows.append(copy.deepcopy(f.rows[0]))
            else: f.rows.pop()
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError): f.run()
    def test_role_contamination_refuses(self):
        f = GateFixture(); f.proof['registered_fold_ids'] = {**f.split, 'train': f.split['test']}
        with self.assertRaises(ValueError): f.run()
    def test_any_pending_fit_refuses_before_live_or_gpu(self):
        f = GateFixture()
        key = f.desc(G.OUT / 'runs' / f.rows[-1]['fit_id'] / 'status.json')['path']
        f.documents[key]['status'] = 'training'
        with self.assertRaises(ValueError): f.run()
        self.assertEqual((f.live_calls, f.gpu_calls), (0, 0))
    def test_experimental_missing_duplicate_or_changed_source_refuses(self):
        for mode in ('missing', 'duplicate', 'hash'):
            f = GateFixture()
            if mode == 'missing': f.experimental.pop(0)
            elif mode == 'duplicate': f.experimental.append(copy.deepcopy(f.experimental[0]))
            else: f.experimental[0]['sha256'] = 'changed'
            with self.subTest(mode=mode), self.assertRaises(ValueError): f.run()
    def test_recovery80_exact_source_contract(self):
        for mode in ('79', '81', 'protected_missing', 'protected_extra'):
            f = GateFixture()
            if mode == '79': f.sources.pop()
            elif mode == '81': f.sources.append(f.desc(G.WORK / 'extra.py'))
            elif mode == 'protected_missing': f.control['protected_sources'] = f.sources[:-1]
            else: f.control['protected_sources'] = f.sources + [f.desc(G.WORK / 'extra.py')]
            with self.subTest(mode=mode), self.assertRaises(ValueError): f.run()
    def test_control_source_schema_version_and_lineage_refuse(self):
        for key, value in [('schema', 'old'), ('version', True), ('version', 2), ('registration', {}),
                           ('source', {}), ('recovery_registration', {}), ('recovery_closure', {})]:
            f = GateFixture(); f.control[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): f.run()
    def test_actual_owner_core_and_recovery_fields_refuse(self):
        for key, value in [('source', {}), ('execution_version', True), ('execution_version', 4),
                           ('execution_registration', {}), ('child_pid', None), ('child_job', 'other-fit'),
                           ('recovery_registration', {}), ('recovery_execution_source', {}), ('recovery_closure', {})]:
            f = GateFixture(); f.owner[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): f.run()
    def test_missing_or_wrong_closure_verification_refuses(self):
        for value in (None, {'passed': False}, {'passed': True, 'closure': {}, 'recovery_registration': {}}):
            f = GateFixture(); f.closure_result = value
            with self.subTest(value=value), self.assertRaises(ValueError): f.run()
        f = GateFixture(); f.closure_result['passed'] = False
        with self.assertRaises(ValueError): f.run()
    def test_history_wrong_sequence_and_bool_refuse(self):
        for row in ({'epoch': True}, {'epoch': 2}):
            f = GateFixture(); f.history[0] = row
            with self.subTest(row=row), self.assertRaises(ValueError): f.run()
    def test_wrong_checkpoint_reference_refuses_without_tensor_read(self):
        f = GateFixture(); f.proof['checkpoint'] = f.desc(G.OUT / 'runs/wrong/latest.pt')
        with self.assertRaises(ValueError): f.run()
    def test_paused_or_failed_live_guard_never_queries_gpu(self):
        f = GateFixture(); f.live_error = RuntimeError('durable user pause')
        with self.assertRaisesRegex(RuntimeError, 'user pause'): f.run()
        self.assertEqual(f.gpu_calls, 0)
        for key in ('passed', 'gpu_children_exit_verified', 'held_dispatcher_running_suspended'):
            f = GateFixture(); f.live[key] = False
            with self.subTest(key=key), self.assertRaises(ValueError): f.run()
            self.assertEqual(f.gpu_calls, 0)
    def test_all_gpu_context_types_refuse_measurement(self):
        for kind in ('G', 'C', 'C+G'):
            f = GateFixture(); f.observation = {'idle': False, 'contexts': [{'pid': 999, 'type': kind}]}
            with self.subTest(kind=kind), self.assertRaises(ValueError): f.run()
        f = GateFixture(); f.observation = {'idle': True, 'contexts': [{'pid': 999, 'type': 'G'}]}
        with self.assertRaises(ValueError): f.run()


class DelegateTests(unittest.TestCase):
    def test_diagnostic_ast_only_changes_single_gate_import(self):
        original = ast.parse(D.DELEGATE.read_bytes()); transformed = D.compatibility_tree(D.DELEGATE.read_bytes())
        count = 0
        for node in transformed.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == 'sync_qualification_gate_v3': alias.name = 'sync_qualification_gate_v1'; count += 1
        self.assertEqual(count, 1)
        self.assertEqual(ast.dump(original), ast.dump(transformed))
        self.assertEqual(D.qualify.__code__.co_filename, str(D.DELEGATE))
        self.assertIs(D.D.G, G)
    def test_training_only_schema_guard_recompiled(self):
        original = types.ModuleType('old_training_source_for_code_compare'); original.__file__ = str(T.DELEGATE)
        exec(compile(T.DELEGATE.read_bytes(), str(T.DELEGATE), 'exec'), original.__dict__)
        for name, value in original.__dict__.items():
            if isinstance(value, types.FunctionType) and name not in ('validate_fixed_identity', 'helpers'):
                with self.subTest(name=name): self.assertEqual(value.__code__, T.D.__dict__[name].__code__)
        source = ast.parse(T.DELEGATE.read_bytes())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_fixed_identity')
        anchors = [n for n in ast.walk(node) if isinstance(n, ast.Constant) and n.value == 'sync_qualification_safe_boundary_v1']
        self.assertEqual(len(anchors), 1); anchors[0].value = 'sync_qualification_safe_boundary_v3'
        expected = dict(original.__dict__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(T.SOURCE), 'exec',
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), expected)
        self.assertEqual(expected['validate_fixed_identity'].__code__, T.D.validate_fixed_identity.__code__)
    def test_training_budget96_and_roles_unchanged(self):
        self.assertEqual(T.D.FIXED_COUNTS, {'forward': 96, 'encoder_forward': 96, 'encoder_windows': 768,
                                          'backward': 96, 'optimizer_step': 0, 'optimizer_step_stub': 12})
        self.assertEqual(len(T.D.matrix_plan()), 12)
        self.assertEqual({r['epoch_phase'] for r in T.D.matrix_plan()}, {1, 11})
        f = GateFixture(); fit = next(r for r in f.rows if r['seed'] == 22 and r['fold'] == 5)
        people, split = T.D.validate_fixed_identity(f.proof, f.registry, fit, f.history)
        self.assertEqual(people, f.split['train'][:8]); self.assertEqual(split, f.split)
        fit['train_ids'][0] = fit['test_ids'][0]
        with self.assertRaises(ValueError): T.D.validate_fixed_identity(f.proof, f.registry, fit, f.history)
    def test_exact_ce_signed_zero_and_nan(self):
        self.assertTrue(D.exact_ce({'ce': 1.25}, {'ce': 1.25}))
        self.assertFalse(D.exact_ce({'ce': 0.0}, {'ce': -0.0}))
        self.assertFalse(D.exact_ce({'ce': float('nan')}, {'ce': float('nan')}))
    def test_actual_v3_and_delegated_sources_truthful(self):
        for module, schema in ((D, 'diagnostic_sync_qualification_attempt_v1'), (T, 'training_sync_qualification_attempt_v1')):
            value = module.provenance({'schema': schema, 'status': 'failed'})
            self.assertTrue(value['schema'].endswith('_v3')); self.assertEqual(value['qualification_execution_version'], 3)
            self.assertEqual(value['source']['sha256'], hashlib.sha256(module.SOURCE.read_bytes()).hexdigest())
            self.assertIs(value['formal_activation_authorized'], False)
        self.assertEqual(D.OWNER, G.OWNER)
        self.assertEqual(T.D.OWNER, G.OWNER)
        self.assertEqual(T.GATE_SHA, hashlib.sha256(G.SOURCE.read_bytes()).hexdigest())
    def test_no_tensor_library_imported(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    start = time.perf_counter()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(GateContractTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(DelegateTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [G.SOURCE, D.SOURCE, T.SOURCE, Path(__file__)]
    value = {'schema': 'independent_sync_qualification_v3_stdlib_tests_v1', 'passed': result.wasSuccessful(),
             'sources': [{'path': p.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
             'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
             'failure_details': [text for _, text in result.failures], 'error_details': [text for _, text in result.errors],
             'seconds': time.perf_counter() - start, 'real_process_queries': 0, 'real_process_control': 0,
             'GPU_operations': 0, 'model_constructions': 0, 'checkpoint_reads': 0, 'raw_or_test_reads': 0,
             'mocked_preflight_is_not_actual_qualification': True, 'formal_activation_authorized': False,
             'original_artifacts_modified': False}
    path = REPORT / ('independent_sync_qualification_v3_stdlib_tests_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8') as stream: json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'report': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'passed': result.wasSuccessful()}))
    sys.exit(0 if result.wasSuccessful() else 1)
