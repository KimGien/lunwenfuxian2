"""Synthetic JSON/source tests only; no audit of actual checkpoint/probability files."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

REPORT = Path(__file__).resolve().parent
SOURCE = REPORT / 'audit_full_plan_acceptance_v5.py'
spec = importlib.util.spec_from_file_location('_wide_v5_tests', SOURCE)
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix='wide_v5_fixture_', dir=REPORT)).resolve()
        self.out = self.work / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
        self.queue = self.out / 'continuation_queue_v5'
        self.report = self.work / 'output/analysis/eeg_all_candidates_v1_20261008'
        self.mod = self.work / 'tridep_ccml/eeg_all_candidates'
        self.patches = []
        self.binary_reads = 0
        self.registry = {'formal_fits': 360, 'fits': [
            {'fit_id': 'condition/seed_11/fold_' + str(i), 'stage': '3' if i < 135 else '2'} for i in range(360)]}
        self.put(self.out / 'registration.json', self.registry)
        self.registry_binding = self.desc(self.out / 'registration.json')
        old_jobs = V.S.read(V.OUT / 'continuation_queue_v4/registration.json')['jobs']
        rows = []
        for i in range(59):
            p = self.mod / ('old_' + str(i) + '.py')
            self.put(p, '# source\n'); rows.append(self.desc(p))
        q3 = {'registry': self.registry_binding, 'sources': rows[:47]}
        q3path = self.out / 'continuation_queue_v3/registration.json'
        self.put(q3path, q3)
        q4 = {'registry': self.registry_binding, 'sources': rows, 'jobs': old_jobs,
              'previous_execution_registration': self.desc(q3path)}
        q4path = self.out / 'continuation_queue_v4/registration.json'
        self.put(q4path, q4)
        self.q5 = {**q4, 'version': 5, 'previous_execution_registration': self.desc(q4path),
                   'formal_fits_in_reused_matrix': 195, 'one_gpu_sequential': True,
                   'training_protocol_matrix_or_hyperparameters_changed': False,
                   'training_micro_subjects': 1, 'training_window_chunk': 8, 'test_gate_unchanged': True,
                   'sources': rows + [self.desc(q4path)], 'jobs': copy.deepcopy(old_jobs)}
        for row in self.q5['jobs']:
            row['module'] = {'acceleration_prepare_v1': 'acceleration_prepare_v2', 'linux_entry_v3': 'linux_entry_v4'}.get(row['module'], row['module'])
        for name in ('continuation_queue_v5.py', 'acceleration_prepare_v2.py', 'linux_entry_v4.py',
                     'finetune_v3.py', 'evaluate_v3.py', 'execution_acceleration_v1.py'):
            p = self.mod / name; self.put(p, '# synthetic source\n'); self.q5['sources'].append(self.desc(p))
        cp = self.report / 'audit_completed_finetune_checkpoints_v4.py'
        self.put(cp, '# synthetic CPU source\n'); self.q5['sources'].append(self.desc(cp))
        self.put(self.queue / 'registration.json', self.q5)
        self.terminal = {'source': self.desc(self.mod / 'continuation_queue_v5.py'),
                         'execution_registration': self.desc(self.queue / 'registration.json'), 'execution_version': 5,
                         'status': 'all_training_complete', 'test_evaluation_completed': True,
                         'test_evaluation_frozen': False, 'test_evaluation_status': 'complete',
                         'outcomes': {row['id']: 'complete' for row in old_jobs}}
        self.put(self.queue / 'status.json', self.terminal)
        self.cpu = {'schema_version': 4, 'registration': self.registry_binding, 'registered_total_fits': 360,
                    'registered_finetune_fits': 135, 'training_complete_finetune_observed': 135,
                    'verified_completed_finetune_fits': 135, 'logical_completed_fits_counted': 135,
                    'checkpoint_loads': 405, 'status': 'all_finetune_verified',
                    'all_135_finetune_checkpoints_verified': True, 'partial_snapshot': False,
                    'pending_fits': [], 'deferred_completed_fits': [], 'errors': [],
                    'auditor_source': self.desc(cp), 'completed_rows': [],
                    'observed_auditor_operations_including_failed_checks': {
                        'cpu_checkpoint_loads_started': 405, 'cpu_checkpoint_loads_returned': 405,
                        'model_construction_attempts_blocked': 0, 'gpu_initialization_attempts_blocked': 0}}
        for fit in self.registry['fits'][:135]:
            folder = self.out / 'runs' / fit['fit_id']
            row = {'fit_id': fit['fit_id'], 'passed': True}
            for key, name in (('status_artifact', 'status.json'), ('metadata', 'metadata.json'), ('completion', 'complete.json')):
                self.put(folder / name, {'fit_id': fit['fit_id'], 'artifact': name}); row[key] = self.desc(folder / name)
            row['cpu_checkpoint_audit'] = {'cpu_tensors_verified': True, 'checkpoint_loads': 3,
                'only_map_location_cpu': True, 'gpu_operations': 0, 'model_constructions': 0,
                'checkpoints': [{'path': (folder / name).relative_to(self.work).as_posix(), 'sha256': 'a' * 64}
                                for name in ('initial.pt', 'latest.pt', 'best.pt')]}
            self.cpu['completed_rows'].append(row)
        self.write_cpu()
        fake_s = types.SimpleNamespace(desc=self.desc, checked=self.checked, read=self.read)
        fake_r = types.SimpleNamespace(registration_descriptor=lambda: self.registry_binding,
                                       load_registration=lambda **kwargs: self.registry)
        for name, value in [('WORK', self.work), ('OUT', self.out), ('QUEUE', self.queue), ('REPORT', self.report),
                            ('S', fake_s), ('R', fake_r), ('Q3_SHA', self.desc(q3path)['sha256']),
                            ('Q4_SHA', self.desc(q4path)['sha256'])]:
            p = patch.object(V, name, value); p.start(); self.patches.append(p)

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.assertEqual(self.binary_reads, 0)
        self.assertTrue(self.work.is_relative_to(REPORT.resolve()))
        shutil.rmtree(self.work)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding='utf-8')

    def desc(self, path):
        path = Path(path).resolve()
        if path.suffix not in ('.json', '.py'):
            self.binary_reads += 1; raise AssertionError('Binary read forbidden in lifecycle verifier')
        return {'path': path.relative_to(self.work).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def checked(self, binding):
        path = self.work / binding['path']
        V.require(self.desc(path) == binding, 'fixture bytes changed')
        return path.resolve()

    def read(self, path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def write_cpu(self, *, directory=None):
        directory = directory or self.queue
        path = directory / 'pre_test_finetune_cpu_audit_fixture.json'
        self.put(path, self.cpu)
        self.put(self.queue / 'pre_test_finetune_cpu_gate.json', {'passed': True, 'checkpoint_loads': 405,
                                                              'test_forward_calls_this_check': 0, 'audit': self.desc(path)})

    def test_valid_lifecycle_no_binary_or_model_operation(self):
        result = V.queue_terminal(self.registry)
        self.assertTrue(result['passed']); self.assertEqual(result['preserved_Q4_sources'], 59)
        self.assertEqual(result['pre_test_finetune_cpu_verification']['actual_cpu_checkpoint_loads'], 405)

    def test_job_identity_math_args_dependencies_must_not_change(self):
        for key in ('args', 'dependencies', 'fits'):
            q = copy.deepcopy(self.q5); q['jobs'][1][key] = [] if key != 'fits' else 16
            self.put(self.queue / 'registration.json', q)
            with self.assertRaises(ValueError): V.queue_lineage(self.registry)

    def test_source_inheritance_and_unique_paths_required(self):
        for sources in (self.q5['sources'][1:], self.q5['sources'] + [self.q5['sources'][0]]):
            self.put(self.queue / 'registration.json', {**self.q5, 'sources': sources})
            with self.assertRaises(ValueError): V.queue_lineage(self.registry)

    def test_changed_sealed_source_rejected(self):
        self.put(self.mod / 'old_0.py', '# changed')
        with self.assertRaises(ValueError): V.queue_lineage(self.registry)

    def test_q4_or_unbound_terminal_not_accepted_as_q5(self):
        for changes in ({'execution_version': 4}, {'execution_version': True}, {'source': {}}, {'execution_registration': {}}):
            self.put(self.queue / 'status.json', {**self.terminal, **changes})
            with self.assertRaises(ValueError): V.queue_terminal(self.registry)

    def test_missing_or_failed_job_blocks(self):
        for outcomes in ({}, {**self.terminal['outcomes'], 'mamba-check': 'failed'}):
            self.put(self.queue / 'status.json', {**self.terminal, 'outcomes': outcomes})
            with self.assertRaises(ValueError): V.queue_terminal(self.registry)

    def test_evaluation_pending_blocks(self):
        for changes in ({'test_evaluation_completed': False}, {'test_evaluation_frozen': True}, {'test_evaluation_status': 'pending'}):
            self.put(self.queue / 'status.json', {**self.terminal, **changes})
            with self.assertRaises(ValueError): V.queue_terminal(self.registry)

    def test_each_old_or_current_owner_blocks(self):
        for name in ('continuation_queue_v3', 'continuation_queue_v4', 'continuation_queue_v5'):
            p = self.out / name / 'owner.json'; self.put(p, {'pid': 123})
            with self.assertRaises(ValueError): V.queue_terminal(self.registry)
            p.unlink()

    def test_old_q4_cpu_report_does_not_substitute(self):
        self.write_cpu(directory=self.out / 'continuation_queue_v4')
        with self.assertRaises(ValueError): V.queue_terminal(self.registry)

    def test_partial_duplicate_or_bool_cpu_counts_block(self):
        original = copy.deepcopy(self.cpu)
        for change in ({'checkpoint_loads': True}, {'deferred_completed_fits': [{'fit_id': 'pending'}]},
                       {'completed_rows': [original['completed_rows'][0]] * 135}):
            self.cpu = {**original, **change}; self.write_cpu()
            with self.assertRaises(ValueError): V.queue_terminal(self.registry)

    def test_stale_current_metadata_rejects_cpu_proof(self):
        path = self.work / self.cpu['completed_rows'][0]['metadata']['path']; self.put(path, {'changed': True})
        with self.assertRaises(ValueError): V.queue_terminal(self.registry)

    def test_gate_binds_no_additional_checkpoint_read(self):
        self.assertFalse(list(self.out.rglob('*.pt')))
        self.assertTrue(V.queue_terminal(self.registry)['passed'])
        self.assertEqual(self.binary_reads, 0)

    def test_evaluation_hook_refuses_before_original_verifier(self):
        self.put(self.queue / 'status.json', {**self.terminal, 'test_evaluation_completed': False})
        with patch.object(V, '_delegate_evaluation_audit') as delegate:
            with self.assertRaises(ValueError): V.evaluation_audit(self.registry)
            delegate.assert_not_called()

    def test_cost_proof_must_match_same_q5_terminal(self):
        life = V.queue_terminal(self.registry)
        expected = {'registration': life['q5_registration'], 'terminal_status': life['terminal_status'],
                    'source': life['q5_source'], 'execution_version': 5, 'previous_q4_registration': life['q4_registration']}
        p = self.report / 'cost.json'
        self.put(p, {'completion_gate': {'q5_execution_proof': expected}})
        with patch.object(V, '_delegate_final_cost_audit', return_value={'passed': True}):
            self.assertTrue(V.final_cost_audit(p, None, [])['passed'])
            self.put(p, {'completion_gate': {'q5_execution_proof': {**expected, 'execution_version': 4}}})
            with self.assertRaises(ValueError): V.final_cost_audit(p, None, [])


class DelegationTests(unittest.TestCase):
    def test_original_math_helpers_reused_without_recompilation_edits(self):
        old = V.load_pinned(V.DELEGATE, V.DELEGATE_SHA, '_wide_v4_reference')
        for name in ('legacy_check_fit', 'check_fit', 'deep_checkpoint', 'state_digest', 'policy_curve',
                     'cost_recovery_audit', 'verify_registered_identity', 'verify_exported_tables', 'semantic_review_audit'):
            self.assertEqual(getattr(V.D, name).__code__.co_code, getattr(old, name).__code__.co_code)
            self.assertEqual(getattr(V.D, name).__code__.co_filename, str(V.DELEGATE))
        self.assertEqual(V.D.SOURCE, V.SOURCE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
