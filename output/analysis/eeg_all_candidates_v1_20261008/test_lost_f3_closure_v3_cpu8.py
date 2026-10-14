"""Pure stdlib/fake evidence tests. Never invoke real X/L, CPU audit or processes."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPORT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


A = load('test_cpu8_driver', REPORT / 'audit_completed_finetune_checkpoints_v8.py')
L = A.L
OLD = load('test_cpu5_compat_from_cpu8', REPORT / 'test_lost_f3_closure_and_cpu_v5.py')


class Cpu8Tests(OLD.CpuCompatibilityTests):
    def setUp(self):
        self.old_a, self.old_l = OLD.A, OLD.L
        OLD.A, OLD.L = A.B, L
        self.addCleanup(setattr, OLD, 'A', self.old_a)
        self.addCleanup(setattr, OLD, 'L', self.old_l)
        self.previous_inherited_a, self.previous_inherited_source = OLD.C.T.A, OLD.C.T.SOURCE
        OLD.C.T.A, OLD.C.T.SOURCE = A.B, Path(A.B.__file__)
        self.addCleanup(setattr, OLD.C.T, 'A', self.previous_inherited_a)
        self.addCleanup(setattr, OLD.C.T, 'SOURCE', self.previous_inherited_source)
        super().setUp()

    def test_private_delegate_not_patch_global_or_math(self):
        old = load('unmodified_cpu5_compare', A.DELEGATE)
        self.assertIs(A.B.L, L)
        self.assertIsNot(old.L, L)
        for name in ('audit', 'cpu_audit', 'checkpoint_payload', 'optimizer_group_boundary',
                     'trace_boundary', 'legacy_trace_boundary', 'completed_fit_audit', 'trace_inventory'):
            self.assertEqual(getattr(A.B, name).__code__.co_code, getattr(old, name).__code__.co_code, name)
            self.assertEqual(getattr(A.B, name).__code__.co_consts, getattr(old, name).__code__.co_consts, name)

    def test_wrapper_reports_actual6_without_auditing(self):
        binding = L.descriptor(A.SOURCE)
        with patch.object(A.B, 'audit', return_value={'schema_version': 5, 'auditor_source': binding, 'sentinel': True}):
            result = A.audit()
        self.assertEqual(result['schema_version'], 8)
        self.assertEqual(result['auditor_source'], binding)
        self.assertTrue(result['sentinel'])
        self.assertTrue(result['historical_weight_bytes_may_be_hashed_for_source_binding'])


class Loss3Tests(unittest.TestCase):
    def setUp(self):
        self.values = {}
        self.bindings = {}
        def d(name, value=None):
            binding = {'path': 'output/synthetic_loss2/' + name, 'sha256': hashlib.sha256(name.encode()).hexdigest()}
            self.bindings[binding['path']] = binding
            if value is not None: self.values[binding['path']] = value
            return binding
        self.d = d
        self.entry = d('entry.py')
        self.prior = {'passed': True, 'registration': d('previous.json'), 'base_queue_registration': d('q5.json'),
                      'source': d('old_entry.py'), 'value': {'sources': []}}
        self.source = d('collector.py')
        self.cpu_source = d('cpu_proof.py')
        self.q5source = L.descriptor(L.SOURCE.with_name('continuation_queue_v5.py'))
        self.bindings[self.q5source['path']] = self.q5source
        identities = [{'pid': p, 'parent_pid': pp, 'creation_100ns': str(100000 + p),
                       'image': 'synthetic/python.exe', 'command_line': 'synthetic experiment'}
                      for p, pp in ((10, 9), (11, 10), (12, 11), (13, 12))]
        self.absence = {'schema': 'q5_process_absence_observation_v3', 'query_succeeded': True,
                        'all_expected_processes_absent': True, 'expected_identities': identities,
                        'queried_pids': [10,11,12,13], 'absent_pids': [10,11,12,13],
                        'workspace_training_workers': [], 'observed_rows': [],
                        'exit_codes': None, 'actual_exit_time': None, 'reason': None, 'observed_at': 'synthetic'}
        self.owner = {'pid': 11, 'child_pid': 12, 'child_job': L.IDENTITY[0] + '-fit',
                      'recovery_registration': self.prior['registration'], 'source': self.q5source,
                      'execution_registration': self.prior['base_queue_registration'], 'execution_version': 5,
                      'recovery_execution_source': self.prior['source']}
        self.queue = {**self.owner, 'status': 'running', 'job': {'id': self.owner['child_job']}}
        self.record = {'condition': L.IDENTITY[0], 'seed': 33, 'fold': 1, 'kind': 'formal', 'status': 'running',
                       'implementation': {'version': 3}, 'graceful_terminal_flush': False,
                       'optimizer_step_returned': 8, 'backward_returned': 52,
                       'latest_durable_checkpoint': {'epoch': 1, 'optimizer_updates': 4, 'backward_calls': 26,
                                                     'path': 'output/synthetic_loss2/latest.pt', 'checkpoint_save_returned': True}}
        self.progress = {'epoch': 2, 'optimizer_updates_completed': 8, 'backward_calls_completed': 52, 'last_durable_epoch': 1}
        self.history = [{'epoch': 1, 'optimizer_updates': 4, 'backward_calls': 26}]
        def pair(name, value):
            original = d(name + '.json', value); snapshot = d(name + '_snapshot.json', value)
            snapshot['sha256'] = original['sha256']
            return original, snapshot
        attempt, attempt_snapshot = pair('attempt', self.record)
        owner, owner_snapshot = pair('owner', self.owner)
        queue, queue_snapshot = pair('queue', self.queue)
        history, history_snapshot = pair('history', self.history)
        checkpoint = d('latest.pt'); preserved = d('preserved.pt'); preserved['sha256'] = checkpoint['sha256']
        absence = d('absence.json', self.absence)
        self.cpu = {'passed': True, 'source': self.cpu_source, 'condition': L.IDENTITY[0], 'seed': 33, 'fold': 1,
                    'checkpoint_loads': 2, 'model_constructions': 0, 'GPU_operations': 0, 'raw_arrays_loaded': 0,
                    'test_probability_reads': 0, 'process_absence_evidence': absence,
                    'buffered_trace_pair': {'original': attempt, 'snapshot': attempt_snapshot},
                    'preserved_checkpoint': preserved, 'resume_checkpoint': checkpoint, 'durable_history_matches_current_and_snapshot': True,
                    'checkpoint_validation': {'epoch': 1, 'optimizer_updates': 4, 'backward_calls': 26,
                        'resume_full_state_verified': True, 'checkpoint_load_device': 'cpu',
                        'optimizer_group_order_and_moment_shapes_verified': True, 'best_selected_parameter_finiteness_verified': True,
                        'cpu_rng_local_restore_and_cuda_saved_byte_shapes_verified': True,
                        'raw_role_descriptors_verified_without_loading_raw': True, 'original_checkpoint_bindings': [checkpoint]}}
        self.loss = {'schema': 'q5_f3_process_loss_evidence_v3', 'passed': True, 'source': self.source,
                     'condition': L.IDENTITY[0], 'seed': 33, 'fold': 1,
                     'previous_recovery_registration': self.prior['registration'], 'base_queue_registration': self.prior['base_queue_registration'],
                     'original_attempt': attempt, 'attempt_snapshot': attempt_snapshot,
                     'old_owner_original': owner, 'old_owner_snapshot': owner_snapshot,
                     'old_queue_attempt_original': queue, 'old_queue_attempt_snapshot': queue_snapshot,
                     'resume_checkpoint': checkpoint, 'preserved_checkpoint': preserved,
                     'train_history': history, 'history_snapshot': history_snapshot, 'progress_snapshot': d('progress.json', self.progress),
                     'durable': self.history[0], 'observed_progress': {'epoch': 2, 'optimizer_updates': 8, 'backward_calls': 52},
                     'known_extra_lower_bounds': {'optimizer_updates': 4, 'backward_calls': 26},
                     'checkpoint_cpu_resume_proof': d('cpu.json', self.cpu), 'process_absence_evidence': absence,
                     'expected_native_identities': identities, 'graceful_terminal_flush': False,
                     **{k: None for k in L.UNKNOWN_FIELDS}}
        self.loss_binding = d('loss.json', self.loss)
        self.snapshot = {'schema': 'q5_loss_byte_snapshot_v3', 'identity': list(L.IDENTITY), 'absence': absence,
                         'source': self.cpu_source, 'retained': {
            name: {'original': self.loss[original], 'snapshot': self.loss[copied]}
            for name, original, copied in (('attempt', 'original_attempt', 'attempt_snapshot'),
                ('owner', 'old_owner_original', 'old_owner_snapshot'), ('queue_attempt', 'old_queue_attempt_original', 'old_queue_attempt_snapshot'),
                ('train_history.json', 'train_history', 'history_snapshot'), ('latest.pt', 'resume_checkpoint', 'preserved_checkpoint'))}}
        self.snapshot['retained']['progress.json'] = {'snapshot': self.loss['progress_snapshot']}
        self.cpu['snapshot_manifest'] = d('snapshot.json', self.snapshot)
        for ident, role in zip(identities, ('dispatcher_launcher','dispatcher','launcher','gpu_child')):
            ident['role'] = role
        self.pause = {'status': 'paused_by_explicit_user', 'automatic_resume_authorized': False,
            'runtime_memory_preserved': True, 'owner_snapshot': self.owner, 'identities': identities,
            'actions': [{'pid': ident['pid'], 'native_status': 0} for ident in identities],
            'verification': [{'pid': ident['pid'], 'role': ident['role'], 'creation_100ns': ident['creation_100ns'],
                'threads': [{'id': 10 + ident['pid'], 'state': 'Wait', 'wait_reason': 'Suspended'}]}
                for ident in identities]}
        self.capture = {'owner_snapshot': self.owner, 'identities': identities}
        self.paused_progress = {'progress_JSON_reported': self.progress,
            'running_observer_records': [{**attempt, 'value': self.record}]}
        for key, value, digest in [('pause_native_proof', self.pause, L.PAUSE_NATIVE_SHA),
                                 ('pause_capture', self.capture, L.PAUSE_CAPTURE_SHA),
                                 ('pause_progress_and_observer', self.paused_progress, L.PAUSE_PROGRESS_SHA)]:
            binding = d(key+'.json', value); binding['sha256'] = digest; self.loss[key] = binding
        self.cpu['pause_native_proof'] = self.loss['pause_native_proof']
        self.observation = {'unflushed_tail_unknown': True, 'unflushed_tail_counters': None}

    def fake_descriptor(self, path):
        path = Path(path).resolve().relative_to(L.S.WORK).as_posix()
        self.assertNotEqual(Path(path).suffix, '.pt', 'No checkpoint byte hash in fake closure verifier')
        return self.bindings[path]

    def run_fixture(self):
        with patch.object(L, 'read_json', side_effect=lambda b: deepcopy(self.values[b['path']])), \
                patch.object(L, 'descriptor', side_effect=self.fake_descriptor), \
                patch.object(L, 'LOSS_BINDING', self.loss_binding), \
                patch.object(L, 'previous_registration', return_value=self.prior), \
                patch.object(L.X, 'verify_observer_record', return_value=self.observation):
            return L.verify_recovery_inputs(self.loss_binding)

    def test_loss_without_pause_keeps_unknowns_and_lower_bounds(self):
        before = deepcopy(self.values); result = self.run_fixture()
        self.assertTrue(result['passed']); self.assertNotIn('pause_proof', result)
        self.assertEqual(result['known_extra_lower_bounds'], {'optimizer_updates': 4, 'backward_calls': 26})
        self.assertEqual(before, self.values)
        for field in L.UNKNOWN_FIELDS: self.assertIsNone(result[field])

    def test_no_invented_exit_time_zero_tail_or_elapsed(self):
        for field in L.UNKNOWN_FIELDS:
            with self.subTest(field=field):
                self.loss[field] = 0
                with self.assertRaises(ValueError): self.run_fixture()
                self.loss[field] = None

    def test_absence_rejects_partial_or_unknown_workers(self):
        for key, bad in [('query_succeeded', False), ('all_expected_processes_absent', False),
                         ('absent_pids', [10,11,12]), ('observed_rows', [{}]), ('workspace_training_workers', [{}]),
                         ('exit_codes', {'13': 0}), ('actual_exit_time', 'fake'), ('reason', 'quota')]:
            with self.subTest(key=key):
                old = self.absence[key]; self.absence[key] = bad
                with self.assertRaises(ValueError): self.run_fixture()
                self.absence[key] = old

    def test_original_trace_cannot_change_or_become_complete(self):
        self.record['status'] = 'complete'
        with self.assertRaises(ValueError): self.run_fixture()

    def test_bool_counters_and_guessed_delta_rejected(self):
        for key, bad in [('optimizer_updates', True), ('backward_calls', 27)]:
            old = self.loss['known_extra_lower_bounds'][key]; self.loss['known_extra_lower_bounds'][key] = bad
            with self.assertRaises(ValueError): self.run_fixture()
            self.loss['known_extra_lower_bounds'][key] = old

    def test_unverified_cpu_full_state_and_loads_rejected(self):
        for field, bad in [('checkpoint_loads', True), ('checkpoint_loads', 1), ('GPU_operations', 1),
                           ('model_constructions', 1), ('durable_history_matches_current_and_snapshot', False)]:
            old = self.cpu[field]; self.cpu[field] = bad
            with self.assertRaises(ValueError): self.run_fixture()
            self.cpu[field] = old
        self.cpu['checkpoint_validation']['resume_full_state_verified'] = False
        with self.assertRaises(ValueError): self.run_fixture()

    def test_checkpoint_descriptor_mismatch_rejected_without_binary_read(self):
        self.loss['preserved_checkpoint']['sha256'] = 'f' * 64
        with self.assertRaises(ValueError): self.run_fixture()

    def test_history_commit_and_progress_mismatch_rejected(self):
        self.record['latest_durable_checkpoint']['epoch'] = 2
        with self.assertRaises(ValueError): self.run_fixture()

    def test_parent_must_remain_unknown_running(self):
        for key, value in [('status', 'complete'), ('returncode', 0), ('seconds', 0)]:
            old = self.queue.get(key); self.queue[key] = value
            with self.assertRaises(ValueError): self.run_fixture()
            if old is None: del self.queue[key]
            else: self.queue[key] = old

    def test_recovery_lineage_owner_and_ancestry_rejected(self):
        self.owner['execution_version'] = True
        with self.assertRaises(ValueError): self.run_fixture()
        self.owner['execution_version'] = 5
        self.loss['expected_native_identities'][2]['parent_pid'] = 99
        with self.assertRaises(ValueError): self.run_fixture()

    def test_closure_preserves_extra_not_added_and_parent_unknown(self):
        context = self.run_fixture()
        proof = {**context, 'schema': 'lost_f3_process_closure_v3', 'status': 'process_absence_observed',
                 'closure_source': self.entry, 'recovery_registration': self.d('reg2.json'),
                 'runtime_process_absence_evidence': self.loss['process_absence_evidence'],
                 'known_extra_not_added_again_to_original_returned_counters': True}
        with patch.object(L, 'verify_recovery_registration', return_value={'context': context, 'source': self.entry,
                     'registration': proof['recovery_registration']}), \
                patch.object(L, 'read_json', side_effect=lambda b: deepcopy(self.values[b['path']])):
            result = L.validate_closure(self.d('closure.json'), proof, self.loss['original_attempt'], self.record,
                                       dict(zip(('condition', 'seed', 'fold'), L.IDENTITY)))
            self.assertEqual(result['known_extra_backward_returns_outside_checkpoint'], 26)
            self.assertIsNone(result['queue_attempt_final_elapsed_seconds'])
            self.assertEqual(result['original_queue_attempt'], self.loss['old_queue_attempt_original'])
            proof['known_extra_not_added_again_to_original_returned_counters'] = False
            with self.assertRaises(ValueError):
                L.validate_closure(self.d('closure.json'), proof, self.loss['original_attempt'], self.record,
                                   dict(zip(('condition', 'seed', 'fold'), L.IDENTITY)))

    def test_old_schema_delegates_exactly_without_new_pause_requirements(self):
        proof = {'schema': 'lost_f3_process_closure_v1'}
        with patch.object(L, 'verify_delegated_source'), patch.object(L.L2, 'validate_closure', return_value='old') as old:
            self.assertEqual(L.validate_closure('binding', proof, 'artifact', 'record', 'fit'), 'old')
            old.assert_called_once_with('binding', proof, 'artifact', 'record', 'fit')

    def test_duplicate_closures_rejected_before_validation(self):
        proof = {'original_attempt': self.loss['original_attempt']}
        paths = [L.S.WORK / 'output/synthetic_loss2/one.json', L.S.WORK / 'output/synthetic_loss2/two.json']
        for path in paths: self.d(path.name, proof)
        with patch.object(L, 'ROOT', SimpleNamespace(glob=lambda pattern: paths)), \
                patch.object(L, 'descriptor', side_effect=self.fake_descriptor), \
                patch.object(L, 'read_json', side_effect=lambda b: deepcopy(self.values[b['path']])), \
                patch.object(L, 'validate_closure', side_effect=AssertionError('Ambiguity must fail first')):
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                L.verify_lost_f3_attempt(self.loss['original_attempt'], self.record, self.loss)

    def test_exact88_source_registration_and_missing_old_source_fail(self):
        context = self.run_fixture()
        old_sources = [self.d('old_source_' + str(i) + '.py') for i in range(84)]
        self.prior['value']['sources'] = old_sources
        required = [L.ENTRY_SOURCE, L.SOURCE, L.S.REPORT / 'audit_completed_finetune_checkpoints_v8.py',
                    L.S.REPORT / 'cost_aggregation_v9.py']
        extra = []
        for path in required:
            relative = path.relative_to(L.S.WORK).as_posix()
            binding = {'path': relative, 'sha256': hashlib.sha256(relative.encode()).hexdigest()}
            self.bindings[relative] = binding; extra.append(binding)
        registry = self.d('registry.json')
        manifest = {'schema': 'q5_checkpoint_recovery_execution_v3', 'version': 3, 'source': extra[0],
                    'base_queue_registration': self.prior['base_queue_registration'],
                    'previous_recovery_registration': self.prior['registration'], 'registry': registry,
                    'loss_evidence': self.loss_binding, 'process_absence_evidence': context['process_absence_evidence'],
                    'checkpoint_cpu_resume_proof': context['checkpoint_cpu_resume_proof'],
                    'sources': old_sources + extra, 'cpu_auditor': extra[2], 'cost_collector': extra[3],
                    'same_formal_F3_and_E3_core': True, 'training_protocol_or_hyperparameters_changed': False}
        binding = self.d('registration2.json', manifest)
        with patch.object(L, 'read_json', side_effect=lambda b: deepcopy(self.values[b['path']])), \
                patch.object(L, 'descriptor', side_effect=self.fake_descriptor), \
                patch.object(L, 'previous_registration', return_value=self.prior), \
                patch.object(L, 'verify_recovery_inputs', return_value=context), \
                patch.object(L.R, 'registration_descriptor', return_value=registry):
            self.assertTrue(L.verify_recovery_registration(binding)['passed'])
            for key, bad in [('version', True), ('same_formal_F3_and_E3_core', False),
                             ('previous_recovery_registration', registry), ('cpu_auditor', extra[3]),
                             ('sources', manifest['sources'][1:]), ('sources', manifest['sources'] + [extra[0]])]:
                with self.subTest(key=key):
                    before = manifest[key]; manifest[key] = bad
                    with self.assertRaises(ValueError): L.verify_recovery_registration(binding)
                    manifest[key] = before

    def test_pause_exact_native_success_and_identity_required(self):
        for key, bad in [('status','running'), ('automatic_resume_authorized', True),
                         ('runtime_memory_preserved', False), ('actions', [])]:
            with self.subTest(key=key):
                old = self.pause[key]; self.pause[key] = bad
                with self.assertRaises(ValueError): self.run_fixture()
                self.pause[key] = old

    def test_original_pause_threads_and_trace_hash_required(self):
        self.pause['verification'][0]['threads'][0]['wait_reason'] = 'UserRequest'
        with self.assertRaises(ValueError): self.run_fixture()
        self.pause['verification'][0]['threads'][0]['wait_reason'] = 'Suspended'
        self.paused_progress['running_observer_records'][0]['sha256'] = 'f'*64
        with self.assertRaises(ValueError): self.run_fixture()

    def test_cpu_proof_must_bind_pause(self):
        self.cpu['pause_native_proof'] = self.d('wrong_pause.json')
        with self.assertRaises(ValueError): self.run_fixture()

    def test_old_loss2_schema_delegates_without_new_pause_contract(self):
        proof = {'schema': 'lost_f3_process_closure_v2'}
        with patch.object(L, 'verify_delegated_source'), patch.object(L.L2, 'validate_closure', return_value='old2') as old:
            self.assertEqual(L.validate_closure('binding', proof, 'artifact', 'record', 'fit'), 'old2')
            old.assert_called_once_with('binding', proof, 'artifact', 'record', 'fit')

    def test_readers_reject_binary_and_outside_workspace_before_read(self):
        for path in ('output/synthetic.pt', '../outside.json'):
            with self.assertRaises(ValueError): L.read_json({'path': path, 'sha256': 'a' * 64})

    def test_arbitrary_loss_certificate_rejected_before_read(self):
        with patch.object(L, 'read_json', side_effect=AssertionError('unregistered data read')):
            with self.assertRaisesRegex(ValueError, 'Only the reviewed actual'):
                L.verify_recovery_inputs(self.loss_binding)


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (Loss3Tests, Cpu8Tests))
    stream = io.StringIO(); start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'seconds': time.perf_counter() - start, 'output': stream.getvalue(),
              'sources': [L.descriptor(Path(__file__)), L.descriptor(A.SOURCE), L.descriptor(L.SOURCE)],
              'scope': 'Pure stdlib fake evidence and unchanged CPU5 counter/math delegation; actual X/L/audit not invoked',
              'actual_checkpoint_loads': 0, 'model_constructions': 0, 'GPU_operations': 0, 'process_operations': 0}
    target = REPORT / ('lost_f3_v3_cpu8_stdlib_tests_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf8') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({'passed': report['passed'], 'tests_run': result.testsRun, 'proof': L.descriptor(target),
                      'output': stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
