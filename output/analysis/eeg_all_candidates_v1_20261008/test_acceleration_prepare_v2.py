"""In-memory stdlib safety tests; no actual device, data, checkpoint or model."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import acceleration_prepare_v2 as A
spec = importlib.util.spec_from_file_location('q5_qualification_under_test', Path(__file__).with_name('qualify_acceleration_v2.py'))
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def d(path):
    return {'path': str(path), 'sha256': 'a' * 64}


def observation(busy=True):
    return {'idle': not busy, 'contexts': [{'pid': 123, 'type': 'C+G', 'process_name': 'synthetic game'}] if busy else [],
            'query': 'synthetic test only; never ran nvidia-smi'}


def proof_fixture():
    context = {'queue': Q.QUEUE, 'queue_registration': d('q5.json'), 'queue_owner': d('owner.json'),
               'queue_child_job': 'C-acceleration-prepare', 'queue_parent_identity': {'ProcessId': 100},
               'queue_child_identity': {'ProcessId': 200}}
    proof = {'schema': 'acceleration_safe_stop_v2', 'mode': 'queue_exclusive_post_prepare',
             'source': d(Q.BUILDER), 'registration': d(Q.D.REGISTRY), 'condition': 'C', 'seed': 11, 'fold': 1,
             'checkpoints': [], 'stopped_process_ids': [], 'zero_optimizer_qualification_only': True,
             'gpu_idle': True, 'passed': True, **context}
    registry = {'folds': {'fold_1': {'train': ['a', 'b'], 'val': ['c'], 'test': ['d']}}}
    fit = {'condition': 'C', 'seed': 11, 'fold': 1}
    return proof, context, registry, fit


def deferred_fixture():
    report = {'schema': 'diagnostic_batching_qualification_attempt_v2', 'condition': 'C', 'status': 'deferred',
              'qualifier_source': d(A.QUALIFIER), 'expanded_batch_activation_authorized': False,
              'production_diagnostic_chunk': 8, 'gpu_stage_entered': False, 'qualification_performed': False,
              'external_gpu_observation': observation()}
    for key in ('optimizer_updates', 'backward_calls', 'test_forward_count', 'model_constructions', 'raw_reads',
                'checkpoint_reads', 'torch_imports', 'forward_started', 'forward_returned', 'encoder_forward_started',
                'encoder_forward_returned', 'encoder_windows_started', 'encoder_windows_returned',
                'forbidden_backward_attempts', 'forbidden_optimizer_attempts'):
        report[key] = 0
    return {'status': 'deferred', 'passed': False, 'artifact': d('deferred.json')}, report


class DeviceParsingTests(unittest.TestCase):
    def test_all_graphics_and_compute_are_busy(self):
        xml = '<nvidia_smi_log><gpu id="0"><processes>' + ''.join(
            f'<process_info><pid>{i}</pid><type>{kind}</type><process_name>fake</process_name></process_info>'
            for i, kind in enumerate(('C', 'G', 'C+G'), 1)) + '</processes></gpu></nvidia_smi_log>'
        self.assertEqual([r['type'] for r in Q.parse_gpu_xml(xml)], ['C', 'G', 'C+G'])

    def test_observed_empty_only_is_idle(self):
        self.assertEqual(Q.parse_gpu_xml('<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>'), [])
        for xml in ('<bad/>', '<nvidia_smi_log/>', '<nvidia_smi_log><gpu/></nvidia_smi_log>',
                    '<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>',
                    '<nvidia_smi_log><gpu><processes><process_info><pid>N/A</pid></process_info></processes></gpu></nvidia_smi_log>'):
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                Q.parse_gpu_xml(xml)

    def test_idle_query_error_is_not_deferral(self):
        with patch.object(Q, 'gpu_observation', side_effect=RuntimeError('query failed')):
            with self.assertRaisesRegex(RuntimeError, 'query failed'):
                Q.strict_idle()


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.proof, self.context, self.registry, self.fit = proof_fixture()
        self.patches = [patch.object(Q, 'json_read', return_value=self.proof), patch.object(Q, 'descriptor', side_effect=d),
                        patch.object(Q, 'checked', return_value=Q.BUILDER.resolve()),
                        patch.object(Q, 'queue_context', return_value=(self.context, self.registry, self.fit)),
                        patch.object(Q, 'gpu_observation', return_value=observation(False))]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def test_quiet_exact_fresh_probe(self):
        result = Q.preflight(Path('fake.json'))
        self.assertEqual(result[4:6], ([], None))
        self.assertEqual(result[3], self.registry['folds']['fold_1'])

    def test_external_context_defers_before_data(self):
        with patch.object(Q, 'gpu_observation', return_value=observation()):
            with self.assertRaises(Q.ExternalGPUContext): Q.preflight(Path('fake.json'))

    def test_busy_at_creation_never_authorizes_later_quiet(self):
        self.proof.update(passed=False, gpu_idle=False, gpu_observation=observation())
        with self.assertRaises(Q.ExternalGPUContext): Q.preflight(Path('fake.json'))

    def test_wrong_sources_queue_identity_or_checkpoint_rejected(self):
        for key, value in [('schema', 'acceleration_safe_stop_v1'), ('queue_owner', d('wrong.json')),
                           ('seed', True), ('fold', 2), ('checkpoints', [d('latest.pt')]), ('zero_optimizer_qualification_only', False)]:
            old = self.proof[key]; self.proof[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): Q.preflight(Path('fake.json'))
            self.proof[key] = old

    def test_actual_delegated_run_defers_without_importing_torch(self):
        records = []
        with patch.object(Q, 'gpu_observation', return_value=observation()), \
             patch.object(Q, 'atomic_json', side_effect=lambda path, value, exclusive=False: records.append(copy.deepcopy(value))), \
             patch.object(Q.D, 'run', wraps=Q.D.run):
            result = Q.execute_explicit(types.SimpleNamespace(stop_proof=Path('fake.json'), repeats=2))
        self.assertEqual(result['status'], 'deferred')
        self.assertNotIn('torch', sys.modules)
        record = records[-1]
        self.assertEqual(record['status'], 'deferred')
        self.assertFalse(record['gpu_stage_entered'])
        self.assertEqual([record[k] for k in ('model_constructions', 'raw_reads', 'checkpoint_reads', 'optimizer_updates')], [0] * 4)
        self.assertEqual(record['schema'], 'diagnostic_batching_qualification_attempt_v2')

    def test_late_external_context_releases_owned_lease_without_torch(self):
        records = []
        owner = {}
        class MemoryOwner:
            def exists(self): return bool(owner)
            def unlink(self): owner.clear()
        lease = MemoryOwner()
        def save(path, value, exclusive=False):
            if path is lease:
                owner.update(value)
            records.append(copy.deepcopy(value))
        def read(path):
            return dict(owner) if path is lease else self.proof
        with patch.object(Q, 'atomic_json', side_effect=save), patch.object(Q.D, 'OWNER', lease), patch.object(Q.D, 'descriptor', side_effect=d), \
             patch.object(Q.D, 'json_read', side_effect=read), \
             patch.object(Q, 'gpu_observation', side_effect=[observation(False), observation()]):
            result = Q.execute_explicit(types.SimpleNamespace(stop_proof=Path('fake.json'), repeats=2))
        self.assertFalse(owner)
        self.assertEqual(result['status'], 'deferred')
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(records[-1]['model_constructions'], 0)


class QueueIdentityTests(unittest.TestCase):
    def setUp(self):
        q5_path = Q.OUT / Q.QUEUE / 'registration.json'
        self.owner = {'pid': 100, 'child_pid': os.getppid(), 'child_job': 'C-acceleration-prepare',
                      'source': d(Q.MOD / (Q.QUEUE + '.py')), 'execution_registration': d(q5_path), 'execution_version': 5}
        self.previous_sources = [d('old_source_' + str(i) + '.py') for i in range(59)]
        self.sources = self.previous_sources + [d(Q.SOURCE), d(Q.BUILDER), d(Q.MOD / (Q.QUEUE + '.py'))]
        self.registry = {'fits': [{'condition': 'C', 'stage': '3', 'seed': 11, 'fold': 1}]}
        self.q4_binding = {'path': str(Q.OUT / 'continuation_queue_v4/registration.json'), 'sha256': Q.Q4_SHA}
        self.queue = {'version': 5, 'registry': {'path': str(Q.D.REGISTRY), 'sha256': Q.D.REGISTRY_SHA},
                      'previous_execution_registration': self.q4_binding, 'sources': self.sources}
        def desc(path):
            if path == Q.D.REGISTRY: return self.queue['registry']
            if path == Q.OUT / 'continuation_queue_v4/registration.json': return self.q4_binding
            if Path(path).name in Q.D.DELEGATE_HASHES:
                return {'path': str(path), 'sha256': Q.D.DELEGATE_HASHES[Path(path).name]}
            return d(path)
        def read(path):
            if path == Q.D.REGISTRY: return self.registry
            if str(path) == self.q4_binding['path']: return {'sources': self.previous_sources}
            if Path(path).name == 'registration.json': return self.queue
            return self.owner
        self.patches = [patch.object(Q, 'descriptor', side_effect=desc), patch.object(Q, 'json_read', side_effect=read),
                        patch.object(Q, 'checked', side_effect=lambda binding: Path(binding['path'])), patch.object(Q.D, 'pid_alive', return_value=True),
                        patch.object(Q.D, 'windows_identity', side_effect=lambda pid: {'ProcessId': pid, 'ParentProcessId': 100}),
                        patch.object(Path, 'exists', return_value=False)]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    @unittest.skipUnless(os.name == 'nt', 'Windows in-memory process ownership fixture')
    def test_exact_queue_identity_is_read_only(self):
        with patch.object(Q, 'gpu_observation', side_effect=AssertionError('queue context cannot query GPU')):
            context, _, fit = Q.queue_context('C')
        self.assertEqual(context['queue_child_job'], 'C-acceleration-prepare')
        self.assertEqual(fit['fold'], 1)

    @unittest.skipUnless(os.name == 'nt', 'Windows in-memory process ownership fixture')
    def test_wrong_job_pid_missing_source_or_duplicate_source_rejected(self):
        for key, value in [('child_job', 'C-formal-fit'), ('child_pid', -100), ('source', d('wrong.py')),
                           ('execution_registration', d('wrong.json')), ('execution_version', True), ('execution_version', 4)]:
            old = self.owner[key]; self.owner[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): Q.queue_context('C')
            self.owner[key] = old
        for sources in ([d(Q.BUILDER)], self.sources + [self.sources[0]]):
            self.queue['sources'] = sources
            with self.subTest(sources=sources), self.assertRaises(ValueError): Q.queue_context('C')

    @unittest.skipUnless(os.name == 'nt', 'Windows in-memory process ownership fixture')
    def test_child_ancestry_and_old_live_queue_are_rejected(self):
        with patch.object(Q.D, 'windows_identity', return_value={'ProcessId': 1, 'ParentProcessId': -100}):
            with self.assertRaises(ValueError): Q.queue_context('C')
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaises(ValueError): Q.queue_context('C')

    def test_linux_uses_actual_linux_pid_and_windows_cim(self):
        linux_binding = d('linux_owner.json')
        original_reader = Q.json_read
        fake_os = types.SimpleNamespace(name='posix', getpid=lambda: 700, getppid=lambda: 600)
        with patch.object(Q, 'os', fake_os), \
             patch.object(Q, 'json_read', side_effect=lambda p: {'pid': 700} if str(p) == 'linux_owner.json' else original_reader(p)), \
             patch.object(Q.D, 'pid_alive', side_effect=lambda pid: pid == 700) as linux_live:
            context, _, _ = Q.queue_context('C', linux_owner=linux_binding)
            self.assertEqual(context['windows_queue_owner'], d(Q.OUT / Q.QUEUE / 'owner.json'))
            linux_live.assert_called_once_with(700)

    def test_q4_lineage_cannot_be_omitted_or_truncated(self):
        self.queue['previous_execution_registration'] = d('fake_q4.json')
        with self.assertRaises(ValueError): Q.queue_context('C')
        self.queue['previous_execution_registration'] = self.q4_binding
        self.queue['sources'] = self.sources[1:]
        with self.assertRaises(ValueError): Q.queue_context('C')


class ResultTests(unittest.TestCase):
    def test_deferred_is_not_qualified(self):
        result, report = deferred_fixture()
        with patch.object(A.S, 'desc', side_effect=d):
            self.assertEqual(A.classify_result('C', result, report), 'retained_original8_optional_probe_deferred_external_gpu')

    def test_any_model_work_or_false_zero_refuses_deferral(self):
        result, original = deferred_fixture()
        with patch.object(A.S, 'desc', side_effect=d):
            for key in [k for k, v in original.items() if type(v) is int and v == 0]:
                for value in (1, False):
                    report = copy.deepcopy(original); report[key] = value
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        A.classify_result('C', result, report)
            for key, value in [('gpu_stage_entered', True), ('qualification_performed', True),
                               ('expanded_batch_activation_authorized', True), ('production_diagnostic_chunk', 16),
                               ('external_gpu_observation', observation(False))]:
                report = copy.deepcopy(original); report[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError): A.classify_result('C', result, report)

    def test_faster_qualified_result_still_requires_future_contract(self):
        _, report = deferred_fixture()
        report.update(schema='diagnostic_batching_qualification_v2', passed=True, selected_chunk=16)
        with patch.object(A.S, 'desc', side_effect=d):
            self.assertEqual(A.classify_result('C', {'status': 'complete', 'passed': True}, report),
                             'expanded_batch_activation_pending_versioned_contract')

    def test_sidecar_always_original_chunk_and_no_qualification(self):
        result, report = deferred_fixture()
        for status, passed in [('deferred', False), ('complete', True), ('complete', False)]:
            result['status'], result['passed'] = status, passed
            report.update(schema='diagnostic_batching_qualification_attempt_v2' if status == 'deferred' else 'diagnostic_batching_qualification_v2', passed=passed)
            captures = []
            with patch.object(A, 'registration_path', return_value=types.SimpleNamespace(exists=lambda: False)), \
                 patch.object(A.S, 'desc', side_effect=lambda p: d(str(p))), patch.object(A.S, 'checked', return_value='memory'), \
                 patch.object(A.S, 'read', return_value=report), patch.object(A.R, '_condition_lock', return_value=d('lock.json')), \
                 patch.object(A.R, 'registration_descriptor', return_value=d('registration.json')), \
                 patch.object(A.S, 'save', side_effect=lambda p, v, **k: captures.append(v)), \
                 patch.object(A.A, 'load_execution_acceleration') as verifier:
                A.create_registration('C', result)
            entry = captures[0]['conditions']['C']
            self.assertEqual(entry['diagnostic_window_chunk'], 8)
            self.assertIsNone(entry['qualification'])
            self.assertEqual(entry['tested_qualification'], result['artifact'])
            verifier.assert_called_once()

    def test_existing_sidecar_is_verified_and_never_rewritten(self):
        with patch.object(A, 'registration_path', return_value=types.SimpleNamespace(exists=lambda: True)), \
             patch.object(A.S, 'desc', return_value=d('existing.json')), patch.object(A.A, 'load_execution_acceleration') as verify, \
             patch.object(A.S, 'save', side_effect=AssertionError('must not rewrite')), \
             patch.object(A, 'load_qualifier', side_effect=AssertionError('must not probe')):
            self.assertEqual(A.run('C'), d('existing.json'))
            verify.assert_called_once()

    def test_failed_only_actual_cuda_oom_is_optional(self):
        _, report = deferred_fixture()
        report.update(status='failed', gpu_stage_entered=True, error='CUDA out of memory')
        result = {'status': 'failed_candidate_oom', 'passed': False}
        with patch.object(A.S, 'desc', side_effect=d):
            self.assertEqual(A.classify_result('C', result, report), 'retained_original8_candidate_oom')
            report['error'] = 'source hash mismatch'
            with self.assertRaises(ValueError): A.classify_result('C', result, report)

    def test_delegate_run_code_unchanged_and_source_truthful(self):
        old = ast.parse(Q.DELEGATE.read_text(encoding='utf-8'))
        run = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        # The loaded function comes from compiling the sealed complete V1 text.
        self.assertEqual(Q.D.run.__code__.co_filename, str(Q.DELEGATE))
        self.assertEqual(Q.D.run.__globals__['__file__'], str(Q.SOURCE))
        self.assertTrue(any(isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names) for n in ast.walk(run)))
        value = Q.provenance({'schema': 'diagnostic_batching_qualification_v1'})
        self.assertEqual(value['qualifier_source'], Q.descriptor(Q.SOURCE))
        self.assertEqual(value['delegated_measurement_source']['sha256'], Q.DELEGATE_SHA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
