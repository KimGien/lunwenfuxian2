"""Independent loss3 semantic fault injection. Existing JSON/source + stdlib only."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import hashlib
import importlib.util
import io
import json
import sys
import time
import unittest

A = Path(__file__).resolve().parent
WORK = A.parents[2]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v3 as L


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load(A / 'cost_aggregation_v9.py', 'independent_cost9')
CPU = load(A / 'audit_completed_finetune_checkpoints_v8.py', 'independent_cpu8')


def d(path):
    path = Path(path).resolve()
    if path.suffix not in ('.json', '.py'):
        raise ValueError('Review reads text/source only')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class Loss3Faults(unittest.TestCase):
    def setUp(self):
        self.values = {}
        def cache(binding):
            if not isinstance(binding, dict): return
            if set(binding) == {'path', 'sha256'} and Path(binding['path']).suffix == '.json':
                if binding['path'] in self.values: return
                path = WORK / binding['path']
                if d(path) != binding: raise ValueError('Actual evidence bytes changed')
                value = json.loads(path.read_text(encoding='utf-8-sig'))
                self.values[binding['path']] = value
                return value
        def walk(value):
            if isinstance(value, dict):
                cache(value)
                for v in value.values(): walk(v)
            elif isinstance(value, list):
                for v in value: walk(v)
        cache(L.LOSS_BINDING)
        self.loss = self.values[L.LOSS_BINDING['path']]
        # Keep this event's explicit evidence scope. Historical manifests bind
        # mutable paths at their past snapshots, so recursive traversal of all
        # prior manifests is neither required nor a current-byte assertion.
        for binding in self.loss.values(): cache(binding)
        self.cpu = self.values[self.loss['checkpoint_cpu_resume_proof']['path']]
        cache(self.cpu['snapshot_manifest'])
        for binding in self.values[self.cpu['snapshot_manifest']['path']]['retained'].values():
            cache(binding['snapshot']); cache(binding['original'])
        self.pause = self.values[self.loss['pause_native_proof']['path']]
        self.record = self.values[self.loss['attempt_snapshot']['path']]
        self.prior = {'passed': True, 'registration': self.loss['previous_recovery_registration'],
            'value': self.values[self.loss['previous_recovery_registration']['path']],
            'base_queue_registration': self.loss['base_queue_registration'],
            'source': self.values[self.loss['previous_recovery_registration']['path']]['source']}
        self.observer = {'unflushed_tail_unknown': True, 'unflushed_tail_counters': None}

    def run_case(self):
        with patch.object(L, 'read_json', side_effect=lambda b: deepcopy(self.values[b['path']])), \
             patch.object(L, 'previous_registration', return_value=deepcopy(self.prior)), \
             patch.object(L.X, 'verify_observer_record', return_value=deepcopy(self.observer)):
            return L.verify_recovery_inputs(L.LOSS_BINDING)

    def reject_change(self, obj, field, value):
        old = deepcopy(obj.get(field))
        obj[field] = value
        try:
            with self.assertRaises((ValueError, KeyError, TypeError)):
                self.run_case()
        finally:
            obj[field] = old

    def test_actual_json_fixture_semantics_and_no_mutation(self):
        before = deepcopy(self.values)
        result = self.run_case()
        self.assertTrue(result['passed'])
        self.assertEqual(result['durable'], {'epoch': 54, 'optimizer_updates': 216, 'backward_calls': 1458})
        self.assertEqual(result['known_extra_lower_bounds'], {'optimizer_updates': 4, 'backward_calls': 27})
        self.assertEqual(before, self.values)

    def test_all_unknown_endpoints_must_remain_explicit_null(self):
        for field in L.UNKNOWN_FIELDS:
            with self.subTest(field=field): self.reject_change(self.loss, field, 0)

    def test_three_native_pause_proofs_not_interchangeable(self):
        for field in ('pause_native_proof', 'pause_capture', 'pause_progress_and_observer'):
            with self.subTest(field=field): self.reject_change(self.loss, field, self.loss['old_owner_snapshot'])

    def test_fake_pause_success_missing_birth_or_resumed_thread_reject(self):
        self.reject_change(self.pause, 'automatic_resume_authorized', True)
        self.reject_change(self.pause['actions'][0], 'native_status', 1)
        self.reject_change(self.pause['verification'][0], 'creation_100ns', '42')
        self.reject_change(self.pause['verification'][0]['threads'][0], 'wait_reason', 'UserRequest')

    def test_unknown_or_partial_absence_reject(self):
        absence = self.values[self.loss['process_absence_evidence']['path']]
        for key, value in [('query_succeeded', False), ('observed_rows', [{}]), ('workspace_training_workers', [{}]), ('absent_pids', []), ('exit_codes', [0])]:
            with self.subTest(key=key): self.reject_change(absence, key, value)

    def test_guessed_delta_bool_and_mismatched_checkpoint_reject(self):
        self.reject_change(self.loss['known_extra_lower_bounds'], 'backward_calls', 26)
        self.reject_change(self.loss['known_extra_lower_bounds'], 'optimizer_updates', True)
        self.reject_change(self.loss['durable'], 'epoch', 55)
        self.reject_change(self.loss['preserved_checkpoint'], 'sha256', 'a' * 64)

    def test_not_full_cpu_or_wrong_role_reject(self):
        for key, value in [('checkpoint_loads', True), ('checkpoint_loads', 1), ('fold', 2), ('GPU_operations', 1), ('test_probability_reads', 1)]:
            with self.subTest(key=key): self.reject_change(self.cpu, key, value)
        self.reject_change(self.cpu['checkpoint_validation'], 'resume_full_state_verified', False)
        self.reject_change(self.cpu['checkpoint_validation'], 'checkpoint_load_device', 'cuda')

    def test_both_old_schemas_delegate_without_reinterpreting_identity(self):
        for version in (1, 2):
            proof = {'schema': 'lost_f3_process_closure_v' + str(version)}
            with patch.object(L, 'verify_delegated_source'), patch.object(L.L2, 'validate_closure', return_value='original') as fn:
                self.assertEqual(L.validate_closure('binding', proof, 'artifact', 'record', 'fit'), 'original')
                fn.assert_called_once_with('binding', proof, 'artifact', 'record', 'fit')
        for version in (1, 2):
            with patch.object(L, 'read_json', return_value={'schema': 'q5_checkpoint_recovery_execution_v' + str(version)}), patch.object(L, 'verify_delegated_source'), patch.object(L.L2, 'verify_recovery_registration', return_value='original') as fn:
                self.assertEqual(L.verify_recovery_registration('binding'), 'original')
                fn.assert_called_once_with('binding')

    def test_duplicate_closure_rejects_before_any_semantic_validation(self):
        files = [WORK / 'output/fake_loss3_one.json', WORK / 'output/fake_loss3_two.json']
        proof = {'original_attempt': self.loss['original_attempt']}
        with patch.object(L, 'ROOT', SimpleNamespace(glob=lambda *_: files)), patch.object(L, 'descriptor', side_effect=lambda p: {'path': str(p), 'sha256': 'a'*64}), patch.object(L, 'read_json', return_value=proof), patch.object(L, 'validate_closure', side_effect=AssertionError('Ambiguity must reject first')):
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                L.verify_lost_f3_attempt(self.loss['original_attempt'], self.record, self.loss)

    def test_cpu5_original_math_and_three_checkpoint_functions_unchanged(self):
        original = load(CPU.DELEGATE, 'independent_original_cpu5')
        self.assertIs(CPU.B.L, L)
        self.assertIsNot(original.L, L)
        for name in ('audit', 'cpu_audit', 'checkpoint_payload', 'optimizer_group_boundary', 'trace_boundary', 'legacy_trace_boundary', 'completed_fit_audit', 'trace_inventory'):
            with self.subTest(name=name):
                self.assertEqual(getattr(CPU.B, name).__code__, getattr(original, name).__code__)


class Cost9Faults(unittest.TestCase):
    def setUp(self):
        old = load(A / 'test_cost_aggregation_v7.py', 'independent_synthetic_cost7_fixture')
        fixture = old.CostV7Guards()
        fixture.setUp()
        self.base, self.records, self.proofs = fixture.base, fixture.records, fixture.proofs
        self.old = old
        for event, updates, back in [('LaBraM4s', 4, 27)]:
            path = 'out/finetune_v2_attempts/' + event + '/third/attempt.json'
            self.records[path] = {'implementation': {'version': 3}, 'condition': event, 'seed': 33, 'fold': 1, 'status': 'running'}
            self.proofs[path] = old.proof(event, updates, back)
            self.base['cost_records'].append({'artifact': old.descriptor(path), 'category': 'formal_training_attempt', 'aggregation_role': 'primary_observed_invocation', 'status': 'running', 'seconds_observed': 90000, 'observed_counters': {'optimizer_returned': 220, 'backward_returned': 1485}})
        self.proofs[list(self.proofs)[1]]['known_extra_backward_returns_outside_checkpoint'] = 26

    def run_case(self, base=None):
        return C.augment(deepcopy(base or self.base), verifier=lambda b,*_: deepcopy(self.proofs[b['path']]), reader=lambda b: deepcopy(self.records[b['path']]), verifier_source=self.old.descriptor('L3.py'))

    def test_three_event_deltas_distinct_and_not_added(self):
        value = self.run_case()
        self.assertEqual(len(value['source_bound_lost_f3_attempts']), 3)
        self.assertEqual([p['known_extra_backward_returns_outside_checkpoint'] for p in value['source_bound_lost_f3_attempts']], [27, 26, 27])
        self.assertEqual([r['observed_counters'] for r in value['cost_records']], [r['observed_counters'] for r in self.base['cost_records']])
        self.assertEqual(value['formal_durable_updates_known_subtotal'], self.base['formal_durable_updates_known_subtotal'])
        self.assertTrue(all(r['additional_optimizer_returns_added_for_recovery'] == r['additional_backward_returns_added_for_recovery'] == 0 for r in value['cost_records']))
        self.assertFalse(value['input_read_errors'])

    def test_three_lost_elapsed_unknown_even_when_raw_numeric(self):
        value = self.run_case()
        self.assertTrue(all(r['seconds_observed'] is None and r['unflushed_tail_counters'] is None for r in value['cost_records']))
        self.assertIsNone(value['all_plan_wall_seconds_exact'])
        self.assertIsNone(value['all_plan_physical_optimizer_returns_exact'])

    def test_old_schema_reaugmentation_refused(self):
        for version in (6, 7, 8, 9):
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.run_case({**self.base, 'schema_version': version})

    def test_loss_parent_supplemental_unknown_not_training(self):
        p = self.old.descriptor('out/continuation_queue_v5/attempts/third/status.json')
        self.records[p['path']] = {'status': 'running', 'seconds': None}
        self.proofs[list(self.proofs)[-1]].update(original_queue_attempt=p, queue_attempt_final_elapsed_seconds=None)
        self.base['cost_records'].append({'artifact': p, 'category': 'dispatcher_elapsed', 'aggregation_role': 'supplemental_nested_or_unmeasured', 'status': 'running', 'seconds_observed': 999999})
        value = self.run_case()
        self.assertIsNone(value['cost_records'][-1]['seconds_observed'])
        self.assertFalse(value['cost_records'][-1]['parent_elapsed_added_to_formal_or_primary_work'])
        self.assertFalse(value['input_read_errors'])


def main():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (Loss3Faults, Cost9Faults))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    value = {'schema': 'independent_loss3_CPU8_cost9_stdlib_review_v1', 'passed': result.wasSuccessful(), 'reviewer': '/root/loss3_independent_review', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'output': stream.getvalue(), 'sources': [d(Path(__file__)), d(Path(L.__file__)), d(CPU.SOURCE), d(C.SOURCE)], 'scope': 'Independent semantic fault injection over existing JSON loaded once; original full verifiers/audits/collector and process/native/GPU/model/Tensor paths are mocked or never called.', 'checkpoint_loads': 0, 'checkpoint_binary_reads': 0, 'process_actions': 0, 'model_operations': 0, 'GPU_operations': 0, 'test_probability_reads': 0}
    path = A / ('independent_loss3_CPU8_cost9_stdlib_review_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': d(path), 'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'output': value['output'] if not result.wasSuccessful() else 'all passed'}, ensure_ascii=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
