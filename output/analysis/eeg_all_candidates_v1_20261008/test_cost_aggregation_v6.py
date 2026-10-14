"""Pure JSON fixtures: process loss never creates time, steps or a zero tail."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cost_v6_tests', ROOT / 'cost_aggregation_v6.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


def desc(name): return {'path': name, 'sha256': 'a' * 64}


class CostV6Tests(unittest.TestCase):
    def setUp(self):
        self.record = {'implementation': {'version': 3}, 'status': 'running', 'condition': 'C', 'seed': 33, 'fold': 4}
        self.proof = {'passed': True, 'persisted_counters_are_lower_bounds': True, 'unflushed_tail_unknown': True,
            'extra_work_not_added_again': True, 'original_status': 'running', 'graceful_terminal_flush': False,
            'unflushed_tail_counters': None, 'final_elapsed_seconds': None,
            'known_extra_optimizer_returns_outside_checkpoint': 4, 'known_extra_backward_returns_outside_checkpoint': 27,
            **{key: desc(key + '.json') for key in ('closure', 'recovery_registration', 'closure_source', 'checkpoint_cpu_resume_proof')}}
        self.row = {'artifact': desc('out/finetune_v2_attempts/C/id/attempt.json'), 'category': 'formal',
            'aggregation_role': 'primary_observed_invocation', 'status': 'running', 'seconds_observed': 50000,
            'observed_counters': {'optimizer_returned': 36, 'backward_returned': 243}}
        self.base = {'source': desc('v5.py'), 'cost_records': [self.row], 'category_observations': {'formal': {
            'counter_totals': {'optimizer_returned': {'known_observed_subtotal': 36}, 'backward_returned': {'known_observed_subtotal': 243}}}},
            'inputs': [], 'input_read_errors': [], 'rules': [], 'source_bound_stopped_attempts': [{'extra': [0, 2]}],
            'legacy_resume_events': [{'extra': [4, 27]}]}

    def run_case(self, base=None, proof=None, verifier=None):
        return C.augment(deepcopy(base or self.base), reader=lambda _: deepcopy(self.record),
                         verifier=verifier or (lambda *_: deepcopy(self.proof if proof is None else proof)))

    def test_no_added_physical_or_durable_work(self):
        result = self.run_case(); row = result['cost_records'][0]
        self.assertEqual(self.row['observed_counters'], row['observed_counters'])
        self.assertEqual(36, result['category_observations']['formal']['counter_totals']['optimizer_returned']['known_observed_subtotal'])
        self.assertEqual(0, row['additional_optimizer_returns_added_for_recovery'])
        self.assertEqual(0, row['additional_backward_returns_added_for_recovery'])

    def test_endpoint_inferred_overnight_removed(self):
        self.row['finished_at'] = '2026-10-10T20:00:00+08:00'
        result = self.run_case()
        self.assertIsNone(result['cost_records'][0]['seconds_observed'])
        self.assertIsNone(result['category_observations']['formal']['known_recorded_scope_seconds_subtotal'])
        self.assertEqual(1, result['category_observations']['formal']['seconds_unknown_records'])

    def test_real_zero_other_attempt_retained(self):
        other = deepcopy(self.row); other.update(artifact=desc('out/other/a.json'), seconds_observed=0)
        self.base['cost_records'].append(other)
        result = self.run_case()
        self.assertEqual(0, result['category_observations']['formal']['known_recorded_scope_seconds_subtotal'])
        self.assertEqual(1, result['category_observations']['formal']['seconds_unknown_records'])

    def test_original_running_and_unknown_tail_preserved(self):
        result = self.run_case(); row = result['cost_records'][0]
        self.assertEqual('running', row['status']); self.assertIsNone(row['unflushed_tail_counters'])
        self.assertIsNone(result['all_plan_physical_optimizer_returns_exact'])
        self.assertIsNone(result['all_plan_wall_seconds_exact'])
        self.assertEqual(self.base['source_bound_stopped_attempts'], result['source_bound_stopped_attempts'])
        self.assertEqual(self.base['legacy_resume_events'], result['legacy_resume_events'])

    def test_bad_closure_cannot_pass(self):
        def fail(*_): raise ValueError('source changed')
        result = self.run_case(verifier=fail)
        self.assertFalse(result['lost_f3_closure_verification']['passed_for_observed_closures'])
        self.assertEqual(1, len(result['input_read_errors']))
        self.assertIsNone(result['cost_records'][0]['seconds_observed'])

    def test_unknown_closure_not_invented(self):
        result = self.run_case(verifier=lambda *_: None)
        self.assertEqual([], result['source_bound_lost_f3_attempts'])
        self.assertIsNone(result['cost_records'][0]['seconds_observed'])

    def test_invalid_disclosure_rejected(self):
        for key, value in [('final_elapsed_seconds', 0), ('unflushed_tail_counters', {}), ('graceful_terminal_flush', True),
                           ('known_extra_optimizer_returns_outside_checkpoint', True), ('known_extra_backward_returns_outside_checkpoint', 28)]:
            with self.subTest(key=key):
                proof = deepcopy(self.proof); proof[key] = value
                self.assertTrue(self.run_case(proof=proof)['input_read_errors'])

    def test_duplicate_primary_closure_rejected(self):
        self.base['cost_records'].append(deepcopy(self.row))
        self.assertTrue(self.run_case()['input_read_errors'])

    def test_duplicate_alias_not_added(self):
        other = deepcopy(self.row); other['aggregation_role'] = 'duplicate_invocation_artifact'
        self.base['cost_records'].append(other)
        result = self.run_case()
        self.assertEqual([], result['input_read_errors'])
        self.assertEqual(1, result['category_observations']['formal']['seconds_unknown_records'])
        self.assertEqual(4, len(result['inputs']))

    def test_completed_attempt_unchanged(self):
        self.record['status'] = 'complete'
        result = self.run_case()
        self.assertEqual(self.row, result['cost_records'][0])

    def test_delegate_hash_and_math_unchanged(self):
        delegate = C.load_v5()
        self.assertEqual(C.DELEGATE_SHA, C.L.descriptor(C.DELEGATE)['sha256'])
        self.assertEqual('46af6b28f80332f99b22f9373bb5902c0f7c42e9676c54a56da9c2f62eddbd40', delegate.DELEGATE_SHA)


if __name__ == '__main__': unittest.main(verbosity=2)
