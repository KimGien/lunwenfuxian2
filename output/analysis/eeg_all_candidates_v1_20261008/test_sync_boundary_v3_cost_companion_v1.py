"""Small stdlib fixtures only; no actual boundary/probe/final-cost execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPORT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_sync_control_cost_test', REPORT / 'sync_boundary_v3_cost_companion_v1.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


def d(name): return {'path': name, 'sha256': 'a' * 64}


class SpanTests(unittest.TestCase):
    def test_nested_spans_not_summed(self):
        value = {'hold_started_monotonic': 10., 'hold_ended_monotonic': 110., 'hold_seconds': 100.,
                 'children_wait_ended_monotonic': 60., 'children_wait_seconds': 50.,
                 'probe_intervals': [{'qualifier': C.QUALIFIERS[0], 'started_monotonic': 65., 'ended_monotonic': 80., 'seconds': 15.},
                                     {'qualifier': C.QUALIFIERS[1], 'started_monotonic': 85., 'ended_monotonic': 105., 'seconds': 20.}]}
        result = C.control_spans(value)
        self.assertEqual(100., result['hold']['seconds']); self.assertTrue(result['intervals_are_nested_not_additive'])
        self.assertIsNone(result['GPU_busy_seconds']); self.assertFalse(result['pause_or_overnight_added_to_training'])

    def test_unknown_duration_not_inferred_from_known_endpoints(self):
        self.assertIsNone(C.elapsed_span(10., 100., None)['seconds'])
        self.assertIsNone(C.elapsed_span(None, None, None)['seconds'])

    def test_real_zero_duration_retained(self): self.assertEqual(0, C.elapsed_span(4, 4, 0)['seconds'])

    def test_invalid_numbers_and_endpoint_mismatch_rejected(self):
        for args in ((1, 2, True), (1, 2, float('nan')), (2, 1, 1), (1, 2, 8), (None, None, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError): C.elapsed_span(*args)

    def test_probe_cannot_overlap_drain_or_another_probe(self):
        base = {'hold_started_monotonic': 0, 'hold_ended_monotonic': 100, 'hold_seconds': 100,
                'children_wait_ended_monotonic': 50, 'children_wait_seconds': 50,
                'probe_intervals': [{'qualifier': C.QUALIFIERS[0], 'started_monotonic': 40, 'ended_monotonic': 60, 'seconds': 20}]}
        with self.assertRaises(ValueError): C.control_spans(base)
        base['probe_intervals'][0].update(started_monotonic=50, seconds=10)
        base['probe_intervals'].append({'qualifier': C.QUALIFIERS[1], 'started_monotonic': 59, 'ended_monotonic': 70, 'seconds': 11})
        with self.assertRaises(ValueError): C.control_spans(base)

    def test_probe_order_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            C.control_spans({'probe_intervals': [{'qualifier': C.QUALIFIERS[1]}]})


class PrimaryTests(unittest.TestCase):
    def setUp(self):
        self.source, self.boundary, self.artifact = d('source.py'), d('boundary.json'), d('attempt.json')
        self.value = {'schema': 'training_sync_qualification_attempt_v3', 'source': self.source,
            'qualification_execution_version': 3, 'boundary_proof': self.boundary,
            'condition': 'CSBrain-lastblock-finetune', 'seed': 22, 'fold': 5,
            'status': 'complete', 'formal_fits': 0, 'optimizer_updates': 0, 'test_forward_count': 0,
            'formal_activation_authorized': False, 'unflushed_hard_crash_tail_exact': None,
            'fixed_engineering_budget': deepcopy(C.FIXED), 'seconds': 9.5, 'peak_allocated_bytes': 123,
            'measurements': [{'synchronized_wall_seconds': 9999}],
            **{key + '_' + suffix: value for key, value in C.FIXED.items() for suffix in ('started', 'returned')}}

    def summary(self, value=None, qualifier=None):
        return C.primary_summary(self.value if value is None else value, self.artifact,
            qualifier=qualifier or C.QUALIFIERS[1], boundary=self.boundary, source=self.source, train_count=27, val_count=3)

    def test_success96_back_and12_stub_are_not_actual_steps(self):
        result = self.summary()
        self.assertEqual(96, result['observed_counters']['backward_returned'])
        self.assertEqual(0, result['observed_counters']['optimizer_returned'])
        self.assertEqual(12, result['optimizer_step_stub_returned'])
        self.assertEqual(9.5, result['seconds_observed']); self.assertFalse(result['nested_warmup_and_measurement_seconds_added'])

    def test_failure_partial_cost_retained_not_promoted_to_full_budget(self):
        self.value.update(status='failed', backward_started=9, backward_returned=8, peak_allocated_bytes_observed_at_failure=200)
        result = self.summary()
        self.assertEqual(8, result['observed_counters']['backward_returned']); self.assertEqual(200, result['peak_allocated_bytes'])
        self.assertIsNone(result['hard_crash_unflushed_tail_exact'])

    def test_unknown_elapsed_not_filled_by_overnight_finished_at(self):
        self.value.update(status='running', seconds=None, started_at='2026-10-09T20:00:00+08:00', finished_at='2026-10-10T10:00:00+08:00')
        self.assertIsNone(self.summary()['seconds_observed'])

    def test_real_zero_elapsed_kept(self):
        self.value.update(status='failed', seconds=0)
        self.assertEqual(0, self.summary()['seconds_observed'])

    def test_changed_budget_real_step_or_bool_count_rejected(self):
        for mutation in ({'backward_returned': 95}, {'optimizer_step_started': 1}, {'forward_returned': True},
                         {'qualification_execution_version': True}, {'unflushed_hard_crash_tail_exact': 0}, {'peak_allocated_bytes': True}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.summary({**self.value, **mutation})

    def test_wrong_boundary_source_or_activation_rejected(self):
        for mutation in ({'boundary_proof': d('other')}, {'source': d('old.py')}, {'formal_activation_authorized': True}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.summary({**self.value, **mutation})

    def test_diagnostic_count_scope_zero_backwards(self):
        value = {k: v for k, v in self.value.items() if not k.endswith(('_started', '_returned'))}
        value.update(schema='diagnostic_sync_qualification_attempt_v3', safe_stop_proof=self.boundary, backward_calls=0)
        value.update({key + '_' + suffix: count for key, count in {'forward': 180, 'encoder_forward': 1440, 'encoder_windows': 10800}.items()
                      for suffix in ('started', 'returned')})
        result = self.summary(value, C.QUALIFIERS[0]); self.assertEqual(0, result['observed_counters']['backward_returned'])
        value['encoder_windows_returned'] -= 60
        with self.assertRaises(ValueError): self.summary(value, C.QUALIFIERS[0])


class DeferralAndSamplingTests(unittest.TestCase):
    def test_deferred_zero_work_does_not_mean_zero_elapsed_or_qualified(self):
        value = {'source': d('guard.py'), 'control_registration': d('reg.json'), 'status': 'deferred_external_GPU_contexts',
            'observed_GPU_contexts': {'contexts': [{'pid': 5, 'type': 'G'}]}, 'GPU_speedup_claimed': False,
            **{k: 0 for k in ('model_constructions', 'GPU_forward_calls', 'backward_calls', 'optimizer_updates', 'formal_fits')}}
        result = C.verify_deferred(value, d('reg.json'), d('guard.py'))
        self.assertIsNone(result['seconds_observed']); self.assertFalse(result['performance_qualification_passed'])
        for change in ({'model_constructions': True}, {'GPU_forward_calls': 1}, {'observed_GPU_contexts': {'contexts': []}}):
            with self.subTest(change=change), self.assertRaises(ValueError): C.verify_deferred({**value, **change}, d('reg.json'), d('guard.py'))

    def monitor(self):
        return {'continuous_exclusivity_verified': False, 'between_sample_exclusivity': 'unknown',
            'samples': [{'started_monotonic': 1, 'returned_monotonic': 2, 'external_contexts': [],
                         'allowed_current_qualifier_pids': [42], 'observation': {'contexts': [{'pid': '42'}]}}],
            'post_probe_GPU_observation': {'contexts': []}, 'observation_errors': [], 'observer_query_seconds_observed': 1,
            'external_context_observed': False, 'sampled_exclusivity_passed': True, 'performance_contamination_observed_or_unknown': False}

    def test_clean_sampling_never_promoted_to_continuous_clean(self):
        result = C.sampled_performance(self.monitor())
        self.assertTrue(result['sampled_clean']); self.assertFalse(result['continuous_exclusivity_verified'])

    def test_external_context_invalidates_performance_even_with_success_report(self):
        value = self.monitor(); value['samples'][0]['external_contexts'] = [{'pid': 5}]
        value['samples'][0]['observation']['contexts'].append({'pid': 5})
        value.update(external_context_observed=True, sampled_exclusivity_passed=False, performance_contamination_observed_or_unknown=True)
        self.assertFalse(C.sampled_performance(value)['sampled_clean'])

    def test_observation_error_or_no_samples_is_unknown(self):
        for mutation in ({'samples': []}, {'observation_errors': [{'error': 'query failed'}]}):
            value = self.monitor(); value.update(**mutation, sampled_exclusivity_passed=False, performance_contamination_observed_or_unknown=True)
            self.assertFalse(C.sampled_performance(value)['sampled_clean'])

    def test_forged_continuous_clean_and_hidden_external_rejected(self):
        value = self.monitor(); value['continuous_exclusivity_verified'] = True
        with self.assertRaises(ValueError): C.sampled_performance(value)
        value = self.monitor(); value['post_probe_GPU_observation']['contexts'] = [{}]
        with self.assertRaises(ValueError): C.sampled_performance(value)

    def test_external_classification_must_match_raw_process_observation(self):
        value = self.monitor(); value['samples'][0]['observation']['contexts'].append({'pid': '5'})
        with self.assertRaises(ValueError): C.sampled_performance(value)
        value = self.monitor(); value['samples'][0]['allowed_current_qualifier_pids'] = [True]
        with self.assertRaises(ValueError): C.sampled_performance(value)


class RetainedAndFinalTests(unittest.TestCase):
    def snapshots(self):
        control = {'roles': {'dispatcher': {'pid': 5}, 'launcher': {'pid': 6}}, 'registration': d('q5.json'),
                   'recovery_registration': d('recovery.json'), 'recovery_closure': d('closure.json'),
                   'queue_attempt': d('attempt/status.json')}
        common = {'execution_registration': control['registration'], 'execution_version': 5,
                  'recovery_registration': control['recovery_registration'], 'recovery_closure': control['recovery_closure']}
        owner = {**common, 'pid': 5, 'child_pid': 6, 'child_job': 'CSBrain-lastblock-finetune-fit', 'linux_owner': 'attempt/linux_owner.json'}
        attempt = {**common, 'status': 'running', 'child_pid': 6, 'registration': control['registration'],
                   'job': {'id': owner['child_job'], 'module': 'finetune_v3'}}
        return owner, attempt, control

    def test_retained_snapshot_keeps_unknown_parent_elapsed_without_current_owner(self):
        value = C.verify_retained_control(*self.snapshots())
        self.assertIsNone(value['recorded_snapshot_seconds']); self.assertTrue(value['terminal_parent_seconds_not_read_during_companion'])

    def test_retained_wrong_child_route_or_lineage_rejected(self):
        for change in ({'child_pid': 7}, {'status': 'complete'}, {'execution_version': True}, {'recovery_registration': d('other.json')}):
            owner, attempt, control = self.snapshots()
            with self.subTest(change=change), self.assertRaises(ValueError): C.verify_retained_control(owner, {**attempt, **change}, control)
        owner, attempt, control = self.snapshots(); owner['linux_owner'] = 'other/linux_owner.json'
        with self.assertRaises(ValueError): C.verify_retained_control(owner, attempt, control)

    def final_shape(self):
        ids = ['fixture_' + str(i) for i in range(360)]
        registration = {'fits': [{'fit_id': i} for i in ids]}
        cost = {'schema_version': 6, 'formal_fits_expected': 360, 'formal_status_counts': {'training_complete': 360},
                'input_read_errors': [], 'formal_fits': [{'fit_id': i, 'formal_durable_updates': 4} for i in ids],
                'formal_durable_updates_known_subtotal': 1440,
                'source_bound_cpu_checkpoint_verification': {'passed': True, 'verified_fits': 135,
                    'prior_actual_cpu_checkpoint_loads': 405, 'new_cpu_checkpoint_loads': 0},
                'actual_completed_report_verifier_called': True, 'existing_outer_prediction_artifacts_read': True,
                **{key: 0 for key in ('gpu_calls_by_finalizer', 'model_constructions_by_finalizer', 'optimizer_updates_by_finalizer', 'new_predictions_computed')}}
        acceptance = {'schema_version': 6, 'formal_identity_count': 360, 'completed_training_json_verified': 360,
            'deep_cpu_checkpoint_scope_verified': 360, 'status_counts': {'training_complete': 360}, 'verification_errors': [], 'incomplete_fits': [],
            'training_rows': [{'fit_id': i, 'updates': 4} for i in ids], 'cpu_checkpoint_proofs': [{} for _ in ids],
            'requirements': [{'id': 'R%02d' % i, 'status': 'verified_complete'} for i in range(1, 31)],
            **{key: {'passed': True} for key in ('evaluation', 'semantic_review', 'final_cost', 'q5_execution_lifecycle')}}
        return cost, acceptance, registration

    def test_full_shape_only_does_not_execute_or_create_complete_certificate(self):
        value = C.verify_final_shape(*self.final_shape())
        self.assertEqual(405, value['existing_finetune_CPU_load_proof']); self.assertFalse(value['scientific_checks_reexecuted'])

    def test_final_markers_alone_never_accept_partial_or_duplicate_identity(self):
        for key, change in [('formal_fits', [{'fit_id': 'fixture_0', 'formal_durable_updates': 4}] * 360),
                            ('input_read_errors', ['unread']), ('formal_status_counts', {'training_complete': 359}),
                            ('all_plan_wall_seconds_exact', 0)]:
            cost, acceptance, registration = self.final_shape(); cost[key] = change
            with self.subTest(key=key), self.assertRaises(ValueError): C.verify_final_shape(cost, acceptance, registration)

    def test_final_missing_required_audit_cpu_or_evaluation_rejected(self):
        for target in ('requirements', 'semantic_review', 'evaluation', 'cpu'):
            cost, acceptance, registration = self.final_shape()
            if target == 'requirements': acceptance[target][-1]['status'] = 'pending'
            elif target == 'cpu': cost['source_bound_cpu_checkpoint_verification']['prior_actual_cpu_checkpoint_loads'] = 402
            else: acceptance[target]['passed'] = False
            with self.subTest(target=target), self.assertRaises(ValueError): C.verify_final_shape(cost, acceptance, registration)

    def test_final_parent_timer_remains_overlapping_supplemental(self):
        from types import SimpleNamespace
        owner, retained, control = self.snapshots()
        terminal = {**retained, 'status': 'complete', 'returncode': 0, 'seconds': 1200}
        snapshot = d('snapshot.json'); artifact = d('attempt/status.json')
        reader = SimpleNamespace(read=lambda binding: retained if binding == snapshot else terminal)
        companion = {'parent_attempt_timer': {'original_attempt': artifact},
                     'retained_pre_resume_control_snapshots': {'queue_attempt_snapshot': snapshot}}
        row = {'artifact': artifact, 'category': 'dispatcher_elapsed', 'aggregation_role': 'supplemental_nested_or_unmeasured', 'seconds_observed': 1200}
        cost = {'cost_records': [row]}
        result = C.final_parent_timer(reader, cost, companion)
        self.assertEqual(1200, result['seconds_observed_including_hold_and_probe']); self.assertIsNone(result['pure_training_seconds'])
        row['aggregation_role'] = 'primary_observed_invocation'
        with self.assertRaises(ValueError): C.final_parent_timer(reader, cost, companion)


class IOAndPendingTests(unittest.TestCase):
    def test_reader_never_opens_tensor_or_reuses_changed_json(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix='sync_companion_') as temporary:
            root = Path(temporary).resolve(); reader = C.Reader(root)
            with self.assertRaises(ValueError): reader.descriptor(root / 'never_open.pt')
            path = root / 'proof.json'; path.write_text('{}', encoding='utf8'); binding = reader.descriptor(path)
            path.write_text('{"changed":true}', encoding='utf8')
            with self.assertRaises(ValueError): reader.read(binding)
            self.assertEqual(root.parent, REPORT.resolve())

    def test_missing_terminal_pending_without_guardian_or_process_import(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix='sync_companion_') as temporary, \
             patch.object(C, 'ROOT', Path(temporary)), patch.object(C, 'load_normalizer', side_effect=AssertionError('No primary cost read')):
            value = C.collect()
        self.assertEqual('pending', value['status']); self.assertFalse(value['passed'])
        self.assertEqual(0, value['GPU_operations']); self.assertEqual(0, value['checkpoint_reads'])

    def test_bundle_requires_three_actual_final_artifacts(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix='sync_companion_') as temporary:
            path = Path(temporary) / 'pending.json'; path.write_text(json.dumps({'passed': False}), encoding='utf8')
            with self.assertRaises(ValueError): C.build_bundle(path, path, path)


if __name__ == '__main__': unittest.main(verbosity=2)
