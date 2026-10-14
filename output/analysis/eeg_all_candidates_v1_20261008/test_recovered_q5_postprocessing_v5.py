"""CPU/std JSON fixtures for recovered Q5 final gates, never a final certificate."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPORT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('_recovery_test_' + name, REPORT / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


M = load('supplemental_head_resource_measurement_v5')
F = load('finalize_full_plan_cost_v5')
C = load('cost_aggregation_v6').load_v5().load_v4()


def d(name): return {'path': name, 'sha256': 'a' * 64}


class RecoveryGateTests(unittest.TestCase):
    def setUp(self):
        self.expected = {key: d(key + '.json') for key in ('recovery_registration', 'recovery_execution_source', 'recovery_closure')}
        self.lineage = {'registration': self.expected['recovery_registration'], 'source': self.expected['recovery_execution_source']}
        self.closure = {'original_attempt': d('original.json'), 'condition': 'C', 'seed': 33, 'fold': 4}
        self.proof = {'passed': True, 'closure': self.expected['recovery_closure'],
                      'recovery_registration': self.expected['recovery_registration'], 'closure_source': self.expected['recovery_execution_source']}

    def check(self, terminal=None, proof=None):
        with patch.object(M.L, 'verify_recovery_registration', return_value=self.lineage), \
             patch.object(M.L, 'read_json', return_value=self.closure), \
             patch.object(M.L, 'verify_lost_f3_attempt', return_value=self.proof if proof is None else proof):
            return M.verify_recovery_terminal(self.expected if terminal is None else terminal)

    def test_exact_recovery_terminal(self): self.assertEqual(self.expected, self.check())

    def test_original_q5_terminal_without_recovery_rejected(self):
        with self.assertRaises(ValueError): self.check({})

    def test_wrong_recovery_entry_or_closure_rejected(self):
        for key in self.expected:
            value = deepcopy(self.expected); value[key] = d('wrong.json')
            with self.subTest(key=key), self.assertRaises(ValueError): self.check(value)

    def test_bad_closure_does_not_prove_completed_recovery(self):
        for mutation in ({'passed': False}, {'closure': d('other')}, {'recovery_registration': d('old')}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.check(proof={**self.proof, **mutation})

    def test_closure_verifier_errors_propagate(self):
        with patch.object(M.L, 'verify_recovery_registration', side_effect=ValueError('source changed')):
            with self.assertRaisesRegex(ValueError, 'source changed'): M.verify_recovery_terminal(self.expected)

    def test_cpu5_exact135_shape_and_schema(self):
        reg = {'fits': [{'fit_id': str(i), 'stage': '3'} for i in range(135)]}; binding = d('registry.json')
        row = {'passed': True, 'cpu_checkpoint_audit': {'cpu_tensors_verified': True, 'checkpoint_loads': 3,
                'only_map_location_cpu': True, 'gpu_operations': 0, 'model_constructions': 0}}
        value = {'schema_version': 5, 'registration': binding, 'registered_total_fits': 360,
            'registered_finetune_fits': 135, 'training_complete_finetune_observed': 135,
            'verified_completed_finetune_fits': 135, 'logical_completed_fits_counted': 135, 'checkpoint_loads': 405,
            'status': 'all_finetune_verified', 'all_135_finetune_checkpoints_verified': True, 'partial_snapshot': False,
            'pending_fits': [], 'deferred_completed_fits': [], 'errors': [],
            'completed_rows': [{**deepcopy(row), 'fit_id': str(i)} for i in range(135)],
            'observed_auditor_operations_including_failed_checks': {'cpu_checkpoint_loads_started': 405,
                'cpu_checkpoint_loads_returned': 405, 'model_construction_attempts_blocked': 0, 'gpu_initialization_attempts_blocked': 0}}
        self.assertEqual(135, len(M.verify_cpu_shape(value, reg, binding)))
        for key, bad in [('schema_version', 4), ('checkpoint_loads', 402), ('verified_completed_finetune_fits', True),
                         ('partial_snapshot', True), ('deferred_completed_fits', [{}])]:
            with self.subTest(key=key), self.assertRaises(ValueError): M.verify_cpu_shape({**value, key: bad}, reg, binding)
        bad = deepcopy(value); bad['completed_rows'][1] = bad['completed_rows'][0]
        with self.assertRaises(ValueError): M.verify_cpu_shape(bad, reg, binding)

    def test_actual_measurement_math_still_exact_original_delegate(self):
        self.assertIs(M.measure, M.D.measure); self.assertEqual(str(M.V2_SOURCE), M.measure.__code__.co_filename)
        self.assertEqual(M.D.SOURCE, M.SOURCE)
        self.assertEqual(M.V4_SHA, hashlib.sha256(M.V4_SOURCE.read_bytes()).hexdigest())

    def test_resource_provenance_version5(self):
        value = {'subtype': 'supplemental_head_resource_measurement', 'measurement_scope': {'seconds': 'E2 prerequisite'}}
        with patch.object(M, 'write_new') as write:
            M.write_measurement(Path('fixture.json'), value)
        self.assertEqual(5, write.call_args.args[1]['resource_measurement_version'])
        self.assertEqual(M.delegated_source(), value['delegated_resource_measurement_source'])


class RecoveryCostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=REPORT, prefix='recovery_gate_')
        self.root = Path(self.temp.name).resolve(); self.out = self.root / 'out'
        self.closed = self.out / 'process_loss_closures/lost_f3_fixture.json'; self.closed.parent.mkdir(parents=True)
        self.closed.write_text('{}', encoding='utf8'); self.binding = F.desc(self.closed)
        self.cpu = self.root / 'cpu.json'
        self.cpu.write_text(json.dumps({'passed': True, 'checkpoint_loads': 2, 'seconds': 14.4206806,
            **{key: 0 for key in ('model_constructions', 'GPU_operations', 'raw_arrays_loaded', 'test_probability_reads', 'optimizer_updates', 'backward_calls')}}), encoding='utf8')
        self.recovery = {'recovery_registration': self.binding, 'recovery_execution_source': self.binding, 'recovery_closure': self.binding}
        self.closure = {'passed': True, 'closure': self.binding, 'recovery_registration': self.binding, 'closure_source': self.binding,
            'aggregation_role': 'primary_observed_invocation', 'artifact': d('original.json'),
            'known_extra_optimizer_returns_outside_checkpoint': 4, 'known_extra_backward_returns_outside_checkpoint': 27,
            'durable_epoch': 8, 'durable_optimizer_updates': 32, 'durable_backward_calls': 216,
            'unflushed_tail_unknown': True, 'unflushed_tail_counters': None, 'final_elapsed_seconds': None,
            'graceful_terminal_flush': False, 'extra_work_not_added_again': True, 'checkpoint_cpu_resume_proof': F.desc(self.cpu)}
        self.counts = dict(optimizer_returned=36, backward_returned=243, model_forward_returned=483,
                           encoder_forward_returned=2163, encoder_windows_returned=16344)
        self.snapshot = {'lost_f3_closure_verification': {'passed_for_observed_closures': True, 'errors': []},
            'source_bound_lost_f3_attempts': [self.closure], 'inputs': [], 'category_observations': {},
            'cost_records': [{'artifact': d('original.json'), 'category': 'formal_training_attempt',
                'aggregation_role': 'primary_observed_invocation', 'status': 'running', 'seconds_observed': None,
                'observed_counters': deepcopy(self.counts)}]}

    def tearDown(self):
        assert self.root.parent == REPORT.resolve() and self.root.name.startswith('recovery_gate_'); self.temp.cleanup()

    def bind(self):
        with patch.object(F, 'OUT', self.out): return F.bind_lost_f3_recovery_cost(self.snapshot, self.recovery, C)

    def test_physical_counters_unchanged_and_cpu_timer_independent(self):
        inventory = self.bind()
        self.assertEqual(self.counts, self.snapshot['cost_records'][0]['observed_counters'])
        self.assertEqual(2, len(inventory)); self.assertFalse(self.snapshot['cost_records'][1]['counts_added_again'])
        self.assertEqual(14.4206806, self.snapshot['category_observations']['recovery_cpu_preflight']['known_recorded_scope_seconds_subtotal'])
        self.assertEqual(0, self.snapshot['cost_records'][2]['observed_counters']['optimizer_returned'])

    def test_graceful_or_zero_tail_or_changed_delta_rejected(self):
        for key, value in [('graceful_terminal_flush', True), ('unflushed_tail_counters', {}), ('final_elapsed_seconds', 0),
                           ('known_extra_optimizer_returns_outside_checkpoint', 0), ('known_extra_backward_returns_outside_checkpoint', True)]:
            old = self.closure[key]; self.closure[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.bind()
            self.closure[key] = old

    def test_duplicate_primary_attempt_rejected(self):
        self.snapshot['cost_records'].append(deepcopy(self.snapshot['cost_records'][0]))
        with self.assertRaises(ValueError): self.bind()

    def test_unaccounted_closure_rejected(self):
        (self.closed.parent / 'lost_f3_other.json').write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'omitted or ambiguous'): self.bind()

    def test_cpu_preflight_never_double_added(self):
        self.snapshot['cost_records'].append({'artifact': F.desc(self.cpu)})
        with self.assertRaisesRegex(ValueError, 'already classified'): self.bind()

    def test_incomplete_training_pending_before_recovery_e3_or_cost(self):
        fits = [{'fit_id': f'C{c}/seed_{s}/fold_{f}', 'condition': f'C{c}',
                 'objective': 'self_supervised' if c == 23 else 'supervised'}
                for c in range(24) for s in (11, 22, 33) for f in range(1, 6)]
        (self.out / 'registration.json').write_text(json.dumps({'fits': fits}), encoding='utf8')
        with patch.object(F, 'OUT', self.out), patch.object(F, 'verify_existing_evaluation', side_effect=AssertionError('E3 forbidden')), \
             patch.object(F, 'collect_cost', side_effect=AssertionError('collect forbidden')), \
             patch.object(F, 'recovery_execution_proof', side_effect=AssertionError('recovery should not be read')):
            result = F.finalize()
        self.assertEqual('pending', result['status']); self.assertFalse(result['outer_prediction_artifacts_read'])

    def test_math_and_old_evidence_helpers_unmodified(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef)}
        old = functions(REPORT / 'finalize_full_plan_cost_v4.py'); new = functions(REPORT / 'finalize_full_plan_cost_v5.py')
        for name in ('measurement_plan', 'head_resource_gate', 'bind_mamba_recheck_evidence', 'append_optional_qualification_v2_cost',
                     'append_boundary_control_evidence', 'verify_phase_coverage', 'render_report', 'verify_existing_evaluation'):
            self.assertEqual(old[name], new[name], name)


if __name__ == '__main__': unittest.main(verbosity=2)
