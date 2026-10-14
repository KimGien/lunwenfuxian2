"""Small stdlib fixtures for recovery-aware source/version routing; no wide audit."""
from copy import deepcopy
import ast
import importlib.util
import io
import json
from pathlib import Path
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wide6_tests_target', ROOT / 'audit_full_plan_acceptance_v6.py')
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def binding(name):
    return {'path': 'synthetic/' + name, 'sha256': 'a' * 64}


class RecoveryWideTests(unittest.TestCase):
    def setUp(self):
        self.lineage = {'registration': binding('recovery.json'), 'source': binding('entry.py'), 'value': {'sources': []}}
        self.terminal = {'recovery_registration': self.lineage['registration'],
                         'recovery_execution_source': self.lineage['source'], 'recovery_closure': binding('closure.json')}
        self.closure = {'original_attempt': binding('attempt.json'), 'condition': 'CSBrain-lastblock-finetune', 'seed': 33, 'fold': 4}
        self.proof = {'passed': True, 'closure': self.terminal['recovery_closure'], 'recovery_registration': self.lineage['registration']}

    def verify(self):
        with patch.object(W.L, 'verify_recovery_registration', return_value=self.lineage), \
                patch.object(W.L, 'read_json', side_effect=lambda b: self.closure if b == self.terminal.get('recovery_closure') else {'status': 'running'}), \
                patch.object(W.L, 'verify_lost_f3_attempt', return_value=self.proof):
            return W.verify_recovery_terminal(self.terminal)

    def test_actual_recovery_three_field_proof(self):
        self.assertEqual(self.verify(), self.terminal)

    def test_wrong_recovery_registration_or_entry_rejected(self):
        for key in ('recovery_registration', 'recovery_execution_source'):
            old = self.terminal[key]; self.terminal[key] = binding('wrong.json')
            with self.subTest(key=key), self.assertRaises(ValueError): self.verify()
            self.terminal[key] = old

    def test_missing_or_unverified_closure_rejected(self):
        self.proof['passed'] = False
        with self.assertRaises(ValueError): self.verify()
        self.proof['passed'] = True; self.proof['closure'] = binding('wrong.json')
        with self.assertRaises(ValueError): self.verify()

    def test_original_check_fit_routes_actual_cpu5_only(self):
        constants = W.D.check_fit.__code__.co_consts
        self.assertIn('audit_completed_finetune_checkpoints_v5.py', constants)
        self.assertNotIn('audit_completed_finetune_checkpoints_v4.py', constants)
        self.assertIn('trace_inventory', W.D.check_fit.__code__.co_names)
        self.assertIn('trace_boundary', W.D.check_fit.__code__.co_names)
        self.assertIn('recheck_trace_inventory', W.D.check_fit.__code__.co_names)

    def test_original_final_cost_routes_real_schema6_and_source(self):
        constants = W._delegate_final_cost_audit.__code__.co_consts
        self.assertIn('cost_aggregation_v6.py', constants); self.assertNotIn('cost_aggregation_v5.py', constants)
        self.assertIn(6, constants); self.assertNotIn(5, constants)

    def test_original_math_cpu_sampler_and_selection_code_preserved(self):
        original = W.load_pinned(W.DELEGATE, W.DELEGATE_SHA, '_wide6_math_compare')
        for name in ('legacy_check_fit', 'deep_checkpoint', 'state_digest', 'policy_curve', 'cost_recovery_audit',
                     'verify_registered_identity', 'verify_exported_tables', 'semantic_review_audit'):
            a, b = getattr(original, name).__code__, getattr(W.D, name).__code__
            self.assertEqual(a.co_code, b.co_code, name); self.assertEqual(a.co_consts, b.co_consts, name)

    def cpu_fixture(self, schema=5, recovery=True):
        cpu = W.S.desc(ROOT / 'audit_completed_finetune_checkpoints_v5.py')
        self.lineage['value']['sources'] = [cpu]
        proof_path = W.QUEUE / 'pre_test_finetune_cpu_audit_fixture.json'
        proof_binding = {'path': proof_path.relative_to(W.WORK).as_posix(), 'sha256': 'a' * 64}
        gate = {'passed': True, 'checkpoint_loads': 405, 'test_forward_calls_this_check': 0,
                'audit': proof_binding, 'recovery_registration': self.lineage['registration'] if recovery else None}
        value = {'schema_version': schema, 'auditor_source': cpu,
                 'delegated_sources': {'lost_f3_closure_verifier': W.S.desc(Path(W.L.__file__)),
                     'previous_cpu_auditor_v4': W.S.desc(ROOT / 'audit_completed_finetune_checkpoints_v4.py')}}
        seen = []
        def shape(v, registration, registry):
            seen.append(deepcopy(v)); self.assertEqual(v['schema_version'], 4); return []
        original_desc = W.S.desc
        with patch.object(W.L, 'verify_recovery_registration', return_value=self.lineage), \
                patch.object(W.S, 'desc', side_effect=lambda p: binding('gate.json') if Path(p).name == 'pre_test_finetune_cpu_gate.json' else original_desc(p)), \
                patch.object(W, 'json_bound', side_effect=lambda b: value if b == proof_binding else gate), \
                patch.object(W, 'load_pinned', return_value=types.SimpleNamespace(verify_cpu_shape=shape)):
            result = W.pre_test_cpu_gate({}, {'sources': []})
        self.assertEqual(value['schema_version'], schema)
        self.assertEqual(result['auditor_source'], cpu)
        self.assertEqual(result['new_checkpoint_loads_by_this_lifecycle_check'], 0)
        self.assertEqual(len(seen), 1)
        return result

    def test_cpu5_shape_normalization_only_in_temporary_copy(self):
        self.cpu_fixture()

    def test_cpu4_or_missing_recovery_gate_cannot_masquerade(self):
        with self.assertRaises(ValueError): self.cpu_fixture(schema=4)
        with self.assertRaises(ValueError): self.cpu_fixture(recovery=False)

    def test_final_cost_requires_separate_recovery_proof_and_closure(self):
        lifecycle = {'q5_registration': binding('q5.json'), 'terminal_status': binding('status.json'),
                     'q5_source': binding('q5.py'), 'q4_registration': binding('q4.json'),
                     'recovery_execution_proof': self.terminal}
        base = {'registration': lifecycle['q5_registration'], 'terminal_status': lifecycle['terminal_status'],
                'source': lifecycle['q5_source'], 'execution_version': 5, 'previous_q4_registration': lifecycle['q4_registration']}
        data = {'completion_gate': {'q5_execution_proof': base, 'recovery_execution_proof': self.terminal},
                'lost_f3_closure_verification': {'passed_for_observed_closures': True, 'errors': []},
                'source_bound_lost_f3_attempts': [{'closure': self.terminal['recovery_closure'], 'aggregation_role': 'primary_observed_invocation'}]}
        with patch.object(W, '_delegate_final_cost_audit', return_value={'passed': True}), \
                patch.object(W, 'queue_terminal', return_value=lifecycle), patch.object(W.R, 'load_registration', return_value={}), \
                patch.object(W.S, 'read', return_value=data):
            self.assertEqual(W.final_cost_audit('synthetic', None, [])['recovery_execution_proof'], self.terminal)
            data['source_bound_lost_f3_attempts'] = []
            with self.assertRaises(ValueError): W.final_cost_audit('synthetic', None, [])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryWideTests)
    stream = io.StringIO(); start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue()); raise SystemExit(1)
    value = {'passed': True, 'tests_run': result.testsRun, 'failures': 0, 'errors': 0,
             'seconds': time.perf_counter() - start, 'source': W.S.desc(Path(W.__file__)), 'tests': W.S.desc(Path(__file__)),
             'output': stream.getvalue(), 'actual_wide_audit_executed': False, 'checkpoint_loads': 0,
             'models': 0, 'GPU_calls': 0, 'raw_feature_probability_reads': 0}
    path = ROOT / ('wide_v6_stdlib_tests_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf8') as f: json.dump(value, f, indent=2); f.write('\n')
    print(json.dumps({'passed': True, 'tests_run': result.testsRun, 'proof': W.S.desc(path)}))
