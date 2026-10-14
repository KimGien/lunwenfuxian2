"""Execution-only routing and fail-closed handoff regression checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from . import continuation_queue as Q1, continuation_queue_v3 as Q3, continuation_queue_v4 as Q4
from . import evaluate_v3 as E3


class RoutingTests(unittest.TestCase):
    def test_exact_formal_identity_and_total(self):
        old = {job['id']: job for job in Q3.jobs() if job.get('fits')}
        new = {job['id']: job for job in Q4.jobs() if job.get('fits')}
        self.assertEqual(set(old), set(new))
        self.assertEqual(sum(job['fits'] for job in new.values()), 195)
        for key in old:
            for field in ('id', 'fits', 'conditions', 'args', 'linux'):
                self.assertEqual(old[key].get(field), new[key].get(field))

    def test_each_finetune_has_own_engineering_dependency(self):
        jobs = Q4.jobs()
        seen = set()
        for job in jobs:
            self.assertTrue(set(job['dependencies']) <= seen)
            seen.add(job['id'])
            if job.get('fits') and job['conditions'][0] in Q1.FINETUNE:
                self.assertEqual(job['dependencies'], [job['conditions'][0] + '-acceleration-prepare'])
        self.assertEqual(sum(job['id'].endswith('-acceleration-prepare') for job in jobs), 9)

    def test_unchanged_non_finetune_jobs(self):
        new = {job['id']: job for job in Q4.jobs()}
        for job in Q3.jobs():
            if not job.get('fits') or job['conditions'][0] not in Q1.FINETUNE:
                self.assertEqual(new[job['id']], job)

    def test_missing_safe_handoff_fails_before_driver(self):
        with patch.object(Q4.S, 'read', return_value={'passed': False}), self.assertRaises(ValueError):
            Q4.require_handoff()

    def test_nonexistent_previous_owner_proof_not_enough(self):
        value = {'passed': True, 'all_training_processes_stopped': True, 'gpu_idle': False}
        with patch.object(Q4.S, 'read', return_value=value), self.assertRaises(ValueError):
            Q4.require_handoff()

    def test_accelerated_command_has_exact_bound_registration(self):
        driver = Q4.isolated_driver()
        job = next(j for j in Q4.jobs() if j['id'] == 'CSBrain-lastblock-finetune-fit')
        binding = {'path': 'registered/condition.json', 'sha256': 'a' * 64}
        with patch.object(Q4.S, 'desc', return_value=binding):
            command = driver.command(job, resume=True)
        self.assertIn('tridep_ccml.eeg_all_candidates.finetune_v3', command)
        self.assertEqual(command[command.index('--acceleration-registration') + 1], binding['path'])
        self.assertEqual(command[command.index('--acceleration-registration-sha256') + 1], binding['sha256'])
        self.assertEqual(command[-1], '--resume')

    def test_evaluation_tail_routes_v3(self):
        command = Q4.isolated_driver().command({'module': 'evaluate', 'args': [], 'conditions': []})
        self.assertIn('tridep_ccml.eeg_all_candidates.evaluate_v3', command)

    def test_evaluator_uses_same_math_runner_and_new_verifier(self):
        value = E3.execution_provenance('CSBrain-lastblock-finetune')
        self.assertEqual(value['version'], 3)
        self.assertEqual(value['finetune_runner_source'], E3.E2.execution_provenance()['finetune_runner_source'])
        self.assertTrue(value['diagnostic_acceleration_does_not_change_test_window_chunk'])
        E3.verify_execution(value, 'CSBrain-lastblock-finetune')

    def test_false_execution_rejected(self):
        value = E3.execution_provenance()
        value['version'] = 2
        with self.assertRaises(ValueError):
            E3.verify_execution(value)

    def test_metadata_verification_is_complete_not_only_extra_field(self):
        metadata = {'execution_acceleration': {}, 'condition': 'CSBrain-lastblock-finetune', 'seed': 22, 'fold': 4}
        with patch.object(E3.A, 'verify_execution_acceleration', return_value={'passed': True}) as verify:
            E3.verify_training_execution(metadata)
        self.assertIs(verify.call_args.args[0], metadata)

    def test_claimed_ready_without_actual_safe_stop_rejected(self):
        value = {'passed': True, 'all_training_processes_stopped': True, 'gpu_idle': True,
                 'q4_registration': {'path': 'q4.json', 'sha256': 'a' * 64},
                 'protected_sources': [], 'checkpoint_evidence': []}
        with patch.object(Q4.S, 'read', return_value=value), patch.object(Q4.S, 'desc', return_value=value['q4_registration']), self.assertRaises(ValueError):
            Q4.require_handoff()

    def test_completed_fits_reused_before_command_or_new_qualification(self):
        driver = Q4.isolated_driver()
        job = next(j for j in Q4.jobs() if j['id'] == 'EEGPT-lastblock-finetune-acceleration-prepare')
        with patch.object(Q1, 'all_complete', return_value=True), patch.object(Q4.S, 'save') as save, patch.object(Q4.S, 'desc', return_value={}):
            self.assertEqual(driver.run_child(job, {}, resume=True), 'complete')
        self.assertEqual(save.call_args.args[1]['new_model_calls'], 0)
        self.assertTrue(save.call_args.args[1]['qualification_for_future_work_skipped_because_all_fits_complete'])

    def test_failed_full_cpu_audit_prevents_evaluation_tail(self):
        driver = Q4.isolated_driver()
        audit = {'all_135_finetune_checkpoints_verified': False, 'errors': [], 'deferred_completed_fits': [], 'checkpoint_loads': 402}
        module = SimpleNamespace(audit=lambda: audit)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        with patch.object(Q1, 'all_complete', return_value=True), patch.object(Q4.importlib.util, 'spec_from_file_location', return_value=spec), patch.object(Q4.importlib.util, 'module_from_spec', return_value=module), patch.object(Q4.S, 'save'), self.assertRaises(ValueError):
            driver.evaluate_tail({})


if __name__ == '__main__':
    unittest.main()
