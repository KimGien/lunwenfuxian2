"""Execution-only routing, real Q5 provenance, and native CPU gate checks."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import continuation_queue as Q1, continuation_queue_v4 as Q4, continuation_queue_v5 as Q5
from . import linux_entry_v4 as L4


class Q5Tests(unittest.TestCase):
    def test_exact36_jobs195fits(self):
        self.assertEqual(len(Q5.jobs()), 36)
        self.assertEqual(sum(j.get('fits', 0) for j in Q5.jobs()), 195)
        for before, after in zip(Q4.jobs(), Q5.jobs()):
            for field in ('id', 'args', 'dependencies', 'conditions', 'fits', 'linux'):
                self.assertEqual(before.get(field), after.get(field))

    def test_only_two_module_routings_change(self):
        mapping = {'acceleration_prepare_v1': 'acceleration_prepare_v2', 'linux_entry_v3': 'linux_entry_v4'}
        for old, new in zip(Q4.jobs(), Q5.jobs()):
            self.assertEqual(new['module'], mapping.get(old['module'], old['module']))

    def test_terminal_and_owner_are_truthful_Q5(self):
        binding = {'path': 'q5/reg.json', 'sha256': 'a' * 64}
        for name in ('status.json', 'owner.json'):
            with patch.object(Q5.S, 'desc', return_value=binding), patch.object(Q5.S, 'save') as save:
                Q5.ActualExecutionStorage().save(Q5.ROOT / name, {'status': 'running'})
            self.assertEqual(save.call_args.args[1]['execution_version'], 5)
            self.assertEqual(save.call_args.args[1]['source'], binding)
            self.assertEqual(save.call_args.args[1]['execution_registration'], binding)

    def test_storage_does_not_relabel_old_Q4(self):
        value = {'status': 'running'}
        with patch.object(Q5.S, 'save') as save:
            Q5.ActualExecutionStorage().save(Q4.ROOT / 'status.json', value)
        self.assertIs(save.call_args.args[1], value)

    def test_formal_F3_command_unchanged_and_bound(self):
        driver = Q5.isolated_driver()
        job = next(j for j in Q5.jobs() if j['id'] == 'REVE-lastblock-finetune-fit')
        binding = {'path': 'original-sidecar.json', 'sha256': 'b' * 64}
        with patch.object(Q5.S, 'desc', return_value=binding): command = driver.command(job, resume=True)
        self.assertIn('tridep_ccml.eeg_all_candidates.finetune_v3', command)
        self.assertIn('--resume', command)
        self.assertEqual(command[command.index('--acceleration-registration') + 1], binding['path'])

    def test_completed_condition_reused_without_launch(self):
        driver = Q5.isolated_driver()
        job = next(j for j in Q5.jobs() if j['id'] == 'CSBrain-lastblock-finetune-fit')
        with patch.object(Q1, 'all_complete', return_value=True), patch.object(Q5.S, 'save') as save, patch.object(Q5.S, 'desc', return_value={}):
            self.assertEqual(driver.run_child(job, {}, resume=True), 'complete')
        self.assertEqual(save.call_args.args[1]['new_model_calls'], 0)

    def test_old_engineering_pass_is_rechecked_before_reuse(self):
        job = next(j for j in Q5.jobs() if j['id'] == 'mamba-check')
        binding = {'path': 'q4/reg.json', 'sha256': 'c' * 64}
        value = {'job': job, 'status': 'complete', 'returncode': 0, 'registration': binding, 'log': binding}
        with patch.object(Path, 'glob', return_value=[Path('attempt/status.json')]), patch.object(Q5.S, 'read', return_value=value), patch.object(Q5.S, 'desc', return_value=binding), patch.object(Q5.S, 'checked'), self.assertRaises(ValueError):
            Q5.previous_completed_engineering(job, SimpleNamespace(postcondition=lambda job: False))

    def test_absent_old_engineering_is_not_invented(self):
        job = next(j for j in Q5.jobs() if j['id'] == 'mamba-check')
        with patch.object(Path, 'glob', return_value=[]):
            self.assertIsNone(Q5.previous_completed_engineering(job, SimpleNamespace(postcondition=lambda job: True)))

    def test_real_cpu_gate_uses_Q5_directory_before_evaluation(self):
        driver = Q5.isolated_driver()
        result = {'all_135_finetune_checkpoints_verified': True, 'errors': [], 'deferred_completed_fits': [], 'checkpoint_loads': 405}
        module = SimpleNamespace(audit=lambda: result)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        with patch.object(Q1, 'all_complete', return_value=True), patch.object(Q5.importlib.util, 'spec_from_file_location', return_value=spec), patch.object(Q5.importlib.util, 'module_from_spec', return_value=module), patch.object(Q5.S, 'save') as save, patch.object(Q5.S, 'desc', return_value={}), patch.object(Q5.S, 'read', return_value={'sources': []}), patch.object(Q5.R, 'assert_test_unlocked', side_effect=ValueError('blocked synthetic evaluation')):
            self.assertFalse(driver.evaluate_tail({}))
        targets = [Path(call.args[0]) for call in save.call_args_list]
        self.assertIn(Q5.ROOT / 'pre_test_finetune_cpu_gate.json', targets)
        self.assertTrue(all(path.is_relative_to(Q5.ROOT) for path in targets))
        native_gate = next(c.args[1] for c in save.call_args_list if Path(c.args[0]).name == 'pre_test_finetune_cpu_gate.json')
        self.assertEqual(native_gate['execution_version'], 5)

    def test_failed_cpu_gate_never_delegates_to_test(self):
        driver = Q5.isolated_driver()
        result = {'all_135_finetune_checkpoints_verified': False, 'errors': [], 'deferred_completed_fits': [], 'checkpoint_loads': 402}
        module = SimpleNamespace(audit=lambda: result)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        with patch.object(Q1, 'all_complete', return_value=True), patch.object(Q5.importlib.util, 'spec_from_file_location', return_value=spec), patch.object(Q5.importlib.util, 'module_from_spec', return_value=module), patch.object(Q5.S, 'save'), patch.object(Q5.R, 'assert_test_unlocked') as gate, self.assertRaises(ValueError):
            driver.evaluate_tail({})
        gate.assert_not_called()

    def test_linux_formal_delegates_qualified_old_F3(self):
        binding = {'path': 'formal-sidecar.json', 'sha256': 'd' * 64}
        with patch.object(L4.L3, 'execute', return_value='old_F3') as execute:
            self.assertEqual(L4.execute('mamba-finetune', resume=True, owner_record='owner.json', acceleration_registration=binding), 'old_F3')
        self.assertEqual(execute.call_args.kwargs['acceleration_registration'], binding)

    def test_false_handoff_rejected(self):
        with patch.object(Q5.S, 'read', return_value={'passed': False, 'sources': []}), self.assertRaises(ValueError):
            Q5.require_handoff()

    def handoff_fixture(self):
        binding = {'path': 'fixture.json', 'sha256': 'e' * 64}
        oldjob = next(j for j in Q4.jobs() if j['id'] == 'CSBrain-lastblock-finetune-fit')
        owner = {'id': '3266366ec77046d09bbd58372bd9bee5', 'pid': 40840, 'child_pid': 18224, 'child_job': oldjob['id']}
        attempt = {'status': 'running', 'child_pid': 18224, 'registration': binding, 'job': oldjob}
        proof = {'passed': True, 'version': 5, 'q5_registration': binding, 'old_q4_registration': binding,
                 'old_dispatcher_terminated_after_completed_children': True, 'launcher_exit_code': 0,
                 'gpu_child_exit_code': 0, 'completed_CSBrain_fits_verified': 15,
                 'GPU_training_child_suspend_or_terminate_calls': 0, 'protected_sources': [binding, binding],
                 'completed_CSBrain_artifacts': [binding] * 90, 'source': binding,
                 'old_owner_snapshot': {'path': 'old_owner.json', 'sha256': 'e' * 64},
                 'old_attempt_snapshot': {'path': 'old_attempt.json', 'sha256': 'e' * 64},
                 'old_dispatcher_exit_code': 0xEED80008,
                 'held_process_identities': {name: {'pid': p, 'parent_pid': parent, 'creation_100ns': birth}
                      for name, p, parent, birth in [('dispatcher', 40840, 15428, '134360214625265856'),
                          ('launcher', 18224, 40840, '134360215362961641'), ('gpu_child', 32880, 18224, '134360215363064824')]}}
        def read(path):
            if Path(path).name == 'handoff_complete.json': return proof
            if Path(path).name == 'old_owner.json': return owner
            if Path(path).name == 'old_attempt.json': return attempt
            return {'sources': [binding]}
        return proof, binding, read

    def run_handoff_fixture(self, proof, binding, read):
        with patch.object(Q5.S, 'read', side_effect=read), patch.object(Q5.S, 'desc', return_value=binding), patch.object(Q5.S, 'checked', side_effect=lambda r: Path(r['path'])), patch.object(Q5.R, 'fit_status', return_value={'evidence': {name: binding for name in ('checkpoint', 'train_history', 'validation_history', 'metadata')}}), patch.object(Q5.R, '_validate_complete'), patch.object(Path, 'exists', return_value=False), patch.object(Q1, 'all_complete', return_value=True):
            return Q5.require_handoff()

    def test_exact90_artifact_handoff_shape(self):
        proof, binding, read = self.handoff_fixture()
        self.assertIs(self.run_handoff_fixture(proof, binding, read), proof)

    def test_short_completed_artifact_list_rejected(self):
        proof, binding, read = self.handoff_fixture()
        proof['completed_CSBrain_artifacts'] = [binding]
        with self.assertRaises(ValueError): self.run_handoff_fixture(proof, binding, read)

    def test_wrong_parent_exit_and_process_creation_rejected(self):
        proof, binding, read = self.handoff_fixture()
        proof['old_dispatcher_exit_code'] = 0
        with self.assertRaises(ValueError): self.run_handoff_fixture(proof, binding, read)
        proof['old_dispatcher_exit_code'] = 0xEED80008
        proof['held_process_identities']['gpu_child']['creation_100ns'] = 'reused_PID'
        with self.assertRaises(ValueError): self.run_handoff_fixture(proof, binding, read)


if __name__ == '__main__': unittest.main()
