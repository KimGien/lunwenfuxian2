"""Pure stdlib Q5/E3/CPU-proof gate tests; synthetic fixtures are not certificates."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('supplemental_head_resource_measurement_v4.py')
spec = importlib.util.spec_from_file_location('supplemental_v4_under_test', SOURCE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.fixture_root = M.WORK / '.sht'; self.fixture_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.fixture_root, prefix='v4_')
        self.work = Path(self.temp.name); self.out = self.work / M.OUT_REL
        self.fits = [{'fit_id': f'c_{c}/seed_{seed}/fold_{fold}', 'condition': f'c_{c}', 'seed': seed, 'fold': fold,
                      'stage': '3' if c < 9 else '1'} for c in range(24) for seed in (11, 22, 33) for fold in range(1, 6)]
        self.registration = {'formal_fits': 360, 'fits': self.fits}
        self.queue = {'status': 'all_training_complete', 'test_evaluation_completed': True,
                      'test_evaluation_frozen': False, 'outcomes': {'job': 'complete'}}
        self.patchers = []

    def tearDown(self):
        for p in reversed(self.patchers): p.stop()
        target = Path(self.temp.name).resolve()
        assert target.is_relative_to(self.fixture_root.resolve()) and target != self.fixture_root.resolve()
        assert target.is_relative_to(M.WORK.resolve())
        self.temp.cleanup()

    def write(self, path, value):
        M.write_new(path, value)
        return M.descriptor(path, self.work)

    def fixture(self, complete=False):
        reg = self.write(self.out / 'registration.json', self.registration)
        self.write(self.out / 'continuation_queue_v5/status.json', self.queue)
        if not complete: return
        for row in self.fits:
            self.write(self.out / 'runs' / row['fit_id'] / 'status.json', {'status': 'training_complete'})
        sources = []
        for n in range(47):
            p = self.work / f'source_{n}.py'; p.write_text(f'# fixture source {n}\n', encoding='utf-8')
            sources.append(M.descriptor(p, self.work))
        q3_binding = self.write(self.out / 'continuation_queue_v3/registration.json',
                               {'sources': sources, 'registry': reg, 'jobs': [{'id': 'old-job'}]})
        p = patch.object(M, 'Q3_SHA', q3_binding['sha256']); p.start(); self.patchers.append(p)
        sources.append(q3_binding)
        for name in ('finetune_v3.py', 'evaluate_v3.py', 'execution_acceleration_v1.py', 'validation_batching_v1.py'):
            file = self.work / 'tridep_ccml/eeg_all_candidates' / name
            file.parent.mkdir(parents=True, exist_ok=True); file.write_text('# fixture source\n', encoding='utf-8')
            sources.append(M.descriptor(file, self.work))
            if name == 'evaluate_v3.py': self.evaluator_source = sources[-1]
        auditor = self.work / M.REPORT_REL / 'audit_completed_finetune_checkpoints_v4.py'
        auditor.parent.mkdir(parents=True, exist_ok=True); auditor.write_text('# fixture auditor\n', encoding='utf-8')
        self.auditor_source=M.descriptor(auditor,self.work);sources.append(self.auditor_source)
        while len(sources)<59:
            file=self.work/f'q4_extra_{len(sources)}.py';file.write_text('# fixture\n',encoding='utf-8')
            sources.append(M.descriptor(file,self.work))
        jobs=[{'id':'job','fits':195,'conditions':['fixture'],'args':['--unchanged'],'linux':False,'module':'acceleration_prepare_v1'}]+[{'id':'job_'+str(i),'module':'unchanged','dependencies':['job'],'args':[]} for i in range(35)]
        self.queue['outcomes']={row['id']:'complete' for row in jobs}
        self.q4={'version':4,'registry':reg,'previous_execution_registration':q3_binding,'sources':list(sources),'jobs':jobs}
        q4_binding=self.write(self.out/'continuation_queue_v4/registration.json',self.q4)
        p=patch.object(M,'Q4_SHA',q4_binding['sha256']);p.start();self.patchers.append(p)
        sources.append(q4_binding)
        for name in ('continuation_queue_v5.py','acceleration_prepare_v2.py','linux_entry_v4.py'):
            file=self.work/'tridep_ccml/eeg_all_candidates'/name;file.write_text('# fixture '+name+'\n',encoding='utf-8')
            sources.append(M.descriptor(file,self.work))
            if name=='continuation_queue_v5.py':
                self.q5_source=sources[-1]
                p=patch.object(M,'Q5_SOURCE_SHA',self.q5_source['sha256']);p.start();self.patchers.append(p)
        self.q5={'version':5,'registry':reg,'previous_execution_registration':q4_binding,'sources':sources,
                 'jobs':[{**jobs[0],'module':'acceleration_prepare_v2'}]+jobs[1:],'one_gpu_sequential':True,'test_gate_unchanged':True,
                 'training_protocol_matrix_or_hyperparameters_changed':False,'formal_fits_in_reused_matrix':195,
                 'training_micro_subjects':1,'training_window_chunk':8}
        q5_binding=self.write(self.out/'continuation_queue_v5/registration.json',self.q5)
        self.queue.update(source=self.q5_source,execution_registration=q5_binding,execution_version=5,test_evaluation_status='complete')
        M.replace_own_trace(self.out/'continuation_queue_v5/status.json',self.queue)
        rows = []
        for fit in self.fits:
            if fit['stage'] != '3': continue
            folder = self.out / 'runs' / fit['fit_id']
            rows.append({'fit_id': fit['fit_id'], 'passed': True,
                         'status_artifact': M.descriptor(folder / 'status.json', self.work),
                         'metadata': self.write(folder / 'metadata.json', {'fit_id': fit['fit_id']}),
                         'completion': self.write(folder / 'complete.json', {'passed': True}),
                         'cpu_checkpoint_audit': {'cpu_tensors_verified': True, 'checkpoint_loads': 3,
                             'only_map_location_cpu': True, 'gpu_operations': 0, 'model_constructions': 0,
                             'checkpoints': [{'path': (folder / n).relative_to(self.work).as_posix(), 'sha256': 'a' * 64}
                                             for n in ('initial.pt', 'latest.pt', 'best.pt')]}})
        self.audit = {'schema_version': 4, 'registration': reg, 'auditor_source': self.auditor_source,
                      'registered_total_fits': 360, 'registered_finetune_fits': 135, 'training_complete_finetune_observed': 135,
                      'verified_completed_finetune_fits': 135, 'logical_completed_fits_counted': 135, 'checkpoint_loads': 405,
                      'status': 'all_finetune_verified', 'all_135_finetune_checkpoints_verified': True, 'partial_snapshot': False,
                      'pending_fits': [], 'deferred_completed_fits': [], 'errors': [], 'completed_rows': rows,
                      'observed_auditor_operations_including_failed_checks': {'cpu_checkpoint_loads_started': 405,
                          'cpu_checkpoint_loads_returned': 405, 'model_construction_attempts_blocked': 0, 'gpu_initialization_attempts_blocked': 0}}
        self.audit_path = self.out / 'continuation_queue_v5/pre_test_finetune_cpu_audit_fixture.json'
        audit_binding = self.write(self.audit_path, self.audit)
        self.cpu_gate_path = self.out / 'continuation_queue_v5/pre_test_finetune_cpu_gate.json'
        self.write(self.cpu_gate_path, {'passed': True, 'audit': audit_binding, 'checkpoint_loads': 405, 'test_forward_calls_this_check': 0})

    def proof(self):
        return {'passed': True, 'formal_fits_verified': 360, 'supervised_fits_evaluated': 345,
                'supervised_conditions': 23, 'model_constructions': 0, 'gpu_forward_calls': 0,
                'evaluator_execution': {'version': 3, 'wrapper_source': self.evaluator_source}}

    def test_pending_no_e3_import_or_invocation(self):
        self.fixture()
        value = M.inspect_gate(work=self.work, verifier=lambda: self.fail('No E3 permitted'))
        self.assertEqual(value['status'], 'pending')
        self.assertFalse(value['test_or_feature_reads_performed'])
        self.assertFalse(value['actual_completed_report_verifier_called'])
        self.assertNotIn('torch', sys.modules); self.assertNotIn('numpy', sys.modules)

    def test_old_q3_terminal_cannot_replace_q5(self):
        self.fixture(complete=True)
        self.write(self.out / 'continuation_queue_v3/status.json', self.queue)
        (self.out / 'continuation_queue_v5/status.json').unlink()
        value = M.inspect_gate(work=self.work, verifier=lambda: self.fail('No E3 permitted'))
        self.assertEqual(value['status'], 'pending')

    def test_q5_q4_or_q3_owner_blocks_before_e3(self):
        self.fixture(complete=True)
        for folder in ('continuation_queue_v3', 'continuation_queue_v4', 'continuation_queue_v5'):
            path = self.out / folder / 'owner.json'; self.write(path, {'id': 'active'})
            self.assertEqual(M.inspect_gate(work=self.work, verifier=lambda: self.fail('No E3 permitted'))['status'], 'pending')
            path.unlink()

    def test_ready_requires_real_shape_and_e3_provenance(self):
        self.fixture(complete=True)
        for bad in ({'passed': True}, {**self.proof(), 'evaluator_execution': {'version': 2}},
                    {**self.proof(), 'model_constructions': False}):
            with self.assertRaises(ValueError): M.inspect_gate(work=self.work, verifier=lambda: bad)
        value = M.inspect_gate(work=self.work, verifier=self.proof)
        self.assertEqual(value['status'], 'ready_after_full_e3_verification')
        self.assertEqual(value['pre_test_finetune_cpu_verification']['actual_cpu_checkpoint_loads'], 405)

    def test_partial_failed_missing_q5_job_blocks_before_e3(self):
        self.fixture(complete=True)
        for outcomes in ({}, {'job': 'failed'}, {'job': 'complete', 'unregistered': 'complete'}):
            M.replace_own_trace(self.out / 'continuation_queue_v5/status.json', {**self.queue, 'outcomes': outcomes})
            with self.assertRaisesRegex(ValueError, 'fixed Q5 job'):
                M.inspect_gate(work=self.work, verifier=lambda: self.fail('No E3 permitted'))

    def test_copied_terminal_wrong_source_version_or_registration_rejected(self):
        self.fixture(complete=True)
        for mutation in ({'execution_version':4},{'execution_version':True},{'source':M.descriptor(M.V3_SOURCE)},
                         {'execution_registration':M.descriptor(self.out/'continuation_queue_v4/registration.json',self.work)},
                         {'test_evaluation_status':'running'}):
            M.replace_own_trace(self.out/'continuation_queue_v5/status.json',{**self.queue,**mutation})
            with self.assertRaisesRegex(ValueError,'Terminal status'):
                M.inspect_gate(work=self.work,verifier=lambda:self.fail('No E3 permitted'))

    def test_q5_lineage_and_formal_budget_not_weakened(self):
        self.fixture(complete=True)
        changes=[lambda q:q.update(previous_execution_registration={}),lambda q:q['sources'].pop(0),
                 lambda q:q.update(training_micro_subjects=True),lambda q:q.update(training_window_chunk=16),
                 lambda q:q['jobs'][0].update(fits=194),lambda q:q['jobs'][0].update(args=['changed']),lambda q:q['jobs'][1].update(dependencies=[])]
        for mutate in changes:
            value=copy.deepcopy(self.q5);mutate(value)
            M.replace_own_trace(self.out/'continuation_queue_v5/registration.json',value)
            terminal={**self.queue,'execution_registration':M.descriptor(self.out/'continuation_queue_v5/registration.json',self.work)}
            with self.assertRaises(ValueError):
                M.verify_q5_sources_and_terminal(value,terminal,out=self.out,
                    registration_binding=M.descriptor(self.out/'registration.json',self.work),work=self.work)

    def test_q4_cpu_evidence_cannot_substitute_for_q5(self):
        self.fixture(complete=True)
        binding=self.write(self.out/'continuation_queue_v4/pre_test_finetune_cpu_audit_fixture.json',self.audit)
        M.replace_own_trace(self.cpu_gate_path,{'passed':True,'audit':binding,'checkpoint_loads':405,'test_forward_calls_this_check':0})
        with self.assertRaisesRegex(ValueError,'actual Q5 pre-test audit'):
            M.inspect_gate(work=self.work,verifier=lambda:self.fail('No E3 permitted'))

    def test_cpu_shape_refuses_missing_duplicate_load_errors_and_deferred(self):
        self.fixture(complete=True)
        changes = [lambda a: a['completed_rows'].pop(), lambda a: a['completed_rows'].__setitem__(1, a['completed_rows'][0]),
                   lambda a: a.update(checkpoint_loads=402), lambda a: a.update(partial_snapshot=True),
                   lambda a: a.update(errors=[{'failed': True}]), lambda a: a.update(deferred_completed_fits=[{}]),
                   lambda a: a.update(pending_fits=['missing']),
                   lambda a: a['completed_rows'][0]['cpu_checkpoint_audit'].update(checkpoint_loads=True)]
        for mutate in changes:
            value = copy.deepcopy(self.audit); mutate(value)
            with self.assertRaises(ValueError):
                M.verify_cpu_shape(value, self.registration, M.descriptor(self.out / 'registration.json', self.work))

    def test_cpu_source_and_bound_metadata_cannot_change(self):
        self.fixture(complete=True)
        path = self.work / self.audit['completed_rows'][0]['metadata']['path']
        path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Evidence changed'):
            M.inspect_gate(work=self.work, verifier=lambda: self.fail('No E3 permitted'))

    def test_owner_or_terminal_mutation_during_verification_rejected(self):
        self.fixture(complete=True)
        owner = self.out / 'continuation_queue_v5/owner.json'
        def owner_mutate(): self.write(owner, {'id': 'new'}); return self.proof()
        with self.assertRaisesRegex(ValueError, 'owner appeared'):
            M.inspect_gate(work=self.work, verifier=owner_mutate)
        owner.unlink()
        def status_mutate():
            M.replace_own_trace(self.out / 'continuation_queue_v5/status.json', {**self.queue, 'test_evaluation_completed': False})
            return self.proof()
        with self.assertRaisesRegex(ValueError, 'changed during verification'):
            M.inspect_gate(work=self.work, verifier=status_mutate)

    def test_v2_source_and_delegated_measurement_code_unchanged(self):
        self.assertEqual(M.descriptor(M.V2_SOURCE)['sha256'], M.V2_SHA)
        self.assertEqual(M.descriptor(M.V3_SOURCE)['sha256'], M.V3_SHA)
        self.assertIs(M.measure, M.D.measure)
        self.assertEqual(M.D.SOURCE, M.SOURCE)
        self.assertEqual(M.measure.__code__.co_filename, str(M.V2_SOURCE))
        self.assertIs(M.probe_plan, M.D.probe_plan)
        self.assertIs(M.choose_records, M.D.choose_records)

    def test_measurement_provenance_wrapper_does_not_break_lease(self):
        with M.exclusive_measurement_lease(self.out) as owner:
            self.assertTrue(owner['id'])
        path = self.out / 'measurement.json'
        record = {'subtype': 'supplemental_head_resource_measurement', 'measurement_scope': {'seconds': 'excludes E2 prerequisite verification'}}
        M.write_measurement(path, record)
        saved = M.read(path)
        self.assertEqual(saved['delegated_resource_measurement_source'], M.delegated_source())
        self.assertEqual(saved['resource_measurement_version'], 4)
        self.assertIn('E3 prerequisite', saved['measurement_scope']['seconds'])

    def test_old_v2_measurement_not_reused(self):
        path = self.out / 'old.json'; self.write(path, {'subtype': 'supplemental_head_resource_measurement', 'passed': True})
        self.assertFalse(M.successful_existing(path, plan={}, gate={}, source={}))

    def test_existing_test_gate_cannot_be_created(self):
        training = {'passed': True, 'registration': {'path': 'r', 'sha256': 'a'}, 'formal_fits_verified': 360, 'statuses': []}
        with self.assertRaisesRegex(ValueError, 'Existing exact source-bound'):
            M.existing_test_gate(self.out, training, work=self.work)
        self.assertFalse((self.out / 'evaluation').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
