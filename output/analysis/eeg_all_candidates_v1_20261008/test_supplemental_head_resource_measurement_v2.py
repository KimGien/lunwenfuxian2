"""Stdlib-only gate, selection, ownership and trace safety tests; no torch/models."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest

SOURCE = Path(__file__).with_name('supplemental_head_resource_measurement_v2.py')
spec = importlib.util.spec_from_file_location('supplemental_resource_under_test', SOURCE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class SupplementalSafety(unittest.TestCase):
    def setUp(self):
        self.fixture_root = M.WORK / '.sht'
        self.fixture_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.fixture_root, prefix='t')
        self.work = Path(self.temp.name)
        self.out = self.work / M.OUT_REL
        fits = [{'fit_id': f'condition_{c}/seed_{s}/fold_{f}', 'condition': f'condition_{c}', 'seed': s, 'fold': f}
                for c in range(24) for s in (11,22,33) for f in range(1,6)]
        self.registration = {'formal_fits': 360, 'fits': fits}
        self.queue = {'status': 'all_training_complete', 'test_evaluation_completed': True,
                      'test_evaluation_frozen': False, 'outcomes': {'job': 'complete'}}

    def tearDown(self):
        target = Path(self.temp.name).resolve()
        assert target.is_relative_to(self.fixture_root.resolve()) and target != self.fixture_root.resolve()
        assert target.is_relative_to(M.WORK.resolve())
        self.temp.cleanup()

    def write_fixture(self, *, complete=False, owner=False):
        M.write_new(self.out/'registration.json', self.registration)
        M.write_new(self.out/'continuation_queue_v3/status.json', self.queue)
        if owner:
            M.write_new(self.out/'continuation_queue_v3/owner.json', {'id': 'active'})
        if complete:
            for fit in self.registration['fits']:
                p=self.out/'runs'/fit['condition']/f"seed_{fit['seed']}"/f"fold_{fit['fold']}"/'status.json'
                M.write_new(p, {'status':'training_complete'})
            sources=[]
            for n in range(47):
                p=self.work/f'source_{n}.py';p.write_text(f'# source {n}\n', encoding='utf-8')
                sources.append(M.descriptor(p, self.work))
            M.write_new(self.out/'continuation_queue_v3/registration.json',
                        {'sources':sources,'registry':M.descriptor(self.out/'registration.json',self.work),'jobs':[{'id':'job'}]})

    @staticmethod
    def proof():
        return {'passed':True,'formal_fits_verified':360,'supervised_fits_evaluated':345,
                'supervised_conditions':23,'model_constructions':0,'gpu_forward_calls':0}

    def test_import_is_stdlib_only(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('numpy',sys.modules)

    def test_pending_never_calls_final_verifier(self):
        self.write_fixture()
        def forbidden(): raise AssertionError('E2 must not inspect test artifacts yet')
        result=M.inspect_gate(work=self.work,verifier=forbidden)
        self.assertEqual(result['status'],'pending')
        self.assertFalse(result['test_or_feature_reads_performed'])
        self.assertEqual(result['training_status_counts'],{'engineering_pending':360})

    def test_active_owner_blocks_even_all360(self):
        result=M.screening(self.registration,['training_complete']*360,self.queue,True)
        self.assertEqual(result['status'],'pending')
        self.assertIn('Q3 owner has not been released',result['reasons'])

    def test_training_done_without_evaluation_blocks(self):
        queue={**self.queue,'test_evaluation_completed':False}
        result=M.screening(self.registration,['training_complete']*360,queue,False)
        self.assertEqual(result['status'],'pending')

    def test_complete_metadata_requires_strong_e2_proof(self):
        self.write_fixture(complete=True)
        with self.assertRaisesRegex(ValueError,'E2 full report verification'):
            M.inspect_gate(work=self.work,verifier=lambda:{'passed':True})
        self.assertEqual(M.inspect_gate(work=self.work,verifier=self.proof)['status'],'ready_after_full_e2_verification')

    def test_one_failed_q3_job_blocks_before_e2(self):
        self.queue['outcomes']['job']='failed';self.write_fixture(complete=True)
        with self.assertRaisesRegex(ValueError,'fixed Q3 job'):
            M.inspect_gate(work=self.work,verifier=lambda:(_ for _ in ()).throw(AssertionError('not called')))

    def test_queue_mutation_during_verification_rejected(self):
        self.write_fixture(complete=True)
        def mutate():
            p=self.out/'continuation_queue_v3/status.json'
            M.replace_own_trace(p,{**self.queue,'test_evaluation_completed':False})
            return self.proof()
        with self.assertRaisesRegex(ValueError,'changed during verification'):
            M.inspect_gate(work=self.work,verifier=mutate)

    def test_owner_appearing_during_verification_rejected(self):
        self.write_fixture(complete=True)
        def mutate():
            M.write_new(self.out/'continuation_queue_v3/owner.json',{'id':'new'})
            return self.proof()
        with self.assertRaisesRegex(ValueError,'owner appeared'):
            M.inspect_gate(work=self.work,verifier=mutate)

    def test_tampered_source_rejected_before_e2(self):
        self.write_fixture(complete=True)
        (self.work/'source_23.py').write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Evidence changed'):
            M.inspect_gate(work=self.work,verifier=self.proof)

    def test_feature_selection_never_opens_unselected_people(self):
        manifest={'records':[{'subject_id':'train','features':{'path':'train.npz','sha256':'a'},'archive_key':'native'},
                             {'subject_id':'outer-test','features':{'path':'DO_NOT_OPEN','sha256':'b'}}]}
        rows=M.choose_records(manifest,['train'])
        self.assertEqual([r['subject_id'] for r in rows],['train'])
        self.assertEqual(rows[0]['archive_key'],'native')
        self.assertEqual(M.choose_records(manifest,['train'],key_override='eegpt')[0]['archive_key'],'eegpt')

    def test_duplicate_feature_identity_rejected(self):
        r={'subject_id':'x','features':{'path':'x','sha256':'a'}}
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            M.choose_records({'records':[r,r]},['x'])

    def test_lease_collision_preserves_original_owner(self):
        with M.exclusive_measurement_lease(self.out) as value:
            p=self.out/'supervised_engineering/supplemental_head_resource_owner.json'
            before=p.read_bytes()
            with self.assertRaises(FileExistsError):
                with M.exclusive_measurement_lease(self.out):pass
            self.assertEqual(p.read_bytes(),before)
            self.assertEqual(M.read(p)['id'],value['id'])
        self.assertFalse(p.exists())

    def test_new_trace_cannot_overwrite_historical_evidence(self):
        p=self.out/'event.json';M.write_new(p,{'cost':1})
        with self.assertRaises(FileExistsError):M.write_new(p,{'cost':2})
        self.assertEqual(M.read(p),{'cost':1})
        M.replace_own_trace(p,{'cost':3})
        self.assertEqual(M.read(p),{'cost':3})

    def test_outside_workspace_descriptors_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside workspace'):
            M.descriptor(SOURCE,self.work)

    def test_success_reuse_requires_resources_and_one_real_backward(self):
        p=self.out/'measurement.json';source={'path':'source','sha256':'x'};lock={'path':'lock','sha256':'y'}
        plan={'condition':'c','condition_source_lock':lock,'probe_subject_ids':['train'],
              'feature_manifest':source,'feature_records':[{'subject_id':'train','features':source,'archive_key':'native'}],
              'contract':{'trainable_parameter_names':['head.weight']}}
        gate={'registration':source,'evaluation_verification':{'complete_manifest':lock}}
        x={'subtype':'supplemental_head_resource_measurement','condition':'c','source':source,'registration':source,
           'condition_source_lock':lock,'status':'complete','passed':True,'optimizer_updates':0,'gradient_passes':1,
           'forward_started':1,'forward_returned':1,'backward_started':1,'backward_returned':1,
           'new_measurement_not_historical_backfill':True,'resume_checked':True,'seconds':2.0,
           'forward_backward_seconds':1.0,'peak_allocated_bytes':123,'seed':11,'fold':1,'input_role':'train',
           'probe_subject_ids':['train'],'feature_manifest':source,'loaded_feature_sources':[source],
           'feature_archive_keys_read':[{'subject_id':'train','key':'native'}],'actual_trainable_parameter_names':['head.weight'],
           'gate':gate,'optimizer_step_started':0,'optimizer_step_returned':0,'formal_fits':0,'new_encoder_forwards':0,
           'model_constructions_started':1,'model_constructions_returned':1,'historical_resource_fields_modified':False}
        M.write_new(p,x)
        args={'plan':plan,'source':source,'gate':gate}
        self.assertTrue(M.successful_existing(p,**args))
        for key, invalid in [('peak_allocated_bytes',None),('input_role','test'),('seed',22),
                             ('optimizer_step_started',1),('probe_subject_ids',['outer-test']),
                             ('loaded_feature_sources',[]),('new_encoder_forwards',1),('forward_returned',True),
                             ('seconds',True),('gradient_passes',True),('model_constructions_returned',0),
                             ('historical_resource_fields_modified',True)]:
            with self.subTest(field=key):
                M.replace_own_trace(p,{**x,key:invalid})
                self.assertFalse(M.successful_existing(p,**args))
        p.write_text(json.dumps({**x,'seconds':float('inf')}),encoding='utf-8')
        self.assertFalse(M.successful_existing(p,**args))

    def test_missing_e2_gate_cannot_be_created_by_verification(self):
        training={'passed':True,'registration':{'path':'r','sha256':'a'},'formal_fits_verified':360,'statuses':[]}
        with self.assertRaisesRegex(ValueError,'Existing exact source-bound'):
            M.existing_test_gate(self.out,training,work=self.work)
        self.assertFalse((self.out/'evaluation').exists())
        digest=hashlib.sha256(json.dumps(training,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        gate_path=self.out/'evaluation'/('test_gate_'+digest+'.json')
        M.write_new(gate_path,training)
        M.write_new(self.out/'evaluation/complete.json',{'test_gate':M.descriptor(gate_path,self.work)})
        before=gate_path.read_bytes()
        self.assertEqual(M.existing_test_gate(self.out,training,work=self.work),M.descriptor(gate_path,self.work))
        self.assertEqual(gate_path.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
