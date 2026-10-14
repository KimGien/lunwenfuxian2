"""Pure new-queue native/WSL admission and default guards; no actual inventory."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tridep_ccml.eeg_all_candidates import repair_runtime_owner_v1 as O,repair_owned_worker_v1 as W,continuation_queue_v6 as C

def fixture(platform='win32'):
    queue={'pid':10,'parent':9,'birth':'100','command':O.QUEUE_MODULE+' --run'}
    parent={'pid':9,'parent':2,'birth':'99','command':queue['command']}
    command=O.WORKER_MODULE+' --run --token actual-token'
    if platform=='linux':command='wsl.exe '+command
    launch={'pid':20,'parent':10,'birth':'200','command':command}
    actual={'pid':21,'parent':20,'birth':'210','command':command}
    owner={'id':'q6-owner','execution_version':6,'pid':10,'native_roles':[queue,parent],
           'child_token':'actual-token','child_job':'CodeBrain-lastblock-finetune-fit',
           'child_pid':20,'child_native_identity':launch}
    admission={'queue_owner_id':'q6-owner','token':'actual-token','job':owner['child_job']}
    return [queue,parent,launch]+([actual] if platform=='win32' else []),owner,admission

class Guards(unittest.TestCase):
    def test_default_without_tensor_or_queries(self):
        self.assertNotIn('torch',sys.modules)
        for module in (W,C):
            output=io.StringIO()
            with patch.object(O.Q,'host_inventory',side_effect=AssertionError),patch.object(O.Q,'gpu_empty',side_effect=AssertionError),contextlib.redirect_stdout(output):module.main([])
            value=json.loads(output.getvalue());self.assertEqual(value['model_calls'],0);self.assertEqual(value['GPU_queries'],0)
    def test_exact_native_tree(self):
        items,owner,admission=fixture()
        self.assertEqual(O.validate_native(items,owner,admission,worker_pid=21,platform='win32'),items)
    def test_exact_WSL_tree(self):
        items,owner,admission=fixture('linux')
        self.assertEqual(O.validate_native(items,owner,admission,worker_pid=1234,platform='linux'),items)
    def test_PID_reuse_birth_changed_rejected(self):
        items,owner,admission=fixture();items=copy.deepcopy(items);items[0]['birth']='101'
        with self.assertRaises(ValueError):O.validate_native(items,owner,admission,worker_pid=21,platform='win32')
    def test_other_experimental_process_rejected(self):
        items,owner,admission=fixture();items.append({'pid':70,'parent':2,'birth':'700','command':'other'})
        with self.assertRaises(ValueError):O.validate_native(items,owner,admission,worker_pid=21,platform='win32')
    def test_wrong_child_parent_rejected(self):
        items,owner,admission=fixture();items[-1]['parent']=77
        with self.assertRaises(ValueError):O.validate_native(items,owner,admission,worker_pid=21,platform='win32')
    def test_wrong_command_and_nonce_rejected(self):
        items,owner,admission=fixture();items[-1]['command']='some other child'
        with self.assertRaises(ValueError):O.validate_native(items,owner,admission,worker_pid=21,platform='win32')
        items,owner,admission=fixture();admission['token']='old-token'
        with self.assertRaises(ValueError):O.validate_native(items,owner,admission,worker_pid=21,platform='win32')
    def test_old_version_or_owner_cannot_admit(self):
        for key,value in (('execution_version',5),('id','old-owner')):
            items,owner,admission=fixture();owner[key]=value
            with self.assertRaises(ValueError):O.validate_native(items,owner,admission,worker_pid=21,platform='win32')
    def test_missing_child_publish_rejected(self):
        items,owner,admission=fixture();owner['child_native_identity']=None
        with self.assertRaises(ValueError):O.validate_native(items,owner,admission,worker_pid=21,platform='win32')
    def test_native_dispatcher_roles(self):
        items,_,_=fixture()
        self.assertEqual(O.queue_roles(items,10),items[:2])
        with self.assertRaises(ValueError):O.queue_roles(items,77)
    def test_Q5_active_queue_blocks_acquisition(self):
        with patch.object(C.Q,'check_pause'),patch.object(C.Q,'check_boundary',side_effect=ValueError('Q5 owner present')),patch.object(C.Q,'host_inventory',side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError,'Q5 owner'):C.acquire({})
    def test_incomplete_actual_inputs_refuse_registration(self):
        with self.assertRaises(ValueError):C.main(['--register'])
    def test_worker_actual_admission_mandatory(self):
        with self.assertRaises(ValueError):W.main(['--run','--condition','CodeBrain-lastblock-finetune'])
    def test_default_resume_refused(self):
        for module in (C,W):
            with self.assertRaises(ValueError):module.main(['--resume'])
    def test_registration_and_run_are_distinct(self):
        with self.assertRaises(ValueError):C.main(['--run','--register'])
    def test_actual_source_chain_still_missing(self):
        with self.assertRaisesRegex(ValueError,'not prepared'):C.sources()

if __name__=='__main__':
    import time
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    if result.wasSuccessful():
        path=Path(__file__).resolve().parent/('repair_queue_ownership_preparation_'+str(time.time_ns())+'.json')
        value={'status':'preparation_only','tests_run':result.testsRun,
               'sources':[C.S.desc(p) for p in (C.SOURCE,W.SOURCE,O.SOURCE,Path(__file__))],
               'actual_native_or_GPU_queries':0,'Tensor_imports':0,'model_calls':0,'formal_fits':0,
               'actual_registration_or_queue_started':False,'new_CPU7_E4_cost8_and_complete_source_review_required':True}
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
        print(json.dumps(C.S.desc(path)))
    sys.exit(0 if result.wasSuccessful() else 1)
