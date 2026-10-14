"""Original buffered counter-guard and CPU7 namespace tests; no tensors."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tridep_ccml.eeg_all_candidates import repair_observation_v1 as V,finetune_v3 as T,storage as S

def fixture():
    execution={'condition':'CodeBrain-lastblock-finetune','seed':11,'fold':1,
               'delegated_finetune_implementation':V.X.f2_provenance()}
    with patch.object(S,'save'):
        trace=T.BufferedAttempt(execution['condition'],11,1,execution)
        trace.observe('forward',lambda raw:None)(None)
        trace.observe('encoder_forward',lambda model,tensor:None)(None,[0]*8)
        trace.observe('backward',lambda:None)()
        trace.observe('optimizer_step',lambda:None)()
        trace.flush('latest_checkpoint_committed',latest_checkpoint={
            'path':(V.X.R.fit_directory(execution['condition'],11,1)/'latest.pt').relative_to(S.WORK).as_posix(),
            'epoch':1,'optimizer_updates':1,'backward_calls':1,'checkpoint_save_returned':True,
            'checkpoint_hash':None,'hash_scope':'not_rehashed_each_epoch; final_checkpoint_audit_binds_bytes'})
        trace.finish()
    return trace.record,{'condition':execution['condition'],'seed':11,'fold':1}

class Guards(unittest.TestCase):
    def test_same_original_guard_accepts_terminal(self):
        record,fit=fixture();self.assertTrue(V.counter_guard()(record,fit)['terminal'])
    def test_count_snapshot_tamper_rejected(self):
        record,fit=fixture();record['backward_returned']=0
        with self.assertRaises(ValueError):V.counter_guard()(record,fit)
    def test_alias_mismatch_rejected(self):
        record,fit=fixture();record['encode_returned']=0;record['durable_counter_snapshot']['encode_returned']=0
        with self.assertRaises(ValueError):V.counter_guard()(record,fit)
    def test_unreturned_call_cannot_complete(self):
        record,fit=fixture();record['forward_started']=2;record['durable_counter_snapshot']['forward_started']=2
        record['started_without_observed_return']['forward']=1
        with self.assertRaises(ValueError):V.counter_guard()(record,fit)
    def test_dirty_snapshot_rejected(self):
        record,fit=fixture();record['dirty_since_last_flush']=True
        with self.assertRaises(ValueError):V.counter_guard()(record,fit)
    def test_tail_not_zero_filled(self):
        record,fit=fixture()
        record.update(status='running',graceful_terminal_flush=False,counter_snapshot_exact_for_process=False,
                      unflushed_tail_unknown=True,unflushed_tail_counters=None)
        self.assertTrue(V.counter_guard()(record,fit)['unflushed_tail_unknown'])
        record['unflushed_tail_counters']={}
        with self.assertRaises(ValueError):V.counter_guard()(record,fit)
    def test_bad_durable_commit_path_rejected(self):
        record,fit=fixture();record['latest_durable_checkpoint']['path']='wrong.pt'
        with self.assertRaises(ValueError):V.counter_guard()(record,fit)
    def test_flush_sequence_strict(self):
        record,fit=fixture();record['flush_sequence']+=1
        with self.assertRaises(ValueError):V.counter_guard()(record,fit)
    def test_version3_cannot_claim_repair4(self):
        with self.assertRaises(ValueError):V.verify_execution({'observer_version':3},condition='CodeBrain-lastblock-finetune',seed=11,fold=1)
    def test_false_completed_metadata_detected(self):
        expected={'observer_version':4,'registration':{'path':'mock','sha256':'mock'}}
        value={'actual_deterministic_repair_execution':expected,'execution_acceleration':{'observer_version':3}}
        with patch.object(V.D,'load_execution',return_value=expected):
            with self.assertRaises(ValueError):V.verify_execution(value,condition='CodeBrain-lastblock-finetune',seed=11,fold=1)
    def test_CPU7_default_and_private_old_namespace(self):
        path=Path(__file__).with_name('audit_completed_finetune_checkpoints_v7.py')
        spec=importlib.util.spec_from_file_location('_test_CPU7_private',path)
        cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
        stream=io.StringIO()
        with patch.object(cpu,'audit',side_effect=AssertionError),contextlib.redirect_stdout(stream):cpu.main([])
        self.assertEqual(json.loads(stream.getvalue())['checkpoint_tensor_loads'],0)
        self.assertEqual(cpu.B.SOURCE,path.resolve())
        self.assertNotIn('torch',sys.modules)
    def test_no_second_checkpoint_math_implementation(self):
        path=Path(__file__).with_name('audit_completed_finetune_checkpoints_v7.py')
        text=path.read_text(encoding='utf-8')
        self.assertNotIn('torch.load(',text);self.assertNotIn('deep_checkpoint(',text)
        self.assertIn('result=B.audit()',text)
    def test_unknown_actual_observer_rejected(self):
        record,fit=fixture()
        with patch.object(V.D,'load_execution',side_effect=AssertionError):
            with self.assertRaises(ValueError):V.verify_execution({'observer_version':5},condition=fit['condition'],seed=11,fold=1)
    def test_actual4_provenance_not_relabelled(self):
        record,fit=fixture();artifact={'path':'synthetic','sha256':'synthetic'}
        with patch.object(S,'checked',return_value=Path('synthetic')),patch.object(S,'desc',return_value=artifact),patch.object(S,'read',return_value=record),patch.object(V,'verify_execution',return_value={**record['execution_acceleration'],'observer_version':4,
            'deterministic_backward_repair':{'accumulation_order_changed':False},
            'qualification':{'path':'synthetic','sha256':'synthetic'},'registration':{'path':'synthetic','sha256':'synthetic'}}):
            with self.assertRaises(ValueError):V.verify_observer(record,artifact,fit)

if __name__=='__main__':
    import time
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    if result.wasSuccessful():
        target=Path(__file__).parent/('repair_observation_CPU7_preparation_'+str(time.time_ns())+'.json')
        value={'status':'preparation_only','tests_run':result.testsRun,'sources':[S.desc(x) for x in
               (V.SOURCE,Path(__file__),Path(__file__).with_name('audit_completed_finetune_checkpoints_v7.py'))],
               'Tensor_imports':0,'checkpoint_tensor_loads':0,'model_calls':0,'GPU_calls':0,
               'actual_CPU7_audit_started':False,'actual_repair4_completion_claimed':False}
        with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
        print(json.dumps(S.desc(target)))
    sys.exit(0 if result.wasSuccessful() else 1)
