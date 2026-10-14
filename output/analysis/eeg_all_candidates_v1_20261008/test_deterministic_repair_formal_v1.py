"""Pure qualification-data/observer/activation guards; no model/Tensor/GPU."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tridep_ccml.eeg_all_candidates import deterministic_repair_execution_v1 as D,finetune_v4 as F

def data(condition):
    names=['head.weight','encoder.last.weight']
    contract={'input_contract':{'micro_subjects':1,'window_chunk':8,'effective_subjects':8,
        'windows_per_training_subject':8,'all_window_validation':60,'coverage_cycle_epochs':8,'precision':'float32'},
        'trainable_parameter_names':names,'frozen_parameter_names':['encoder.frozen.weight'],
        'engineering_backward_calls':4,'engineering_optimizer_updates':0,'probe_subject_ids':['train1'],
        'raw_sources':{'train1':{'path':'synthetic_raw','sha256':'synthetic'}},
        'native_CUDA_gradient_equivalence_established':False,'accumulation_order_changed':condition==D.REPAIR_CONDITIONS[1]}
    for k in ('forward_checked','gradient_checked','resume_checked','training_only_engineering_checked',
              'frozen_buffers_and_parameters_zero_update_verified','separate_new_formal_execution_and_auditor_required'):contract[k]=True
    value={'condition':condition,'status':'engineering_verified','kind':'deterministic_repair_engineering',
        'GPU_engineering_checks_passed':True,'new_formal_execution_required':True,'formal_fits':0,
        'native_CUDA_gradient_equivalence_established':False,'performance_improvement_claimed':False,
        'accumulation_order_changed':condition==D.REPAIR_CONDITIONS[1],'outer_test_read':False,
        'original_native_forward_bitwise_equal':True,'zero_step_optimizer_and_rng_resume_verified':True,
        'primary_engineering_seconds':1.,'probe_person':'train1','raw_sources':contract['raw_sources'],
        'runtime_contract':contract,'checks':[{'epoch_phase':1,'repeat_bitwise_equal':True,'gradient_parameter_names':['head.weight']},
            {'epoch_phase':11,'repeat_bitwise_equal':True,'gradient_parameter_names':names}]}
    for kind,total in (('forward',6),('backward',4),('encoder_forward',6),('encoder_windows',48),('optimizer_step',0)):
        value.update({kind+'_started':total,kind+'_returned':total})
    return value,{'train_ids':['train1'],'val_ids':['val1'],'test_ids':['test1']}

def execution():
    return {'condition':D.REPAIR_CONDITIONS[0],'seed':11,'fold':1,
            'delegated_finetune_implementation':D.X.f2_provenance(),
            'deterministic_backward_repair':{'accumulation_order_changed':False},
            'qualification':{'path':'synthetic','sha256':'synthetic'},'registration':{'path':'synthetic','sha256':'synthetic'}}

class Guards(unittest.TestCase):
    def test_default_import_without_torch(self):
        self.assertNotIn('torch',sys.modules)
        stream=io.StringIO()
        with patch.object(D,'load_execution',side_effect=AssertionError),contextlib.redirect_stdout(stream):F.main([])
        self.assertEqual(json.loads(stream.getvalue())['formal_fits'],0)
    def test_no_actual_registration_refuses_before_tensor(self):
        with patch.object(F.Q,'check_pause'),patch.object(D,'load_execution',side_effect=ValueError('unqualified')):
            with self.assertRaisesRegex(ValueError,'unqualified'):F.fit(D.REPAIR_CONDITIONS[0],11,1,repair_registration={})
        self.assertNotIn('torch',sys.modules)
    def test_both_qualification_data_paths(self):
        for condition in D.REPAIR_CONDITIONS:
            value,fit=data(condition);self.assertEqual(D.validate_report_data(value,condition,fit),value['runtime_contract'])
    def test_started_unreturned_backward_rejected(self):
        value,fit=data(D.REPAIR_CONDITIONS[0]);value['backward_returned']=3
        with self.assertRaises(ValueError):D.validate_report_data(value,D.REPAIR_CONDITIONS[0],fit)
    def test_optimizer_step_rejected(self):
        value,fit=data(D.REPAIR_CONDITIONS[0]);value['optimizer_step_started']=1
        with self.assertRaises(ValueError):D.validate_report_data(value,D.REPAIR_CONDITIONS[0],fit)
    def test_outer_role_rejected(self):
        value,fit=data(D.REPAIR_CONDITIONS[0]);value['probe_person']='test1'
        with self.assertRaises(ValueError):D.validate_report_data(value,D.REPAIR_CONDITIONS[0],fit)
    def test_Mamba_order_disclosure_required(self):
        value,fit=data(D.REPAIR_CONDITIONS[1]);value['accumulation_order_changed']=False
        with self.assertRaises(ValueError):D.validate_report_data(value,D.REPAIR_CONDITIONS[1],fit)
    def test_native_backward_equivalence_cannot_be_claimed(self):
        value,fit=data(D.REPAIR_CONDITIONS[0]);value['native_CUDA_gradient_equivalence_established']=True
        with self.assertRaises(ValueError):D.validate_report_data(value,D.REPAIR_CONDITIONS[0],fit)
    def test_partition_and_phase_scope_strict(self):
        value,fit=data(D.REPAIR_CONDITIONS[0]);value['runtime_contract']['frozen_parameter_names']+=['head.weight']
        with self.assertRaises(ValueError):D.validate_report_data(value,D.REPAIR_CONDITIONS[0],fit)
        value,fit=data(D.REPAIR_CONDITIONS[0]);value['checks'][0]['gradient_parameter_names']+=['encoder.last.weight']
        with self.assertRaises(ValueError):D.validate_report_data(value,D.REPAIR_CONDITIONS[0],fit)
    def test_nonrepeatable_gradients_rejected(self):
        value,fit=data(D.REPAIR_CONDITIONS[0]);value['checks'][1]['repeat_bitwise_equal']=False
        with self.assertRaises(ValueError):D.validate_report_data(value,D.REPAIR_CONDITIONS[0],fit)
    def test_initial_trace_truthfully_version4(self):
        saves=[]
        with patch.object(F.S,'save',side_effect=lambda path,value,**kwargs:saves.append(copy.deepcopy(value))):
            trace=F.RepairAttempt(D.REPAIR_CONDITIONS[0],11,1,execution())
        self.assertEqual(saves[0]['implementation']['version'],4)
        self.assertTrue(saves[0]['implementation']['backward_graph_privately_changed'])
        self.assertEqual(saves[0]['actual_deterministic_repair_execution'],execution())
    def test_captured_original_builder_and_exception_restoration(self):
        count=[]
        def base_builder(condition,*,seed=11,device='cuda',window_chunk=8):
            count.append((condition,seed))
            return types.SimpleNamespace(forward=lambda raw:0,adapter=types.SimpleNamespace()),{}
        old=lambda *a,**k:None
        v1=types.SimpleNamespace(build_model=old,R=types.SimpleNamespace(fit_directory=lambda *a:Path('/synthetic')))
        v2=types.SimpleNamespace(V1=v1,_BUILD=base_builder,provenance=lambda:{'source':'fake'})
        trace=types.SimpleNamespace(record={'condition':D.REPAIR_CONDITIONS[0],'seed':11,'fold':1,'execution_acceleration':execution()},
                                    observe=lambda kind,fn:fn,bind_adapter=lambda adapter:None)
        @contextlib.contextmanager
        def template(*args):
            before=v1.build_model
            def wrong(*args,**kwargs):raise AssertionError('Recursively used active wrapper')
            v1.build_model=wrong
            try:yield
            finally:v1.build_model=before
        with patch.object(F.TEMPLATE,'activate',template),patch.object(F.S,'save'):
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with F.activate(trace,v2,None):
                    model,audit=v1.build_model(D.REPAIR_CONDITIONS[0],seed=11)
                    self.assertEqual(audit['actual_deterministic_repair_execution'],execution())
                    raise RuntimeError('intentional')
        self.assertIs(v1.build_model,old);self.assertFalse(F._ACTIVE);self.assertEqual(len(count),1)
    def test_default_rejects_resume_options(self):
        with self.assertRaises(ValueError):F.main(['--resume'])
    def test_missing_new_sources_prevents_future_registration(self):
        with self.assertRaisesRegex(ValueError,'not prepared'):D.implementation_sources()

if __name__=='__main__':
    import time
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    if result.wasSuccessful():
        a=Path(__file__).resolve().parent
        report=a/('deterministic_repair_formal_preparation_'+str(time.time_ns())+'.json')
        value={'status':'preparation_only','tests_run':result.testsRun,'formal_source':F.S.desc(F.SOURCE),
               'contract_source':F.S.desc(D.SOURCE),'test_source':F.S.desc(Path(__file__)),
               'Tensor_imports':0,'model_calls':0,'GPU_calls':0,'formal_fits':0,'actual_registration_created':False,
               'new_CPU_evaluator_cost_queue_sources_still_required':True,'actual_GPU_qualification_complete':False}
        with report.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
        print(json.dumps(F.S.desc(report)))
    sys.exit(0 if result.wasSuccessful() else 1)
