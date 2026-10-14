"""E4 strict pretest/default/source-context guards; no probabilities/models."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tridep_ccml.eeg_all_candidates import evaluate_v4 as E

def gates():
    source={'path':'CPU7','sha256':'synthetic'}
    gate={'passed':True,'original_training_complete':360,'FT':135,'CPU_loads':405}
    cpu={'schema_version':7,'auditor_source':source,'all_135_finetune_checkpoints_verified':True,
         'verified_completed_finetune_fits':135,'training_complete_finetune_observed':135,
         'registered_finetune_fits':135,'checkpoint_loads':405,'errors':[],
         'deferred_completed_fits':[],'pending_fits':[]}
    return gate,cpu,source

class Guards(unittest.TestCase):
    def test_default_zero_probability_and_tensor(self):
        output=io.StringIO()
        with patch.object(E,'controller',side_effect=AssertionError),patch.object(E,'verify_gate',side_effect=AssertionError),contextlib.redirect_stdout(output):E.main([])
        result=json.loads(output.getvalue())
        self.assertEqual(result['outer_probability_reads'],0);self.assertEqual(result['GPU_queries'],0)
        self.assertNotIn('torch',sys.modules)
    def test_partial_fit_or_CPU_gate_rejected(self):
        gate,cpu,source=gates();E.validate_gate_data(gate,cpu,source)
        for key,value in (('original_training_complete',359),('FT',134),('CPU_loads',402)):
            bad=copy.deepcopy(gate);bad[key]=value
            with self.assertRaises(ValueError):E.validate_gate_data(bad,cpu,source)
    def test_CPU_errors_deferred_pending_rejected(self):
        gate,cpu,source=gates()
        for key in ('errors','deferred_completed_fits','pending_fits'):
            bad=copy.deepcopy(cpu);bad[key]=['synthetic unresolved']
            with self.assertRaises(ValueError):E.validate_gate_data(gate,bad,source)
    def test_wrong_CPU_source_version_count_rejected(self):
        gate,cpu,source=gates()
        for key,value in (('schema_version',6),('auditor_source',{}),('checkpoint_loads',404),('all_135_finetune_checkpoints_verified',False)):
            bad=copy.deepcopy(cpu);bad[key]=value
            with self.assertRaises(ValueError):E.validate_gate_data(gate,bad,source)
    def test_active_training_owner_refuses_before_file_read(self):
        with patch.object(E.Q,'check_pause'),patch.object(Path,'exists',return_value=True),patch.object(E.S,'checked',side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError,'owner still present'):E.verify_gate({})
    def test_actual_run_requires_CPU7_binding(self):
        with self.assertRaises(ValueError):E.main(['--run'])
        with self.assertRaises(ValueError):E.main(['--worker'])
    def test_controller_worker_distinct(self):
        with self.assertRaises(ValueError):E.main(['--run','--worker'])
    def test_old_and_new_training_verifiers_separate(self):
        old={'execution_acceleration':{},'condition':'old','seed':11,'fold':1}
        new={**old,'actual_deterministic_repair_execution':{}}
        with patch.object(E.X,'verify_execution_acceleration',return_value='old proof'),patch.object(E.V,'verify_execution',return_value='new proof'):
            self.assertEqual(E.verify_training_execution(old),'old proof')
            self.assertEqual(E.verify_training_execution(new),'new proof')
    def test_native_child_tree_and_other_experiment_rejected(self):
        parent={'pid':10,'parent':3,'birth':'100','command':'controller'}
        launcher={'pid':20,'parent':10,'birth':'200','command':'evaluate_v4 --worker token'}
        actual={'pid':21,'parent':20,'birth':'210','command':launcher['command']}
        owner={'source':E.S.desc(E.SOURCE),'pid':10,'native_roles':[parent],'child_pid':20,'child_identity':launcher,'child_token':'token'}
        roles=E.admit_worker([parent,launcher,actual],owner,'token',21,'win32')
        self.assertEqual(len(roles),3)
        with self.assertRaises(ValueError):E.admit_worker([parent,launcher,actual,{'pid':22,'birth':'999'}],owner,'token',21,'win32')
    def test_reused_child_PID_wrong_birth_refused(self):
        parent={'pid':10,'parent':3,'birth':'100','command':'controller'}
        launcher={'pid':20,'parent':10,'birth':'200','command':'evaluate_v4 --worker token'}
        owner={'source':E.S.desc(E.SOURCE),'pid':10,'native_roles':[parent],'child_pid':20,'child_identity':launcher,'child_token':'token'}
        changed={**launcher,'birth':'201'}
        with self.assertRaises(ValueError):E.admit_worker([parent,changed],owner,'token',20,'win32')
    def test_private_E2_driver_before_any_probability_read(self):
        before=(E.E2.SOURCE,E.E2.execution_provenance,E.E2.materialize_finetune)
        driver=E.isolated_driver()
        self.assertEqual(driver.SOURCE,E.SOURCE)
        self.assertEqual((E.E2.SOURCE,E.E2.execution_provenance,E.E2.materialize_finetune),before)
        self.assertNotIn('torch',sys.modules)
    def test_threshold_and_metrics_original_bytecode(self):
        driver=E.isolated_driver()
        self.assertEqual(driver.metrics.__code__,E.E2.E1.metrics.__code__)
        self.assertEqual(driver.predicted_class.__code__,E.E2.E1.predicted_class.__code__)
        self.assertEqual(driver.predicted_class(.5),0)
        self.assertEqual(driver.predicted_class(.50001),1)

if __name__=='__main__':
    import time
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    if result.wasSuccessful():
        target=Path(__file__).parent/('owned_evaluation_v4_preparation_'+str(time.time_ns())+'.json')
        value={'status':'preparation_only','tests_run':result.testsRun,'source':E.S.desc(E.SOURCE),
               'test_source':E.S.desc(Path(__file__)),'Tensor_imports':0,'outer_probability_reads':0,
               'actual_native_or_GPU_queries':0,'model_calls':0,'actual_evaluation_started':False,
               'full_new_source_chain_review_passed':False}
        with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
        print(json.dumps(E.S.desc(target)))
    sys.exit(0 if result.wasSuccessful() else 1)
