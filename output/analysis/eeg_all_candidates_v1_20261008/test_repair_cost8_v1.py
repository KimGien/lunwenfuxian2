"""Pure Cost8 scope/dedup tests; no actual collect, model or GPU."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
path=Path(__file__).with_name('cost_aggregation_v8.py')
spec=importlib.util.spec_from_file_location('_test_Cost8',path)
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
NORMALIZE=C.delegate().load_v5().load_v4().normalized
def value():
    d={'source':C.S.desc(C.Q.SOURCE),'kind':'deterministic_repair_engineering','condition':C.Q.CONDITIONS[0],
       'status':'engineering_verified','formal_fits':0,'primary_engineering_seconds':2.5}
    for key,n in (('forward',6),('backward',4),('encoder_forward',6),('encoder_windows',48),('optimizer_step',0)):
        d[key+'_started']=d[key+'_returned']=n
    return d

class Guards(unittest.TestCase):
    def test_default_does_not_collect(self):
        out=io.StringIO()
        with patch.object(C,'collect',side_effect=AssertionError),contextlib.redirect_stdout(out):C.main([])
        self.assertFalse(json.loads(out.getvalue())['actual_cost_collection_started'])
        self.assertNotIn('torch',sys.modules)
    def test_primary_engineering_once(self):
        row=C.repair_worker_record({'path':'synthetic','sha256':'synthetic'},value(),NORMALIZE)
        self.assertEqual(row['seconds_observed'],2.5);self.assertEqual(row['formal_fits'],0)
        self.assertEqual(row['observed_counters']['optimizer_returned'],0)
    def test_running_time_not_inferred(self):
        d=value();d.update(status='running');d.pop('primary_engineering_seconds')
        row=C.repair_worker_record({'path':'synthetic','sha256':'synthetic'},d,NORMALIZE);self.assertIsNone(row['seconds_observed'])
    def test_parent_elapsed_not_added(self):
        row=C.repair_worker_record({'path':'synthetic','sha256':'synthetic'},value(),NORMALIZE)
        nested={'aggregation_role':'supplemental_nested_or_unmeasured','observed_counters':{},
                'seconds_observed':90000}
        total=C.category_totals([row,nested])
        self.assertEqual(total['known_recorded_scope_seconds_subtotal'],2.5)
        self.assertEqual(total['primary_invocations'],1);self.assertEqual(total['supplemental_records'],1)
    def test_unknown_counter_not_zero_filled(self):
        row=C.repair_worker_record({'path':'synthetic','sha256':'synthetic'},value(),NORMALIZE)
        self.assertTrue(row['unknown_counter_fields'])
        total=C.category_totals([row]);self.assertIsNone(total['counter_totals']['outer_test_forward_returned']['known_observed_subtotal'])
    def test_step_or_formal_repair_rejected(self):
        for change in ({'formal_fits':1},{'optimizer_step_started':1},{'optimizer_step_returned':1}):
            d=value();d.update(change)
            with self.assertRaises(ValueError):C.repair_worker_record({'path':'synthetic','sha256':'synthetic'},d,NORMALIZE)
    def test_unknown_source_rejected(self):
        d=value();d['source']={'path':'wrong','sha256':'wrong'}
        with self.assertRaises(ValueError):C.repair_worker_record({'path':'synthetic','sha256':'synthetic'},d,NORMALIZE)
    def test_old_actual_records_delegate_unchanged(self):
        sentinel=object();seen=[]
        def old(data,artifact):seen.append((data,artifact));return sentinel
        self.assertIs(C.buffered_observation(old,{'implementation':{'version':3}},{}),sentinel)
        self.assertEqual(len(seen),1)
    def test_existing_Q6_parent_scope_is_not_added_twice(self):
        p=C.S.OUT/'continuation_queue_v6/attempts/synthetic/status.json'
        artifact={'path':p.relative_to(C.WORK).as_posix(),'sha256':'synthetic'}
        existing={'artifact':artifact,'aggregation_role':'supplemental_nested_or_unmeasured','category':'dispatcher_elapsed'}
        base={'cost_records':[existing],'inputs':[artifact],'input_read_errors':[],'category_observations':{},
              'unknown_or_opaque_records':[],'max_observed_peak_allocated_bytes':None}
        def glob(pattern):return [p] if pattern=='continuation_queue_v6/attempts/*/status.json' else []
        with patch.object(Path,'glob',side_effect=glob),patch.object(C.S,'desc',return_value=artifact),patch.object(C.S,'checked',return_value=p),patch.object(C.S,'read',return_value={'inclusive_parent_seconds':25.}):
            result=C.add_repair_ledgers(base,NORMALIZE)
        self.assertEqual(len(result['cost_records']),1);self.assertEqual(result['input_read_errors'],[])
        self.assertFalse(result['cost_records'][0]['parent_seconds_added_to_formal_or_primary_work'])

if __name__=='__main__':
    import time
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    if result.wasSuccessful():
        target=Path(__file__).parent/('repair_cost8_preparation_'+str(time.time_ns())+'.json')
        d={'status':'preparation_only','tests_run':result.testsRun,'source':C.S.desc(path),
           'test_source':C.S.desc(Path(__file__)),'actual_collection_started':False,'model_calls':0,'GPU_calls':0,
           'checkpoint_tensor_loads':0,'original_losses_and_unknown_tails_preserved':True,'final_cost_complete':False}
        with target.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2);f.write('\n')
        print(json.dumps(C.S.desc(target)))
    sys.exit(0 if result.wasSuccessful() else 1)
