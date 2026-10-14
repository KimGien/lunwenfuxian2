"""Pure stdlib guard/math tests; never produces a completed-stage certificate."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve()
TARGET = SOURCE.with_name('review_reve_first_seed5_v1.py')
spec = importlib.util.spec_from_file_location('_reve_first_seed5_test_target', TARGET)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class DenyReader:
    def snapshot(self, *args, **kwargs): raise AssertionError('Pending branch attempted an evidence read')


class GuardTests(unittest.TestCase):
    def states(self, n):
        return [{'fit_id':f'{M.CONDITION}/seed_11/fold_{i}', 'status':'training_complete' if i <= n else 'training',
                 'artifact':{'path':f'fold{i}/status.json','sha256':'0'*64}, 'value':{}} for i in range(1, 6)]

    def test_partial_gate_never_opens_supplied_CPU_path(self):
        with patch.object(M, 'Reader', DenyReader), patch.object(M, 'gate', return_value=({}, {}, [], self.states(4))):
            result, _ = M.review(M.WORK/'forbidden_checkpoint.pt', '0'*64)
        self.assertFalse(result['passed']); self.assertFalse(result['actual_CPU_report_read'])
        self.assertEqual(result['completed_statuses_observed'], 4)

    def test_check_only_even_all_five_never_opens_CPU(self):
        with patch.object(M, 'Reader', DenyReader), patch.object(M, 'gate', return_value=({}, {}, [], self.states(5))):
            result, _ = M.review(M.WORK/'forbidden_checkpoint.pt', '0'*64, True)
        self.assertFalse(result['passed']); self.assertFalse(result['actual_CPU_report_read'])

    def test_no_CPU_proof_remains_pending(self):
        with patch.object(M, 'Reader', DenyReader), patch.object(M, 'gate', return_value=({}, {}, [], self.states(5))):
            result, _ = M.review()
        self.assertFalse(result['passed'])

    def test_binary_reader_rejects_before_open(self):
        with self.assertRaisesRegex(ValueError, 'Only workspace JSON and Python'):
            M.Reader().snapshot(M.WORK/'never_open.pt')

    def test_architecture_adaptation_retains_pure_policy_objects(self):
        review_fit = M.semantic_functions()
        self.assertEqual(review_fit.__globals__['CONDITION'], M.CONDITION)
        self.assertIs(review_fit.__globals__['decision'], M.H.decision)
        self.assertIs(review_fit.__globals__['lr_multiplier'], M.H.lr_multiplier)
        self.assertEqual(M.H.CONDITION, 'CSBrain-lastblock-finetune')

    def test_pristine_formula_and_wrong_counter(self):
        expected = {'forward':2339, 'encoder_forward':10963, 'encode':10963,
                    'encoder_windows':82776, 'backward':1107, 'optimizer_step':164}
        value = {'attempts':[{'status':'complete'}], 'recorded_physical_call_subtotals':
                 {k+s:n for k,n in expected.items() for s in ('_started','_returned')}}
        fit = {'train_ids':list(range(27)), 'val_ids':list(range(3))}
        with patch.object(M.B, 'verify_traces', return_value=copy.deepcopy(value)):
            out = M.physical(fit, {'epochs':41}, {}, [], {})
        self.assertTrue(out['single_complete_attempt_math_checked'])
        value['recorded_physical_call_subtotals']['backward_returned'] += 1
        with patch.object(M.B, 'verify_traces', return_value=value), self.assertRaises(ValueError):
            M.physical(fit, {'epochs':41}, {}, [], {})

    def test_minimum40_patience_and_epoch200_cap(self):
        policy = {'minimum_improvement':.0001, 'initial_epochs':120, 'final_epochs':200,
                  'minimum_stop_epoch':40, 'patience':30}
        self.assertFalse(M.H.decision(policy, [1.]*39, False)['stop'])
        self.assertTrue(M.H.decision(policy, [1.]*40, False)['stop'])
        result = M.H.decision(policy, [2.-i*.005 for i in range(200)], True)
        self.assertEqual(result['best_epoch'],200); self.assertTrue(result['stop'])
        self.assertFalse(result['extend']); self.assertEqual(result['reason'],'epoch_cap')

    def test_extension_one_percent_versus_subthreshold(self):
        policy = {'minimum_improvement':.0001, 'initial_epochs':120, 'final_epochs':200,
                  'minimum_stop_epoch':40, 'patience':30}
        good = [1.5]*80+[1.]*20+[.98]*19+[.979]
        bad = [1.5]*80+[1.]*20+[.999]*19+[.998]
        self.assertTrue(M.H.decision(policy, good, False)['extend'])
        self.assertFalse(M.H.decision(policy, bad, False)['extend'])
        self.assertTrue(M.H.decision(policy, bad, False)['stop'])


def main():
    start=time.perf_counter(); stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests))
    def binding(path): return {'path':path.relative_to(M.WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    value={'schema':'REVE_first_seed5_guard_stdlib_tests_v1','source':binding(SOURCE),'target':binding(TARGET),
           'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
           'elapsed_seconds':time.perf_counter()-start,'output':stream.getvalue(),
           'synthetic_completed_stage_certificate_written':False,'actual_CPU_report_read':False,
           'tensor_model_GPU_binary_raw_test_process_operations':0,'torch_imported':'torch' in sys.modules}
    target=SOURCE.parent/('REVE_first_seed5_guard_tests_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as handle: json.dump(value,handle,indent=2,allow_nan=False)
    print(json.dumps({'report':binding(target),'passed':result.wasSuccessful(),'tests':result.testsRun}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
