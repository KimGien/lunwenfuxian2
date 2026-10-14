"""Independent recovery3 control tests, no native/process/model execution."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout

SOURCE = Path(__file__).resolve(); A = SOURCE.parent; WORK = SOURCE.parents[3]
ENTRY = WORK / 'tridep_ccml/eeg_all_candidates/recover_q5_after_process_loss_v3.py'
spec = importlib.util.spec_from_file_location('independent_recovery3_control', ENTRY)
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)


class FakeBackend:
    def __init__(self, fail=None): self.events=[]; self.fail=fail
    def __getattr__(self, name):
        def action(*args):
            self.events.append(name)
            if self.fail == (name, self.events.count(name)): raise ValueError('injected interruption')
            return {'synthetic': name}
        return action


class IndependentControlTests(unittest.TestCase):
    def test_order_before_owner_removed_and_no_runtime_retry(self):
        b=FakeBackend();E.prepare_recovery({'synthetic':'loss'},b)
        self.assertEqual(b.events,['no_pause','verify','intent','no_pause','absence','register','closure','recheck','no_pause','absence','no_pause','recheck','remove_owner','no_pause','ready'])
        for event,index in [('intent',1),('absence',1),('register',1),('closure',1),('recheck',1),('absence',2),('recheck',2)]:
            b=FakeBackend((event,index))
            with self.assertRaises(ValueError):E.prepare_recovery({},b)
            self.assertNotIn('remove_owner',b.events)
            self.assertNotIn('ready',b.events)

    def test_pause_at_each_checkpoint_never_leads_ready(self):
        for nth in range(1,6):
            b=FakeBackend(('no_pause',nth))
            with self.assertRaises(ValueError):E.prepare_recovery({},b)
            self.assertNotIn('ready',b.events)
            if nth<5:self.assertNotIn('remove_owner',b.events)

    def test_permanent_intent_exclusive_on_disk(self):
        with tempfile.TemporaryDirectory(dir=A,prefix='_independent_recovery3_') as directory:
            path=Path(directory)/'intent.json'
            first=E.write_new(path,{'once':True})
            with self.assertRaises(FileExistsError):E.write_new(path,{'second':True})
            self.assertEqual(E.desc(path),first)

    def test_every_present_marker_including_malformed_blocks(self):
        with tempfile.TemporaryDirectory(dir=A,prefix='_independent_recovery3_') as directory:
            marker=Path(directory)/'pause.json'
            for content in ('{"paused":true}','{"paused":false}','malformed'):
                marker.write_text(content,encoding='utf8')
                with patch.object(E,'PAUSE_PATHS',(marker,)),self.assertRaises(ValueError):E.assert_not_paused()

    def test_default_main_never_inspects_or_launches(self):
        with patch.object(E,'run',side_effect=AssertionError('no run')),patch.object(E,'native_birth',side_effect=AssertionError('no native')),patch.object(E,'gpu_and_wsl_observation',side_effect=AssertionError('no GPU')),redirect_stdout(io.StringIO()):
            self.assertEqual(E.main([]),2)

    def test_evaluation_is_unconditionally_disabled(self):
        with self.assertRaisesRegex(RuntimeError,'permanently disabled'):E.disable_evaluation({'id':'fake'})

    def test_strict_cpu8_full135405_gate_rejects_other_versions_and_bool(self):
        auditor={'path':'fake.py','sha256':'a'*64}
        value={'schema_version':8,'auditor_source':auditor,'all_135_finetune_checkpoints_verified':True,'errors':[],'deferred_completed_fits':[],'pending_fits':[],'status':'all_finetune_verified','registered_finetune_fits':135,'training_complete_finetune_observed':135,'verified_completed_finetune_fits':135,'checkpoint_loads':405}
        E.validate_cpu_gate(value,auditor)
        for key,bad in [('schema_version',6),('schema_version',True),('checkpoint_loads',404),('checkpoint_loads',True),('verified_completed_finetune_fits',134),('pending_fits',[{}]),('errors',[{}]),('all_135_finetune_checkpoints_verified',1)]:
            with self.subTest(key=key,bad=bad),self.assertRaises(ValueError):E.validate_cpu_gate({**value,key:bad},auditor)

    def rows(self):
        values=json.loads((E.OUT/'registration.json').read_text('utf8'))['fits']
        rows=[]
        for r in values:
            condition,seed,fold=(r[k] for k in ('condition','seed','fold'))
            state='training_complete'
            if condition in E.BLOCKED_CONDITIONS:state='blocked'
            elif condition==E.TARGET_CONDITION and seed==33:state='training' if fold==1 else 'ready'
            rows.append({'condition':condition,'seed':seed,'fold':fold,'status':state})
        return rows

    def test_only_original_target_and_30_blocked_not_false_plan_complete(self):
        rows=self.rows()
        self.assertEqual(E.validate_training_scope(rows,before_launch=True),{'training_complete':325,'remaining_registered_repair_fits':30,'target_completed':10})
        for r in rows:
            if r['condition']==E.TARGET_CONDITION:r['status']='training_complete'
        self.assertEqual(E.validate_training_scope(rows,before_launch=False),{'training_complete':330,'remaining_registered_repair_fits':30,'target_completed':15})
        for r in rows:r['status']='training_complete'
        with self.assertRaises(ValueError):E.validate_training_scope(rows,before_launch=False)

    def test_changed_fit_identity_completed_row_and_blocked_state_reject(self):
        for which in ('duplicate','bool','completed','blocked'):
            rows=self.rows()
            if which=='duplicate':rows[-1]=deepcopy(rows[0])
            elif which=='bool':rows[0]['seed']=True
            elif which=='completed':next(r for r in rows if r['condition']==E.TARGET_CONDITION and r['seed']==11)['status']='ready'
            else:next(r for r in rows if r['condition'] in E.BLOCKED_CONDITIONS)['status']='ready'
            with self.subTest(which=which),self.assertRaises(ValueError):E.validate_training_scope(rows,before_launch=True)

    def test_recovery_absence_rejects_other_worker_old_pid_and_wrong_birth(self):
        ids=[{'pid':p} for p in (1,2,3,4)];loss={'sha256':'a'*64}
        row={'pid':10,'parent_pid':11,'creation_time_100ns':123,'image_path':'C:/python.exe','command_line':E.ENTRY_MODULE+' --run '+loss['sha256']}
        payload={'query_succeeded':True,'queried_pids':[1,2,3,4],'exact_getprocess_alive_pids':[],'rows':[row]}
        self.assertTrue(E.validate_absence_rows(payload,ids,current_pid=10,parent_pid=11,loss_binding=loss)['all_expected_processes_absent'])
        for change in ('other','old','birth','command'):
            bad=deepcopy(payload)
            if change=='other':bad['rows'].append({**row,'pid':12})
            elif change=='old':bad['exact_getprocess_alive_pids']=[1]
            elif change=='birth':bad['rows'][0]['creation_time_100ns']=True
            else:bad['rows'][0]['command_line']='oldrecover --run '+loss['sha256']
            with self.subTest(change=change),self.assertRaises(ValueError):E.validate_absence_rows(bad,ids,current_pid=10,parent_pid=11,loss_binding=loss)


def main():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentControlTests))
    value={'schema':'independent_recovery3_control_tests_v1','passed':result.wasSuccessful(),'reviewer':'/root/loss3_independent_review','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source':E.desc(SOURCE),'reviewed_entry':E.desc(ENTRY),'output':stream.getvalue(),'scope':'stdlib synthetic control faults and exact original matrix JSON only; no actual control/inspection/queue/model/Tensor paths','model_operations':0,'process_actions':0,'GPU_operations':0,'checkpoint_loads':0,'test_probability_reads':0}
    path=A/('independent_recovery3_control_tests_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({'artifact':E.desc(path),'passed':result.wasSuccessful(),'tests_run':result.testsRun,'output':stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
