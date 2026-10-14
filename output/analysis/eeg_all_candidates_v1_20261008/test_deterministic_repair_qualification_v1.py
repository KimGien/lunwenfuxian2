"""Boundary/ownership tests; no torch, model, actual GPU or queue mutation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v1 as Q

class Guards(unittest.TestCase):
    def test_default_zero_operations(self):
        with patch.object(Q,'source_check',side_effect=AssertionError),patch.object(Q,'host_inventory',side_effect=AssertionError),patch.object(Q,'gpu_empty',side_effect=AssertionError),patch.object(Q,'read',side_effect=AssertionError):
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout):Q.main([])
        value=json.loads(stdout.getvalue())
        self.assertEqual(value['model_calls'],0);self.assertEqual(value['writes'],0)
        self.assertFalse(value['GPU_qualification_complete'])
    def test_cli_conflicting_modes(self):
        with self.assertRaises(ValueError):Q.main(['--run','--worker'])
    def test_cli_requires_explicit_condition(self):
        with self.assertRaises(ValueError):Q.main(['--run'])
    def test_pause_sticky_and_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for name in ('sync_boundary','sync_boundary_v2','sync_boundary_v3'):
                p=out/'execution_acceleration_v1'/name/'user_pause.json';p.parent.mkdir(parents=True)
                for value in ({'paused':True},{'paused':False,'extra':1},{'paused':0}):
                    p.write_text(json.dumps(value),encoding='utf-8')
                    with self.assertRaises(ValueError):Q.check_pause(out)
                p.unlink()
            Q.check_pause(out)
    def test_running_owner_blocks_before_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            q=Path(d);(q/'owner.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'owner still present'):Q.check_boundary(q)
    def test_terminal_strict_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            q=Path(d);p=q/'status.json'
            valid={'status':'required_work_remaining','execution_version':5,'recovery_execution_version':2,
                   'recovery_registration':{'sha256':Q.MANIFEST_SHA}}
            p.write_text(json.dumps(valid));self.assertEqual(Q.check_boundary(q),valid)
            for change in ({'status':'running'},{'execution_version':4},{'recovery_execution_version':1},
                           {'recovery_registration':{'sha256':'wrong'}}):
                p.write_text(json.dumps({**valid,**change}))
                with self.assertRaises(ValueError):Q.check_boundary(q)
    def test_exclusive_intent_and_root(self):
        with tempfile.TemporaryDirectory() as d,patch.object(Q,'ROOT',Path(d)):
            p=Path(d)/'intent.json';Q.exclusive(p,{'x':1})
            with self.assertRaises(FileExistsError):Q.exclusive(p,{'x':2})
            self.assertEqual(Q.read(p),{'x':1})
            with self.assertRaises(ValueError):Q.exclusive(Path(d).parent/'escape.json',{})
    def test_birth_identity_required(self):
        allowed=[{'pid':22,'birth':'100'}]
        Q.admit_host([{'pid':22,'birth':'100'}],allowed)
        for items in ([{'pid':22,'birth':'101'}],[{'pid':23,'birth':'100'}],None):
            with self.assertRaises(ValueError):Q.admit_host(items,allowed)
    def test_gpu_inventory_empty_single_only(self):
        xml='<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>'
        with patch.object(Q.sys,'platform','win32'),patch.object(Q.subprocess,'run',return_value=types.SimpleNamespace(stdout=xml)):
            self.assertEqual(Q.gpu_empty()['context_count'],0)
        for xml in ('<nvidia_smi_log/>','<nvidia_smi_log><gpu><processes><process_info><pid>9</pid><type>G</type></process_info></processes></gpu></nvidia_smi_log>',
                    '<nvidia_smi_log><gpu/><gpu/></nvidia_smi_log>'):
            with patch.object(Q.sys,'platform','win32'),patch.object(Q.subprocess,'run',return_value=types.SimpleNamespace(stdout=xml)):
                with self.assertRaises(ValueError):Q.gpu_empty()
    def test_gpu_query_failure_blocks(self):
        with patch.object(Q.sys,'platform','win32'),patch.object(Q.subprocess,'run',side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):Q.gpu_empty()
    def test_private_builder_globals_unchanged(self):
        def build(condition,*,seed=11,device='cuda',window_chunk=8):return condition,seed,device,window_chunk
        original=build.__globals__.copy()
        f=types.SimpleNamespace(build_model=build)
        new=Q.isolated_builder(f)
        self.assertIs(new.__code__,build.__code__);self.assertIs(f.build_model,build)
        self.assertEqual(new('x'),('x',11,'cuda',8));self.assertEqual(build.__globals__,original)
    def test_worker_claim_rejects_wrong_source(self):
        with patch.object(Q,'desc',return_value={'sha256':'real'}):
            with self.assertRaises(ValueError):Q.validate_worker({'condition':Q.CONDITIONS[0],'source':{'sha256':'wrong'}},Q.CONDITIONS[0])
    def test_windows_worker_roles(self):
        controller={'pid':10,'birth':'1','parent':3,'command':'controller'}
        launcher={'pid':20,'birth':'2','parent':10,'command':'qualify_deterministic_repairs_v1 --worker CodeBrain-lastblock-finetune'}
        actual={'pid':21,'birth':'3','parent':20,'command':launcher['command']}
        admission={'condition':Q.CONDITIONS[0],'source':{'key':'same'},'terminal':{'key':'same'},'sources':[],
                   'allowed_host':[controller]}
        owner={'condition':Q.CONDITIONS[0],'native_roles':[controller],'pid':10}
        with patch.object(Q.sys,'platform','win32'),patch.object(Q.os,'getpid',return_value=21),patch.object(Q,'desc',return_value={'key':'same'}),patch.object(Q,'check_pause'),patch.object(Q,'check_boundary'),patch.object(Q,'source_check'),patch.object(Q,'read',return_value=owner),patch.object(Q,'host_inventory',return_value=[controller,launcher,actual]):
            value=Q.validate_worker(admission,Q.CONDITIONS[0]);self.assertEqual(len(value['allowed_host']),3)
            with patch.object(Q,'host_inventory',return_value=[controller,launcher,actual,{'pid':77,'birth':'8','command':'another train'}]):
                with self.assertRaises(ValueError):Q.validate_worker(admission,Q.CONDITIONS[0])
    def test_linux_worker_host_roles(self):
        controller={'pid':10,'birth':'1','parent':3,'command':'controller'}
        host={'pid':40,'birth':'4','parent':10,'command':'wsl.exe qualify_deterministic_repairs_v1 --worker --admission-sha256 abc'}
        admission={'condition':Q.CONDITIONS[1],'source':{'key':'same'},'terminal':{'key':'same'},'sources':[],'allowed_host':[controller]}
        owner={'condition':Q.CONDITIONS[1],'native_roles':[controller],'pid':10}
        with patch.object(Q.sys,'platform','linux'),patch.object(Q,'desc',return_value={'key':'same'}),patch.object(Q,'check_pause'),patch.object(Q,'check_boundary'),patch.object(Q,'source_check'),patch.object(Q,'read',return_value=owner),patch.object(Q,'host_inventory',return_value=[controller,host]):
            self.assertEqual(len(Q.validate_worker(admission,Q.CONDITIONS[1])['allowed_host']),2)
    def test_healthy_queue_run_refuses_without_queries(self):
        with patch.object(Q,'check_pause'),patch.object(Q,'check_boundary',side_effect=ValueError('owner still present')),patch.object(Q,'source_check',side_effect=AssertionError),patch.object(Q,'host_inventory',side_effect=AssertionError),patch.object(Q,'gpu_empty',side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError,'owner still present'):Q.run_controller(Q.CONDITIONS[0])

if __name__=='__main__':
    import hashlib,time,sys
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    if result.wasSuccessful():
        report=Q.WORK/'output/analysis/eeg_all_candidates_v1_20261008'/('deterministic_repair_qualification_preparation_'+str(time.time_ns())+'.json')
        value={'status':'preparation_only','source':Q.desc(Q.SOURCE),'test_source':Q.desc(Path(__file__)),
               'tests_run':result.testsRun,'errors':[],'model_calls':0,'GPU_calls':0,'checkpoint_reads':0,
               'current_queue_mutated':False,'GPU_qualification_complete':False,'new_formal_execution_required':True}
        with report.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
        print(json.dumps(Q.desc(report)))
    sys.exit(0 if result.wasSuccessful() else 1)
