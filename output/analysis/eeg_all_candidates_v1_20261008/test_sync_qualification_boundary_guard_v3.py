"""Synthetic stdlib lifecycle tests; never reads a live process or tensor/GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

REPORT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_guard3_fake_only', REPORT / 'sync_qualification_boundary_guard_v3.py')
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
VERIFY_CONTROL = G.verify_control


class Resume:
    def __init__(self, native): self.native = native
    def __call__(self, handle):
        self.native.calls.append(('resume', handle))
        if self.native.ambiguous_resume: raise RuntimeError('resume effect may have occurred before backend error')
        return 0


class Native:
    def __init__(self, fixture):
        self.f = fixture; self.calls = []; self.children_done = False; self.extra = set(); self.codes = [0, 0]
        self.ambiguous_resume = False; self.ambiguous_suspend = False; self.ntdll = types.SimpleNamespace(NtResumeProcess=Resume(self))
        self.parents_map = {999: 88}
    def capture(self, role, *, arm=False):
        self.calls.append(('capture', role, arm))
        value = copy.deepcopy(self.f.roles[role])
        if self.f.bad_capture == role: value['creation_100ns'] = 'bad'
        return {**value, 'handle': role}
    def exited(self, ref): return self.children_done if ref['role'] != 'dispatcher' else False
    def exit_code(self, ref): return self.codes[0 if ref['role'] == 'launcher' else 1]
    def descendants(self): return self.extra | (set() if self.children_done else {12, 13})
    def parents(self): return dict(self.parents_map)
    def suspend_dispatcher(self, ref):
        assert ref['role'] == 'dispatcher'; self.calls.append(('suspend', ref['role']))
        if self.ambiguous_suspend:raise RuntimeError('uncertain successful suspend effect')
    def close(self, ref): self.calls.append(('close', ref['role']))
    def terminate_dispatcher(self, ref): raise AssertionError('Formal process termination forbidden')


class Process:
    def __init__(self, f, behavior, number): self.f=f;self.behavior=behavior;self.pid=1000+number;self.returncode=None;self.seen=False
    def poll(self):
        if self.returncode is not None: return self.returncode
        if self.behavior.get('live'): return None
        if not self.seen:
            self.seen=True
            if self.behavior.get('pause'): self.f.pause(); return None
            if self.behavior.get('lease'): self.f.save(self.f.probe_lease, {'fake':'residue'})
            if self.behavior.get('orphan'): self.f.native.parents_map[self.pid+9000]=self.pid
        self.returncode=self.behavior.get('code',0);return self.returncode


class Guardian3Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='guard3_fixture_',dir=REPORT)
        self.work=Path(self.temp.name).resolve();self.out=self.work/'out';self.root=self.out/'execution_acceleration_v1/sync_boundary_v3'
        self.root.mkdir(parents=True);self.report=self.work/'report';self.report.mkdir();self.q5=self.out/'q5';self.q5.mkdir()
        self.source=self.report/'guard.py';self.source.write_text('# fixture source\n')
        self.probe_lease=self.out/'execution_acceleration_v1/sync_qualification_owner.json'
        self.roles={role:{'role':role,'pid':11+i,'parent_pid':10+i,'creation_100ns':str(100+i),
                          'created_local':'fixture','image':'fixture-python.exe'} for i,role in enumerate(G.ROLES)}
        self.native=Native(self);self.clock=0.0;self.processes=[];self.behaviors=[{},{}];self.contexts=[]
        self.wait_hook=None;self.bad_capture=None;self.complete=True;self.save_failure_once=False;self.control_bad=False
        self.owner={'id':'actual-fixture-owner','pid':11,'child_pid':12}
        self.save(self.q5/'owner.json',self.owner);self.save(self.q5/'attempt.json',{'status':'running'})
        self.save(self.root/'inspection.json',{'source':self.desc(self.source),'roles':self.roles})
        self.save(self.root/'registration.json',{})
        self.graphics_evidence={'schema':'sync_v3_observed_external_qq_gpu_process_v1',
            'process':{'ProcessId':15304,'ParentProcessId':34776,'ExecutablePath':'C:\\Program Files\\Tencent\\QQNT\\QQ.exe',
                       'CommandLine':'"C:\\Program Files\\Tencent\\QQNT\\QQ.exe" --type=gpu-process --fixture'},
            'native_creation_100ns':'134360680926134212','nvidia_observed_type':'C+G',
            'nvidia_observed_path':'C:/Program Files/Tencent/QQNT/QQ.exe','model_or_process_control_operations':0}
        self.graphics_path=self.root/'qq_graphics.json';self.save(self.graphics_path,self.graphics_evidence)
        self.expected=self.desc(self.root/'inspection.json')
        self.control={'expected':self.expected,'roles':self.roles,'queue_owner':self.desc(self.q5/'owner.json'),
                      'registration':self.desc(self.q5/'attempt.json'),'recovery_registration':self.desc(self.q5/'attempt.json'),
                      'recovery_closure':self.desc(self.q5/'attempt.json'),'protected_sources':[self.desc(self.source)],
                      'experimental_sources':[self.desc(self.source)]}
        self.identity={'pid':999,'parent_pid':88,'creation_100ns':'777','source':self.desc(self.source)}
        self.fit=self.out/'fit';self.fit.mkdir();self.save(self.fit/'train_history.json',[{'epoch':1}]);(self.fit/'latest.pt').write_bytes(b'NOT A TENSOR: synthetic fixture')
        self.artifacts=[self.desc(self.fit/'train_history.json') for _ in range(90)]
        storage=types.SimpleNamespace(WORK=self.work,OUT=self.out,REPORT=self.report,desc=self.desc,checked=self.checked,
                                      read=self.read,save=self.save,now=lambda:'fixture-time')
        registry=types.SimpleNamespace(fit_directory=lambda *a:self.fit,registration_descriptor=lambda:self.desc(self.q5/'attempt.json'),
                                       load_registration=lambda:{'folds':{'fold_5':{'train':['a'],'val':['b'],'test':['c']}}})
        self.patchers=[patch.object(G,'S',storage),patch.object(G,'R',registry),patch.object(G,'SOURCE',self.source),
          patch.object(G,'ROOT',self.root),patch.object(G,'REGISTRATION',self.root/'registration.json'),
          patch.object(G,'Q5',types.SimpleNamespace(ROOT=self.q5)),
          patch.object(G,'EXTERNAL_GRAPHICS',self.graphics_path),
          patch.object(G,'backend',lambda roles:self.native),patch.object(G,'guardian_identity',lambda native:copy.deepcopy(self.identity)),
          patch.object(G,'verify_control',self.verify_control),patch.object(G,'boundary',lambda roles=None,**kwargs:(copy.deepcopy(self.owner),self.q5/'attempt.json',{})),
          patch.object(G,'completed_artifacts',self.completed),patch.object(G,'gpu_observation',lambda:{'contexts':copy.deepcopy(self.contexts)}),
          patch.object(G,'time',types.SimpleNamespace(time_ns=__import__('time').time_ns,monotonic=lambda:self.clock,sleep=self.tick)),
          patch.object(G,'subprocess',types.SimpleNamespace(Popen=self.spawn,STDOUT=-2)),
          patch.object(G.G1,'current_process_identity',self.current_identity)]
        for p in self.patchers:p.start()

    def tearDown(self):
        for p in reversed(self.patchers):p.stop()
        assert self.work.parent==REPORT.resolve() and self.work.name.startswith('guard3_fixture_');self.temp.cleanup()
    def desc(self,p):
        p=Path(p).resolve();assert p.is_relative_to(self.work)
        return {'path':p.relative_to(self.work).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    def checked(self,d):
        p=self.work/d['path'];self.assertEqual(self.desc(p),d);return p
    def read(self,p):return json.loads(Path(p).read_text(encoding='utf-8'))
    def save(self,p,v,*,immutable=False):
        if self.save_failure_once and v.get('phase')=='qualification_finished_same_Q5_resumed':
            self.save_failure_once=False;raise OSError('synthetic post-resume disk write failure')
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x' if immutable else 'w',encoding='utf-8') as f:json.dump(v,f)
    def verify_control(self,*a):
        if self.control_bad:raise ValueError('source80 changed')
        return copy.deepcopy(self.control)
    def completed(self):
        if not self.complete:raise ValueError('actual15 incomplete')
        return copy.deepcopy(self.artifacts)
    def current_identity(self,native,identity):
        if identity!=self.identity:raise ValueError('guardian birth/source changed')
    def tick(self,seconds):
        self.clock+=max(seconds,600.1) if self.processes and self.behaviors[-1].get('fast_timeout') else seconds
        if self.wait_hook:self.wait_hook()
        else:self.native.children_done=True
    def pause(self,name='sync_boundary_v3',value=None):
        self.save(self.out/'execution_acceleration_v1'/name/'user_pause.json',{'paused':True} if value is None else value)
    def spawn(self,args,**kwargs):
        self.assertTrue(self.native.children_done);self.assertTrue(any(c[0]=='suspend' for c in self.native.calls))
        self.assertEqual(kwargs.get('creationflags'),0x08000000)
        p=Process(self,self.behaviors[len(self.processes)],len(self.processes));self.processes.append(p);return p
    def arm(self):return G.arm(self.expected)
    def state(self):return self.read(self.root/'state.json')
    def resumes(self):return [c for c in self.native.calls if c[0]=='resume']

    def test_success_same_parent_once_sequential_and_observed_timing(self):
        result=self.arm();self.assertTrue(result['passed']);self.assertTrue(result['all_qualifications_passed'])
        self.assertEqual(self.resumes(),[('resume','dispatcher')]);self.assertEqual(len(self.processes),2)
        self.assertEqual([c for c in self.native.calls if c[0]=='suspend'],[('suspend','dispatcher')])
        self.assertTrue(all(not c[2] for c in self.native.calls if c[0]=='capture' and c[1]!='dispatcher'))
        self.assertEqual(result['formal_process_terminate_calls'],0);self.assertEqual(result['formal_queue_spawn_calls'],0)
        self.assertFalse(result['actual_sync_optimization_applied']);self.assertEqual(len(result['probe_intervals']),2)
        self.assertGreaterEqual(result['hold_seconds'],result['children_wait_seconds'])
    def test_duplicate_lease_refuses_before_capture_or_second_suspend(self):
        G.write_exclusive(self.root/'guardian_owner.json',{'fixture':True})
        with self.assertRaises(FileExistsError):self.arm()
        self.assertFalse(self.native.calls);self.assertFalse(self.processes)
    def test_changed_birth_blocks_before_suspend(self):
        self.bad_capture='gpu_child'
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.resumes());self.assertFalse([c for c in self.native.calls if c[0]=='suspend'])
    def test_child_nonzero_blocks_probe_but_allows_clean_parent_to_handle_failure(self):
        self.native.codes=[0,1]
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.processes);self.assertEqual(len(self.resumes()),1)
    def test_incomplete_or_short_full15_never_probes(self):
        self.artifacts=self.artifacts[:-1]
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.processes)
    def test_source80_changes_keep_parent_held(self):
        self.wait_hook=lambda:(setattr(self,'control_bad',True),setattr(self.native,'children_done',True))
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.resumes());self.assertTrue(self.state()['dispatcher_suspended'])
    def test_known_graphics_defers_and_resumes_without_probes(self):
        self.contexts=[{'pid':91,'type':'G','process_name':'desktop'}]
        value=self.arm();self.assertFalse(self.processes);self.assertEqual(len(self.resumes()),1)
        self.assertFalse(value['all_qualifications_passed']);self.assertIn('deferred_qualification',value)
    def test_unknown_gpu_compute_blocks_parent_resume(self):
        self.contexts=[{'pid':91,'type':'C','process_name':'other.py'}]
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.processes or self.resumes())
    def qq_actual(self,expected):
        return {**expected,'created_local':'fixture','image':'C:\\Program Files\\Tencent\\QQNT\\QQ.exe'}
    def test_exact_source_bound_qq_defers_both_probes_and_resumes_same_parent(self):
        self.contexts=[{'pid':15304,'type':'C+G','process_name':'C:/Program Files/Tencent/QQNT/QQ.exe'}]
        with patch.object(G,'external_native_identity',self.qq_actual):value=self.arm()
        self.assertFalse(self.processes);self.assertEqual(len(self.resumes()),1)
        self.assertFalse(value['all_qualifications_passed'])
        deferred=self.read(self.checked(value['deferred_qualification']))
        self.assertEqual(deferred['model_constructions'],0);self.assertFalse(deferred['GPU_speedup_claimed'])
    def test_qq_reused_pid_or_changed_birth_does_not_allow_training_resume(self):
        self.contexts=[{'pid':15304,'type':'C+G','process_name':'C:/Program Files/Tencent/QQNT/QQ.exe'}]
        with patch.object(G,'external_native_identity',lambda e:{**self.qq_actual(e),'creation_100ns':'different'}):
            with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.processes or self.resumes())
    def test_qq_wrong_pid_or_same_filename_different_path_rejected(self):
        for context in ({'pid':99,'type':'C+G','process_name':'C:/Program Files/Tencent/QQNT/QQ.exe'},
                        {'pid':15304,'type':'C+G','process_name':'C:/other/QQ.exe'},
                        {'pid':15304,'type':'C','process_name':'C:/Program Files/Tencent/QQNT/QQ.exe'}):
            with self.subTest(context=context),patch.object(G,'external_native_identity',side_effect=AssertionError('Must reject before native read')):
                with self.assertRaises(ValueError):G.training_gpu_scope({'contexts':[context]})
    def test_qq_evidence_without_exact_gpu_process_flag_refused(self):
        self.graphics_evidence['process']['CommandLine']='QQ.exe --type=renderer'
        self.save(self.graphics_path,self.graphics_evidence)
        with self.assertRaises(ValueError):G.external_graphics_evidence()
    def test_probe_monitor_allows_only_current_qualifier_descendants(self):
        process=types.SimpleNamespace(pid=700);self.native.parents_map[701]=700
        monitor={'samples':[],'external_context_observed':False,'observation_errors':[],'observer_query_seconds_observed':0.0}
        self.contexts=[{'pid':701,'type':'C','process_name':'actual-qualifier-python'}]
        G.observe_probe_gpu(self.native,process,monitor)
        self.assertFalse(monitor['external_context_observed'])
        self.contexts=[{'pid':13,'type':'C','process_name':'old-CSBrain'}]
        G.observe_probe_gpu(self.native,process,monitor)
        self.assertTrue(monitor['external_context_observed'])
    def test_probe_monitor_query_failure_remains_unknown_not_clean(self):
        monitor={'samples':[],'external_context_observed':False,'observation_errors':[],'observer_query_seconds_observed':0.0}
        with patch.object(G,'gpu_observation',side_effect=OSError('inventory unavailable')):
            G.observe_probe_gpu(self.native,types.SimpleNamespace(pid=700),monitor)
        self.assertFalse(monitor['samples']);self.assertEqual(len(monitor['observation_errors']),1)
    def test_transient_graphics_during_probe_rejects_clean_performance_despite_code0(self):
        original=G.observe_probe_gpu
        def transient(native,process,monitor):
            self.contexts=[{'pid':15304,'type':'C+G','process_name':'QQ graphics'}]
            original(native,process,monitor)
            self.contexts=[]
        with patch.object(G,'observe_probe_gpu',transient):value=self.arm()
        self.assertTrue(all(row['qualifier_report_passed'] for row in value['outcomes']))
        self.assertFalse(value['all_qualifications_passed']);self.assertEqual(len(self.resumes()),1)
        self.assertTrue(all(not row['performance_sampled_clean'] for row in value['outcomes']))
    def test_dynamic_control_registration_binds_inspection_and_exact_sources(self):
        expected={'schema':'sync_q5_dynamic_boundary_inspection_v3','source':self.desc(self.source),
                  'queue_owner':self.desc(self.q5/'owner.json'),'registration':self.control['registration'],
                  'queue_attempt':self.desc(self.q5/'attempt.json'),
                  'recovery_registration':self.control['recovery_registration'],'recovery_closure':self.control['recovery_closure'],
                  'native_birth_observation':self.desc(self.q5/'attempt.json'),'roles':self.roles}
        self.save(self.root/'inspection.json',expected);binding=self.desc(self.root/'inspection.json')
        lineage={'base_queue_registration':expected['registration'],'registration':expected['recovery_registration'],
                 'value':{'sources':[self.desc(self.source)]}}
        owner={**self.owner,'recovery_closure':expected['recovery_closure']}
        (self.root/'registration.json').unlink()
        with patch.object(G,'boundary',lambda *a,**k:(owner,self.q5/'attempt.json',lineage)), \
             patch.object(G,'experimental_sources',lambda:[self.desc(self.source)]):
            registered=G.register_control(binding);self.assertEqual(VERIFY_CONTROL(registered)['roles'],self.roles)
            with self.assertRaises(FileExistsError):G.register_control(binding)
            value=self.read(self.root/'registration.json');value['roles']['gpu_child']['creation_100ns']='bad'
            self.save(self.root/'registration.json',value)
            with self.assertRaises(ValueError):VERIFY_CONTROL(self.desc(self.root/'registration.json'))
    def test_stale_shared_lease_blocks_before_probe_and_resume(self):
        self.save(self.probe_lease,{'foreign':True})
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.processes or self.resumes())
    def test_cpu_orphan_retains_parent_hold(self):
        self.behaviors=[{'orphan':True},{}]
        with self.assertRaises(ValueError):self.arm()
        self.assertEqual(len(self.processes),1);self.assertFalse(self.resumes())
    def test_alive_cpu_probe_timeout_blocks_even_idle_gpu(self):
        self.behaviors=[{'live':True},{'fast_timeout':True}]
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.resumes());self.assertTrue(self.state()['dispatcher_suspended'])
    def test_declined_qualification_is_not_applied_or_allpassed(self):
        self.behaviors=[{'code':2},{}];v=self.arm()
        self.assertFalse(v['all_qualifications_passed']);self.assertTrue(v['outcomes'][0]['qualification_declined'])
        self.assertFalse(v['actual_sync_optimization_applied'])
    def test_failed_clean_probe_skips_second_and_resumes_once(self):
        self.behaviors=[{'code':1},{}];v=self.arm()
        self.assertEqual(len(self.processes),1);self.assertEqual(len(self.resumes()),1);self.assertTrue(v['outcomes'][0]['execution_failed'])
    def test_pause_before_arm_blocks_all_actions(self):
        self.pause('sync_boundary')
        with self.assertRaises(G.UserPauseRequested):self.arm()
        self.assertFalse(self.native.calls or self.processes)
    def test_both_old_markers_and_new_marker_revoke_waiting_resume(self):
        for name in ('sync_boundary','sync_boundary_v2','sync_boundary_v3'):
            with self.subTest(name=name):
                self.wait_hook=lambda:self.pause(name)
                v=self.arm();self.assertTrue(v['automatic_resume_revoked']);self.assertFalse(self.processes or self.resumes())
                (self.out/'execution_acceleration_v1'/name/'user_pause.json').unlink();(self.root/'guardian_owner.json').unlink()
                self.native.calls.clear()
    def test_malformed_pause_marker_fails_closed(self):
        self.wait_hook=lambda:self.pause(value={'paused':False})
        v=self.arm();self.assertTrue(v['automatic_resume_revoked']);self.assertFalse(self.resumes())
    def test_pause_during_active_probe_is_polled_without_waiting1200(self):
        self.behaviors=[{'pause':True},{}];v=self.arm()
        self.assertEqual(len(self.processes),1);self.assertFalse(self.resumes());self.assertTrue(v['automatic_resume_revoked'])
        self.assertLess(self.clock,3)
    def test_successful_resume_then_write_failure_never_resumes_again(self):
        self.save_failure_once=True
        with self.assertRaises(OSError):self.arm()
        self.assertEqual(len(self.resumes()),1);self.assertFalse(self.state()['dispatcher_suspended'])
    def test_ambiguous_native_resume_is_never_retried_or_claimed_held(self):
        self.native.ambiguous_resume=True
        with self.assertRaises(RuntimeError):self.arm()
        self.assertEqual(len(self.resumes()),1);self.assertIsNone(self.state()['dispatcher_suspended'])
        self.assertTrue(self.state()['automatic_resume_retry_forbidden'])
    def test_ambiguous_native_suspend_is_not_retried_or_claimed_absent(self):
        self.native.ambiguous_suspend=True
        with self.assertRaises(RuntimeError):self.arm()
        self.assertEqual(len([c for c in self.native.calls if c[0]=='suspend']),1)
        self.assertFalse(self.resumes());self.assertIsNone(self.state()['dispatcher_suspended'])
    def test_live_verifier_rejects_wrong_closure_and_requires_current_state(self):
        # Run the real verifier during fake process launch; the normal lifecycle
        # already exercises it once before the first probe.
        self.arm();p=next(self.root.glob('boundary_*.json'));v=self.read(p);v['recovery_closure']={'path':'bad','sha256':'bad'}
        self.save(p,v)
        with self.assertRaises(ValueError):G.verify_live_boundary(self.desc(p))
    def test_resume_precheck_rejects_new_user_pause(self):
        self.pause()
        with self.assertRaises(G.UserPauseRequested):G.resume_dispatcher(self.native,self.native.capture('dispatcher'))
        self.assertFalse(self.resumes())


class NativeRightsTests(unittest.TestCase):
    def fixture(self, role):
        backend=object.__new__(G.DynamicNative);expected={'role':role,'pid':7,'parent_pid':6,'creation_100ns':'123456789'}
        backend.roles={role:expected};backend.parents=lambda:{7:6};calls=[]
        def times(handle,*values):values[0]._obj.value=123456789;return True
        def image(handle,flags,buffer,size):buffer.value='fixture.exe';return True
        backend.kernel=types.SimpleNamespace(OpenProcess=lambda rights,*args:(calls.append(('open',rights)) or 99),
            GetProcessTimes=times,QueryFullProcessImageNameW=image,CloseHandle=lambda h:calls.append(('close',h)))
        return backend,calls
    def test_dispatcher_handle_has_suspend_only_not_terminate_right(self):
        backend,calls=self.fixture('dispatcher');backend.capture('dispatcher',arm=True)
        self.assertTrue(calls[0][1]&0x0800);self.assertFalse(calls[0][1]&0x0001)
    def test_child_readonly_and_child_arm_rejected(self):
        backend,calls=self.fixture('gpu_child');backend.capture('gpu_child')
        self.assertFalse(calls[0][1]&(0x0800|0x0001))
        with self.assertRaises(ValueError):backend.capture('gpu_child',arm=True)
        self.assertEqual(len(calls),1)
    def test_full100ns_change_rejects_and_closes_handle(self):
        backend,calls=self.fixture('dispatcher');backend.roles['dispatcher']['creation_100ns']='123456788'
        with self.assertRaises(ValueError):backend.capture('dispatcher')
        self.assertEqual(calls[-1],('close',99))


if __name__=='__main__':unittest.main(verbosity=2)
