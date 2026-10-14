"""Fake-only native/process lifecycle tests. No real process control or GPU."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

REPORT = Path(__file__).resolve().parent
SOURCE = REPORT / 'sync_qualification_boundary_guard_v1.py'


def load_guard():
    spec = importlib.util.spec_from_file_location('_fake_only_sync_guard', SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ResumeCall:
    def __init__(self, native): self.native = native
    def __call__(self, handle):
        self.native.calls.append(('resume', handle))
        return self.native.resume_status


class FakeKernel:
    def __init__(self, native): self.native = native
    def OpenProcess(self, rights, inherit, pid): return ('guardian_handle', pid)
    def WaitForSingleObject(self, handle, milliseconds): return 0x102 if self.native.guardian_alive else 0
    def GetProcessTimes(self, handle, *times):
        times[0]._obj.value = self.native.guardian_birth
        return True
    def CloseHandle(self, handle): self.native.calls.append(('close_identity', handle))


class FakeNative:
    def __init__(self, guard):
        self.g, self.calls = guard, []
        self.children_done = False; self.exit_codes = {'launcher': 0, 'gpu_child': 0}
        self.resume_status = 0; self.guardian_alive = True; self.guardian_birth = 777
        self.parents_map = {999: 42}; self.extra_training = set()
        self.ntdll = types.SimpleNamespace(NtResumeProcess=ResumeCall(self)); self.kernel = FakeKernel(self)
        self.refs = {role: {'role': role, 'pid': pid, 'parent_pid': guard.PARENTS[role],
                           'creation_100ns': str(i + 100), 'created_at': guard.BIRTHS[role],
                           'handle': 'fake-' + role, 'image': 'fixture-python.exe'}
                     for i, (role, pid) in enumerate(guard.PIDS.items())}
    def capture(self, role, *, arm=False):
        self.calls.append(('capture', role, arm)); return dict(self.refs[role])
    def exited(self, ref): return False if ref['role'] == 'dispatcher' else self.children_done
    def exit_code(self, ref): return self.exit_codes[ref['role']]
    def descendants(self):
        return set(self.extra_training) | (set() if self.children_done else {self.g.PIDS['launcher'], self.g.PIDS['gpu_child']})
    def parents(self): return dict(self.parents_map)
    def suspend_dispatcher(self, ref):
        assert ref['role'] == 'dispatcher'; self.calls.append(('suspend', ref['role']))
    def terminate_dispatcher(self, ref): raise AssertionError('No termination permitted')
    def close(self, ref): self.calls.append(('close', ref['role']))


class FakeProcess:
    def __init__(self, fixture, number, behavior):
        self.fixture, self.pid, self.behavior = fixture, 1000 + number, behavior
        self.returncode = None
    def wait(self, timeout):
        f = self.fixture; f.actions.append(('wait_qualifier', self.pid, timeout))
        if self.behavior.get('lease'):
            f.save(f.probe_lease, {'id': 'fake-leftover'})
        if self.behavior.get('orphan'):
            f.native.parents_map[self.pid + 10000] = self.pid
        if self.behavior.get('pause'):
            f.pause()
        if self.behavior.get('timeout'):
            if not self.behavior.get('alive', True): self.returncode = 1
            raise TimeoutError('fake qualifier timeout')
        self.returncode = self.behavior.get('code', 0)
        return self.returncode
    def poll(self): return self.returncode


class GuardianTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sync_guard_fixture_', dir=REPORT)
        self.work = Path(self.temp.name).resolve(); self.g = load_guard()
        self.out = self.work / 'out'; self.report = self.work / 'report'; self.report.mkdir()
        self.root = self.out / 'execution_acceleration_v1/sync_boundary'; self.root.mkdir(parents=True)
        self.probe_lease = self.out / 'execution_acceleration_v1/sync_qualification_owner.json'
        self.actions, self.behaviors = [], [{}, {}]; self.processes = []
        self.complete = True; self.gpu_available = True; self.wait_hook = None
        self.source = self.report / SOURCE.name; self.source.write_bytes(SOURCE.read_bytes())
        self.g.SOURCE, self.g.ROOT = self.source, self.root
        self.g.S = types.SimpleNamespace(WORK=self.work, OUT=self.out, REPORT=self.report,
                read=self.read, save=self.save, desc=self.desc, checked=self.checked, now=lambda: 'fixture-time')
        self.q4root = self.out / 'continuation_queue_v4'; self.q4root.mkdir()
        self.sources = []
        for n in range(59):
            p = self.work / f'source_{n}.py'; p.write_text('# fixture\n', encoding='utf-8'); self.sources.append(self.desc(p))
        self.save(self.q4root / 'registration.json', {'sources': self.sources})
        attempt = self.q4root / 'attempts/fixture/status.json'
        self.save(attempt, {'status': 'running', 'child_pid': self.g.PIDS['launcher'], 'job': {'module': 'finetune_v3'}})
        self.owner = {'id': self.g.OWNER_ID, 'pid': self.g.PIDS['dispatcher'], 'child_pid': self.g.PIDS['launcher'],
                      'child_job': self.g.CONDITION + '-fit',
                      'linux_owner': (attempt.with_name('owner.json')).relative_to(self.work).as_posix()}
        self.save(self.q4root / 'owner.json', self.owner)
        self.g.Q4 = types.SimpleNamespace(ROOT=self.q4root, register=lambda: {'sources': self.sources})
        def completed(conditions):
            self.actions.append(('verify_all15', conditions)); return self.complete
        self.g.Q1 = types.SimpleNamespace(all_complete=completed)
        self.fit = self.out / 'runs/CSBrain-lastblock-finetune/seed_22/fold_5'; self.fit.mkdir(parents=True)
        (self.fit / 'latest.pt').write_bytes(b'SYNTHETIC FIXTURE, NOT A TENSOR FILE')
        self.save(self.fit / 'train_history.json', [{'epoch': 1}])
        self.save(self.out / 'registration.json', {'folds': {'fold_5': {'train': ['a'], 'val': ['b'], 'test': ['c']}}})
        self.g.R = types.SimpleNamespace(fit_directory=lambda *args: self.fit,
                   registration_descriptor=lambda: self.desc(self.out/'registration.json'),
                   load_registration=lambda: self.read(self.out/'registration.json'))
        for name in ('sync_reduction_v1.py', 'sync_subject_update_v1.py', 'sync_state_digest_v1.py', 'sync_qualification_gate_v1.py'):
            p = self.work/'tridep_ccml/eeg_all_candidates'/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# fixture\n')
        for name in self.g.QUALIFIERS: (self.report/name).write_text('# fixture\n')
        self.native = FakeNative(self.g); self.g.backend = lambda: self.native
        self.identity = {'pid': 999, 'creation_100ns': '777', 'parent_pid': 42, 'source': self.desc(self.source)}
        self.g.self_identity = lambda native: copy.deepcopy(self.identity)
        self.g.time = types.SimpleNamespace(time_ns=__import__('time').time_ns, sleep=self.tick)
        self.g.gpu_idle = self.gpu_idle
        self.g.subprocess = types.SimpleNamespace(Popen=self.spawn, STDOUT=-2)
        self.expected = {'source': self.desc(self.source), 'owner': self.desc(self.q4root/'owner.json'),
                         'registration': self.desc(self.q4root/'registration.json'),
                         'roles': {r: self.g.H2.public_identity(v) for r,v in self.native.refs.items()}}
        self.expected_path = self.root/'inspection_fixture.json'; self.save(self.expected_path, self.expected)

    def tearDown(self):
        assert self.work.parent == REPORT.resolve() and self.work.name.startswith('sync_guard_fixture_')
        self.temp.cleanup()

    def desc(self, path):
        path = Path(path).resolve(); assert path.is_relative_to(self.work)
        return {'path': path.relative_to(self.work).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    def checked(self, binding):
        p = self.work / binding['path']; self.assertEqual(self.desc(p), binding); return p
    def read(self, path): return json.loads(Path(path).read_text(encoding='utf-8'))
    def save(self, path, value, *, immutable=False):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x' if immutable else 'w', encoding='utf-8') as f: json.dump(value, f)
    def pause(self, value=None): self.save(self.root/'user_pause.json', {'paused': True} if value is None else value)
    def tick(self, seconds):
        self.actions.append(('wait_children', seconds))
        if self.wait_hook: self.wait_hook()
        else: self.native.children_done = True
    def gpu_idle(self):
        self.actions.append(('gpu_idle',))
        if not self.gpu_available: raise ValueError('fake GPU busy')
        return {'passed': True}
    def spawn(self, arguments, **kwargs):
        self.assertTrue(self.native.children_done); self.assertTrue(any(x[0]=='suspend' for x in self.native.calls))
        self.actions.append(('spawn', Path(arguments[3]).name))
        process = FakeProcess(self, len(self.processes), self.behaviors[len(self.processes)])
        self.processes.append(process); return process
    def arm(self): return self.g.arm(self.desc(self.expected_path))
    def resumed(self): return [x for x in self.native.calls if x[0]=='resume']
    def state(self): return self.read(self.root/'state.json')
    def live_proof(self):
        self.native.children_done = True
        lease = self.root/'guardian_owner.json'; self.save(lease, {'identity': self.identity})
        proof = {'schema':'sync_qualification_safe_boundary_v1','passed':True,'source':self.desc(self.source),
                 'queue_owner':self.desc(self.q4root/'owner.json'),'registration':self.desc(self.q4root/'registration.json'),
                 'guardian_state_path':(self.root/'state.json').relative_to(self.work).as_posix(),
                 'guardian_owner':self.desc(lease),'guardian_identity':self.identity,
                 'held_process_identities':self.expected['roles'],'gpu_children_exit_verified':True,
                 'launcher_exit_code':0,'gpu_child_exit_code':0}
        p=self.root/'proof.json'; self.save(p,proof); binding=self.desc(p)
        self.save(self.root/'state.json',{'phase':'qualification_running','dispatcher_suspended':True,
                  'boundary_proof':binding,'source':proof['source'],'guardian_identity':self.identity,
                  'held_process_identities':proof['held_process_identities']})
        return binding

    def test_natural_success_only_dispatcher_control_and_sequential_probes(self):
        result=self.arm()
        self.assertTrue(result['passed']); self.assertTrue(result['all_qualifications_passed'])
        self.assertEqual([x for x in self.native.calls if x[0]=='suspend'], [('suspend','dispatcher')])
        self.assertEqual(len(self.resumed()),1)
        self.assertEqual([x[1] for x in self.actions if x[0]=='spawn'], list(self.g.QUALIFIERS))
        self.assertEqual(len(self.processes),2)
        self.assertTrue(all(not x[2] for x in self.native.calls if x[0]=='capture' and x[1]!='dispatcher'))
        self.assertTrue(result['gpu_children_suspended_or_terminated'] is False)
        self.assertTrue((self.root/'guardian_owner.json').is_file())

    def test_declined_qualification_not_claimed_passed(self):
        self.behaviors=[{'code':2},{}]; result=self.arm()
        self.assertTrue(result['boundary_workflow_completed']); self.assertFalse(result['all_qualifications_passed'])
        self.assertTrue(result['outcomes'][0]['qualification_declined']); self.assertFalse(result['outcomes'][0]['qualification_passed'])
        self.assertEqual(len(self.resumed()),1)

    def test_changed_inspection_identity_refuses_before_suspend(self):
        self.expected['roles']['gpu_child']['creation_100ns']='wrong'; self.save(self.expected_path,self.expected)
        with self.assertRaises(ValueError): self.arm()
        self.assertFalse([x for x in self.native.calls if x[0] in ('suspend','resume')]); self.assertFalse(self.processes)

    def test_child_failure_or_incomplete15_never_starts_probe(self):
        self.native.exit_codes['gpu_child']=1
        with self.assertRaises(ValueError): self.arm()
        self.assertFalse(self.processes); self.assertEqual(len(self.resumed()),1)

    def test_missing_full15_evidence_refuses_probe(self):
        self.complete=False
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.processes); self.assertEqual(len(self.resumed()),1)

    def test_source_change_refuses_before_probe(self):
        self.wait_hook=lambda: ((self.work/'source_0.py').write_text('# changed'), setattr(self.native,'children_done',True))
        with self.assertRaises(AssertionError):self.arm()
        self.assertFalse(self.processes); self.assertEqual(len(self.resumed()),1)

    def test_successful_exit_with_stale_lease_keeps_dispatcher_held(self):
        self.behaviors=[{'lease':True},{}]
        with self.assertRaises(ValueError):self.arm()
        self.assertEqual(len(self.processes),1); self.assertFalse(self.resumed())
        self.assertEqual(self.state()['phase'],'probe_cleanup_unverified_Q4_held')

    def test_successful_exit_with_cpu_orphan_keeps_dispatcher_held(self):
        self.behaviors=[{'orphan':True},{}]
        with self.assertRaises(ValueError):self.arm()
        self.assertEqual(len(self.processes),1); self.assertFalse(self.resumed())

    def test_timeout_live_cpu_qualifier_not_resumed_even_gpu_idle(self):
        self.behaviors=[{'timeout':True},{}]
        with self.assertRaises(TimeoutError):self.arm()
        self.assertFalse(self.resumed()); self.assertEqual(self.state()['phase'],'probe_cleanup_unverified_Q4_held')

    def test_failed_exited_clean_qualifier_resumes_and_skips_second(self):
        self.behaviors=[{'code':1},{}]
        with self.assertRaises(ValueError):self.arm()
        self.assertEqual(len(self.processes),1); self.assertEqual(len(self.resumed()),1)

    def test_gpu_busy_after_original_children_exit_blocks_resume_before_any_probe(self):
        self.gpu_available=False
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.processes);self.assertFalse(self.resumed())
        self.assertEqual(self.state()['phase'],'probe_cleanup_unverified_Q4_held')

    def test_leftover_probe_lease_blocks_resume_before_any_probe(self):
        self.save(self.probe_lease, {'id':'already-present'})
        with self.assertRaises(ValueError):self.arm()
        self.assertFalse(self.processes);self.assertFalse(self.resumed())

    def test_user_pause_before_arm_performs_no_process_control(self):
        self.pause();result=self.arm()
        self.assertEqual(result['phase'],'user_pause_before_suspend');self.assertFalse(self.resumed());self.assertFalse(self.processes)
        self.assertFalse([x for x in self.native.calls if x[0]=='suspend'])

    def test_user_pause_waiting_retains_native_suspend(self):
        self.wait_hook=self.pause; result=self.arm()
        self.assertEqual(result['phase'],'user_pause_Q4_held');self.assertFalse(self.resumed());self.assertFalse(self.processes)

    def test_user_pause_during_probe_revokes_resume_and_next_probe(self):
        self.behaviors=[{'pause':True},{}];result=self.arm()
        self.assertEqual(result['phase'],'user_pause_Q4_held');self.assertEqual(len(self.processes),1);self.assertFalse(self.resumed())

    def test_user_pause_during_timed_out_probe_takes_precedence(self):
        self.behaviors=[{'pause':True,'timeout':True},{}];result=self.arm()
        self.assertEqual(result['phase'],'user_pause_Q4_held');self.assertTrue(result['automatic_resume_revoked_by_user']);self.assertFalse(self.resumed())

    def test_malformed_marker_fails_closed_after_suspend(self):
        self.wait_hook=lambda:self.pause({'paused':False})
        with self.assertRaises(ValueError):self.arm()
        self.assertEqual(self.state()['phase'],'invalid_user_pause_marker_Q4_held');self.assertFalse(self.resumed())

    def test_resume_itself_rechecks_user_pause(self):
        self.pause()
        with self.assertRaises(self.g.UserPauseRequested):self.g.resume_dispatcher(self.native,self.native.refs['dispatcher'])
        self.assertFalse(self.resumed())

    def test_live_boundary_readonly_and_birth_lease_state_binding(self):
        binding=self.live_proof();result=self.g.verify_live_boundary(binding)
        self.assertTrue(result['passed']);self.assertEqual(result['process_control_calls'],0)
        self.assertFalse([x for x in self.native.calls if x[0] in ('suspend','resume')])
        self.native.guardian_birth=778
        with self.assertRaises(ValueError):self.g.verify_live_boundary(binding)
        self.native.guardian_birth=777;self.native.parents_map[999]=43
        with self.assertRaises(ValueError):self.g.verify_live_boundary(binding)

    def test_live_boundary_rejects_new_child_bad_state_or_pause(self):
        binding=self.live_proof();self.native.extra_training={555}
        with self.assertRaises(ValueError):self.g.verify_live_boundary(binding)
        self.native.extra_training=set();state=self.state();state['source']={'path':'wrong','sha256':'wrong'};self.save(self.root/'state.json',state)
        with self.assertRaises(ValueError):self.g.verify_live_boundary(binding)
        self.pause()
        with self.assertRaises(self.g.UserPauseRequested):self.g.verify_live_boundary(binding)


if __name__=='__main__':unittest.main(verbosity=2)
