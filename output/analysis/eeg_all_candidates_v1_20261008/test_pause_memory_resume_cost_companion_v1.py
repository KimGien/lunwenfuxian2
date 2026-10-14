"""Pure stdlib guards; no plan tests, process controls or scientific execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPORT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_pause_cost_guard', REPORT / 'pause_memory_resume_cost_companion_v1.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


def point(qpc, wall='2026-10-10T15:00:00+08:00'):
    return {'wall': wall, 'qpc': str(qpc), 'qpc_frequency': '1000'}


class TimingGuards(unittest.TestCase):
    def test_resume_qpc_window_is_not_pause_or_active_duration(self):
        value = C.clock_window(point(1000), point(1005))
        self.assertEqual(.005, value['resume_call_observed_window_seconds'])
        self.assertIsNone(value['pause_monotonic_exact_seconds']); self.assertIsNone(value['active_training_elapsed_exact_seconds'])

    def test_wall_jump_does_not_create_pause_subtraction(self):
        value = C.clock_window(point(1000, '2026-10-10T15:00:00+08:00'), point(1005, '2026-10-09T15:00:00+08:00'))
        self.assertEqual(.005, value['resume_call_observed_window_seconds']); self.assertIsNone(value['pause_monotonic_exact_seconds'])

    def test_invalid_qpc_or_frequency_rejected(self):
        for change in ({'qpc': True}, {'qpc': '-1'}, {'qpc_frequency': '0'}, {'qpc_frequency': '1001'}, {'wall': '2026-10-10T15:00:00'}):
            with self.subTest(change=change), self.assertRaises(ValueError): C.clock_window(point(1000), {**point(1005), **change})
        with self.assertRaises(ValueError): C.clock_window(point(1000), point(999))

    def test_inclusive_timer_never_active_known_contribution(self):
        row = C.timer_row('F3_formal_attempt', {'path': 'trace.json', 'sha256': 'a'*64}, 10000., 'actual_terminal')
        self.assertEqual(10000., row['recorded_elapsed_including_native_pause_seconds'])
        self.assertFalse(row['eligible_for_active_known_subtotal']); self.assertIsNone(row['active_known_subtotal_contribution'])
        self.assertIsNone(row['active_training_elapsed_exact_seconds']); self.assertFalse(row['wall_pause_subtracted'])

    def test_real_zero_recorded_timer_is_not_zero_active_time(self):
        row = C.timer_row('Q5_parent_dispatcher', seconds=0, status='actual_terminal')
        self.assertEqual(0, row['recorded_elapsed_including_native_pause_seconds']); self.assertIsNone(row['active_training_elapsed_exact_seconds'])
        self.assertTrue(row['nested_with_formal_attempt'])

    def test_pending_or_invalid_elapsed_cannot_pass_terminal(self):
        for value in (True, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError): C.timer_row('F3_formal_attempt', seconds=value, status='actual_terminal')
        with self.assertRaises(ValueError): C.timer_row('F3_formal_attempt', seconds=8)
        self.assertIsNone(C.timer_row('F3_formal_attempt')['recorded_elapsed_including_native_pause_seconds'])


class ReadGuards(unittest.TestCase):
    def test_binary_guard_before_access(self):
        with tempfile.TemporaryDirectory(dir=REPORT) as temp:
            r=C.Reader(temp)
            for name in ('weights.pth','latest.pt','signals.npy'):
                with self.subTest(name=name), self.assertRaises(ValueError): r.bind(Path(temp)/name)

    def test_nonterminal_never_becomes_immutable_input(self):
        with tempfile.TemporaryDirectory(dir=REPORT) as temp:
            path=Path(temp)/'trace.json';path.write_text(json.dumps({'status':'running','seconds':9999}),encoding='utf8')
            reader=C.Reader(temp);value,binding,reason=C.read_terminal(reader,path)
            self.assertIsNone(value);self.assertIsNone(binding);self.assertEqual('still_nonterminal',reason);self.assertEqual({},reader.inputs)

    def test_terminal_actual_zero_time_bound(self):
        with tempfile.TemporaryDirectory(dir=REPORT) as temp:
            path=Path(temp)/'trace.json';path.write_text(json.dumps({'status':'complete','seconds':0}),encoding='utf8')
            reader=C.Reader(temp);value,binding,reason=C.read_terminal(reader,path)
            self.assertEqual(0,value['seconds']);self.assertIsNone(reason);self.assertEqual(binding,reader.inputs['trace.json'])

    def test_transient_permission_has_bounded_retry_and_defers(self):
        with tempfile.TemporaryDirectory(dir=REPORT) as temp:
            path=Path(temp)/'trace.json';path.write_text('{}',encoding='utf8');reader=C.Reader(temp)
            with patch.object(Path,'read_bytes',side_effect=PermissionError('locked')) as call,patch.object(C.time,'sleep'):
                value,binding,reason=C.read_terminal(reader,path)
            self.assertEqual(3,call.call_count);self.assertIsNone(value);self.assertIsNone(binding)
            self.assertEqual('transient_read_unresolved',reason);self.assertEqual({},reader.inputs)

    def test_mutating_matching_terminal_defers(self):
        with tempfile.TemporaryDirectory(dir=REPORT) as temp:
            path=Path(temp)/'trace.json';path.write_text('{}',encoding='utf8');reader=C.Reader(temp)
            with patch.object(Path,'read_bytes',side_effect=[b'{"status":"complete"}',b'{"status":"running"}']):
                value,binding,reason=C.read_terminal(reader,path)
            self.assertEqual('changed_during_observation',reason);self.assertEqual({},reader.inputs)

    def test_immutable_binding_change_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPORT) as temp:
            path=Path(temp)/'proof.json';path.write_text('{}',encoding='utf8');reader=C.Reader(temp);binding=reader.bind(path)
            path.write_text('{"changed":true}',encoding='utf8')
            with self.assertRaises(ValueError):reader.checked(binding)

    def test_native_birth_set_strict(self):
        rows=[{'role':role,'pid':pid,'creation_100ns':birth} for role,(pid,birth) in C.ROLES.items()]
        self.assertEqual(set(C.ROLES),set(C.identity_rows(rows)))
        rows[0]['creation_100ns']='1'
        with self.assertRaises(ValueError):C.identity_rows(rows)


class ResumeProofGuards(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=REPORT);self.root=Path(self.temp.name).resolve()
        self.out=self.root/'out';self.pause=self.out/'pause';self.folder=self.out/'resume'
        self.pause.mkdir(parents=True);self.folder.mkdir()
        self.patches=[patch.object(C,'PAUSE',self.pause),patch.object(C,'OUT',self.out)]
        for p in self.patches:p.start()
        self.source={'path':'resumer.ps1','sha256':'a'*64};self.owner={'pid':45980}
        self.identities=[{'role':r,'pid':p,'creation_100ns':b,'image':'python.exe','command_line':'command'} for r,(p,b) in C.ROLES.items()]
        self.native={'identities':self.identities,'owner_snapshot':self.owner};self.pause_value={'owner_snapshot':self.owner}
        paths=[self.out/'user_pause_q5_owner.json',*(self.out/'execution_acceleration_v1'/n/'user_pause.json' for n in ('sync_boundary','sync_boundary_v2','sync_boundary_v3'))]
        self.archives=[]
        for index,path in enumerate(paths,1):
            archive=self.folder/('control_%d.json'%index)
            value={'folder':str(self.pause),'automatic_resume_authorized':False} if index==1 else {'paused':True}
            self.write(archive,value);binding=C.Reader(self.root).bind(archive)
            self.archives.append({'original':str(path),'archive':str(archive),'sha256':binding['sha256']})
        actions=[]
        for index,(role,(pid,birth)) in enumerate(C.ROLES.items(),1):
            start={'source':self.source,'role':role,'pid':pid,'birth':birth,'resume_started':True,'native_result':None,'clock':point(index*1000)}
            action={'role':role,'pid':pid,'creation_100ns':birth,'native_status':0,'returned':point(index*1000+5)}
            self.write(self.folder/('event_%02d_%s_started.json'%(index,role)),start)
            self.write(self.folder/('event_%02d_%s_returned.json'%(index,role)),action);actions.append(action)
        self.complete={'status':'same_four_objects_resumed_by_explicit_continue','source':self.source,'actions':actions,
            'verification':self.identities,'controls_archived':self.archives,'native_resume_calls':4,'new_queue_processes':0,
            'optimizer_operations_by_controller':0,'model_reads_by_controller':0,'process_terminate_calls':0,
            'pause_boundary_qpc_not_measured':True,'exact_active_elapsed_seconds':None,'pause_not_training':True}
        self.pre={'source':self.source,'pause_proof':str(self.pause/'pause_complete.json'),'pause_proof_sha256':C.PAUSE_SHA,
            'native_pause':str(self.pause/'native_pause_complete.json'),'native_pause_sha256':C.NATIVE_SHA,
            'owner_snapshot':self.owner,'identities':self.identities,'controls':self.archives,
            'pause_boundary_qpc_not_measured':True,'paused_interval_qpc_exact':None}
        self.release={'status':'explicit_continue_controls_archived_then_released','source':self.source,'controls':self.archives}

    def write(self,path,value):path.write_text(json.dumps(value),encoding='utf8')
    def verify(self):
        for name,value in [('native_resume_complete.json',self.complete),('captured_resume_preflight.json',self.pre),('resume_controls_released.json',self.release)]:self.write(self.folder/name,value)
        return C.verify_resume(C.Reader(self.root),self.folder,self.pause_value,self.native,self.source)
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def test_valid_four_object_proof_has_unknown_pause_and_no_native_actions(self):
        value=self.verify();self.assertEqual(4,len(value['resume_call_windows']));self.assertIsNone(value['pause_monotonic_exact_seconds'])

    def test_failed_or_duplicate_resume_is_rejected(self):
        self.complete['actions'][0]['native_status']=1
        with self.assertRaises(ValueError):self.verify()
        self.complete['actions'][0]['native_status']=0;self.complete['actions'][-1]=self.complete['actions'][0]
        with self.assertRaises(ValueError):self.verify()

    def test_missing_controls_release_is_not_confirmed_resume(self):
        self.release['status']='pending'
        with self.assertRaises(ValueError):self.verify()

    def test_fabricated_exact_pause_or_active_seconds_rejected(self):
        self.pre['paused_interval_qpc_exact']=12
        with self.assertRaises(ValueError):self.verify()
        self.pre['paused_interval_qpc_exact']=None;self.complete['exact_active_elapsed_seconds']=0
        with self.assertRaises(ValueError):self.verify()

    def test_resume_event_must_match_bound_action(self):
        self.complete['actions'][0]['returned']=point(9999)
        with self.assertRaises(ValueError):self.verify()

    def test_control_snapshot_cannot_substitute_current_control(self):
        self.archives[0]['archive']=str(self.out/'user_pause_q5_owner.json')
        with self.assertRaises(ValueError):self.verify()


if __name__=='__main__':unittest.main(verbosity=2)
