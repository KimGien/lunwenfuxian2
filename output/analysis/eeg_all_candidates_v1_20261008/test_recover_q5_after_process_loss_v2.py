"""Synthetic control-flow tests only: no project verifier, process or tensor I/O."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
ENTRY = WORK / 'tridep_ccml/eeg_all_candidates/recover_q5_after_process_loss_v2.py'
SPEC = importlib.util.spec_from_file_location('recovery_v2_control_under_test', ENTRY)
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)
LOSS = {'path': 'fixture/loss.json', 'sha256': 'a' * 64}


class Fake:
    def __init__(self, fail=None):
        self.events = []; self.fail = fail; self.pauses = 0; self.owner = True; self.intent_kept = False
    def event(self, name):
        self.events.append(name)
        if name == self.fail:
            raise ValueError('Synthetic ' + name)
    def no_pause(self):
        self.pauses += 1; self.event('pause' + str(self.pauses))
    def verify(self, loss):
        self.event('verify'); return {'durable': {'epoch': 38}}
    def intent(self, loss):
        self.event('intent')
        if self.intent_kept: raise FileExistsError('Permanent intent retained')
        self.intent_kept = True
    def absence(self, context, loss, index):
        self.event('absence' + str(index)); return {'fresh': index}
    def register(self, context, loss, absence):
        self.event('register'); return {'manifest': 2}
    def closure(self, context, loss, manifest, absence):
        self.event('closure'); return {'closure': 2}
    def recheck(self, context, manifest, closure):
        self.event('recheck')
    def remove_owner(self, context):
        self.event('remove_owner'); self.owner = False
    def ready(self, context, manifest, closure, second):
        self.event('ready'); return {'passed': True}


class Control(unittest.TestCase):
    def test_transaction_order_two_absence_archive_and_rechecks(self):
        b = Fake(); result = M.prepare_recovery(LOSS, b)
        self.assertEqual(b.events, ['pause1','verify','intent','pause2','absence1','register','closure',
                        'recheck','pause3','absence2','pause4','recheck','remove_owner','pause5','ready'])
        self.assertEqual(result, ({'manifest':2}, {'closure':2}, {'passed':True}))
        self.assertFalse(b.owner)

    def test_all_failures_before_removal_preserve_owner(self):
        for point in ('pause1','verify','intent','pause2','absence1','register','closure','recheck','pause3','absence2','pause4'):
            with self.subTest(point=point):
                b = Fake(point)
                with self.assertRaises(ValueError): M.prepare_recovery(LOSS, b)
                self.assertTrue(b.owner); self.assertNotIn('ready', b.events)

    def test_pause_after_owner_archive_never_ready_or_launch(self):
        b = Fake('pause5')
        with self.assertRaises(ValueError): M.prepare_recovery(LOSS, b)
        self.assertFalse(b.owner); self.assertTrue(b.intent_kept); self.assertNotIn('ready', b.events)

    def test_failed_second_query_keeps_intent_and_blocks_blind_retry(self):
        b = Fake('absence2')
        with self.assertRaises(ValueError): M.prepare_recovery(LOSS, b)
        self.assertTrue(b.intent_kept); b.fail = None
        with self.assertRaises(FileExistsError): M.prepare_recovery(LOSS, b)
        self.assertTrue(b.owner)

    def test_no_runtime_imports_for_pending(self):
        with patch.object(M, 'Runtime', side_effect=AssertionError('No runtime')), patch.object(M.subprocess, 'run', side_effect=AssertionError('No process')):
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(M.main([]), 2)
        value = json.loads(stream.getvalue())
        self.assertEqual(value['actual_process_queries'], 0)
        self.assertEqual(value['process_controls'], 0)

    def test_run_requires_bound_loss(self):
        with patch.object(M, 'run', side_effect=AssertionError('Must not launch')):
            for argv in (['--run'], ['--run','--loss-evidence','x','--loss-evidence-sha256','bogus']):
                with self.assertRaises(ValueError): M.main(argv)

    def test_exclusive_intent_never_overwrites(self):
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            path = Path(folder) / 'intent.json'
            M.write_new(path, {'attempt':1}); before = path.read_bytes()
            with self.assertRaises(FileExistsError): M.write_new(path, {'attempt':2})
            self.assertEqual(path.read_bytes(), before)

    def test_exact_archive_and_original_retained(self):
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            source = Path(folder) / 'original.json'; target = Path(folder) / 'retained.json'
            raw = b'{"unknown": null}\r\n'; source.write_bytes(raw)
            binding = M.desc(source); archived = M.archive(binding, target)
            self.assertEqual(archived['sha256'], binding['sha256'])
            self.assertEqual(source.read_bytes(), raw); self.assertEqual(target.read_bytes(), raw)
            with self.assertRaises(FileExistsError): M.archive(binding, target)

    def test_archive_race_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            source = Path(folder) / 'original.json'; source.write_text('{}')
            binding = M.desc(source); source.write_text('{"raced":1}')
            with self.assertRaises(ValueError): M.archive(binding, Path(folder) / 'retained.json')

    def test_any_pause_marker_even_malformed_blocks(self):
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            path = Path(folder) / 'pause.json'
            with patch.object(M, 'PAUSE_PATHS', (path,)):
                M.assert_not_paused()
                for text in ('{"paused":true}', '{', '{"paused":false}'):
                    path.write_text(text)
                    with self.assertRaises(ValueError): M.assert_not_paused()

    def test_existing_other_closure_retained_same_attempt_rejected(self):
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            root = Path(folder); closures = root / 'process_loss_closures'; closures.mkdir()
            with patch.object(M, 'OUT', root):
                (closures / 'lost_f3_old.json').write_text(json.dumps({'original_attempt':{'path':'old-CSB.json'}}))
                M.assert_no_existing_closure({'path':'new-LUNA.json'})
                (closures / 'lost_f3_v2_existing.json').write_text(json.dumps({'original_attempt':{'path':'new-LUNA.json'}}))
                with self.assertRaises(ValueError): M.assert_no_existing_closure({'path':'new-LUNA.json'})
                self.assertEqual(len(list(closures.iterdir())),2)


class Absence(unittest.TestCase):
    def setUp(self):
        self.expected = [{'pid':p} for p in (45980,38860,25396,50296)]
        self.command = 'python.exe -m ' + M.ENTRY_MODULE + ' --run --loss-evidence loss.json --loss-evidence-sha256 ' + LOSS['sha256']
        self.rows = [dict(pid=100, parent_pid=101, creation_time_100ns=12345, image_path='python.exe', command_line=self.command),
                     dict(pid=101, parent_pid=9, creation_time_100ns=12344, image_path='python.exe', command_line=self.command)]
        self.payload = dict(query_succeeded=True, queried_pids=[x['pid'] for x in self.expected], exact_getprocess_alive_pids=[], rows=self.rows)
    def validate(self):
        return M.validate_absence_rows(self.payload, self.expected, current_pid=100,parent_pid=101,loss_binding=LOSS)
    def test_current_exact_recovery_pair_is_explicit_not_hidden(self):
        value = self.validate()
        self.assertEqual(value['allowed_recovery_processes'], self.rows)
        self.assertEqual(value['raw_observation']['rows'], self.rows)
        self.assertFalse(value['workspace_substring_filter_used'])
        self.assertIsNone(value['exit_codes']); self.assertIsNone(value['actual_exit_time'])
    def test_old_pid_reused_reject(self):
        self.rows.append({**self.rows[0], 'pid':45980})
        with self.assertRaises(ValueError): self.validate()
    def test_cim_empty_exact_getprocess_alive_reject(self):
        self.payload['exact_getprocess_alive_pids'] = [45980]
        with self.assertRaises(ValueError): self.validate()
    def test_other_checkout_worker_reject(self):
        self.rows.append({**self.rows[0], 'pid':222, 'command_line':r'Z:\other\python.exe -m tridep_ccml.eeg_all_candidates.finetune_v3'})
        with self.assertRaises(ValueError): self.validate()
        self.assertIsNotNone(M.re.search(M.WORKER_PATTERN, self.rows[-1]['command_line']))
    def test_missing_self_or_wrong_parent_reject(self):
        self.rows[0]['parent_pid'] = 333
        with self.assertRaises(ValueError): self.validate()
        self.rows.pop(0)
        with self.assertRaises(ValueError): self.validate()
    def test_unrelated_parent_not_allowed(self):
        self.rows[1]['command_line'] = 'python.exe other_job.py'
        with self.assertRaises(ValueError): self.validate()
    def test_wrong_loss_or_missing_run_reject(self):
        for bad in (self.command.replace(LOSS['sha256'], 'b'*64), self.command.replace(' --run ', ' ')):
            self.rows[0]['command_line'] = bad
            with self.assertRaises(ValueError): self.validate()
    def test_native_birth_and_image_required(self):
        self.rows[0]['creation_time_100ns'] = None
        with self.assertRaises(ValueError): self.validate()
        self.rows[0]['creation_time_100ns'] = 123; self.rows[0]['image_path'] = 'powershell.exe'
        with self.assertRaises(ValueError): self.validate()
    def test_broad_selector_has_no_workspace_gate(self):
        text = ENTRY.read_text(encoding='utf-8')
        self.assertNotIn("-like '*lunwen2*'", text)
        for module in ('finetune_v3', 'recover_q5_after_process_loss_v2', 'continuation_queue_v5', 'linux_entry_v4'):
            self.assertIsNotNone(M.re.search(M.WORKER_PATTERN, 'python -m tridep_ccml.eeg_all_candidates.'+module))


class CPUAndProvenance(unittest.TestCase):
    def setUp(self):
        self.auditor = {'path':'cpu6.py','sha256':'b'*64}
        self.value = dict(schema_version=6,auditor_source=self.auditor,all_135_finetune_checkpoints_verified=True,
            errors=[],deferred_completed_fits=[],pending_fits=[],status='all_finetune_verified',
            registered_finetune_fits=135,training_complete_finetune_observed=135,verified_completed_finetune_fits=135,checkpoint_loads=405)
    def test_complete_cpu6_gate_shape(self):
        M.validate_cpu_gate(self.value,self.auditor)
    def test_cpu_partial_oldschema_wrongsource_reject(self):
        for key,bad in [('schema_version',5),('auditor_source',{}),('checkpoint_loads',404),('pending_fits',['x']),
                        ('errors',['x']),('deferred_completed_fits',['x']),('verified_completed_finetune_fits',134),
                        ('all_135_finetune_checkpoints_verified',False),('status','partial_verified')]:
            with self.subTest(key=key):
                value = {**self.value,key:bad}
                with self.assertRaises(ValueError): M.validate_cpu_gate(value,self.auditor)
    def test_cpu_integer_not_bool_or_float(self):
        for key in ('schema_version','checkpoint_loads','registered_finetune_fits'):
            value = {**self.value,key:float(self.value[key])}
            with self.assertRaises(ValueError): M.validate_cpu_gate(value,self.auditor)
    def test_storage_preserves_actual_q5_core_and_adds_recovery(self):
        class Store:
            def save(self,path,value,**kwargs): self.value = value; return value
        store = Store(); fields = dict(recovery_registration={'v':2},recovery_execution_source={'entry':2},
            recovery_closure={'loss':2},previous_recovery_registration={'v':1},previous_recovery_closure={'loss':1},recovery_execution_version=2)
        with patch.object(M,'provenance',return_value=fields):
            wrapper = M.RecoveryStorage(store,{}, {})
        original = {'source':{'Q5':True},'execution_registration':{'base':True},'execution_version':5,'status':'running'}
        value = wrapper.save(M.ROOT/'owner.json',original)
        self.assertEqual(value,{**original,**fields}); self.assertNotIn('recovery_execution_version',original)
        self.assertEqual(wrapper.save(M.OUT/'untouched.json',original),original)
    def test_no_native_control_api_or_old_recovery_execution(self):
        text = ENTRY.read_text(encoding='utf-8')
        for forbidden in ('NtSuspendProcess','NtResumeProcess','TerminateProcess','Stop-Process','taskkill','Q5.run(', 'recover_q5_from_pause_v1 import'):
            self.assertNotIn(forbidden,text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
