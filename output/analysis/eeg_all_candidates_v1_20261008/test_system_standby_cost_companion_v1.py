"""Standard-library source/JSON fixtures only; never invoke actual collect."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

REPORT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_standby_guard', REPORT/'system_standby_cost_companion_v1.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


class EventGuards(unittest.TestCase):
    def setUp(self):
        self.events = json.loads((C.WORK/C.EVENTS['path']).read_text(encoding='utf-8-sig'))
        self.review = json.loads((C.WORK/C.REVIEW['path']).read_text(encoding='utf-8-sig'))
    def test_actual_event_json_parses_without_timer_deduction(self):
        result = C.parse_events(self.events, self.review)
        self.assertEqual(result['scenario_instance_v2'], 253)
        self.assertEqual(result['raw_provider_residency_microseconds']['ActiveResidencyInUs'], 1928093259)
        self.assertTrue(result['system_interval_is_not_per_process_zero_work'])
        self.assertIsNone(result['timer_deduction_seconds'])
        self.assertFalse(result['current_sleep_state_inferred'])
        self.assertFalse(result['unpaired_exit_observed_in_window'])
    def test_xml_header_mismatch_rejected(self):
        self.events['all_power_provider_events_in_same_window'][1]['event_id'] = 507
        with self.assertRaises(ValueError): C.parse_events(self.events, self.review)
    def test_wrong_v2_pair_rejected(self):
        for row in self.events['all_power_provider_events_in_same_window']:
            if row['record_id'] == 149310: row['xml'] = row['xml'].replace("Name='ScenarioInstanceIdV2'>253", "Name='ScenarioInstanceIdV2'>255")
        for row in self.review['structured_provider_event_data']:
            if row['record_id'] == 149310: row['data']['ScenarioInstanceIdV2'] = '255'
        with self.assertRaises(ValueError): C.parse_events(self.events, self.review)
    def test_fabricated_exact_time_rejected(self):
        self.review['exact_nontraining_interval_seconds'] = 3031.7
        with self.assertRaises(ValueError): C.parse_events(self.events, self.review)
    def test_duplicate_record_rejected(self):
        self.events['all_power_provider_events_in_same_window'].append(self.events['all_power_provider_events_in_same_window'][0])
        with self.assertRaises(ValueError): C.parse_events(self.events, self.review)
    def test_entity_declaration_rejected(self):
        self.events['all_power_provider_events_in_same_window'][0]['xml'] = '<!DOCTYPE Event>'+self.events['all_power_provider_events_in_same_window'][0]['xml']
        with self.assertRaises(ValueError): C.parse_events(self.events, self.review)


class TimerGuards(unittest.TestCase):
    def test_zero_recorded_not_zero_active(self):
        row = C.timer_row('F3_formal_attempt', {'path': 'a.json', 'sha256': 'a'*64}, 0, 'actual_terminal')
        self.assertEqual(row['recorded_elapsed_including_observed_low_power_window_seconds'], 0)
        self.assertIsNone(row['active_known_subtotal_contribution']); self.assertIsNone(row['standby_nontraining_exact_seconds'])
        self.assertFalse(row['eligible_for_active_known_subtotal'])
    def test_pending_malformed_seconds_rejected(self):
        for seconds in (1, True, -1, float('nan'), float('inf')):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError): C.timer_row('F3_formal_attempt', seconds=seconds)
    def test_metadata_parent_remain_nested(self):
        for kind in C.SCOPES[1:]: self.assertTrue(C.timer_row(kind)['nested_with_formal_attempt'])
    def test_original_fit_and_overlap_are_required(self):
        value = {'started_at': C.STARTS['F3_formal_attempt'], 'condition': 'LUNA-lastblock-finetune', 'seed': 33, 'fold': 1, 'kind': 'formal',
                 'implementation': {'version': 3, 'wrapper_source': C.PINNED['F3']}, 'status': 'running'}
        interval = {'entry_utc': '2026-10-10T11:30:15Z', 'exit_utc': '2026-10-10T12:21:31Z'}
        C.verify_attempt(value, 'F3_formal_attempt', interval)
        for change in ({'fold': True}, {'started_at': '2026-10-10T21:00:00+08:00'}, {'seed': 22}):
            with self.subTest(change=change), self.assertRaises(ValueError): C.verify_attempt({**value, **change}, 'F3_formal_attempt', interval)
    def test_terminal_must_span_exit_and_be_graceful(self):
        value = {'started_at': C.STARTS['F3_formal_attempt'], 'condition': 'LUNA-lastblock-finetune', 'seed': 33, 'fold': 1, 'kind': 'formal',
                 'implementation': {'version': 3, 'wrapper_source': C.PINNED['F3']}, 'status': 'complete', 'seconds': 5000,
                 'finished_at': '2026-10-10T20:30:00+08:00', 'graceful_terminal_flush': True, 'counter_snapshot_exact_for_process': True}
        interval = {'entry_utc': '2026-10-10T11:30:15Z', 'exit_utc': '2026-10-10T12:21:31Z'}
        C.verify_attempt(value, 'F3_formal_attempt', interval)
        for change in ({'finished_at': '2026-10-10T20:00:00+08:00'}, {'graceful_terminal_flush': False}, {'seconds': True}):
            with self.subTest(change=change), self.assertRaises(ValueError): C.verify_attempt({**value, **change}, 'F3_formal_attempt', interval)


class NativeGuards(unittest.TestCase):
    def setUp(self):
        self.value=json.loads((C.WORK/C.NATIVE['path']).read_text(encoding='utf-8-sig'))
        self.interval={'exit_utc':'2026-10-10T12:21:31Z'}
    def test_saved_observation_only_not_full_window_liveness(self):
        result=C.verify_native(self.value,self.interval)
        self.assertTrue(result['original_native_objects_observed_alive'])
        self.assertFalse(result['continuous_activity_through_entire_window_proven'])
        self.assertFalse(result['live_owner_or_progress_binding_rechecked'])
        self.assertTrue(result['observation_does_not_prove_current_liveness_at_final_delivery'])
    def test_wrong_birth_or_suspended_state_rejected(self):
        for change in ({'creation_100ns':'1'},{'suspended_threads':1},{'native_exit_code':True}):
            value=copy.deepcopy(self.value);value['identities'][-1].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError): C.verify_native(value,self.interval)
    def test_future_mutable_owner_reference_not_followed(self):
        self.value['owner_binding']={'path':'now_missing_owner.json','sha256':'not_current'}
        self.value['progress']['sha256']='old_not_current'
        self.assertTrue(C.verify_native(self.value,self.interval)['original_native_objects_observed_alive'])
    def test_exact_retained_owner_and_embedded_owner_must_match(self):
        link=json.loads((C.WORK/C.OWNER_LINK['path']).read_text(encoding='utf-8-sig'))
        snapshot=json.loads((C.WORK/C.OWNER_SNAPSHOT['path']).read_text(encoding='utf-8-sig'))
        result=C.verify_owner_snapshot(self.value,link,snapshot)
        self.assertFalse(result['current_live_owner_read'])
        with self.assertRaises(ValueError): C.verify_owner_snapshot(self.value,link,{**snapshot,'child_pid':1})


class ReaderGuards(unittest.TestCase):
    def test_binary_and_outside_paths_rejected_before_access(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix="standby_fixture_") as tmp:
            reader=C.Reader(tmp)
            for path in ('a.pt', 'a.pth', 'a.npy', '../outside.json', 'test_predictions.json'):
                with self.subTest(path=path), self.assertRaises(ValueError): reader.path(path)
    def test_running_not_immutable(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix="standby_fixture_") as tmp:
            p=Path(tmp)/'a.json'; p.write_text('{"status":"running"}',encoding='utf-8'); reader=C.Reader(tmp)
            record,binding,reason=C.observe(reader,p)
            self.assertEqual(reason,'still_nonterminal'); self.assertIsNone(binding); self.assertEqual(reader.inputs,{})
    def test_permission_retry_is_bounded_unknown(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix="standby_fixture_") as tmp:
            reader=C.Reader(tmp)
            with patch.object(Path,'read_bytes',side_effect=PermissionError('busy')) as read, patch.object(C.time,'sleep'):
                self.assertEqual(C.observe(reader,Path(tmp)/'a.json'),(None,None,'transient_read_unresolved'))
                self.assertEqual(read.call_count,3)
            self.assertFalse(reader.inputs)
    def test_changing_terminal_deferred(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix="standby_fixture_") as tmp:
            reader=C.Reader(tmp)
            with patch.object(Path,'read_bytes',side_effect=[b'{"status":"complete"}',b'{"status":"running"}']):
                self.assertEqual(C.observe(reader,Path(tmp)/'a.json'),(None,None,'changed_during_observation'))
            self.assertFalse(reader.inputs)
    def test_terminal_binding_rechecked(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix="standby_fixture_") as tmp:
            p=Path(tmp)/'a.json'; p.write_text('{"status":"complete"}',encoding='utf-8'); reader=C.Reader(tmp)
            _,binding,_=C.observe(reader,p); self.assertIsNotNone(binding)
            p.write_text('{"status":"failed"}',encoding='utf-8')
            with self.assertRaises(ValueError): reader.recheck()


class CollectionGuards(unittest.TestCase):
    def test_failed_original_cannot_claim_later_retry_metadata(self):
        with tempfile.TemporaryDirectory(dir=REPORT, prefix='standby_fixture_') as tmp:
            complete=Path(tmp)/'complete.json'; complete.write_text('{"passed":true,"metadata":{"path":"later_retry.json","sha256":"a"}}',encoding='utf-8')
            class FakeReader:
                inputs={}
                def bind(self,path,raw): return {'path':'source.py','sha256':'test'}
                def read(self,binding): return {}
                def path(self,path): return complete
                def recheck(self): pass
            with ExitStack() as stack:
                stack.enter_context(patch.object(C,'Reader',FakeReader))
                stack.enter_context(patch.object(C,'parse_events',return_value={}))
                stack.enter_context(patch.object(C,'verify_native',return_value={}))
                stack.enter_context(patch.object(C,'verify_owner_snapshot',return_value={}))
                stack.enter_context(patch.object(C,'verify_attempt'))
                stack.enter_context(patch.object(C,'observe',side_effect=[
                    ({'status':'failed','seconds':99}, {'path':C.F3,'sha256':'a'},None),
                    ({'status':'complete','seconds':199}, {'path':C.PARENT,'sha256':'b'},None)]))
                result=C.collect()
            rows=result['affected_timer_observations']
            self.assertEqual(rows[0]['producer_terminal_status'],'failed')
            self.assertEqual(rows[1]['status'],'pending'); self.assertIsNone(rows[1]['artifact'])
            self.assertIn('later successful retry',rows[1]['pending_reason'])
            self.assertFalse(result['all_affected_producer_timers_terminal'])


if __name__=='__main__': unittest.main(verbosity=2)
