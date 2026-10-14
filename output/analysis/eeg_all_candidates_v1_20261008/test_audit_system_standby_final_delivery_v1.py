"""Stdlib in-memory guards; actual event XML is read, never a final gate run."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('audit_system_standby_final_delivery_v1.py')
spec=importlib.util.spec_from_file_location('standby_delivery_guard_tests',P);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def d(path):return {'path':path,'sha256':'a'*64}
class Reader:
    def __init__(self,values=None):self.values=values or {};self.calls=[];self.inputs={}
    def checked(self,b):self.calls.append(b);return copy.deepcopy(self.values[b['path']])
    def snapshot(self,path):return self.checked(d(str(path))),d(str(path))
    def recheck(self):pass

def base():
    binds={k:d('fixture/'+k+'.json') for k in ('final_cost','wide','semantic_review','g3_bundle','pause_companion','resume_review')};binds['pause_companion']=M.NOON
    ids=[M.FIT]+['fixture/%d'%i for i in range(359)];req=[{'id':'R%02d'%i,'status':'verified_complete'} for i in range(1,31)]
    cost={'schema_version':6,'source':M.PINNED['cost'],'actual_cost_collector_source':M.PINNED['cost'],'finalizer_source':M.PINNED['finalizer'],
        'status':'verified_final_cost','final_snapshot':True,'registration':d('fixture/registry.json'),
        'formal_fits':[{'fit_id':x} for x in ids],'formal_status_counts':{'training_complete':360},'input_read_errors':[],
        'source_bound_cpu_checkpoint_verification':{'passed':True,'verified_fits':135,'prior_actual_cpu_checkpoint_loads':405,'new_cpu_checkpoint_loads':0}}
    wide={'source':M.PINNED['wide'],'objective_satisfied':True,'full_plan_completed':True,'requirements':req,'registration':cost['registration'],
        'final_cost':{'passed':True,'snapshot':binds['final_cost']},'semantic_review':{'passed':True,'artifact':binds['semantic_review']},
        'training_rows':[{'fit_id':x} for x in ids],'status_counts':{'training_complete':360},'verification_errors':[],'incomplete_fits':[]}
    bundle={'source':M.PINNED['g3'],'passed':True,'final_cost':binds['final_cost'],'full_plan_acceptance':binds['wide'],'companion_costs_added_again':False}
    semantic={'passed':True,'binding':{'cost_snapshot':binds['final_cost'],'audit_source':M.PINNED['wide']}}
    delivery={'schema':'memory_pause_final_delivery_gate_v1','source':M.MEMORY_SOURCE,'status':'verified_final_delivery_with_memory_pause',
        'passed':True,'objective_satisfied':True,'full_plan_completed':True,'requirements':req,'R01_R30_derived_after_all_prerequisites':True,
        'R28_pause_timer_scope':{'passed':True},'R29_memory_resume_control':{'passed':True},'all_plan_exact_active_seconds':None,
        'base_recorded_elapsed_totals_not_relabelled_as_active':True,'base_final_artifacts':binds}
    values={binds[k]['path']:v for k,v in {'final_cost':cost,'wide':wide,'g3_bundle':bundle,'semantic_review':semantic,'pause_companion':{'all_affected_producer_timers_terminal':True},'resume_review':{}}.items()}
    values['fixture/registry.json']={'fits':[{'fit_id':x} for x in ids]}
    return Reader(values),delivery

def timer_fixture():
    values={'fixture/human.md':'fixture'};cost={'cost_records':[{'artifact':d(M.REVE_F3),'category':'formal_training_attempt','aggregation_role':'primary_observed_invocation','seconds_observed':70}], 'formal_fits':[]};wide={'training_rows':[]};rows=[]
    for kind,(path,category,role,field) in M.TIMERS.items():
        desc=d(path);seconds=500.25;producer={field:seconds,'status':'complete','condition':'LUNA-lastblock-finetune','seed':33,'fold':1,'returncode':0,'graceful_terminal_flush':True,'counter_snapshot_exact_for_process':True}
        row={'scope':kind,'status':'actual_terminal','artifact':desc,'recorded_elapsed_including_observed_low_power_window_seconds':seconds,
            'active_training_elapsed_exact_seconds':None,'standby_nontraining_exact_seconds':None,'active_known_subtotal_contribution':None,
            'eligible_for_active_known_subtotal':False,'producer_elapsed_preserved':True,'wall_or_residency_subtracted':False,'nested_with_formal_attempt':kind!='F3_formal_attempt'}
        if kind=='F1_metadata_subscope':
            row['completion_binding']=d('fixture/complete.json');values['fixture/complete.json']={'passed':True,'metadata':desc}
            cost['formal_fits']=[{'fit_id':M.FIT,'metadata_artifact':desc,'last_attempt_seconds':seconds}];wide['training_rows']=[{'fit_id':M.FIT,'completion':row['completion_binding']}]
        values[path]=producer;rows.append(row);cost['cost_records'].append({'artifact':desc,'category':category,'aggregation_role':role,'seconds_observed':seconds})
    value={'schema':'system_standby_cost_companion_v1','source':M.STANDBY_SOURCE,'status':'verified_low_power_scope_and_recorded_terminal_timers',
        'all_affected_producer_timers_terminal':True,'passed':True,'affected_formal_fit_id':M.FIT,'system_event_snapshot':M.EVENT,'independent_system_event_review':M.EVENT_REVIEW,
        'final_scope_closure_ready':False,'final_scope_closure_required':True,'extra_optimizer_updates_added':0,'extra_backward_calls_added':0,'affected_timer_observations':rows,'immutable_inputs':[],'human_readable_report':d('fixture/human.md')}
    for k in ('observed_low_power_overlap','system_interval_is_not_per_process_zero_work','no_exact_timer_subtraction','no_new_process_or_loss_closure_created','same_attempt_identity_preserved','future_low_power_other_timers_require_additional_bound_evidence'):value[k]=True
    for k in ('standby_nontraining_exact_seconds','active_training_elapsed_exact_seconds','affected_active_known_subtotal','CPU_audit_standby_adjusted_exact_elapsed'):value[k]=None
    value.update(affected_active_unknown_formal_fit_count=1,affected_active_unknown_scopes=3)
    for k in ('final_plan_completion_certificate','all_future_sleep_or_timer_scopes_covered','old_cost6_or_noon_gate_already_excludes_this_timer','new_loss_closure_created','controller_or_parent_seconds_added_to_formal_cost','CPU_audit_observer_elapsed_added_to_formal_cost','native_process_continuity_independently_verified_by_this_helper'):value[k]=False
    return Reader(values),value,cost,wide

def events():
    return tuple(json.loads((M.WORK/x['path']).read_text(encoding='utf8')) for x in (M.EVENT,M.EVENT_REVIEW))

def closure_fixture():
    r,delivery=base();linked,binds=M.verify_base(r,delivery)
    for row in linked['wide']['training_rows']:row['completion']=d('fixture/'+row['fit_id']+'/complete.json')
    linked['final_cost']['captured_at']='2026-10-12T10:00:00+00:00';linked['final_cost']['cost_records']=[{'artifact':d(M.TIMERS['F3_formal_attempt'][0])}]
    r.values['fixture/registry.json']['registered_at']='2026-10-08T03:16:40+00:00'
    source=d('fixture/reviewer.py');events=d('fixture/final_range.json');r.values[source['path']]='source fixture'
    original=json.loads((M.WORK/M.EVENT['path']).read_text(encoding='utf8'));r.values[M.EVENT['path']]=original
    full_events=copy.deepcopy(original['all_power_provider_events_in_same_window'])
    fake_exit=copy.deepcopy(full_events[3]);fake_exit.update(record_id=149320,time_created_utc='2026-10-10T12:24:00Z',time_created_local='2026-10-10T20:24:00+08:00')
    fake_exit['xml']=fake_exit['xml'].replace('149310','149320').replace('2026-10-10T12:21:31.4372363Z','2026-10-10T12:24:00Z').replace("Name='ScenarioInstanceIdV2'>253<","Name='ScenarioInstanceIdV2'>256<")
    full_events.append(fake_exit)
    r.values[events['path']]={'schema':'readonly_windows_power_event_window_v1','query_status':'success','query_utc_start':'2026-10-08T00:00:00Z','query_utc_end':'2026-10-12T11:00:00Z','all_power_provider_events_in_same_window':full_events}
    review={'schema':'independent_final_system_low_power_scope_review_v1','passed':True,'final_scope_closed':True,'reviewer':'synthetic component fixture only','source':source,
        'binding':{'final_cost':binds['final_cost'],'full_plan_acceptance':binds['wide'],'formal_complete_artifacts':sorted((x['completion'] for x in linked['wide']['training_rows']),key=lambda d:d['path']),
            'original_system_event_snapshot':M.EVENT,'original_independent_event_review':M.EVENT_REVIEW},
        'all_identified_affected_standby_fit_ids':[M.FIT],'all_identified_affected_standby_formal_attempts':[d(M.TIMERS['F3_formal_attempt'][0])],
        'additional_affected_fit_ids':[],'unresolved_scenarios':[],'original_boot218_scenario256_resolved':True,'full_formal_run_event_range_reviewed':True,
        'exact_timer_subtraction_performed':False,'checks':[{'claim':'fixture','observed':'fixture','evidence_locations':['fixture']}],
        'reviewed_range_start_utc':'2026-10-08T00:00:00Z','reviewed_range_end_utc':'2026-10-12T11:00:00Z','event_range_snapshots':[events],'evidence':[events]}
    return r,review,linked,binds

class Guards(unittest.TestCase):
    def test_missing_input_pending_no_reads(self):
        r=Reader();v,_=M.audit({},r);self.assertFalse(v['passed']);self.assertEqual(v['status'],'pending');self.assertEqual(r.calls,[])
    def test_missing_later_scope_review_stays_pending_before_reads(self):
        r=Reader();v,_=M.audit({k:'fixture.json' for k in M.INPUT_NAMES if k!='scope_review'},r)
        self.assertFalse(v['passed']);self.assertEqual(v['status'],'pending');self.assertEqual(r.calls,[])
    def test_scope_component_bound_full_range_only(self):self.assertTrue(M.verify_scope_closure(*closure_fixture())['passed'])
    def test_scope_unpaired_or_extra_fit_must_reject(self):
        for key,value in [('unresolved_scenarios',[256]),('all_identified_affected_standby_fit_ids',[M.FIT,'extra']),('additional_affected_fit_ids',['extra']),('original_boot218_scenario256_resolved',False)]:
            r,v,c,b=closure_fixture();v[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
    def test_scope_cannot_self_narrow_required_actual_date_range(self):
        r,v,c,b=closure_fixture();v['reviewed_range_start_utc']='2026-10-10T11:28:00Z'
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
        r,v,c,b=closure_fixture();r.values['fixture/final_range.json']['query_utc_start']='2026-10-10T11:28:00Z'
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
        r,v,c,b=closure_fixture();r.values['fixture/final_range.json']['query_utc_end']='2026-10-11T00:00:00Z'
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
    def test_scope_not_passed_cannot_close(self):
        r,v,c,b=closure_fixture();v['final_scope_closed']=False
        with self.assertRaises(M.Pending):M.verify_scope_closure(r,v,c,b)
    def test_scope_range_headers_without_actual_events_reject(self):
        r,v,c,b=closure_fixture();del r.values['fixture/final_range.json']['all_power_provider_events_in_same_window']
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
    def test_scope_boolean_cannot_replace_real256_exit(self):
        r,v,c,b=closure_fixture();r.values['fixture/final_range.json']['all_power_provider_events_in_same_window'].pop()
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
    def test_scope_wrong256_and_overlap_conflict_reject(self):
        r,v,c,b=closure_fixture();rows=r.values['fixture/final_range.json']['all_power_provider_events_in_same_window'];rows[-1]['xml']=rows[-1]['xml'].replace("Name='ScenarioInstanceIdV2'>256<","Name='ScenarioInstanceIdV2'>257<")
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
        r,v,c,b=closure_fixture();second=copy.deepcopy(r.values['fixture/final_range.json']);second['all_power_provider_events_in_same_window'][-1]['xml']=second['all_power_provider_events_in_same_window'][-1]['xml'].replace("Name='EnergyDrain'>0<","Name='EnergyDrain'>1<");r.values['fixture/second_range.json']=second;v['event_range_snapshots'].append(d('fixture/second_range.json'))
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
    def test_scope_substantive_artifacts_required(self):
        r,v,c,b=closure_fixture();v['evidence']=[]
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
    def test_scope_original_event_omission_and_other_unpaired_reject(self):
        r,v,c,b=closure_fixture();r.values['fixture/final_range.json']['all_power_provider_events_in_same_window'].pop(0)
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
        r,v,c,b=closure_fixture();rows=r.values['fixture/final_range.json']['all_power_provider_events_in_same_window'];row=copy.deepcopy(rows[-2]);row['record_id']=149321;row['xml']=row['xml'].replace('149313','149321').replace("Name='ScenarioInstanceIdV2'>256<","Name='ScenarioInstanceIdV2'>257<");rows.append(row)
        with self.assertRaises(ValueError):M.verify_scope_closure(r,v,c,b)
    def test_base_fixture_complete(self):self.assertEqual(len(M.verify_base(*base())[0]['final_cost']['formal_fits']),360)
    def test_base_pending_30_or_cpu_incomplete_rejects(self):
        r,v=base();v['status']='pending'
        with self.assertRaises(M.Pending):M.verify_base(r,v)
        r,v=base();v['requirements'].pop()
        with self.assertRaises(ValueError):M.verify_base(r,v)
        r,v=base();r.values['fixture/final_cost.json']['source_bound_cpu_checkpoint_verification']['verified_fits']=134
        with self.assertRaises(ValueError):M.verify_base(r,v)
    def test_same_actual_cost_and_noon_binding_required(self):
        r,v=base();r.values['fixture/g3_bundle.json']['final_cost']=d('other.json')
        with self.assertRaises(ValueError):M.verify_base(r,v)
        r,v=base();v['base_final_artifacts']['pause_companion']=d('pending_noon.json')
        with self.assertRaises(ValueError):M.verify_base(r,v)
    def test_actual_bounded_XML_component_only(self):
        value=M.verify_events(*events());self.assertTrue(value['passed']);self.assertGreater(value['reported_active_residency_us'],0);self.assertIsNone(value['exact_active_seconds'])
    def test_XML_tamper_and_summary_mismatch_reject(self):
        raw,review=events();raw['all_power_provider_events_in_same_window'][1]['xml']=raw['all_power_provider_events_in_same_window'][1]['xml'].replace('>218<','>219<')
        with self.assertRaises(ValueError):M.verify_events(raw,review)
        raw,review=events();raw['all_power_provider_events_in_same_window'][1]['record_id']=0
        with self.assertRaises(ValueError):M.verify_events(raw,review)
    def test_system_sleep_does_not_make_whole_interval_zero_work(self):
        raw,review=events();review['exact_active_training_seconds']=0
        with self.assertRaises(ValueError):M.verify_events(raw,review)
    def test_actual_native_JSON_component_only_no_live_queries(self):
        value=json.loads((M.WORK/M.NATIVE['path']).read_text(encoding='utf8'));r=Reader({M.NATIVE['path']:value,M.OWNER_LINK['path']:{},M.OWNER_SNAPSHOT['path']:value['owner_snapshot']})
        c={'native_observation_verification':{'artifact':M.NATIVE,'observed_at':value['observed_at'],'original_native_objects_observed_alive':True,'continuous_activity_through_entire_window_proven':False,'live_owner_or_progress_binding_rechecked':False,'retained_owner':{'link':M.OWNER_LINK,'snapshot':M.OWNER_SNAPSHOT,'matches_original_owner_bytes':True,'current_live_owner_read':False}}}
        self.assertTrue(M.verify_native_observation(r,c)['passed']);self.assertEqual(r.calls,[M.NATIVE,M.OWNER_LINK,M.OWNER_SNAPSHOT])
        value['identities'][-1]['creation_100ns']='999';r=Reader({M.NATIVE['path']:value,M.OWNER_LINK['path']:{},M.OWNER_SNAPSHOT['path']:value['owner_snapshot']})
        with self.assertRaises(ValueError):M.verify_native_observation(r,c)
    def test_native_observation_does_not_claim_continuous_activity(self):
        value=json.loads((M.WORK/M.NATIVE['path']).read_text(encoding='utf8'));r=Reader({M.NATIVE['path']:value,M.OWNER_LINK['path']:{},M.OWNER_SNAPSHOT['path']:value['owner_snapshot']})
        c={'native_observation_verification':{'artifact':M.NATIVE,'observed_at':value['observed_at'],'original_native_objects_observed_alive':True,'continuous_activity_through_entire_window_proven':True,'live_owner_or_progress_binding_rechecked':False,'retained_owner':{'link':M.OWNER_LINK,'snapshot':M.OWNER_SNAPSHOT,'matches_original_owner_bytes':True,'current_live_owner_read':False}}}
        with self.assertRaises(ValueError):M.verify_native_observation(r,c)
        raw,review=events();review['second_entry']['current_state_not_inferred_from_bounded_window']=False
        with self.assertRaises(ValueError):M.verify_events(raw,review)
    def test_three_terminal_scopes_only(self):
        v=M.verify_standby(*timer_fixture());self.assertTrue(v['passed']);self.assertIsNone(v['affected_exact_standby_seconds'])
    def test_companion_passed_but_pending_rejects(self):
        r,v,c,w=timer_fixture();v['status']='verified_low_power_scope_with_terminal_timers_pending'
        with self.assertRaises(M.Pending):M.verify_standby(r,v,c,w)
    def test_timers_null_not_zero_no_subtraction(self):
        for key,value in [('active_training_elapsed_exact_seconds',0),('standby_nontraining_exact_seconds',0),('active_known_subtotal_contribution',0),('wall_or_residency_subtracted',True)]:
            r,v,c,w=timer_fixture();v['affected_timer_observations'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_standby(r,v,c,w)
    def test_noon_key_is_not_standby_contract(self):
        r,v,c,w=timer_fixture();row=v['affected_timer_observations'][0];row['wall_pause_subtracted']=row.pop('wall_or_residency_subtracted')
        with self.assertRaises(ValueError):M.verify_standby(r,v,c,w)
    def test_top_level_contradictions_rejected(self):
        for key,value in [('standby_nontraining_exact_seconds',0),('affected_active_unknown_formal_fit_count',True),('affected_active_unknown_scopes',1),('all_future_sleep_or_timer_scopes_covered',True),('CPU_audit_observer_elapsed_added_to_formal_cost',True)]:
            r,v,c,w=timer_fixture();v[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_standby(r,v,c,w)
    def test_original_failed_retry_requires_new_scope(self):
        r,v,c,w=timer_fixture();r.values[M.TIMERS['F3_formal_attempt'][0]]['status']='failed'
        with self.assertRaises(ValueError):M.verify_standby(r,v,c,w)
    def test_no_extra_counts_or_future_coverage_claim(self):
        for key,value in [('extra_optimizer_updates_added',True),('extra_backward_calls_added',1),('same_attempt_identity_preserved',False),('future_low_power_other_timers_require_additional_bound_evidence',False)]:
            r,v,c,w=timer_fixture();v[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_standby(r,v,c,w)
    def test_formal_subset_excludes_both_no_nested_or_CPU(self):
        _,_,c,_=timer_fixture()
        for path,seconds,category,role in [('known.json',3.0,'formal_training_attempt','primary_observed_invocation'),('unknown.json',None,'formal_training_attempt','primary_observed_invocation'),('cpu.json',90.0,'CPU_observer','primary_observed_invocation')]:c['cost_records'].append({'artifact':d(path),'seconds_observed':seconds,'category':category,'aggregation_role':role})
        v=M.formal_subset(c);self.assertEqual(v['known_recorded_formal_seconds_excluding_bound_native_pause_and_standby_timers'],3);self.assertEqual(len(v['excluded_affected_formal_timer_records']),2);self.assertEqual(len(v['retained_unknown_formal_timer_records']),1)
    def test_no_known_null_actualzero_and_strict_numbers(self):
        _,_,c,_=timer_fixture();self.assertIsNone(M.formal_subset(c)['known_recorded_formal_seconds_excluding_bound_native_pause_and_standby_timers'])
        c['cost_records'].append({'artifact':d('zero.json'),'category':'formal_training_attempt','aggregation_role':'primary_observed_invocation','seconds_observed':0})
        self.assertEqual(M.formal_subset(c)['known_recorded_formal_seconds_excluding_bound_native_pause_and_standby_timers'],0)
        for value in (True,float('nan'),float('inf'),-1):
            c['cost_records'][-1]['seconds_observed']=value
            with self.subTest(value=value),self.assertRaises(ValueError):M.formal_subset(c)
    def test_duplicate_or_missing_affected_primary_rejects(self):
        _,_,c,_=timer_fixture();c['cost_records'].append(copy.deepcopy(c['cost_records'][0]))
        with self.assertRaises(ValueError):M.formal_subset(c)
        _,_,c,_=timer_fixture();c['cost_records'].pop(0)
        with self.assertRaises(ValueError):M.formal_subset(c)
    def test_metadata_and_parent_role_guard(self):
        r,v,c,w=timer_fixture();c['cost_records'][-1]['aggregation_role']='primary_observed_invocation'
        with self.assertRaises(ValueError):M.verify_standby(r,v,c,w)
        r,v,c,w=timer_fixture();r.values['fixture/complete.json']['metadata']=d('wrong.json')
        with self.assertRaises(ValueError):M.verify_standby(r,v,c,w)
    def test_text_reader_blocks_binary_predictions_escape_and_changes(self):
        with tempfile.TemporaryDirectory(dir=P.parent) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(P.parent.resolve()));r=M.Reader(tmp)
            for path in ('model.pt','x.pth','x.safetensors','prediction.json','../escape.json'):
                with self.subTest(path=path),self.assertRaises(ValueError):r.path(path)
            p=Path(tmp)/'x.xml';p.write_text('<x/>');_,b=r.snapshot(p);self.assertEqual(r.checked(b),'<x/>');p.write_text('<y/>')
            with self.assertRaises(ValueError):r.recheck()
    def test_require_complete_pending_exit_two(self):
        with tempfile.TemporaryDirectory(dir=P.parent) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(P.parent.resolve()));fake=Reader({'fixture.py':'fixture'});fake.snapshot=lambda path:('fixture',d('fixture.py'))
            with patch.object(M,'REPORT',Path(tmp)),patch.object(M,'WORK',Path(tmp)),patch.object(M,'Reader',return_value=fake),patch.object(M,'audit',return_value=({'status':'pending','passed':False},fake)):
                self.assertEqual(M.main(['--require-complete']),2)
            self.assertTrue(all(json.loads(p.read_text())['passed'] is False for p in Path(tmp).glob('*.json')))

if __name__=='__main__':unittest.main(verbosity=2)
