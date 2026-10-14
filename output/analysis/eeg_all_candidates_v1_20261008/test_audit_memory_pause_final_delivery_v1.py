"""Pure stdlib/in-memory guards; no actual final certificate or model is run."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('audit_memory_pause_final_delivery_v1.py')
spec=importlib.util.spec_from_file_location('memory_delivery_under_test',SOURCE)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def desc(path): return {'path':path,'sha256':'a'*64}


class FakeReader:
    def __init__(self, values=None): self.values=values or {};self.calls=[];self.inputs={}
    def checked(self,binding):
        self.calls.append(binding);return copy.deepcopy(self.values[binding['path']])
    def snapshot(self,path):
        self.calls.append(path);return copy.deepcopy(self.values[str(path)]),desc(str(path))
    def recheck(self): pass


def base_fixture():
    ids=[M.FIT]+['synthetic/%03d'%i for i in range(359)]
    bindings={k:desc('synthetic/'+k+'.json') for k in M.INPUT_NAMES}
    train=[{'fit_id':x,'updates':4,'completion':desc('synthetic/'+x+'/complete.json')} for x in ids]
    parent=desc('synthetic/parent.json')
    cost={'source':M.PINNED['cost'],'actual_cost_collector_source':M.PINNED['cost'],'finalizer_source':M.PINNED['finalizer'],
        'status':'verified_final_cost','final_snapshot':True,'draft_not_completion_certificate':False,
        'all_available_artifacts_accounted':True,'coverage_gaps_disclosed':True,'schema_version':6,'registration':M.REGISTRY,
        'formal_fits_expected':360,'formal_fits':[{'fit_id':x,'formal_durable_updates':4} for x in ids],
        'formal_durable_updates_known_subtotal':1440,'formal_status_counts':{'training_complete':360},'input_read_errors':[],
        'source_bound_cpu_checkpoint_verification':{'passed':True,'verified_fits':135,'prior_actual_cpu_checkpoint_loads':405,'new_cpu_checkpoint_loads':0},
        'human_readable_cost_report':desc('synthetic/cost.md'),'all_plan_wall_seconds_exact':None,'all_plan_physical_optimizer_returns_exact':None,
        'cost_records':[{'artifact':parent,'category':'dispatcher_elapsed','aggregation_role':'supplemental_nested_or_unmeasured','seconds_observed':40.0}]}
    wide={'source':M.PINNED['wide'],'objective_satisfied':True,'full_plan_completed':True,'schema_version':6,'registration':M.REGISTRY,
        'training_rows':train,'status_counts':{'training_complete':360},'verification_errors':[],'incomplete_fits':[],
        'formal_identity_count':360,'completed_training_json_verified':360,'deep_cpu_checkpoint_scope_verified':360,'cpu_checkpoint_proofs':[{}]*360,
        'requirements':[{'id':'R%02d'%i,'status':'verified_complete'} for i in range(1,31)],
        'evaluation':{'passed':True,'complete':desc('synthetic/eval.json')},'semantic_review':{'passed':True,'artifact':bindings['semantic_review']},
        'final_cost':{'passed':True,'snapshot':bindings['final_cost']},'q5_execution_lifecycle':{'passed':True}}
    semantic={'kind':'independent_final_semantic_review','passed':True,'reviewer':'synthetic fixture only',
        'binding':{'registration':M.REGISTRY,'audit_source':M.PINNED['wide'],'formal_complete_artifacts':sorted([x['completion'] for x in train],key=lambda x:x['path']),
            'evaluation_complete':wide['evaluation']['complete'],'cost_snapshot':bindings['final_cost']},
        'items':[{'id':i,'conclusion':'verified','finding':'fixture','checks':[{'claim':'fixture','observed':'fixture','evidence_locations':['fixture']}],'evidence':[desc('synthetic/source.py')]} for i in sorted(M.SEMANTIC_IDS)]}
    bundle={'source':M.PINNED['g3'],'passed':True,'schema':'eeg_final_cost_with_sync_boundary_v3_bundle_v1','final_cost':bindings['final_cost'],
        'full_plan_acceptance':bindings['wide'],'final_cost_human_report':cost['human_readable_cost_report'],
        'all_probe_primaries_already_in_final_cost_once':True,'companion_costs_added_again':False,'full_plan_verification_rerun':False,'GPU_or_evaluation_calls':0,
        'existing_final_artifact_shape_verified':{'registered_and_verified_fits':360,'existing_finetune_CPU_fit_proof':135,'existing_finetune_CPU_load_proof':405,'scientific_checks_reexecuted':False},
        'parent_attempt_timer':{'artifact':parent,'seconds_observed_including_hold_and_probe':40.0,'seconds_added_to_formal_or_primary_probe':False,'pure_training_seconds':None,'hold_subtracted_to_infer_training_time':False}}
    return cost,wide,semantic,bundle,bindings,{'fits':[{'fit_id':x} for x in ids]}


def timer_fixture():
    cost={'cost_records':[],'formal_fits':[]};wide={'training_rows':[]};values={};timers=[]
    for scope,(path,category,role,field) in M.TIMERS.items():
        binding=desc(path);seconds=101.25
        row={'scope':scope,'status':'actual_terminal','artifact':binding,'recorded_elapsed_including_native_pause_seconds':seconds,
            'active_training_elapsed_exact_seconds':None,'pause_monotonic_exact_seconds':None,'active_known_subtotal_contribution':None,
            'eligible_for_active_known_subtotal':False,'producer_elapsed_preserved':True,'wall_pause_subtracted':False,
            'nested_with_formal_attempt':scope!='F3_formal_attempt'}
        producer={field:seconds,'condition':'REVE-lastblock-finetune','seed':22,'fold':2,'status':'complete',
            'graceful_terminal_flush':True,'counter_snapshot_exact_for_process':True,'returncode':0}
        if scope=='F1_metadata_subscope':
            row['completion_binding']=desc('synthetic/complete.json');values['synthetic/complete.json']={'passed':True,'metadata':binding}
            cost['formal_fits'].append({'fit_id':M.FIT,'metadata_artifact':binding,'last_attempt_seconds':seconds})
            wide['training_rows'].append({'fit_id':M.FIT,'completion':row['completion_binding']})
        timers.append(row);values[path]=producer
        cost['cost_records'].append({'artifact':binding,'category':category,'aggregation_role':role,'seconds_observed':seconds})
    pause={'schema':'pause_memory_resume_cost_companion_v1','source':M.PINNED['pause'],'passed':True,
        'status':'verified_pause_scope_and_recorded_terminal_timers','all_affected_producer_timers_terminal':True,
        'affected_formal_fit_id':M.FIT,'pause_monotonic_exact_seconds':None,'active_training_elapsed_exact_seconds':None,'affected_active_known_subtotal':None,
        'pause_boundary_QPC_recorded':False,'old_cost6_already_excludes_this_completed_pause':False,'pause_snapshot_counters_added_to_terminal':False,
        'new_loss_closure_created':False,'affected_active_unknown_formal_fit_count':1,'affected_active_unknown_scopes':3,
        'unknown_scope_count_includes_nested_metadata_and_parent_not_three_formal_fits':True,'affected_timer_observations':timers}
    return FakeReader(values),pause,cost,wide


def resume_fixture():
    release=desc('synthetic/release.json');pre=desc('synthetic/preflight.json');archives=[desc('synthetic/control%d.json'%i) for i in range(4)]
    identities={k:{'pid':v[0],'creation_100ns':v[1]} for k,v in M.ROLES.items()}
    resume={'status':'actual_native_resume_and_controls_release_verified','native_resume':M.NATIVE_RESUME,'controls_release':release,
        'preflight':pre,'same_native_identities':identities,'pause_monotonic_exact_seconds':None,'active_training_elapsed_exact_seconds':None,'archived_controls':archives}
    pause={'resumer_source':M.PINNED['resumer'],'pause_proof':M.PAUSE_PROOF,'native_pause':M.NATIVE_PAUSE,'resume_verification':resume,'immutable_inputs':[]}
    review={'schema':'independent_q5_memory_resume_review_v1','passed':True,'review_source':M.PINNED['resume_review'],
        'resume_source':M.PINNED['resumer'],'controls_release_evidence':release,'same_four_original_objects':True,
        'total_original_threads_verified':67,'total_original_threads_still_suspended_after_recorded_resume':0,
        'processes':[{'role':role,**identity,'native_suspend_calls_in_bound_pause':1,'native_resume_calls_in_bound_resume':1,'native_resume_status':0,'post_resume_original_suspended':0} for role,identity in identities.items()],
        'exact_pause_elapsed_seconds':None,'exact_active_elapsed_seconds':None,'pause_clock':{'pause_boundary_qpc':None},'inputs':[M.PAUSE_PROOF,M.NATIVE_PAUSE,M.NATIVE_RESUME]}
    values={x['path']:{} for x in [release,pre,*archives,M.PAUSE_PROOF,M.NATIVE_PAUSE]}
    values[M.NATIVE_RESUME['path']]={'source':M.PINNED['resumer'],'status':'same_four_objects_resumed_by_explicit_continue',
        'actions':[{'role':k,'native_status':0} for k in identities]}
    return FakeReader(values),pause,review,M.RESUME_REVIEW


class Guards(unittest.TestCase):
    def test_missing_final_inputs_pending_without_reads(self):
        r=FakeReader();value,_=M.audit({},r);self.assertEqual(value['status'],'pending');self.assertFalse(value['passed']);self.assertEqual(r.calls,[])
    def test_base_complete_in_memory_only(self): self.assertEqual(len(M.verify_base(*base_fixture())),30)
    def test_base_partial_and_mismatch_rejected(self):
        for field,value in [('full_plan_completed',False),('deep_cpu_checkpoint_scope_verified',359)]:
            args=base_fixture();args[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):M.verify_base(*args)
        args=base_fixture();args[1]['final_cost']['snapshot']=desc('different.json')
        with self.assertRaises(ValueError):M.verify_base(*args)
    def test_semantic_must_bind_same_cost_and_30_unique_requirements(self):
        args=base_fixture();args[2]['binding']['cost_snapshot']=desc('different.json')
        with self.assertRaises(ValueError):M.verify_base(*args)
        args=base_fixture();args[1]['requirements'][-1]=args[1]['requirements'][0]
        with self.assertRaises(ValueError):M.verify_base(*args)
    def test_exact135405_and_no_new_loads(self):
        for field,value in [('verified_fits',True),('prior_actual_cpu_checkpoint_loads',404),('new_cpu_checkpoint_loads',1)]:
            args=base_fixture();args[0]['source_bound_cpu_checkpoint_verification'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):M.verify_base(*args)
    def test_g3_parent_supplemental_not_active(self):
        args=base_fixture();args[3]['parent_attempt_timer']['pure_training_seconds']=0
        with self.assertRaises(ValueError):M.verify_base(*args)
        args=base_fixture();args[0]['cost_records'][0]['aggregation_role']='primary_observed_invocation'
        with self.assertRaises(ValueError):M.verify_base(*args)
    def test_terminal_timers_keep_raw_and_unknown(self):
        value=M.verify_pause_timers(*timer_fixture());self.assertIsNone(value['affected_active_known_seconds_subtotal']);self.assertEqual(value['active_unknown_formal_fits'],1)
        self.assertTrue(all(x['recorded_inclusive_seconds']==101.25 and x['active_known_contribution'] is None for x in value['observed_timers']))
    def test_pending_companion_passed_flag_is_insufficient(self):
        args=timer_fixture();args[1]['status']='verified_pause_scope_with_terminal_timers_pending'
        with self.assertRaises(M.Pending):M.verify_pause_timers(*args)
    def test_pause_null_cannot_be_zero_or_subtracted(self):
        for key,value in [('active_training_elapsed_exact_seconds',0),('active_known_subtotal_contribution',0),('wall_pause_subtracted',True),('eligible_for_active_known_subtotal',True)]:
            args=timer_fixture();args[1]['affected_timer_observations'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_pause_timers(*args)
    def test_timer_roles_duplicates_nested_counts(self):
        args=timer_fixture();args[2]['cost_records'].append(copy.deepcopy(args[2]['cost_records'][0]))
        with self.assertRaises(ValueError):M.verify_pause_timers(*args)
        args=timer_fixture();args[1]['affected_active_unknown_formal_fit_count']=3
        with self.assertRaises(ValueError):M.verify_pause_timers(*args)
        args=timer_fixture();args[2]['cost_records'][2]['aggregation_role']='primary_observed_invocation'
        with self.assertRaises(ValueError):M.verify_pause_timers(*args)
    def test_timer_numbers_strict(self):
        for number in (True,float('inf'),float('nan'),-1):
            args=timer_fixture();args[1]['affected_timer_observations'][0]['recorded_elapsed_including_native_pause_seconds']=number
            with self.subTest(number=number),self.assertRaises(ValueError):M.verify_pause_timers(*args)
    def test_producer_or_metadata_binding_mismatch_rejected(self):
        args=timer_fixture();args[0].values[M.TIMERS['F3_formal_attempt'][0]]['seconds']=99
        with self.assertRaises(ValueError):M.verify_pause_timers(*args)
        args=timer_fixture();args[0].values['synthetic/complete.json']['metadata']=desc('other.json')
        with self.assertRaises(ValueError):M.verify_pause_timers(*args)
    def test_failed_original_attempt_needs_explicit_new_scope(self):
        args=timer_fixture();args[0].values[M.TIMERS['F3_formal_attempt'][0]]['status']='failed'
        with self.assertRaises(ValueError):M.verify_pause_timers(*args)
    def test_known_formal_subset_excludes_pause_nested_and_unknown(self):
        _,_,cost,_=timer_fixture()
        def row(path,seconds):return {'artifact':desc(path),'category':'formal_training_attempt','aggregation_role':'primary_observed_invocation','seconds_observed':seconds}
        cost['cost_records'].extend([row('known1.json',3.25),row('known2.json',0),row('unknown.json',None)])
        value=M.recorded_formal_subset(cost)
        self.assertEqual(value['known_recorded_formal_seconds_excluding_affected_native_pause_timer'],3.25)
        self.assertEqual(len(value['retained_unknown_formal_timer_records']),1)
        self.assertEqual(len(value['excluded_native_pause_formal_timer_records']),1)
        self.assertFalse(value['subset_is_exact_active_training_time'])
    def test_known_formal_empty_null_but_actual_zero_zero(self):
        _,_,cost,_=timer_fixture();value=M.recorded_formal_subset(cost)
        self.assertIsNone(value['known_recorded_formal_seconds_excluding_affected_native_pause_timer'])
        cost['cost_records'].append({'artifact':desc('zero.json'),'category':'formal_training_attempt','aggregation_role':'primary_observed_invocation','seconds_observed':0})
        self.assertEqual(M.recorded_formal_subset(cost)['known_recorded_formal_seconds_excluding_affected_native_pause_timer'],0)
        cost['cost_records'][-1]['seconds_observed']=True
        with self.assertRaises(ValueError):M.recorded_formal_subset(cost)
    def test_resume_four_same_objects_once(self):self.assertTrue(M.verify_resume(*resume_fixture())['passed'])
    def test_resume_birth_count_and_return_guard(self):
        for key,value in [('creation_100ns','wrong'),('native_resume_calls_in_bound_resume',2),('native_resume_status',True),('post_resume_original_suspended',1)]:
            args=resume_fixture();args[2]['processes'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_resume(*args)
    def test_resume_unknown_time_and_archive_count(self):
        args=resume_fixture();args[2]['exact_pause_elapsed_seconds']=123
        with self.assertRaises(ValueError):M.verify_resume(*args)
        args=resume_fixture();args[1]['resume_verification']['archived_controls'].pop()
        with self.assertRaises(ValueError):M.verify_resume(*args)
    def test_reader_binary_prediction_escape_and_changed_hash(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(SOURCE.parent.resolve()))
            r=M.Reader(tmp)
            for path in ('x.pt','weights.pth','x.safetensors','prediction.json','oof_x.json','../escape.json'):
                with self.subTest(path=path),self.assertRaises(ValueError):r.path(path)
            p=Path(tmp)/'allowed.json';p.write_text('{"x":1}',encoding='utf-8');_,binding=r.snapshot(p)
            self.assertEqual(r.checked(binding),{'x':1});p.write_text('{"x":2}',encoding='utf-8')
            with self.assertRaises(ValueError):r.recheck()
    def test_require_complete_pending_exit2_only(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as tmp,patch.object(M,'REPORT',Path(tmp)),patch.object(M,'WORK',Path(tmp)):
            self.assertTrue(Path(tmp).resolve().is_relative_to(SOURCE.parent.resolve()))
            fake=FakeReader({'fixture.py':'fixture'});fake.snapshot=lambda path: ('fixture',desc('fixture.py'))
            with patch.object(M,'Reader',return_value=fake),patch.object(M,'audit',return_value=({'status':'pending','passed':False},fake)):
                self.assertEqual(M.main(['--require-complete']),2)
            files=list(Path(tmp).glob('*.json'));self.assertEqual(len(files),1);self.assertFalse(json.loads(files[0].read_text())['passed'])


if __name__=='__main__':unittest.main(verbosity=2)
