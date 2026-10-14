"""Full-range timer-scope gate retaining the original bounded low-power proof.

Standard-library text/JSON/XML reads only. No earlier verifier is imported or
executed. Producer timers stay unchanged; exclusions are not active-time estimates.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

SOURCE=Path(__file__).resolve();REPORT=SOURCE.parent;WORK=SOURCE.parents[3]
REL='output/analysis/eeg_all_candidates_v1_20261008/'
OUT='tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/'
FIT='LUNA-lastblock-finetune/seed_33/fold_1'
def binding(path,sha):return {'path':path,'sha256':sha}
MEMORY_SOURCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/audit_memory_pause_final_delivery_v4.py', 'sha256': '0072923160057c18932f6ccdf05de4ec673e647916ff25ccd015a3fab6b0b71f'}
STANDBY_SOURCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/system_standby_cost_companion_v4.py', 'sha256': '9d4f389c4d1d7dc98e1ed7eaf5f27e23d328dafca48735924f8aa5fff4c55e5f'}
EVENT=binding(REL+'system_power_event_window_1791635064778.json','8b55ffd71637b10b583daaadb1aef2f2263f10ec9e94b1c2c99194c3439d8670')
EVENT_REVIEW=binding(REL+'independent_system_sleep_overlap_review_1791635154808605000.json','29727e34ca640854af6e6594677da32a0495797dc490737193affbccda1c2544')
NATIVE=binding(OUT+'monitoring/system_standby/native_observation_1791635509483.json','c2b499e646c93698c4baee377be514104b0558b382d0153a2e5b19410a38c83c')
OWNER_LINK=binding(OUT+'monitoring/system_standby/owner_snapshot_link_1791635509483.json','adcb75e61501f9516fd54c1fb57fb20773ba4d5f5326355286a7716cd006fdcf')
OWNER_SNAPSHOT=binding(OUT+'monitoring/system_standby/owner_exact_1791635509483.json','83dced47231f58bd28ef41c4f00d106cc36331c1b5d86851710865eb52eb0791')
NATIVE_OBJECTS={38860:(50164,'134360707153787861'),45980:(38860,'134360707154222972'),25396:(45980,'134360988331542432'),50296:(25396,'134360988331626948')}
NOON=binding(REL+'pause_memory_resume_cost_companion_1791625875034611700.json','60217de0cb993be4b43bfe28cb4a0f79f9f5cfe314bf62249fd002b764a7b7db')
PINNED={'cost': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/cost_aggregation_v19.py', 'sha256': 'e03e1832cbff9472d17377a8b38e63526f340294912aece43472c450a97cfc6d'}, 'finalizer': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/finalize_full_plan_cost_v9.py', 'sha256': '4752ada431a8008c7b97a8908210bd45dbd55e2f6e68a037c6f4f4b9757b70ac'}, 'wide': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/audit_full_plan_acceptance_v9.py', 'sha256': 'df743c655c85422091b3c371f2a73b587ca48963c642742926a4901e49c15ce7'}, 'g3': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/sync_boundary_v3_cost_companion_v4.py', 'sha256': '214b6c7fa1ec0a00b31cecf7514b1c7dbdc8127cf2048c0258897a759a240944'}, 'pause': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/pause_memory_resume_cost_companion_v4.py', 'sha256': '379c913fc0e6fb479e1685f0ba37c81f6e5280e452b8e967a893ba8d53eb2999'}}
REVE_F3=OUT+'finetune_v2_attempts/REVE-lastblock-finetune/f_1791607293628173200_cafa8752/attempt.json'
TIMERS={
 'F3_formal_attempt':(OUT+'finetune_v2_attempts/LUNA-lastblock-finetune/f_1791631397731004200_7e6d59cd/attempt.json','formal_training_attempt','primary_observed_invocation','seconds'),
 'F1_metadata_subscope':(OUT+'runs/'+FIT+'/metadata.json','formal_legacy_metadata','supplemental_durable_metadata','seconds_this_attempt'),
 'Q5_parent_dispatcher':(OUT+'continuation_queue_v5/attempts/1791625233145752000-LUNA-lastblock-finetune-fit/status.json','dispatcher_elapsed','supplemental_nested_or_unmeasured','seconds')}
EXCLUDED={REVE_F3,TIMERS['F3_formal_attempt'][0]}
FULL_SCOPE_SOURCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/full_system_power_timer_scope_v1.py', 'sha256': 'c28bdd912c3a97510cd17d725ce7542c078640ec3e77f2c48340d6c0e923caec'}
INPUT_NAMES=('memory_pause_delivery','standby_companion','event_review','scope_review')
REQUIREMENTS={'R%02d'%i for i in range(1,31)}

def require(ok,reason):
    if not ok:raise ValueError(reason)
def number(v):return type(v) in (int,float) and math.isfinite(v) and v>=0
class Pending(ValueError):pass

class Reader:
    def __init__(self,root=WORK):self.root=Path(root).resolve();self.inputs={}
    def path(self,path):
        p=(self.root/path).resolve()
        require(p.is_relative_to(self.root) and p.suffix.lower() in ('.py','.json','.xml','.md','.ps1'),'Only allowlisted workspace source/JSON/XML/Markdown evidence permitted')
        require(not any(x in p.name.lower() for x in ('probabilit','prediction','oof_')),'Prediction content is outside this gate')
        return p
    def snapshot(self,path):
        p=self.path(path);raw=p.read_bytes();d=binding(p.relative_to(self.root).as_posix(),hashlib.sha256(raw).hexdigest())
        require(d['path'] not in self.inputs or self.inputs[d['path']]==d,'Evidence changed during review')
        self.inputs[d['path']]=d
        return (json.loads(raw.decode('utf-8-sig')) if p.suffix.lower()=='.json' else raw.decode('utf-8-sig')),d
    def checked(self,d):
        require(type(d) is dict and set(d)=={'path','sha256'},'Exact source descriptor required')
        value,actual=self.snapshot(d['path']);require(actual==d,'Source/evidence hash differs');return value
    def recheck(self):
        for d in list(self.inputs.values()):self.checked(d)

def all30(value):
    rows=value.get('requirements',[])
    return len(rows)==30 and {r['id'] for r in rows}==REQUIREMENTS and all(r.get('status')=='verified_complete' for r in rows)

def verify_base(reader,delivery):
    require(delivery.get('schema')=='memory_pause_final_delivery_gate_v4' and delivery.get('source')==MEMORY_SOURCE,'Actual sealed memory-pause gate required')
    if not (delivery.get('status')=='verified_final_delivery_with_memory_pause' and delivery.get('passed') is True
            and delivery.get('objective_satisfied') is True and delivery.get('full_plan_completed') is True):
        raise Pending('Original memory-pause final delivery has not completed')
    require(all30(delivery) and delivery.get('R01_R30_derived_after_all_prerequisites') is True,'Original complete30 unavailable')
    require(delivery.get('R28_pause_timer_scope',{}).get('passed') is True and delivery.get('R29_memory_resume_control',{}).get('passed') is True,'Noon pause gates absent')
    require(delivery.get('all_plan_exact_active_seconds','missing') is None and delivery.get('base_recorded_elapsed_totals_not_relabelled_as_active') is True,'Base elapsed was relabelled active')
    bindings=delivery['base_final_artifacts']
    require(set(bindings)=={'final_cost','wide','semantic_review','g3_bundle','pause_companion','resume_review'},'Exact six inputs including actual terminal noon report required')
    values={k:reader.checked(v) for k,v in bindings.items()};cost=values['final_cost'];wide=values['wide'];bundle=values['g3_bundle'];semantic=values['semantic_review']
    require(cost.get('schema_version')==19 and type(cost['schema_version']) is int and cost.get('source')==PINNED['cost']
            and cost.get('actual_cost_collector_source')==PINNED['cost'] and cost.get('finalizer_source')==PINNED['finalizer']
            and cost.get('status')=='verified_final_cost' and cost.get('final_snapshot') is True,'Actual final cost6 differs')
    require(wide.get('source')==PINNED['wide'] and wide.get('objective_satisfied') is True and wide.get('full_plan_completed') is True and all30(wide),'Actual complete wide9/30 unavailable')
    require(wide['final_cost'].get('passed') is True and wide['final_cost'].get('snapshot')==bindings['final_cost']
            and wide['semantic_review'].get('artifact')==bindings['semantic_review'] and wide['semantic_review'].get('passed') is True,'Wide binding differs')
    require(bundle.get('source')==PINNED['g3'] and bundle.get('passed') is True and bundle.get('final_cost')==bindings['final_cost']
            and bundle.get('full_plan_acceptance')==bindings['wide'] and bundle.get('companion_costs_added_again') is False,'Same actual G3 bundle required')
    require(semantic.get('passed') is True and semantic.get('binding',{}).get('cost_snapshot')==bindings['final_cost']
            and semantic['binding'].get('audit_source')==PINNED['wide'],'Same actual semantic proof required')
    registry=reader.checked(cost['registration']);expected={x['fit_id'] for x in registry['fits']}
    require(len(registry['fits'])==len(expected)==360 and cost.get('registration')==wide.get('registration'),'Original360 differs')
    for rows in (cost.get('formal_fits',[]),wide.get('training_rows',[])):
        require(len(rows)==360 and {x['fit_id'] for x in rows}==expected,'Final360 actual fit coverage differs')
    require(cost.get('formal_status_counts')==wide.get('status_counts')=={'training_complete':360}
            and cost.get('input_read_errors')==wide.get('verification_errors')==wide.get('incomplete_fits')==[],'Base incomplete/error remains')
    cpu=cost.get('source_bound_cpu_checkpoint_verification',{})
    require(cpu.get('passed') is True and all(type(cpu.get(k)) is int and cpu[k]==v for k,v in
            (('verified_fits',135),('prior_actual_cpu_checkpoint_loads',405),('new_cpu_checkpoint_loads',0))),'Actual prior135/405 CPU proof differs')
    require(values['pause_companion'].get('all_affected_producer_timers_terminal') is True and values['pause_companion'].get('original_noon_companion')==NOON
        and values['pause_companion'].get('source')==PINNED['pause'] and delivery.get('three_loss_lifecycle_scope')==values['pause_companion'].get('loss_chain')
        and delivery['three_loss_lifecycle_scope'].get('passed') is True,'Noon terminal and three actual losses unavailable')
    return values,bindings

def verify_events(raw,review):
    require(raw.get('schema')=='readonly_windows_power_event_window_v1' and raw.get('query_status')=='success','Successful original event window required')
    require(review.get('schema')=='independent_system_sleep_overlap_review_v1' and review.get('passed_for_observed_event_evidence') is True
            and review.get('source_event_snapshot')==EVENT,'Independent actual event review differs')
    structured=[];ns={'e':'http://schemas.microsoft.com/win/2004/08/events/event'}
    for row in raw['all_power_provider_events_in_same_window']:
        tree=ET.fromstring(row['xml']);sys=tree.find('e:System',ns);require(sys is not None,'Raw XML System absent')
        data={n.attrib['Name']:n.text for n in tree.findall('e:EventData/e:Data',ns)}
        observed={'record_id':int(sys.find('e:EventRecordID',ns).text),'provider':sys.find('e:Provider',ns).attrib['Name'],
            'event_id':int(sys.find('e:EventID',ns).text),'time_created_utc':sys.find('e:TimeCreated',ns).attrib['SystemTime'],
            'time_created_local':row['time_created_local'],'data':data}
        require(all(observed[k]==row[k] for k in ('record_id','provider','event_id','time_created_utc')),'Raw XML summary differs')
        structured.append(observed)
    require(len(structured)==7 and structured==review.get('structured_provider_event_data'),'Independent structured data differs from raw XML')
    by_id={x['record_id']:x for x in structured};require(len(by_id)==7,'Repeated event records')
    entry,end,unpaired=[by_id[k] for k in (149303,149310,149313)]
    require(entry['event_id']==506 and end['event_id']==507 and entry['data']['BootId']==end['data']['BootId']=='218'
            and entry['data']['ScenarioInstanceIdV2']==end['data']['ScenarioInstanceIdV2']=='253','Boot/scenario pair differs')
    require(end['data']['SleepEntered']=='true' and int(end['data']['ActiveResidencyInUs'])>0,'System interval includes active residency; do not imply zero work')
    require(unpaired['event_id']==506 and unpaired['data']['ScenarioInstanceIdV2']=='256'
            and not any(x['event_id']==507 and x['data'].get('ScenarioInstanceIdV2')=='256' for x in structured),'Unpaired entry scope differs')
    require(review.get('second_entry',{}).get('current_state_not_inferred_from_bounded_window') is True,'Unpaired entry cannot establish current sleep')
    require(all(k in review and review[k] is None for k in ('exact_nontraining_interval_seconds','exact_active_training_seconds','exact_deduction_from_producer_timers_seconds'))
            and review.get('producer_timers_modified') is False and review.get('tool_latency_treated_as_training_or_pure_pause') is False,'Unsafe timer inference from system data')
    return {'passed':True,'system_event_snapshot':EVENT,'independent_event_review':EVENT_REVIEW,
        'boot_id':218,'paired_scenario_instance_v2':253,'unpaired_scenario_instance_v2':256,
        'reported_active_residency_us':int(end['data']['ActiveResidencyInUs']),
        'system_interval_is_not_per_process_zero_work':True,'exact_nontraining_seconds':None,'exact_active_seconds':None,
        'current_sleep_state_inferred':False,'future_low_power_other_timers_require_additional_bound_evidence':True}

def formal_subset(cost):
    known=[];unknown=[];excluded=[];seen=set()
    for row in cost['cost_records']:
        if row.get('category')!='formal_training_attempt' or row.get('aggregation_role')!='primary_observed_invocation':continue
        d=row['artifact'];require(d['path'] not in seen,'Duplicate primary formal record');seen.add(d['path'])
        if d['path'] in EXCLUDED:excluded.append(d);continue
        seconds=row.get('seconds_observed')
        if seconds is None:unknown.append(d)
        else:require(number(seconds),'Invalid known primary seconds');known.append({'artifact':d,'recorded_seconds':seconds})
    require(len(excluded)==2 and {d['path'] for d in excluded}==EXCLUDED,'Both affected formal timers must be excluded exactly once')
    return {'known_recorded_formal_seconds_excluding_bound_native_pause_and_standby_timers':sum(x['recorded_seconds'] for x in known) if known else None,
        'known_recorded_formal_timer_records':known,'retained_unknown_formal_timer_records':unknown,'excluded_affected_formal_timer_records':excluded,
        'metadata_dispatcher_CPU_observer_seconds_added':False,'subset_is_exact_active_training_time':False,'unknown_records_zero_filled':False}

def verify_native_observation(reader,companion):
    proof=companion.get('native_observation_verification',{})
    require(proof.get('artifact')==NATIVE and proof.get('original_native_objects_observed_alive') is True
            and proof.get('continuous_activity_through_entire_window_proven') is False and proof.get('live_owner_or_progress_binding_rechecked') is False,'Bound native observation scope differs')
    observed=reader.checked(NATIVE)
    require(observed.get('schema')=='system_standby_readonly_native_observation_v1' and proof.get('observed_at')==observed.get('observed_at'),'Actual native observation differs')
    objects=observed.get('identities',[])
    require(len(objects)==4 and {r['pid'] for r in objects}==set(NATIVE_OBJECTS),'Four observed native objects required')
    entry=datetime.fromisoformat('2026-10-10T11:30:15.131953+00:00')
    entry_filetime=int((entry-datetime(1601,1,1,tzinfo=timezone.utc)).total_seconds()*10000000)
    for row in objects:
        parent,birth=NATIVE_OBJECTS[row['pid']]
        require(type(row['pid']) is int and row.get('parent_pid')==parent and row.get('creation_100ns')==birth
                and int(birth)<entry_filetime and row.get('alive') is True
                and type(row.get('native_exit_code')) is int and row['native_exit_code']==259
                and type(row.get('suspended_threads')) is int and row['suspended_threads']==0,'Native birth/chain/observed state differs')
    owner=observed['owner_snapshot']
    require(owner.get('pid')==45980 and owner.get('child_pid')==25396 and owner.get('child_job')=='LUNA-lastblock-finetune-fit'
            and type(owner.get('execution_version')) is int and owner['execution_version']==5,'Embedded owner scope differs')
    require(datetime.fromisoformat(observed['observed_at'])>datetime.fromisoformat('2026-10-10T12:21:31.437236+00:00')
            and observed.get('no_pause_or_replay_added_by_observer') is True
            and all(type(x) is int and x==0 for x in observed['operations'].values()),'Observation did not remain read-only after the paired exit')
    retained=proof.get('retained_owner',{})
    require(retained.get('link')==OWNER_LINK and retained.get('snapshot')==OWNER_SNAPSHOT
            and retained.get('matches_original_owner_bytes') is True and retained.get('current_live_owner_read') is False,'Retained owner scope differs')
    reader.checked(OWNER_LINK);snapshot=reader.checked(OWNER_SNAPSHOT)
    require(snapshot==owner and observed['owner_binding']['sha256']==OWNER_SNAPSHOT['sha256'],'Immutable original owner bytes differ')
    return {'passed':True,'artifact':NATIVE,'observed_at':observed['observed_at'],'native_objects':NATIVE_OBJECTS,
        'objects_born_before_paired_entry_observed_alive_after_exit':True,'continuous_activity_through_entire_window_proven':False,
        'claims_current_liveness':False,'live_owner_or_progress_paths_rechecked':False,'new_restart_loss_closure_or_counts_inferred':False}

def verify_standby(reader,value,cost,wide,bindings):
    require(value.get('schema')=='system_standby_cost_companion_v4' and value.get('source')==STANDBY_SOURCE,'Standby companion source/schema differs')
    if not (value.get('status')=='verified_low_power_scope_and_closed_timer_lifecycles' and value.get('all_affected_timer_lifecycles_closed') is True):
        raise Pending('Every standby timer requires real terminal or exact immutable lost-event closure')
    require(value.get('passed') is True and value.get('affected_formal_fit_id')==FIT,'Actual standby fit scope differs')
    require(value.get('final_scope_closure_ready') is False and value.get('final_scope_closure_required') is True,'Bounded companion cannot itself close final event coverage')
    require(all(k in value and value[k] is None for k in ('standby_nontraining_exact_seconds','active_training_elapsed_exact_seconds','affected_active_known_subtotal','CPU_audit_standby_adjusted_exact_elapsed')),'Top-level unknown standby/active/observer seconds must remain null')
    require(type(value.get('affected_active_unknown_formal_fit_count')) is int and value['affected_active_unknown_formal_fit_count']==1
            and type(value.get('affected_active_unknown_scopes')) is int and value['affected_active_unknown_scopes']==3,'Standby scopes are one fit, not three')
    for key in ('final_plan_completion_certificate','all_future_sleep_or_timer_scopes_covered','old_cost6_or_noon_gate_already_excludes_this_timer',
                'new_loss_closure_created','controller_or_parent_seconds_added_to_formal_cost','CPU_audit_observer_elapsed_added_to_formal_cost','native_process_continuity_independently_verified_by_this_helper'):
        require(value.get(key) is False,'Unsupported top-level standby claim: '+key)
    require(value.get('system_event_snapshot')==EVENT and value.get('independent_system_event_review')==EVENT_REVIEW,'Standby system evidence differs')
    for key in ('observed_low_power_overlap','system_interval_is_not_per_process_zero_work','no_exact_timer_subtraction',
                'no_new_process_or_loss_closure_created','same_attempt_identity_preserved','future_low_power_other_timers_require_additional_bound_evidence'):
        require(value.get(key) is True,'Required standby scope claim absent: '+key)
    for key in ('extra_optimizer_updates_added','extra_backward_calls_added'):
        require(type(value.get(key)) is int and value[key]==0,'Standby must not create additional work counts')
    timers=value.get('affected_timer_observations',[]);require(len(timers)==3 and {r['scope'] for r in timers}==set(TIMERS),'Three distinct standby timer scopes required')
    helper_path=reader.path(PINNED['pause']['path']);source=reader.checked(PINNED['pause'])
    helper={'__file__':str(helper_path),'__name__':'_text_only_lost_scope'}
    exec(compile(source,str(helper_path),'exec'),helper)
    pair=helper['verify_final_pair'](reader.checked,cost,wide,bindings['final_cost'],bindings['wide'])
    losses=helper['verify_loss_chain'](reader.checked,cost)
    require(value.get('final_artifacts')==pair and value.get('loss_chain')==losses,'Standby companion belongs to different final evidence')
    require(value.get('all_affected_producer_timers_terminal') is False,'Lost LUNA parent cannot claim producer-terminal')
    observed=[]
    for row in timers:
        kind=row['scope'];path,category,role,field=TIMERS[kind];d=row.get('artifact',{})
        if kind=='Q5_parent_dispatcher':
            matches=[r for r in losses['lost_parent_scopes'] if r['artifact']==d and d['path']==path]
            require(len(matches)==1 and row.get('status')=='lost_event_closed_unknown_elapsed'
                and row.get('loss_closure')==matches[0]['closure'] and row.get('lifecycle_closed_by_actual_loss') is True
                and row.get('producer_status')=='running' and row.get('producer_terminal') is False
                and row.get('producer_terminal_status','missing') is None,'LUNA parent must bind actual loss2, not a fabricated terminal')
            require(all(k in row and row[k] is None for k in ('recorded_elapsed_including_observed_low_power_window_seconds',
                'active_training_elapsed_exact_seconds','standby_nontraining_exact_seconds','active_known_subtotal_contribution')),'Lost parent time must remain unknown')
            require(row.get('producer_elapsed_preserved') is True and row.get('eligible_for_active_known_subtotal') is False
                and row.get('wall_or_residency_subtracted') is False and row.get('nested_with_formal_attempt') is True,'Lost nested timer accounting differs')
            producer=reader.checked(d);require(producer.get('status')=='running' and producer.get(field) is None and producer.get('returncode') is None,'Original lost parent was rewritten')
            observed.append({'scope':kind,**matches[0],'recorded_inclusive_seconds':None,'exact_active_seconds':None,'exact_standby_seconds':None,'active_known_contribution':None})
            continue
        require(row.get('status')=='actual_terminal' and d.get('path')==path,'Timer path/state differs')
        seconds=row.get('recorded_elapsed_including_observed_low_power_window_seconds');require(number(seconds) and seconds>0,'Actual inclusive seconds required')
        require(all(k in row and row[k] is None for k in ('active_training_elapsed_exact_seconds','standby_nontraining_exact_seconds','active_known_subtotal_contribution')),'Unknown active/standby time must stay null')
        require(row.get('eligible_for_active_known_subtotal') is False and row.get('producer_elapsed_preserved') is True
                and row.get('wall_or_residency_subtracted') is False and row.get('nested_with_formal_attempt') is (kind!='F3_formal_attempt'),'Unsafe standby timer classification')
        records=[x for x in cost['cost_records'] if x['artifact']['path']==path]
        require(len(records)==1 and records[0]['artifact']==d and records[0]['category']==category and records[0]['aggregation_role']==role
                and number(records[0].get('seconds_observed')) and records[0]['seconds_observed']==seconds,'Final cost timer absent/duplicate/different')
        producer=reader.checked(d);require(number(producer.get(field)) and producer[field]==seconds,'Bound actual producer time differs')
        if kind=='F1_metadata_subscope':
            require((producer.get('condition'),producer.get('seed'),producer.get('fold'))==('LUNA-lastblock-finetune',33,1),'Metadata identity differs')
            complete=reader.checked(row['completion_binding']);require(complete.get('passed') is True and complete.get('metadata')==d,'Complete metadata binding differs')
            fit=[x for x in cost['formal_fits'] if x['fit_id']==FIT];wf=[x for x in wide['training_rows'] if x['fit_id']==FIT]
            require(len(fit)==len(wf)==1 and fit[0]['metadata_artifact']==d and fit[0]['last_attempt_seconds']==seconds and wf[0]['completion']==row['completion_binding'],'Fit nested timer binding differs')
        else:
            require(producer.get('status')=='complete','Failed/retried original invocation requires new explicit scope review')
            if kind=='F3_formal_attempt':
                require((producer.get('condition'),producer.get('seed'),producer.get('fold'))==('LUNA-lastblock-finetune',33,1)
                        and producer.get('graceful_terminal_flush') is True and producer.get('counter_snapshot_exact_for_process') is True,'F3 terminal identity/flush differs')
            else:require(type(producer.get('returncode')) is int and producer['returncode']==0,'Q5 parent did not complete normally')
        observed.append({'scope':kind,'artifact':d,'recorded_inclusive_seconds':seconds,'exact_active_seconds':None,'exact_standby_seconds':None,'active_known_contribution':None,'added_again':False})
    for d in value.get('immutable_inputs',[]):reader.checked(d)
    reader.checked(value['human_readable_report'])
    return {'passed':True,'affected_fit':FIT,'observed_timers':observed,'original_timers_modified':False,'wall_QPC_event_duration_subtracted':False,
        'affected_exact_active_seconds':None,'affected_exact_standby_seconds':None,**formal_subset(cost)}

def verify_scope_closure(reader,review,linked,bindings):
    require(review.get('schema')=='independent_final_system_low_power_scope_review_v2','Independent full-range v2 review required')
    if review.get('passed') is not True or review.get('final_scope_closed') is not True: raise Pending('Independent complete-range review has not passed')
    require(isinstance(review.get('reviewer'),str) and review['reviewer'].strip(),'Independent reviewer required')
    reader.checked(review['source']);helper_source=reader.checked(FULL_SCOPE_SOURCE)
    scope=reader.checked(review['scope_analysis']);require(scope.get('source')==FULL_SCOPE_SOURCE and scope.get('status')=='full_scope_analyzed_active_time_unknown','Actual complete-range scope artifact required')
    namespace={'__file__':str(reader.path(FULL_SCOPE_SOURCE['path'])),'__name__':'private_full_scope_gate_verification'}
    exec(compile(helper_source,namespace['__file__'],'exec'),namespace)
    expected=namespace['analyze'](reader,linked['final_cost'],bindings['final_cost'],linked['wide'],bindings['wide'],scope['event_inventory']['snapshots'])
    require({k:scope.get(k) for k in expected}==expected,'Full timer/event scope differs from exact independent recomputation')
    required_binding={'final_cost':bindings['final_cost'],'full_plan_acceptance':bindings['wide'],'scope_analysis':review['scope_analysis'],
        'formal_complete_artifacts':expected['formal_complete_artifacts'],'original_system_event_snapshot':EVENT,'original_independent_event_review':EVENT_REVIEW}
    require(review.get('binding')==required_binding,'Independent final scope inputs differ')
    require(review.get('all_identified_possibly_affected_fit_ids')==expected['all_identified_possibly_affected_fit_ids']
        and review.get('inventory_count')==expected['inventory_count'] and review.get('unresolved_scenarios')==[],'Independent timer inventory/unknown disclosure differs')
    require(expected['unresolved_event_or_inventory_coverage']==[] and expected['event_inventory']['original_boot218_scenario256_resolved'] is True,'Event or inventory coverage unresolved')
    require(review.get('exact_timer_subtraction_performed') is False and review.get('all_plan_exact_active_seconds','missing') is None
        and review.get('old_active_named_subtotals_not_valid_as_full_plan_active_seconds') is True,'Review relabelled recorded elapsed as active')
    checks=review.get('checks',[]);require(checks and all(x.get('claim') and x.get('observed') and x.get('evidence_locations') for x in checks),'Substantive independent observed checks required')
    require(review.get('evidence'),'Independent input evidence required')
    for d in review['evidence']:reader.checked(d)
    return {'passed':True,'reviewer':review['reviewer'],'source':review['source'],'scope_analysis':review['scope_analysis'],
        'event_snapshots':expected['event_inventory']['snapshots'],'required_start':expected['review_range']['start'],'required_end':expected['review_range']['end'],
        'all_identified_possibly_affected_fit_ids':expected['all_identified_possibly_affected_fit_ids'],'inventory_count':expected['inventory_count'],
        'unknown_exact_endpoint_count':expected['unknown_exact_endpoint_count'],'all_inventory_rows_retained':True,'unresolved_scenarios':[],
        'conservative_envelopes_are_not_process_lifetimes':True,'system_sessions_do_not_prove_per_process_stoppage':True,
        'all_plan_exact_active_seconds':None,'exact_timer_subtraction_performed':False,'old_active_named_subtotals_not_valid_as_full_plan_active_seconds':True}


def audit(paths,reader=None):
    reader=reader or Reader();base={'schema':'system_standby_final_delivery_gate_v5','status':'pending','passed':False,'objective_satisfied':False,'full_plan_completed':False}
    if any(not paths.get(k) for k in INPUT_NAMES):return {**base,'reason':'Actual complete memory-pause delivery, terminal standby companion, event review and later complete-range scope review required'},reader
    try:
        values={};inputs={}
        for k in INPUT_NAMES:values[k],inputs[k]=reader.snapshot(paths[k])
        require(inputs['event_review']==EVENT_REVIEW,'Fixed actual independent event review required')
        for d in (MEMORY_SOURCE,STANDBY_SOURCE,*PINNED.values()):reader.checked(d)
        linked,linked_bindings=verify_base(reader,values['memory_pause_delivery'])
        event=verify_events(reader.checked(EVENT),values['event_review'])
        timers=verify_standby(reader,values['standby_companion'],linked['final_cost'],linked['wide'],linked_bindings)
        event['native_observation']=verify_native_observation(reader,values['standby_companion'])
        event['final_scope_closure']=verify_scope_closure(reader,values['scope_review'],linked,linked_bindings)
        rows=[dict(x) for x in values['memory_pause_delivery']['requirements']]
        for row in rows:
            if row['id'] in ('R28','R29'):row['additional_system_standby_gate']='bounded_standby_timer_scope' if row['id']=='R28' else 'bounded_system_event_and_same_invocation_evidence'
            if row['id'] in ('R01','R30'):row['derived_from_all_R02_R29_plus_memory_and_standby_gates']=True
        reader.recheck()
        return {**base,'status':'verified_final_delivery_with_bound_standby_scope','passed':True,'objective_satisfied':True,'full_plan_completed':True,
            'input_artifacts':inputs,'base_final_artifacts':linked_bindings,'requirements':rows,'R28_standby_timer_scope':timers,'R29_system_event_scope':event,
            'R01_R30_derived_after_all_prerequisites':True,'old_finalized_artifacts_modified':False,'all_plan_exact_active_seconds':None,
            'covers_complete_registered_run_inventory_with_unknown_endpoints_disclosed':True,'future_after_final_capture_not_covered':True,
            'old_bounded_companion_active_subtotal_not_valid_as_full_plan_active':True,'exact_timer_subtraction_performed':False,
            'original_scientific_or_final_cost_audits_reexecuted':False},reader
    except Pending as e:return {**base,'reason':str(e)},reader
    except (OSError,ValueError,KeyError,TypeError,ET.ParseError) as e:return {**base,'status':'failed','reason':str(e),'error_type':type(e).__name__},reader

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for k in INPUT_NAMES:parser.add_argument('--'+k.replace('_','-'),type=Path)
    parser.add_argument('--require-complete',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args(argv)
    if args.require_complete and not args.run: parser.error('--require-complete requires --run')
    if not args.run: print(json.dumps({'status':'prepared_only','operations':0}));return 0
    reader=Reader();_,source=reader.snapshot(SOURCE);value,reader=audit({k:getattr(args,k) for k in INPUT_NAMES},reader);reader.checked(source)
    value.update(source=source,created_at=datetime.now(timezone.utc).isoformat(),input_bindings=list(reader.inputs.values()),
        operations={'binary_tensor_model_GPU_process_calls':0,'project_verifier_calls':0,'prior_artifact_modifications':0})
    target=REPORT/('system_standby_final_delivery_v5_'+value['status']+'_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(binding(target.relative_to(WORK).as_posix(),hashlib.sha256(target.read_bytes()).hexdigest())))
    return 0 if value['passed'] or (value['status']=='pending' and not args.require_complete) else 2

if __name__=='__main__':raise SystemExit(main())
