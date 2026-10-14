"""Read-only full-run power-event/timer envelopes. Never estimates active time.

All inventory rows are retained, including records lacking historical endpoints.
An envelope is directional evidence or an explicitly bounded review-domain, not
the lifetime of a process. No wall-clock interval is subtracted from QPC seconds.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

SOURCE=Path(__file__).resolve(); WORK=SOURCE.parents[3]; REPORT=SOURCE.parent
REL='output/analysis/eeg_all_candidates_v1_20261008/'
OUT='tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/'
MOD='tridep_ccml/eeg_all_candidates/'
PREFIX={'path':REL+'system_power_event_full_range_prefix_1791876271590.json','sha256':'1269fec163aaf846676c88b54f2e252879467a641eeba5d7871111380c307a87'}
OLD_EVENT={'path':REL+'system_power_event_window_1791635064778.json','sha256':'8b55ffd71637b10b583daaadb1aef2f2263f10ec9e94b1c2c99194c3439d8670'}
COST_SOURCE={'path':REL+'cost_aggregation_v19.py','sha256':'e03e1832cbff9472d17377a8b38e63526f340294912aece43472c450a97cfc6d'}
WIDE_SOURCE={'path':REL+'audit_full_plan_acceptance_v9.py','sha256':'df743c655c85422091b3c371f2a73b587ca48963c642742926a4901e49c15ce7'}
Q14={'path':OUT+'continuation_queue_v14/registration.json','sha256':'93b19228fb8d3f032f87be4072f4ff2fb3a0bbe2d89adce93117896b5bc59742'}
LOSSES=[{'path':OUT+'process_loss_closures/lost_f3_1791597116209422300.json','sha256':'e56bbb201c067bf0bab31c0d405520fb2860deb12796f088072bb30bdd73de65'},
 {'path':OUT+'process_loss_closures/lost_f3_v2_1791706969935734100.json','sha256':'14286a281be2d896653fb717ea2faefb86b84b3d80e572e36795154b746bfc8e'},
 {'path':OUT+'process_loss_closures/lost_f3_v3_1791767849813215500.json','sha256':'83d4558818cd8727a4ec4b04d18ebfc18653aed6f3a03a611edb5a41f659e791'}]
NOON={'path':REL+'pause_memory_resume_cost_companion_1791625875034611700.json','sha256':'60217de0cb993be4b43bfe28cb4a0f79f9f5cfe314bf62249fd002b764a7b7db'}
NS={'e':'http://schemas.microsoft.com/win/2004/08/events/event'}

def require(ok,why):
    if not ok: raise ValueError(why)
def instant(value):
    require(isinstance(value,str),'Timestamp must be explicit text')
    x=datetime.fromisoformat(value.replace('Z','+00:00'));require(x.tzinfo is not None,'Timestamp lacks timezone');return x
def iso(value): return instant(value).astimezone(timezone.utc).isoformat()
def descriptor(path):
    p=Path(path).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

class Reader:
    def __init__(self,root=WORK):self.root=Path(root).resolve();self.inputs={}
    def snapshot(self,path):
        p=(self.root/path).resolve();require(p.is_relative_to(self.root) and p.suffix.lower() in ('.json','.py','.ps1','.xml','.md'),'Text-only workspace evidence required')
        require(not any(x in p.name.lower() for x in ('probabilit','prediction','oof_')),'Prediction payload outside scope')
        raw=p.read_bytes();d={'path':p.relative_to(self.root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}
        require(d['path'] not in self.inputs or self.inputs[d['path']]==d,'Evidence changed')
        self.inputs[d['path']]=d;return (json.loads(raw.decode('utf-8-sig')) if p.suffix=='.json' else raw.decode('utf-8-sig')),d
    def checked(self,d):
        require(type(d) is dict and set(d)=={'path','sha256'},'Exact descriptor required')
        value,observed=self.snapshot(d['path']);require(observed==d,'Descriptor bytes differ');return value
    def recheck(self):
        for d in list(self.inputs.values()):self.checked(d)

def event_inventory(reader,snapshots,start,end):
    require(PREFIX in snapshots and len(snapshots)==len({d['path'] for d in snapshots}),'Actual prefix and unique event snapshots required')
    events={};windows=[];old=reader.checked(OLD_EVENT)
    for d in snapshots:
        value=reader.checked(d);require(value.get('schema')=='readonly_windows_power_event_window_v1' and value.get('query_status')=='success','Actual provider-complete query required')
        a,b=instant(value['query_utc_start']),instant(value['query_utc_end']);require(a<=b,'Reversed query bounds')
        require(instant(value['oldest_available_system_record_utc'])<=instant(start),'System log retention omits required start')
        windows.append((a,b));seen=set()
        for row in value['all_power_provider_events_in_same_window']:
            tree=ET.fromstring(row['xml']);s=tree.find('e:System',NS)
            fields={'record_id':int(s.find('e:EventRecordID',NS).text),'provider':s.find('e:Provider',NS).attrib['Name'],
                'event_id':int(s.find('e:EventID',NS).text),'time_created_utc':s.find('e:TimeCreated',NS).attrib['SystemTime']}
            require(all(fields[k]==row[k] for k in fields),'XML/header mismatch')
            require(a<=instant(fields['time_created_utc'])<=b and fields['record_id'] not in seen,'Event outside query or duplicated')
            require(fields['provider'] in ('Microsoft-Windows-Kernel-Power','Microsoft-Windows-Power-Troubleshooter'),'Unexpected provider')
            seen.add(fields['record_id']);fields['data']={x.attrib['Name']:x.text for x in tree.findall('e:EventData/e:Data',NS)}
            fields['xml_sha256']=hashlib.sha256(row['xml'].encode('utf-8')).hexdigest()
            require(fields['record_id'] not in events or events[fields['record_id']]==fields,'Repeated record changed across queries')
            events[fields['record_id']]=fields
    cursor=instant(start)
    for a,b in sorted(windows):
        if b<cursor:continue
        require(a<=cursor,'Event range has a coverage gap');cursor=max(cursor,b)
    require(cursor>=instant(end),'Event suffix does not reach final cost capture')
    for row in old['all_power_provider_events_in_same_window']:
        require(row['record_id'] in events and events[row['record_id']]['xml_sha256']==hashlib.sha256(row['xml'].encode('utf-8')).hexdigest(),'Original bounded event omitted/changed')
    enters={};exits={};sleep=[];support=[]
    for row in sorted(events.values(),key=lambda r:r['record_id']):
        d=row['data'];eid=row['event_id']
        if row['provider']=='Microsoft-Windows-Kernel-Power' and eid in (506,507):
            k=(d.get('BootId'),d.get('ScenarioInstanceIdV2'));require(None not in k,'Missing modern-session identity')
            bucket=enters if eid==506 else exits;require(k not in bucket,'Ambiguous repeated modern-session identity');bucket[k]=row
        elif row['provider']=='Microsoft-Windows-Power-Troubleshooter' and eid==1:
            a,b=d.get('SleepTime'),d.get('WakeTime');require(instant(a)<instant(b)<=instant(row['time_created_utc']),'Invalid event-declared sleep/wake interval')
            sleep.append({'id':'sleep_event_'+str(row['record_id']),'kind':'event_declared_sleep_wake','start':a,'end':b,'records':[row['record_id']],
                'sleep_entered':None,'per_process_stopped_not_proven':True,'event_data':d})
        elif row['provider']=='Microsoft-Windows-Kernel-Power' and eid in (42,107):support.append(row)
    require(set(enters)==set(exits),'Unpaired modern-session event')
    sessions=[]
    for key,a in sorted(enters.items()):
        b=exits[key];require(a['record_id']<b['record_id'] and instant(a['time_created_utc'])<instant(b['time_created_utc']),'Reversed modern-session pair')
        d=b['data'];no_sleep=d.get('SleepEntered')=='false' and d.get('SleepDurationInUs')=='0'
        sessions.append({'id':'modern_'+key[0]+'_'+key[1],'kind':'screen_off_session_no_sleep_reported' if no_sleep else 'modern_session_residency_not_process_stop',
            'start':a['time_created_utc'],'end':b['time_created_utc'],'records':[a['record_id'],b['record_id']], 'sleep_entered':d.get('SleepEntered'),
            'per_process_stopped_not_proven':True,'event_data':d})
    # 42/107 creation timestamps are supporting sequence evidence, never sleep endpoints.
    assigned=[]
    for window in sleep:
        matches=[r for r in support if instant(window['start'])<=instant(r['time_created_utc'])<=instant(window['end']) and r['record_id']<window['records'][0]]
        ids=[r['event_id'] for r in matches]
        if matches:require(ids==[42,107],'Unmatched/ambiguous 42/107 support sequence')
        else:require(instant(window['end'])<instant(start),'In-scope sleep/wake event lacks 42/107 support')
        window['support_records']=[r['record_id'] for r in matches];assigned.extend(window['support_records'])
    require(sorted(assigned)==sorted(r['record_id'] for r in support),'Unassigned sleep transition support event')
    intervals=sorted(sessions+sleep,key=lambda r:(instant(r['start']),r['id']))
    for r in intervals:
        r['review_range_relation']='outside_registered_run' if instant(r['end'])<=instant(start) or instant(r['start'])>=instant(end) else 'intersects_registered_run'
    return {'snapshots':snapshots,'events':sorted(events.values(),key=lambda r:r['record_id']),'intervals':intervals,
        'query_windows':[{'start':a.isoformat(),'end':b.isoformat()} for a,b in sorted(windows)],'required_start':start,'required_end':end,
        'modern_session_pairs':len(sessions),'declared_sleep_wake_events':len(sleep),'original_boot218_scenario256_resolved':('218','256') in exits,
        'unpaired_or_unassigned_events':[],'all_retained_original_XML_verified':True,'42_107_creation_difference_is_not_sleep_duration':True}

def source_directions(reader,registry):
    execution=reader.checked(Q14);sources=execution['sources']
    require(len(sources)==194 and execution['registry']==descriptor(reader.root/OUT/'registration.json'),'Registered execution inheritance differs')
    # Exact registered hashes, not a fresh unbound source description.
    wanted=['registry.py','finetune.py','supervised.py','ssl.py','frozen_heads.py']
    found=[]
    for name in wanted:
        p=MOD+name
        descriptors=[x for x in sources if isinstance(x,dict) and x.get('path')==p]
        require(len(descriptors)==1,'Original registration does not pin required direction source '+name)
        d=descriptors[0];text=reader.checked(d);require('load_registration' in text,'Registration-before-work source missing');found.append(d)
    return {'execution_registration':Q14,'registered_execution_sources':found,'lower_bound':'original registration exists and is loaded before registered formal work',
        'upper_bound':'registry records training_complete with recorded_at after checking completion evidence',
        'metadata_seconds_not_used_to_derive_wall_endpoint':True}

def loss_and_pause(reader,cost):
    rows=[];mapping={}
    for desc in LOSSES:
        c=reader.checked(desc);require(c.get('passed') is True and c.get('status')=='process_absence_observed' and c.get('actual_exit_time','missing') is None,'Actual unknown-exit loss required')
        evidence=[]
        for k in ('pause_proof','pause_native_proof','pause_capture','pause_progress_and_observer','process_absence_evidence','runtime_process_absence_evidence'):
            if k in c:reader.checked(c[k]);evidence.append(c[k])
        artifacts=[c['original_attempt']]+([c['old_queue_attempt_original']] if 'old_queue_attempt_original' in c else [])
        for a in artifacts:reader.checked(a);mapping[a['path']]={'closure':desc,'observed_absent_no_later_than':c['observed_at']}
        rows.append({'closure':desc,'artifacts':artifacts,'pause_and_absence_evidence':evidence,'observed_absent_no_later_than':c['observed_at'],
            'actual_exit_time':None,'active_seconds_exact':None,'extra_work_added_again':False})
    noon=reader.checked(NOON)
    for k in ('pause_proof','native_pause'):reader.checked(noon[k])
    for call in noon['resume_verification']['resume_call_windows']:
        reader.checked(call['started_event']);reader.checked(call['returned_event'])
    return {'three_losses':rows,'noon_pause_companion':NOON,'noon_pause_observed_fields':{k:v for k,v in noon.items() if k in ('pause_wall_observed_at','pause_monotonic_exact_seconds','pause_proof','native_pause','resume_verification')},
        'no_nested_wait_pause_or_loss_cost_added':True},mapping

def overlap(a,b,x,y):return instant(a)<instant(y) and instant(x)<instant(b)

def analyze(reader,cost,cost_desc,wide,wide_desc,snapshots):
    require(cost.get('status')=='verified_final_cost' and cost.get('schema_version')==19 and cost.get('source')==COST_SOURCE,'Actual frozen final Cost19 required')
    require(wide.get('source')==WIDE_SOURCE and wide.get('full_plan_completed') is True and wide.get('objective_satisfied') is True,'Actual final wide9 required')
    require(wide.get('final_cost',{}).get('snapshot')==cost_desc and wide.get('registration')==cost.get('registration'),'Same final cost/wide binding required')
    registry=reader.checked(cost['registration']);start=registry['registered_at'];end=cost['captured_at'];directions=source_directions(reader,registry)
    events=event_inventory(reader,snapshots,start,end);pause,lossmap=loss_and_pause(reader,cost)
    fits={r['fit_id']:r for r in cost['formal_fits']};require(len(fits)==360,'All original360 required')
    bounds={}
    for fid,fit in fits.items():
        s=reader.checked(fit['status_artifact']);require(s.get('status')=='training_complete' and s.get('fit_id')==fid and s.get('registration')==cost['registration'],'Actual completed fit status binding required')
        require(s['evidence']['metadata']==fit['metadata_artifact'],'Metadata completion binding differs')
        require(instant(start)<=instant(s['recorded_at'])<=instant(end),'Formal completion outside registration/capture')
        bounds[fid]={'start':start,'end':s['recorded_at'],'evidence':[cost['registration'],fit['status_artifact']], 'kind':'registration_before_formal_work_and_recorded_completion_after_work'}
    rows=[]
    for index,row in enumerate(cost['cost_records']):
        artifact=row['artifact'];data=reader.checked(artifact);data=data if isinstance(data,dict) else {}
        fid=row.get('fit_id')
        if fid is None:
            identity=data.get('binding',data);condition=identity.get('condition') or row.get('condition');seed=identity.get('seed');fold=identity.get('fold')
            candidate=str(condition)+'/seed_'+str(seed)+'/fold_'+str(fold)
            if candidate in fits:fid=candidate
        a=row.get('started_at') or data.get('started_at');b=row.get('finished_at') or data.get('finished_at')
        if a:instant(a)
        if b:instant(b)
        exact=bool(a and b);why=[];evidence=[artifact];lo,hi=a,b
        if exact:require(instant(a)<=instant(b),'Timer endpoints reversed')
        elif artifact['path'] in lossmap:
            loss=lossmap[artifact['path']];hi=loss['observed_absent_no_later_than'];lo=a or start;evidence.append(loss['closure']);why.append('process_loss_observation_is_upper_bound_not_actual_exit')
        elif fid in bounds and row['category']=='formal_legacy_metadata':
            bound=bounds[fid];lo=a or bound['start'];hi=b or bound['end'];evidence+=bound['evidence'];why.append(bound['kind'])
        elif not a and b and instant(b)<instant(start):
            # A known pre-registration finish excludes the registered domain,
            # without inventing a start or a zero-duration historical invocation.
            lo=hi=start;why.append('recorded_finish_before_registered_review_domain_empty_intersection_not_zero_runtime')
        else:
            lo=a or start;hi=b or end;why.append('full_registered_run_review_domain_only_not_a_claim_of_actual_process_endpoints')
        require(instant(lo)<=instant(hi),'Conservative envelope reversed')
        comparisons=[]
        for event in events['intervals']:
            inside=event['review_range_relation']=='intersects_registered_run'
            crosses=inside and instant(lo)<instant(hi) and overlap(lo,hi,event['start'],event['end'])
            relation=('recorded_timer_interval_overlap' if exact else 'possible_overlap_unknown_exact_endpoint') if crosses else 'outside_timer_review_envelope'
            comparisons.append({'event_id':event['id'],'relation':relation})
        rows.append({'inventory_index':index,'artifact':artifact,'category':row['category'],'aggregation_role':row['aggregation_role'],'fit_id':fid,
            'producer_status':row.get('status'),'exact_started_at':a,'exact_finished_at':b,'both_recorded_endpoints':exact,
            'review_envelope':{'start':lo,'end':hi,'basis':why or ['producer_recorded_wall_endpoints'],'evidence':evidence},
            'recorded_seconds_observed':row.get('seconds_observed'),'event_relations':comparisons,'active_seconds_exact':None,'nontraining_seconds_exact':None,
            'costs_or_counters_added_again':False,'endpoint_unknowns_retained':not exact})
    affected=sorted({r['fit_id'] for r in rows if r['fit_id'] and any(x['relation']!='outside_timer_review_envelope' for x in r['event_relations'])})
    categories=Counter(r['category'] for r in rows)
    return {'schema':'full_system_power_timer_scope_v1','final_cost':cost_desc,'full_plan_acceptance':wide_desc,'registration':cost['registration'],
        'formal_complete_artifacts':sorted((r['completion'] for r in wide['training_rows']),key=lambda x:x['path']),
        'review_range':{'start':start,'end':end},'event_inventory':events,'directional_source_evidence':directions,'pause_and_loss_scope':pause,
        'timer_rows':rows,'inventory_count':len(rows),'category_counts':dict(sorted(categories.items())),
        'all_identified_possibly_affected_fit_ids':affected,'unknown_exact_endpoint_count':sum(not r['both_recorded_endpoints'] for r in rows),
        'all_original360_completion_bounds':bounds,'unresolved_event_or_inventory_coverage':[],
        'scope_is_complete_over_registered_run_only':True,'before_registration_cost_scope_not_claimed_to_have_complete_events':True,
        'all_plan_exact_active_seconds':None,'affected_exact_active_seconds':None,'exact_timer_subtraction_performed':False,
        'old_active_named_subtotals_not_valid_as_full_plan_active_seconds':True,'recorded_cost_bytes_unchanged':True,
        'modern_session_is_not_per_process_stopped_time':True,'nested_overlap_seconds_not_summed':True,
        'new_model_Tensor_GPU_process_calls':0}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--final-cost');p.add_argument('--wide');p.add_argument('--event-snapshot',action='append',default=[])
    args=p.parse_args(argv)
    if not args.run:print(json.dumps({'status':'prepared_only','operations':0}));return 0
    require(args.final_cost and args.wide and args.event_snapshot,'Complete actual inputs required')
    r=Reader();cost,cd=r.snapshot(args.final_cost);wide,wd=r.snapshot(args.wide);ds=[r.snapshot(x)[1] for x in args.event_snapshot]
    value=analyze(r,cost,cd,wide,wd,ds);value['source']=descriptor(SOURCE);r.checked(value['source']);r.recheck()
    value.update(created_at=datetime.now(timezone.utc).isoformat(),input_bindings=list(r.inputs.values()),status='full_scope_analyzed_active_time_unknown')
    target=REPORT/('full_system_power_timer_scope_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(descriptor(target)));return 0

if __name__=='__main__':raise SystemExit(main())
