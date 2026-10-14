"""Independent actual full-range inventory review; no producer imports or models."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,re,time
from pathlib import Path
import xml.etree.ElementTree as ET
SOURCE=Path(__file__).resolve();A=SOURCE.parent;WORK=SOURCE.parents[3]
REL='output/analysis/eeg_all_candidates_v1_20261008/'
OUT='tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/'
FIXED={
 'cost':(REL+'final_full_plan_cost_1791878935037825200.json','1213c45eaceeb78b0644ade79aa71a9b9638d05a5b7e6bce3f87d4a228515354'),
 'wide':(REL+'full_plan_acceptance_v9_1791880718467386600.json','a25798add925ba129df97aff7f2c2e2043fc90e568b6977479676d84f27d65aa'),
 'scope':(REL+'full_system_power_timer_scope_1791881536952146900.json','6b9f3f6cc223ba782eab79955e8fb2ed026324224a3fae56b460fa677f7c7e23'),
 'prefix':(REL+'system_power_event_full_range_prefix_1791876271590.json','1269fec163aaf846676c88b54f2e252879467a641eeba5d7871111380c307a87'),
 'suffix':(REL+'system_power_event_full_range_suffix_1791879383255.json','2781a3d581eb0d1dcba0916e15e83b61e9bdd07baaa12f6a09a57b0d50caa66b'),
 'source_review':(REL+'independent_full_power_scope_source_review_1791878543512820500.json','4f8a16a25a90f2b2b677f2a62e3454e1ffac6f399c98215ea3799d873b3eb2dc'),
 'helper':(REL+'full_system_power_timer_scope_v1.py','c28bdd912c3a97510cd17d725ce7542c078640ec3e77f2c48340d6c0e923caec'),
 'original_event':(REL+'system_power_event_window_1791635064778.json','8b55ffd71637b10b583daaadb1aef2f2263f10ec9e94b1c2c99194c3439d8670'),
 'original_review':(REL+'independent_system_sleep_overlap_review_1791635154808605000.json','29727e34ca640854af6e6594677da32a0495797dc490737193affbccda1c2544')}
def require(ok,reason):
    if not ok:raise ValueError(reason)
def binding(item):return {'path':item[0],'sha256':item[1]}
def descriptor(path):
    p=Path(path).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def ticks(value):
    m=re.fullmatch(r'(.+?)(Z|[+-]\d{2}:\d{2})',value);require(m is not None,'Timestamp timezone absent')
    body,tz=m.groups();head,sep,fraction=body.partition('.');require(not fraction or (fraction.isdigit() and len(fraction)<=7),'Unsupported timestamp precision')
    base=datetime.fromisoformat(head+('+00:00' if tz=='Z' else tz)).astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
    return (base.days*86400+base.seconds)*10000000+int((fraction+'0000000')[:7])

def run():
    cache={};inputs={};checks=[]
    def read(d):
        p=(WORK/d['path']).resolve();require(p.is_relative_to(WORK) and p.suffix in ('.json','.py','.ps1','.xml','.md'),'Text evidence outside scope')
        require(not any(x in p.name.lower() for x in ('probabilit','prediction','oof_')),'Probability payload forbidden')
        if d['path'] in cache:require(inputs[d['path']]==d,'Conflicting descriptor');return cache[d['path']]
        raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==d['sha256'],'Evidence changed: '+d['path']);inputs[d['path']]=d
        cache[d['path']]=json.loads(raw.decode('utf-8-sig')) if p.suffix=='.json' else raw.decode('utf-8-sig');return cache[d['path']]
    def check(claim,ok,observed,evidence):
        require(ok,claim);checks.append({'claim':claim,'observed':observed,'evidence_locations':evidence})
    ds={k:binding(v) for k,v in FIXED.items()};values={k:read(d) for k,d in ds.items()};cost,wide,scope=values['cost'],values['wide'],values['scope']
    check('Actual same immutable cost/wide/scope chain',cost['status']=='verified_final_cost' and cost['schema_version']==19 and wide['final_cost']['snapshot']==ds['cost'] and wide['objective_satisfied'] is True and wide['full_plan_completed'] is True and scope['final_cost']==ds['cost'] and scope['full_plan_acceptance']==ds['wide'] and scope['source']==ds['helper'],'Cost19 and wide9 actual bytes agree with the scope',[ds[k] for k in ('cost','wide','scope')])
    check('Thirty original requirements actually complete',len(wide['requirements'])==30 and {r['id'] for r in wide['requirements']}=={'R%02d'%i for i in range(1,31)} and all(r['status']=='verified_complete' for r in wide['requirements']),'30/30 verified_complete; no model or CPU re-audit here',[ds['wide']])
    registry=read(cost['registration']);lo=ticks(registry['registered_at']);hi=ticks(cost['captured_at']);require(lo<hi,'Reversed review range')
    check('Review bounds cover registration through actual final cost capture',scope['review_range']=={'start':registry['registered_at'],'end':cost['captured_at']},scope['review_range'],[cost['registration'],ds['cost']])
    p,s=values['prefix'],values['suffix'];require(ticks(p['query_utc_start'])<=lo and ticks(p['query_utc_end'])==ticks(s['query_utc_start']) and ticks(s['query_utc_end'])==hi,'Provider query range gap')
    require(all(x['query_status']=='success' and ticks(x['oldest_available_system_record_utc'])<=lo for x in (p,s)),'Retention/query failure')
    check('Provider-complete contiguous event range including verified empty suffix',s['all_power_provider_events_in_same_window']==[] and s['native_query_returned_no_matching_events'] is True and scope['event_inventory']['snapshots']==[ds['prefix'],ds['suffix']],'Prefix and suffix join at exact 100ns timestamp; suffix ends at final cost capture; oldest retained log July8',[ds['prefix'],ds['suffix']])
    namespace={'e':'http://schemas.microsoft.com/win/2004/08/events/event'};raws={};events={}
    for query in (p,s):
        require(query['system_log']=='System','Wrong event log')
        require("Provider[@Name='Microsoft-Windows-Kernel-Power']" in query['query_xpath'] and "Provider[@Name='Microsoft-Windows-Power-Troubleshooter']" in query['query_xpath'],'Provider-filter coverage absent')
        for row in query['all_power_provider_events_in_same_window']:
            xml=ET.fromstring(row['xml']);system=xml.find('e:System',namespace);eid=int(system.find('e:EventID',namespace).text);rid=int(system.find('e:EventRecordID',namespace).text)
            record={'record_id':rid,'provider':system.find('e:Provider',namespace).attrib['Name'],'event_id':eid,'time_created_utc':system.find('e:TimeCreated',namespace).attrib['SystemTime'],'data':{n.attrib['Name']:n.text for n in xml.findall('e:EventData/e:Data',namespace)},'xml_sha256':hashlib.sha256(row['xml'].encode()).hexdigest()}
            require(all(record[k]==row[k] for k in ('record_id','provider','event_id','time_created_utc')),'XML/header disagreement')
            require(ticks(query['query_utc_start'])<=ticks(record['time_created_utc'])<=ticks(query['query_utc_end']),'Event outside query')
            require(rid not in events or events[rid]==record,'Conflicting duplicate event');events[rid]=record;raws[rid]=row['xml']
    require(sorted(events.values(),key=lambda x:x['record_id'])==scope['event_inventory']['events'],'Parsed event inventory differs')
    for row in values['original_event']['all_power_provider_events_in_same_window']:require(raws[row['record_id']]==row['xml'],'Original event bytes not retained')
    enter={};leave={};intervals={};support=[]
    for row in events.values():
        data=row['data'];eid=row['event_id'];provider=row['provider']
        if provider=='Microsoft-Windows-Kernel-Power' and eid in (506,507):
            key=(data['BootId'],data['ScenarioInstanceIdV2']);target=enter if eid==506 else leave;require(key not in target,'Repeated session identity');target[key]=row
        if provider=='Microsoft-Windows-Power-Troubleshooter' and eid==1:
            a,b=data['SleepTime'],data['WakeTime'];require(ticks(a)<ticks(b)<=ticks(row['time_created_utc']),'Invalid event-declared sleep window');intervals['sleep_event_'+str(row['record_id'])]=(a,b,'event_declared_sleep_wake',[row['record_id']])
        if provider=='Microsoft-Windows-Kernel-Power' and eid in (42,107):support.append(row)
    require(set(enter)==set(leave),'Unpaired session')
    for key in enter:
        first,last=enter[key],leave[key];a,b=first['time_created_utc'],last['time_created_utc'];require(first['record_id']<last['record_id'] and ticks(a)<ticks(b),'Reversed modern pair');data=last['data']
        kind='screen_off_session_no_sleep_reported' if data.get('SleepEntered')=='false' and data.get('SleepDurationInUs')=='0' else 'modern_session_residency_not_process_stop'
        intervals['modern_'+'_'.join(key)]=(a,b,kind,[first['record_id'],last['record_id']])
    lookup={x['id']:x for x in scope['event_inventory']['intervals']};require(set(lookup)==set(intervals),'Missing system interval')
    assigned=[]
    for key,(a,b,kind,rids) in intervals.items():
        x=lookup[key];require((x['start'],x['end'],x['kind'],x['records'])==(a,b,kind,rids) and x['per_process_stopped_not_proven'] is True,'Scope event classification differs')
        require(x['review_range_relation']==('intersects_registered_run' if ticks(a)<hi and lo<ticks(b) else 'outside_registered_run'),'Wrong registered event relation')
        if key.startswith('sleep_event_'):
            matches=sorted((v for v in support if ticks(a)<=ticks(v['time_created_utc'])<=ticks(b) and v['record_id']<rids[0]),key=lambda v:v['record_id'])
            require([v['event_id'] for v in matches]==[42,107] or (not matches and ticks(b)<lo),'Unpaired support sequence')
            require(x['support_records']==[v['record_id'] for v in matches],'Wrong support binding');assigned.extend(x['support_records'])
    require(sorted(assigned)==sorted(x['record_id'] for x in support),'Unassigned/duplicate transition support')
    check('All actual events independently parsed and paired',len(events)==208 and len(enter)==31 and len(intervals)==38 and ('218','256') in leave,'208 raw records, 31 modern sessions, 7 sleep/wake windows; all support assigned',[ds['scope'],ds['prefix'],ds['suffix'],ds['original_event']])
    check('Long screen-off interval cannot be deducted as stopped time',lookup['modern_220_11']['kind']=='screen_off_session_no_sleep_reported' and leave[('220','11')]['data']['SleepEntered']=='false','Boot220/scenario11 spans Oct12 15:17Z to Oct13 01:09Z; SleepDurationInUs=0, not a 9.9h training stop',[ds['prefix'],ds['scope']])
    fits={x['fit_id']:x for x in cost['formal_fits']};require(len(fits)==len(cost['formal_fits'])==360 and set(fits)=={x['fit_id'] for x in registry['fits']}=={x['fit_id'] for x in wide['training_rows']},'Formal identities differ')
    bounds={}
    for fid,row in fits.items():
        status=read(row['status_artifact']);require(status['fit_id']==fid and status['status']=='training_complete' and status['registration']==cost['registration'] and status['evidence']['metadata']==row['metadata_artifact'],'Completion status mismatch')
        require(lo<=ticks(status['recorded_at'])<=hi,'Completion outside run');bounds[fid]=status['recorded_at']
    lossmap={}
    for row in scope['pause_and_loss_scope']['three_losses']:
        closure=read(row['closure']);require(closure['passed'] and closure['status']=='process_absence_observed' and closure['actual_exit_time'] is None,'Invalid loss closure')
        original=[closure['original_attempt']]+([closure['old_queue_attempt_original']] if 'old_queue_attempt_original' in closure else [])
        require(row['artifacts']==original and row['observed_absent_no_later_than']==closure['observed_at'] and row['actual_exit_time'] is None and row['extra_work_added_again'] is False,'Loss disclosure changed')
        for d in original:read(d);lossmap[d['path']]=(closure['observed_at'],row['closure'])
        for d in row['pause_and_absence_evidence']:read(d)
    require(len(scope['pause_and_loss_scope']['three_losses'])==3,'Missing loss')
    require(len(scope['timer_rows'])==len(cost['cost_records'])==scope['inventory_count']==1240,'Inventory omitted')
    possible=set();relations=0;category=Counter();unknown=0;loss_count=0
    for i,(c,t) in enumerate(zip(cost['cost_records'],scope['timer_rows'])):
        d=read(c['artifact']);d=d if isinstance(d,dict) else {};a=c.get('started_at') or d.get('started_at');b=c.get('finished_at') or d.get('finished_at')
        for k in ('started_at','finished_at'):
            if c.get(k) and d.get(k):require(c[k]==d[k],'Contradictory artifact/ledger timestamp')
        fid=c.get('fit_id')
        if fid is None:
            identity=d.get('binding',d);candidate=str(identity.get('condition') or c.get('condition'))+'/seed_'+str(identity.get('seed'))+'/fold_'+str(identity.get('fold'))
            if candidate in fits:fid=candidate
        exact=bool(a and b)
        if exact: lower,upper=a,b;basis=['producer_recorded_wall_endpoints']; evidence=[c['artifact']]
        elif c['artifact']['path'] in lossmap:
            upper,closure=lossmap[c['artifact']['path']];lower=a or registry['registered_at'];basis=['process_loss_observation_is_upper_bound_not_actual_exit'];evidence=[c['artifact'],closure];loss_count+=1
        elif fid in bounds and c['category']=='formal_legacy_metadata':lower=a or registry['registered_at'];upper=b or bounds[fid];basis=['registration_before_formal_work_and_recorded_completion_after_work'];evidence=[c['artifact'],cost['registration'],fits[fid]['status_artifact']]
        elif not a and b and ticks(b)<lo:lower=upper=registry['registered_at'];basis=['recorded_finish_before_registered_review_domain_empty_intersection_not_zero_runtime'];evidence=[c['artifact']]
        else:lower=a or registry['registered_at'];upper=b or cost['captured_at'];basis=['full_registered_run_review_domain_only_not_a_claim_of_actual_process_endpoints'];evidence=[c['artifact']]
        require(ticks(lower)<=ticks(upper),'Reversed envelope')
        require(t['inventory_index']==i and t['artifact']==c['artifact'] and t['category']==c['category'] and t['aggregation_role']==c['aggregation_role'] and t['fit_id']==fid and t['producer_status']==c.get('status') and t['recorded_seconds_observed']==c.get('seconds_observed'),'Inventory row relabelled or omitted')
        require(t['exact_started_at']==a and t['exact_finished_at']==b and t['both_recorded_endpoints']==exact and t['endpoint_unknowns_retained']==(not exact),'Unknown endpoint changed')
        require(t['review_envelope']=={'start':lower,'end':upper,'basis':basis,'evidence':evidence},'Conservative envelope or evidence changed')
        require(t['active_seconds_exact'] is None and t['nontraining_seconds_exact'] is None and t['costs_or_counters_added_again'] is False,'Invented active time/cost')
        actual_rel={x['event_id']:x['relation'] for x in t['event_relations']};require(len(actual_rel)==len(t['event_relations'])==38,'Event relation omitted/duplicated')
        for key,(x,y,_,_) in intervals.items():
            crosses=ticks(x)<hi and lo<ticks(y) and ticks(lower)<ticks(upper) and ticks(lower)<ticks(y) and ticks(x)<ticks(upper)
            expect=('recorded_timer_interval_overlap' if exact else 'possible_overlap_unknown_exact_endpoint') if crosses else 'outside_timer_review_envelope'
            require(actual_rel[key]==expect,'Wrong 100ns interval relation at row '+str(i))
            if crosses and fid:possible.add(fid)
            relations+=1
        category[t['category']]+=1;unknown+=not exact
    check('All recorded timer rows and event relations independently recomputed at 100ns precision',relations==1240*38 and unknown==513 and loss_count==5 and dict(sorted(category.items()))==scope['category_counts'],'1,240 rows, 47,120 relations, 727 explicit double endpoints, 513 unknown endpoints, 5 loss-bound original timers retained',[ds['cost'],ds['scope']])
    check('Possible fit set is conservative coverage, not proof of actual interruption',sorted(possible)==scope['all_identified_possibly_affected_fit_ids'] and len(possible)==360,'All360 enter possible set due broad directional envelopes; this does not claim all360 actually slept or paused',[ds['scope']])
    complete=sorted((r['completion'] for r in wide['training_rows']),key=lambda x:x['path'])
    require(complete==scope['formal_complete_artifacts'],'Completion set differs')
    check('Unknown active time and original producer bytes preserved',scope['all_plan_exact_active_seconds'] is None and scope['exact_timer_subtraction_performed'] is False and scope['old_active_named_subtotals_not_valid_as_full_plan_active_seconds'] is True and scope['unresolved_event_or_inventory_coverage']==[],'No QPC/wall subtraction, no replacement of running statuses or unknown exit times, no nested cost addition',[ds['cost'],ds['scope']])
    for d in inputs.values():require(descriptor(WORK/d['path'])==d,'Evidence changed after review')
    value={'schema':'independent_final_system_low_power_scope_review_v2','passed':True,'final_scope_closed':True,'reviewer':'loss3_independent_review independent agent','source':descriptor(SOURCE),'scope_analysis':ds['scope'],'binding':{'final_cost':ds['cost'],'full_plan_acceptance':ds['wide'],'scope_analysis':ds['scope'],'formal_complete_artifacts':complete,'original_system_event_snapshot':ds['original_event'],'original_independent_event_review':ds['original_review']},'all_identified_possibly_affected_fit_ids':sorted(possible),'inventory_count':len(scope['timer_rows']),'unresolved_scenarios':[],'unknown_exact_endpoint_count':unknown,'unknown_endpoints_are_disclosed_not_solved':True,'all360_possible_is_not_all360_proven_interrupted':True,'coverage_closed_only_for_registered_run_through_cost_capture':True,'exact_timer_subtraction_performed':False,'all_plan_exact_active_seconds':None,'old_active_named_subtotals_not_valid_as_full_plan_active_seconds':True,'checks':checks,'evidence':[ds[k] for k in FIXED]+[cost['registration']]+[r['closure'] for r in scope['pause_and_loss_scope']['three_losses']],'read_scope':{'source_JSON_XML_only':True,'unique_text_evidence_files':len(inputs),'new_models':0,'new_Tensor_reads':0,'GPU_queries_or_calls':0,'native_process_queries_or_control':0,'probability_reads':0,'CPU_checkpoint_audit_calls':0,'prior_cost_or_wide_audits_reexecuted':0},'limitations':['Exact historical process endpoints remain unknown for513 rows; conservative review envelopes are not actual lifetimes.','System session data does not establish each process stopped; Boot220/scenario11 explicitly reported no sleep.','Pre-registration costs and events after final cost capture are outside this completed review domain.'],'created_at':datetime.now(timezone.utc).isoformat()}
    target=A/('independent_final_system_low_power_scope_review_v2_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'report':descriptor(target),'passed':True,'inventory':len(scope['timer_rows']),'relations':relations,'unknown':unknown}));return value
if __name__=='__main__':run()
