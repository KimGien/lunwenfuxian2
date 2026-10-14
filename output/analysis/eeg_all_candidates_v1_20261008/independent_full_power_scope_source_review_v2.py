"""Independent stdlib source and conservative-envelope review; no final certificate."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import time

SOURCE=Path(__file__).resolve(); A=SOURCE.parent; WORK=SOURCE.parents[3]
REL='output/analysis/eeg_all_candidates_v1_20261008/'
EXPECTED={
 'full_system_power_timer_scope_v1.py':'c28bdd912c3a97510cd17d725ce7542c078640ec3e77f2c48340d6c0e923caec',
 'audit_system_standby_final_delivery_v5.py':'5bda427e754deaa5f0f1aa246a29dc0ff5274de36bc579598e1d93a2325bee29'}
def desc(p):
    p=Path(p).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def require(ok,why):
    if not ok:raise ValueError(why)

def run():
    checks=[]
    def check(name,ok):
        require(ok,name);checks.append({'name':name,'passed':True})
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError):check(name,True)
        else:raise ValueError('Accepted negative: '+name)
    sources=[desc(A/name) for name in EXPECTED]
    check('Exact two candidate source bytes',all(d['sha256']==EXPECTED[Path(d['path']).name] for d in sources))
    ns={'__file__':str(A/'full_system_power_timer_scope_v1.py'),'__name__':'independent_source_review_no_main'}
    exec(compile((A/'full_system_power_timer_scope_v1.py').read_text(encoding='utf-8'),ns['__file__'],'exec'),ns)
    reader=ns['Reader']()
    prefix=reader.checked(ns['PREFIX']); reg,regd=reader.snapshot(ns['OUT']+'registration.json')
    actual=ns['event_inventory'](reader,[ns['PREFIX']],reg['registered_at'],prefix['query_utc_end'])
    check('Actual raw prefix independently reparsed: 208 provider events, 31 modern pairs, 7 declared sleep windows',len(actual['events'])==208 and actual['modern_session_pairs']==31 and actual['declared_sleep_wake_events']==7)
    window=next(v for v in actual['intervals'] if v['id']=='modern_220_11')
    check('Long Boot220 scenario11 is screen-off with SleepEntered false, never a process stop interval',window['kind']=='screen_off_session_no_sleep_reported' and window['event_data']['SleepEntered']=='false' and window['event_data']['SleepDurationInUs']=='0' and window['per_process_stopped_not_proven'])
    check('Original unmatched 218/256 is actually paired in complete prefix',actual['original_boot218_scenario256_resolved'] is True)
    check('42/107 creation times retained only as supporting sequence, not sleep duration',actual['42_107_creation_difference_is_not_sleep_duration'] is True and all('support_records' in v for v in actual['intervals'] if v['kind']=='event_declared_sleep_wake'))
    original=copy.deepcopy(prefix)
    class EventReader:
        def __init__(self,value):self.value=value
        def checked(self,d):return self.value if d==ns['PREFIX'] else reader.checked(d)
    bad=copy.deepcopy(original);bad['all_power_provider_events_in_same_window']=[r for r in bad['all_power_provider_events_in_same_window'] if r['record_id']!=150976]
    rejects('Missing modern-session exit rejected',lambda:ns['event_inventory'](EventReader(bad),[ns['PREFIX']],reg['registered_at'],prefix['query_utc_end']))
    bad=copy.deepcopy(original);bad['all_power_provider_events_in_same_window'][0]['event_id']=-1
    rejects('XML versus summary contradiction rejected',lambda:ns['event_inventory'](EventReader(bad),[ns['PREFIX']],reg['registered_at'],prefix['query_utc_end']))
    rejects('Prefix alone cannot certify later final cost capture',lambda:ns['event_inventory'](reader,[ns['PREFIX']],reg['registered_at'],'2026-10-14T00:00:00+00:00'))
    bad=copy.deepcopy(original);bad['oldest_available_system_record_utc']='2026-10-09T00:00:00+00:00'
    rejects('Log-retention gap rejected',lambda:ns['event_inventory'](EventReader(bad),[ns['PREFIX']],reg['registered_at'],prefix['query_utc_end']))
    directions=ns['source_directions'](reader,reg)
    check('Five original directional sources retain exact Q14 source bindings',len(directions['registered_execution_sources'])==5)
    texts={Path(d['path']).name:reader.checked(d) for d in directions['registered_execution_sources']}
    def funcs(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    rfunc=funcs(texts['registry.py'])
    completion=ast.unparse(rfunc['record_fit_status'])
    check('Registry validates completion before recording status time',completion.index('_validate_complete(evidence, folder)')<completion.index("'recorded_at': now()"))
    check('Condition lock binds existing original registration before work','registration_descriptor()' in ast.unparse(rfunc['_condition_lock']) and 'descriptor(REGISTRATION)' in ast.unparse(rfunc['registration_descriptor']))
    for name,fn in [('finetune.py','fit'),('supervised.py','fit_one'),('ssl.py','fit_one'),('frozen_heads.py','fit')]:
        body=ast.unparse(funcs(texts[name])[fn])
        check(name+' records complete after metadata production',body.rfind('record_fit_status')>body.rfind('metadata.json') and ('_condition_lock' in body or "record_fit_status(condition, seed, fold, 'training')" in body))
    pause,lossmap=ns['loss_and_pause'](reader,{})
    check('Three exact losses and noon proofs retained with unknown exit and no re-added counters',len(pause['three_losses'])==3 and all(v['actual_exit_time'] is None and v['active_seconds_exact'] is None and v['extra_work_added_again'] is False for v in pause['three_losses']) and pause['no_nested_wait_pause_or_loss_cost_added'] is True)
    old=funcs((A/'audit_system_standby_final_delivery_v4.py').read_text(encoding='utf-8'));new=funcs((A/'audit_system_standby_final_delivery_v5.py').read_text(encoding='utf-8'))
    changes={k for k in old if ast.dump(old[k],include_attributes=False)!=ast.dump(new[k],include_attributes=False)}
    check('Original 30, same-cost bindings, bounded standby, native and memory verifiers retain identical AST',changes=={'verify_scope_closure','audit','main'})
    gate={'__file__':str(A/'audit_system_standby_final_delivery_v5.py'),'__name__':'independent_gate_source_no_main'}
    exec(compile((A/'audit_system_standby_final_delivery_v5.py').read_text(encoding='utf-8'),gate['__file__'],'exec'),gate)
    check('Gate pins exact reviewed helper',gate['FULL_SCOPE_SOURCE']==sources[0])
    check('Gate Reader supplies compatible root and checked/snapshot semantics',gate['Reader']().root==reader.root and callable(gate['Reader']().checked))
    scopebody=ast.unparse(new['verify_scope_closure'])
    check('Final gate recomputes all scope fields and binds exact actual cost/wide plus complete artifacts',"ns['analyze']" in scopebody and 'scope.get(key) == value' in scopebody and "review.get('binding') == expected_binding" in scopebody)

    # Independent fake domain, exact dates intentionally close to show exclusion errors.
    # No producer modules, models, real cost collection or live queries are imported.
    fake=ns.copy(); start='2026-10-08T00:00:00+00:00'; finish='2026-10-13T00:00:00+00:00'; complete='2026-10-09T00:00:00+00:00'
    def d(path):return {'path':path,'sha256':'fake-only'}
    registry=d('fake_registration.json'); costd=d('fake_cost.json'); wided=d('fake_wide.json')
    fitids=['fake/seed_11/fold_'+str(i) for i in range(360)]
    data={registry['path']:{'registered_at':start}}
    fits=[]
    for fid in fitids:
        sd=d(fid+'/status.json');md=d(fid+'/metadata.json');fits.append({'fit_id':fid,'status_artifact':sd,'metadata_artifact':md})
        data[sd['path']]={'fit_id':fid,'status':'training_complete','registration':registry,'recorded_at':complete,'evidence':{'metadata':md}}
    rows=[]
    def add(label,category='engineering',a=None,b=None,fid=None):
        artifact=d('fake_'+label+'.json');item={'artifact':artifact,'category':category,'aggregation_role':'fake','fit_id':fid,'started_at':a,'finished_at':b,'seconds_observed':None,'status':'running' if b is None else 'complete'}
        data[artifact['path']]={'started_at':a,'finished_at':b};rows.append(item);return len(rows)-1
    full=add('unknown'); tail=add('tail',a='2026-10-10T00:00:00+00:00'); head=add('head',b='2026-10-11T00:00:00+00:00')
    metadata=add('metadata','formal_legacy_metadata',fid=fitids[0]); wrapper=add('wrapper','formal_training_attempt',fid=fitids[0]); failure=add('failure','legacy_failure',fid=fitids[0])
    lost=add('loss','formal_training_attempt',a=start,fid=fitids[1]); exact=add('exact',a='2026-10-10T00:00:00+00:00',b='2026-10-12T00:00:00+00:00')
    pre=add('prefinish',b='2026-10-07T00:00:00+00:00'); duplicate=len(rows);rows.append(copy.deepcopy(rows[full]))
    events={'snapshots':[],'intervals':[{'id':'later','start':'2026-10-10T12:00:00+00:00','end':'2026-10-10T13:00:00+00:00','review_range_relation':'intersects_registered_run'}]}
    class FakeReader:
        def checked(self,x):return copy.deepcopy(data[x['path']])
    fake['source_directions']=lambda *args:{};fake['event_inventory']=lambda *args:events
    fake['loss_and_pause']=lambda *args:({'fake':True},{rows[lost]['artifact']['path']:{'closure':d('fake_loss.json'),'observed_absent_no_later_than':'2026-10-11T00:00:00+00:00'}})
    # Bind a fresh function to patched globals; original function code is unchanged.
    import types
    analyze=types.FunctionType(ns['analyze'].__code__,fake)
    cost={'status':'verified_final_cost','schema_version':19,'source':ns['COST_SOURCE'],'registration':registry,'captured_at':finish,'formal_fits':fits,'cost_records':rows}
    wide={'source':ns['WIDE_SOURCE'],'full_plan_completed':True,'objective_satisfied':True,'final_cost':{'snapshot':costd},'registration':registry,'training_rows':[]}
    result=analyze(FakeReader(),cost,costd,wide,wided,[]); out=result['timer_rows']
    check('Every inventory occurrence retained, including repeated-artifact rows',len(out)==len(rows) and [r['inventory_index'] for r in out]==list(range(len(rows))) and out[duplicate]['artifact']==out[full]['artifact'])
    check('Unknown both endpoints conservatively cover whole review domain',out[full]['review_envelope']['start']==start and out[full]['review_envelope']['end']==finish and out[full]['exact_started_at'] is None and out[full]['exact_finished_at'] is None)
    check('Known start with missing finish retains possible later overlap',out[tail]['review_envelope']['end']==finish and out[tail]['event_relations'][0]['relation']=='possible_overlap_unknown_exact_endpoint')
    check('Known finish with missing start uses registration lower bound',out[head]['review_envelope']['start']==start and out[head]['exact_started_at'] is None)
    check('Only formal metadata receives completion-time upper bound',out[metadata]['review_envelope']['end']==complete)
    check('Formal wrapper unknown tail cannot be excluded by an earlier completion status',out[wrapper]['review_envelope']['end']==finish and out[wrapper]['event_relations'][0]['relation']=='possible_overlap_unknown_exact_endpoint')
    check('Legacy failure unknown tail cannot be excluded by a completion status',out[failure]['review_envelope']['end']==finish)
    check('Loss observation is upper bound, never an invented actual finish',out[lost]['review_envelope']['end']=='2026-10-11T00:00:00+00:00' and out[lost]['exact_finished_at'] is None and 'process_loss_observation_is_upper_bound_not_actual_exit' in out[lost]['review_envelope']['basis'])
    check('Recorded exact wall interval uses explicit endpoints without elapsed deduction',out[exact]['both_recorded_endpoints'] is True and out[exact]['event_relations'][0]['relation']=='recorded_timer_interval_overlap')
    check('Pre-registration known finish gives empty review intersection, not zero runtime',out[pre]['review_envelope']['start']==out[pre]['review_envelope']['end']==start and out[pre]['exact_started_at'] is None and out[pre]['recorded_seconds_observed'] is None and out[pre]['event_relations'][0]['relation']=='outside_timer_review_envelope')
    check('No per-process active time, nontraining time or subtraction invented',result['all_plan_exact_active_seconds'] is None and result['exact_timer_subtraction_performed'] is False and result['old_active_named_subtotals_not_valid_as_full_plan_active_seconds'] is True and all(r['active_seconds_exact'] is None and r['nontraining_seconds_exact'] is None and r['costs_or_counters_added_again'] is False for r in out))
    badcost=copy.deepcopy(cost);badcost['cost_records'][exact]['finished_at']='2026-10-09T00:00:00+00:00'
    rejects('Reversed explicit timer endpoints rejected',lambda:analyze(FakeReader(),badcost,costd,wide,wided,[]))
    badwide=copy.deepcopy(wide);badwide['final_cost']['snapshot']=d('wrong_cost.json')
    rejects('Different final cost rejected',lambda:analyze(FakeReader(),cost,costd,badwide,wided,[]))
    reader.recheck()
    check('Candidate source bytes remain exact after read-only tests',sources==[desc(A/name) for name in EXPECTED])
    value={'schema':'independent_full_power_scope_source_review_v1','status':'full_power_scope_sources_reviewed_only','passed':True,'reviewer':'loss3_independent_review independent agent','reviewer_source':desc(SOURCE),'reviewed_sources':sources,'checks':checks,'check_count':len(checks),'actual_event_prefix':ns['PREFIX'],'training_registration':ns['Q14'],'actual_read_evidence':list(reader.inputs.values()),'source_directions':directions,'three_loss_evidence':pause,'scope_review_final_certificate_issued':False,'final_cost_wide_suffix_not_reviewed_by_this_report':True,'limits':['This is source/fake-envelope and existing-prefix review only; real final cost, wide, suffix and every actual row still require independent review.','Conservative envelopes can overinclude possible overlaps; they never recover true active seconds.','Modern Standby and event-declared sleep/wake are system observations, not proof of per-process stopped time.','Timestamp comparisons use Python microsecond precision while original seven-digit event timestamps are retained; no exact durations are derived.'],'new_model_calls':0,'new_Tensor_reads':0,'GPU_queries_or_calls':0,'native_process_queries_or_control':0,'test_probability_reads':0,'actual_cost_collection_calls':0}
    target=A/('independent_full_power_scope_source_review_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'report':desc(target),'passed':True,'checks':len(checks)}));return value

if __name__=='__main__':run()
