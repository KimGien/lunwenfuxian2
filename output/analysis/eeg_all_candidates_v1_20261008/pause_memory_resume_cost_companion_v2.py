"""Final JSON-only compatibility: original noon pause and three real process losses.

This never reruns a collector, scientific verifier, native query, or old pause.
It retains the verified noon report and binds the immutable loss closures to the
same actual cost17/wide7. A lost timer is lifecycle-closed, never producer-terminal.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

SOURCE = Path(__file__).resolve()
REPORT = SOURCE.parent
WORK = SOURCE.parents[3]
REL = 'output/analysis/eeg_all_candidates_v1_20261008/'
OUT = 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/'
def binding(path, sha): return {'path': path, 'sha256': sha}
NOON = binding(REL+'pause_memory_resume_cost_companion_1791625875034611700.json', '60217de0cb993be4b43bfe28cb4a0f79f9f5cfe314bf62249fd002b764a7b7db')
NOON_SOURCE = binding(REL+'pause_memory_resume_cost_companion_v1.py', 'c1709cfc9c2f96d810215bbd3d7960cd5feb7d2382a2a5a36c4570bb74d92ca4')
FINALIZER = binding(REL+'finalize_full_plan_cost_v7.py', 'ab5d4d892f3d6a0e75c2c82db5bd3bbcfdffd086b4be4f92858234612982a712')
COST = {'path': 'output/analysis/eeg_all_candidates_v1_20261008/cost_aggregation_v17.py', 'sha256': 'aa75239a0a9114006e128d4b7bb0713453aeadc211b6aa8fef7c5f88e1e169a1'}
WIDE = binding(REL+'audit_full_plan_acceptance_v7.py', 'ad8ddc3b50fd8c4537d5aaffe02451196762f27a6d3ffaa33fdf282fee65e795')
Q14 = binding(OUT+'continuation_queue_v14/registration.json', '93b19228fb8d3f032f87be4072f4ff2fb3a0bbe2d89adce93117896b5bc59742')
LOSSES = [
    binding(OUT+'process_loss_closures/lost_f3_1791597116209422300.json', 'e56bbb201c067bf0bab31c0d405520fb2860deb12796f088072bb30bdd73de65'),
    binding(OUT+'process_loss_closures/lost_f3_v2_1791706969935734100.json', '14286a281be2d896653fb717ea2faefb86b84b3d80e572e36795154b746bfc8e'),
    binding(OUT+'process_loss_closures/lost_f3_v3_1791767849813215500.json', '83d4558818cd8727a4ec4b04d18ebfc18653aed6f3a03a611edb5a41f659e791'),
]
ERRATA = [
    binding(REL+'historical_binary_hash_scope_erratum_1791601956592773300.json', 'c7542365ed904799da58242605a3d46dabfd164ec00cdb18a409887405279038'),
    binding(REL+'independent_closure_review_binary_hash_scope_erratum_1791602447024092800.json', '462f2f3beea08b79640c4d8f1a0889cb02bc29fe9579dcb667232d3bb79b0962'),
    binding(REL+'sync_control_companion_read_scope_erratum_1791602404565633400.json', 'bc114598e11c57bb2cc5f09115c764054b6709932b9b3fa75b1f125edd8a2461'),
]

def require(ok, why):
    if not ok: raise ValueError(why)

class Reader:
    def __init__(self, root=WORK): self.root, self.inputs = Path(root).resolve(), {}
    def snapshot(self, path):
        p = (self.root/path).resolve()
        require(p.is_relative_to(self.root) and p.suffix.lower() in ('.json','.py','.ps1','.md'), 'Only workspace text evidence allowed')
        require(not any(x in p.name.lower() for x in ('prediction','probabilit','oof_')), 'Outer prediction content is outside scope')
        raw = p.read_bytes(); d = binding(p.relative_to(self.root).as_posix(), hashlib.sha256(raw).hexdigest())
        require(d['path'] not in self.inputs or self.inputs[d['path']] == d, 'Input changed during read')
        self.inputs[d['path']] = d
        return (json.loads(raw.decode('utf-8-sig')) if p.suffix.lower()=='.json' else raw.decode('utf-8-sig')), d
    def checked(self, d):
        require(type(d) is dict and set(d)=={'path','sha256'}, 'Exact descriptor required')
        value, actual = self.snapshot(d['path']); require(actual==d, 'Bound source/evidence changed'); return value
    def recheck(self):
        for d in list(self.inputs.values()): self.checked(d)

def verify_final_pair(read, cost, wide, cost_binding, wide_binding):
    read(FINALIZER); read(WIDE); read(Q14); read(COST)
    require(cost.get('source')==COST and cost.get('actual_cost_collector_source')==COST and wide.get('actual_cost_collector_source')==COST, 'Actual Cost17 collector binding required')
    require(cost.get('schema_version')==17 and type(cost['schema_version']) is int and cost.get('finalizer_source')==FINALIZER
        and cost.get('status')=='verified_final_cost' and cost.get('final_snapshot') is True, 'Actual final cost17 required')
    require(wide.get('source')==WIDE and wide.get('schema_version')==7 and type(wide['schema_version']) is int
        and wide.get('full_plan_completed') is True and wide.get('objective_satisfied') is True
        and wide.get('final_cost',{}).get('passed') is True and wide['final_cost'].get('snapshot')==cost_binding, 'Same complete wide7 required')
    rows=wide.get('requirements',[])
    require(len(rows)==30 and {r['id'] for r in rows}=={'R%02d'%i for i in range(1,31)}
        and all(r.get('status')=='verified_complete' for r in rows), 'All thirty original requirements required')
    lc=cost.get('actual_training_lifecycle',{})
    require(lc==wide.get('actual_training_lifecycle') and lc.get('passed') is True and lc.get('training_registration')==Q14
        and lc.get('formal_fits_verified')==360 and lc.get('FT_verified')==135 and lc.get('CPU_loads_verified')==405
        and len(lc.get('training_sources',[]))==194, 'Actual mixed Q13/Q14/E11/CPU16 lifecycle differs')
    for k in ('training_registration','training_terminal','CPU_gate','CPU_audit','CPU_auditor','evaluation_control_completion','evaluation_runtime','evaluation_released_owner','training_source_review'):
        read(lc[k])
    for d in lc['training_sources']: read(d)
    for k in ('registration','terminal','CPU_audit','child'): read(lc['CodeBrain_completion'][k])
    cpu=cost.get('source_bound_cpu_checkpoint_verification',{})
    require(cpu.get('passed') is True and all(type(cpu.get(k)) is int and cpu[k]==v for k,v in
        (('verified_fits',135),('prior_actual_cpu_checkpoint_loads',405),('new_cpu_checkpoint_loads',0))), 'Actual prior135/405 CPU evidence required')
    return {'final_cost':cost_binding, 'full_plan_acceptance':wide_binding, 'training_lifecycle':lc}

def verify_loss_chain(read, cost):
    """Bind already-verified cost records to original loss JSON, without binaries."""
    proof=cost.get('completion_gate',{}).get('recovery_execution_proof',{})
    require([proof.get(k) for k in ('original_recovery_closure','previous_recovery_closure','recovery_closure')]==LOSSES,
        'Exactly three actual historical closures required')
    rows=cost.get('source_bound_lost_f3_attempts',[])
    primary=[r for r in rows if r.get('aggregation_role')=='primary_observed_invocation']
    require(len(primary)==3 and {r['closure']['path']:r['closure'] for r in primary}=={d['path']:d for d in LOSSES}, 'Missing/duplicate lost primary')
    require(all(r.get('aggregation_role') in ('primary_observed_invocation','duplicate_invocation_artifact') and r.get('closure') in LOSSES for r in rows), 'Unknown loss alias')
    result=[]; parents=[]
    for index,d in enumerate(LOSSES):
        closure=read(d); row=next(r for r in primary if r['closure']==d)
        manifest=read(closure['recovery_registration']); read(closure['closure_source'])
        require(closure.get('schema')=='lost_f3_process_closure_v'+str(index+1) and closure.get('passed') is True
            and closure.get('status')=='process_absence_observed' and closure.get('old_attempt_bytes_unchanged') is True
            and closure.get('known_extra_not_added_again_to_original_returned_counters') is True, 'Actual immutable loss closure differs')
        require(row.get('passed') is True and row.get('recovery_registration')==closure['recovery_registration']
            and row.get('closure_source')==closure['closure_source']==manifest['source'] and row['artifact']==closure['original_attempt'], 'Loss source/attempt binding differs')
        require(all(closure.get(k,'missing') is None for k in ('unflushed_tail_counters','final_elapsed_seconds','actual_exit_time'))
            and closure.get('graceful_terminal_flush') is False and row.get('final_elapsed_seconds','missing') is None
            and row.get('unflushed_tail_unknown') is True and row.get('extra_work_not_added_again') is True, 'Unknown loss tail/time was made exact')
        original=read(closure['original_attempt']); snapshot=read(closure['attempt_snapshot'])
        require(original==snapshot and original.get('status')=='running' and original.get('seconds') is None, 'Lost producer must retain original running/unknown bytes')
        records=[r for r in cost['cost_records'] if r['artifact']==closure['original_attempt'] and r.get('aggregation_role')=='primary_observed_invocation']
        require(len(records)==1 and records[0]['category']=='formal_training_attempt' and records[0]['status']=='running'
            and records[0]['seconds_observed'] is None, 'Lost formal primary absent/duplicated/relabelled')
        for f,ck in (('durable_epoch','epoch'),('durable_optimizer_updates','optimizer_updates'),('durable_backward_calls','backward_calls')):
            require(type(row.get(f)) is int and row[f]==closure['durable'][ck], 'Durable loss boundary differs')
        for f,ck in (('known_extra_optimizer_returns_outside_checkpoint','optimizer_updates'),('known_extra_backward_returns_outside_checkpoint','backward_calls')):
            require(type(row.get(f)) is int and row[f]==closure['known_extra_lower_bounds'][ck], 'Extra returned work differs')
        for k in ('checkpoint_cpu_resume_proof','process_absence_evidence','old_owner_snapshot','old_queue_attempt_snapshot'):
            read(closure[k])
        cpu=read(closure['checkpoint_cpu_resume_proof']); require(cpu.get('passed') is True and row['checkpoint_cpu_resume_proof']==closure['checkpoint_cpu_resume_proof'], 'Real prior CPU resume proof required')
        for k in ('pause_proof','pause_native_proof','pause_capture','pause_progress_and_observer','runtime_process_absence_evidence','loss_evidence'):
            if k in closure: read(closure[k])
        if index:
            require(closure['previous_recovery_registration']==proof['original_recovery_registration' if index==1 else 'previous_recovery_registration'], 'Lost lineage changed')
            parent_d=closure['old_queue_attempt_original']; parent=read(parent_d)
            require(parent==read(closure['old_queue_attempt_snapshot']) and parent.get('status')=='running'
                and parent.get('seconds') is None and parent.get('returncode') is None, 'Lost parent must remain running and unknown')
            ps=[r for r in cost.get('source_bound_lost_parent_observer_scopes',[]) if r.get('closure')==d]
            cr=[r for r in cost['cost_records'] if r['artifact']==parent_d]
            require(len(ps)==len(cr)==1 and ps[0]['artifact']==parent_d and ps[0].get('final_elapsed_seconds','missing') is None
                and ps[0].get('added_again') is False and cr[0]['category']=='dispatcher_elapsed'
                and cr[0]['aggregation_role']=='supplemental_nested_or_unmeasured' and cr[0]['seconds_observed'] is None, 'Lost parent missing/duplicated or added')
            parents.append({'artifact':parent_d,'closure':d,'status':'lost_event_closed_unknown_elapsed','producer_status':'running',
                'producer_terminal':False,'seconds_observed':None,'active_seconds_exact':None,'added_again':False})
        result.append({'closure':d,'artifact':closure['original_attempt'],'recovery_registration':closure['recovery_registration'],
            'durable':closure['durable'],'known_extra_lower_bounds_already_in_original_trace':closure['known_extra_lower_bounds'],
            'lifecycle_closed_by_actual_loss':True,'producer_status':'running','producer_terminal':False,'seconds_observed':None,
            'active_seconds_exact':None,'unflushed_tail_counters':None,'counts_added_again':False})
    require(len(cost.get('source_bound_lost_parent_observer_scopes',[]))==2, 'Unexpected lost parent scope')
    for d in ERRATA: read(d)
    return {'passed':True,'three_loss_events':result,'lost_parent_scopes':parents,'all_loss_lifecycles_closed':True,
        'unknown_elapsed_preserved':True,'new_loss_closures_created':False,'new_tensor_loads':0,'extra_work_added_again':False,'io_scope_errata':ERRATA}

def collect(final_cost, final_acceptance):
    reader=Reader(); cost,cd=reader.snapshot(final_cost); wide,wd=reader.snapshot(final_acceptance)
    pair=verify_final_pair(reader.checked,cost,wide,cd,wd); noon=reader.checked(NOON); reader.checked(NOON_SOURCE)
    require(noon.get('source')==NOON_SOURCE and noon.get('schema')=='pause_memory_resume_cost_companion_v1'
        and noon.get('status')=='verified_pause_scope_and_recorded_terminal_timers' and noon.get('passed') is True
        and noon.get('all_affected_producer_timers_terminal') is True, 'Original actual terminal noon companion required')
    for d in noon['immutable_inputs']: reader.checked(d)
    reader.checked(noon['human_readable_report']); losses=verify_loss_chain(reader.checked,cost)
    value={**noon,'schema':'pause_memory_resume_cost_companion_v2','source':reader.snapshot(SOURCE)[1],
        'original_noon_companion':NOON,'original_noon_human_report':noon['human_readable_report'],
        'final_artifacts':pair,'loss_chain':losses,'all_affected_timer_lifecycles_closed':True,
        'old_noon_companion_recollected':False,'all_plan_active_seconds_exact':None}
    reader.recheck(); value['immutable_inputs']=list(reader.inputs.values()); return value

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true')
    p.add_argument('--final-cost',type=Path); p.add_argument('--final-acceptance',type=Path); args=p.parse_args(argv)
    if not args.run: print(json.dumps({'status':'prepared_only','operations':0})); return 0
    require(args.final_cost and args.final_acceptance, 'Both actual final artifacts required')
    value=collect(args.final_cost,args.final_acceptance); stamp=str(time.time_ns())
    human=REPORT/('pause_memory_and_three_loss_cost_v2_'+stamp+'.md')
    content='# 原进程暂停及三次丢失成本附页\n\n沿用真实中午暂停终态附页；三次丢失以原始封存证明关闭生命周期。丢失生产者仍为 running，尾部、退出时点、总秒数和有效秒数仍未知。暂停与守护等待不计训练，额外返回操作已包含原记录，不再加一次；不用墙钟与 QPC 混减。\n\n原中午附页：'+value['original_noon_human_report']['path']+'\n'
    with human.open('x',encoding='utf8') as f: f.write(content)
    value['human_readable_report']=Reader().snapshot(human)[1]; value['created_at']=datetime.now(timezone.utc).isoformat()
    target=REPORT/('pause_memory_resume_cost_companion_v2_'+stamp+'.json')
    with target.open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(Reader().snapshot(target)[1])); return 0

if __name__=='__main__': raise SystemExit(main())
