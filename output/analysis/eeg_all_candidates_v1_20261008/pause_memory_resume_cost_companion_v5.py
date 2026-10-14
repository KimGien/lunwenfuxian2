"""Final JSON-only compatibility: original noon pause and three real process losses.

This never reruns a collector, scientific verifier, native query, or old pause.
It retains the verified noon report and binds the immutable loss closures to the
same actual cost19/wide9. A lost timer is lifecycle-closed, never producer-terminal.
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
FINALIZER = binding(REL+'finalize_full_plan_cost_v9.py', '4752ada431a8008c7b97a8908210bd45dbd55e2f6e68a037c6f4f4b9757b70ac')
COST = {'path': 'output/analysis/eeg_all_candidates_v1_20261008/cost_aggregation_v19.py', 'sha256': 'e03e1832cbff9472d17377a8b38e63526f340294912aece43472c450a97cfc6d'}
WIDE = binding(REL+'audit_full_plan_acceptance_v9.py', 'df743c655c85422091b3c371f2a73b587ca48963c642742926a4901e49c15ce7')
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

MAMBA_MANIFEST = {'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation/EEGMamba-lastblock-finetune/predictions_manifest.json', 'sha256': '9d3a15c32609ae74f7db91f1ad9105f616a0135dfa51de5a0a484645a71d5661'}

def fixed_manifest_path(path):
    return Path(path).resolve() == (WORK / MAMBA_MANIFEST['path']).resolve()

def validate_fixed_manifest_metadata(path, raw):
    if not fixed_manifest_path(path): return None
    require(hashlib.sha256(raw).hexdigest()==MAMBA_MANIFEST['sha256'], 'Fixed E12 manifest bytes changed')
    value=json.loads(raw.decode('utf-8-sig'))
    require(type(value) is dict and set(value)=={'passed','condition','test_gate','registration','supervised_fits_evaluated','primary_oof_predictions','distinct_subjects','evaluation_fit_artifacts','evaluator_source','global_report_generated','evaluator_execution'}, 'Only fixed descriptor/count manifest metadata permitted')
    require(value['passed'] is True and value['condition']=='EEGMamba-lastblock-finetune' and value['global_report_generated'] is False, 'E12 metadata condition/status differs')
    require(all(type(value[k]) is int and value[k]==n for k,n in (('supervised_fits_evaluated',15),('primary_oof_predictions',111),('distinct_subjects',37))), 'Metadata counts differ')
    expected={OUT+'evaluation/EEGMamba-lastblock-finetune/seed_'+str(seed)+'/fold_'+str(fold)+'/best/complete.json' for seed in (11,22,33) for fold in range(1,6)}
    rows=value['evaluation_fit_artifacts']
    require(type(rows) is list and len(rows)==15 and all(type(d) is dict and set(d)=={'path','sha256'} and type(d['sha256']) is str and len(d['sha256'])==64 for d in rows) and {d['path'] for d in rows}==expected, 'Fixed fifteen completion descriptors required')
    require(value['evaluator_source']=={'path':'tridep_ccml/eeg_all_candidates/evaluate_v12.py','sha256':'9a84b1cb5994571f1c486abf59a58a5100f111dbda2bd9ed93dec643bb532473'} and value['evaluator_execution']['version']==12 and value['evaluator_execution']['wrapper_source']==value['evaluator_source'], 'Actual E12 metadata provenance differs')
    return value

class Reader:
    def __init__(self, root=WORK): self.root, self.inputs = Path(root).resolve(), {}
    def snapshot(self, path):
        p = (self.root/path).resolve()
        require(p.is_relative_to(self.root) and p.suffix.lower() in ('.json','.py','.ps1','.md'), 'Only workspace text evidence allowed')
        require(fixed_manifest_path(p) or not any(x in p.name.lower() for x in ('prediction','probabilit','oof_')), 'Outer prediction content is outside scope')
        raw = p.read_bytes(); validate_fixed_manifest_metadata(p, raw); d = binding(p.relative_to(self.root).as_posix(), hashlib.sha256(raw).hexdigest())
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
    require(cost.get('source')==COST and cost.get('actual_cost_collector_source')==COST and wide.get('actual_cost_collector_source')==COST, 'Actual Cost19 collector binding required')
    require(cost.get('schema_version')==19 and type(cost['schema_version']) is int and cost.get('finalizer_source')==FINALIZER
        and cost.get('status')=='verified_final_cost' and cost.get('final_snapshot') is True, 'Actual final cost19 required')
    require(wide.get('source')==WIDE and wide.get('schema_version')==9 and type(wide['schema_version']) is int
        and wide.get('full_plan_completed') is True and wide.get('objective_satisfied') is True
        and wide.get('final_cost',{}).get('passed') is True and wide['final_cost'].get('snapshot')==cost_binding, 'Same complete wide9 required')
    rows=wide.get('requirements',[])
    require(len(rows)==30 and {r['id'] for r in rows}=={'R%02d'%i for i in range(1,31)}
        and all(r.get('status')=='verified_complete' for r in rows), 'All thirty original requirements required')
    lc=cost.get('actual_training_lifecycle',{})
    require(lc==wide.get('actual_training_lifecycle') and lc.get('passed') is True and lc.get('training_registration')==Q14
        and lc.get('formal_fits_verified')==360 and lc.get('FT_verified')==135 and lc.get('CPU_loads_verified')==405
        and len(lc.get('training_sources',[]))==194, 'Actual mixed Q13/Q14/E13/CPU17 recovered lifecycle differs')
    for k in ('training_registration','training_terminal','CPU_gate','CPU_audit','CPU_auditor','evaluation_control_completion','evaluation_runtime','evaluation_released_owner','training_source_review'):
        read(lc[k])
    for d in lc['training_sources']: read(d)
    for k in ('registration','terminal','CPU_audit','child'): read(lc['CodeBrain_completion'][k])
    cpu=cost.get('source_bound_cpu_checkpoint_verification',{})
    require(cpu.get('passed') is True and all(type(cpu.get(k)) is int and cpu[k]==v for k,v in
        (('verified_fits',135),('prior_actual_cpu_checkpoint_loads',405),('new_cpu_checkpoint_loads',0))), 'Actual prior135/405 CPU evidence required')
    gate=read(lc['post_training_recovery_gate'])
    require(lc['CPU_gate']==lc['post_training_recovery_gate'] and gate.get('schema')=='post_training_cpu_recovery_v1' and gate.get('passed') is True, 'Independent post-training recovery gate required')
    require(gate.get('CPU_audit')==lc['CPU_audit'] and gate.get('failed_CPU16')==lc['failed_CPU16'] and gate.get('original_Q14_failed_terminal')==lc['training_terminal'] and gate.get('original_Q14_released_owner')==lc['original_Q14_released_owner'] and gate.get('original_child_complete')==lc['original_child_complete'], 'Exact failed/released/exit0 lineage required')
    require(gate.get('original_training_complete')==360 and gate.get('FT')==135 and gate.get('CPU_loads')==405 and gate.get('actual_new_training_or_optimizer_updates')==0 and gate.get('original_Q14_status_rewritten') is False, 'No training or history relabelling permitted')
    review=read(lc['post_training_source_review'])
    require(gate.get('post_training_source_review')==lc['post_training_source_review'] and gate.get('post_training_sources')==lc['post_training_sources'] and review.get('passed') is True and review.get('status')=='post_training_CPU_recovery_sources_reviewed' and review.get('reviewed_sources')==lc['post_training_sources'], 'Actual new post-training source review required')
    read(gate['source']);read(review['reviewer_source'])
    for item in lc['post_training_sources']:read(item)
    failed=read(lc['failed_CPU16']);passed=read(lc['CPU_audit']);terminal=read(lc['training_terminal']);child=read(lc['original_child_complete']);released=read(lc['original_Q14_released_owner'])
    require(terminal.get('status')=='failed' and child.get('status')=='complete' and type(child.get('returncode')) is int and child['returncode']==0 and released.get('child_pid') is None, 'Historical Q14 failure and successful training child must both remain')
    require(failed.get('schema_version')==16 and failed.get('checkpoint_loads')==360 and failed.get('verified_completed_finetune_fits')==120 and len(failed.get('errors',[]))==15, 'Historical failed CPU16 scope must remain exact')
    require(passed.get('schema_version')==17 and passed.get('checkpoint_loads')==405 and passed.get('verified_completed_finetune_fits')==135 and passed.get('errors')==[] and passed.get('pending_fits')==[] and passed.get('deferred_completed_fits')==[] and passed.get('learning_rate_tolerance_used') is False, 'Actual strict CPU17 full135/405 proof required')
    accounting=cost.get('post_training_CPU_recovery_accounting',{})
    require(accounting.get('gate')==lc['CPU_gate'] and accounting.get('failed_CPU16')==lc['failed_CPU16'] and accounting.get('successful_CPU17')==lc['CPU_audit'] and accounting.get('failed_CPU16_loads')==360 and accounting.get('successful_CPU17_loads')==405 and accounting.get('nested_times_added') is False, 'Actual post-training independent costs must be bound')
    require(accounting.get('platform_captures')==[lc['Linux_LR_reference'],lc['Windows_LR_reference']], 'Both exact platform capture scopes required')
    for item in [lc['failed_CPU16'],lc['CPU_audit'],lc['Linux_LR_reference'],lc['Windows_LR_reference']]:
        data=read(item);rows=[r for r in cost['cost_records'] if r['artifact']==item]
        require(len(rows)==1 and rows[0]['aggregation_role']=='primary_observed_invocation','Actual independent post-training scope must occur once')
        if item in accounting['platform_captures']:
            require(rows[0]['seconds_observed']==data['primary_stdlib_capture_seconds'] and rows[0]['seconds_source_field']=='primary_stdlib_capture_seconds','Actual capture elapsed changed')
        else:
            count=360 if item==lc['failed_CPU16'] else 405
            require(rows[0]['observed_counters'].get('cpu_checkpoint_loads_started')==count and rows[0]['observed_counters'].get('cpu_checkpoint_loads_returned')==count and rows[0]['seconds_observed'] is None,'Real CPU counts/unknown elapsed changed')
    mixed=lc['mixed_evaluation_lifecycle'];closure=read(mixed['recovery_closure']);review=read(mixed['recovery_source_review'])
    require(closure.get('schema')=='evaluation_storage_recovery_v1' and closure.get('passed') is True and closure.get('sources')==mixed['recovery_sources'] and closure.get('source_review')==mixed['recovery_source_review'] and closure.get('CPU_gate')==lc['CPU_gate'],'Mixed evaluation recovery and original full CPU gate required')
    require(review.get('passed') is True and review.get('status')=='evaluation_storage_recovery_sources_reviewed' and review.get('reviewed_sources')==mixed['recovery_sources'],'Mixed evaluation independent source review required')
    for item in mixed['recovery_sources']:read(item)
    read(review['reviewer_source'])
    parent=read(mixed['original_E12_parent']);released=read(mixed['original_E12_released']);mamba=read(mixed['Mamba_E12_worker']);failed=read(mixed['failed_E12_Windows_worker']);windows=read(mixed['Windows_E13_worker']);manifest=read(mixed['Mamba_E12_manifest']);evaluation=read(lc['evaluation_runtime'])
    require(parent['status']=='running' and parent['stages']=={'mamba_linux':{'returncode':0,'worker_report':mixed['Mamba_E12_worker']},'all_windows':{'returncode':1,'worker_report':mixed['failed_E12_Windows_worker']}} and released['child_pid'] is None,'Original E12 running parent with actual two child outcomes retained')
    require(mamba['status']=='complete' and failed['status']=='failed' and windows['status']=='complete' and mamba['mode']=='mamba_linux' and failed['mode']==windows['mode']=='all_windows' and mamba['source']==failed['source']==manifest['evaluator_source'] and windows['source']==evaluation['source'] and mamba['source']!=windows['source'],'True mixed E12 Mamba/E13 Windows sources required')
    require(manifest['supervised_fits_evaluated']==15 and set(evaluation['stages'])=={'all_windows'} and evaluation['stages']['all_windows']['returncode']==0,'No second Mamba stage or incomplete Windows stage permitted')
    accounting=cost.get('evaluation_storage_recovery_accounting',{})
    require(accounting.get('closure')==mixed['recovery_closure'] and accounting.get('failed_header_CPU_loads')==1 and accounting.get('failed_header_elapsed_seconds') is None and accounting.get('nested_times_added') is False and accounting.get('Mamba_predictions_recomputed') is False,'Known CPU1/unknown elapsed and no repeated Mamba inference required')
    expected_primary=[mixed['failed_E12_Windows_worker'],mixed['metrics_reuse_comparison'],{'path': 'output/analysis/eeg_all_candidates_v1_20261008/evaluation_storage_recovery_stdlib_contract_1791838170182583300.json', 'sha256': '6e03dff9eb0249ecbb4210c0433ca530c914b9462f056018a02aa81b3b809f65'}]
    require(len(accounting.get('primary_artifacts',[]))==3 and len({x['path'] for x in accounting['primary_artifacts']})==3 and {tuple(sorted(x.items())) for x in accounting['primary_artifacts']}=={tuple(sorted(x.items())) for x in expected_primary},'Exact three independent evaluation recovery primary scopes required')
    for item in accounting['primary_artifacts']:
        data=read(item);rows=[r for r in cost['cost_records'] if r['artifact']==item]
        require(len(rows)==1 and rows[0]['aggregation_role']=='primary_observed_invocation','Each evaluation recovery primary scope once')
        if item==mixed['failed_E12_Windows_worker']:
            require(rows[0]['seconds_observed'] is None and rows[0]['observed_counters'].get('cpu_checkpoint_loads_returned')==1,'Failed unpersisted header scope must stay exact')
        else:
            expected=data['primary_stdlib_seconds'] if item==mixed['metrics_reuse_comparison'] else data['actual_windows_write_reproduction']['primary_stdlib_seconds']
            require(rows[0]['seconds_observed']==expected,'Actual independent stdlib elapsed differs')
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
    value={**noon,'fixed_manifest_metadata_exception':MAMBA_MANIFEST,'new_probability_or_OOF_payload_reads':0,'schema':'pause_memory_resume_cost_companion_v5','source':reader.snapshot(SOURCE)[1],
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
    human=REPORT/('pause_memory_and_three_loss_cost_v5_'+stamp+'.md')
    content='# 原进程暂停及三次丢失成本附页\n\n沿用真实中午暂停终态附页；三次丢失以原始封存证明关闭生命周期。丢失生产者仍为 running，尾部、退出时点、总秒数和有效秒数仍未知。暂停与守护等待不计训练，额外返回操作已包含原记录，不再加一次；不用墙钟与 QPC 混减。\n\n原中午附页：'+value['original_noon_human_report']['path']+'\n'
    with human.open('x',encoding='utf8') as f: f.write(content)
    value['human_readable_report']=Reader().snapshot(human)[1]; value['created_at']=datetime.now(timezone.utc).isoformat()
    target=REPORT/('pause_memory_resume_cost_companion_v5_'+stamp+'.json')
    with target.open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(Reader().snapshot(target)[1])); return 0

if __name__=='__main__': raise SystemExit(main())
