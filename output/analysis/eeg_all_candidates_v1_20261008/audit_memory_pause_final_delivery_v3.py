"""Final delivery gate for the Oct10 same-process memory pause.

This is an additional gate over completed cost18/wide8/semantic/G3 artifacts.
Only allowlisted source/JSON/Markdown bytes are read. It never executes prior
auditors, project verifiers, tensor operations, evaluation or process controls.
Original producer elapsed values stay inclusive; no precise pause subtraction.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
REPORT = SOURCE.parent
REL = 'output/analysis/eeg_all_candidates_v1_20261008/'
OUT = 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/'
FIT = 'REVE-lastblock-finetune/seed_22/fold_2'
REGISTRY = {'path': OUT + 'registration.json', 'sha256': '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'}
PINNED = {'cost': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/cost_aggregation_v18.py', 'sha256': '8f8d37c3fae0e408c36ddeb85643f4c2d37334bf973e0d5da80e62124e62ccd0'}, 'finalizer': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/finalize_full_plan_cost_v8.py', 'sha256': '8454f50133fde2b86aaf48cda632d0062bbda550ef1412cca50f0cfd09277c36'}, 'wide': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/audit_full_plan_acceptance_v8.py', 'sha256': '5b18cfbd5989d57b9451dc54230ad4284be1f38481e07246a15e8a57d32b58e3'}, 'g3': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/sync_boundary_v3_cost_companion_v3.py', 'sha256': '8a15e73227b1fc081b57986918f7af665b4b879ed1bd863a8d93f49dd13cf7eb'}, 'pause': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/pause_memory_resume_cost_companion_v3.py', 'sha256': '03c6f0543ade7d86daa972e1cc27e1dc4dedad20754205712b825ff9361d0790'}, 'resume_review': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/review_q5_memory_resume_20261010_v1.py', 'sha256': 'ef7464e3d987ab4cf56ce9405ae93f3050546c6e847e3d070aecd83649e6a224'}, 'resumer': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/resume_paused_q5_20261010_v1.ps1', 'sha256': '1f227764bc5b82351067319d0815e5d5926bb80baf3a7446b2ff55f761684761'}, 'original_g3': {'path': 'output/analysis/eeg_all_candidates_v1_20261008/sync_boundary_v3_cost_companion_v1.py', 'sha256': 'e2eb75c9b436c7d083a6446a92699d85c7261ce84db743141f1c15cfea0d3744'}}
RESUME_REVIEW = {'path': REL+'independent_q5_memory_resume_review_1791618064344382300.json','sha256':'ab6666acd982cc1c08b4a1c52094b6b8574ab2ab9146657385ed410d1ddceedf'}
PAUSE_PROOF = {'path': OUT+'pause_events/q5_user_pause_1791607365360/pause_complete.json','sha256':'fcbd717850d8cee8bc57ceb0594cf8dab482d10363f618f8d5cae7406bc37b3b'}
NATIVE_PAUSE = {'path': OUT+'pause_events/q5_user_pause_1791607365360/native_pause_complete.json','sha256':'acb9b67e4a15548cbcab476fb308ce29bfdc8e2d61a2fd35a39337cce5ac42b2'}
NATIVE_RESUME = {'path': OUT+'pause_events/q5_memory_resume_1791617874108/native_resume_complete.json','sha256':'87dc54eb1efc9e95aefa9fc9ce431e9294a628fb404e17151516c7ae4b257c78'}
SCOPE_ERRATUM = {'path': REL+'sync_control_companion_read_scope_erratum_1791602404565633400.json','sha256':'bc114598e11c57bb2cc5f09115c764054b6709932b9b3fa75b1f125edd8a2461'}
ROLES = {'gpu_child': (31128, '134360741986202743'), 'launcher': (16096, '134360741986106595'),
         'dispatcher': (45980, '134360707154222972'), 'dispatcher_launcher': (38860, '134360707153787861')}
TIMERS = {
    'F3_formal_attempt': (OUT+'finetune_v2_attempts/REVE-lastblock-finetune/f_1791607293628173200_cafa8752/attempt.json', 'formal_training_attempt', 'primary_observed_invocation', 'seconds'),
    'F1_metadata_subscope': (OUT+'runs/'+FIT+'/metadata.json', 'formal_legacy_metadata', 'supplemental_durable_metadata', 'seconds_this_attempt'),
    'Q5_parent_dispatcher': (OUT+'continuation_queue_v5/attempts/1791600598599694900-REVE-lastblock-finetune-fit/status.json', 'dispatcher_elapsed', 'supplemental_nested_or_unmeasured', 'seconds'),
}
INPUT_NAMES = ('final_cost', 'wide', 'semantic_review', 'g3_bundle', 'pause_companion', 'resume_review')
SEMANTIC_IDS = {'R04','R06','R10','R27','R29'}


def require(ok, message):
    if not ok: raise ValueError(message)


def integer(value): return type(value) is int and value >= 0
def number(value): return type(value) in (int,float) and math.isfinite(value) and value >= 0


class Pending(ValueError): pass


class Reader:
    def __init__(self, root=WORK): self.root, self.inputs = Path(root).resolve(), {}
    def path(self, path):
        target = (self.root / path).resolve()
        require(target.is_relative_to(self.root) and target.suffix.lower() in ('.py','.json','.md','.ps1'), 'Only workspace Python/JSON/Markdown/PowerShell evidence permitted')
        require(not any(x in target.name.lower() for x in ('probabilit','prediction','oof_')), 'Prediction/probability artifacts are outside this delivery gate')
        return target
    def snapshot(self, path):
        target = self.path(path); raw = target.read_bytes()
        binding = {'path':target.relative_to(self.root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}
        require(binding['path'] not in self.inputs or self.inputs[binding['path']] == binding, 'Evidence changed during review')
        self.inputs[binding['path']] = binding
        value = json.loads(raw.decode('utf-8-sig')) if target.suffix.lower()=='.json' else raw.decode('utf-8-sig')
        return value,binding
    def checked(self, binding):
        require(type(binding) is dict and set(binding)=={'path','sha256'}, 'Exact path/SHA descriptor required')
        value,actual = self.snapshot(binding['path']); require(actual==binding,'Bound evidence hash changed')
        return value
    def recheck(self):
        for binding in list(self.inputs.values()): self.checked(binding)


def verify_base(cost, wide, semantic, bundle, bindings, registry):
    require(cost.get('source')==PINNED['cost'] and cost.get('actual_cost_collector_source')==PINNED['cost']
            and cost.get('finalizer_source')==PINNED['finalizer'],'Final cost source differs')
    require(wide.get('source')==PINNED['wide'] and bundle.get('source')==PINNED['g3'],'Wide/G3 source differs')
    if not (cost.get('status')=='verified_final_cost' and cost.get('final_snapshot') is True
            and wide.get('objective_satisfied') is True and wide.get('full_plan_completed') is True
            and bundle.get('passed') is True): raise Pending('All actual base final artifacts must complete first')
    require(cost.get('draft_not_completion_certificate') is False and cost.get('all_available_artifacts_accounted') is True
            and cost.get('coverage_gaps_disclosed') is True,'Final cost flags differ')
    require(type(cost.get('schema_version')) is int and cost['schema_version']==18 and type(wide.get('schema_version')) is int and wide['schema_version']==8,'Base versions differ')
    require(cost.get('registration')==wide.get('registration')==REGISTRY,'Original registration differs')
    expected={r['fit_id'] for r in registry['fits']}
    require(len(registry['fits'])==len(expected)==360,'Original unique360 required')
    cost_rows, train = cost.get('formal_fits',[]), wide.get('training_rows',[])
    require(type(cost.get('formal_fits_expected')) is int and cost['formal_fits_expected']==360,'Final expected count differs')
    require(len(cost_rows)==len(train)==360 and {r['fit_id'] for r in cost_rows}=={r['fit_id'] for r in train}==expected,'Base unique360 identity coverage differs')
    updates={r['fit_id']:r['updates'] for r in train}
    require(all(integer(v) and v>0 for v in updates.values()) and all(type(r['formal_durable_updates']) is int and r['formal_durable_updates']==updates[r['fit_id']] for r in cost_rows)
            and cost['formal_durable_updates_known_subtotal']==sum(updates.values()),'Durable update counts differ')
    require(cost.get('formal_status_counts')==wide.get('status_counts')=={'training_complete':360}
            and cost.get('input_read_errors')==wide.get('verification_errors')==wide.get('incomplete_fits')==[],'Base partial/errors remain')
    require(all(type(wide.get(k)) is int and wide[k]==360 for k in ('formal_identity_count','completed_training_json_verified','deep_cpu_checkpoint_scope_verified'))
            and len(wide.get('cpu_checkpoint_proofs',[]))==360,'Full base verification shape differs')
    cpu=cost.get('source_bound_cpu_checkpoint_verification',{})
    require(cpu.get('passed') is True and type(cpu.get('verified_fits')) is int and cpu['verified_fits']==135
            and type(cpu.get('prior_actual_cpu_checkpoint_loads')) is int and cpu['prior_actual_cpu_checkpoint_loads']==405
            and type(cpu.get('new_cpu_checkpoint_loads')) is int and cpu['new_cpu_checkpoint_loads']==0,'Existing135/405 proof absent')
    requirements=wide.get('requirements',[])
    require(len(requirements)==30 and {r['id'] for r in requirements}=={'R%02d'%i for i in range(1,31)}
            and all(r.get('status')=='verified_complete' for r in requirements),'Original30 requirements not all verified')
    require(all(wide.get(k,{}).get('passed') is True for k in ('evaluation','semantic_review','final_cost','actual_training_lifecycle')),'Base evaluation/semantic/cost/lifecycle proof absent')
    require(wide['final_cost'].get('snapshot')==bindings['final_cost'] and wide['semantic_review'].get('artifact')==bindings['semantic_review'],'Wide artifact bindings differ')
    require(bundle.get('schema')=='eeg_final_cost_with_sync_boundary_v3_bundle_v3'
            and bundle.get('final_cost')==bindings['final_cost'] and bundle.get('full_plan_acceptance')==bindings['wide']
            and bundle.get('final_cost_human_report')==cost.get('human_readable_cost_report'),'G3 bundle belongs to different final artifacts')
    require(bundle.get('all_probe_primaries_already_in_final_cost_once') is True and bundle.get('companion_costs_added_again') is False
            and bundle.get('full_plan_verification_rerun') is False and type(bundle.get('GPU_or_evaluation_calls')) is int and bundle['GPU_or_evaluation_calls']==0,'G3 bundle scope differs')
    require(bundle.get('existing_final_artifact_shape_verified')=={'registered_and_verified_fits':360,'existing_finetune_CPU_fit_proof':135,'existing_finetune_CPU_load_proof':405,'scientific_checks_reexecuted':False},'G3 existing shape differs')
    parent=bundle.get('parent_attempt_timer',{})
    require(parent.get('seconds_added_to_formal_or_primary_probe') is False and 'pure_training_seconds' in parent
            and parent['pure_training_seconds'] is None and parent.get('hold_subtracted_to_infer_training_time') is False,'G3 parent hold/probe duration was relabelled or added')
    rows=[r for r in cost.get('cost_records',[]) if r.get('artifact')==parent.get('artifact')]
    require(len(rows)==1 and rows[0]['category']=='dispatcher_elapsed' and rows[0]['aggregation_role']=='supplemental_nested_or_unmeasured'
            and rows[0]['seconds_observed']==parent.get('seconds_observed_including_hold_and_probe'),'G3 parent timer binding/scope differs')
    require(semantic.get('kind')=='independent_final_semantic_review' and semantic.get('passed') is True and semantic.get('reviewer'),'Actual independent semantic review absent')
    expected_binding={'registration':REGISTRY,'audit_source':PINNED['wide'],
        'formal_complete_artifacts':sorted((r['completion'] for r in train),key=lambda x:x['path']),
        'evaluation_complete':wide['evaluation']['complete'],'cost_snapshot':bindings['final_cost']}
    require(semantic.get('binding')==expected_binding,'Semantic review bound to different final evidence')
    items=semantic.get('items',[])
    require(len(items)==5 and {r['id'] for r in items}==SEMANTIC_IDS,'Five semantic requirements required')
    for item in items:
        require(item.get('conclusion')=='verified' and type(item.get('finding')) is str and item['finding'].strip() and item.get('checks') and item.get('evidence'),'Substantive semantic finding absent')
        require(all(type(c) is dict and c.get('claim') and c.get('observed') and c.get('evidence_locations') for c in item['checks']),'Semantic check lacks evidence')
    require(cost.get('all_plan_wall_seconds_exact') is None and cost.get('all_plan_physical_optimizer_returns_exact') is None,'Unknown global history was made exact')
    return requirements


def verify_pause_timers(reader, pause, cost, wide):
    require(pause.get('schema')=='pause_memory_resume_cost_companion_v3' and pause.get('source')==PINNED['pause'],'Pause companion source/schema differs')
    if pause.get('status')!='verified_pause_scope_and_recorded_terminal_timers' or pause.get('all_affected_producer_timers_terminal') is not True:
        raise Pending('Pause scope is not a final three-producer terminal companion')
    require(pause.get('passed') is True and pause.get('affected_formal_fit_id')==FIT,'Pause fit/result differs')
    for key in ('pause_monotonic_exact_seconds','active_training_elapsed_exact_seconds','affected_active_known_subtotal'):
        require(key in pause and pause[key] is None,'Unknown pause/active duration must be explicit null')
    require(pause.get('pause_boundary_QPC_recorded') is False and pause.get('old_cost6_already_excludes_this_completed_pause') is False
            and pause.get('pause_snapshot_counters_added_to_terminal') is False and pause.get('new_loss_closure_created') is False,'Pause observation or accounting claim differs')
    require(type(pause.get('affected_active_unknown_formal_fit_count')) is int and pause['affected_active_unknown_formal_fit_count']==1
            and type(pause.get('affected_active_unknown_scopes')) is int and pause['affected_active_unknown_scopes']==3
            and pause.get('unknown_scope_count_includes_nested_metadata_and_parent_not_three_formal_fits') is True,'Nested scope/fit count differs')
    timers=pause.get('affected_timer_observations',[])
    require(len(timers)==3 and {r['scope'] for r in timers}==set(TIMERS),'Exactly three affected scopes required')
    evidence=[]
    for timer in timers:
        kind=timer['scope'];path,category,role,field=TIMERS[kind]
        require(timer.get('status')=='actual_terminal' and timer.get('artifact',{}).get('path')==path,'Affected timer path/state differs')
        seconds=timer.get('recorded_elapsed_including_native_pause_seconds')
        require(number(seconds) and seconds>0,'Actual positive inclusive elapsed required')
        for key in ('active_training_elapsed_exact_seconds','pause_monotonic_exact_seconds','active_known_subtotal_contribution'):
            require(key in timer and timer[key] is None,'Unknown affected duration was zero-filled or inferred')
        require(timer.get('eligible_for_active_known_subtotal') is False and timer.get('producer_elapsed_preserved') is True
                and timer.get('wall_pause_subtracted') is False and timer.get('nested_with_formal_attempt') is (kind!='F3_formal_attempt'),'Unsafe active/overlap classification')
        rows=[r for r in cost['cost_records'] if r['artifact']['path']==path]
        require(len(rows)==1 and rows[0]['artifact']==timer['artifact'] and rows[0]['category']==category and rows[0]['aggregation_role']==role,'Affected final-cost record missing/duplicate/misclassified')
        require(rows[0].get('seconds_observed')==seconds and number(rows[0]['seconds_observed']),'Final cost changed original producer seconds')
        producer=reader.checked(timer['artifact'])
        require(number(producer.get(field)) and producer[field]==seconds,'Companion no longer binds actual producer elapsed')
        if kind=='F1_metadata_subscope':
            require((producer.get('condition'),producer.get('seed'),producer.get('fold'))==('REVE-lastblock-finetune',22,2),'Metadata fit differs')
            completion=reader.checked(timer['completion_binding'])
            require(completion.get('passed') is True and completion.get('metadata')==timer['artifact'],'Metadata completion differs')
            final_fit=[r for r in cost['formal_fits'] if r['fit_id']==FIT]
            wide_fit=[r for r in wide['training_rows'] if r['fit_id']==FIT]
            require(len(final_fit)==len(wide_fit)==1 and final_fit[0]['metadata_artifact']==timer['artifact']
                    and final_fit[0]['last_attempt_seconds']==seconds and wide_fit[0]['completion']==timer['completion_binding'],'Per-fit nested final timer differs')
        else:
            require(producer.get('status')=='complete','Paused invocation must itself have completed; retry scenarios need another explicit scope review')
            if kind=='F3_formal_attempt':
                require((producer.get('condition'),producer.get('seed'),producer.get('fold'))==('REVE-lastblock-finetune',22,2)
                        and producer.get('graceful_terminal_flush') is True and producer.get('counter_snapshot_exact_for_process') is True,'F3 actual terminal identity differs')
            else: require(type(producer.get('returncode')) is int and producer['returncode']==0,'Original Q5 parent invocation did not return0')
        evidence.append({'scope':kind,'artifact':timer['artifact'],'recorded_inclusive_seconds':seconds,
            'active_known_eligible':False,'active_known_contribution':None,'exact_active_seconds':None,'exact_pause_seconds':None,
            'added_again':False,'base_category':category,'base_role':role})
    subtotal=recorded_formal_subset(cost)
    return {'passed':True,'affected_fit':FIT,'observed_timers':evidence,'affected_active_known_seconds_subtotal':None,
        'active_unknown_formal_fits':1,'overlapping_unknown_scopes':3,'producer_seconds_modified':False,
        'wall_or_resume_QPC_subtracted_from_producer_elapsed':False,'base_recorded_subtotals_are_inclusive_not_active':True,
        'no_plan_active_training_exact_total_claim':True,**subtotal}


def recorded_formal_subset(cost):
    """Filter actual primary formal records; this is not an active-time estimate."""
    affected=TIMERS['F3_formal_attempt'][0];known=[];unknown=[];excluded=[];seen=set()
    for row in cost['cost_records']:
        if row.get('category')!='formal_training_attempt' or row.get('aggregation_role')!='primary_observed_invocation': continue
        binding=row['artifact'];require(binding['path'] not in seen,'Duplicate primary formal elapsed record');seen.add(binding['path'])
        if binding['path']==affected: excluded.append(binding);continue
        seconds=row.get('seconds_observed')
        if seconds is None: unknown.append(binding)
        else:
            require(number(seconds),'Malformed known primary formal seconds')
            known.append({'artifact':binding,'recorded_seconds':seconds})
    require(len(excluded)==1,'Affected paused formal timer must be excluded exactly once')
    return {'known_recorded_formal_seconds_excluding_affected_native_pause_timer':sum(x['recorded_seconds'] for x in known) if known else None,
        'known_recorded_formal_timer_records':known,'retained_unknown_formal_timer_records':unknown,
        'excluded_native_pause_formal_timer_records':excluded,'subset_includes_metadata_or_dispatcher_seconds':False,
        'subset_is_exact_active_training_time':False,'unknown_records_zero_filled':False}


def verify_resume(reader, pause, review, binding):
    require(binding==RESUME_REVIEW and review.get('schema')=='independent_q5_memory_resume_review_v1'
            and review.get('passed') is True and review.get('review_source')==PINNED['resume_review']
            and review.get('resume_source')==pause.get('resumer_source')==PINNED['resumer'],'Independent resume proof differs')
    require(pause.get('pause_proof')==PAUSE_PROOF and pause.get('native_pause')==NATIVE_PAUSE,'Original pause proof differs')
    resume=pause.get('resume_verification',{})
    require(resume.get('status')=='actual_native_resume_and_controls_release_verified' and resume.get('native_resume')==NATIVE_RESUME,'Actual completed resume required')
    require(resume.get('controls_release')==review.get('controls_release_evidence'),'Control release bindings differ')
    require(resume.get('same_native_identities')=={role:{'pid':pid,'creation_100ns':birth} for role,(pid,birth) in ROLES.items()},'Same native objects differ')
    require(review.get('same_four_original_objects') is True and type(review.get('total_original_threads_verified')) is int and review['total_original_threads_verified']==67
            and type(review.get('total_original_threads_still_suspended_after_recorded_resume')) is int and review['total_original_threads_still_suspended_after_recorded_resume']==0,'Original threads release proof differs')
    rows=review.get('processes',[])
    require(len(rows)==4 and {r['role'] for r in rows}==set(ROLES),'Four original roles required')
    for row in rows:
        pid,birth=ROLES[row['role']]
        require(type(row.get('pid')) is int and row['pid']==pid and row.get('creation_100ns')==birth,'PID/birth differs')
        for key,expected in (('native_suspend_calls_in_bound_pause',1),('native_resume_calls_in_bound_resume',1),('native_resume_status',0),('post_resume_original_suspended',0)):
            require(type(row.get(key)) is int and row[key]==expected,'Resume exact-once/return field differs')
    for value,keys in ((review,('exact_pause_elapsed_seconds','exact_active_elapsed_seconds')),
                       (resume,('pause_monotonic_exact_seconds','active_training_elapsed_exact_seconds'))):
        require(all(k in value and value[k] is None for k in keys),'Unknown pause/active time must remain null')
    require(review.get('pause_clock',{}).get('pause_boundary_qpc','missing') is None,'Pause unexpectedly claims a QPC boundary')
    for evidence in review.get('inputs',[]): reader.checked(evidence)
    require(PAUSE_PROOF in review['inputs'] and NATIVE_RESUME in review['inputs'] and NATIVE_PAUSE in review['inputs'],'Independent proof did not bind actual pause/resume')
    for evidence in pause.get('immutable_inputs',[]): reader.checked(evidence)
    for key in ('native_resume','preflight','controls_release'): reader.checked(resume[key])
    require(len(resume.get('archived_controls',[]))==4,'Four archived controls required')
    for evidence in resume['archived_controls']: reader.checked(evidence)
    native=reader.checked(NATIVE_RESUME)
    require(native.get('source')==PINNED['resumer'] and native.get('status')=='same_four_objects_resumed_by_explicit_continue','Native resume source/result differs')
    actions=native.get('actions',[])
    require(len(actions)==4 and {r['role'] for r in actions}==set(ROLES)
            and all(type(r.get('native_status')) is int and r['native_status']==0 for r in actions),'Four successful native calls required')
    return {'passed':True,'pause_proof':PAUSE_PROOF,'native_resume':NATIVE_RESUME,'independent_review':binding,
        'resumer_source':PINNED['resumer'],'same_native_identities':resume['same_native_identities'],
        'one_resume_per_original_object':True,'controls_release':resume['controls_release'],
        'archived_controls':resume['archived_controls'],'claims_current_process_liveness':False,
        'exact_pause_seconds':None,'exact_active_seconds':None}



def verify_three_loss_companion(reader,pause,cost,wide,bindings):
    path=reader.path(PINNED['pause']['path']); reader.checked(PINNED['pause'])
    ns={'__file__':str(path),'__name__':'_text_only_loss_companion'}
    exec(compile(path.read_text(encoding='utf8'),str(path),'exec'),ns)
    pair=ns['verify_final_pair'](reader.checked,cost,wide,bindings['final_cost'],bindings['wide'])
    require(pause.get('final_artifacts')==pair and pause.get('original_noon_companion')==ns['NOON']
        and pause.get('old_noon_companion_recollected') is False and pause.get('all_affected_timer_lifecycles_closed') is True,'Pause companion belongs to another final pair or loses noon lineage')
    noon=reader.checked(ns['NOON']);reader.checked(ns['NOON_SOURCE']);reader.checked(pause['original_noon_human_report'])
    require(pause['original_noon_human_report']==noon['human_readable_report'],'Original noon report changed')
    for k in ('affected_timer_observations','resume_verification','pause_proof','native_pause','resumer_source'):
        require(pause.get(k)==noon.get(k),'Original noon evidence changed: '+k)
    losses=ns['verify_loss_chain'](reader.checked,cost)
    require(pause.get('loss_chain')==losses and pause.get('all_plan_active_seconds_exact','missing') is None,'Three loss proof or unknown active scope differs')
    return losses

def audit(paths, reader=None):
    reader=reader or Reader()
    base={'schema':'memory_pause_final_delivery_gate_v3','passed':False,'objective_satisfied':False,'full_plan_completed':False,
          'status':'pending','R01_R30_derived_after_all_prerequisites':False}
    missing=[key for key in INPUT_NAMES if not paths.get(key)]
    if missing: return {**base,'reason':'Actual final inputs required','missing_inputs':missing},reader
    try:
        values,bindings={},{}
        for key in INPUT_NAMES: values[key],bindings[key]=reader.snapshot(paths[key])
        for binding in PINNED.values(): reader.checked(binding)
        registry=reader.checked(REGISTRY)
        requirements=verify_base(values['final_cost'],values['wide'],values['semantic_review'],values['g3_bundle'],bindings,registry)
        extra=verify_three_loss_companion(reader,values['pause_companion'],values['final_cost'],values['wide'],bindings)
        r28=verify_pause_timers(reader,values['pause_companion'],values['final_cost'],values['wide'])
        r29=verify_resume(reader,values['pause_companion'],values['resume_review'],bindings['resume_review'])
        for evidence in (values['final_cost']['human_readable_cost_report'],values['g3_bundle']['sync_control_human_report'],values['pause_companion']['human_readable_report']): reader.checked(evidence)
        erratum=reader.checked(SCOPE_ERRATUM)
        require(erratum.get('passed') is True and erratum.get('affected_companion')==values['g3_bundle']['sync_control_cost_companion'],'G3 scope erratum does not bind actual bundled companion')
        g3_companion=reader.checked(values['g3_bundle']['sync_control_cost_companion'])
        require(g3_companion.get('passed') is True and g3_companion.get('status')=='verified_terminal_control_cost' and g3_companion.get('source')==PINNED['original_g3'],'Actual G3 companion differs')
        result_requirements=[{**row,'base_artifact':bindings['wide']} for row in requirements]
        for row in result_requirements:
            if row['id'] in ('R28','R29'):
                row['additional_memory_pause_gate']='R28_pause_timer_scope' if row['id']=='R28' else 'R29_memory_resume_control'
        complete=all(row['status']=='verified_complete' for row in result_requirements if row['id'] not in ('R01','R30')) and r28['passed'] and r29['passed']
        for row in result_requirements:
            if row['id'] in ('R01','R30'): row.update(status='verified_complete' if complete else 'pending',derived_from_all_R02_R29_plus_memory_pause_gates=True)
        reader.recheck()
        return {**base,'passed':complete,'objective_satisfied':complete,'full_plan_completed':complete,
            'status':'verified_final_delivery_with_memory_pause' if complete else 'pending',
            'R01_R30_derived_after_all_prerequisites':True,'base_final_artifacts':bindings,'scope_erratum':SCOPE_ERRATUM,
            'R28_pause_timer_scope':r28,'R29_memory_resume_control':r29,'three_loss_lifecycle_scope':extra,'requirements':result_requirements,
            'base_30_requirements_preserved':True,'old_finalized_bytes_modified':False,
            'base_recorded_elapsed_totals_not_relabelled_as_active':True,'all_plan_exact_active_seconds':None,
            'new_final_cost_snapshot_or_scientific_reaudit_created':False},reader
    except Pending as error: return {**base,'reason':str(error)},reader
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {**base,'status':'failed','reason':str(error),'error_type':type(error).__name__},reader


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in INPUT_NAMES: parser.add_argument('--'+name.replace('_','-'),type=Path)
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args(argv)
    if args.require_complete and not args.run: parser.error('--require-complete requires --run')
    if not args.run: print(json.dumps({'status':'prepared_only','operations':0})); return 0
    reader=Reader();_,source=reader.snapshot(SOURCE)
    value,reader=audit({name:getattr(args,name) for name in INPUT_NAMES},reader)
    reader.checked(source)
    value.update(source=source,created_at=datetime.now(timezone.utc).isoformat(),input_bindings=list(reader.inputs.values()),
        operations={'binary_or_prediction_reads':0,'tensor_loads':0,'models_GPU_process_operations':0,'project_verifier_imports_or_calls':0,
                    'existing_final_artifacts_modified':False,'training_or_evaluation_reexecuted':False})
    target=REPORT/('memory_pause_final_delivery_v3_'+value['status']+'_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as stream: json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'path':target.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'status':value['status'],'passed':value['passed']}))
    return 0 if value['passed'] or (value['status']=='pending' and not args.require_complete) else 2


if __name__=='__main__': raise SystemExit(main())
