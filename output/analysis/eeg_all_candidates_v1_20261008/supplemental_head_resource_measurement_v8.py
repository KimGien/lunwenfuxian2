"""Unchanged twelve zero-update head probes behind actual Q14/CPU17/E12 gates.

Default is no operation. --check-only reads status JSON. --run requires the
separate exact-source review, full completed evaluation, and released owners.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
SOURCE=Path(__file__).resolve(); WORK=SOURCE.parents[3]; REPORT=SOURCE.parent
OUT_REL='tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
OUT=WORK/OUT_REL; REPORT_REL='output/analysis/eeg_all_candidates_v1_20261008'
sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S
V2_SOURCE=REPORT/'supplemental_head_resource_measurement_v2.py'
V2_SHA='351f67b6941b12cfb8139372783d92158c5040d7ad20abf4a8b2d435e5a9a54b'
POST_SOURCE=REPORT/'repair_postprocessing_gate_v9.py'
POST_SHA='604b1cbcb917fdb0b71c463089db677f386c90473b649563c44ab5f4bb377661'

def require(ok,message):
    if not ok:raise ValueError(message)

def load_pinned(path,digest,name):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==digest,'Sealed source changed: '+Path(path).name)
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module

D=load_pinned(V2_SOURCE,V2_SHA,'_resource8_private_unchanged_v2')
D.SOURCE=SOURCE
P=load_pinned(POST_SOURCE,POST_SHA,'_resource8_reviewed_post9')
read,descriptor,checked=D.read,D.descriptor,D.checked
write_new,now=D.write_new,D.now
probe_plan,exclusive_measurement_lease=D.probe_plan,D.exclusive_measurement_lease

def delegated_source():
    require(S.digest(V2_SOURCE)==V2_SHA,'Original twelve resource probe source changed')
    return descriptor(V2_SOURCE)

def source_review(binding):
    require(isinstance(binding,dict),'Actual independent postprocessing source review required')
    value=read(checked(binding))
    expected=[descriptor(REPORT/name) for name in ('cost_aggregation_v18.py', 'repair_postprocessing_gate_v9.py', 'supplemental_head_resource_measurement_v8.py', 'finalize_full_plan_cost_v8.py', 'audit_full_plan_acceptance_v8.py', 'pause_memory_resume_cost_companion_v3.py', 'sync_boundary_v3_cost_companion_v3.py', 'audit_memory_pause_final_delivery_v3.py', 'system_standby_cost_companion_v3.py', 'audit_system_standby_final_delivery_v3.py')]
    require(value.get('passed') is True and value.get('status')=='final_postprocessing_sources_reviewed'
            and value.get('reviewed_sources')==expected
            and value.get('training_registration')==descriptor(OUT/'continuation_queue_v14/registration.json'),
            'Independent review does not bind these ten actual sources and Q14')
    checked(value['reviewer_source'])
    registration=read(checked(value['training_registration']))
    require(len(registration['sources'])==194,'Actual final194 training sources required')
    for item in registration['sources']:checked(item)
    return binding

def lifecycle_evidence():
    """Reuse post9's complete prefix, stopping before its first report read."""
    raw=POST_SOURCE.read_text(encoding='utf-8')
    require(S.digest(POST_SOURCE)==POST_SHA,'Reviewed post9 source changed')
    function=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='verify_existing')
    boundaries=[i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='manifest' for t in n.targets)]
    require(len(boundaries)==1,'Exact pre-report lifecycle boundary required')
    function.name='_verified_lifecycle_prefix'
    function.body=function.body[:boundaries[0]]+ast.parse("return {'passed':True,'registration':reg_binding,'training_registration':gate['registration'],'training_terminal':S.desc(Q6.ROOT/'status.json'),'CPU_gate':gate_binding,'CPU_audit':gate['CPU_audit'],'CPU_auditor':cpu_source,'evaluation_control_completion':S.desc(control_path),'evaluation_runtime':control['status'],'evaluation_released_owner':S.desc(released),'CodeBrain_completion':gate['previous_CodeBrain_completion'],'training_sources':queue['sources'],'training_source_review':queue['source_review'],'formal_fits_verified':360,'FT_verified':135,'CPU_loads_verified':405,'post_training_recovery_gate':gate_binding,'post_training_source_review':gate['post_training_source_review'],'post_training_sources':gate['post_training_sources'],'failed_CPU16':gate['failed_CPU16'],'original_Q14_released_owner':gate['original_Q14_released_owner'],'original_child_complete':gate['original_child_complete'],'Linux_LR_reference':RECOVERY.LINUX_REFERENCE,'Windows_LR_reference':RECOVERY.WINDOWS_REFERENCE,'existing_report_verified':False}").body
    namespace=dict(vars(P))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(POST_SOURCE)+':lifecycle_prefix','exec'),namespace)
    result=namespace['_verified_lifecycle_prefix']()
    cpu=read(checked(result['CPU_audit']))
    for row in cpu['completed_rows']:
        completion=read(checked(row['completion']))
        require(read(checked(row['status_artifact'])).get('evidence')==completion and row['metadata']==completion['metadata'],'CPU17 status/completion/metadata bytes differ')
        require(completion['checkpoint'] in row['cpu_checkpoint_audit']['checkpoints'],'Selected checkpoint absent from actual CPU17 proof')
        for binding in row['cpu_checkpoint_audit']['checkpoints']:checked(binding)
    result['CPU_checkpoint_same_bytes_verified']=405
    return result

def inspect_gate(*, review=None, verify_report=True):
    value=P.inspect()
    value.update(source=descriptor(SOURCE),checked_at=now(),model_constructions=0,gpu_calls=0,actual_completed_report_verifier_called=False)
    if value['status']!='metadata_ready':return value
    reviewed=source_review(review)
    lifecycle=lifecycle_evidence()
    if not verify_report:
        return {**value,'status':'ready_for_readonly_verification','training_lifecycle':lifecycle,'postprocessing_source_review':reviewed,'reasons':[]}
    proof=P.verify_existing()
    require(proof['CPU_gate']==lifecycle['CPU_gate'] and proof['q14_registration']==lifecycle['training_registration'], 'Post8 lifecycle changed')
    return {**value,'status':'ready_after_full_e12_verification','reasons':[],
            'registration':lifecycle['registration'],'training_lifecycle':lifecycle,
            'evaluation_verification':proof['evaluation_verification'],
            'postprocessing_source_review':reviewed,'delegated_resource_measurement_source':delegated_source(),
            'actual_completed_report_verifier_called':True,'outer_existing_results_verified':True}

def assert_terminal_still_bound(gate,out):
    P.Q.check_pause()
    require(Path(out).resolve()==OUT.resolve(),'Actual experiment required')
    require(all(not (OUT/name).exists() for name in P.OWNERS),'A training/evaluation/qualification owner appeared')
    source_review(gate['postprocessing_source_review'])
    lifecycle=gate['training_lifecycle']
    for key in ('registration','training_registration','training_terminal','CPU_gate','CPU_audit','CPU_auditor','evaluation_control_completion','evaluation_runtime','evaluation_released_owner','post_training_recovery_gate','post_training_source_review','failed_CPU16','original_Q14_released_owner','original_child_complete','Linux_LR_reference','Windows_LR_reference'):
        checked(lifecycle[key])
    for key in ('registration','terminal','CPU_audit','child'):checked(lifecycle['CodeBrain_completion'][key])
    checked(gate['evaluation_verification']['complete_manifest'])
    delegated_source()

def write_measurement(path,value):
    if value.get('subtype')=='supplemental_head_resource_measurement' and 'measurement_scope' in value:
        value['delegated_resource_measurement_source']=delegated_source()
        value['resource_measurement_version']=8
        value['measurement_scope']['seconds']=value['measurement_scope']['seconds'].replace('E2 prerequisite','E12 prerequisite')
    write_new(path,value)

def successful_existing(path,*,plan,gate,source):
    record=read(path)
    return (record.get('resource_measurement_version')==8
            and record.get('delegated_resource_measurement_source')==delegated_source()
            and record.get('gate',{}).get('training_lifecycle')==gate['training_lifecycle']
            and record.get('gate',{}).get('postprocessing_source_review')==gate['postprocessing_source_review']
            and D.successful_existing(path,plan=plan,gate=gate,source=source))

D.assert_terminal_still_bound=assert_terminal_still_bound
D.write_new=write_measurement
measure=D.measure

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--run',action='store_true');group.add_argument('--check-only',action='store_true')
    parser.add_argument('--source-review');parser.add_argument('--source-review-sha256')
    args=parser.parse_args(argv)
    if not args.run:
        print(json.dumps(P.inspect() if args.check_only else {'status':'preparation_only','model_calls':0,'GPU_queries':0,'new_outputs':0}));return 0
    require(args.source_review and args.source_review_sha256,'Actual reviewed source binding required')
    review=descriptor(WORK/args.source_review)
    require(review['sha256']==args.source_review_sha256,'Explicit independent review SHA differs')
    gate=inspect_gate(review=review)
    if gate['status']!='ready_after_full_e12_verification':print(json.dumps(gate));return 2
    gate_path=REPORT/('supplemental_head_resource_v8_gate_'+str(time.time_ns())+'.json')
    write_new(gate_path,gate)
    registration=read(checked(gate['registration']))
    conditions=[r['condition_id'] for r in registration['configuration']['conditions'] if r['policy_key'] in ('diagnostic','frozen')]
    require(len(conditions)==len(set(conditions))==12,'Exactly twelve original head probes required')
    completed=[]
    with exclusive_measurement_lease(OUT):
        for condition in conditions:
            assert_terminal_still_bound(gate,OUT)
            plan=probe_plan(condition,registration)
            existing=[p for p in (OUT/'supervised_engineering'/condition).glob('measurement_*.json') if successful_existing(p,plan=plan,gate=gate,source=descriptor(SOURCE))]
            require(len(existing)<=1,'Duplicate completed measurements need explicit cost review')
            if existing:checked(read(existing[0])['resume_probe']);completed.append(descriptor(existing[0]))
            else:completed.append(measure(plan,gate))
    value={'subtype':'supplemental_head_resource_measurement_completion','passed':True,'source':descriptor(SOURCE),'delegated_resource_measurement_source':delegated_source(),'resource_measurement_version':8,'gate':descriptor(gate_path),'conditions':conditions,'measurements':completed,'optimizer_updates':0,'historical_missing_measurements_backfilled':False,'completed_at':now()}
    path=REPORT/('supplemental_head_resource_v8_complete_'+str(time.time_ns())+'.json');write_new(path,value)
    print(json.dumps({'complete':descriptor(path),'conditions':12,'optimizer_updates':0}));return 0
if __name__=='__main__':raise SystemExit(main())
