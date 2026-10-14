"""Original thirty requirements bound to actual Q13/Q14/CPU16/E11/Cost17.

Default zero operation; pending --run reads prerequisite status JSON only.
Completed FT reuses exact CPU16 three-file proofs without repeating Tensor
loads. All other original CPU, metric, OOF, cost and semantic checks remain.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import types
SOURCE=Path(__file__).resolve();WORK=SOURCE.parents[3];REPORT=SOURCE.parent
OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
DELEGATE=REPORT/'audit_full_plan_acceptance_v4.py'
DELEGATE_SHA='6e659d2ca7a9d3a364f04aaef8d17385608d7db6a25e9e8b5e941f657b440f84'
RESOURCE_SOURCE=REPORT/'supplemental_head_resource_measurement_v7.py'
RESOURCE_SHA='3b85f4c444fff69711a1299ec260b0695f5896b8caff48a631fcc0ec28f36711'
FINALIZER_SOURCE=REPORT/'finalize_full_plan_cost_v7.py'
FINALIZER_SHA='ab5d4d892f3d6a0e75c2c82db5bd3bbcfdffd086b4be4f92858234612982a712'
CPU_SOURCE=REPORT/'audit_completed_finetune_checkpoints_v16.py'
CPU_SHA='108c7cb87fa2e8e64e7882fa0365fa431c55e2e67a34cfbe95310ee8386fa9e3'

def require(ok,message):
    if not ok:raise ValueError(message)

def load_pinned(path,digest,name):
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==digest,'Sealed source changed: '+path.name)
    module=types.ModuleType(name);module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__);return module

D=load_pinned(DELEGATE,DELEGATE_SHA,'_wide7_original_thirty_math');D.SOURCE=SOURCE
S,R,P=D.S,D.R,D.P
RES=load_pinned(RESOURCE_SOURCE,RESOURCE_SHA,'_wide7_resource7')
FIN=load_pinned(FINALIZER_SOURCE,FINALIZER_SHA,'_wide7_finalizer7')
CPU=load_pinned(CPU_SOURCE,CPU_SHA,'_wide7_CPU16')
from tridep_ccml.eeg_all_candidates import evaluate_v11 as E
ACTUAL_REVIEW=None
CPU_ROWS={}
LIFECYCLE=None

def __getattr__(name):return getattr(D,name)

def check_fit(fit,status,registration,lock,contract):
    row=D.legacy_check_fit(fit,status,registration,lock,contract)
    metadata=S.read(S.checked(row['metadata']))
    if P.CONDITION_BY_ID[fit['condition']].policy_key=='finetune':
        inputs=CPU.B.trace_inventory(fit)
        proof=CPU.B.trace_boundary(fit,metadata,inputs)
        CPU.B.recheck_trace_inventory(fit,inputs)
        row['actual_execution_observer_proof']=proof
        row['execution_auditor_source']=S.desc(CPU_SOURCE)
        row['actual_training_execution_proof']=E.verify_training_execution(metadata)
    else:require('execution_acceleration' not in metadata and 'actual_deterministic_repair_execution' not in metadata,'FT execution appeared on non-FT identity')
    return row

_original_deep=D.deep_checkpoint

def deep_checkpoint(fit,row,contract,metadata):
    if P.CONDITION_BY_ID[fit['condition']].policy_key!='finetune':return _original_deep(fit,row,contract,metadata)
    require(LIFECYCLE is not None and fit['fit_id'] in CPU_ROWS,'Actual complete CPU16 proof required before FT tensor reuse')
    prior=CPU_ROWS[fit['fit_id']]
    for key in ('fit_id','condition','seed','fold','epochs','updates','best_epoch','stop','metadata','completion','condition_source_lock'):
        require(prior[key]==row[key],'CPU16 proof differs from final training: '+key)
    require(prior['f2_physical_durable_boundary']==row['actual_execution_observer_proof'],'CPU16 physical trace changed')
    deep=prior['cpu_checkpoint_audit']
    require(deep.get('cpu_tensors_verified') is True and deep.get('checkpoint_loads')==3 and deep.get('only_map_location_cpu') is True and deep.get('gpu_operations')==0,'Actual CPU16 three-file proof incomplete')
    for binding in deep['checkpoints']:S.checked(binding)
    return {**deep,'fit_id':fit['fit_id'],'reused_actual_CPU16_audit':LIFECYCLE['CPU_audit'],'prior_checkpoint_loads':3,'new_checkpoint_loads':0,'scope':'Prior actual CPU16 Tensor checks plus current same-byte proof; no new FT Tensor load.'}

def bind_original_consumers():
    tree=ast.parse(DELEGATE.read_text(encoding='utf-8'))
    for name in ('evaluation_audit','final_cost_audit'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        changed=0
        if name=='evaluation_audit':
            for item in ast.walk(node):
                if isinstance(item,ast.ImportFrom) and item.module=='tridep_ccml.eeg_all_candidates':
                    for alias in item.names:
                        if alias.name=='evaluate_v3' and alias.asname=='E2':alias.name='evaluate_v11';changed+=1
            require(changed==1,'Exact original E3 import replacement required')
        else:
            replacements={'cost_aggregation_v5.py':'cost_aggregation_v17.py',5:17};counts={k:0 for k in replacements}
            for item in ast.walk(node):
                if isinstance(item,ast.Constant) and type(item.value) in (str,int) and item.value in replacements:
                    counts[item.value]+=1;item.value=replacements[item.value]
            require(all(v==1 for v in counts.values()),'Exact original cost source/schema replacement required')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE)+':unchanged_math','exec'),D.__dict__)
bind_original_consumers()
_original_evaluation=D.evaluation_audit
_original_cost=D.final_cost_audit

def evaluation_audit(registration):
    require(RES.lifecycle_evidence()==LIFECYCLE,'Actual mixed lifecycle changed')
    return _original_evaluation(registration)

def final_cost_audit(path,companion,training_rows):
    require(path is not None,'Actual Cost17 final artifact required')
    value=S.read(path)
    require(value.get('finalizer_source')==S.desc(FINALIZER_SOURCE) and value.get('actual_cost_collector_source')==S.desc(REPORT/'cost_aggregation_v17.py'),'Actual finalizer7/Cost17 source required')
    require(value.get('postprocessing_source_review')==ACTUAL_REVIEW and value.get('actual_training_lifecycle')==LIFECYCLE,'Final cost belongs to another review or lifecycle')
    FIN.ACTUAL_REVIEW=ACTUAL_REVIEW
    gate=FIN.preflight()
    require(gate['status']=='ready_for_readonly_verification','Actual twelve-resource/CPU16/lifecycle gate required')
    recorded=value.get('completion_gate',{})
    for key in ('training_lifecycle','verified_final_sources','gate_artifacts','training_status_descriptors','supplemental_head_resource_gate','completed_finetune_cpu_gate','recovery_execution_proof','postprocessing_source_review'):
        require(recorded.get(key)==gate.get(key),'Final cost gate changed: '+key)
    require(value.get('source_bound_cpu_checkpoint_verification')==gate['completed_finetune_cpu_gate'],'Cost CPU16 proof differs')
    require(value.get('repair_ledger_coverage',{}).get('errors')==[] and value.get('process_loss_closures_aggregated_once') is True,'Actual repair/loss coverage missing')
    primary=[x for x in value['source_bound_lost_f3_attempts'] if x['aggregation_role']=='primary_observed_invocation']
    require(len(primary)==3 and len({x['closure']['path'] for x in primary})==3,'Three actual losses required')
    return _original_cost(path,companion,training_rows)

D.check_fit=check_fit;D.deep_checkpoint=deep_checkpoint;D.evaluation_audit=evaluation_audit;D.final_cost_audit=final_cost_audit

def pending(cheap):
    preparation=S.read(D.PREPARATION)
    require(len(preparation['requirements'])==30 and {x['id'] for x in preparation['requirements']}=={'R'+str(i).zfill(2) for i in range(1,31)},'Original thirty requirements required')
    return {'schema_version':7,'source':S.desc(SOURCE),'status':'pending','objective_satisfied':False,'full_plan_completed':False,'formal_identity_count':360,'status_counts':cheap['actual_fit_statuses'],'reasons':cheap['reasons'],'requirements':[{**x,'status':'pending','reason':'Actual full training, CPU16 and E11 prerequisites still pending.'} for x in preparation['requirements']],'outer_test_artifacts_read':False,'new_checkpoint_loads':0,'model_calls':0,'GPU_queries':0}

def audit(*,cost_artifact=None,cost_report=None,semantic_review=None,source_review=None):
    global ACTUAL_REVIEW,CPU_ROWS,LIFECYCLE
    cheap=RES.P.inspect()
    if cheap['status']!='metadata_ready':return pending(cheap)
    ACTUAL_REVIEW=RES.source_review(source_review)
    FIN.ACTUAL_REVIEW=ACTUAL_REVIEW
    fullgate=FIN.preflight()
    require(fullgate['status']=='ready_for_readonly_verification','Actual twelve resources and full lifecycle required')
    LIFECYCLE=fullgate['training_lifecycle']
    cpu=S.read(S.checked(LIFECYCLE['CPU_audit']))
    CPU_ROWS={row['fit_id']:row for row in RES.P.validate_CPU_shape(cpu,S.read(OUT/'registration.json'),S.desc(OUT/'registration.json'),S.desc(CPU_SOURCE))}
    result=D.audit(cost_artifact=cost_artifact,cost_report=cost_report,semantic_review=semantic_review)
    require(RES.lifecycle_evidence()==LIFECYCLE,'Actual lifecycle changed during final acceptance')
    result.update(schema_version=7,source=S.desc(SOURCE),actual_training_lifecycle=LIFECYCLE,postprocessing_source_review=ACTUAL_REVIEW,actual_CPU16_auditor=S.desc(CPU_SOURCE),prior_actual_FT_CPU_loads=405,new_FT_CPU_loads=0,actual_evaluator_source=S.desc(E.SOURCE),actual_cost_collector_source=S.desc(REPORT/'cost_aggregation_v17.py'),original_thirty_requirement_math_preserved=True,backward_arithmetic_repair_not_original_cuda_equivalence=True,separate_G3_pause_standby_delivery_gates_still_required=True)
    require(len(result['requirements'])==30,'Original thirty requirements were reduced')
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--require-complete',action='store_true')
    for name in ('cost-artifact','cost-report','semantic-review','source-review','source-review-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args(argv)
    if not args.run:
        require(not args.require_complete,'Actual --require-complete needs --run')
        print(json.dumps({'status':'preparation_only','Tensor_loads':0,'model_calls':0,'GPU_queries':0,'new_outputs':0}));return 0
    review=None
    if args.source_review or args.source_review_sha256:
        require(args.source_review and args.source_review_sha256,'Both independent review fields required')
        review=S.desc(WORK/args.source_review)
        require(review['sha256']==args.source_review_sha256,'Explicit source-review SHA differs')
    result=audit(cost_artifact=args.cost_artifact,cost_report=args.cost_report,semantic_review=args.semantic_review,source_review=review)
    path=REPORT/('full_plan_acceptance_v7_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'artifact':S.desc(path),'full_plan_completed':result['full_plan_completed'],'requirements':len(result['requirements'])}))
    return 1 if args.require_complete and not result['full_plan_completed'] else 0
if __name__=='__main__':raise SystemExit(main())
