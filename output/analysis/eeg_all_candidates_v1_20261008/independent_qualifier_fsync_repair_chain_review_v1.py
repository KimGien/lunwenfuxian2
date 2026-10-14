"""Independent Qual5 fsync-only / repair115 chain review; no actual run entries."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
SOURCE=Path(__file__).resolve();A=SOURCE.parent;W=SOURCE.parents[3];M=W/'tridep_ccml/eeg_all_candidates';OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
checks={}
def require(ok,message):
    if not ok:raise ValueError(message)
def check(label,value):require(value,label);checks[label]=True
def reject(label,fn):
    try:fn()
    except (ValueError,KeyError,AssertionError):checks[label]=True
    else:raise AssertionError('Unsafe acceptance '+label)
def desc(p):
    p=Path(p).resolve();return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def checked(binding):
    p=W/binding['path'];require(desc(p)==binding,'Changed bound input '+str(p));return p
def parsed(p):return ast.parse(p.read_text(encoding='utf-8-sig'))
def function(p,name):return next(n for n in parsed(p).body if isinstance(n,ast.FunctionDef) and n.name==name)
def functions(p,names,ns):
    nodes=[copy.deepcopy(function(p,n)) for n in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preparation',required=True);args=parser.parse_args()
    preparation_path=W/args.preparation;preparation=read(preparation_path)
    check('fresh_contract_preparation_passed',preparation.get('passed') is True)
    # Exact full qualifier AST equivalence after only explicit version paths and fsync mode mapping.
    old=M/'qualify_deterministic_repairs_v4.py';new=M/'qualify_deterministic_repairs_v5.py'
    text=new.read_text(encoding='utf-8').replace('qualify_deterministic_repairs_v5','qualify_deterministic_repairs_v4').replace('deterministic_repair_qualification_v5','deterministic_repair_qualification_v4')
    check('qualifier_exactly_one_fsync_mode_fix',text.count("checkpoint.open('r+b')")==1)
    text=text.replace("checkpoint.open('r+b')","checkpoint.open('rb')")
    check('qualifier_full_AST_same_except_version_paths_and_fsync_handle',ast.dump(ast.parse(text),include_attributes=False)==ast.dump(parsed(old),include_attributes=False))
    check('qualifier_real_main_after_all_definitions',isinstance(parsed(new).body[-1],ast.If) and '__name__' in ast.unparse(parsed(new).body[-1].test))
    diagnosis=read(A/'actual_windows_fsync_diagnosis_1791776209166909300.json')
    check('real_storage_diagnosis_rb9_rplusb_unchanged_bytes',diagnosis['actual_platform_fsync']['rb']['errno']==9 and diagnosis['actual_platform_fsync']['r+b']['success'] is True and diagnosis['bytes_unchanged'] is True)
    failure_path=OUT/'deterministic_repair_qualification_v4/CodeBrain-lastblock-finetune/1791775861976395100/attempt.json'
    failure=read(failure_path)
    check('real_failed_qual4_six_four_48_zero_preserved',failure['status']=='failed' and failure['source']==desc(old) and failure['forward_returned']==6 and failure['backward_returned']==4 and failure['encoder_windows_returned']==48 and failure['optimizer_step_returned']==0 and failure['primary_engineering_seconds']==16.700772199998028)
    cleanup=read(A/'qualifier4_failure_cleanup_1791776170659.json')
    check('existing_failure_cleanup_records_empty_contexts_and_no_owner',cleanup['experimental_native_processes']==cleanup['gpu_contexts']==[] and cleanup['qualifier4_owner_present'] is False)

    # CPU10 continues exactly original CPU8, whose private CPU5 math/loss3 were previously reviewed.
    cpu=A/'audit_completed_finetune_checkpoints_v10.py'
    raw=cpu.read_text(encoding='utf-8');node=function(cpu,'load_delegate')
    check('CPU10_delegate_is_CPU8',"DELEGATE = SOURCE.with_name('audit_completed_finetune_checkpoints_v8.py')" in raw and "43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f" in raw)
    check('CPU10_actual6_not_relabelled_as5',"versions == {6}" in ast.unparse(node) and 'actual_observer6_sources_verified_before_counter_arithmetic=True' in raw and 'actual_observer5_' not in raw)
    check('CPU10_report_filename_and_schema10',"completed_finetune_checkpoint_audit_v10_" in raw and 'schema_version=10' in raw)

    # Cost11 old/new ledgers routed to their own sources; full cost collection is never called.
    cost=A/'cost_aggregation_v11.py';cost_tree=parsed(cost)
    insert_index=next(i for i,n in enumerate(cost_tree.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='sys.path.insert')
    previous_import_index=next(i for i,n in enumerate(cost_tree.body) if isinstance(n,ast.ImportFrom) and any(a.asname=='PREVIOUS_Q' for a in n.names))
    check('Cost11_script_import_after_workspace_path',insert_index<previous_import_index)
    default=subprocess.run([str(W/'S00/.venv/Scripts/python.exe'),'-I','-B',str(cost)],cwd=W,capture_output=True,text=True,check=True)
    default_value=json.loads(default.stdout.strip().splitlines()[-1])
    check('Cost11_isolated_default_CLI_zero_operation',default_value['status']=='preparation_only' and default_value['actual_cost_collection_started'] is False and default_value['model_calls']==default_value['GPU_calls']==default_value['new_output_writes']==0)
    sources={'old':desc(old),'new':desc(new)}
    ns={'require':require,'copy':copy,'Q':types.SimpleNamespace(SOURCE=new,CONDITIONS=('CodeBrain-lastblock-finetune','EEGMamba-lastblock-finetune')),
        'PREVIOUS_Q':types.SimpleNamespace(SOURCE=old),'S':types.SimpleNamespace(desc=desc),'COUNTER_CATEGORY':'qualification','NESTED_CATEGORY':'nested'}
    functions(cost,{'repair_worker_record','category_totals'},ns)
    def normalize(value,**kwargs):
        return {'observed_counters':{k:value.get(k+'_returned') for k in ('forward','backward','optimizer_step')},'unknown_counter_fields':[]}
    old_row=ns['repair_worker_record'](desc(failure_path),failure,normalize)
    future=copy.deepcopy(failure);future.update(source=sources['new'],status='engineering_verified',primary_engineering_seconds=8.0)
    new_artifact={'path':(OUT/'deterministic_repair_qualification_v5/CodeBrain-lastblock-finetune/fixture/attempt.json').relative_to(W).as_posix(),'sha256':'fixture'}
    new_row=ns['repair_worker_record'](new_artifact,future,normalize)
    check('Cost11_old_failure_real_source_and_elapsed_retained',old_row['status']=='failed' and old_row['seconds_observed']==failure['primary_engineering_seconds'] and old_row['aggregation_role']=='primary_observed_invocation')
    reject('Cost11_reject_old_failure_relabelled_new_source',lambda:ns['repair_worker_record'](desc(failure_path),future,normalize))
    nested={'aggregation_role':'supplemental_nested_or_unmeasured','observed_counters':{'forward':6,'backward':4,'optimizer_step':0},'seconds_observed':1000.0}
    total=ns['category_totals']([old_row,new_row,nested])
    check('Cost11_two_primary_once_nested_not_added',total['primary_invocations']==2 and total['known_recorded_scope_seconds_subtotal']==failure['primary_engineering_seconds']+8.0 and total['counter_totals']['backward']['known_observed_subtotal']==8 and total['nested_driver_times_added'] is False)
    check('Cost11_exact_active_elapsed_still_unknown',total['full_category_wall_or_gpu_busy_time'] is None and total['counter_totals']['backward']['exact_category_total'] is None)
    calls=[]
    base=types.SimpleNamespace(buffered_observation=lambda *a:None,collect=lambda:(calls.append('inventory') or {'rules':[]}),load_v4=lambda:types.SimpleNamespace(normalized='normalizer'))
    previous=types.SimpleNamespace(load_v5=lambda:base,augment=lambda result,**kw:(calls.append('three_loss_augment') or result))
    ns={'delegate':lambda:previous,'buffered_observation':lambda *a:None,'add_repair_ledgers':lambda result,n:(calls.append('repair_ledgers') or result),
        'L':types.SimpleNamespace(verify_lost_f3_attempt='verify3',read_json='read3',__file__='closure3.py'),
        'S':types.SimpleNamespace(desc=lambda p:{'path':str(p),'sha256':'fixture'}),'Path':Path,'SOURCE':'cost11.py','DELEGATE':'cost9.py','V':types.SimpleNamespace(SOURCE='observer3.py')}
    functions(cost,{'collect'},ns);result=ns['collect']()
    check('Cost11_original_inventory_and_Cost9_augment_each_once',calls==['inventory','three_loss_augment','repair_ledgers'] and result['schema_version']==11)

    sys.path.insert(0,str(W))
    from tridep_ccml.eeg_all_candidates import continuation_queue_v8 as Q8,repair_observation_v3 as V,deterministic_repair_execution_v3 as D
    V.source_check();records=Q8.sources()
    prior_path=A/'independent_recovery3_repair_chain_sources_review_1791771797426373100.json';prior=read(prior_path)
    check('all115_unique_exact104plus_prior_review_and_new10',len(records)==115 and len({x['path'] for x in records})==115 and all(x in records for x in prior['reviewed_sources']) and desc(prior_path) in records)
    for item in records:checked(item)
    check('all115_current_hashes_verified',True)
    check('D3_qualifier5_pin_matches',D.QUALIFIER_SHA==desc(new)['sha256'])
    check('all_source_review_imports_remain_stdlib',not any(x in sys.modules for x in ('torch','numpy')))
    check('original104_independent_review_immutable',desc(prior_path)['sha256']=='9f4c37179767a8e8dfd51060968094b6cf12e9a0b542cdb0d2e33c5e47ea17b6')
    # The actual preparation report must bind exactly this current execution source set.
    prepared_records=preparation.get('candidate_execution_sources',preparation.get('sources',preparation.get('reviewed_sources')))
    check('fresh_preparation_binds_same115',prepared_records==records)
    auxiliary=[desc(A/'repair_postprocessing_gate_v3.py'),desc(A/'review_actual_repair_engineering_v3.py')]
    result={'schema':'independent_qualifier_fsync_repair_chain_source_review_v1','status':'new_repair_chain_sources_reviewed','passed':True,
       'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'reviewed_sources':records,'auxiliary_sources':auxiliary,
       'checks':checks,'preparation_contracts':desc(preparation_path),'prior104_review':desc(prior_path),'actual_failure_attempt':desc(failure_path),
       'actual_storage_diagnosis':desc(A/'actual_windows_fsync_diagnosis_1791776209166909300.json'),'existing_cleanup_observation':desc(A/'qualifier4_failure_cleanup_1791776170659.json'),
       'review_scope':['Full qualifier AST unchanged except explicit module/output version labels and rb to r+b fsync handle; no numerical or role change.',
       'New D3/F6/Q8/O3/W3/E6/CPU10/cost11 source/versions reviewed; old104 source bytes remain unchanged and included.',
       'Control comparisons include retained exact Windows/Linux birth/parent/nonce, exclusive claims, per-boundary live validation and original360/CPU135405 gates.',
       'CPU10 delegates original CPU8 checkpoint math/loss3; cost11 delegates original cost9 augmentation exactly once and retains actual failed Qual4 plus new Qual5 primary ledgers once.',
       'No actual qualifier5/report, original30 fitting, actual CPU10 full gate, E6 evaluation, resource supplementation or final cost/semantic/acceptance is claimed.'],
       'numeric_backward_plan_changed_since_qualifier4':False,'new_numeric_GPU_equivalence_established':False,'performance_improvement_claimed':False,
       'model_calls':0,'GPU_queries':0,'actual_process_inventory_queries':0,'process_controls':0,'checkpoint_tensor_loads':0,'official_weight_or_checkpoint_byte_reads':0,'outer_test_probability_reads':0,
       'default_only_CLI_invocations':1,'source_byte_hashing_performed':True,'new_registration_created':False,'actual_GPU_qualification_complete':False,'actual_new_formal_fits_completed':0,'all_plan_complete':False,
       'remaining_limitations':['Mamba backward numeric accumulation order differs; original CUDA backward equivalence not established.','EEGMamba external-pretraining participant overlap exclusion remains unknown.','Final resource/cost/wide/companion delivery sources and actual certificates remain required.'],
       'captured_ns':time.time_ns()}
    require(records==Q8.sources(),'Candidate changed during independent review')
    target=A/('independent_qualifier_fsync_repair_chain_source_review_'+str(result['captured_ns'])+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'report':desc(target),'passed':True,'reviewed_sources':len(records),'checks':len(checks)}))
if __name__=='__main__':main()
