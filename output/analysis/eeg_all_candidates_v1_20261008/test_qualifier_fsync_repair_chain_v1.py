"""Pure fixture tests for reviewed control blockers; no live inventory/model."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types

W=Path('C:/Users/jin  qiyan/Desktop/lunwen2');M=W/'tridep_ccml/eeg_all_candidates';A=W/'output/analysis/eeg_all_candidates_v1_20261008'
checks={}
def require(ok,msg):
    if not ok:raise ValueError(msg)
def check(label,value):require(value,label);checks[label]=True
def reject(label,fn):
    try:fn()
    except (ValueError,KeyError,IndexError,FileExistsError):checks[label]=True
    else:raise AssertionError('Unsafe acceptance: '+label)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def desc(p):return {'path':p.relative_to(W).as_posix(),'sha256':sha(p)}
def parsed(p):return ast.parse(p.read_text(encoding='utf-8'))
def function(p,name):return next(n for n in parsed(p).body if isinstance(n,ast.FunctionDef) and n.name==name)
def load_functions(p,names,ns):
    nodes=[copy.deepcopy(function(p,n)) for n in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns
lp=M/'linux_worker_identity_v1.py'
spec=importlib.util.spec_from_file_location('_pure_linux_control_verifier',lp);L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
module='tridep_ccml.eeg_all_candidates.repair_owned_worker_v3'
flags={'--token':'nonce','--condition':'EEGMamba-lastblock-finetune','--worker-admission-sha256':'a'*64,'--repair-registration-sha256':'r'*64}
argv=['/mnt/c/experiment/env/bin/python','-B','-u','-m',module,'--run']
for k,v in flags.items():argv.extend([k,v])
row={'pid':777,'parent':4,'start_ticks':'1234','argv':argv,'executable':'/usr/bin/python3.11',
     'parent_identity':{'pid':4,'parent':1,'start_ticks':'1200','argv':['/init'],'executable':'/init'}}
snapshot={'query_succeeded':True,'errors':[],'boot_id':'boot-real','rows':[row]}
binding=L.select_unique(snapshot,module=module,flags=flags)
check('Linux_exact_pid_parent_birth_argv_boot_positive',L.validate_bound(snapshot,binding,module=module,flags=flags,worker_pid=777)==binding)
reject('Linux_wrong_caller_pid',lambda:L.validate_bound(snapshot,binding,module=module,flags=flags,worker_pid=888))
for label,mutate in [
    ('Linux_duplicate_process',lambda v:v['rows'].append({**copy.deepcopy(row),'pid':888})),
    ('Linux_query_errors',lambda v:v.update(errors=[{'pid':5,'error_type':'unknown'}])),
    ('Linux_missing_query',lambda v:v.update(query_succeeded=False)),
    ('Linux_new_boot',lambda v:v.update(boot_id='new-boot')),
    ('Linux_changed_start_ticks',lambda v:v['rows'][0].update(start_ticks='2345')),
    ('Linux_changed_parent_birth',lambda v:v['rows'][0]['parent_identity'].update(start_ticks='2222')),
    ('Linux_changed_executable',lambda v:v['rows'][0].update(executable='/tmp/imposter')),
    ('Linux_changed_nonce',lambda v:v['rows'][0]['argv'].__setitem__(v['rows'][0]['argv'].index('--token')+1,'wrong')),
    ('Linux_wrong_module',lambda v:v['rows'][0]['argv'].__setitem__(v['rows'][0]['argv'].index('-m')+1,'wrong.module')),
    ('Linux_duplicate_nonce_flag',lambda v:v['rows'][0]['argv'].extend(['--token','nonce'])),
    ('Linux_parent_alias',lambda v:v['rows'][0]['parent_identity'].update(pid=9)),
]:
    bad=copy.deepcopy(snapshot);mutate(bad)
    reject(label,lambda bad=bad:L.validate_bound(bad,binding,module=module,flags=flags,worker_pid=777))

q=M/'qualify_deterministic_repairs_v5.py'
host_ns=load_functions(q,['admit_host'],{'require':require})
controller={'pid':1,'parent':0,'birth':'100','command':'dispatcher'}
launch={'pid':2,'parent':1,'birth':'200','command':'wsl.exe -m '+module+' --token nonce'}
items=[controller,launch]
check('Native_all_captured_roles_present',host_ns['admit_host'](items,items) is None)
reject('Native_missing_controller',lambda:host_ns['admit_host']([launch],items))
reject('Native_controller_birth_change',lambda:host_ns['admit_host']([{**controller,'birth':'101'},launch],items))
reject('Native_unexpected_extra_process',lambda:host_ns['admit_host'](items+[{'pid':3,'parent':0,'birth':'300','command':'other'}],items))
admission={'queue_owner_id':'owner','token':'nonce','job':'job','condition':flags['--condition'],'repair_registration':{'path':'repair','sha256':'r'*64}}
owner={'id':'owner','execution_version':8,'child_token':'nonce','child_job':'job','native_roles':[controller],
       'child_pid':2,'pid':1,'child_native_identity':launch,'linux_worker_identity':binding}
o=M/'repair_runtime_owner_v3.py'
native_ns=load_functions(o,['validate_native'],{'require':require,'L':L,'WORKER_MODULE':module})
kwargs={'worker_pid':777,'platform':'linux','linux_snapshot':snapshot,'admission_binding':{'sha256':'a'*64}}
check('FormalLinux_matches_controller_capture',native_ns['validate_native'](items,owner,admission,**kwargs)==items)
reject('FormalLinux_wrong_pid',lambda:native_ns['validate_native'](items,owner,admission,**{**kwargs,'worker_pid':888}))
reject('FormalLinux_missing_capture',lambda:native_ns['validate_native'](items,{**owner,'linux_worker_identity':None},admission,**kwargs))
reject('FormalLinux_missing_controller',lambda:native_ns['validate_native']([launch],owner,admission,**kwargs))
bad=copy.deepcopy(snapshot);bad['rows'][0]['start_ticks']='999'
reject('FormalLinux_changed_birth',lambda:native_ns['validate_native'](items,owner,admission,**{**kwargs,'linux_snapshot':bad}))

with tempfile.TemporaryDirectory(prefix='eeg_atomic_worker_claim_') as temp:
    root=Path(temp);folder=root/'attempts/fake';folder.mkdir(parents=True);path=folder/'worker_owner.json'
    ns=load_functions(o,['claim_worker_owner'],{'ROOT':root,'require':require,'json':json,'os':os})
    gate=threading.Barrier(2)
    def contender(pid):
        gate.wait()
        try:ns['claim_worker_owner'](path,{'pid':pid});return ('won',pid)
        except FileExistsError:return ('rejected',pid)
    with ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(contender,[777,888]))
    check('Atomic_two_concurrent_claims_exactly_one_winner',sum(x[0]=='won' for x in result)==1 and sum(x[0]=='rejected' for x in result)==1)
    check('Atomic_winner_identity_not_overwritten',json.loads(path.read_text())['pid']==next(x[1] for x in result if x[0]=='won'))
    reject('Atomic_wrong_path_rejected',lambda:ns['claim_worker_owner'](root/'bad.json',{}))

# E5 uses the same independently captured Linux identity and per-fit recheck.
e=M/'evaluate_v6.py';eval_module='tridep_ccml.eeg_all_candidates.evaluate_v6'
eval_flags={'--token':'nonce','--mode':'mamba_linux','--admission-sha256':'a'*64,'--cpu-gate-sha256':'g'*64}
eval_row=copy.deepcopy(row);eval_row['argv']=['python','-m',eval_module]
for k,v in eval_flags.items():eval_row['argv'].extend([k,v])
eval_snapshot={**snapshot,'rows':[eval_row]};eval_binding=L.select_unique(eval_snapshot,module=eval_module,flags=eval_flags)
eval_launch={**launch,'command':'wsl.exe -m '+eval_module+' --worker --token nonce'}
eval_owner={**owner,'source':{'path':'E5','sha256':'E5'},'child_identity':eval_launch,
            'child_mode':'mamba_linux','child_admission':{'sha256':'a'*64},'CPU_gate':{'sha256':'g'*64},'linux_worker_identity':eval_binding}
eval_ns=load_functions(e,['admit_worker'],{'require':require,'S':types.SimpleNamespace(desc=lambda p:eval_owner['source']),
       'SOURCE':e,'Q':types.SimpleNamespace(admit_host=host_ns['admit_host']),'MODULE':eval_module,
       'L':types.SimpleNamespace(snapshot=lambda:eval_snapshot,validate_bound=L.validate_bound)})
check('EvaluatorLinux_matches_controller_capture',eval_ns['admit_worker']([controller,eval_launch],eval_owner,'nonce',777,'linux')==[controller,eval_launch])
reject('EvaluatorLinux_wrong_pid',lambda:eval_ns['admit_worker']([controller,eval_launch],eval_owner,'nonce',888,'linux'))
reject('EvaluatorLinux_missing_controller',lambda:eval_ns['admit_worker']([eval_launch],eval_owner,'nonce',777,'linux'))
reject('EvaluatorLinux_missing_nonce_capture',lambda:eval_ns['admit_worker']([controller,eval_launch],{**eval_owner,'linux_worker_identity':None},'nonce',777,'linux'))

# Boundary fixtures now require source5/base, the exact completed child exit0,
# immutable completion events and the recovery3 linkage, before GPU operations.
with tempfile.TemporaryDirectory(prefix='eeg_natural_boundary_') as temp:
    tw=Path(temp);out=tw/'out';queue=out/'continuation_queue_v5';queue.mkdir(parents=True)
    special={'registration.json':{'path':'base','sha256':'base'},'continuation_queue_v5.py':{'path':'Q5','sha256':'Q5'},
             'recover_q5_after_process_loss_v3.py':{'path':'entry3','sha256':'entry3'},
             'recovery_registration_v3.json':{'path':'manifest3','sha256':'653df2183b1a9cdcd74755f69629d2cf89ca3de8a6b7e9bf26b7df0bfcabbaf4'},
             'lost_f3_v3_1791767849813215500.json':{'path':'loss3','sha256':'loss3'}}
    target='LaBraM-native4s-lastblock-finetune-fit';job={'id':target,'fits':15}
    matrix={'path':'matrix','sha256':'matrix'}
    rows=[{'fit_id':'normal/seed_11/fold_'+str(i),'condition':'normal'} for i in range(330)]
    status={'status':'required_work_remaining','execution_version':5,'source':special['continuation_queue_v5.py'],
       'execution_registration':special['registration.json'],'recovery_execution_version':3,
       'recovery_registration':special['recovery_registration_v3.json'],'recovery_execution_source':special['recover_q5_after_process_loss_v3.py'],
       'recovery_closure':special['lost_f3_v3_1791767849813215500.json'],'outcomes':{target:'complete'},'actual_resumed_job_ids':[target],'other_jobs_executed_this_recovery':0}
    attempt={**status,'job':job,'status':'complete','returncode':0,'registration':special['registration.json'],'log':{'path':'log','sha256':'log'}}
    data={}
    def fake_desc(p):return matrix if p==out/'registration.json' else special[p.name]
    def fake_read(p):
        if p==queue/'status.json':return data['status']
        if p==queue/'registration.json':return {'jobs':[job]}
        if p==out/'registration.json':return {'fits':rows}
        if isinstance(p,Path) and p.parent.name==target+'FAKE':raise AssertionError('Wrong attempt')
        if isinstance(p,Path) and p.parent.name=='1791767857117600800-'+target:return data['attempt']
        if p=='loss3':return {'recovery_registration':data['status']['recovery_registration'],'closure_source':data['status']['recovery_execution_source']}
        if isinstance(p,str) and p.startswith('event:'):
            fit=p[6:];return {k:v for k,v in fit_status(fit).items() if k!='event_record'}
        if isinstance(p,Path) and p.name=='status.json':return fit_status(p.parent.relative_to(out/'runs').as_posix())
        raise AssertionError('Unexpected fixture read '+str(p))
    def fit_status(fit):return data.get('bad_fit',{}).get(fit,{'fit_id':fit,'status':'training_complete','registration':matrix,'evidence':{'passed':True},'event_record':{'path':'event:'+fit,'sha256':'event'}})
    ns=load_functions(q,['check_boundary'],{'QUEUE':queue,'OUT':out,'WORK':tw,'CONDITIONS':('repair1','repair2'),
                    'MANIFEST_SHA':special['recovery_registration_v3.json']['sha256'],'require':require,'read':fake_read,
                    'desc':fake_desc,'checked':lambda x:x['path']})
    def reset():data.clear();data.update(status=copy.deepcopy(status),attempt=copy.deepcopy(attempt))
    reset();check('Natural_bound_Q5_exit0_330_events_positive',ns['check_boundary']()==status)
    for label,mutate in [
      ('Natural_failed_parent_rejected',lambda:data['status'].update(status='failed')),
      ('Natural_wrong_Q3_source_rejected',lambda:data['status'].update(source={'path':'Q3','sha256':'Q3'})),
      ('Natural_wrong_base_registration_rejected',lambda:data['status'].update(execution_registration=matrix)),
      ('Natural_empty_outcomes_rejected',lambda:data['status'].update(outcomes={})),
      ('Natural_nonzero_child_exit_rejected',lambda:data['attempt'].update(returncode=2)),
      ('Natural_boolean_exit_rejected',lambda:data['attempt'].update(returncode=False)),
      ('Natural_running_child_rejected',lambda:data['attempt'].update(status='running')),
      ('Natural_old_child_recovery_rejected',lambda:data['attempt'].update(recovery_execution_version=2)),
      ('Natural_wrong_child_job_rejected',lambda:data['attempt'].update(job={'id':'wrong','fits':15})),
      ('Natural_missing_complete_evidence_rejected',lambda:data.update(bad_fit={rows[-1]['fit_id']:{'fit_id':rows[-1]['fit_id'],'status':'training_complete','registration':matrix,'evidence':{}}})),
    ]:
        reset();mutate();reject(label,lambda:ns['check_boundary']())
    reset();(queue/'owner.json').write_text('{}');reject('Natural_still_owned_rejected',lambda:ns['check_boundary']())

check('Qualifier_module_main_guard_after_all_definitions', isinstance(parsed(q).body[-1],ast.If) and '__name__' in ast.unparse(parsed(q).body[-1].test))
check('Full330_byte_complete_verification_in_admission',any(isinstance(n,ast.Call) and ast.unparse(n.func)=='R._validate_complete' for n in ast.walk(function(q,'verify_original330_complete'))))
safe=ast.unparse(function(q,'safe_before_model'))
check('Full330_verification_after_empty_GPU_no_repeated_hashing_on_graphic_deferral',safe.index('gpu = gpu_empty()')<safe.index('completion = verify_original330_complete()'))
check('Full330_immutable_rows_in_real_admission',"'original330_complete_proof': completion" in safe)
for name,func in [('continuation_queue_v8.py','child'),('evaluate_v6.py','controller'),('qualify_deterministic_repairs_v5.py','run_controller')]:
    raw=ast.unparse(function(M/name,func));check('Controller_capture_before_wait:'+name,'L.wait_capture' in raw and raw.index('L.wait_capture')<raw.index('.wait()'))
for name,func in [('repair_runtime_owner_v3.py','require_live'),('evaluate_v6.py','worker'),('qualify_deterministic_repairs_v5.py','runtime_worker')]:
    raw=ast.unparse(function(M/name,func));check('Linux_identity_revalidated_at_owned_boundaries:'+name,'validate_native' in raw or 'admit_worker' in raw or 'L.validate_bound' in raw)
f=(M/'finetune_v6.py').read_text();cpu=(A/'audit_completed_finetune_checkpoints_v10.py').read_text()
check('F5_output_actual_version5',"'actual_observer_version': 6" in f)
check('CPU9_report_filename_version9','completed_finetune_checkpoint_audit_v10_' in cpu and 'completed_finetune_checkpoint_audit_v7_' not in cpu)
check('E5_output_repair5_label','old_and_repair6_training_provenance_verified' in e.read_text())
sys.path.insert(0,str(W))
from tridep_ccml.eeg_all_candidates import continuation_queue_v8 as Q7,repair_observation_v3 as Obs
Obs.source_check();records=Q7.sources();prior=json.loads((W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v3.json').read_text(encoding='utf-8'))['sources']
check('All88_source_descriptors_still_inherited',len(prior)==88 and all(r in records for r in prior))
check('All_candidate_source_records_unique',len(records)==len({r['path'] for r in records}))
check('Shared_Linux_identity_source_bound',desc(lp) in records)
for r in records:check('Current_source_SHA:'+r['path'],desc(W/r['path'])==r)

class NumericNormalize(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            for old,new in sorted(mapping.items(),key=lambda kv:-len(kv[1])):node.value=node.value.replace(new,old)
            if node.value=='r+b':node.value='rb'
        return node
mapping={'qualify_deterministic_repairs_v4': 'qualify_deterministic_repairs_v5', 'deterministic_repair_execution_v2': 'deterministic_repair_execution_v3', 'finetune_v5': 'finetune_v6', 'repair_observation_v2': 'repair_observation_v3', 'repair_runtime_owner_v2': 'repair_runtime_owner_v3', 'repair_owned_worker_v2': 'repair_owned_worker_v3', 'continuation_queue_v7': 'continuation_queue_v8', 'evaluate_v5': 'evaluate_v6', 'evaluation_v5_control': 'evaluation_v6_control', 'deterministic_repair_qualification_v4': 'deterministic_repair_qualification_v5', 'audit_completed_finetune_checkpoints_v9': 'audit_completed_finetune_checkpoints_v10', 'cost_aggregation_v10': 'cost_aggregation_v11', 'repair_postprocessing_gate_v2': 'repair_postprocessing_gate_v3', 'review_actual_repair_engineering_v2': 'review_actual_repair_engineering_v3'}
for name in ('isolated_builder','runtime_worker'):
    oldfn=function(M/'qualify_deterministic_repairs_v4.py',name)
    newfn=NumericNormalize().visit(copy.deepcopy(function(M/'qualify_deterministic_repairs_v5.py',name)))
    check('Numeric_qualification_unchanged_except_fsync:'+name,ast.dump(oldfn,include_attributes=False)==ast.dump(newfn,include_attributes=False))
prior104=json.loads((A/'independent_recovery3_repair_chain_sources_review_1791771797426373100.json').read_text(encoding='utf-8'))['reviewed_sources']
check('All104_previous_descriptors_inherited',len(prior104)==104 and all(row in records for row in prior104))
cp=A/'cost_aggregation_v11.py';ct=parsed(cp)
previous_import=next(i for i,n in enumerate(ct.body) if isinstance(n,ast.ImportFrom) and any(x.asname=='PREVIOUS_Q' for x in n.names))
path_setup=next(i for i,n in enumerate(ct.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='sys.path.insert')
check('Cost11_package_path_before_previous_qualifier_import',path_setup<previous_import)
collector=function(cp,'collect');calls=[ast.unparse(n.func) for n in ast.walk(collector) if isinstance(n,ast.Call)]
check('Cost9_augment_exactly_once',calls.count('previous.augment')==1 and 'previous.collect' not in calls and 'old.collect' in calls)
patterns=next(n.value for n in function(cp,'add_repair_ledgers').body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='patterns' for t in n.targets))
pat=ast.literal_eval(patterns)
check('Both_failed4_and_new5_qualification_invocations_inventory',len(pat)==len(set(pat)) and 'deterministic_repair_qualification_v4/*/*/attempt.json' in pat and 'deterministic_repair_qualification_v5/*/*/attempt.json' in pat)
actual_failure=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/deterministic_repair_qualification_v4/CodeBrain-lastblock-finetune/1791775861976395100/attempt.json'
failed=json.loads(actual_failure.read_text(encoding='utf-8'))
check('Actual_old_failure_retains_6_4_48_0_and_seconds',failed['status']=='failed' and failed['forward_returned']==6 and failed['backward_returned']==4 and failed['encoder_windows_returned']==48 and failed['optimizer_step_returned']==0 and failed['primary_engineering_seconds']==16.700772199998028)
check('No_new_model_work_inherited_from_old_failure',failed['formal_fits']==0)

import time
report={'schema':'qualifier_fsync_repair_chain_contract_tests_v1','passed':True,'source':desc(Path(__file__).resolve()),
        'checks':checks,'candidate_execution_sources':records,'candidate_source_count':len(records),
        'captured_ns':time.time_ns(),'model_calls':0,'GPU_queries':0,'actual_process_inventory_queries':0,
        'checkpoint_tensor_loads':0,'outer_probability_reads':0,'new_registration_created':False,
        'actual_GPU_qualification_complete':False,'independent_rereview_passed':False,'all_plan_complete':False,
        'source_byte_hashing_performed':True}
target=A/('qualifier_fsync_repair_chain_contract_tests_'+str(report['captured_ns'])+'.json')
with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'artifact':desc(target),'checks':len(checks),'candidate_sources':len(records),'model_calls':0,'GPU_queries':0}))
