"""CodeBrain-only source/registration/control contracts; stdlib/fake only."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,copy,hashlib,json,time
A=Path(__file__).resolve().parent;WORK=A.parents[2];M=WORK/'tridep_ccml/eeg_all_candidates';OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
checks=[]
def require(ok,message):
 if not ok:raise ValueError(message)
def eq(a,b):assert a==b,(a,b)
def check(name,fn):fn();checks.append(name)
def reject(fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):return
 raise AssertionError('Unexpected admission')
def desc(p):return {'path':Path(p).relative_to(WORK).as_posix(),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def checked(x):p=WORK/x['path'];eq(desc(p),x);return p
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def funcs(p,names,env=None):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 eq(set(n.name for n in nodes),set(names));space={'require':require,'Path':Path,**(env or {})}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p)+':AST-only','exec'),space);return space
def value(p,name):return ast.literal_eval(next(n.value for n in ast.parse(p.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
def source(p,name):
 raw=p.read_text();node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name);return ast.get_source_segment(raw,node)
CB='CodeBrain-lastblock-finetune';MA='EEGMamba-lastblock-finetune';conditions=(CB,MA)
prep=read(sorted(A.glob('codebrain_only_source_preparation_*.json'))[-1]);prior=read(checked(prep['previous166_review']))
S=NS(WORK=WORK,OUT=OUT,REPORT=A,desc=desc,checked=checked,read=read,digest=lambda p:desc(p)['sha256'])
check('old166_exact_bytes',lambda:[checked(x) for x in prior['reviewed_sources']])
new=[desc(WORK/x['path']) for x in prep['new_sources']]
check('new8_compile_only',lambda:[compile((WORK/x['path']).read_text(),x['path'],'exec') for x in new])
qfile=M/'qualify_deterministic_repairs_v9.py';pinenv={'REL':'tridep_ccml/eeg_all_candidates/'}
for n in ast.parse(qfile.read_text()).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='PINS') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='PINS') for t in n.targets):exec(compile(ast.Module(body=[n],type_ignores=[]),'pins','exec'),pinenv)
Dfile=M/'deterministic_repair_execution_v8.py';Qfile=M/'continuation_queue_v13.py'
D=funcs(Dfile,['implementation_sources','qualification','load_execution'],{'S':S,'SOURCE':Dfile,'read_bound':lambda b:read(checked(b)),'NEW_ENTRY_NAMES':value(Dfile,'NEW_ENTRY_NAMES'),'NEW_ANALYSIS_NAMES':value(Dfile,'NEW_ANALYSIS_NAMES'),'Q':NS(__file__=str(qfile),PINS=pinenv['PINS']),'F3_SHA':value(Dfile,'F3_SHA'),'REPAIR_CONDITIONS':conditions})
Q=funcs(Qfile,['sources'],{'S':S,'SOURCE':Qfile,'D':NS(implementation_sources=D['implementation_sources'])})
chain=Q['sources']()
check('exact175_unique_source_chain',lambda:eq((len(chain),len({x['path'] for x in chain})),(175,175)))
check('all_original166_inherit',lambda:require(all(x in chain for x in prior['reviewed_sources']),'Lost historical source'))
check('only_two_execution_entries_no_evaluator',lambda:eq(value(Dfile,'NEW_ENTRY_NAMES'),('finetune_v11.py','continuation_queue_v13.py')))
check('qualifier9_old_source_pin_preserved',lambda:eq(value(Dfile,'QUALIFIER_SHA'),desc(qfile)['sha256']))
check('observer_F11_pin',lambda:eq(value(M/'repair_observation_v8.py','F4_SHA'),desc(M/'finetune_v11.py')['sha256']))
check('observer_D8_pin',lambda:eq(value(M/'repair_observation_v8.py','D_SHA'),desc(Dfile)['sha256']))
check('D8_reject_Mamba_qualification',lambda:reject(lambda:D['qualification']({},MA)))
check('D8_reject_Mamba_load',lambda:reject(lambda:D['load_execution']({},condition=MA,seed=11,fold=1)))
F=funcs(M/'finetune_v11.py',['run'],{'D':NS(REPAIR_CONDITIONS=conditions)})
check('F11_reject_Mamba_run',lambda:reject(lambda:F['run'](MA,repair_registration={})))
O=funcs(M/'repair_runtime_owner_v8.py',['enter'],{'sys':NS(platform='win32')})
check('O8_reject_Mamba_before_admission',lambda:reject(lambda:O['enter']({},'nonce',condition=MA,repair_registration={})))
W=funcs(M/'repair_owned_worker_v8.py',['run'],{'sys':NS(platform='win32')})
check('W8_reject_Mamba_before_admission',lambda:reject(lambda:W['run'](condition=MA,repair_registration={},admission={},token='nonce')))
check('Q13_conditions_exact_CB',lambda:eq(value(Qfile,'CONDITIONS'),(CB,)))
nodes=ast.parse(Qfile.read_text()).body
check('Q13_no_test_gate_function',lambda:require(not any(isinstance(n,ast.FunctionDef) and n.name=='cpu_gate' for n in nodes),'Full gate exists'))
check('Q13_no_evaluator_import',lambda:require(not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any('evaluate' in a.name for a in n.names) for n in nodes),'Evaluation imported'))
check('Q13_cli_has_no_Mamba_report_option',lambda:require('mamba-report' not in source(Qfile,'main'),'Mamba report requested'))
check('Q13_terminal_explicit_partial',lambda:require("codebrain15_complete_mamba15_blocked_no_evaluation" in source(Qfile,'run') and 'all_plan_complete=False' in source(Qfile,'run') and 'full360_CPU_gate_passed=False' in source(Qfile,'run'),'Misleading full terminal'))
check('Q13_partial_audit120_360_not_testgate',lambda:require("== 120" in source(Qfile,'cpu_partial_audit') and "== 360" in source(Qfile,'cpu_partial_audit') and "== 345" in source(Qfile,'cpu_partial_audit') and "is False" in source(Qfile,'cpu_partial_audit'),'Partial audit scope changed'))
# Fake registration exercises the actual function without any registration writes.
matrix=read(OUT/'registration.json');ids=[x['fit_id'] for x in matrix['fits'] if x['condition']==CB];eq(len(ids),15)
blocked=[{'path':'mamba'+str(i),'sha256':str(i)} for i in range(15)]
calls=[];writes=[]
proof={'original_fit_ids':ids,'contract':{'accumulation_order_changed':False}}
source_review={'status':'new_repair_chain_sources_reviewed','reviewed_sources':chain,'passed':True,'reviewer_source':{}}
fakeS=NS(OUT=OUT,read=lambda p:{'status':'blocked'},desc=lambda p:{'path':str(p),'sha256':'fake'},checked=lambda p:p,now=lambda:'fake')
fakeD=NS(REPAIR_CONDITIONS=conditions,read_bound=lambda b:source_review,qualification=lambda binding,condition:(calls.append(('qualification',condition)) or proof),ROOT=OUT/'deterministic_repair_execution_v8',SCHEMA=value(Dfile,'SCHEMA'),implementation_sources=lambda:chain,lock_contract=lambda p,b:{'source':'newF11','binding':b})
fakeR=NS(registration_descriptor=lambda:{'registry':'original360'},lock_condition=lambda c,l:calls.append(('lock',c)),record_fit_status=lambda c,s,f,status:calls.append(('ready',c,s,f,status)))
def create(path,val):writes.append((str(path),copy.deepcopy(val)));return {'path':str(path),'sha256':'fake'}
reg=funcs(Qfile,['register'],{'Q':NS(check_pause=lambda:None,check_boundary=lambda:None,QUEUE=OUT/'continuation_queue_v5'),'D':fakeD,'S':fakeS,'R':fakeR,'CONDITIONS':(CB,),'SOURCE':Qfile,'ROOT':OUT/'continuation_queue_v13','SCHEMA':value(Qfile,'SCHEMA'),'original_others_complete':lambda:['old'+str(i) for i in range(330)],'pending_mamba':lambda:blocked,'sources':lambda:chain,'review_engineering':lambda b,p:None,'create':create,'time':NS(time_ns=lambda:1)})
check('register_reject_both_groups',lambda:reject(lambda:reg['register']({CB:{},MA:{}},{CB:{},MA:{}},{})))
check('reject_before_any_mutation',lambda:eq((calls,writes),([],[])))
reg['register']({CB:{}},{CB:{}},{})
check('register_only_CB_qualification_and_lock',lambda:eq([x for x in calls if x[0]!='ready'],[('qualification',CB),('lock',CB)]))
check('register_exact15_original_ready_transitions',lambda:eq([x for x in calls if x[0]=='ready'],[('ready',CB,s,f,'ready') for s in (11,22,33) for f in (1,2,3,4,5)]))
registration=writes[-1][1]
check('partial_registration_actual_identity15',lambda:eq((registration['formal_fits_to_run'],registration['original_fit_ids']),(15,ids)))
check('partial_registration_Mamba_never_qualified',lambda:eq((registration['mamba_qualification_passed'],registration['pending_mamba_statuses']),(False,blocked)))
check('partial_registration_no_evaluation_or_full_gate',lambda:eq((registration['evaluation_permanently_disabled'],registration['full360_CPU_gate_passed'],registration['all360_completion_claimed']),(True,False,False)))
check('partial_registration_own_version13',lambda:eq((registration['version'],registration['schema']),(13,'eeg_deterministic_repair_queue_v13')))
sidecar=next(v for p,v in writes if '/deterministic_repair_execution_v8/' in p.replace('\\','/'))
check('sidecar_own_D8_observer11',lambda:eq((sidecar['version'],sidecar['new_actual_observer_version'],sidecar['condition']),(8,11,CB)))
check('CPU15_filename',lambda:require('completed_finetune_checkpoint_audit_v15_' in (A/'audit_completed_finetune_checkpoints_v15.py').read_text(),'Old CPU label'))
check('cost16_collects_Qual9_failed_ledger',lambda:require('deterministic_repair_qualification_v9/*/*/attempt.json' in source(A/'cost_aggregation_v16.py','add_repair_ledgers'),'Qual9 failure missing'))
check('F11_training_constants_unchanged',lambda:require('micro_subjects=1, window_chunk=8' in source(M/'finetune_v11.py','fit'),'Training constants changed'))
check('owner13_strict_version',lambda:require("get('execution_version') == 13" in source(M/'repair_runtime_owner_v8.py','validate_native'),'Old owner version'))
check('owner_atomic_claim_unchanged',lambda:require("path.open('x'" in source(M/'repair_runtime_owner_v8.py','claim_worker_owner'),'Nonatomic worker claim'))
check('Q13_single_CB_child_first_guard',lambda:require(source(Qfile,'child').splitlines()[1].strip().startswith('require(condition in CONDITIONS'),'Child can admit Mamba before guard'))
check('Q13_only_CB_outcomes_loop',lambda:require('for condition in CONDITIONS:' in source(Qfile,'run') and 'D.REPAIR_CONDITIONS' not in source(Qfile,'run'),'Run loops unqualified Mamba'))
report={'schema':'codebrain_only_partial_chain_stdlib_contract_v1','passed':True,'checks':checks,'check_count':len(checks),'new_sources':new,'source_chain':chain,'previous166_review':prep['previous166_review'],'model_operations':0,'GPU_operations':0,'Tensor_loads':0,'process_queries':0,'process_controls':0,'real_registration_writes':0,'fake_registration_only':True,'outer_test_reads':0,'actual_training_started':False,'source':desc(Path(__file__))}
target=A/('codebrain_only_partial_chain_contracts_'+str(time.time_ns())+'.json')
with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps({'artifact':desc(target),'checks':len(checks),'sources':len(chain),'passed':True}))
