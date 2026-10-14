"""Pure stdlib/AST contracts: no real process, GPU, model, Tensor or registration."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast, copy, hashlib, json, time
A = Path(__file__).resolve().parent
WORK = A.parents[2]
M = WORK / 'tridep_ccml/eeg_all_candidates'
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
RESULTS = []
def require(ok, message):
    if not ok: raise ValueError(message)
def check(name, fn):
    fn()
    RESULTS.append(name)
def eq(a,b):
    assert a == b, (a,b)
def reject(fn):
    try: fn()
    except (ValueError, KeyError, AssertionError): return
    raise AssertionError('Negative case was admitted')
def desc(path):
    path = Path(path)
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
def checked(item):
    path = WORK / item['path']
    eq(desc(path),item)
    return path
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def functions(path, names, env=None):
    nodes = [n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    eq({n.name for n in nodes},set(names))
    space = {'require':require, 'Path':Path, '__builtins__':__builtins__, **(env or {})}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+':AST-functions-only','exec'),space)
    return space
def fsource(path,name):
    raw = Path(path).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(raw,node)
N = NS(**functions(M/'native_launcher_tree_v1.py', ['capture','verify']))
qenv=functions(M/'qualify_deterministic_repairs_v8.py',['admit_host'])
admit=qenv['admit_host']
module='tridep_ccml.eeg_all_candidates.repair_owned_worker_v6'
def fixture(module=module, nonce='nonce'):
    controller={'pid':1,'parent':0,'birth':'100','command':'python controller'}
    command='wsl.exe -m '+module+' --worker --token '+nonce
    primary={'pid':2,'parent':1,'birth':'200','command':command}
    shim={'pid':3,'parent':2,'birth':'300','command':command}
    return controller,primary,shim
c,p,s=fixture()
kw=dict(launcher_pid=2,controller_pid=1,module=module,nonce='nonce')
check('helper_primary_only',lambda:eq(N.capture([c,p],**kw),[p]))
check('helper_realistic_primary_and_shim',lambda:eq(N.capture([c,p,s],**kw),[p,s]))
for field, val in [('parent',9),('birth','bad'),('command','unknown')]:
    check('helper_reject_primary_'+field,lambda field=field,val=val:reject(lambda:N.capture([c,{**p,field:val},s],**kw)))
for field,val in [('birth','bad'),('command','unknown')]:
    check('helper_reject_shim_'+field,lambda field=field,val=val:reject(lambda:N.capture([c,p,{**s,field:val}],**kw)))
check('helper_reject_multiple_shims',lambda:reject(lambda:N.capture([c,p,s,{**s,'pid':4}],**kw)))
check('helper_reject_changed_bound_birth',lambda:reject(lambda:N.verify([c,p,{**s,'birth':'301'}],[p,s],**kw)))
check('host_reject_unknown_sibling',lambda:reject(lambda:admit([c,p,s,{**s,'pid':4,'parent':1}],[c,p,s])))
check('host_reject_bound_role_disappearance',lambda:reject(lambda:admit([c,p],[c,p,s])))

class FakeLinux:
    def snapshot(self): return {'fixture':True}
    def validate_bound(self, snapshot, binding, **kwargs):
        eq(snapshot,{'fixture':True}); eq(binding,{'identity':'linux'}); eq(kwargs['worker_pid'],900)
        return binding
L=FakeLinux()
S=NS(desc=desc,checked=checked,read=read,WORK=WORK,OUT=OUT,REPORT=A,digest=lambda p:desc(p)['sha256'])
O=functions(M/'repair_runtime_owner_v6.py',['validate_native'],{'N':N,'L':L,'WORKER_MODULE':module})
admission={'queue_owner_id':'id','token':'nonce','job':'job','condition':'EEGMamba-lastblock-finetune','repair_registration':{'sha256':'repair'}}
owner={'id':'id','execution_version':11,'child_token':'nonce','child_job':'job','native_roles':[c],'child_pid':2,'pid':1,'child_native_identity':p,'child_native_roles':[p,s],'linux_worker_identity':{'identity':'linux'}}
def oval(items=None,own=None,adm=None,platform='linux',worker=900):
    return O['validate_native'](items or [c,p,s],own or owner,adm or admission,worker_pid=worker,platform=platform,linux_snapshot={'fixture':True},admission_binding={'sha256':'admit'})
check('formal_Linux_both_native_objects_admitted',lambda:eq(oval(),[c,p,s]))
for label,items in [('changed_shim_birth',[c,p,{**s,'birth':'333'}]),('changed_shim_command',[c,p,{**s,'command':'other'}]),('sibling',[c,p,s,{**s,'pid':4,'parent':1}]),('unknown_extra',[c,p,s,{'pid':5,'parent':0,'birth':'500','command':'other'}]),('missing_shim',[c,p]),('missing_controller',[p,s])]:
    check('formal_reject_'+label,lambda items=items:reject(lambda:oval(items=items)))
for label,values in [('wrong_version',{'execution_version':10}),('no_tree',{'child_native_roles':None}),('no_Linux',{'linux_worker_identity':None}),('wrong_owner',{'id':'other'}),('wrong_token',{'child_token':'other'})]:
    check('formal_reject_'+label,lambda values=values:reject(lambda:oval(own={**owner,**values})))
check('formal_wrong_actual_Linux_pid',lambda:reject(lambda:oval(worker=901)))
check('formal_Windows_venv_shim_dynamic',lambda:eq(oval(own={**owner,'child_native_roles':None,'linux_worker_identity':None},platform='win32',worker=3),[c,p,s]))
check('formal_Windows_primary_only',lambda:eq(oval(items=[c,p],own={**owner,'child_native_roles':None,'linux_worker_identity':None},platform='win32',worker=2),[c,p]))

emodule='tridep_ccml.eeg_all_candidates.evaluate_v9'
ec,ep,es=fixture(emodule)
eowner={'source':desc(M/'evaluate_v9.py'),'native_roles':[ec],'child_token':'nonce','child_pid':2,'pid':1,'child_identity':ep,'child_native_roles':[ep,es],'linux_worker_identity':{'identity':'linux'},'child_mode':'mamba_linux','child_admission':{'sha256':'admit'},'CPU_gate':{'sha256':'cpu'}}
E=functions(M/'evaluate_v9.py',['admit_worker','validate_gate_data'],{'N':N,'L':L,'S':S,'SOURCE':M/'evaluate_v9.py','MODULE':emodule,'Q':NS(admit_host=admit)})
def eval_admit(items=None,own=None,platform='linux',worker=900):
    return E['admit_worker'](items or [ec,ep,es],own or eowner,'nonce',worker,platform)
check('evaluation_Linux_both_native_objects_admitted',lambda:eq(eval_admit(),[ec,ep,es]))
for label,items in [('changed_birth',[ec,ep,{**es,'birth':'301'}]),('sibling',[ec,ep,es,{**es,'pid':4,'parent':1}]),('missing_shim',[ec,ep]),('missing_controller',[ep,es])]:
    check('evaluation_reject_'+label,lambda items=items:reject(lambda:eval_admit(items=items)))
for label,values in [('no_tree',{'child_native_roles':None}),('no_Linux',{'linux_worker_identity':None}),('wrong_token',{'child_token':'other'})]:
    check('evaluation_reject_'+label,lambda values=values:reject(lambda:eval_admit(own={**eowner,**values})))
check('evaluation_reject_wrong_Linux_pid',lambda:reject(lambda:eval_admit(worker=901)))
check('evaluation_Windows_venv_dynamic',lambda:eq(eval_admit(own={**eowner,'child_native_roles':None,'linux_worker_identity':None},platform='win32',worker=3),[ec,ep,es]))

qmodule='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v8'
qc,qp,qs=fixture(qmodule,'admit')
condition='EEGMamba-lastblock-finetune'
for r in (qp,qs): r['command']+=' --condition '+condition
qowner={'condition':condition,'native_roles':[qc],'pid':1,'child_pid':2,'child_native_roles':[qp,qs],'linux_worker_identity':{'identity':'linux'}}
qsource=M/'qualify_deterministic_repairs_v8.py'
qadmission={'condition':condition,'source':desc(qsource),'terminal':{'path':'terminal','sha256':'terminal'},'sources':[],'allowed_host':[qc]}
state={'owner':qowner,'items':[qc,qp,qs],'platform':'linux','uid':1000,'pid':900}
counter={'t':0}
def monotonic(): counter['t']+=21; return counter['t']
def qdesc(path):
    if path==qsource: return desc(path)
    return {'path':'terminal','sha256':'terminal'}
Q=functions(qsource,['validate_worker','admit_host'],{'N':N,'L':L,'sys':NS(platform='linux'),'os':NS(getuid=lambda:state['uid'],getpid=lambda:state['pid']),'SOURCE':qsource,'ROOT':Path('root'),'QUEUE':Path('queue'),'desc':qdesc,'check_pause':lambda:None,'check_boundary':lambda:None,'source_check':lambda:None,'checked':lambda x:x,'read':lambda p:state['owner'],'host_inventory':lambda:state['items'],'time':NS(monotonic=monotonic,sleep=lambda n:None)})
def qvalidate(own=None,items=None,platform='linux',uid=1000,pid=900):
    state.update(owner=own or qowner,items=items or [qc,qp,qs],uid=uid,pid=pid)
    Q['sys'].platform=platform
    return Q['validate_worker'](qadmission,condition,'admit')
check('qualifier_Linux_both_native_objects',lambda:eq(qvalidate()['allowed_host'],[qc,qp,qs]))
for label,items in [('changed_birth',[qc,qp,{**qs,'birth':'301'}]),('sibling',[qc,qp,qs,{**qs,'pid':4,'parent':1}]),('missing_shim',[qc,qp])]:
    check('qualifier_reject_'+label,lambda items=items:reject(lambda:qvalidate(items=items)))
check('qualifier_reject_root_worker',lambda:reject(lambda:qvalidate(uid=0)))
check('qualifier_reject_wrong_Linux_pid',lambda:reject(lambda:qvalidate(pid=901)))
check('qualifier_waits_for_native_tree',lambda:reject(lambda:qvalidate(own={**qowner,'child_native_roles':None})))
check('qualifier_waits_for_Linux_binding',lambda:reject(lambda:qvalidate(own={**qowner,'linux_worker_identity':None})))
check('qualifier_Windows_venv_dynamic',lambda:eq(set(x['pid'] for x in qvalidate(own={**qowner,'child_native_roles':None,'linux_worker_identity':None},platform='win32',pid=3)['allowed_host']),{1,2,3}))

for file, name, publish in [('qualify_deterministic_repairs_v8.py','run_controller','current.update(linux_worker_identity'),('continuation_queue_v11.py','child','owner.update(linux_worker_identity'),('evaluate_v9.py','controller','owner.update(linux_worker_identity')]:
    source=fsource(M/file,name)
    check(file+'_capture_order',lambda source=source,publish=publish:require(source.index('L.wait_capture(')<source.index('N.capture(')<source.index(publish), 'Capture/publication order'))
    check(file+'_tree_inventory_exclusion',lambda source=source:require('admit_host(items,' in source and 'native_tree' in source,'Controller lacks exclusion'))
for file,name in [('qualify_deterministic_repairs_v8.py','validate_worker'),('repair_runtime_owner_v6.py','enter'),('evaluate_v9.py','worker')]:
    source=fsource(M/file,name)
    check(file+'_double_identity_wait',lambda source=source:require("owner.get('linux_worker_identity') and owner.get('child_native_roles')" in source,'Worker can leave gate early'))
    check(file+'_uid1000',lambda source=source:require('os.getuid() == 1000' in source,'Model uid hard gate missing'))
for file,name in [('repair_runtime_owner_v6.py','claim_worker_owner'),('qualify_deterministic_repairs_v8.py','exclusive'),('evaluate_v9.py','create'),('continuation_queue_v11.py','create')]:
    check(file+'_atomic_claim',lambda file=file,name=name:require(".open('x'," in fsource(M/file,name),'Claim is not CreateNew'))
def normalized_builder(source):
    tree = ast.parse(source)
    mapping = next(n.value for n in tree.body[0].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mapping' for t in n.targets))
    values = ast.literal_eval(mapping)
    eq(values,{'.adapters.codebrain_v2':'.adapters.codebrain_v3','.adapters.eegmamba':'.adapters.eegmamba_v3'})
    for key,value in zip(mapping.keys,mapping.values):
        if key.value=='.adapters.eegmamba': value.value='.adapters.eegmamba_v2'
    return ast.dump(tree)
check('builder_exact_only_Mamba_route_changed',lambda:eq(normalized_builder(fsource(M/'qualify_deterministic_repairs_v8.py','isolated_builder')),ast.dump(ast.parse(fsource(M/'qualify_deterministic_repairs_v7.py','isolated_builder')))))
check('qualifier_only_Mamba_new_execution',lambda:require("condition == CONDITIONS[1]" in fsource(qsource,'safe_before_model'),'CodeBrain would rerun'))
check('qualification_reuse_original_Qual5_provider',lambda:require("provider = PREVIOUS_Q if condition == REPAIR_CONDITIONS[0] else Q" in fsource(M/'deterministic_repair_execution_v6.py','qualification'),'Wrong provider'))
check('CPU13_output_filename',lambda:require('completed_finetune_checkpoint_audit_v13_' in (A/'audit_completed_finetune_checkpoints_v13.py').read_text(),'CPU identity mismatch'))

cpu_source={'path':'cpu12','sha256':'12'}
gate={'passed':True,'original_training_complete':360,'FT':135,'CPU_loads':405}
cpu={'schema_version':13,'auditor_source':cpu_source,'all_135_finetune_checkpoints_verified':True,'verified_completed_finetune_fits':135,'training_complete_finetune_observed':135,'registered_finetune_fits':135,'checkpoint_loads':405,'errors':[],'deferred_completed_fits':[],'pending_fits':[]}
check('evaluation_gate_actual_CPU13_135_405',lambda:E['validate_gate_data'](gate,cpu,cpu_source))
for key,val in [('schema_version',12),('checkpoint_loads',404),('verified_completed_finetune_fits',134),('errors',['error']),('pending_fits',['pending']),('deferred_completed_fits',['deferred'])]:
    check('evaluation_gate_reject_'+key,lambda key=key,val=val:reject(lambda:E['validate_gate_data'](gate,{**cpu,key:val},cpu_source)))
check('evaluation_gate_reject_359',lambda:reject(lambda:E['validate_gate_data']({**gate,'original_training_complete':359},cpu,cpu_source)))

prep=read(sorted(A.glob('core_prefix_repair_source_preparation_*.json'))[-1])
new_sources=[desc(WORK/x['path']) for x in prep['new_sources']]
prior=read(checked(prep['previous139_review']))
check('all_prior139_bytes_unchanged',lambda:[checked(x) for x in prior['reviewed_sources']])
check('all_new15_compile_without_execution',lambda:[compile((WORK/x['path']).read_text(encoding='utf-8'),x['path'],'exec') for x in new_sources])
# Evaluate only assignments with pure literal/path expressions to check static pins.
pin_env={'REL':'tridep_ccml/eeg_all_candidates/','S':S}
for node in ast.parse(qsource.read_text()).body:
    if isinstance(node,ast.Assign) and (any(isinstance(t,ast.Name) and t.id=='PINS' for t in node.targets) or any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='PINS' for t in node.targets)):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'pins','exec'),pin_env)
check('qualifier_all_static_PINS_match',lambda:[eq(desc(WORK/path)['sha256'],digest) for path,digest in pin_env['PINS'].items()])
Dsource=M/'deterministic_repair_execution_v6.py'
D=functions(Dsource,['implementation_sources'],{'S':S,'SOURCE':Dsource,'read_bound':lambda x:read(checked(x)),'NEW_ENTRY_NAMES':('finetune_v9.py','continuation_queue_v11.py','evaluate_v9.py'),'NEW_ANALYSIS_NAMES':('audit_completed_finetune_checkpoints_v13.py','cost_aggregation_v14.py'),'Q':NS(__file__=str(qsource),PINS=pin_env['PINS']),'F3_SHA':'13601bd5f78b5c543bbc9d6c4dcb856fd9e72a6d09c55d18a5a38febeb670e59'})
queues=functions(M/'continuation_queue_v11.py',['sources'],{'S':S,'SOURCE':M/'continuation_queue_v11.py','D':NS(implementation_sources=D['implementation_sources'])})
chain=queues['sources']()
check('source_chain153_unique',lambda:eq((len(chain),len({x['path'] for x in chain})),(153,153)))
check('source_chain_retains_exact139',lambda:require(all(x in chain for x in prior['reviewed_sources']),'Prior sources changed/lost'))
check('native_tree_source_bound_in_chain',lambda:require(desc(M/'native_launcher_tree_v1.py') in chain,'Helper not in source binding'))
for file in ['repair_postprocessing_gate_v6.py']:
    env={'S':S}
    node=next(n for n in ast.parse((A/file).read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'postpins','exec'),env)
    check(file+'_all_pins',lambda env=env:[eq(desc(path)['sha256'],sha) for path,sha in env['PINS'].items()])
def assignment(path,name):
    return ast.literal_eval(next(n.value for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
check('D5_qualifier_pin',lambda:eq(assignment(Dsource,'QUALIFIER_SHA'),desc(qsource)['sha256']))
check('V5_D5_pin',lambda:eq(assignment(M/'repair_observation_v6.py','D_SHA'),desc(Dsource)['sha256']))
check('V5_F9_pin',lambda:eq(assignment(M/'repair_observation_v6.py','F4_SHA'),desc(M/'finetune_v9.py')['sha256']))

cost_source=A/'cost_aggregation_v14.py'
C=functions(cost_source,['repair_worker_record','category_totals','add_preserved_failure_closure'],{'S':S,'PREVIOUS_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v4.py'),'PREVIOUS_SUCCESS_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v5.py'),'PREVIOUS_IDENTITY_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v6.py'),'PREVIOUS_TREE_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v7.py'),'Q':NS(SOURCE=qsource,CONDITIONS=('CodeBrain-lastblock-finetune',condition)),'copy':copy,'COUNTER_CATEGORY':'engineering','FAILURE_CLOSURE':assignment(cost_source,'FAILURE_CLOSURE')})
def ledger(version,wrong=False):
    value={'source':desc(M/('qualify_deterministic_repairs_v'+str(version)+'.py')),'kind':'deterministic_repair_engineering','condition':condition,'status':'running','formal_fits':0,'primary_engineering_seconds':None}
    for k in ('forward','backward','encoder_forward','encoder_windows','optimizer_step'):
        value[k+'_started']=value[k+'_returned']=0
    if wrong:value['source']=desc(qsource)
    return C['repair_worker_record']({'path':'tridep_ccml/experiments/deterministic_repair_qualification_v'+str(version)+'/condition/id/attempt.json'},value,lambda v:{'observed_counters':{},'seconds_observed':0})
for v in (4,5,6,7,8):
    check('cost14_retains_qual'+str(v)+'_unknown_seconds',lambda v=v:eq(ledger(v)['seconds_observed'],None))
for v in (4,5,6,7):
    check('cost14_reject_relabel_qual'+str(v),lambda v=v:reject(lambda:ledger(v,True)))
closure_result={'inputs':[],'cost_records':[{'sentinel':'unchanged'}]}
C['add_preserved_failure_closure'](closure_result)
check('actual98bd_closure_bound_without_new_primary',lambda:eq(closure_result['cost_records'],[{'sentinel':'unchanged'}]))
check('actual98bd_elapsed_stays_unknown',lambda:eq(closure_result['repair_qualification_failure_closures'][0]['original_elapsed'],None))
check('actual98bd_original_status_preserved',lambda:eq(closure_result['repair_qualification_failure_closures'][0]['original_status_preserved'],'worker_starting'))
before=copy.deepcopy(closure_result)
C['add_preserved_failure_closure'](closure_result)
check('actual98bd_closure_idempotent_inputs',lambda:eq(closure_result,before))
check('original_missing_owner_not_reopened',lambda:require(not any(x['path'].endswith('deterministic_repair_qualification_v5/owner.json') for x in closure_result['inputs']),'Missing owner improperly reused'))
# Actual preserved qualifier7 failure is a primary row once, with unknown internal tail.
failure=OUT/'deterministic_repair_qualification_v7/EEGMamba-lastblock-finetune/1791782206453969000/attempt.json'
failed=read(failure)
def normalize_actual(v):
    return {'observed_counters':{key:v[key] for key in ('forward_returned','backward_started','backward_returned','encoder_windows_returned','optimizer_step_returned')}}
actual_row=C['repair_worker_record'](desc(failure),failed,normalize_actual)
check('actual_Qual7_failed_primary_48_540251404',lambda:eq(actual_row['seconds_observed'],48.540251404))
check('actual_Qual7_unreturned_internal_work_unknown',lambda:eq(actual_row['unreturned_call_internal_work_unknown'],True))
check('actual_Qual7_observed_counts_preserved',lambda:eq(actual_row['observed_counters'],{'forward_returned':5,'backward_started':3,'backward_returned':2,'encoder_windows_returned':40,'optimizer_step_returned':0}))
check('actual_Qual7_failure_status_not_relabelled',lambda:eq(actual_row['status'],'failed'))
check('all_C2_G2_M3_bound_in_chain',lambda:require(all(desc(M/name) in chain for name in ('mamba_cumsum_cpu_fp32_v2.py','mamba_private_backward_graph_v2.py','adapters/eegmamba_v3.py')),'Mamba private sources missing'))
# Execute the actual nested D6 normalizer on source objects, never adapters.
nested=next(n for n in ast.walk(ast.parse(Dsource.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='builder_ast')
qmod=NS(__file__=str(qsource));oldqmod=NS(__file__=str(M/'qualify_deterministic_repairs_v5.py'))
ns={'ast':ast,'Path':Path,'require':require,'Q':qmod}
exec(compile(ast.Module(body=[nested],type_ignores=[]),'D6_nested_source_normalizer','exec'),ns)
check('actual_D6_Qual5_CB_builder_reuse_gate',lambda:eq(ns['builder_ast'](qmod),ns['builder_ast'](oldqmod)))

report={'schema':'core_prefix_repair_chain_stdlib_contract_v1','status':'source_preparation_contracts_passed','passed':True,'checks':RESULTS,'check_count':len(RESULTS),'sources':new_sources,'source_chain':chain,'previous139_review':prep['previous139_review'],'actual_failure_closure':C['FAILURE_CLOSURE'],'model_operations':0,'GPU_operations':0,'Tensor_loads':0,'process_queries':0,'process_controls':0,'registration_writes':0,'outer_test_reads':0,'real_qualification_or_training_executed':False,'reviewer_source':desc(Path(__file__))}
target=A/('core_prefix_repair_chain_contracts_'+str(time.time_ns())+'.json')
with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False);stream.write('\n')
print(json.dumps({'artifact':desc(target),'check_count':len(RESULTS),'source_count':len(chain),'passed':True}))
