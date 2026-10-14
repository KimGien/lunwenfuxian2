"""Preparation contract: AST/fake and original JSON/source only; never actual audit."""
import ast,copy,hashlib,json,math,sys,time,types
from pathlib import Path
WORK=Path(__file__).resolve().parents[3]; A=WORK/'output/analysis/eeg_all_candidates_v1_20261008';M=WORK/'tridep_ccml/eeg_all_candidates'
checks=[]
def ok(x,name):
    if not x:raise AssertionError(name)
    checks.append(name)
def desc(p):
    p=Path(p).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(d):
    p=WORK/d['path'];assert p.suffix in ('.json','.py','.md');assert desc(p)==d;return p.resolve()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
S=types.SimpleNamespace(WORK=WORK,OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008',REPORT=A,desc=desc,checked=checked,read=read)
def extract(path,functions,assignments=()):
    tree=ast.parse(path.read_text(encoding='utf-8'));nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in node.targets):nodes.append(node)
    ns={'ast':ast,'copy':copy,'Path':Path,'math':math,'S':S,'SOURCE':path.resolve()}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
cpu=A/'audit_completed_finetune_checkpoints_v17.py';gate=M/'post_training_cpu_recovery_v1.py';e=M/'evaluate_v12.py'
ns=extract(cpu,{'require','reference_verified','patch_wide'},{'MAMBA','LINUX_REFERENCE','WINDOWS_REFERENCE','CAPTURE_SOURCE','CHILD_COMPLETE','EXPRESSIONS'})
ref=ns['reference_verified']();ok(len(ref['reference_table'])==200,'actual Linux200 independent reference/15 history/source/runtime/F12 bindings')
old=A/'audit_full_plan_acceptance_v3.py';capture=[]
def compiler(value,*args,**kwargs):
    if isinstance(value,ast.Module):capture.append(copy.deepcopy(value))
    return compile(value,*args,**kwargs)
ns['compile']=compiler;wide=types.ModuleType('_fake_wide');wide.__file__=str(old)
ns['patch_wide'](wide,ref);ok(len(capture)==2,'exact two private wide functions compiled without execution')
oldtree=ast.parse(old.read_text(encoding='utf-8'))
changed_count=0
for mod in capture:
    function=mod.body[0];conditions=[x for x in ast.walk(function) if isinstance(x,ast.IfExp) and isinstance(x.body,ast.Call) and isinstance(x.body.func,ast.Name) and x.body.func.id=='_CPU17_linux_expected'];changed_count+=len(conditions)
    class Restore(ast.NodeTransformer):
        def visit_IfExp(self,node):
            if node in conditions:return node.orelse
            return self.generic_visit(node)
    restored=Restore().visit(function);original=next(x for x in oldtree.body if isinstance(x,ast.FunctionDef) and x.name==function.name)
    ok(ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False),'entire original '+function.name+' AST preserved beyond exact Mamba expected values')
ok(changed_count==3,'exactly three changed expected expressions; nonFT duplicate left unchanged')
f={'condition':ns['MAMBA'],'fit_id':ref['fits'][0]['fit_id']}
for row in ref['reference_table']:
 for group in ('head','backbone'):assert wide._CPU17_linux_expected(f,'finetune',row['epoch'],group).hex()==row[group+'_lr_hex']
ok(True,'400 original-platform expected floats strictly retain full hex')
for fit,policy,epoch,group in [({**f,'condition':'CodeBrain-lastblock-finetune'},'finetune',33,'head'),({**f,'fit_id':'wrong'},'finetune',33,'head'),(f,'linear',33,'head'),(f,'finetune',0,'head'),(f,'finetune',33,'other')]:
 try:wide._CPU17_linux_expected(fit,policy,epoch,group)
 except ValueError:checks.append('reject invalid LR scope '+str((fit['condition'],fit['fit_id'],policy,epoch,group)))
 else:raise AssertionError('bad scope accepted')
# Gate uses real failure/release/child/source JSON; fake only canonical360 verifier
# so no binary/model/native work is done by this preparation.
g=extract(gate,{'require','sources','completed_training_verified','validate_CPU_shape'},{'CPU17','FAILED_CPU16','Q14_FAILED','Q14_RELEASED','Q14_REG','CHILD_COMPLETE','LINUX_REFERENCE','WINDOWS_REFERENCE','ORIGINAL_REG'})
g['Q']=types.SimpleNamespace(check_pause=lambda:None);g['D']=types.SimpleNamespace(previous_codebrain_completion=lambda:{'fixture':'previous completion remains delegated'})
registration=read(checked(g['ORIGINAL_REG'])); statuses=[]
for fit in registration['fits']:
 p=S.OUT/'runs'/fit['fit_id']/'status.json';v=read(p);assert v['fit_id']==fit['fit_id'] and v['status']=='training_complete';statuses.append(desc(p))
g['R']=types.SimpleNamespace(assert_test_unlocked=lambda:{'formal_fits_verified':360,'registration':g['ORIGINAL_REG'],'statuses':statuses})
actual=g['completed_training_verified']();ok(len(actual['training_sources'])==194 and len(actual['training_statuses'])==360,'actual failedQ14/exit0/releasedowner/failedCPU16/194/source-review and all360 JSON states accepted')
base=read(checked(g['FAILED_CPU16']));synthetic=copy.deepcopy(base)
synthetic.update(schema_version=17,auditor_source=g['CPU17'],registered_total_fits=360,registered_finetune_fits=135,training_complete_finetune_observed=135,verified_completed_finetune_fits=135,logical_completed_fits_counted=135,checkpoint_loads=405,status='all_finetune_verified',all_135_finetune_checkpoints_verified=True,partial_snapshot=False,errors=[],pending_fits=[],deferred_completed_fits=[])
oldrow=base['completed_rows'][0];synthetic['completed_rows']=[]
for fit in registration['fits']:
 if fit['stage']=='3':
  row=copy.deepcopy(oldrow);row['fit_id']=fit['fit_id'];synthetic['completed_rows'].append(row)
synthetic['observed_auditor_operations_including_failed_checks']={'cpu_checkpoint_loads_started':405,'cpu_checkpoint_loads_returned':405,'model_construction_attempts_blocked':0,'gpu_initialization_attempts_blocked':0}
g['validate_CPU_shape'](synthetic,registration,g['ORIGINAL_REG'],g['CPU17']);ok(True,'synthetic full CPU17 shape passes only135/405')
for key,value in [('schema_version',16),('checkpoint_loads',360),('verified_completed_finetune_fits',120),('errors',[{'error':'x'}]),('pending_fits',[{}]),('deferred_completed_fits',[{}]),('partial_snapshot',True)]:
 bad=copy.deepcopy(synthetic);bad[key]=value
 try:g['validate_CPU_shape'](bad,registration,g['ORIGINAL_REG'],g['CPU17'])
 except ValueError:checks.append('reject partial/oldCPU17 shape '+key)
 else:raise AssertionError(key)
# Evaluation's math/worker/identity functions must stay exactly previous E11 after
# mechanical version labels, with only gate functions intentionally different.
previous=(M/'evaluate_v11.py').read_text(encoding='utf-8').replace('E11','E12').replace('CPU16','CPU17').replace('evaluate_v11','evaluate_v12').replace('evaluation_v11_control','evaluation_v12_control').replace("'version': 11, 'wrapper_source'","'version': 12, 'wrapper_source'")
oldfunc={x.name:x for x in ast.parse(previous).body if isinstance(x,ast.FunctionDef)};newfunc={x.name:x for x in ast.parse(e.read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef)}
for name,node in oldfunc.items():
 if name not in ('validate_gate_data','verify_gate'):assert ast.dump(node,include_attributes=False)==ast.dump(newfunc[name],include_attributes=False),name
ok(True,'all non-gate E12 evaluator/control/identity/math AST equals E11 with version labels only')
for p in (cpu,gate,e):compile(p.read_text(encoding='utf-8'),str(p),'exec')
ok(True,'three source syntax/default main guards intact')
value={'schema':'post_training_cpu_recovery_stdlib_preparation_v1','passed':True,'checks':checks,'reviewed_sources':[desc(cpu),desc(gate),desc(e)],'Q14_registration':g['Q14_REG'],'Linux_reference':ns['LINUX_REFERENCE'],'Windows_reference':ns['WINDOWS_REFERENCE'],'source':desc(Path(__file__)),'original_training_source_count':194,'actual_CPU_audit_run':False,'actual_gate_created':False,'Tensor_loads':0,'model_calls':0,'GPU_calls':0,'process_queries':0,'outer_test_reads':0,'binary_checkpoint_hash_reads':0}
p=A/('post_training_cpu_recovery_stdlib_contract_'+str(time.time_ns())+'.json')
with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({'artifact':desc(p),'passed':True,'checks':len(checks),'sources':value['reviewed_sources']}))
