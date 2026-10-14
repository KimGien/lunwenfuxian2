"""Source/AST/fake-only preparation. Execute through python -B - stdin."""
from pathlib import Path
import ast, copy, contextlib, hashlib, io, json, sys, time, types
SOURCE=Path(__file__).resolve();A=SOURCE.parent;WORK=A.parents[2]
def desc(p):
    p=Path(p).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
checks=[]
def ok(name,value=True):
    assert value,name;checks.append(name)
def reject(name,fn):
    try:fn()
    except (ValueError,AssertionError,KeyError):ok(name);return
    raise AssertionError(name+' did not reject')
def load(name):
    p=A/name;m=types.ModuleType('_stdlib_'+name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
before=set(sys.modules)
res=load('supplemental_head_resource_measurement_v7.py');fin=load('finalize_full_plan_cost_v7.py');wide=load('audit_full_plan_acceptance_v7.py')
for m in (res,fin,wide):
    out=io.StringIO()
    with contextlib.redirect_stdout(out):code=m.main([])
    value=json.loads(out.getvalue());ok(Path(m.__file__).name+' default zero',code==0 and value['status']=='preparation_only' and value['new_outputs']==0)
ok('Default imports add no Torch',not any(x=='torch' or x.startswith('torch.') for x in set(sys.modules)-before))
reg=json.loads((WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v14/registration.json').read_text(encoding='utf-8'))
ok('194 exact inherited source bytes',len(reg['sources'])==194 and all(desc(WORK/x['path'])==x for x in reg['sources']))
ok('Probe boundaries use new pause-aware guard',res.D.assert_terminal_still_bound is res.assert_terminal_still_bound and 'check_pause' in res.assert_terminal_still_bound.__code__.co_names)
ok('Unchanged resource delegate body',res.measure is res.D.measure and res.delegated_source()['sha256']==res.V2_SHA)
def forbidden(*a,**kw):raise AssertionError('Forbidden deeper operation')
saved=(res.P.inspect,res.source_review,res.lifecycle_evidence,res.P.verify_existing)
res.P.inspect=lambda:{'status':'pending','reasons':['training'],'actual_fit_statuses':{'training_complete':350}}
res.source_review=res.lifecycle_evidence=res.P.verify_existing=forbidden
ok('Pending resource never calls downstream',res.inspect_gate()['status']=='pending')
res.P.inspect,res.source_review,res.lifecycle_evidence,res.P.verify_existing=saved
reject('No independent review rejected',lambda:res.source_review(None))
tree=ast.parse(res.POST_SOURCE.read_text(encoding='utf-8'));node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='verify_existing')
boundary=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='manifest' for t in n.targets)]
prefix=node.body[:boundary[0]]
ok('Unique pre-report boundary',len(boundary)==1)
ok('Prefix no completed report verification',not any(isinstance(x,ast.Attribute) and x.attr=='completed_report_verified' for n in prefix for x in ast.walk(n)))
ok('Prefix retains CPU shape and actual test lock',all(any(isinstance(x,(ast.Name,ast.Attribute)) and getattr(x,'id',getattr(x,'attr',None))==key for n in prefix for x in ast.walk(n)) for key in ('validate_CPU_shape','assert_test_unlocked')))
fakecheap={'actual_fit_statuses':{'training_complete':350},'reasons':['pending']}
pending=wide.pending(fakecheap)
ok('All thirty pending requirements',len(pending['requirements'])==30 and not pending['full_plan_completed'] and not pending['outer_test_artifacts_read'] and {x['status'] for x in pending['requirements']}=={'pending'})
reject('Strict completion requires actual run',lambda:wide.main(['--require-complete']))
# Synthetic CPU evidence: 135 distinct original-role placeholders, no filesystem/data access.
fakefits=[{'fit_id':str(i),'stage':'3'} for i in range(135)]
cpu={'schema_version':16,'registration':{'path':'r','sha256':'r'},'auditor_source':{'path':'c','sha256':'c'},'registered_total_fits':360,'registered_finetune_fits':135,'training_complete_finetune_observed':135,'verified_completed_finetune_fits':135,'logical_completed_fits_counted':135,'checkpoint_loads':405,'status':'all_finetune_verified','all_135_finetune_checkpoints_verified':True,'partial_snapshot':False,'errors':[],'deferred_completed_fits':[],'pending_fits':[],'completed_rows':[{'fit_id':str(i),'passed':True,'cpu_checkpoint_audit':{'cpu_tensors_verified':True,'checkpoint_loads':3,'only_map_location_cpu':True,'gpu_operations':0,'model_constructions':0}} for i in range(135)],'observed_auditor_operations_including_failed_checks':{'cpu_checkpoint_loads_started':405,'cpu_checkpoint_loads_returned':405,'model_construction_attempts_blocked':0,'gpu_initialization_attempts_blocked':0}}
def shape(v):return res.P.validate_CPU_shape(v,{'fits':fakefits},cpu['registration'],cpu['auditor_source'])
ok('Full135405 shape passes',len(shape(cpu))==135)
for key,bad in [('checkpoint_loads',404),('schema_version',15),('errors',['bad']),('deferred_completed_fits',['x']),('pending_fits',['x']),('partial_snapshot',True),('auditor_source',{'path':'wrong','sha256':'wrong'})]:
    v=copy.deepcopy(cpu);v[key]=bad;reject('CPU rejects '+key,lambda v=v:shape(v))
v=copy.deepcopy(cpu);v['completed_rows'][-1]=v['completed_rows'][0];reject('CPU rejects duplicate identity',lambda:shape(v))
# Fake reuse checks, with file hashing replaced by an in-memory binding check.
condition='EEGMamba-lastblock-finetune';fit={'fit_id':'fake','condition':condition}
row={k:k for k in ('fit_id','condition','seed','fold','epochs','updates','best_epoch','stop','metadata','completion','condition_source_lock')};row.update(fit)
row['actual_execution_observer_proof']={'same':'trace'}
prior={**row,'f2_physical_durable_boundary':row['actual_execution_observer_proof'],'cpu_checkpoint_audit':{'cpu_tensors_verified':True,'checkpoint_loads':3,'only_map_location_cpu':True,'gpu_operations':0,'model_constructions':0,'checkpoints':[{'path':x,'sha256':x} for x in ('initial','latest','best')]}}
wide.CPU_ROWS={'fake':prior};wide.LIFECYCLE={'CPU_audit':{'path':'audit','sha256':'audit'}}
checked=wide.S.checked
wide.S.checked=lambda b:b
ok('FT tensor proof reused with zero loads',wide.deep_checkpoint(fit,row,{},{} )['new_checkpoint_loads']==0)
v=copy.deepcopy(row);v['epochs']='changed';reject('Reuse rejects changed epoch',lambda:wide.deep_checkpoint(fit,v,{},{}))
v=copy.deepcopy(row);v['actual_execution_observer_proof']={};reject('Reuse rejects changed physical trace',lambda:wide.deep_checkpoint(fit,v,{},{}))
wide.S.checked=lambda b:(_ for _ in ()).throw(ValueError('changed checkpoint'))
reject('Reuse rejects changed checkpoint bytes',lambda:wide.deep_checkpoint(fit,row,{},{}))
wide.S.checked=checked
oldast=ast.parse((A/'finalize_full_plan_cost_v6.py').read_text(encoding='utf-8'));newast=ast.parse(Path(fin.__file__).read_text(encoding='utf-8'))
for name in ('measurement_plan','head_resource_gate','bind_mamba_recheck_evidence','append_optional_qualification_v2_cost','append_boundary_control_evidence','verify_phase_coverage'):
    a=next(x for x in oldast.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in newast.body if isinstance(x,ast.FunctionDef) and x.name==name)
    ok(name+' original AST retained',ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
ok('All actual repair families named',all('repair_qualification_'+str(i) in fin.PHASE_PATTERNS for i in range(4,11)) and 'mamba_repeat_diagnosis' in fin.PHASE_PATTERNS)
ok('Cost schema17 exact gate',17 in fin.verify_training_cost.__code__.co_consts)
ok('Three losses explicitly required',3 in fin.bind_lost_f3_recovery_cost.__code__.co_consts)
sources=[desc(A/x) for x in ('supplemental_head_resource_measurement_v7.py','finalize_full_plan_cost_v7.py','audit_full_plan_acceptance_v7.py')]
result={'schema':'final_postprocessing_v7_stdlib_contract_v1','passed':True,'source':desc(SOURCE),'sources':sources,'training_registration':desc(WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v14/registration.json'),'checks':checks,'check_count':len(checks),'actual_training_source_count':194,'sealed_training_source_bytes_unchanged':True,'Tensor_loads':0,'GPU_queries':0,'model_calls':0,'native_process_queries':0,'test_probability_reads':0,'actual_postprocessing_started':False}
target=A/('final_postprocessing_v7_stdlib_contract_'+str(time.time_ns())+'.json')
with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'report':desc(target),'checks':len(checks),'passed':True}))
