"""Author preparation tests only: source/AST and synthetic JSON, no collection."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time

P=Path(__file__).resolve().parent; ROOT=P.parents[2]
NAMES=['pause_memory_resume_cost_companion_v2.py','sync_boundary_v3_cost_companion_v2.py',
       'audit_memory_pause_final_delivery_v2.py','system_standby_cost_companion_v2.py','audit_system_standby_final_delivery_v2.py']
CHECKS=[]
def ok(name,condition):
    assert condition,name
    CHECKS.append({'name':name,'passed':True})
def reject(name,fn):
    try: fn()
    except (ValueError,KeyError,TypeError): ok(name,True)
    else: raise AssertionError(name)
def desc(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(n):
    p=P/n;ns={'__file__':str(p),'__name__':'_synthetic_contract'}
    exec(compile(p.read_text(encoding='utf8'),str(p),'exec'),ns);return ns
mods={n:load(n) for n in NAMES};h=mods[NAMES[0]]
for n,ns in mods.items():
    with contextlib.redirect_stdout(io.StringIO()):
        if n=='system_standby_cost_companion_v2.py':
            old=sys.argv;sys.argv=['prepared'];ns['main']();sys.argv=old
        else: ns['main']([])
    ok(n+' defaults to zero collection',True)

def d(label):return {'path':label+'.json','sha256':'a'*64}
def key(v):return json.dumps(v,sort_keys=True)
objects={}
def add(v,value):objects[key(v)]=value;return v
def read(v):return copy.deepcopy(objects[key(v)])
closures=[];primary=[];parents=[];records=[]
for i,closure_d in enumerate(h['LOSSES']):
    reg=d('recovery'+str(i));src=d('source'+str(i));orig=d('attempt'+str(i));snap=d('attempt_snapshot'+str(i))
    cpu=d('cpu'+str(i));owner=d('owner'+str(i));q_snap=d('queue_snapshot'+str(i));absence=d('absence'+str(i))
    original={'status':'running','seconds':None};add(orig,original);add(snap,original)
    add(reg,{'source':src});add(src,'source');add(cpu,{'passed':True});add(owner,{});add(q_snap,{'status':'running','seconds':None});add(absence,{})
    durable={'epoch':8+i,'optimizer_updates':32+i,'backward_calls':216+i};extra={'optimizer_updates':4,'backward_calls':27}
    closure={'schema':'lost_f3_process_closure_v'+str(i+1),'passed':True,'status':'process_absence_observed',
        'old_attempt_bytes_unchanged':True,'known_extra_not_added_again_to_original_returned_counters':True,
        'recovery_registration':reg,'closure_source':src,'original_attempt':orig,'attempt_snapshot':snap,
        'checkpoint_cpu_resume_proof':cpu,'old_owner_snapshot':owner,'old_queue_attempt_snapshot':q_snap,
        'process_absence_evidence':absence,'unflushed_tail_counters':None,'final_elapsed_seconds':None,
        'actual_exit_time':None,'graceful_terminal_flush':False,'durable':durable,'known_extra_lower_bounds':extra}
    row={'artifact':orig,'closure':closure_d,'aggregation_role':'primary_observed_invocation','passed':True,
        'recovery_registration':reg,'closure_source':src,'checkpoint_cpu_resume_proof':cpu,
        'final_elapsed_seconds':None,'unflushed_tail_unknown':True,'extra_work_not_added_again':True,
        'durable_epoch':durable['epoch'],'durable_optimizer_updates':durable['optimizer_updates'],'durable_backward_calls':durable['backward_calls'],
        'known_extra_optimizer_returns_outside_checkpoint':4,'known_extra_backward_returns_outside_checkpoint':27}
    if i:
        parent_d=d('parent'+str(i));add(parent_d,{'status':'running','seconds':None,'returncode':None});add(q_snap,objects[key(parent_d)])
        closure.update(previous_recovery_registration=d('recovery'+str(i-1)),old_queue_attempt_original=parent_d)
        parents.append({'closure':closure_d,'artifact':parent_d,'final_elapsed_seconds':None,'added_again':False})
        records.append({'artifact':parent_d,'category':'dispatcher_elapsed','aggregation_role':'supplemental_nested_or_unmeasured','seconds_observed':None})
    add(closure_d,closure);closures.append(closure);primary.append(row)
    records.append({'artifact':orig,'aggregation_role':'primary_observed_invocation','category':'formal_training_attempt','status':'running','seconds_observed':None})
for v in h['ERRATA']:add(v,{})
proof=dict(zip(('original_recovery_closure','previous_recovery_closure','recovery_closure'),h['LOSSES']))
proof.update(original_recovery_registration=d('recovery0'),previous_recovery_registration=d('recovery1'),recovery_registration=d('recovery2'))
cost={'completion_gate':{'recovery_execution_proof':proof},'source_bound_lost_f3_attempts':primary,
      'source_bound_lost_parent_observer_scopes':parents,'cost_records':records}
losses=h['verify_loss_chain'](read,cost)
ok('three actual identities lifecycle close without producer terminal',losses['passed'] and len(losses['three_loss_events'])==3 and len(losses['lost_parent_scopes'])==2 and all(r['producer_terminal'] is False and r['seconds_observed'] is None for r in losses['lost_parent_scopes']))
def altered_cost(fn):
    value=copy.deepcopy(cost);fn(value);return h['verify_loss_chain'](read,value)
reject('omitted loss rejected',lambda:altered_cost(lambda c:c['source_bound_lost_f3_attempts'].pop()))
reject('duplicate loss primary rejected',lambda:altered_cost(lambda c:c['source_bound_lost_f3_attempts'].append(copy.deepcopy(c['source_bound_lost_f3_attempts'][0]))))
reject('parent cost zero-filled rejected',lambda:altered_cost(lambda c:c['cost_records'][1].update(seconds_observed=0)))
reject('lost physical elapsed inferred rejected',lambda:altered_cost(lambda c:c['source_bound_lost_f3_attempts'][0].update(final_elapsed_seconds=1)))
reject('extra work added twice rejected',lambda:altered_cost(lambda c:c['source_bound_lost_f3_attempts'][1].update(extra_work_not_added_again=False)))
reject('durable counter changed rejected',lambda:altered_cost(lambda c:c['source_bound_lost_f3_attempts'][2].update(durable_epoch=100)))
reject('missing parent binding rejected',lambda:altered_cost(lambda c:c['source_bound_lost_parent_observer_scopes'].pop()))
original=copy.deepcopy(objects)
objects[key(d('parent1'))]['status']='complete'
reject('fabricated lost producer natural terminal rejected',lambda:h['verify_loss_chain'](read,cost));objects=copy.deepcopy(original)
objects[key(h['LOSSES'][2])]['previous_recovery_registration']=d('recovery0')
reject('recovery ancestor drift rejected',lambda:h['verify_loss_chain'](read,cost));objects=copy.deepcopy(original)

lc={'passed':True,'training_registration':h['Q14'],'formal_fits_verified':360,'FT_verified':135,'CPU_loads_verified':405,
    'training_sources':[d('source_'+str(i)) for i in range(194)],'CodeBrain_completion':{k:d('cb_'+k) for k in ('registration','terminal','CPU_audit','child')}}
for k in ('training_terminal','CPU_gate','CPU_audit','CPU_auditor','evaluation_control_completion','evaluation_runtime','evaluation_released_owner','training_source_review'):lc[k]=d(k)
cd,wd=d('finalcost'),d('finalwide')
cost.update(schema_version=17,source=h['COST'],actual_cost_collector_source=h['COST'],finalizer_source=h['FINALIZER'],status='verified_final_cost',final_snapshot=True,
    actual_training_lifecycle=lc,source_bound_cpu_checkpoint_verification={'passed':True,'verified_fits':135,'prior_actual_cpu_checkpoint_loads':405,'new_cpu_checkpoint_loads':0})
wide={'source':h['WIDE'],'schema_version':7,'actual_cost_collector_source':h['COST'],'full_plan_completed':True,'objective_satisfied':True,
    'final_cost':{'passed':True,'snapshot':cd},'requirements':[{'id':'R%02d'%i,'status':'verified_complete'} for i in range(1,31)],'actual_training_lifecycle':lc}
pair=h['verify_final_pair'](lambda d:{},cost,wide,cd,wd)
ok('same actual final pair strict shape',pair['final_cost']==cd and pair['full_plan_acceptance']==wd)
def bad_pair(part,fn):
    c,w=copy.deepcopy(cost),copy.deepcopy(wide);fn(c if part=='cost' else w);return h['verify_final_pair'](lambda d:{},c,w,cd,wd)
reject('old cost6 rejected',lambda:bad_pair('cost',lambda c:c.update(schema_version=6)))
reject('old wide6 rejected',lambda:bad_pair('wide',lambda w:w.update(schema_version=6)))
reject('cost mismatch rejected',lambda:bad_pair('wide',lambda w:w['final_cost'].update(snapshot=d('other'))))
reject('partial thirty rejected',lambda:bad_pair('wide',lambda w:w['requirements'].pop()))
reject('mixed lifecycle mismatch rejected',lambda:bad_pair('wide',lambda w:w['actual_training_lifecycle'].update(formal_fits_verified=345)))
reject('fake current collector rejected',lambda:bad_pair('cost',lambda c:c.update(source=d('cost6'))))

def function_ast(n,name):
    tree=ast.parse((P/n).read_text(encoding='utf8'));return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
ok('standby complete-range coverage/pairing/extra-fit rule unchanged',function_ast('audit_system_standby_final_delivery_v1.py','verify_scope_closure')==function_ast('audit_system_standby_final_delivery_v2.py','verify_scope_closure'))
ok('noon exact native same-process resume proof unchanged',function_ast('audit_memory_pause_final_delivery_v1.py','verify_resume')==function_ast('audit_memory_pause_final_delivery_v2.py','verify_resume'))
ok('standby immutable native history proof unchanged',function_ast('audit_system_standby_final_delivery_v1.py','verify_native_observation')==function_ast('audit_system_standby_final_delivery_v2.py','verify_native_observation'))
ok('standby no added elapsed arithmetic unchanged',function_ast('audit_system_standby_final_delivery_v1.py','formal_subset')==function_ast('audit_system_standby_final_delivery_v2.py','formal_subset'))
ok('G3 recollection removed',not any(isinstance(n,ast.FunctionDef) and n.name=='collect' for n in ast.parse((P/NAMES[1]).read_text(encoding='utf8')).body))
for name in NAMES:
    text=(P/name).read_text(encoding='utf8');tree=ast.parse(text)
    bad=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            terms=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            bad.extend(x for x in terms if x.split('.')[0] in ('torch','subprocess','ctypes','tridep_ccml','numpy'))
    ok(name+' no experimental imports',not bad)

training=json.loads((ROOT/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v14/registration.json').read_text(encoding='utf8'))
ok('all194 training sources retain sealed bytes',len(training['sources'])==194 and all(hashlib.sha256((ROOT/d['path']).read_bytes()).hexdigest()==d['sha256'] for d in training['sources']))
for name,expected in [('supplemental_head_resource_measurement_v7.py','3b85f4c444fff69711a1299ec260b0695f5896b8caff48a631fcc0ec28f36711'),('finalize_full_plan_cost_v7.py','ab5d4d892f3d6a0e75c2c82db5bd3bbcfdffd086b4be4f92858234612982a712'),('audit_full_plan_acceptance_v7.py','ad8ddc3b50fd8c4537d5aaffe02451196762f27a6d3ffaa33fdf282fee65e795')]:ok(name+' unchanged',desc(P/name)['sha256']==expected)
value={'schema':'final_companion_v2_stdlib_preparation','passed':True,'source':desc(Path(__file__).resolve()),'sources':[desc(P/n) for n in NAMES],
    'checks':CHECKS,'actual_collect_calls':0,'actual_scientific_or_native_queries':0,'Tensor_GPU_or_test_operations':0,'actual_final_artifacts_created':False}
target=P/('final_companion_v2_stdlib_contract_'+str(time.time_ns())+'.json')
with target.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
print(json.dumps({'artifact':desc(target),'checks':len(CHECKS),'passed':True}))
