"""Independent minimal E13 final compatibility review. No actual collectors or gates."""
from __future__ import annotations
import ast,copy,hashlib,json,re,time,types
from pathlib import Path
SOURCE=Path(__file__).resolve(); A=SOURCE.parent; WORK=SOURCE.parents[3]
M=WORK/'tridep_ccml/eeg_all_candidates'; OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
NAMES=('cost_aggregation_v19.py','repair_postprocessing_gate_v10.py','supplemental_head_resource_measurement_v9.py','finalize_full_plan_cost_v9.py','audit_full_plan_acceptance_v9.py','pause_memory_resume_cost_companion_v4.py','sync_boundary_v3_cost_companion_v4.py','audit_memory_pause_final_delivery_v4.py','system_standby_cost_companion_v4.py','audit_system_standby_final_delivery_v4.py')
OLDNAMES=('cost_aggregation_v18.py','repair_postprocessing_gate_v9.py','supplemental_head_resource_measurement_v8.py','finalize_full_plan_cost_v8.py','audit_full_plan_acceptance_v8.py','pause_memory_resume_cost_companion_v3.py','sync_boundary_v3_cost_companion_v3.py','audit_memory_pause_final_delivery_v3.py','system_standby_cost_companion_v3.py','audit_system_standby_final_delivery_v3.py')
CHECKS=[]
def require(ok,msg):
    if not ok:raise ValueError(msg)
def check(ok,msg): require(ok,msg); CHECKS.append(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def desc(p):
    p=Path(p).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(d):
    p=WORK/d['path'];require(desc(p)==d,'Changed bytes: '+d['path']);return p
def tree(p):return ast.parse(Path(p).read_text(encoding='utf-8'))
def fn(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
def dump(n):return ast.dump(n,include_attributes=False)
def extract(t,names,ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn(t,n)) for n in names],type_ignores=[])),'<independent synthetic accounting>','exec'),ns)
def rejects(call,label):
    try:call()
    except (ValueError,KeyError,TypeError):CHECKS.append(label);return
    raise ValueError('Negative case accepted: '+label)
def consts(t):
    result={}
    for n in t.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
            try:result[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    return result
def allbindings(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'} and isinstance(value['path'],str):yield value
        else:
            for x in value.values():yield from allbindings(x)
    elif isinstance(value,(list,tuple)):
        for x in value:yield from allbindings(x)

def main(contract_path):
    sources=[desc(A/n) for n in NAMES];trees={n:tree(A/n) for n in NAMES}
    require(desc(contract_path)['sha256']=='33e20e0df004097725513ea6e0cd5bca91af209960c88973a87c0fe7db135066','Frozen author contract changed')
    contract=read(contract_path)
    lists=[v for v in contract.values() if isinstance(v,list) and v and isinstance(v[0],dict) and set(v[0])=={'path','sha256'}]
    check(contract['passed'] is True and sources in lists,'Exact frozen ten-source author preparation, not actual completion')
    prior_path=A/'independent_post_training_final_compatibility_source_review_1791833968825479500.json'
    prior=read(prior_path)
    require(desc(prior_path)['sha256']=='8b0f23b56fb645f5ed33b05cf86f67d772704a4a5ab31e93b9d6f36b89847f74','Prior ten-source review changed')
    for d in prior['reviewed_sources']:checked(d)
    training=read(checked(prior['training_registration']))
    for d in training['sources']:checked(d)
    check(len(training['sources'])==194,'Original194 training sources and previous ten postprocessing bytes unchanged')
    e_review=A/'independent_evaluation_storage_recovery_source_review_1791838352376118100.json'; ev=read(e_review)
    require(desc(e_review)['sha256']=='4775863335c3d94bc17fb8f807d6e9cd373f97f3e0eeafcd066ee9e782ad2d1a','E13 review changed')
    for d in ev['reviewed_sources']:checked(d)
    check(ev['passed'] is True,'Previously reviewed E13/source recovery bytes preserved without rerunning evaluation')
    oldhashes=set();newhashes=set()
    for name in OLDNAMES:oldhashes.update(re.findall('[a-f0-9]{64}',(A/name).read_text(encoding='utf-8')))
    for name in NAMES:newhashes.update(re.findall('[a-f0-9]{64}',(A/name).read_text(encoding='utf-8')))
    knownnew={d['sha256'] for d in sources+ev['reviewed_sources']}|{desc(A/'evaluation_storage_recovery_stdlib_contract_1791838170182583300.json')['sha256']}
    check(newhashes <= oldhashes|knownnew,'Every new hash literal is a real successor descriptor; no E12-to-E13 hash text corruption')
    # Evaluate only path/literal descriptor assignments, never imports or module calls.
    bound=0
    for name,t in trees.items():
        ns={'__file__':str(A/name),'Path':Path,'binding':lambda path,sha:{'path':path,'sha256':sha},'S':types.SimpleNamespace(WORK=WORK,OUT=OUT)}
        for n in t.body:
            if not(isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)):continue
            calls=[x for x in ast.walk(n.value) if isinstance(x,ast.Call)]
            if any(not(isinstance(x.func,ast.Name) and x.func.id in ('Path','binding') or isinstance(x.func,ast.Attribute) and x.func.attr=='resolve') for x in calls):continue
            try:v=eval(compile(ast.Expression(n.value),'<descriptor expression>','eval'),ns)
            except (NameError,AttributeError,TypeError):continue
            ns[n.targets[0].id]=v
            for d in allbindings(v):checked(d);bound+=1
            if n.targets[0].id=='PINS':
                for p,h in v.items():require(desc(p)['sha256']==h,'PINS drift');bound+=1
    check(bound>50,'Literal and constructed current/historical source descriptor bindings all match exact bytes')
    cost=trees[NAMES[0]];oldcost=tree(A/OLDNAMES[0])
    for name in ('buffered_observation','category_totals','repair_worker_record','add_repair_ledgers','add_preserved_failure_closure','add_post_training_recovery'):
        require(dump(fn(cost,name))==dump(fn(oldcost,name)),'Historical accounting changed: '+name)
    check(True,'Prior failures, losses, CPU360/405 and platform captures retain original accounting functions')
    text=ast.unparse(fn(cost,'collect'))
    check(text.count('old.collect(')==text.count('previous.augment(')==text.count('add_post_training_recovery(')==text.count('add_evaluation_storage_recovery(')==1,'Original inventory and each augmentation execute once')
    H=consts(tree(M/'evaluation_storage_recovery_v1.py'))
    closurepath=OUT/'evaluation_storage_recovery_v1/complete.json';closure=read(closurepath);closurebinding=desc(closurepath)
    require(closure['source_review']==desc(e_review) and closure['CPU_gate']==ev['CPU_gate'],'Actual independent evaluation recovery closure differs')
    controlroot=Path('synthetic_E13_control');controlsource=M/'evaluate_v13.py'
    fakecontrol={'path':(controlroot/'complete.json').as_posix(),'sha256':'fakecomplete'}
    fakestatus={'path':(controlroot/'runs/fake/status.json').as_posix(),'sha256':'fakestatus'}
    fakeworker={'path':(controlroot/'runs/fake/all_windows/worker.json').as_posix(),'sha256':'fakeworker'}
    fakereleased={'path':(controlroot/'runs/fake/released_owner.json').as_posix(),'sha256':'fakereleased'}
    fakevalues={fakecontrol['path']:{'actual_evaluation_complete':True,'source':desc(controlsource),'evaluation_storage_recovery':closurebinding,'reused_E12_Mamba_worker':H['MAMBA_WORKER'],'status':fakestatus},fakestatus['path']:{'status':'all_evaluation_complete','stages':{'all_windows':{'returncode':0,'worker_report':fakeworker}}},fakeworker['path']:{'status':'complete','source':desc(controlsource),'mode':'all_windows','inclusive_evaluator_worker_seconds':99999.25},fakereleased['path']:{}}
    fakebindings={d['path']:d for d in (fakecontrol,fakestatus,fakeworker,fakereleased)}
    def fdesc(p):
        key=Path(p).as_posix();return fakebindings[key] if key in fakebindings else desc(p)
    def fchecked(d):
        if d['path'] in fakebindings:require(d==fakebindings[d['path']],'Synthetic source changed');return Path(d['path'])
        return checked(d)
    def fread(p):
        key=Path(p).as_posix();return copy.deepcopy(fakevalues[key]) if key in fakevalues else read(p)
    SR=types.SimpleNamespace(ROOT=closurepath.parent,verify_recovery=lambda:copy.deepcopy(closure),**H)
    ns={'S':types.SimpleNamespace(desc=fdesc,checked=fchecked,read=fread),'STORAGE_RECOVERY':SR,'E13':types.SimpleNamespace(CONTROL=controlroot,SOURCE=controlsource),'STORAGE_REPRO':consts(cost)['STORAGE_REPRO'],'require':require}
    extract(cost,['category_totals','add_evaluation_storage_recovery'],ns)
    def blank():return {'inputs':[],'cost_records':[],'category_observations':{},'rules':[]}
    def normalizer(data,**kwargs):return {'observed_counters':{},'seconds_observed':999,'seconds_source_field':'wrong','unknown_counter_fields':[]}
    result=ns['add_evaluation_storage_recovery'](blank(),normalizer)
    rows=result['cost_records'];primaries=[r for r in rows if r['aggregation_role']=='primary_observed_invocation']
    check(len(rows)==7 and len(primaries)==3 and len({r['artifact']['path'] for r in rows})==7,'Synthetic final lifecycle yields exactly three new primaries plus four nested records')
    failed=next(r for r in rows if r['artifact']==H['WINDOWS_WORKER'])
    check(failed['seconds_observed'] is None and failed['observed_counters']=={'cpu_checkpoint_loads_started':1,'cpu_checkpoint_loads_returned':1,'model_constructions':0,'gpu_forward_calls':0,'optimizer_updates':0} and failed['inclusive_worker_seconds_supplemental']==34.58747199999925,'Actual failed header CPU1 has unknown duration; worker34.587s remains a separate nested field')
    check([r['seconds_observed'] for r in primaries[1:]]==[0.4189831000112463,0.056277800002135336] and result['category_observations']['evaluation_controller_and_worker_elapsed']['primary_invocations']==0,'Only actual metrics/short-writer seconds are primary; large synthetic worker time never contributes')
    ns['add_evaluation_storage_recovery'](result,normalizer)
    check(len(result['cost_records'])==7 and len(result['inputs'])==len({r['path'] for r in result['inputs']}),'Repeated synthetic inventory augmentation adds no duplicate records or inputs')
    duplicate=blank();duplicate['cost_records']=[copy.deepcopy(failed),copy.deepcopy(failed)]
    rejects(lambda:ns['add_evaluation_storage_recovery'](duplicate,normalizer),'Duplicate failed-header scope rejected')
    existing=blank();existing['cost_records']=[{'artifact':H['WINDOWS_WORKER'],'category':'old_nested','aggregation_role':'supplemental_nested_elapsed','observed_counters':{},'seconds_observed':34.58747199999925}]
    existing=ns['add_evaluation_storage_recovery'](existing,normalizer)
    check(len(existing['cost_records'])==7 and existing['category_observations']['old_nested']['primary_invocations']==0,'Existing worker inventory is reclassified once rather than appended and double-counted')
    fakevalues[fakestatus['path']]['stages']['all_windows']['returncode']=True
    rejects(lambda:ns['add_evaluation_storage_recovery'](blank(),normalizer),'Boolean return code cannot masquerade as actual integer exit0')
    fakevalues[fakestatus['path']]['stages']['all_windows']['returncode']=0
    post=trees[NAMES[1]];ptext=ast.unparse(fn(post,'verify_existing'))
    check("set(runtime['stages']) == {'all_windows'}" in ptext and "type(stage.get('returncode')) is int" in ptext and 'STORAGE_RECOVERY.verify_recovery()' in ptext and 'mixed_evaluation_lifecycle' in ptext,'Postprocessing requires actual E13 Windows exit0 plus exact old E12 recovery lifecycle')
    check(ptext.index('E.verify_gate(')<ptext.index('manifest = S.read(') and "terminal.get('status') == 'failed'" in ptext,'Actual CPU17 gate remains before probability verification; original Q14 failure retained')
    resource=trees[NAMES[2]];rtext=ast.unparse(fn(resource,'lifecycle_evidence'))
    lifecycle_fn=fn(resource,'lifecycle_evidence')
    generated=[ast.parse(n.args[0].value) for n in ast.walk(lifecycle_fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='ast' and n.func.attr=='parse' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and n.args[0].value.startswith('return ')]
    require(len(generated)==1 and len(generated[0].body)==1 and isinstance(generated[0].body[0],ast.Return) and isinstance(generated[0].body[0].value,ast.Dict),'Exact generated lifecycle return required')
    returns=generated[0].body[0].value
    mixed=[v for k,v in zip(returns.keys,returns.values) if isinstance(k,ast.Constant) and k.value=='mixed_evaluation_lifecycle']
    loops=[n for n in ast.walk(lifecycle_fn) if isinstance(n,ast.For) and dump(n.iter)==dump(ast.parse("row['cpu_checkpoint_audit']['checkpoints']",mode='eval').body)]
    require(len(loops)==1 and isinstance(loops[0].target,ast.Name),'Exact same-byte checkpoint loop required')
    target=loops[0].target.id
    same_bytes=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='checked' and len(n.args)==1 and isinstance(n.args[0],ast.Name) and n.args[0].id==target for n in ast.walk(loops[0]))
    check(len(mixed)==1 and isinstance(mixed[0],ast.Name) and mixed[0].id=='mixed_evaluation' and same_bytes,'Resource pre-report lifecycle includes mixed evaluation and same-byte405 checks')
    check(ast.unparse(fn(resource,'assert_terminal_still_bound')).startswith('def assert_terminal_still_bound(gate, out):\n    P.Q.check_pause()'),'Pause check remains first at every resource boundary')
    check(all(repr(n) in ast.unparse(fn(resource,'source_review')) for n in NAMES),'Resource admission requires exact ordered full ten-source review')
    final=trees[NAMES[3]];rrecord=ast.unparse(fn(final,'verified_resource_record'))
    check("proof['evaluator_execution']['version'] == 13" in rrecord and "evaluate_v13.py" in rrecord and "value['resource_measurement_version'] == 9" in rrecord,'Finalizer expects actual E13 and resource9 source/version')
    for name in ('measurement_plan','bind_mamba_recheck_evidence','append_optional_qualification_v2_cost','append_boundary_control_evidence','verify_phase_coverage','bind_lost_f3_recovery_cost'):
        require(dump(fn(final,name))==dump(fn(tree(A/OLDNAMES[3]),name)),'Original scientific accounting changed: '+name)
    check(True,'Original twelve-probe/scientific/old-loss functions remain unchanged')
    wide=trees[NAMES[4]];at=ast.unparse(fn(wide,'audit'))
    check('D.legacy_check_fit = CPU_BASE.load_wide().check_fit' in at and "len(result['requirements']) == 30" in at,'Reviewed CPU17 original-platform strict math and all30 requirements retained')
    mixed={'failed_E12_Windows_worker':H['WINDOWS_WORKER'],'metrics_reuse_comparison':H['METRICS_REUSE_CHECK']}
    primary_expected=[mixed['failed_E12_Windows_worker'],mixed['metrics_reuse_comparison'],ns['STORAGE_REPRO']]
    for t,name,variable in ((wide,'final_cost_audit','storage'),(trees[NAMES[5]],'verify_final_pair','accounting')):
        body=fn(t,name).body
        index=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='expected_primary')
        snippet=ast.fix_missing_locations(ast.Module(body=copy.deepcopy(body[index:index+2]),type_ignores=[]))
        code=compile(snippet,'<independent exact three-scope guard>','exec')
        env={'mixed':mixed,variable:{'primary_artifacts':primary_expected},'require':require}
        exec(code,env)
        check(True,'Exact three real primary scopes accepted: '+name)
        for bad,label in (([],'empty'),(primary_expected[:2],'missing'),([primary_expected[0]]*3,'duplicate'),([*primary_expected[:2],{'path':'wrong','sha256':'wrong'}],'substituted')):
            env[variable]={'primary_artifacts':bad}
            rejects(lambda:exec(code,env),'Exact scope guard rejects '+label+': '+name)
    for stem,names in [('pause_memory_resume_cost_companion',('verify_loss_chain',)),('system_standby_cost_companion',('parse_events','observe','verify_native','verify_owner_snapshot','verify_attempt','timer_row')),('audit_system_standby_final_delivery',('verify_events','formal_subset','verify_native_observation','verify_scope_closure')),('audit_memory_pause_final_delivery',('recorded_formal_subset','verify_resume','verify_three_loss_companion'))]:
        old=tree(A/(stem+'_v3.py'));new=tree(A/(stem+'_v4.py'))
        for name in names:require(dump(fn(old,name))==dump(fn(new,name)),'Historical/event invariant changed: '+name)
        check(True,'Original loss/unknown timer/full-event-range gate preserved: '+stem)
    for name in ('audit_memory_pause_final_delivery_v4.py','audit_system_standby_final_delivery_v4.py'):
        check(any(isinstance(n,ast.If) and dump(n.test)==dump(ast.parse('args.require_complete and not args.run',mode='eval').body) and 'parser.error' in ast.unparse(n) for n in ast.walk(fn(trees[name],'main'))),'Strict require-complete cannot silently return preparation: '+name)
    for d in sources:checked(d)
    value={'status':'final_postprocessing_sources_reviewed','passed':True,'reviewer_source':desc(SOURCE),'reviewed_sources':sources,'training_registration':prior['training_registration'],'checks':CHECKS,'author_contract':desc(contract_path),'prior_final_compatibility_review':desc(prior_path),'actual_E13_source_review':desc(e_review),'actual_E13_recovery_closure':closurebinding,'process_actions':0,'model_operations':0,'Tensor_reads':0,'GPU_operations':0,'probability_payload_reads':0,'actual_collectors_or_final_gates_called':False,'training194_sources_preserved':True,'scope':'Minimal source/JSON and synthetic final cost accounting; actual E13 completion, twelve resources, cost, semantic review, thirty requirements and additional final delivery gates remain required.'}
    target=A/('independent_E13_final_compatibility_source_review_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({'artifact':desc(target),'checks':len(CHECKS)}))

