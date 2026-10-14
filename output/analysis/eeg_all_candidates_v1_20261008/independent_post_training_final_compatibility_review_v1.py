"""Minimal independent compatibility review; no actual audit/collect/evaluation."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,time,types
from pathlib import Path
SOURCE=Path(__file__).resolve();A=SOURCE.parent;WORK=SOURCE.parents[3]
M=WORK/'tridep_ccml/eeg_all_candidates'
OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
CHECKS=[]
NAMES=('cost_aggregation_v18.py','repair_postprocessing_gate_v9.py','supplemental_head_resource_measurement_v8.py','finalize_full_plan_cost_v8.py','audit_full_plan_acceptance_v8.py','pause_memory_resume_cost_companion_v3.py','sync_boundary_v3_cost_companion_v3.py','audit_memory_pause_final_delivery_v3.py','system_standby_cost_companion_v3.py','audit_system_standby_final_delivery_v3.py')
def require(ok,message):
    if not ok:raise ValueError(message)
def check(ok,message):require(ok,message);CHECKS.append(message)
def desc(p):
    p=Path(p).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(d):
    p=WORK/d['path'];require(desc(p)==d,'Changed byte binding: '+d['path']);return p
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def tree(p):return ast.parse(Path(p).read_text(encoding='utf-8'))
def node(t,name):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name)
def dump(x):return ast.dump(x,include_attributes=False)
def extract(t,names,ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node(t,n)) for n in names],type_ignores=[])),'<independent synthetic check>','exec'),ns)
def rejects(call,message):
    try:call()
    except (ValueError,KeyError,TypeError):CHECKS.append(message);return
    raise ValueError('Negative case accepted: '+message)
def literal_namespace(t):
    ns={}
    for n in t.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:ns[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    return ns

def main():
    contract_path=A/'post_training_final_compatibility_stdlib_contract_1791833808995520100.json'
    check(desc(contract_path)['sha256']=='4e95e761b9d68cc5e678ab19ab15e8b83096c82789f44a251bd397def3d77659','Exact author17 compatibility evidence retained')
    contract=read(contract_path);sources=[desc(A/n) for n in NAMES]
    check(contract['passed'] is True,'Author preparation passed, distinct from actual finalization')
    report_lists=[v for v in contract.values() if isinstance(v,list) and v and isinstance(v[0],dict) and 'path' in v[0] and 'sha256' in v[0]]
    check(sources in report_lists,'Exact ordered ten current source bytes match frozen author evidence')
    prior=read(A/'independent_post_training_cpu_recovery_source_review_1791833255395495800.json')
    for d in prior['reviewed_sources']:checked(d)
    check(prior['passed'] is True,'Previously reviewed CPU17/recovery/E12 bytes remain unchanged; not rerun')
    registration=read(checked(prior['Q14_registration']))
    for d in registration['sources']:checked(d)
    check(len(registration['sources'])==194,'All194 sealed training sources still original')
    trees={n:tree(A/n) for n in NAMES}
    c=trees[NAMES[0]];old=tree(A/'cost_aggregation_v17.py')
    for name in ('delegate','buffered_observation','category_totals','repair_worker_record','add_repair_ledgers','add_preserved_failure_closure'):
        check(dump(node(c,name)).replace('_Cost18_private_original_Cost9','_Cost17_private_original_Cost9')==dump(node(old,name)),'Original Cost17 function unchanged except private namespace label: '+name)
    collect=ast.unparse(node(c,'collect'))
    check(collect.count('previous.augment(')==1 and collect.count('old.collect(')==1 and collect.count('add_post_training_recovery(')==1,'Original Cost9 augmentation and new accounting each execute exactly once')
    rec=literal_namespace(tree(M/'post_training_cpu_recovery_v1.py'))
    gate_binding={'path':'synthetic/complete.json','sha256':'fake-gate'}
    passed_binding={'path':'synthetic/passed_CPU17.json','sha256':'fake-CPU17'}
    source_binding={'path':'synthetic/source.py','sha256':'fake-source'}
    review_binding={'path':'synthetic/review.json','sha256':'fake-review'}
    gate={'CPU_audit':passed_binding,'source':source_binding,'original_Q14_failed_terminal':rec['Q14_FAILED'],'original_Q14_released_owner':rec['Q14_RELEASED'],'original_child_complete':rec['CHILD_COMPLETE'],'post_training_source_review':review_binding,'post_training_sources':prior['reviewed_sources']}
    values={}
    for d in (rec['FAILED_CPU16'],rec['LINUX_REFERENCE'],rec['WINDOWS_REFERENCE'],rec['Q14_FAILED'],rec['Q14_RELEASED'],rec['CHILD_COMPLETE'],*prior['reviewed_sources']):
        p=checked(d);values[d['path']]=read(p) if p.suffix=='.json' else {}
    values[passed_binding['path']]={'status':'all_finetune_verified','observed_auditor_operations_including_failed_checks':{'cpu_checkpoint_loads_started':405,'cpu_checkpoint_loads_returned':405}}
    values.update({gate_binding['path']:gate,source_binding['path']:{},review_binding['path']:{}})
    known={d['path']:d for d in (rec['FAILED_CPU16'],rec['LINUX_REFERENCE'],rec['WINDOWS_REFERENCE'],rec['Q14_FAILED'],rec['Q14_RELEASED'],rec['CHILD_COMPLETE'],*prior['reviewed_sources'],gate_binding,passed_binding,source_binding,review_binding)}
    def fake_checked(d):require(known[d['path']]==d,'Fake binding mismatch');return d['path']
    def normalizer(data,**kwargs):return {'observed_counters':{},'seconds_observed':None,'seconds_source_field':None,'unknown_counter_fields':[]}
    ns={'require':require,'S':types.SimpleNamespace(desc=lambda _:gate_binding,checked=fake_checked,read=lambda p:copy.deepcopy(values[p])),'RECOVERY':types.SimpleNamespace(ROOT=Path('synthetic'),verify_gate=lambda _:copy.deepcopy(gate),**{k:rec[k] for k in ('FAILED_CPU16','LINUX_REFERENCE','WINDOWS_REFERENCE')})}
    extract(c,['category_totals','add_post_training_recovery'],ns)
    def blank():return {'inputs':[],'cost_records':[],'category_observations':{},'rules':[]}
    result=ns['add_post_training_recovery'](blank(),normalizer)
    check(len(result['cost_records'])==4 and len({r['artifact']['path'] for r in result['cost_records']})==4,'Four independent post-training scopes added once to empty synthetic inventory')
    cpus=[r for r in result['cost_records'] if r['category']=='post_training_CPU_checkpoint_audit']
    check([r['observed_counters']['cpu_checkpoint_loads_returned'] for r in cpus]==[360,405] and all(r['seconds_observed'] is None for r in cpus),'Failed360 and new405 actual count shapes separate; audit elapsed remains unknown')
    captures=[r for r in result['cost_records'] if r['category']=='original_platform_LR_stdlib_capture']
    check([r['seconds_observed'] for r in captures]==[0.9818039079999998,0.039485699991928414],'Both real stdlib capture primary elapsed values preserved exactly')
    existing=blank();existing['cost_records']=[{'artifact':rec['FAILED_CPU16'],'category':'old_supplemental','aggregation_role':'supplemental_potentially_cumulative_test_report','observed_counters':{},'seconds_observed':None}]
    result2=ns['add_post_training_recovery'](existing,normalizer)
    check(len(result2['cost_records'])==4 and result2['category_observations']['old_supplemental']['primary_invocations']==0,'Previously inventoried supplemental row reused rather than appended again')
    ns['add_post_training_recovery'](result2,normalizer)
    check(len(result2['cost_records'])==4,'Repeat augmentation does not duplicate four primary artifacts')
    duplicate=blank();duplicate['cost_records']=[copy.deepcopy(result['cost_records'][0]) for _ in range(2)]
    rejects(lambda:ns['add_post_training_recovery'](duplicate,normalizer),'Duplicate existing actual cost scope rejected')
    wrong=copy.deepcopy(values[passed_binding['path']]);values[passed_binding['path']]['observed_auditor_operations_including_failed_checks']['cpu_checkpoint_loads_returned']=404
    rejects(lambda:ns['add_post_training_recovery'](blank(),normalizer),'Incomplete405 accounting rejected');values[passed_binding['path']]=wrong
    post=trees[NAMES[1]];text=ast.unparse(node(post,'verify_existing'))
    check("gate = E.verify_gate(gate_binding)" in text and "terminal.get('status') == 'failed'" in text and "gate['original_Q14_failed_terminal']" in text,'Post9 retains exact Q14 failure and verifies independent recovered gate')
    check(text.index('E.verify_gate(')<text.index('manifest = S.read('),'No existing outer result reads before recovered gate')
    check("set(runtime['stages']) == {'mamba_linux', 'all_windows'}" in text and "stage['returncode'] == 0" in text and "evaluation_released_owner" in text,'Both actual E12 stages exit0 and released evaluator remain required')
    resource=trees[NAMES[2]]
    sr=ast.unparse(node(resource,'source_review'))
    check(all(repr(n) in sr for n in NAMES) and "value.get('reviewed_sources') == expected" in sr,'Actual resource admission binds all ten independent reviewed sources')
    pauseguard=ast.unparse(node(resource,'assert_terminal_still_bound'))
    check(pauseguard.startswith('def assert_terminal_still_bound(gate, out):\n    P.Q.check_pause()'),'Pause cancellation remains first check at every resource boundary')
    lifecycle=ast.unparse(node(resource,'lifecycle_evidence'))
    check("function.body[:boundaries[0]]" in lifecycle and "for binding in row['cpu_checkpoint_audit']['checkpoints']" in lifecycle,'Resource lifecycle retains full pre-report prefix and same-byte405 checkpoint bindings')
    final=trees[NAMES[3]];finalold=tree(A/'finalize_full_plan_cost_v7.py')
    for name in ('measurement_plan','bind_mamba_recheck_evidence','append_optional_qualification_v2_cost','append_boundary_control_evidence','verify_phase_coverage','bind_lost_f3_recovery_cost'):
        check(dump(node(final,name))==dump(node(finalold,name)),'Scientific/resource/old-failure function unchanged: '+name)
    resource_record=ast.unparse(node(final,'verified_resource_record'))
    check("proof['evaluator_execution']['version'] == 12" in resource_record and "value['resource_measurement_version'] == 8" in resource_record,'Resource record uses actual E12 and resource8, resolving old-version draft blocker')
    wide=trees[NAMES[4]];audit_text=ast.unparse(node(wide,'audit'))
    check('D.legacy_check_fit = CPU_BASE.load_wide().check_fit' in audit_text and audit_text.index('D.legacy_check_fit =')<audit_text.index('D.audit('),'Final wide JSON history checks use reviewed CPU17 strict Linux expectations before full audit')
    original3=node(tree(A/'audit_full_plan_acceptance_v3.py'),'check_fit');original4=copy.deepcopy(node(tree(A/'audit_full_plan_acceptance_v4.py'),'legacy_check_fit'));original4.name=original3.name
    check(dump(original3)==dump(original4),'Wide4 legacy check is same original wide3 function patched by CPU17')
    check("len(result['requirements']) == 30" in audit_text and "CPU_ROWS" in audit_text,'Original30 requirements and actual135 row evidence retained')
    deep=ast.unparse(node(wide,'deep_checkpoint'))
    check("_original_deep(fit, row, contract, metadata)" in deep and "for binding in deep['checkpoints']" in deep and "prior['f2_physical_durable_boundary']" in deep,'NonFT deep checks unchanged;FT proof requires current same bytes and exact physical trace')
    for base,names in [('pause_memory_resume_cost_companion',('verify_loss_chain',)),('system_standby_cost_companion',('parse_events','observe','verify_native','verify_owner_snapshot','verify_attempt','timer_row')),('audit_system_standby_final_delivery',('verify_events','formal_subset','verify_native_observation','verify_scope_closure')),('audit_memory_pause_final_delivery',('recorded_formal_subset','verify_resume','verify_three_loss_companion'))]:
        old=tree(A/(base+'_v2.py'));new=tree(A/(base+'_v3.py'))
        for name in names:require(dump(node(old,name))==dump(node(new,name)),'Historical/fullrange invariant changed: '+name)
        check(True,'Original loss/native/elapsed/full-range semantics unchanged: '+base)
    pair=ast.unparse(node(trees[NAMES[5]],'verify_final_pair'))
    check("terminal.get('status') == 'failed'" in pair and "passed.get('learning_rate_tolerance_used') is False" in pair and "rows[0]['seconds_observed'] is None" in pair and "len(rows) == 1" in pair,'Final companion requires unchanged failure, strictCPU17 and each independent cost once with unknown audit elapsed')
    for name in ('audit_memory_pause_final_delivery_v3.py','audit_system_standby_final_delivery_v3.py'):
        maintext=ast.unparse(node(trees[name],'main'))
        check(any(isinstance(n,ast.If) and dump(n.test)==dump(ast.parse('args.require_complete and not args.run',mode='eval').body) and 'parser.error' in ast.unparse(n) for n in ast.walk(node(trees[name],'main'))),'Strict final completion flag cannot silently return preparation: '+name)
    # All recorded path/SHA dictionary literals, including cross-version pins.
    descriptors=[]
    for t in trees.values():
        for n in ast.walk(t):
            if isinstance(n,ast.Dict):
                try:v=ast.literal_eval(n)
                except (ValueError,TypeError):continue
                if isinstance(v,dict) and set(v)=={'path','sha256'} and isinstance(v['path'],str):descriptors.append(v)
    for d in descriptors:checked(d)
    check(True,'All literal current and historical path/SHA bindings match existing bytes')
    for d in sources:checked(d)
    value={'status':'final_postprocessing_sources_reviewed','passed':True,'reviewer_source':desc(SOURCE),'reviewed_sources':sources,'training_registration':prior['Q14_registration'],'prior_core_independent_review':desc(A/'independent_post_training_cpu_recovery_source_review_1791833255395495800.json'),'author_contract':desc(contract_path),'checks':CHECKS,'process_actions':0,'model_operations':0,'Tensor_loads':0,'GPU_operations':0,'outer_test_reads':0,'actual_cost_collectors_or_final_gates_called':False,'actual_resource_probes_run':False,'original_training194_preserved':True,'scope':'Source/JSON reading and extracted stdlib synthetic accounting only. No actual CPU auditor, post-training gate, evaluation, cost collector, resource probe, full acceptance or companion collector executed.','limitations':['Only compatibility source review; actual CPU17, E12,12 resources,cost,semantic review,all30,final bundles and full-range event evidence remain separately required.','Original unknown lost-parent timers and unmeasured CPU audit durations remain unknown; no fake natural terminal or fabricated active time.']}
    target=A/('independent_post_training_final_compatibility_source_review_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'artifact':desc(target),'passed':True,'checks':len(CHECKS)}))
if __name__=='__main__':main()
