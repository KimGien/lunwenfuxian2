"""Independent source, historical JSON, and fake-driver review; never evaluates a model."""
from __future__ import annotations
import ast, copy, hashlib, json, time, types
from pathlib import Path

SOURCE=Path(__file__).resolve(); WORK=SOURCE.parents[3]; A=SOURCE.parent
M=WORK/'tridep_ccml/eeg_all_candidates'; OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
CHECKS=[]
def require(ok,msg):
    if not ok: raise ValueError(msg)
def check(ok,msg): require(ok,msg); CHECKS.append(msg)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def desc(path):
    p=Path(path).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(d):
    p=WORK/d['path'];require(desc(p)==d,'Changed bytes: '+d['path']);return p
def tree(path): return ast.parse(Path(path).read_text(encoding='utf-8'))
def fn(t,name): return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name)
def dump(node): return ast.dump(node,include_attributes=False)
def extract(t,names,ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn(t,n)) for n in names],type_ignores=[])),'<independent extracted source>','exec'),ns)
def reject(call,label):
    try: call()
    except (ValueError,FileNotFoundError): CHECKS.append(label);return
    raise ValueError('Negative case accepted: '+label)

def main():
    helper=M/'evaluation_storage_recovery_v1.py'; evaluator=M/'evaluate_v13.py'
    sources=[desc(helper),desc(evaluator)]
    check([x['sha256'] for x in sources]==['b0583021efaf549004a43250291e7e7e0271eb4d2cf8b6f2658b34b04cabecf9','f7bd1be03785ee5d3db7dd1a934eef546d22f790b9f52911f41a477e9f74b410'],'Frozen two-source bytes')
    H=tree(helper); E=tree(evaluator); OLD=tree(M/'evaluate_v12.py'); E1=tree(M/'evaluate.py')
    constants={}
    for n in H.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
            try: constants[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError): pass
    for name in ('CPU_GATE','E12_SOURCE','E1_SOURCE','STORAGE_SOURCE','PARENT','RELEASED','MAMBA_WORKER','WINDOWS_WORKER','WINDOWS_LOG','MAMBA_MANIFEST','METRICS_REUSE_CHECK'): checked(constants[name])
    check(True,'Exact successful CPU17 gate and E12 source/status/worker/manifest bytes')
    gate=read(checked(constants['CPU_GATE']))
    check(gate['passed'] is True and gate['original_training_complete']==360 and gate['FT']==135 and gate['CPU_loads']==405,'Actual full360/135/405 gate exists')
    S=types.SimpleNamespace(checked=checked,read=read,desc=desc,OUT=OUT,WORK=WORK)
    from tridep_ccml.eeg_all_candidates import protocol as P
    ns={**constants,'S':S,'P':P,'Path':Path,'require':require,'ast':ast,'types':types,'__package__':'tridep_ccml.eeg_all_candidates','SOURCE':helper}
    extract(H,['path_bounds','historical_metadata','sources'],ns)
    evidence=ns['historical_metadata']()
    check(evidence['path_bounds']=={'evaluated_checkpoints':435,'canonical_max_chars':238,'short_temporary_max_chars':218,'future_attempt_ns_digits_upper_bound':20,'all_below_260':True},'All435 registered paths fit native Windows with strict spare attempt digit')
    check(evidence['Windows_failed_header_CPU_loads']==1 and evidence['Windows_failed_model_constructions']==evidence['Windows_failed_forward_calls']==evidence['Windows_failed_optimizer_updates']==0 and evidence['Windows_failed_attempt_elapsed_seconds'] is None and evidence['original_parent_status_preserved']=='running' and evidence['original_parent_elapsed_seconds'] is None,'Failure preserves one header load, zero model work, missing elapsed and unchanged running parent')
    parent=read(checked(constants['PARENT']));linux=read(checked(constants['MAMBA_WORKER']));windows=read(checked(constants['WINDOWS_WORKER']))
    check(parent['stages']['mamba_linux']['returncode']==0 and parent['stages']['all_windows']['returncode']==1 and linux['status']=='complete' and windows['status']=='failed','Actual old child exit outcomes are mixed, never relabelled as old success')
    check(evidence['Windows_inclusive_worker_seconds_nested_only']==windows['inclusive_evaluator_worker_seconds'] and evidence['Mamba_inclusive_worker_seconds_nested_only']==linux['inclusive_evaluator_worker_seconds'],'Old worker inclusive times remain nested, missing attempt time is not replaced')
    metric=read(checked(constants['METRICS_REUSE_CHECK']))
    manifest=read(checked(constants['MAMBA_MANIFEST']))
    check(metric['platform']=='win32' and metric['fit_count']==15 and metric['role_checks']==45 and metric['probability_payload_reads']==555 and metric['differences']==[] and metric['tolerance'] is False,'Actual original Windows metrics report strictly equals all45 cached Linux roles')
    require({json.dumps(x,sort_keys=True) for x in manifest['evaluation_fit_artifacts']}=={json.dumps(x['fit'],sort_keys=True) for x in metric['rows']},'Metrics report fit scope differs')
    for item in metric['rows']:
        fit=read(checked(item['fit']))
        require([x['role'] for x in item['roles']]==['train','val','test'] and all(x['strict_equal'] is True for x in item['roles']),'Role comparison missing')
        require(item['probability_batches']==fit['batches'],'Metric comparison batch descriptors differ from actual completion')
    check(True,'Metrics evidence covers exact original fifteen completions and batch descriptors; payloads not reread')
    captured=[]
    def capture_compile(t,*args,**kwargs):
        captured.append(copy.deepcopy(t));return compile(t,*args,**kwargs)
    storage_ns={**ns,'compile':capture_compile}
    extract(H,['private_storage'],storage_ns)
    private=storage_ns['private_storage']();patched=captured[0]
    writer=fn(patched,'write');assignment=next(x for x in writer.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='temporary')
    check(dump(assignment.value)==dump(ast.parse("path.with_name('.' + uuid.uuid4().hex + '.part')",mode='eval').body),'Private storage uses short same-directory UUID basename')
    assignment.value=ast.parse("path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')",mode='eval').body
    check(dump(patched)==dump(tree(checked(constants['STORAGE_SOURCE']))),'Reversing one temporary assignment restores entire original storage AST, including flush/fsync/replace/retries/immutable guards')
    from tridep_ccml.eeg_all_candidates import storage as ORIGINAL_STORAGE
    check(private.write is not ORIGINAL_STORAGE.write and desc(M/'storage.py')==constants['STORAGE_SOURCE'],'Private writer does not mutate original storage module/source')
    contract_path=A/'evaluation_storage_recovery_stdlib_contract_1791838170182583300.json'
    require(desc(contract_path)['sha256']=='6e03dff9eb0249ecbb4210c0433ca530c914b9462f056018a02aa81b3b809f65','Author contract changed')
    contract=read(contract_path);repro=contract['actual_windows_write_reproduction']
    check(contract['passed'] is True and contract['reviewed_sources']==sources and repro['old_temporary_chars']==275 and repro['short_temporary_chars']==217 and repro['canonical_chars']==237 and repro['old_error']['type']=='FileNotFoundError','Actual Windows safe fixture reproduces long temporary failure and short writer success')
    checked(repro['saved_fixture'])
    check(True,'Actual saved short-writer fixture byte proof retained; no repeated storage repro')
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str): node.value=node.value.replace('E13','E12').replace('evaluation_v13','evaluation_v12').replace('evaluate_v13','evaluate_v12')
            return node
    for name in ('verify_training_execution','admit_worker','validate_gate_data','native_roles'):
        check(dump(Normalize().visit(copy.deepcopy(fn(E,name))))==dump(fn(OLD,name)),'Inherited strict control unchanged: '+name)
    worker=fn(E,'worker');controller=fn(E,'controller')
    check('mode == \'all_windows\'' in ast.unparse(worker.body[0]) and "sys.platform == 'win32'" in ast.unparse(worker.body[0]),'Worker rejects Linux or wrong mode before any gate/model work')
    modes=[n.iter for n in ast.walk(controller) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='mode']
    check(len(modes)==1 and ast.literal_eval(modes[0])==('all_windows',),'Controller owns only Windows continuation')
    check("create(CONTROL / 'intent.json'" in ast.unparse(controller) and "(CONTROL / 'owner.json').unlink()" in ast.unparse(controller),'Exclusive persistent E13 intent and release-on-clean-child semantics retained')
    check('Q.check_pause()' in ast.unparse(worker) and 'admit_worker(Q.host_inventory()' in ast.unparse(worker),'Per-fit pause and exact current native identity rechecks retained')
    calls=[]; mamba=constants['MAMBA'];dest=OUT/'evaluation'/mamba/'seed_11/fold_1/best'
    old_driver=types.SimpleNamespace(_fit_context=lambda *a,**k:('oldmeta','oldplan',{'evaluator_source':constants['E12_SOURCE']},dest),_evaluate_fit=lambda *a,**k:('old_result','old_rows'),_verified_completed_fit=lambda *a,**k:'old_complete',_materialize_predictor=lambda *a,**k:calls.append('forbidden_old'))
    new_driver=types.SimpleNamespace(_fit_context=lambda *a,**k:('newmeta','newplan',{},dest),_evaluate_fit=lambda *a,**k:'new_evaluate',_verified_completed_fit=lambda *a,**k:'new_complete',_materialize_predictor=lambda *a,**k:'new_materialize')
    def fake_module(*a):
        return types.SimpleNamespace(materialize_finetune=lambda *a,**k:None,isolated_driver=lambda:new_driver)
    def fake_compile(text,filename,mode):
        require(text==(M/'evaluate_v2.py').read_text(encoding='utf-8'),'Fake driver must replace exact original E2 compile only')
        return compile('',filename,mode)
    runns={'types':types.SimpleNamespace(ModuleType=fake_module),'Path':Path,'E2':types.SimpleNamespace(__file__=str(M/'evaluate_v2.py')),'OLD_E12':types.SimpleNamespace(isolated_driver=lambda:old_driver),'STORAGE_RECOVERY':types.SimpleNamespace(MAMBA=mamba,private_storage=lambda:private),'SOURCE':evaluator,'ROOT':OUT/'evaluation','__package__':'tridep_ccml.eeg_all_candidates','compile':fake_compile,'execution_provenance':lambda c=None:None,'verify_execution':lambda *a:None,'verify_training_execution':lambda *a:None,'require':require,'P':types.SimpleNamespace(CONDITION_BY_ID={'other':types.SimpleNamespace(policy_key='frozen')})}
    extract(E,['isolated_driver'],runns);driver=runns['isolated_driver']()
    check(driver._fit_context(mamba)[2]['evaluator_source']==constants['E12_SOURCE'] and driver._fit_context('other')[2]=={},'Fake-driver Mamba complete context retains old evaluator source; other context uses current source')
    check(driver._evaluate_fit(mamba,11,1,'best')==('old_result','old_rows') and driver._evaluate_fit('other')=='new_evaluate','Mamba only delegates original completed cache; remaining routes use new evaluator')
    check(driver._verified_completed_fit(None,{'condition':mamba})=='old_complete' and driver._verified_completed_fit(None,{'condition':'other'})=='new_complete','Mixed completed-fit verifier follows actual old/new source')
    reject(lambda:driver._evaluate_fit(mamba,11,1,60),'Mamba non-best evaluation refused')
    reject(lambda:driver._evaluate_fit(mamba,999,1,'best'),'Missing Mamba completed cache refused')
    reject(lambda:driver._materialize_predictor(mamba),'New driver Mamba model construction refused')
    reject(lambda:old_driver._materialize_predictor(mamba),'Old cached driver fallback model construction refused')
    check(driver._materialize_predictor('other')=='new_materialize' and calls==[],'Remaining model routing preserved; fake tests executed no model fallback')
    original_fit=fn(E1,'_evaluate_fit');positions={}
    for i,n in enumerate(original_fit.body):
        text=ast.unparse(n)
        if isinstance(n,ast.If) and 'complete_path.exists()' in text: positions['cache']=i
        if "S.save(attempt_path, ledger" in text:positions.setdefault('ledger',i)
    check(positions['cache']<positions['ledger'],'Original E12 complete fast return precedes attempt creation/materialization')
    ns['sources']=lambda:sources;extract(H,['review_verified'],ns)
    fake_review={'status':'evaluation_storage_recovery_sources_reviewed','passed':True,'reviewed_sources':sources,'CPU_gate':constants['CPU_GATE'],'reviewer_source':desc(SOURCE)}
    original_read=S.read;original_checked=S.checked
    S.read=lambda _:fake_review;S.checked=lambda x:x
    check(ns['review_verified']({}) is fake_review,'Exact source-review admission accepted with matching CPU gate')
    fake_review['reviewed_sources']=list(reversed(sources));reject(lambda:ns['review_verified']({}),'Reordered review source binding refused')
    fake_review['reviewed_sources']=sources;fake_review['CPU_gate']={'wrong':True};reject(lambda:ns['review_verified']({}),'Wrong CPU gate refused')
    S.read=original_read;S.checked=original_checked
    check([desc(helper),desc(evaluator)]==sources,'Candidate bytes unchanged through independent review')
    output={'status':'evaluation_storage_recovery_sources_reviewed','passed':True,'reviewer_source':desc(SOURCE),'reviewed_sources':sources,'CPU_gate':constants['CPU_GATE'],'checks':CHECKS,'author_contract':desc(contract_path),'actual_historical_evidence':evidence,'metrics_reuse_evidence':constants['METRICS_REUSE_CHECK'],'storage_reproduction_evidence':repro,'process_actions':0,'model_operations':0,'Tensor_reads':0,'GPU_operations':0,'probability_payload_reads':0,'review_scope':'Source/JSON byte verification and fake-driver guards only; actual E13 evaluation and final acceptance still pending','old194_and_original_storage_bytes_modified':False,'old_Mamba_probability_packages_rewritten':False,'unknown_failed_attempt_elapsed_seconds':None}
    path=A/('independent_evaluation_storage_recovery_source_review_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'report':desc(path),'checks':len(CHECKS)},ensure_ascii=False))

if __name__=='__main__':main()
