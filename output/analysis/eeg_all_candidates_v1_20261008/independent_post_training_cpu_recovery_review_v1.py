"""Independent source/JSON/stdlib review. Never runs the CPU auditor or evaluator."""
from __future__ import annotations
import ast, copy, hashlib, json, math, time, types
from pathlib import Path

SOURCE=Path(__file__).resolve()
WORK=SOURCE.parents[3]
A=SOURCE.parent
M=WORK/'tridep_ccml/eeg_all_candidates'
OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
CHECKS=[]
def require(ok,message):
    if not ok: raise ValueError(message)
def check(ok,message):
    require(ok,message);CHECKS.append(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def desc(path):
    p=Path(path).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(d):
    p=WORK/d['path'];require(desc(p)==d,'Byte binding changed: '+d['path']);return p
def tree(path):return ast.parse(Path(path).read_text(encoding='utf-8'))
def dump(node):return ast.dump(node,include_attributes=False)
def function(t,name):return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name)
def extract(t,names,ns):
    body=[copy.deepcopy(function(t,name)) for name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'<independent extracted functions>','exec'),ns)
def rejects(call,message):
    try:call()
    except (ValueError,KeyError,TypeError):CHECKS.append(message);return
    raise ValueError('Negative case accepted: '+message)

def main():
    cpu=A/'audit_completed_finetune_checkpoints_v17.py'
    recovery=M/'post_training_cpu_recovery_v1.py'
    evaluator=M/'evaluate_v12.py'
    reviewed=[desc(p) for p in (cpu,recovery,evaluator)]
    linux_path=A/'actual_platform_learning_rate_reference_linux_1791832760707973329.json'
    windows_path=A/'actual_platform_learning_rate_reference_win32_1791832754389436100.json'
    linux=read(linux_path);windows=read(windows_path)
    check(desc(linux_path)['sha256']=='c7109d9ced13c853ecb6de99606b98ab1c2f2b9978535a0a2c614049fb682463' and desc(windows_path)['sha256']=='0ce44e1bbfb87d9e0d3b300e4960bdcafdc726b851b8c76b473d90f8e8ce8b1f','Exact independently captured Linux/Windows references')
    for key in ('source','protocol','Q14_registration','Q14_failed_terminal','failed_CPU16'):
        require(linux[key]==windows[key],'Capture lineage differs');checked(linux[key])
    capture=tree(checked(linux['source']));body=function(capture,'main').body
    table_loop=next(i for i,n in enumerate(body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    history_loop=next(i for i,n in enumerate(body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='seed')
    check(table_loop<history_loop and 'P.learning_rate_multiplier' in ast.unparse(body[table_loop]) and 'history' not in ast.unparse(body[table_loop]),'Reference calculated from original protocol before histories, not copied from outcomes')
    for report in (linux,windows):
        require([x['epoch'] for x in report['reference_table']]==list(range(1,201)),'Whole200 table required')
        for row in report['reference_table']:
            for group in ('head','backbone'):
                for suffix in ('lr','multiplier'):
                    key=group+'_'+suffix
                    require(type(row[key]) is float and math.isfinite(row[key]) and row[key].hex()==row[key+'_hex'],'Float/hex differs')
    check(True,'Both200x2 original tables have finite exact float/hex pairs')
    table_differences=[(l['epoch'],group) for l,w in zip(linux['reference_table'],windows['reference_table']) for group in ('head','backbone') if l[group+'_lr']!=w[group+'_lr']]
    check(table_differences==[(33,'head'),(93,'backbone'),(95,'backbone'),(97,'backbone')],'Full reference preserves unobserved later backbone platform differences too')
    scalars=0;differences=[]
    for lf,wf in zip(linux['fits'],windows['fits']):
        require(lf['fit_id']==wf['fit_id'] and lf['history']==wf['history'] and lf['status']==wf['status'],'Actual fit/source comparison differs')
        status=read(checked(lf['status']));history=read(checked(lf['history']))
        require(status['status']=='training_complete' and status['fit_id']==lf['fit_id'],'Actual completed identity differs')
        require(len(history)==lf['epochs_checked'] and 2*len(history)==lf['scalar_LRs_checked'],'Scope differs')
        for row in history:
            for group in ('head','backbone'):
                value=row[group+'_lr'];wanted=linux['reference_table'][row['epoch']-1][group+'_lr']
                require(value==wanted,'Actual history differs from original Linux arithmetic');scalars+=1
                win=windows['reference_table'][row['epoch']-1][group+'_lr']
                if value!=win:
                    differences.append((lf['fit_id'],row['epoch'],group,value.hex(),win.hex(),abs(value-win)/math.ulp(win)))
    check(scalars==1452 and len(differences)==15 and all(x[1:]==(33,'head','0x1.1d1d6d3ca0cd9p-12','0x1.1d1d6d3ca0cdap-12',1.0) for x in differences),'15 actual histories:1452 strict Linux matches;15 Windows head33 one-ULP differences')
    queue=read(checked(linux['Q14_registration']))
    check(len(queue['sources'])==194,'Original training registration remains194 sources')
    for item in queue['sources']:
        require(Path(item['path']).suffix in ('.py','.json','.md'),'Unexpected binary source');checked(item)
    check(True,'All194 original sealed source bytes unchanged')
    cpu_tree=tree(cpu)
    literals={}
    for n in cpu_tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:literals[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    child=read(checked(literals['CHILD_COMPLETE']))
    check(child['status']=='complete' and type(child['returncode']) is int and child['returncode']==0 and child['linux_worker_identity']['identity']['argv'][0]==linux['runtime']['executable'] and child['linux_worker_identity']['identity']['executable']=='/usr/bin/python3.12','Actual completed original training child uses same Linux3.12 interpreter')
    ns={'require':require,'S':types.SimpleNamespace(read=read,checked=checked),'math':math,**literals}
    extract(cpu_tree,['reference_verified'],ns)
    check(ns['reference_verified']()==linux,'Actual CPU17 reference verifier binds15 F12 metadata/histories and original lineage without Tensor reads')
    compiled=[]
    def capture_compile(t,*args,**kwargs):
        compiled.append(copy.deepcopy(t));return compile(t,*args,**kwargs)
    ns={'require':require,'ast':ast,'copy':copy,'Path':Path,'compile':capture_compile,**literals}
    extract(cpu_tree,['patch_wide'],ns)
    original_path=A/'audit_full_plan_acceptance_v3.py'
    original_tree=tree(original_path)
    wide=types.ModuleType('independent_fake_wide');wide.__file__=str(original_path)
    ns['patch_wide'](wide,linux)
    class Undo(ast.NodeTransformer):
        def visit_IfExp(self,n):
            if isinstance(n.body,ast.Call) and isinstance(n.body.func,ast.Name) and n.body.func.id=='_CPU17_linux_expected':
                self.count+=1;return copy.deepcopy(n.orelse)
            return self.generic_visit(n)
    count=0
    for module in compiled:
        n=module.body[0];undo=Undo();undo.count=0;reverted=undo.visit(copy.deepcopy(n));count+=undo.count
        require(dump(reverted)==dump(function(original_tree,n.name)),'Other original audit AST changed')
    check(count==3 and len(compiled)==2,'Only3 expected expressions change; reversing them restores complete original check_fit/deep_checkpoint AST')
    patched_check=next(x.body[0] for x in compiled if x.body[0].name=='check_fit')
    old_nonft=next(n for n in ast.walk(function(original_tree,'check_fit')) if isinstance(n,ast.Compare) and dump(n.left)==dump(ast.parse("row['lr']",mode='eval').body))
    new_nonft=next(n for n in ast.walk(patched_check) if isinstance(n,ast.Compare) and dump(n.left)==dump(ast.parse("row['lr']",mode='eval').body))
    check(dump(old_nonft)==dump(new_nonft),'NonFT row lr comparison unchanged, resolving double-match initial-draft blocker')
    fit={'condition':literals['MAMBA'],'fit_id':linux['fits'][0]['fit_id']}
    expected=wide.__dict__['_CPU17_linux_expected']
    check(expected(fit,'finetune',33,'head')==float.fromhex('0x1.1d1d6d3ca0cd9p-12'),'Mamba exact native expected value, no tolerance')
    for args in [({**fit,'condition':'CodeBrain-lastblock-finetune'},'finetune',33,'head'),({**fit,'fit_id':'wrong'},'finetune',33,'head'),(fit,'scratch',33,'head'),(fit,'finetune',True,'head'),(fit,'finetune',201,'head'),(fit,'finetune',33,'bad')]:
        rejects(lambda args=args:expected(*args),'Wrong condition/identity/policy/epoch/group rejected: '+str(args[1:]))
    check('result=base.audit()' in ast.unparse(function(cpu_tree,'audit')).replace(' ',''),'CPU17 performs fresh inherited full audit, does not promote CPU16 failed report')
    recovery_tree=tree(recovery);rconstants={}
    for n in recovery_tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:rconstants[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    check(rconstants['CPU17']==desc(cpu),'Recovery source pin closes CPU17 exact bytes')
    failed=read(checked(rconstants['FAILED_CPU16']))
    check(failed['checkpoint_loads']==360 and failed['verified_completed_finetune_fits']==120 and len(failed['errors'])==15 and {x['error'] for x in failed['errors']}=={'Actual head learning-rate schedule differs'},'Actual failed CPU16 360 loads/120 verified/15 LR errors preserved')
    for k in ('Q14_FAILED','Q14_RELEASED','CHILD_COMPLETE','Q14_REG','ORIGINAL_REG'):checked(rconstants[k])
    check(read(checked(rconstants['Q14_FAILED']))['status']=='failed','Original Q14 failed state retained')
    registration=read(checked(rconstants['ORIGINAL_REG']));fits=[x for x in registration['fits'] if x['stage']=='3']
    ns={'require':require};extract(recovery_tree,['validate_CPU_shape'],ns)
    shape={'schema_version':17,'registration':rconstants['ORIGINAL_REG'],'auditor_source':desc(cpu),'registered_total_fits':360,'registered_finetune_fits':135,'training_complete_finetune_observed':135,'verified_completed_finetune_fits':135,'logical_completed_fits_counted':135,'checkpoint_loads':405,'status':'all_finetune_verified','all_135_finetune_checkpoints_verified':True,'partial_snapshot':False,'errors':[],'deferred_completed_fits':[],'pending_fits':[],'completed_rows':[{'fit_id':f['fit_id'],'passed':True,'cpu_checkpoint_audit':{'cpu_tensors_verified':True,'checkpoint_loads':3,'only_map_location_cpu':True,'gpu_operations':0,'model_constructions':0}} for f in fits],'observed_auditor_operations_including_failed_checks':{'cpu_checkpoint_loads_started':405,'cpu_checkpoint_loads_returned':405,'model_construction_attempts_blocked':0,'gpu_initialization_attempts_blocked':0}}
    validate=lambda value:ns['validate_CPU_shape'](value,registration,rconstants['ORIGINAL_REG'],desc(cpu))
    check(len(validate(shape))==135,'Synthetic complete CPU17 shape accepted, without claiming actual audit')
    for key,value in [('schema_version',16),('checkpoint_loads',360),('errors',[{}]),('pending_fits',[{}]),('deferred_completed_fits',[{}])]:
        bad=copy.deepcopy(shape);bad[key]=value;rejects(lambda bad=bad:validate(bad),'Incomplete/old/error/deferred CPU evidence rejected: '+key)
    bad=copy.deepcopy(shape);bad['completed_rows'][1]=bad['completed_rows'][0];rejects(lambda:validate(bad),'Duplicate fit rows rejected')
    create_text=ast.unparse(function(recovery_tree,'create_gate'))
    check(create_text.index('completed_training_verified()')<create_text.index(".open('x'") and create_text.index('validate_CPU17(')<create_text.index(".open('x'") and 'os.fsync' in create_text,'New gate requires actual training/fullCPU before exclusive durable creation')
    complete_text=ast.unparse(function(recovery_tree,'completed_training_verified'))
    check("terminal['status'] == 'failed'" in complete_text and 'R.assert_test_unlocked()' in complete_text and 'D.previous_codebrain_completion()' in complete_text,'New lifecycle retains failedQ14, actualchildexit0/releasedowner, oldQ13 and all360 completion verifier')
    evaluator_tree=tree(evaluator);old_evaluator_tree=tree(M/'evaluate_v11.py')
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str):n.value=n.value.replace('E11','E12').replace('CPU16','CPU17')
            return n
    compared=[]
    for old in old_evaluator_tree.body:
        if not isinstance(old,ast.FunctionDef) or old.name in ('validate_gate_data','verify_gate','execution_provenance'):continue
        new=function(evaluator_tree,old.name)
        require(dump(Normalize().visit(copy.deepcopy(old)))==dump(new),'Evaluation math/control changed: '+old.name);compared.append(old.name)
    check(True,'E12 all'+str(len(compared))+' remaining evaluation/math/identity/control functions AST unchanged except message labels')
    verify=ast.unparse(function(evaluator_tree,'verify_gate'))
    check('RECOVERY.verify_gate(binding)' in verify and "gate['post_training_sources']" in verify,'E12 consumes independently reviewed recovered fullCPU gate, not fake Q14 success')
    for name in ('controller','worker'):
        code=ast.unparse(function(evaluator_tree,name));require(code.index('verify_gate(')<code.index('Q.host_inventory()'),'Evaluator operations precede true full gate')
    check(True,'Both owned evaluation entry points verify full recovered gate before runtime operations')
    for item in reviewed:checked(item)
    value={'status':'post_training_CPU_recovery_sources_reviewed','passed':True,'reviewer_source':desc(SOURCE),'reviewed_sources':reviewed,'Q14_registration':linux['Q14_registration'],'checks':CHECKS,'platform_evidence':[desc(linux_path),desc(windows_path)],'actual_history_LR_scalars_checked':scalars,'full_reference_platform_differences':table_differences,'actual_failed_CPU16':rconstants['FAILED_CPU16'],'actual_new_CPU_audit_performed':False,'actual_CPU17_135_405_still_required':True,'actual_evaluation_performed':False,'training_protocol_or_original194_changed':False,'process_actions':0,'model_operations':0,'Tensor_loads':0,'GPU_operations':0,'outer_test_reads':0,'read_scope':'Existing JSON and source bytes only. No checkpoint binary or raw arrays opened. Extracted function AST/synthetic evidence only for proposed audit/gate.','limitations':['Source review does not establish a successful real CPU17 audit, recovered gate, evaluation, resources or final acceptance.','Windows and Linux capture costs and failed CPU16 actual360 loads remain separate real costs; no repeated training is warranted by this arithmetic discrepancy.']}
    path=A/('independent_post_training_cpu_recovery_source_review_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({'artifact':desc(path),'passed':True,'checks':len(CHECKS)}))
if __name__=='__main__':main()
