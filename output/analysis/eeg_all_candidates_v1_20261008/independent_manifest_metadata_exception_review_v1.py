"""Narrow independent review of fixed metadata-manifest exception, not final audit."""
from __future__ import annotations
import ast,hashlib,json,time
from pathlib import Path
SOURCE=Path(__file__).resolve();A=SOURCE.parent;WORK=SOURCE.parents[3]
PAIRS=[('pause_memory_resume_cost_companion_v4.py','pause_memory_resume_cost_companion_v5.py','0f4bd7219f6669db5eac0c562103ede6c4aa695d709a3459c10311165fdcaf3b'),
 ('audit_memory_pause_final_delivery_v4.py','audit_memory_pause_final_delivery_v5.py','554457591941de6bdfc75d8c265e81eae18118568fbddc8e9ab2dcdcf0fd7784'),
 ('system_standby_cost_companion_v4.py','system_standby_cost_companion_v5.py','444299fd9fde6091945928c05fb43c3f208068ad3ebfd63284ab918ac5fae5b3'),
 ('audit_system_standby_final_delivery_v5.py','audit_system_standby_final_delivery_v6.py','771b45657ed184ff3d3402316fabcc0683afdf8e3b17827badd31e702205b1af')]
def require(ok,why):
    if not ok:raise ValueError(why)
def desc(p):
    p=Path(p).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def funcs(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
class NormalizeSchema(ast.NodeTransformer):
    mapping={'pause_memory_resume_cost_companion_v5':'pause_memory_resume_cost_companion_v4','memory_pause_final_delivery_gate_v5':'memory_pause_final_delivery_gate_v4','system_standby_cost_companion_v5':'system_standby_cost_companion_v4','system_standby_final_delivery_gate_v6':'system_standby_final_delivery_gate_v5'}
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value in self.mapping:return ast.copy_location(ast.Constant(self.mapping[node.value]),node)
        return node
def run():
    checks=[];loaded=[];sources=[];unchanged=[]
    def check(name,ok):require(ok,name);checks.append({'name':name,'passed':True})
    def rejected(name,call):
        try:call()
        except ValueError:check(name,True)
        else:raise ValueError('Negative accepted: '+name)
    for old,new,sha in PAIRS:
        p=A/new;d=desc(p);check(new+' exact source',d['sha256']==sha);sources.append(d)
        text=p.read_text(encoding='utf-8');base=funcs((A/old).read_text(encoding='utf-8'));current=funcs(text)
        exceptions={'Reader','collect','main'}
        preserved=[]
        for name,node in base.items():
            if name in exceptions:continue
            # Reparse before normalizing, leaving namespace source unmodified.
            candidate=NormalizeSchema().visit(funcs(text)[name])
            require(ast.dump(node,include_attributes=False)==ast.dump(candidate,include_attributes=False),'Unexpected old function change: '+new+':'+name);preserved.append(name)
        check(new+' all prior protection functions unchanged after exact schema mapping',bool(preserved));unchanged.append({'source':d,'functions':preserved})
        ns={'__file__':str(p.resolve()),'__name__':'independent_metadata_exception_source_no_main'};exec(compile(text,str(p),'exec'),ns);loaded.append(ns)
    reference=loaded[0]['MAMBA_MANIFEST'];manifest_path=WORK/reference['path'];raw=manifest_path.read_bytes()
    check('Actual fixed manifest metadata SHA',hashlib.sha256(raw).hexdigest()==reference['sha256']=='9d3a15c32609ae74f7db91f1ad9105f616a0135dfa51de5a0a484645a71d5661')
    validators=[]
    for index,ns in enumerate(loaded):
        name=PAIRS[index][1];check(name+' exact singleton manifest descriptor',ns['MAMBA_MANIFEST']==reference)
        validators.append(ast.dump(funcs((A/name).read_text(encoding='utf-8'))['validate_fixed_manifest_metadata'],include_attributes=False))
        value=ns['validate_fixed_manifest_metadata'](manifest_path,raw)
        check(name+' actual15 descriptor metadata passes, no payload arrays',value['supervised_fits_evaluated']==15 and value['primary_oof_predictions']==111 and value['distinct_subjects']==37 and len(value['evaluation_fit_artifacts'])==15 and value['evaluator_execution']['version']==12)
        rejected(name+' same path with altered byte rejected',lambda ns=ns:ns['validate_fixed_manifest_metadata'](manifest_path,raw+b' '))
        reader=ns['Reader']()
        if hasattr(reader,'snapshot'):
            observed,descriptor=reader.snapshot(manifest_path);check(name+' actual Reader uses same fixed metadata',descriptor==reference and observed==value)
        else:check(name+' actual Reader uses same fixed metadata',reader.read(reference)==value)
        wrong=manifest_path.with_name('other_predictions_manifest.json')
        if hasattr(reader,'path'):rejected(name+' other prediction filename refused before reading',lambda reader=reader:reader.path(wrong))
        else:rejected(name+' other prediction filename refused before reading',lambda reader=reader:reader.snapshot(wrong))
        check(name+' resolved path cannot admit sibling manifest',ns['fixed_manifest_path'](wrong) is False and ns['fixed_manifest_path'](manifest_path.parent/'../CodeBrain-lastblock-finetune/predictions_manifest.json') is False)
    check('All four metadata validators use identical AST',len(set(validators))==1)
    pause,memory,standby,system=loaded
    check('Updated four-source pins close exactly',memory['PINNED']['pause']==sources[0] and standby['PAUSE_SOURCE']==sources[0] and system['PINNED']['pause']==sources[0] and system['MEMORY_SOURCE']==sources[1] and system['STANDBY_SOURCE']==sources[2])
    old_gate=funcs((A/PAIRS[3][0]).read_text(encoding='utf-8'));new_gate=funcs((A/PAIRS[3][1]).read_text(encoding='utf-8'))
    check('Full actual scope gate is bytecode-source AST unchanged',ast.dump(old_gate['verify_scope_closure'],include_attributes=False)==ast.dump(new_gate['verify_scope_closure'],include_attributes=False))
    scope_path=A/'independent_final_system_low_power_scope_review_v2_1791881863754362300.json';scope_desc=desc(scope_path);require(scope_desc['sha256']=='3adbfdf666faaaebe1999069297088756685aaa7e0cffc1b8c673ce600bf5755','Actual scope review changed')
    scope=json.loads(scope_path.read_text(encoding='utf-8'));check('Final scope v2 accepted shape is exact and full inventory bound',scope['schema']=='independent_final_system_low_power_scope_review_v2' and scope['passed'] is True and scope['final_scope_closed'] is True and scope['inventory_count']==1240 and scope['unknown_exact_endpoint_count']==513 and len(scope['binding']['formal_complete_artifacts'])==360 and scope['unresolved_scenarios']==[])
    for ns in (memory,system):
        body=ast.unparse(funcs((A/Path(ns['SOURCE']).name).read_text(encoding='utf-8'))['main'])
        check(Path(ns['SOURCE']).name+' strict completion CLI retained',"args.require_complete and (not args.run)" in body and "parser.error('--require-complete requires --run')" in body)
    check('Old Cost19 wide9 G3 and full-scope helper unchanged',all(desc(WORK/d['path'])==d for d in [system['PINNED'][k] for k in ('cost','wide','g3')]+[system['FULL_SCOPE_SOURCE']]))
    check('Candidate four source bytes unchanged after narrow checks',sources==[desc(A/name) for _,name,_ in PAIRS])
    report={'schema':'independent_manifest_metadata_exception_source_review_v1','status':'final_manifest_metadata_exception_sources_reviewed','passed':True,'reviewer':'loss3_independent_review independent agent','reviewer_source':desc(SOURCE),'reviewed_sources':sources,'actual_full_scope_review':scope_desc,'actual_fixed_manifest_metadata':reference,'prior_protection_functions_preserved':unchanged,'checks':checks,'check_count':len(checks),'manual_diff_scope':['Only exact WORK-resolved E12 Mamba metadata manifest path with fixed SHA is exempted from filename rejection.','Validator runs before successful Reader return; exact key set, counts,15 completion descriptors and true E12 source retained.','No numeric probability or OOF payload is opened or followed.','Collect/main changes are output version and scope disclosure; all scientific/cost/loss/scope verification functions remain identical after listed schema renames.'],'operations':{'actual_metadata_manifest_read_only':True,'final_collect_or_audit_run':False,'new_models_Tensor_GPU_native_calls':0,'probability_payload_reads':0,'old_full_scope_or_CPU_audits_reexecuted':0},'actual_execution_pending':True}
    target=A/('independent_manifest_metadata_exception_source_review_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'report':desc(target),'passed':True,'checks':len(checks)}));return report
if __name__=='__main__':run()
