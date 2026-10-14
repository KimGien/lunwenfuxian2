"""Repair and verify unregistered compatibility drafts, with no live operations."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import types

WORK=Path('C:/Users/jin  qiyan/Desktop/lunwen2')
MOD=WORK/'tridep_ccml/eeg_all_candidates'
A=WORK/'output/analysis/eeg_all_candidates_v1_20261008'
STAMP=str(time.time_ns())
OLD_PREP=A/'recovery3_repair_chain_preparation_1791769863748149700.json'
def require(ok,msg):
    if not ok:raise ValueError(msg)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def desc(p):return {'path':p.relative_to(WORK).as_posix(),'sha256':digest(p)}
prep=json.loads(OLD_PREP.read_text()); paths=[WORK/r['path'] for r in prep['new_sources']]
history=WORK/'output/analysis/eeg_all_candidates_v1_20261008/new_repair_source_preparation_history/recovery3_contract_initial_failure_1791770064868596700'
q=MOD/'qualify_deterministic_repairs_v4.py';c=A/'cost_aggregation_v10.py'
checks={}
def check(name,value):require(value,name);checks[name]=True
def tree(p):return ast.parse(p.read_text())
def fn(p,name):return next(n for n in tree(p).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
for p in paths:compile(p.read_text(),str(p),'exec')
check('all_new_sources_compile',True)
for r in prep['previous_sources']:check('old_bytes_preserved:'+r['path'],digest(WORK/r['path'])==r['sha256'])

# The actual numerical worker, model-builder and repair call plan are bytecode
# equivalent after normalizing only the qualifier module-name strings.
class Normalize(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):node.value=node.value.replace('qualify_deterministic_repairs_v4','qualify_deterministic_repairs_v3')
        return node
for name in ('isolated_builder','runtime_worker'):
    original=fn(MOD/'qualify_deterministic_repairs_v3.py',name)
    current=Normalize().visit(copy.deepcopy(fn(q,name)))
    check('original_numerical_qualification_body:'+name,dump(original)==dump(current))

# Pure fake boundary tests must reject partial, held, wrong-lineage and aliased
# observations. No real process inventory, checkpoint or GPU is consulted.
boundary=fn(q,'check_boundary')
with tempfile.TemporaryDirectory(prefix='eeg_recovery3_boundary_') as tmp:
    tw=Path(tmp);out=tw/'experiment';queue=out/'continuation_queue_v5';queue.mkdir(parents=True)
    expected={'path':'manifest3','sha256':'653df2183b1a9cdcd74755f69629d2cf89ca3de8a6b7e9bf26b7df0bfcabbaf4'}
    source={'path':'source3','sha256':'entry3'};closure={'path':'closure3','sha256':'closure3'};lsource={'path':'loss3','sha256':'loss3'}
    fitlist=[{'fit_id':'c/seed_11/fold_'+str(i),'condition':'c'} for i in range(330)]
    base_status={'status':'required_work_remaining','execution_version':5,'recovery_execution_version':3,
                 'recovery_registration':expected,'recovery_execution_source':source,'recovery_closure':closure}
    state={}
    def fake_desc(path):
        if path.name=='recovery_registration_v3.json':return expected
        if path.name=='recover_q5_after_process_loss_v3.py':return source
        if path.name=='lost_f3_closure_v3.py':return lsource
        if path.name=='lost_f3_v3_1791767849813215500.json':return closure
        raise AssertionError('Unexpected descriptor '+str(path))
    def fake_read(path):
        if path==queue/'status.json':return state['status']
        if path==out/'registration.json':return {'fits':state.get('fits',fitlist)}
        if path=='bound_closure':return state['closure']
        return state.get('fit_bad',{}).get(str(path),{'status':'training_complete','fit_id':path.parent.relative_to(out/'runs').as_posix()})
    namespace={'QUEUE':queue,'OUT':out,'WORK':tw,'CONDITIONS':('repair_a','repair_b'),
               'MANIFEST_SHA':expected['sha256'],'require':require,'read':fake_read,'desc':fake_desc,'checked':lambda x:'bound_closure'}
    exec(compile(ast.Module(body=[boundary],type_ignores=[]),str(q),'exec'),namespace)
    def reset():state.clear();state.update(status=copy.deepcopy(base_status),closure={'recovery_registration':expected,'closure_source':source})
    reset();namespace['check_boundary']();check('actual330_released_V3_boundary_accepted',True)
    def reject(label,mutate):
        reset();mutate()
        try:namespace['check_boundary']()
        except (ValueError,KeyError):check(label,True)
        else:raise AssertionError('Accepted invalid boundary '+label)
    reject('partial_running_terminal_rejected',lambda:state['status'].update(status='running'))
    reject('old_recovery2_terminal_rejected',lambda:state['status'].update(recovery_execution_version=2))
    reject('canonical_manifest_descriptor_required',lambda:state['status'].update(recovery_registration={'path':'wrong','sha256':expected['sha256']}))
    reject('closure_loss3_binding_required',lambda:state['closure'].update(recovery_registration={'path':'wrong','sha256':'wrong'}))
    reject('closure_source_required',lambda:state['closure'].update(closure_source=lsource))
    reject('missing_nonrepair_fit_rejected',lambda:state.update(fits=fitlist[:-1]))
    badpath=str(out/'runs'/fitlist[-1]['fit_id']/'status.json')
    reject('incomplete_last_nonrepair_fit_rejected',lambda:state.update(fit_bad={badpath:{'status':'training','fit_id':fitlist[-1]['fit_id']}}))
    reject('aliased_nonrepair_fit_rejected',lambda:state.update(fit_bad={badpath:{'status':'training_complete','fit_id':'other'}}))
    reset();(queue/'owner.json').write_text('{}')
    try:namespace['check_boundary']()
    except ValueError:check('held_live_queue_rejected_before_observation',True)
    else:raise AssertionError('Held owner accepted')

# New observer5 is routed through its source/qualification verifier, never the
# original observer3 route. Old traces retain their existing interpretation.
cost_branch=fn(c,'buffered_observation');seen=[]
ns={'V':types.SimpleNamespace(verify_observer=lambda *args:seen.append('new5') or {'actual_execution':{}})}
exec(compile(ast.Module(body=[cost_branch],type_ignores=[]),str(c),'exec'),ns)
old=lambda *args:seen.append('old') or 'original'
check('new_observer5_cost_route',ns['buffered_observation'](old,{'condition':'repair_a','seed':11,'fold':1,'implementation':{'version':5},'flush_sequence':0,'last_flush_reason':'fake','counter_semantics':{},'durable_counter_snapshot':{}},{})['execution_acceleration']=={})
check('new_observer5_uses_verifier_once',seen==['new5'])
seen.clear();check('original_observer3_cost_route',ns['buffered_observation'](old,{'implementation':{'version':3}},{})=='original' and seen==['old'])
body=fn(c,'collect')
calls=[ast.unparse(n.func) for n in ast.walk(body) if isinstance(n,ast.Call)]
check('Cost9_augmentation_exactly_once',calls.count('previous.augment')==1)
check('Cost9_collect_never_reaugmented','previous.collect' not in calls and 'old.collect' in calls)
check('mixed_CPU9_delegates_all_three_loss_CPU8',"audit_completed_finetune_checkpoints_v8.py" in (A/'audit_completed_finetune_checkpoints_v9.py').read_text())
check('new_E5_requires_schema9_full135405',"cpu.get('schema_version') == 9" in (MOD/'evaluate_v5.py').read_text())
check('original_sampler_budgets_not_edited',"v2._FIT(condition, seed, fold, resume=resume, micro_subjects=1, window_chunk=8)" in (MOD/'finetune_v5.py').read_text())

sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import continuation_queue_v7 as queue7
from tridep_ccml.eeg_all_candidates import repair_observation_v2 as obs
obs.source_check()
records=queue7.sources();prior=json.loads((WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v3.json').read_text())['sources']
check('all88_frozen_descriptors_inherited',len(prior)==88 and all(r in records for r in prior))
check('registered_candidate_sources_unique',len(records)==len({r['path'] for r in records}))
check('actual_new_sources_and_E5_are_bound',all(desc(p) in records for p in paths if p.name not in ('repair_postprocessing_gate_v2.py','review_actual_repair_engineering_v2.py')))
for item in records:check('bound_source_bytes:'+item['path'],digest(WORK/item['path'])==item['sha256'])
report={'schema':'recovery3_repair_chain_contract_tests_v1','status':'prepared_contracts_verified','passed':True,
        'source':{'absolute_path':str(Path(__file__).resolve()),'sha256':digest(Path(__file__))},'checks':checks,
        'current_sources':[desc(p) for p in paths],'candidate_execution_sources':records,'source_count':len(records),
        'preserved_first_draft':history.relative_to(WORK).as_posix(),'captured_ns':time.time_ns(),
        'independent_review_complete':False,'actual_GPU_qualification_complete':False,'actual_new_registration_created':False,
        'actual_CPU9_audit_started':False,'actual_cost10_collection_started':False,'outer_probabilities_read':False,
        'checkpoint_tensor_loads':0,'model_calls':0,'GPU_queries':0,'all_plan_complete':False,
        'source_byte_hashing_performed':True,'full_final_postprocessing_compatibility_complete':False}
target=A/('recovery3_repair_chain_contract_tests_'+str(report['captured_ns'])+'.json')
with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'artifact':desc(target),'checks':len(checks),'candidate_execution_sources':len(records),'actual_new_registration_created':False}))
