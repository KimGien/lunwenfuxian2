"""Independent source/fixture review of recovery3 repair104; never actual execution."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
W=Path(__file__).resolve().parents[3]
A=Path(__file__).resolve().parent
M=W/'tridep_ccml/eeg_all_candidates'
OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
SOURCE=Path(__file__).resolve()
checks={}
def require(value,message):
    if not value:raise ValueError(message)
def check(label,value):
    require(value,label);checks[label]=True
def reject(label,fn):
    try:fn()
    except (ValueError,KeyError,IndexError,AssertionError):checks[label]=True
    else:raise AssertionError('Unsafe acceptance: '+label)
def desc(path):
    path=Path(path).resolve()
    return {'path':path.relative_to(W).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def checked(binding):
    path=W/binding['path'];require(desc(path)==binding,'Changed bound source '+str(path));return path

def tree(path):return ast.parse(path.read_text(encoding='utf-8'))
def functions(path,names,namespace):
    nodes=[copy.deepcopy(n) for n in tree(path).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require(len(nodes)==len(names),'All requested functions present')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def fixture_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value

q=M/'qualify_deterministic_repairs_v4.py'
# Simulate real module definition order without importing or invoking actual main.
qt=tree(q);ordered=[]
for node in qt.body:
    if isinstance(node,ast.FunctionDef):
        if node.name=='main':
            node=ast.parse("def main():\n    assert callable(verify_original330_complete)\n    observed.append('helper_available_at_real_main')").body[0]
        else:
            node=ast.parse('def '+node.name+'(*args, **kwargs):\n    return None').body[0]
        ordered.append(node)
    elif isinstance(node,ast.If) and '__name__' in ast.unparse(node.test):ordered.append(copy.deepcopy(node))
ns={'__name__':'__main__','observed':[]}
exec(compile(ast.Module(body=ordered,type_ignores=[]),'<independent-module-order>','exec'),ns)
check('real_module_entry_reaches_defined_completion_helper',ns['observed']==['helper_available_at_real_main'])

# Real existing boundary: only JSON, source and finished log bytes are read.
ns={'Path':Path,'hashlib':hashlib,'json':json,'WORK':W,'OUT':OUT,'QUEUE':OUT/'continuation_queue_v5',
    'CONDITIONS':('CodeBrain-lastblock-finetune','EEGMamba-lastblock-finetune'),
    'MANIFEST_SHA':'653df2183b1a9cdcd74755f69629d2cf89ca3de8a6b7e9bf26b7df0bfcabbaf4'}
functions(q,{'require','read','sha','desc','checked','check_boundary'},ns)
actual=ns['check_boundary']()
check('actual_Q5_natural_terminal_and330_immutable_events_match',actual['status']=='required_work_remaining')
original_read=ns['read']
for label,filename,change in [
 ('reject_nonnatural_parent','status.json',lambda v:v.update(status='failed')),
 ('reject_boolean_child_returncode','status.json',lambda v:v.update(returncode=False)),
 ('reject_child_source_loss','status.json',lambda v:v.update(recovery_closure={})),
]:
    def mutated_read(path,label=label,change=change):
        value=original_read(path)
        use=(Path(path)==ns['QUEUE']/'status.json') if label=='reject_nonnatural_parent' else (Path(path).parent.name=='1791767857117600800-LaBraM-native4s-lastblock-finetune-fit')
        if use:value=copy.deepcopy(value);change(value)
        return value
    ns['read']=mutated_read
    reject(label,ns['check_boundary'])
ns['read']=original_read
check('boundary_no_completed_checkpoint_bytes_read_by_this_review',True)

# Exercise shared exact Linux identity code with fixtures, including every field.
L=fixture_module(M/'linux_worker_identity_v1.py','_independent_linux_identity_fixture')
module='tridep_ccml.eeg_all_candidates.repair_owned_worker_v2'
flags={'--token':'review-nonce','--condition':'EEGMamba-lastblock-finetune','--worker-admission-sha256':'a'*64,'--repair-registration-sha256':'r'*64}
argv=['/env/python','-m',module,'--run']
for k,v in flags.items():argv.extend([k,v])
row={'pid':717,'parent':17,'start_ticks':'10009','argv':argv,'executable':'/env/python',
     'parent_identity':{'pid':17,'parent':1,'start_ticks':'10001','argv':['/init'],'executable':'/init'}}
snapshot={'query_succeeded':True,'errors':[],'boot_id':'boot-id','rows':[row]}
bound=L.select_unique(snapshot,module=module,flags=flags)
check('exact_linux_fixture_admitted',L.validate_bound(snapshot,bound,module=module,flags=flags,worker_pid=717)==bound)
for label,change in [
 ('changed_boot',lambda v:v.update(boot_id='reboot')),
 ('changed_birth',lambda v:v['rows'][0].update(start_ticks='10010')),
 ('changed_parent_birth',lambda v:v['rows'][0]['parent_identity'].update(start_ticks='10002')),
 ('changed_executable',lambda v:v['rows'][0].update(executable='/imposter')),
 ('changed_nonce',lambda v:v['rows'][0]['argv'].__setitem__(v['rows'][0]['argv'].index('--token')+1,'other')),
 ('duplicate_worker',lambda v:v['rows'].append(copy.deepcopy(row))),
 ('inventory_error',lambda v:v.update(errors=[{'error':'unknown'}])),
]:
    bad=copy.deepcopy(snapshot);change(bad)
    reject('linux_reject_'+label,lambda bad=bad:L.validate_bound(bad,bound,module=module,flags=flags,worker_pid=717))
reject('linux_reject_wrong_actual_caller',lambda:L.validate_bound(snapshot,bound,module=module,flags=flags,worker_pid=718))

# CPU gate semantic checks; no auditor/model/Tensor is invoked.
e=M/'evaluate_v5.py'
ns=functions(e,{'validate_gate_data'},{'require':require})
gate={'passed':True,'original_training_complete':360,'FT':135,'CPU_loads':405}
cpu={'schema_version':9,'auditor_source':{'path':'CPU9','sha256':'fixture'},'all_135_finetune_checkpoints_verified':True,
 'verified_completed_finetune_fits':135,'training_complete_finetune_observed':135,'registered_finetune_fits':135,
 'checkpoint_loads':405,'errors':[],'deferred_completed_fits':[],'pending_fits':[]}
check('E5_fixture_requires_actual_CPU9_135405',ns['validate_gate_data'](gate,cpu,cpu['auditor_source']) is None)
for field,value in [('schema_version',8),('checkpoint_loads',402),('verified_completed_finetune_fits',134),('pending_fits',['missing']),('deferred_completed_fits',['deferred']),('errors',['bad'])]:
    bad={**cpu,field:value}
    reject('E5_reject_'+field,lambda bad=bad:ns['validate_gate_data'](gate,bad,cpu['auditor_source']))

# Cost10 must not collect/augment three historical losses twice or add nested timers.
cp=A/'cost_aggregation_v10.py';calls=[]
old=types.SimpleNamespace(buffered_observation=lambda *a:None,collect=lambda:(calls.append('base_inventory') or {'rules':[]}),load_v4=lambda:types.SimpleNamespace(normalized='normalizer'))
prior=types.SimpleNamespace(load_v5=lambda:(calls.append('load_original') or old),augment=lambda result,**kw:(calls.append('augment_all_three_losses_once') or result))
ns=functions(cp,{'collect'},{'delegate':lambda:prior,'buffered_observation':lambda *a:None,
  'add_repair_ledgers':lambda result,n:(calls.append('repair_ledgers_once') or result),
  'L':types.SimpleNamespace(verify_lost_f3_attempt='verifier',read_json='reader',__file__='closure3.py'),
  'S':types.SimpleNamespace(desc=lambda x:{'path':str(x),'sha256':'fixture'}),'Path':Path,'SOURCE':'cost10.py','DELEGATE':'cost9.py','V':types.SimpleNamespace(SOURCE='observer2.py')})
result=ns['collect']()
check('cost_inventory_then_three_loss_augment_exactly_once',calls==['load_original','base_inventory','augment_all_three_losses_once','repair_ledgers_once'])
check('cost_exact_unknowns_preserved',result['schema_version']==10 and result['all_plan_wall_seconds_exact'] is None and result['all_plan_physical_optimizer_returns_exact'] is None)
ns=functions(cp,{'category_totals'},{})
result=ns['category_totals']([{'aggregation_role':'primary_observed_invocation','observed_counters':{'back':4},'seconds_observed':2.5},{'aggregation_role':'supplemental_nested_or_unmeasured','observed_counters':{'back':4},'seconds_observed':100.0}])
check('nested_controller_timer_and_back_not_added',result['known_recorded_scope_seconds_subtotal']==2.5 and result['counter_totals']['back']['known_observed_subtotal']==4 and result['counter_totals']['back']['exact_category_total'] is None)

# Sources imports below are stdlib-only; no public execution entry is called.
sys.path.insert(0,str(W))
from tridep_ccml.eeg_all_candidates import continuation_queue_v7 as Q7,repair_observation_v2 as O
O.source_check();records=Q7.sources()
prior88=read(OUT/'continuation_queue_v5/recovery_registration_v3.json')['sources']
check('actual104_unique_sources_inherit_exact_original88',len(records)==104 and len({r['path'] for r in records})==104 and len(prior88)==88 and all(r in records for r in prior88))
for item in records:checked(item)
check('every_current104_source_hash_matches',True)
check('review_imports_no_torch_numpy',not any(x in sys.modules for x in ('torch','numpy')))
prepared=read(A/'recovery3_control_hardening_contract_tests_1791771585598558700.json')
check('frozen161_preparation_matches_current104',prepared['passed'] is True and len(prepared['checks'])==161 and all(prepared['checks'].values()) and prepared['candidate_execution_sources']==records)
checked(prepared['source'])
check('Q7_uses_only_owned_repair_worker',Q7.O.WORKER_MODULE=='tridep_ccml.eeg_all_candidates.repair_owned_worker_v2')

# Concrete manual-source findings and scope are retained, never a GPU success certificate.
report={'schema':'independent_recovery3_repair_chain_sources_review_v2','status':'new_repair_chain_sources_reviewed','passed':True,
 'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'reviewed_sources':records,
 'auxiliary_sources':[desc(A/'repair_postprocessing_gate_v2.py'),desc(A/'review_actual_repair_engineering_v2.py')],
 'checks':checks,'preparation_contracts':desc(A/'recovery3_control_hardening_contract_tests_1791771585598558700.json'),
 'prior_blocked_review':desc(A/'independent_recovery3_repair_chain_blocking_review_1791770460762218300.json'),
 'actual_Q5_terminal':desc(OUT/'continuation_queue_v5/status.json'),
 'actual_original_target_child':desc(OUT/'continuation_queue_v5/attempts/1791767857117600800-LaBraM-native4s-lastblock-finetune-fit/status.json'),
 'reviewed_semantics':[
 'Natural original330 handoff uses true Q5 required_work_remaining, immutable source5/base/recovery3, original job15 and real int child exit0; actual existing JSON boundary independently passed.',
 'Full original330 completion-byte verification is after empty-GPU admission and before models; immutable status/completion descriptors are reused later. This review did not itself hash model/checkpoint bytes.',
 'Shared Linux identity includes boot,pid,parent,start_ticks,argv,executable,parent_identity; controller captures module/nonce uniquely; workers compare own PID and binding at premodel/per-fit or per-call boundaries.',
 'Native admission requires every captured controller role present; atomic exclusive worker claim prevents two workers being admitted to a single attempt. Unknown cleanup retains owner, and original user pause markers remain enforced.',
 'Q7 registration consumes exact104 source review plus both real engineering reports and independent actual engineering reviews; permanent intent prevents blind repeat; original blocked30 fit identities and scientific training constants are retained.',
 'F5 delegates original training/sampler/budgets/selection/resume through private builder and buffered trace; only private deterministic-backward repair is new, Mamba numeric order explicitly different and no original CUDA backward equivalence or speed improvement claimed.',
 'CPU9 delegates original CPU8/CPU5 math with original closure3, plus actual observer5 verifier; E5 requires actual360 plus source-bound CPU9 135/405 before reading any test probability.',
 'Cost10 original inventory and all-three-loss augmentation each run once; repaired formal counts and zero-step engineering are primary once, nested Q7/worker/controller timing supplemental, unknown exact elapsed/tails retained.',
 'Postprocessing gate is source/CPU9/E5 completion verification only. Final resource runner, final cost artifact, semantic review, wide acceptance and all companion delivery gates still require actual completion.'
 ],
 'execution_scope':'Only qualified controller/owned-worker/Q7/E5 entries are approved; low-level F5.fit/run are implementation internals, not independent concurrent launch permission.',
 'model_calls':0,'GPU_queries':0,'process_queries':0,'process_actions':0,'checkpoint_tensor_loads':0,'checkpoint_or_official_weight_bytes_hashed':0,'outer_probability_reads':0,
 'actual_GPU_qualification_complete':False,'actual_new_formal_fits_completed_by_this_review':0,'new_registration_created':False,'final_acceptance_completed':False,'all_plan_complete':False,
 'limitations':['Source/fixture review cannot establish actual GPU runtime success or performance.','Original CUDA backward numerical equivalence is not established; Mamba accumulation order changes are explicit.','EEGMamba external-pretraining participant overlap exclusion remains unknown.'],
 'captured_ns':time.time_ns()}
# Final source check prevents handing off a stale review if another agent edits drafts.
require(records==Q7.sources(),'Candidate sources changed while review ran')
target=A/('independent_recovery3_repair_chain_sources_review_'+str(report['captured_ns'])+'.json')
with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'report':desc(target),'passed':True,'reviewed_sources':len(records),'independent_checks':len(checks),'actual_GPU_qualification_complete':False}))
