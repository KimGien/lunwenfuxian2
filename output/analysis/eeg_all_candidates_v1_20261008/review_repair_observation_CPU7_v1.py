"""Independent exact-tail/private-auditor source review; no tensor loads."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
WORK=Path(__file__).resolve().parents[3]
MOD=WORK/'tridep_ccml/eeg_all_candidates';A=WORK/'output/analysis/eeg_all_candidates_v1_20261008'
checks={}
def require(ok,name):
    if not ok:raise ValueError(name)
    checks[name]=True
def desc(path):
    path=Path(path)
    return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
v=MOD/'repair_observation_v1.py';c=A/'audit_completed_finetune_checkpoints_v7.py'
vt=v.read_text(encoding='utf-8');ct=c.read_text(encoding='utf-8')
ast.parse(vt);ast.parse(ct)
require('Counter' not in ct and 'torch.load(' not in ct and 'deep_checkpoint(' not in ct,
        'no_second_tensor_or_counter_arithmetic')
require('result=B.audit()' in ct and "base=private.B" in ct and 'base.SOURCE=SOURCE' in ct,
        'private_unchanged_CPU6_CPU5_three_load_delegate')
require('old_boundary(fit,metadata,inputs)' in ct and "versions=={4}" in ct,
        'old2_3_closure_paths_and_strict_new4_branch')
require('actual_observer4_never_relabelled_as_actual3=True' in ct and "clone['record']['implementation']=X.f2_provenance()" in vt,
        'actual4_proof_separate_from_arithmetic_only_copy')
require("for row in result['attempts']:row['actual_execution_observation']" in vt and "actual_observer_versions=[4]" in vt,
        'actual4_artifacts_sources_versions_retained')
require("record.get('implementation')==F.provenance(execution)" in vt and 'expected=D.load_execution(' in vt,
        'actual_full_repair_registration_before_counter_guard')
clone_scope=vt.split('clone=copy.deepcopy(item)',1)[1].split('normalized.append',1)[0]
require("clone['record']['status']" not in clone_scope and "clone['record']['seconds']" not in clone_scope
        and "clone['record']['finished_at']" not in clone_scope,'no_new_fake_process_endpoint_placeholder')
require("'counter_semantics'" not in ct and 'matches[0]:-1' in vt,
        'exact_original_suffix_extraction_not_guard_reimplementation')
require('tail=copy.deepcopy(function.body[matches[0]:-1])' in vt and 'tail+[return_node]' in vt,
        'all_old_counter_guard_statements_preserved')
require("'status" not in vt.split("clone['record']['implementation']")[1].split('normalized.append')[0],
        'counter_copy_never_changes_actual_status_elapsed_counts')
require('new_CPU_checkpoint_loads_per_completed_fit=3' in ct and 'historical_weight_or_engineering_probe_bytes_may_be_hashed=True' in ct,
        'same_three_load_scope_and_binary_hash_disclosure')
# Independent static equivalence check against original guard AST and actual
# compiled private guard; the compiled code is obtained without any tensors.
import sys
sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import repair_observation_v1 as V
guard=V.counter_guard()
require('record' in guard.__code__.co_varnames and 'fit' in guard.__code__.co_varnames,
        'actual_guard_signature_record_fit')
require('verify_execution_acceleration' not in guard.__code__.co_names,
        'original_actual3_acceptance_not_in_counter_only_guard')
require(all(name in guard.__code__.co_names for name in ('COUNTER_FIELDS','FLUSH_BOUNDARIES','integer','seconds')),
        'original_all_guard_global_helpers_retained')
require('torch' not in sys.modules,'review_no_tensor_import')
manifest=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v2.json'
m=json.loads(manifest.read_text(encoding='utf-8-sig'))
require(len(m['sources'])==84,'original84_count')
for item in m['sources']:require(desc(WORK/item['path'])==item,'original_source:'+item['path'])
result=subprocess.run([str(WORK/'S00/.venv/Scripts/python.exe'),'-B',str(c)],cwd=WORK,capture_output=True,text=True,check=True)
default=json.loads(result.stdout)
require(default['checkpoint_tensor_loads']==default['GPU_calls']==0 and default['actual_CPU_audit_started'] is False,
        'actual_CPU7_default_zero_operation')
queue=MOD/'continuation_queue_v6.py'
require("str(path),'--run']" in queue.read_text(encoding='utf-8'),'actual_Q6_gate_calls_explicit_CPU7_run')
report=A/('independent_repair_observation_CPU7_review_'+str(time.time_ns())+'.json')
value={'status':'preparation_source_reviewed','reviewer_source':desc(Path(__file__)),'sources':[desc(v),desc(c),desc(queue)],
       'checks':checks,'default_result':default,'source_byte_hashing_performed':True,'Tensor_imports':0,
       'checkpoint_tensor_loads':0,'model_calls':0,'GPU_calls':0,'actual_CPU7_audit_started':False,
       'real_observer4_completion_available':False,'full_new_source_chain_review_passed':False,
       'remaining':['Real repair engineering/independent reports and reviewed complete registration.',
                    'E4/cost8 and mixed-source final/companion gates.',
                    'Actual135/405 CPU7 gate only after all new formal fits complete.']}
with report.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(desc(report)))
