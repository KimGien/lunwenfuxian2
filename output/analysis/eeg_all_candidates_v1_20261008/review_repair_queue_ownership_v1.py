"""Source-only independent queue/control review; not actual runtime evidence."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
WORK=Path(__file__).resolve().parents[3]
MOD=WORK/'tridep_ccml/eeg_all_candidates';A=WORK/'output/analysis/eeg_all_candidates_v1_20261008'
checks={}
def require(value,name):
    if not value:raise ValueError(name)
    checks[name]=True
def desc(path):
    path=Path(path)
    return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def methods(path):
    text=Path(path).read_text(encoding='utf-8')
    tree=ast.parse(text)
    return text,{n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
ct,c=methods(MOD/'continuation_queue_v6.py')
ot,o=methods(MOD/'repair_runtime_owner_v1.py')
wt,w=methods(MOD/'repair_owned_worker_v1.py')
require('len(others)==330' in c['original_others_complete'] and 'R._validate_complete(' in c['original_others_complete'],'actual_original_other330_verified')
require(c['register'].index('Q.check_boundary()')<c['register'].index('create(ROOT/'), 'healthy_Q5_blocks_registration_before_writes')
require("'new_repair_chain_sources_reviewed'" in c['register'] and "review.get('reviewed_sources')==all_sources" in c['register'],'complete_independent_source_review_required')
require("condition in D.REPAIR_CONDITIONS" in c['register'] and "status.json')['status']=='blocked'" in c['register'],'only_original_blocked15_groups_registered')
require("D.qualification(" in c['register'] and 'review_engineering(' in c['register'],'actual_GPU_and_independent_report_review_required')
require("'registration_intent.json'" in c['register'] and ".open('x'" in c['create'],'permanent_exclusive_registration_no_blind_retry')
require('repair_runtime_owner_v1.py' in c['sources'] and 'repair_owned_worker_v1.py' in c['sources'],'runtime_wrapper_sources_bound')
require('Q.admit_host(inventory,roles)' in c['acquire'] and 'Q.linux_inventory()==[]' in c['acquire'] and 'Q.gpu_empty()' in c['acquire'],'fresh_native_Linux_GPU_admission')
require(c['child'].index('process=subprocess.Popen(')<c['child'].index("owner['child_native_identity']=native"),'actual_child_birth_published')
require('process.wait()' in c['child'] and "child_pid=None" in c['child'],'sequential_actual_child_return')
require('creationflags=0x08000000' in c['child'],'hidden_Windows_worker_launch')
require('time.monotonic()+20' in o['enter'] and "if owner.get('child_native_identity')" in o['enter'],'bounded_worker_wait_before_GPU_runtime')
require("x==role" in o['validate_native'] and "launch==owner['child_native_identity']" in o['validate_native'],'complete_native_birth_command_parent_verified')
require("actual['parent']==launch['pid']" in o['validate_native'] and "'wsl.exe' in launch['command'].lower()" in o['validate_native'],'native_and_WSL_worker_tree_verified')
require("child_token')==admission.get('token')" in o['validate_native'] and "owner.get('execution_version')==6" in o['validate_native'],'actual_current_Q6_child_nonce_verified')
require('Another experimental process remains' in o['validate_native'],'second_experimental_worker_rejected')
require(w['run'].index('O.enter(')<w['run'].index("importlib.import_module('.finetune_v4'"),'live_source_admission_precedes_formal_import')
require('O.require_live(value,identity)' in w['run'] and 'module.fit=original' in w['run'],'per_fit_live_check_and_wrapper_restoration')
require('timer_is_nested_supplemental_not_F4_primary_cost' in w['run'] and 'inclusive_parent_seconds_primary_cost' in c['child'],'nested_timer_not_added_to_F4_primary')
require("get('training_complete')==360" in c['cpu_gate'] and "checkpoint_loads']==405" in c['cpu_gate'] and "verified_completed_finetune_fits']==135" in c['cpu_gate'],'actual360_and_CPU7_135405_no_override')
require('test_evaluation_frozen=True,all_plan_complete=False' in c['run'] and 'evaluation_pending' in c['run'],'training_CPU_only_not_total_plan_complete')
require("current.get('child_pid') is None" in c['run'],'unverified_child_keeps_owner')
require(all(name not in ct+ot+wt for name in ('NtSuspend','NtResume','terminate(','kill(')),'no_unrelated_process_or_training_control')
manifest=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v2.json'
m=json.loads(manifest.read_text(encoding='utf-8-sig'))
require(len(m['sources'])==84,'original84_count')
for item in m['sources']:require(desc(WORK/item['path'])==item,'original_source:'+item['path'])
defaults={}
for module in ('continuation_queue_v6','repair_owned_worker_v1'):
    result=subprocess.run([str(WORK/'S00/.venv/Scripts/python.exe'),'-B','-m','tridep_ccml.eeg_all_candidates.'+module],
                          cwd=WORK,capture_output=True,text=True,check=True)
    defaults[module]=json.loads(result.stdout)
    require(defaults[module]['model_calls']==defaults[module]['GPU_queries']==0,'actual_default_zero:'+module)
path=A/('independent_repair_queue_ownership_review_'+str(time.time_ns())+'.json')
value={'status':'queue_control_preparation_reviewed','reviewer_source':desc(Path(__file__)),
       'sources':[desc(MOD/x) for x in ('continuation_queue_v6.py','repair_runtime_owner_v1.py','repair_owned_worker_v1.py')],
       'checks':checks,'actual_defaults':defaults,'source_byte_hashing_performed':True,
       'actual_native_or_GPU_queries':0,'Tensor_imports':0,'author_models_or_formal_fits':0,
       'actual_registration_created':False,'full_new_source_chain_review_passed':False,
       'remaining':['Actual GPU qualification and independent real-report review.',
                    'CPU7/E4/cost8 and complete mixed old/new final gates.',
                    'Real Windows/WSL launch, owner/drain/cleanup and full-resume checks at exclusive boundary.',
                    'Future direct F4 CLI is not an owned queue entry; production must use bound Q6 worker.']}
with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(desc(path)))
