"""Independent Cost8 source/default review; no actual cost collection."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
WORK=Path(__file__).resolve().parents[3];A=WORK/'output/analysis/eeg_all_candidates_v1_20261008'
SOURCE=A/'cost_aggregation_v8.py'
text=SOURCE.read_text(encoding='utf-8')
tree=ast.parse(text)
methods={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
checks={}
def require(ok,name):
    if not ok:raise ValueError(name)
    checks[name]=True
def desc(path):
    path=Path(path)
    return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
require(methods['collect'].count('old.collect()')==1 and methods['collect'].count('previous.augment(')==1,
        'original_inventory_and_loss_augmentation_once')
require('V.verify_observer(' in methods['buffered_observation'] and "observer_version" not in methods['buffered_observation'].split("return {**proof",1)[1].split("}",1)[0],
        'actual4_verifier_proof_not_relabelled')
require("value['optimizer_step_started']==value['optimizer_step_returned']==0" in methods['repair_worker_record'],
        'repair_engineering_zero_optimization')
require("primary_engineering_seconds" in methods['repair_worker_record']
        and 'seconds_observed=seconds' in methods['repair_worker_record'],'explicit_worker_time_only_no_endpoint_inference')
require("x['aggregation_role']=='primary_observed_invocation'" in methods['category_totals'],
        'nested_seconds_not_summed')
require("'exact_category_total':None" in methods['category_totals'] and "all_plan_wall_seconds_exact=None" in methods['collect'],
        'historical_exact_time_and_counts_stay_unknown')
require("artifact['path'] in existing" in methods['add_repair_ledgers']
        and "matches[0]['aggregation_role']=='supplemental_nested_or_unmeasured'" in methods['add_repair_ledgers']
        and 'continue' in methods['add_repair_ledgers'],'existing_Q6_parent_reused_not_duplicated')
require("actual2_3_loss_closures_unchanged=True" in methods['collect']
        and 'L.verify_lost_f3_attempt' in methods['collect'],'old_loss_proofs_and_delta_subsets_preserved')
require('result["' not in text and 'assert_test_unlocked' not in text and 'prediction' not in methods['collect'],
        'no_test_gate_or_probability_collection')
require('torch.load(' not in text and 'import torch' not in text,'no_tensor_or_GPU_code')
manifest=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v2.json'
original=json.loads(manifest.read_text(encoding='utf-8-sig'))
require(len(original['sources'])==84,'original84_count')
for item in original['sources']:require(desc(WORK/item['path'])==item,'original_byte:'+item['path'])
result=subprocess.run([str(WORK/'S00/.venv/Scripts/python.exe'),'-B',str(SOURCE)],cwd=WORK,capture_output=True,text=True,check=True)
default=json.loads(result.stdout)
require(default['actual_cost_collection_started'] is False and default['new_output_writes']==0,'actual_default_no_collection')
report=A/('independent_repair_cost8_review_'+str(time.time_ns())+'.json')
body={'status':'preparation_source_reviewed','reviewer_source':desc(Path(__file__)),'source':desc(SOURCE),
      'checks':checks,'actual_default':default,'source_byte_hashing_performed':True,
      'actual_collection_started':False,'checkpoint_tensor_loads':0,'model_calls':0,'GPU_calls':0,
      'final_cost_artifact_or_completion_created':False,'full_new_chain_review_passed':False,
      'remaining':['Actual qualified repair ledgers and formal4 completions.',
                   'Independent source-bound finalizer/wide/companion compatibility.',
                   'Actual final cost semantics review; current source checks are not final acceptance.']}
with report.open('x',encoding='utf-8') as f:json.dump(body,f,indent=2);f.write('\n')
print(json.dumps(desc(report)))
