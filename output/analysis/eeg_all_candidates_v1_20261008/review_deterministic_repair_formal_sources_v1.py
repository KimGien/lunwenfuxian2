"""Independent source-only review of the incomplete new formal chain.

This checks delegation/observer scope and current frozen bytes. It is not a
GPU qualification, actual new registration, final semantic or completion audit.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

WORK=Path(__file__).resolve().parents[3]
A=WORK/'output/analysis/eeg_all_candidates_v1_20261008'
MOD=WORK/'tridep_ccml/eeg_all_candidates'
checks={}
def require(ok,name):
    if not ok:raise ValueError(name)
    checks[name]=True
def desc(path):
    path=Path(path)
    return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
f=MOD/'finetune_v4.py';d=MOD/'deterministic_repair_execution_v1.py';q=MOD/'qualify_deterministic_repairs_v3.py'
text=f.read_text(encoding='utf-8');tree=ast.parse(text)
defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def source(name):return ast.get_source_segment(text,defs[name])
require(not any('torch' in (n.module or '') for n in tree.body if isinstance(n,ast.ImportFrom)),'no_top_level_torch')
require('RepairAttempt(TEMPLATE.BufferedAttempt)' in text,'reuse_original_counter_code')
require(source('RepairAttempt').index("record['implementation']=provenance")<source('RepairAttempt').index('super().flush('),'version4_before_first_durable_flush')
require("'version':4" in source('provenance') and "'backward_graph_privately_changed':True" in source('provenance')
        and "'original_CUDA_backward_equivalence_established':False" in source('provenance'),'new_source_truthful_backward_scope')
require(source('activate').index('v2._BUILD')<source('activate').index('with TEMPLATE.activate('),'capture_sealed_builder_before_wrappers')
require('Q.isolated_builder(v1)' not in source('activate') and 'model,audit=original_builder(' in source('activate'),'no_recursive_current_builder_clone')
require('v1.build_model=previous_builder;S.save=previous_save' in source('activate') and 'finally:_ACTIVE=False' in source('activate'),'exception_restores_runtime_context')
require(source('fit').index('D.load_execution(')<source('fit').index("importlib.import_module('.finetune_v2'"),'actual_complete_qualification_before_tensor_import')
require('Cannot relabel historical completion' in source('fit'),'no_relabelled_Q5_completion')
require('v2._FIT(condition,seed,fold,resume=resume,micro_subjects=1,window_chunk=8)' in source('fit'),'exact_original_fit_sampler_optimizer_stop_resume')
require('v1.subject_update =' not in text and 'set_learning_rates =' not in text and 'POLICY =' not in text,'no_training_math_rule_override')
require('model.forward=trace.observe(' in source('activate') and 'trace.bind_adapter(' in source('activate'),'actual_forward_encoder_observed')
require('actual_finetune_implementation' in source('activate') and 'actual_deterministic_repair_execution' in source('activate'),'metadata_separates_new_observer_and_old_delegate')
dt=d.read_text(encoding='utf-8')
require('Tensor' not in ''.join(ast.get_source_segment(dt,n) for n in ast.parse(dt).body if isinstance(n,ast.ImportFrom)),'contract_checker_no_tensor_import')
require("phase" in dt and 'Repeat' not in dt,'report_structure_explicit')
require('encoder_windows' in dt and "('optimizer_step',0)" in dt and "('backward',4)" in dt,'actual_engineering_counter_scope')
require("value.get('native_CUDA_gradient_equivalence_established') is False" in dt and "value.get('performance_improvement_claimed') is False" in dt,'no_unsupported_numeric_speed_equivalence')
require('runtime_contract' in dt and 'Exact parameter partition missing' in dt,'exact_actual_parameter_contract')
require("'worker_returned'" in dt and 'released_owner' in dt and "intent['folder']" in dt,'terminal_child_and_permanent_intent_binding')
require('New formal entry not prepared' in dt and 'New audit/cost entry not prepared' in dt,'incomplete_chain_cannot_register')
require("'evaluate_v4.py'" in dt and "'audit_completed_finetune_checkpoints_v7.py'" in dt and "'cost_aggregation_v8.py'" in dt,'new_evaluator_CPU_cost_required')
manifest=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v2.json'
m=json.loads(manifest.read_text(encoding='utf-8-sig'))
require(len(m['sources'])==84,'original84_source_count')
for record in m['sources']:require(desc(WORK/record['path'])==record,'original_byte:'+record['path'])
defaults={}
for name in ('finetune_v4','deterministic_repair_execution_v1','qualify_deterministic_repairs_v3'):
    result=subprocess.run([str(WORK/'S00/.venv/Scripts/python.exe'),'-B','-m',
                           'tridep_ccml.eeg_all_candidates.'+name],cwd=WORK,capture_output=True,text=True,check=True)
    defaults[name]=json.loads(result.stdout)
    require(defaults[name]['status']=='preparation_only' and defaults[name]['model_calls']==0,'actual_default_zero:'+name)
report=A/('independent_deterministic_repair_formal_sources_review_'+str(time.time_ns())+'.json')
value={'status':'preparation_sources_reviewed','reviewer_source':desc(Path(__file__)),'formal_source':desc(f),
       'contract_source':desc(d),'qualifier_source':desc(q),'checks':checks,'actual_defaults':defaults,
       'source_byte_hashing_performed':True,'checkpoint_tensor_reads':0,'model_calls':0,'GPU_calls':0,
       'actual_new_formal_registration_created':False,'actual_GPU_qualification_complete':False,
       'remaining_before_formal_launch':['Actual GPU qualifications and independent real-report review.',
          'New Q6 dispatcher/child ownership and exclusive runtime launch binding.',
          'CPU7 mixed observer4 acceptance, E4 inference, cost8 and new final/postprocessing gates.',
          'Freeze and independently review the complete actual source set and canonical engineering locks.']}
with report.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n')
print(json.dumps(desc(report)))
