"""Independent entry-source review; no model/GPU/runtime qualification."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parents[3]
ENTRY=WORK/'tridep_ccml/eeg_all_candidates/qualify_deterministic_repairs_v3.py'
REPORT=WORK/'output/analysis/eeg_all_candidates_v1_20261008'
text=ENTRY.read_text(encoding='utf-8')
tree=ast.parse(text)
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
def part(name):return ast.get_source_segment(text,functions[name])
def descriptor(p):
    p=Path(p)
    return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
checks={}
def check(name,value):
    if not value:raise ValueError('Independent review failed: '+name)
    checks[name]=True

check('top_level_imports_stdlib_only',all(n.module in ('__future__','datetime','pathlib') for n in tree.body if isinstance(n,ast.ImportFrom)))
check('default_returns_before_controller',part('main').index("'status':'preparation_only'")<part('main').index('runtime_worker(')<part('main').index('run_controller('))
check('owner_and_boundary_precede_any_GPU_query',part('safe_before_model').index('check_boundary()')<part('safe_before_model').index('host_inventory()')<part('safe_before_model').index('gpu_empty()'))
check('actual_incomplete_Q5_not_running',"'required_work_remaining'" in part('check_boundary') and "'owner.json'" in part('check_boundary') and "'recovery_execution_version')==2" in part('check_boundary'))
check('all_three_strict_pause_markers',all(x in part('check_pause') for x in ('sync_boundary','sync_boundary_v2','sync_boundary_v3',"{'paused':False}",'type(value')))
check('permanent_intent_and_owner_exclusive',part('run_controller').count('exclusive(')>=4 and "'permanent_intent.json'" in part('run_controller') and "'owner.json'" in part('run_controller'))
check('full84_and_original_matrix_pinned',"len(sources)==84" in part('source_check') and 'MATRIX_SHA' in part('source_check'))
check('exact_host_birth_and_worker_admission',"(x['pid'],x['birth'])" in part('admit_host') and 'admission_sha256' in part('validate_worker') and 'controller_lease' in part('validate_worker'))
check('worker_owner_precedes_torch_import',part('runtime_worker').index("exclusive(folder/'worker_owner.json'")<part('runtime_worker').index('from . import finetune as F'))
check('primary_attempt_timer_precedes_runtime',part('runtime_worker').index("exclusive(folder/'attempt.json'")<part('runtime_worker').index('started=time.monotonic()')<part('runtime_worker').index('from . import finetune as F'))
check('strict_determinism_never_relaxed','not torch.is_deterministic_algorithms_warn_only_enabled()' in part('runtime_worker') and 'use_deterministic_algorithms(False' not in text and 'warn_only=True' not in text)
check('private_build_model_code_reused',all(x in part('isolated_builder') for x in ('types.FunctionType','original.__code__','original.__globals__','F.build_model is original')))
check('training_only_micro1_chunk8',all(x in part('runtime_worker') for x in ("split['train'][0]","person not in split['val']+split['test']",'people=[person]','window_chunk=8')))
check('native_forward_comparison_no_grad',part('runtime_worker').count('with torch.no_grad()')==2 and 'torch.equal(native,adapted)' in part('runtime_worker'))
check('two_phases_two_gradient_repeats_zero_steps',all(x in part('runtime_worker') for x in ('for epoch in (1,11)','for repeat in range(2)','optimizer.step=forbid_step',"record['backward_returned']==4","record['encoder_windows_returned']==48","record['optimizer_step_returned']==0")))
check('frozen_model_buffer_and_resume_verified',all(x in part('runtime_worker') for x in ('Frozen parameter acquired gradient','Zero-update probe changed parameter/buffer bytes','Complete zero-step engineering resume differs')))
check('failure_work_and_cost_not_faked',"'unreturned_call_internal_work_unknown" not in text and 'unreturned_call_internal_work_unknown=True' in part('runtime_worker') and 'primary_engineering_seconds=time.monotonic()-started' in part('runtime_worker'))
check('no_original_registry_or_fit_mutation',all(x not in text for x in ('record_fit_status','lock_condition','NtSuspend','NtResume','terminate(','kill(')))
check('Mamba_order_change_and_no_speed_claim',"'accumulation_order_changed':condition==CONDITIONS[1]" in text and "'native_CUDA_gradient_equivalence_established':False" in text and "'performance_improvement_claimed':False" in text)
check('installed_packages_not_written',"torch.save(value,checkpoint)" in text and 'new_formal_execution_required' in text and "'No engineering checkpoint overwrite'" in text)
check('actual_runtime_contract_only_after_GPU_checks',part('runtime_worker').index("runtime_contract={")>part('runtime_worker').index("'Observed engineering counts differ'") and "set(update_names)==set(phases[1]['gradient_parameter_names'])" in part('runtime_worker'))
# Independently verify actual immutable runtime84 bytes without importing
# registry/lock/model helpers; these are source-file byte hashes, not JSON-only.
manifest=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v2.json'
value=json.loads(manifest.read_text(encoding='utf-8-sig'))
check('actual_manifest84_hash','525f6ed6e838fe267956c54d90d4e6d13d7da3bc710863aca90c35af466782cd'==descriptor(manifest)['sha256'] and len(value['sources'])==84)
for record in value['sources']:
    check('original_source:'+record['path'],descriptor(WORK/record['path'])==record)
result=subprocess.run([str(WORK/'S00/.venv/Scripts/python.exe'),'-B','-m',
                       'tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v3'],
                      cwd=WORK,capture_output=True,text=True,check=True)
default=json.loads(result.stdout)
check('actual_default_zero_model_query_write',all(default[x]==0 for x in ('model_calls','GPU_queries','writes','optimizer_steps','formal_fits')))
report=REPORT/('independent_deterministic_repair_entry_v3_review_'+str(time.time_ns())+'.json')
body={'status':'entry_preparation_source_review_passed','reviewer_source':descriptor(Path(__file__)),
      'entry_source':descriptor(ENTRY),'checks':checks,'original84_sources_verified':True,
      'default_result':default,'actual_model_calls':0,'actual_GPU_calls':0,'checkpoint_tensor_reads':0,
      'source_byte_hashing_performed':True,'GPU_qualification_complete':False,'formal_execution_integrated':False,
      'limitations':['Actual Windows/WSL worker and GPU graph are untested.',
                     'GPU admission is conservative: all contexts must be empty; no unrelated program is controlled.',
                     'No clean speed comparison or original CUDA backward bitwise-equivalence is established.',
                     'Future formal repair, CPU audit, evaluator, costs and all final gates require fresh reviewed source bindings.']}
with report.open('x',encoding='utf-8') as f:json.dump(body,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps(descriptor(report)))
