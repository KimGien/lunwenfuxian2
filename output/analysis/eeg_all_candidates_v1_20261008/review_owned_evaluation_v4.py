"""Independent E4 source/binding/metric-code review; no probabilities."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
WORK=Path(__file__).resolve().parents[3];MOD=WORK/'tridep_ccml/eeg_all_candidates';A=WORK/'output/analysis/eeg_all_candidates_v1_20261008'
ENTRY=MOD/'evaluate_v4.py'
text=ENTRY.read_text(encoding='utf-8');tree=ast.parse(text)
functions={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
checks={}
def require(ok,name):
    if not ok:raise ValueError(name)
    checks[name]=True
def desc(path):
    path=Path(path)
    return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
require('import torch' not in text and 'torch.load(' not in text,'no_added_model_or_tensor_code')
require(functions['verify_gate'].index("/owner.json")<functions['verify_gate'].index("R.status_summary()")
        <functions['verify_gate'].index('S.checked(binding)'),'training_owners_and360_precede_gate_artifact_reads')
require('135' in functions['validate_gate_data'] and '405' in functions['validate_gate_data']
        and "cpu.get('schema_version')==7" in functions['validate_gate_data'],'strict_actual_CPU7_135405_source')
require(all(x in functions['validate_gate_data'] for x in ("cpu.get('errors')==[]","cpu.get('deferred_completed_fits')==[]","cpu.get('pending_fits')==[]")),
        'CPU_errors_deferred_pending_cannot_open_test')
require('V.verify_execution(' in functions['verify_training_execution'] and 'X.verify_execution_acceleration(' in functions['verify_training_execution'],
        'separate_actual4_and_old3_provenance')
require('previous_materialize(*args,**kwargs)' in functions['isolated_driver'] and 'no_backward_for_test' in functions['execution_provenance'],
        'reuse_exact_original_E2_native_inference')
require(functions['worker'].index('verify_gate(gate_binding)')<functions['worker'].index('isolated_driver()'),
        'actual_full_gate_before_any_driver_or_probability_read')
require("owner['id']==admission['queue_owner_id']" in functions['worker'] and "current_roles==roles" in functions['worker'],
        'nonce_owner_and_birth_tree_live_per_evaluated_fit')
require("launch==owner['child_identity']" in functions['admit_worker'] and "actual['parent']==launch['pid']" in functions['admit_worker'],
        'actual_parent_launcher_python_creation_identity')
require("for mode in ('mamba_linux','all_windows')" in functions['controller'] and 'process.wait()' in functions['controller'],
        'singleGPU_Linux_then_Windows_sequential_return')
require('Q.gpu_empty()' in functions['controller'] and 'Q.linux_inventory()==[]' in functions['controller'],
        'fresh_empty_GPU_and_experiment_admission')
require("current.get('child_pid') is None" in functions['controller'],'unverified_child_keeps_owner')
require('all_plan_complete=False' in functions['controller'],'evaluation_completion_not_entire_plan')
require('optimizer.step(' not in text and 'loss.backward(' not in text,'evaluation_never_trains_or_optimizes')
sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import evaluate_v4 as E
driver=E.isolated_driver()
require(driver.metrics.__code__==E.E2.E1.metrics.__code__ and driver.predicted_class.__code__==E.E2.E1.predicted_class.__code__,
        'actual_threshold_and_metrics_original_code_objects_equal')
require(driver.predicted_class(.5)==0 and driver.predicted_class(.500001)==1,'actual_fixed_strict_threshold')
require('torch' not in sys.modules,'source_review_and_compile_no_tensor_import')
m=json.loads((WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v2.json').read_text(encoding='utf-8-sig'))
require(len(m['sources'])==84,'original84_count')
for item in m['sources']:require(desc(WORK/item['path'])==item,'original_byte:'+item['path'])
result=subprocess.run([str(WORK/'S00/.venv/Scripts/python.exe'),'-B','-m','tridep_ccml.eeg_all_candidates.evaluate_v4'],
                      cwd=WORK,capture_output=True,text=True,check=True)
default=json.loads(result.stdout)
require(default['outer_probability_reads']==default['model_calls']==default['GPU_queries']==0
        and default['actual_evaluation_started'] is False,'actual_default_zero_model_test_probability')
target=A/('independent_owned_evaluation_v4_review_'+str(time.time_ns())+'.json')
value={'status':'preparation_source_reviewed','reviewer_source':desc(Path(__file__)),'source':desc(ENTRY),
       'checks':checks,'default_result':default,'source_byte_hashing_performed':True,'Tensor_imports':0,
       'model_calls':0,'GPU_queries':0,'outer_probability_reads':0,'actual_evaluation_started':False,
       'full_new_source_chain_review_passed':False,
       'remaining':['Real GPU repair qualification and fully bound new formal30 source registration.',
                    'Actual old/new135/405 CPU7 and all360 before any evaluation.',
                    'Actual native/WSL evaluation child/cleanup and final mixed-source acceptance evidence.']}
with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(desc(target)))
