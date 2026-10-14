"""Standard-library negative evidence tests, no actual GPU qualification."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
SOURCE=Path(__file__).resolve();WORK=SOURCE.parents[3]
sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import continuation_queue_v6 as Q6, storage as S
spec=importlib.util.spec_from_file_location('_actual_engineering_review',SOURCE.with_name('review_actual_repair_engineering_v1.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
B=lambda n:{'path':n,'sha256':'a'*64}
def fixture(condition):
    report={'optimizer_step_started':0,'optimizer_step_returned':0,'backward_started':4,'backward_returned':4,
            'forward_started':6,'forward_returned':6,'encoder_windows_started':48,'encoder_windows_returned':48,
            'probe_person':'train1','original_native_forward_bitwise_equal':True,'checks':[{'repeat_bitwise_equal':True}]*2,
            'zero_step_optimizer_and_rng_resume_verified':True,'performance_improvement_claimed':False,
            'native_CUDA_gradient_equivalence_established':False,'accumulation_order_changed':condition==M.D.REPAIR_CONDITIONS[1],
            'primary_engineering_seconds':2.0,'admission':B('admission'),'source':B('qualifier'),'outer_test_read':False,'formal_fits':0}
    contract={'probe_subject_ids':['train1'],'trainable_parameter_names':['encoder.w','head.w'],
              'frozen_parameter_names':['encoder.frozen'],'accumulation_order_changed':report['accumulation_order_changed']}
    proof={'observed_engineering':report,'contract':contract,'controller':B('controller'),
           'released_owner':B('released_owner'),'worker_owner':B('worker_owner')}
    evidence={'controller':{'controller_seconds_primary_cost':False},
              'worker_owner':{'GPU_before_runtime':{'context_count':0},'native':{
                  'controller_lease':B('owner.json'),'allowed_host':[{'pid':1,'birth':'2'}]}},
              'admission':{'GPU_admission':{'context_count':0}}}
    return proof,evidence
def call(proof,evidence,condition):
    with patch.object(M.Q,'check_pause'),patch.object(M.Q,'check_boundary'),patch.object(M.D,'qualification',return_value=proof), \
         patch.object(M.S,'checked',side_effect=lambda b:b['path']),patch.object(M.S,'read',side_effect=lambda p:evidence[p]):
        return M.review(B('report'),condition)
def main():
    checks={}
    for condition in M.D.REPAIR_CONDITIONS:
        p,e=fixture(condition);v=call(p,e,condition)
        assert v['passed'] is True and v['new_formal_training_or_final_plan_complete'] is False
        checks['simulated_valid:'+condition]=True
    condition=M.D.REPAIR_CONDITIONS[0]
    bad=[
      ('missing_backward_return',lambda p,e:p['observed_engineering'].update(backward_returned=3)),
      ('optimizer_update',lambda p,e:p['observed_engineering'].update(optimizer_step_returned=1)),
      ('changed_forward',lambda p,e:p['observed_engineering'].update(original_native_forward_bitwise_equal=False)),
      ('gradient_repeat_failed',lambda p,e:p['observed_engineering']['checks'][0].update(repeat_bitwise_equal=False)),
      ('numerical_disclosure_changed',lambda p,e:p['observed_engineering'].update(accumulation_order_changed=True)),
      ('nonpositive_primary_seconds',lambda p,e:p['observed_engineering'].update(primary_engineering_seconds=0)),
      ('worker_external_context',lambda p,e:e['worker_owner']['GPU_before_runtime'].update(context_count=1)),
      ('admission_external_context',lambda p,e:e['admission']['GPU_admission'].update(context_count=1)),
      ('changed_controller_bytes',lambda p,e:e['worker_owner']['native']['controller_lease'].update(sha256='b'*64)),
      ('actual_test_role',lambda p,e:p['observed_engineering'].update(outer_test_read=True))]
    for name,mutate in bad:
        p,e=fixture(condition);mutate(p,e)
        try:call(p,e,condition)
        except ValueError:checks['reject:'+name]=True
        else:raise AssertionError('Invalid evidence accepted: '+name)
    with patch.object(Q6.S,'read',return_value={'sources':[B('changed_source')]}), \
         patch.object(Q6.S,'checked',side_effect=ValueError('source changed')), \
         patch.object(Q6.R,'status_summary') as summary,patch.object(Q6.subprocess,'run') as run:
        try:Q6.cpu_gate()
        except ValueError as error:assert str(error)=='source changed'
        else:raise AssertionError('Changed source accepted before CPU7')
        assert not summary.called and not run.called
    checks['CPU7_changed_source_rejected_before_checkpoint_process']=True
    completed=subprocess.run([sys.executable,'-B',str(M.SOURCE)],cwd=WORK,capture_output=True,text=True,check=True)
    default=json.loads(completed.stdout)
    assert default['actual_engineering_review_started'] is False and default['GPU_calls']==0
    checks['actual_default_review_zero_operation']=True
    value={'status':'preparation_guards_passed','source':S.desc(SOURCE),'reviewer_source':S.desc(M.SOURCE),
           'Q6_source':S.desc(Q6.SOURCE),'checks':checks,'simulated_checks_only':True,
           'actual_engineering_reports_reviewed':0,'model_GPU_operations':0,'outer_probability_reads':0,
           'formal_registration_created':False}
    target=SOURCE.parent/('actual_repair_engineering_review_preparation_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(S.desc(target)))
if __name__=='__main__':main()
