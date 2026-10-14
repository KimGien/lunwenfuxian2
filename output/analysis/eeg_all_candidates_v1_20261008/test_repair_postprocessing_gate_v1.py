"""Synthetic gate failures; no model/checkpoint/test probability operations."""
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
from tridep_ccml.eeg_all_candidates import storage as S
spec=importlib.util.spec_from_file_location('_repair_postprocessing_gate_test',SOURCE.with_name('repair_postprocessing_gate_v1.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def main():
    registration=S.read(S.OUT/'registration.json')
    reg_binding=S.desc(S.OUT/'registration.json')
    cpu_source=S.desc(SOURCE.with_name('audit_completed_finetune_checkpoints_v7.py'))
    rows=[{'fit_id':x['fit_id'],'passed':True,'cpu_checkpoint_audit':{
        'cpu_tensors_verified':True,'checkpoint_loads':3,'only_map_location_cpu':True,
        'gpu_operations':0,'model_constructions':0}} for x in registration['fits'] if x['stage']=='3']
    base={'schema_version':7,'registration':reg_binding,'auditor_source':cpu_source,
          'registered_total_fits':360,'registered_finetune_fits':135,'training_complete_finetune_observed':135,
          'verified_completed_finetune_fits':135,'logical_completed_fits_counted':135,'checkpoint_loads':405,
          'status':'all_finetune_verified','all_135_finetune_checkpoints_verified':True,'partial_snapshot':False,
          'errors':[],'deferred_completed_fits':[],'pending_fits':[],'completed_rows':rows,
          'observed_auditor_operations_including_failed_checks':{'cpu_checkpoint_loads_started':405,
              'cpu_checkpoint_loads_returned':405,'model_construction_attempts_blocked':0,'gpu_initialization_attempts_blocked':0}}
    checks={}
    assert len(M.validate_CPU_shape(base,registration,reg_binding,cpu_source))==135
    checks['valid_synthetic_135_rows_shape_only']=True
    mutations=[
        ('partial_134',lambda v:v.update(verified_completed_finetune_fits=134)),
        ('false_int_boolean',lambda v:v.update(checkpoint_loads=True)),
        ('pending_fit',lambda v:v.update(pending_fits=['pending'])),
        ('deferred_fit',lambda v:v.update(deferred_completed_fits=['deferred'])),
        ('failed_CPU',lambda v:v.update(errors=['failed'])),
        ('wrong_schema_oldCPU6',lambda v:v.update(schema_version=6)),
        ('wrong_auditor_bytes',lambda v:v.update(auditor_source={'path':cpu_source['path'],'sha256':'b'*64})),
        ('missing_row',lambda v:v['completed_rows'].pop()),
        ('duplicate_identity',lambda v:v['completed_rows'][-1].update(fit_id=v['completed_rows'][0]['fit_id'])),
        ('nonCPU_tensor',lambda v:v['completed_rows'][0]['cpu_checkpoint_audit'].update(only_map_location_cpu=False)),
        ('model_constructed',lambda v:v['completed_rows'][0]['cpu_checkpoint_audit'].update(model_constructions=1)),
        ('GPU_call',lambda v:v['observed_auditor_operations_including_failed_checks'].update(gpu_initialization_attempts_blocked=1)),
        ('missing_returned_load',lambda v:v['observed_auditor_operations_including_failed_checks'].update(cpu_checkpoint_loads_returned=404))]
    for name,mutate in mutations:
        v=copy.deepcopy(base);mutate(v)
        try:M.validate_CPU_shape(v,registration,reg_binding,cpu_source)
        except ValueError:checks['reject:'+name]=True
        else:raise AssertionError('Invalid full CPU evidence admitted:'+name)
    complete=['training_complete']*360
    assert M.screening(registration,complete,[],True)['status']=='metadata_ready'
    checks['metadata_ready_is_not_GPU_admission']=True
    for name,statuses,owners,finished in [
        ('one_original_fit_incomplete',complete[:-1]+['training'],[],True),
        ('active_training_owner',complete,['continuation_queue_v5/owner.json'],True),
        ('missing_E4_real_completion',complete,[],False)]:
        assert M.screening(registration,statuses,owners,finished)['status']=='pending'
        checks['screen_reject:'+name]=True
    with patch.object(M,'inspect',return_value={'status':'pending'}), \
         patch.object(M.S,'checked') as checked,patch.object(M.E,'verify_gate') as cpu_check, \
         patch.object(M.R,'assert_test_unlocked') as training,patch.object(M.E,'completed_report_verified') as outer:
        try:M.verify_existing()
        except ValueError:pass
        else:raise AssertionError('Incomplete train gate accepted')
        assert not checked.called and not cpu_check.called and not training.called and not outer.called
    checks['pending_rejected_before_checkpoint_hash_or_outer_evaluation_verifier']=True
    result=subprocess.run([sys.executable,'-B',str(M.SOURCE)],cwd=WORK,capture_output=True,text=True,check=True)
    value=json.loads(result.stdout);assert value=={'status':'preparation_only','model_calls':0,'GPU_queries':0,'outer_probability_reads':0,'writes':0}
    checks['default_zero_operation']=True
    proof={'status':'preparation_guards_passed','source':S.desc(SOURCE),'helper_source':S.desc(M.SOURCE),
           'checks':checks,'actual_CPU7_full135405_proof':False,'actual_E4_completed_or_verified':False,
           'synthetic_evidence_only':True,'model_GPU_calls':0,'checkpoint_tensor_loads':0,'outer_probability_reads':0}
    path=SOURCE.parent/('repair_postprocessing_gate_preparation_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(S.desc(path)))
if __name__=='__main__':main()
