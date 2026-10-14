"""Independent exact88-source + actual loss evidence review; no recovery run."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import io
import json
import time
import unittest

SOURCE=Path(__file__).resolve();A=SOURCE.parent;WORK=SOURCE.parents[3]
M=WORK/'tridep_ccml/eeg_all_candidates'
OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
EXPECTED_ENTRY='69ad989cefdb000304c89f9baf13529f096faa58223bce986e99a33e88612b10'


def require(ok,message):
    if not ok:raise ValueError(message)


def desc(path):
    path=Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix in ('.json','.py','.ps1'),'Restricted source/JSON review')
    return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))


def main():
    old=OUT/'continuation_queue_v5/recovery_registration_v2.json'
    require(desc(old)['sha256']=='525f6ed6e838fe267956c54d90d4e6d13d7da3bc710863aca90c35af466782cd','Actual sealed previous recovery2')
    old_sources=read(old)['sources']
    require(len(old_sources)==len({x['path'] for x in old_sources})==84,'Old84 preserved')
    new=[M/'recover_q5_after_process_loss_v3.py',M/'lost_f3_closure_v3.py',A/'audit_completed_finetune_checkpoints_v8.py',A/'cost_aggregation_v9.py']
    expected=[EXPECTED_ENTRY,'b7067820b9ff0fb3f6cfddf62991acb918d6a6198cdcf6ead07d3f0f5cffbaf1','43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f','4d0863abf0a4dbfe30e8e2bd853ca7d7bb8242212fce65683be36d9cd70811e0']
    require([desc(x)['sha256'] for x in new]==expected,'Reviewed new4 source bytes differ')
    sources=old_sources+[desc(x) for x in new]
    require(len(sources)==len({x['path'] for x in sources})==88,'Exactly inherited84+new4')
    for binding in sources:
        path=WORK/binding['path']
        require(desc(path)==binding,'Sealed runtime source changed: '+binding['path'])
        if path.suffix=='.py':compile(path.read_bytes(),str(path),'exec')
        else:read(path)
    tree=ast.parse(new[0].read_text('utf-8'))
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    require(any(isinstance(n,ast.Raise) for n in ast.walk(funcs['disable_evaluation'])),'Evaluation must unconditionally raise')
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='disable_evaluation' for n in ast.walk(funcs['run'])),'No evaluation run path')
    text=new[0].read_text('utf-8')
    require('driver.evaluate_tail = disable_evaluation' in text and 'original_run_child(job, owner, resume=True)' in text and "job == registered[0] and resume is True" in text,'Target-only original child dispatch')
    require('GetProcessTimes' in text and 'native_birth(pid, allow_absent=True)' in text and 'gpu_and_wsl_observation(graphics_binding)' in text,'Actual native/current GPU admission')
    require('driver.release_owner(owner)' in text and "'all_plan_complete': False" in text,'Keep original cleanup guard and noncomplete state')
    tests=[]
    for name,expected_sha in (
        ('independent_loss3_CPU8_cost9_stdlib_review_1791767311753700200.json','ab8a7c3d18b3fd2e167e8aa54523ab360117d262767e5546ba95ead2ee4de6f9'),
        ('independent_recovery3_control_tests_1791767612887869700.json','97860c2e68083987c812470ec43ca952673a5b6f36fca7107b9b941ffc4650d4'),
        ('q5_recovery_v3_entry_preparation_1791767607182636700.json','26c96f06e96d80392897d455c0a2e0d4db6436646f2b3cbc32807c9bcde23dac'),
        ('lost_f3_v3_cpu8_stdlib_tests_1791767260121847400.json','d7344418439dbf9ad52a69778be51c228e393d77d929b001803aa4a00954ab52')):
        path=A/name;binding=desc(path);value=read(path)
        require(binding['sha256']==expected_sha and value['passed'] is True,'Expected passing actual test artifact')
        tests.append(binding)
    # Independently rerun author control tests under a fresh private module.
    test_path=A/'test_recover_q5_after_process_loss_v3.py'
    spec=importlib.util.spec_from_file_location('independently_rerun_recovery3_guards',test_path)
    test_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(test_module)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_module))
    require(result.wasSuccessful() and result.testsRun==35,'Independent35 control rerun failed: '+stream.getvalue())
    evidence=[]
    for name,sha in (
        ('independent_actual_labram4s_loss_cpu_evidence_review_1791767150384408900.json','224685c60ed6b712da38e9a5e4095d4f423e55c3cee6c8dacaf7e865bec2633a'),
        ('independent_actual_loss3_bound_verifier_review_1791767374204622000.json','f03960af35d1d5484c14d8534c1fee6e75a7c9e570c86ff3ce0632c8f0fbae00')):
        path=A/name;binding=desc(path)
        require(binding['sha256']==sha and read(path)['passed'] is True,'Existing actual evidence review failed')
        evidence.append(binding)
    control=desc(A/'authorize_loss3_continue_20261012_v1.py')
    control_text=(WORK/control['path']).read_text('utf-8')
    require('disk_recovery_continue_intent.json' in control_text and "p.open('x'" in control_text and 'native_pause_complete.json' in control_text and '65977f9551a7621f9df0e1701173bf1845794a6c60b7a0d58012237092a98c40' in control_text and 'p.unlink()' in control_text,'Bound reversible marker release controls differ')
    for binding in sources:require(desc(WORK/binding['path'])==binding,'Source changed during independent review')
    value={'schema':'q5_recovery_v3_independent_source_review','passed':True,'reviewer':'/root/loss3_independent_review','source':desc(SOURCE),'reviewed_entry':sources[84],'sources':sources,'tests':tests,'independent_control_rerun':{'source':desc(test_path),'tests_run':result.testsRun,'passed':True,'output':stream.getvalue()},'actual_evidence_reviews':evidence,'separate_marker_release_control':{'source':control,'review_passed':True,'scope':'Explicit Oct12 human continue; permanently create intent, archive exact native proof/old markers/pointer before release. Zero native resume or model launch. New entry still independently checks live absence/sources and single GPU admission.'},'source_count':88,'original84_preserved':True,'original_formal_matrix_360_preserved':True,'training_protocol_changed':False,'test_evaluation_permanently_disabled':True,'process_actions':0,'model_operations':0,'GPU_operations':0,'checkpoint_tensor_loads':0,'checkpoint_binary_reads_this_source_review':0,'test_probability_reads':0,'scoped_launch_review':{'only_job':'LaBraM-native4s-lastblock-finetune-fit','original_Q5_F3_core':True,'allowed_progress_325_to_330':True,'blocked_original_repair_fits':30,'engineering_or_optional_probes':False,'all_plan_completion_allowed':False,'permanent_intent_and_two_fresh_absence_checks':True,'unknown_child_cleanup_retains_original_owner_guard':True},'closure_review':{'old_L1_L2_strict_delegation':True,'old_running_bytes_retained':True,'unknown_exit_elapsed_tail_preserved':True,'durable':[54,216,1458],'observed':[55,220,1485],'extra_4_27_is_subset_not_added_again':True,'original_CPU5_math_and_three_loads':True,'three_loss_cost_deltas_separate_and_nonnumeric_elapsed_unknown':True},'scope_limitations':['This is an independent source/evidence/control review before recovery, not proof that recovery launched, training advanced, all360 completed, or any final acceptance passed.','Point-in-time current GPU/WSL/native admission is implemented but not executed by this review. Observed QQ graphics is allowed only by exact native identity, never a performance-idle claim.','Actual CPU proof was produced separately with two CPU tensor loads; this source review does not reload tensors. The separately bound full L3 review may hash official weight bytes transitively, count/time unknown.','All new postprocessing and repair chain compatibility with third process loss still needs explicit new source/acceptance binding before use.']}
    target=A/('independent_q5_recovery_v3_source_review_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'artifact':desc(target),'passed':True,'sources':len(sources),'independently_rerun_tests':result.testsRun}))


if __name__=='__main__':main()
