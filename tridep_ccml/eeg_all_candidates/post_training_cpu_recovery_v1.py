"""Independent post-training CPU gate recovery; never starts/relabels training.
Q14 remains failed for its historical CPU16 failure. A new immutable gate can
only bind actual CPU17 135/405 plus all360 original completions and released
Q14 child exit0. Default is zero-operation preparation.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from . import storage as S,registry as R
from . import qualify_deterministic_repairs_v10 as Q
from . import deterministic_repair_execution_v9 as D
SOURCE=Path(__file__).resolve()
ROOT=S.OUT/'post_training_cpu_recovery_v1'
CPU17={'path': 'output/analysis/eeg_all_candidates_v1_20261008/audit_completed_finetune_checkpoints_v17.py', 'sha256': '3845dcf2e23a924700a519d6f687213604b4456b836d9f9b9d7262ddcb406eb8'}
FAILED_CPU16={'path': 'output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v16_1791831771185269400.json', 'sha256': '6418fb669f20e63444074d0e645c78c623519acdaa40de7ccda537b05079615f'}
Q14_FAILED={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v14/status.json', 'sha256': '3c7c24868ef4585034473d70711c6cb0be8ae2b91649ffdeb5fbbf4e7f9a6b3f'}
Q14_RELEASED={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v14/released_owner_1791831771722544000.json', 'sha256': 'e9191304abd1f96561450ce10a51944b918c9d85f4f3b9f369072d341851b87a'}
Q14_REG={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v14/registration.json', 'sha256': '93b19228fb8d3f032f87be4072f4ff2fb3a0bbe2d89adce93117896b5bc59742'}
CHILD_COMPLETE={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v14/attempts/f_1791825911463169900_7c008897/status.json', 'sha256': 'd6178fbe5c5206baf8a1da21f1616fff99cb5bf9d6f34edc28e1778ad14ff980'}
LINUX_REFERENCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/actual_platform_learning_rate_reference_linux_1791832760707973329.json', 'sha256': 'c7109d9ced13c853ecb6de99606b98ab1c2f2b9978535a0a2c614049fb682463'}
WINDOWS_REFERENCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/actual_platform_learning_rate_reference_win32_1791832754389436100.json', 'sha256': '0ce44e1bbfb87d9e0d3b300e4960bdcafdc726b851b8c76b473d90f8e8ce8b1f'}
ORIGINAL_REG={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/registration.json', 'sha256': '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'}

def require(ok,message):
    if not ok:raise ValueError(message)

def sources():
    return [CPU17,S.desc(SOURCE),S.desc(SOURCE.with_name('evaluate_v12.py'))]

def review_verified(binding):
    review=S.read(S.checked(binding))
    require(review.get('status')=='post_training_CPU_recovery_sources_reviewed' and review.get('passed') is True,'Actual independent post-training source review required')
    require(review.get('reviewed_sources')==sources() and review.get('Q14_registration')==Q14_REG,'Exact new three sources and historical194 review scope required')
    S.checked(review['reviewer_source'])
    for item in sources():S.checked(item)
    return review

def validate_CPU_shape(value, registration, registration_binding, cpu_source):
    expected = {x['fit_id'] for x in registration['fits'] if x['stage'] == '3'}
    require(len(expected) == 135 and value.get('schema_version') == 17 and (value.get('registration') == registration_binding) and (value.get('auditor_source') == cpu_source), 'Original FT135 and actual source-bound CPU17 required')
    for key, total in [('registered_total_fits', 360), ('registered_finetune_fits', 135), ('training_complete_finetune_observed', 135), ('verified_completed_finetune_fits', 135), ('logical_completed_fits_counted', 135), ('checkpoint_loads', 405)]:
        require(type(value.get(key)) is int and value[key] == total, 'Incomplete exact CPU17 count: ' + key)
    require(value.get('status') == 'all_finetune_verified' and value.get('all_135_finetune_checkpoints_verified') is True and (value.get('partial_snapshot') is False) and (value.get('errors') == []) and (value.get('deferred_completed_fits') == []) and (value.get('pending_fits') == []), 'Partial/deferred/error CPU17 report rejected')
    rows = value.get('completed_rows', [])
    require(len(rows) == 135 and {x['fit_id'] for x in rows} == expected and (len({x['fit_id'] for x in rows}) == 135), 'CPU17 rows must match original135 unique identities')
    for row in rows:
        deep = row.get('cpu_checkpoint_audit', {})
        require(row.get('passed') is True and deep.get('cpu_tensors_verified') is True and (type(deep.get('checkpoint_loads')) is int) and (deep['checkpoint_loads'] == 3) and (deep.get('only_map_location_cpu') is True) and (type(deep.get('gpu_operations')) is int) and (deep['gpu_operations'] == 0) and (type(deep.get('model_constructions')) is int) and (deep['model_constructions'] == 0), 'Actual three-checkpoint CPU17 row evidence missing')
    operations = value.get('observed_auditor_operations_including_failed_checks', {})
    require(all((type(operations.get(key)) is int and operations[key] == count for key, count in [('cpu_checkpoint_loads_started', 405), ('cpu_checkpoint_loads_returned', 405), ('model_construction_attempts_blocked', 0), ('gpu_initialization_attempts_blocked', 0)])), 'Actual CPU17 operation counts differ')
    return rows

def validate_CPU17(cpu):
    registration=S.read(S.checked(ORIGINAL_REG))
    rows=validate_CPU_shape(cpu,registration,ORIGINAL_REG,CPU17)
    require(cpu.get('strict_Mamba_original_Linux_LR_reference')==LINUX_REFERENCE and cpu.get('Windows_cross_platform_discrepancy')==WINDOWS_REFERENCE and cpu.get('learning_rate_tolerance_used') is False and cpu.get('three_expected_LR_expressions_only') is True and cpu.get('other_checks_unchanged') is True and cpu.get('failed_CPU16_preserved_not_relabelled') is True,'Only the reviewed strict original-platform expectation change permitted')
    for row in rows:
        folder=S.OUT/'runs'/row['fit_id']
        for key,name in [('status_artifact','status.json'),('metadata','metadata.json'),('completion','complete.json')]:
            require(S.checked(row[key])==(folder/name).resolve(),'CPU17 JSON binding differs from the original fit')
        checkpoints=row['cpu_checkpoint_audit']['checkpoints']
        require(len(checkpoints)==3 and {x['path'] for x in checkpoints}=={(folder/name).relative_to(S.WORK).as_posix() for name in ('initial.pt','latest.pt','best.pt')},'CPU17 checkpoint scope differs')
        for item in checkpoints:S.checked(item)
    return rows

def completed_training_verified():
    Q.check_pause()
    require(all(not(S.OUT/name/'owner.json').exists() for name in ('continuation_queue_v5','continuation_queue_v13','continuation_queue_v14','deterministic_repair_qualification_v10')),'Actual training/qualification owner must already be released')
    terminal=S.read(S.checked(Q14_FAILED));released=S.read(S.checked(Q14_RELEASED));child=S.read(S.checked(CHILD_COMPLETE));failed=S.read(S.checked(FAILED_CPU16));queue=S.read(S.checked(Q14_REG))
    qsource=S.desc(SOURCE.with_name('continuation_queue_v14.py'))
    require(terminal['status']=='failed' and terminal['error_type']=='CalledProcessError' and terminal['source']==qsource and terminal['execution_version']==14 and terminal['registration']==Q14_REG and terminal['all_plan_complete'] is False,'Original failed Q14 terminal must remain exact')
    require(terminal['outcomes']=={'EEGMamba-lastblock-finetune':CHILD_COMPLETE},'Original single Mamba15 child outcome must remain exact')
    require(child['status']=='complete' and type(child['returncode']) is int and child['returncode']==0 and child['source']==qsource and child['condition']=='EEGMamba-lastblock-finetune','Actual original training child must be exit0 complete')
    require(released['source']==qsource and released['registration']==Q14_REG and released['execution_version']==14 and all(released.get(key) is None for key in ('child_pid','child_token','child_job','child_native_identity','linux_worker_identity','child_native_roles')),'Actual released original Q14 owner required')
    admission=S.read(S.checked(child['worker_admission']))
    require(admission['queue_owner_id']==released['id'],'Completed child admission and released owner differ')
    require(child['child_native_identity']['parent']==released['pid'] and child['child_native_roles'][0]==child['child_native_identity'],'Historical exact child/controller identity differs')
    expected={f'EEGMamba-lastblock-finetune/seed_{seed}/fold_{fold}' for seed in (11,22,33) for fold in range(1,6)}
    require(failed['schema_version']==16 and failed['verified_completed_finetune_fits']==120 and failed['checkpoint_loads']==360 and failed['training_complete_finetune_observed']==135 and failed['deferred_completed_fits']==[] and failed['pending_fits']==[],'Exact historical failed CPU16 scope required')
    require(len(failed['errors'])==15 and {x['fit_id'] for x in failed['errors']}==expected and all(x['error_type']=='ValueError' and x['error']=='Actual head learning-rate schedule differs' for x in failed['errors']),'Only the proven platform-LR audit failure can use this recovery')
    require(failed['observed_auditor_operations_including_failed_checks']=={'cpu_checkpoint_loads_started':360,'cpu_checkpoint_loads_returned':360,'model_construction_attempts_blocked':0,'gpu_initialization_attempts_blocked':0},'Preserve real failed360 CPU reads')
    require(queue['version']==14 and queue['schema']=='eeg_deterministic_repair_queue_v14' and len(queue['sources'])==194,'Exact original194 registration required')
    for item in queue['sources']:S.checked(item)
    prior_review=S.read(S.checked(queue['source_review']))
    require(prior_review['passed'] is True and prior_review['reviewed_sources']==queue['sources'],'Original194 independent review changed')
    S.checked(prior_review['reviewer_source'])
    for item in (LINUX_REFERENCE,WINDOWS_REFERENCE):S.checked(item)
    previous=D.previous_codebrain_completion()
    training=R.assert_test_unlocked()
    require(training['formal_fits_verified']==360 and training['registration']==ORIGINAL_REG,'Every original360 byte-bound completion required')
    return {'previous_CodeBrain_completion':previous,'training_statuses':training['statuses'],'training_sources':queue['sources'],'source_review':queue['source_review']}

def verify_gate(binding):
    path=S.checked(binding)
    require(path.parent==ROOT.resolve() and path.name=='complete.json','Canonical independent CPU recovery gate required')
    gate=S.read(path)
    require(gate.get('schema')=='post_training_cpu_recovery_v1' and gate.get('passed') is True and gate.get('source')==S.desc(SOURCE),'Actual new post-training gate source differs')
    review_verified(gate['post_training_source_review'])
    expected=completed_training_verified()
    require(gate['original_Q14_failed_terminal']==Q14_FAILED and gate['original_Q14_released_owner']==Q14_RELEASED and gate['original_child_complete']==CHILD_COMPLETE and gate['failed_CPU16']==FAILED_CPU16 and gate['registration']==Q14_REG,'Recovery must preserve all exact historical failure/release evidence')
    require(gate['original_training_complete']==360 and gate['FT']==135 and gate['CPU_loads']==405 and gate['actual_new_training_or_optimizer_updates']==0 and gate['original_Q14_status_rewritten'] is False,'Exact post-training-only gate required')
    require(gate['post_training_sources']==sources(),'New post-training source list changed')
    require(all(gate[key]==value for key,value in expected.items()),'Actual training/source/CodeBrain lineage changed')
    validate_CPU17(S.read(S.checked(gate['CPU_audit'])))
    return gate

def create_gate(cpu_binding,review_binding):
    Q.check_pause();review_verified(review_binding)
    require(not(ROOT/'complete.json').exists(),'Permanent completed recovery gate already exists; do not repeat')
    actual=completed_training_verified();validate_CPU17(S.read(S.checked(cpu_binding)))
    value={'schema':'post_training_cpu_recovery_v1','passed':True,'source':S.desc(SOURCE),'registration':Q14_REG,'original_Q14_failed_terminal':Q14_FAILED,'original_Q14_released_owner':Q14_RELEASED,'original_child_complete':CHILD_COMPLETE,'failed_CPU16':FAILED_CPU16,'CPU_audit':cpu_binding,'CPU_auditor':CPU17,'original_training_complete':360,'FT':135,'CPU_loads':405,'post_training_source_review':review_binding,'post_training_sources':sources(),'created_at':S.now(),'actual_new_training_or_optimizer_updates':0,'original_Q14_status_rewritten':False,'all_plan_complete':False,**actual}
    Q.check_pause();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'complete.json').open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return S.desc(ROOT/'complete.json')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    for key in ('cpu-report','cpu-report-sha256','source-review','source-review-sha256'):parser.add_argument('--'+key)
    args=parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'status':'preparation_only','Tensor_loads':0,'model_calls':0,'GPU_queries':0,'process_queries':0,'writes':0}));return 0
    require(all((args.cpu_report,args.cpu_report_sha256,args.source_review,args.source_review_sha256)),'Actual CPU17 and independent source review bindings required')
    print(json.dumps({'artifact':create_gate({'path':args.cpu_report,'sha256':args.cpu_report_sha256},{'path':args.source_review,'sha256':args.source_review_sha256})}));return 0
if __name__=='__main__':raise SystemExit(main())
