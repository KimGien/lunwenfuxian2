"""Metadata-only E12 failed-Windows closure and exact completed-Mamba reuse.
Default does nothing. --run only seals an independently reviewed recovery
boundary; no model, Tensor, process query or probability payload is read.
"""
from __future__ import annotations
import argparse, ast, json, os, types
from pathlib import Path
from . import storage as S, protocol as P
from . import post_training_cpu_recovery_v1 as RECOVERY
SOURCE=Path(__file__).resolve()
ROOT=S.OUT/'evaluation_storage_recovery_v1'
MAMBA='EEGMamba-lastblock-finetune'
E12_SOURCE={'path': 'tridep_ccml/eeg_all_candidates/evaluate_v12.py', 'sha256': '9a84b1cb5994571f1c486abf59a58a5100f111dbda2bd9ed93dec643bb532473'}
E1_SOURCE={'path': 'tridep_ccml/eeg_all_candidates/evaluate.py', 'sha256': '5956bff6395a23c8bd0e60ffed9abb276db529301e4d56265308e7db70ec523f'}
STORAGE_SOURCE={'path': 'tridep_ccml/eeg_all_candidates/storage.py', 'sha256': '1c5dc51e33633beee8b6b54ffd43249ac53679486f62db18e9872e67fbaf0fbe'}
CPU_GATE={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/post_training_cpu_recovery_v1/complete.json', 'sha256': 'fc3c163fcbd62a73cf4f01cdfb7ddfb919b86dc479dd6f4fc22a01f4f9aa1181'}
PARENT={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation_v12_control/runs/1791834075976739200/status.json', 'sha256': '9c324022ea2bd7365e7fff5d13dd7384fbf9505ebaf8ac8f45c95ade04689c47'}
RELEASED={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation_v12_control/runs/1791834075976739200/released_owner.json', 'sha256': '51a407ad16649f635fbed4244c38195d6b118be0e9f2e256b3f292d1c475925c'}
MAMBA_WORKER={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation_v12_control/runs/1791834075976739200/mamba_linux/worker.json', 'sha256': 'b2bcbb33634fd8a095de4b31e7664e673cce67bcb3815ce6143f804a522152c2'}
WINDOWS_WORKER={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation_v12_control/runs/1791834075976739200/all_windows/worker.json', 'sha256': 'aeb4324e2db912dc8f78649629050bc541502507526fe9c0859d3be5ea9ae168'}
WINDOWS_LOG={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation_v12_control/runs/1791834075976739200/all_windows/process.log', 'sha256': '3cdb44fca164e6b199848eb1b7636731ce9785f4844b9a17e7748590c734a555'}
MAMBA_MANIFEST={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation/EEGMamba-lastblock-finetune/predictions_manifest.json', 'sha256': '9d3a15c32609ae74f7db91f1ad9105f616a0135dfa51de5a0a484645a71d5661'}

METRICS_REUSE_CHECK={'path': 'output/analysis/eeg_all_candidates_v1_20261008/actual_E12_Mamba_Windows_metrics_reuse_check_1791838061112571200.json', 'sha256': '5143600cd2c90293d5ff7aa647216e6a8f4df9e2b04c6a48313efbe4c6c32590'}
FAILED_ATTEMPT='1791837124597736100_a2c099adc8ec417092bd6215e22c827f.json'

def require(ok,message):
    if not ok: raise ValueError(message)

def sources():
    return [S.desc(SOURCE),S.desc(SOURCE.with_name('evaluate_v13.py'))]

def private_storage():
    original=ast.parse(S.checked(STORAGE_SOURCE).read_text(encoding='utf-8'))
    selected=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='write']
    require(len(selected)==1,'Unique original storage writer required')
    changed=0
    for node in selected[0].body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='temporary':
            require(ast.dump(node.value)==ast.dump(ast.parse("path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')",mode='eval').body),'Original temporary expression differs')
            node.value=ast.parse("path.with_name('.' + uuid.uuid4().hex + '.part')",mode='eval').body
            changed+=1
    require(changed==1,'Exactly one temporary-name assignment required')
    module=types.ModuleType(__package__+'._E13_short_temporary_storage')
    module.__package__=__package__;module.__file__=str(S.checked(STORAGE_SOURCE))
    exec(compile(ast.fix_missing_locations(original),module.__file__,'exec'),module.__dict__)
    return module

def path_bounds():
    rows=[]
    for spec in P.CONDITIONS:
        if spec.objective!='supervised': continue
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                for snapshot in (P.SNAPSHOT_EPOCHS if spec.policy_key=='diagnostic' else ('best',)):
                    folder=S.OUT/'evaluation'/spec.condition_id/f'seed_{seed}'/f'fold_{fold}'/('best' if snapshot=='best' else f'epoch_{snapshot}')
                    target=folder/'attempts'/('9'*20+'_'+'f'*32+'.json')
                    # Epoch nanoseconds currently have19 digits;20 gives a strict spare digit.
                    rows.append({'condition':spec.condition_id,'seed':seed,'fold':fold,'snapshot':snapshot,'canonical_max_chars':len(str(target.resolve())),'temporary_max_chars':len(str(target.with_name('.'+'f'*32+'.part').resolve()))})
    require(len(rows)==435 and all(x['canonical_max_chars']<260 and x['temporary_max_chars']<260 for x in rows),'Every original435 canonical/short-temporary path must fit native Windows limit')
    return {'evaluated_checkpoints':len(rows),'canonical_max_chars':max(x['canonical_max_chars'] for x in rows),'short_temporary_max_chars':max(x['temporary_max_chars'] for x in rows),'future_attempt_ns_digits_upper_bound':20,'all_below_260':True}

def historical_metadata():
    require(not(S.OUT/'evaluation_v12_control/owner.json').exists(),'Original E12 owner must be released')
    for item in (E12_SOURCE,E1_SOURCE,STORAGE_SOURCE): S.checked(item)
    parent=S.read(S.checked(PARENT));release=S.read(S.checked(RELEASED))
    linux=S.read(S.checked(MAMBA_WORKER));windows=S.read(S.checked(WINDOWS_WORKER))
    require(parent['status']=='running' and parent['source']==E12_SOURCE and parent['CPU_gate']==CPU_GATE and parent['all_plan_complete'] is False,'Historical parent running bytes must remain unchanged')
    require(parent['stages']=={'mamba_linux':{'returncode':0,'worker_report':MAMBA_WORKER},'all_windows':{'returncode':1,'worker_report':WINDOWS_WORKER}},'Actual two child outcomes must remain unchanged')
    require(release['source']==E12_SOURCE and release['CPU_gate']==CPU_GATE and all(release.get(k) is None for k in ('child_pid','child_token','child_identity','linux_worker_identity','child_native_roles')),'Actual E12 released owner required')
    for worker,mode,status in ((linux,'mamba_linux','complete'),(windows,'all_windows','failed')):
        require(worker['source']==E12_SOURCE and worker['CPU_gate']==CPU_GATE and worker['mode']==mode and worker['status']==status and worker['formal_fits_or_optimizer_updates_added']==0,'Actual E12 child identity/status changed')
        admission=S.read(S.checked(worker['admission']))
        require(admission['queue_owner_id']==release['id'] and admission['mode']==mode and admission['source']==E12_SOURCE and admission['CPU_gate']==CPU_GATE,'Actual child/released owner lineage differs')
    require(release['child_admission']==windows['admission'],'Released final child admission differs')
    require(windows['error_type']=='FileNotFoundError' and FAILED_ATTEMPT in windows['error'] and '.part' in windows['error'],'Only the exact original atomic-temporary failure can recover')
    text=S.checked(WINDOWS_LOG).read_text(encoding='utf-8')
    require('line 324, in _evaluate_fit' in text and 'S.save(attempt_path, ledger, immutable=True)' in text and 'with temporary.open' in text,'Original pre-materialization trace required')
    failed_path=S.OUT/'evaluation/EEGPT-mean-projection4s-originalCNN-300ep/seed_11/fold_1/epoch_60/attempts'/FAILED_ATTEMPT
    require(not failed_path.exists(),'Historical first ledger unexpectedly exists')
    result=linux['evaluation_result'];manifest=S.read(S.checked(MAMBA_MANIFEST))
    require(result['passed'] is True and result['staged_conditions']==[MAMBA] and result['predictions_manifests']==[MAMBA_MANIFEST],'Actual Mamba staged success differs')
    require(manifest['passed'] is True and manifest['condition']==MAMBA and manifest['supervised_fits_evaluated']==15 and manifest['evaluator_source']==E12_SOURCE and manifest['evaluator_execution']['version']==12 and manifest['evaluator_execution']['wrapper_source']==E12_SOURCE and manifest['test_gate']==result['test_gate'],'Original Mamba15 manifest/source differs')
    expected={(S.OUT/'evaluation'/MAMBA/f'seed_{seed}'/f'fold_{fold}'/'best/complete.json').relative_to(S.WORK).as_posix():(seed,fold) for seed in P.SEEDS for fold in P.FOLDS}
    artifacts=manifest['evaluation_fit_artifacts']
    require(len(artifacts)==15 and len({x['path'] for x in artifacts})==15 and {x['path'] for x in artifacts}==set(expected),'All original fifteen unique Mamba completions required')
    for item in artifacts:
        value=S.read(S.checked(item));binding=value['binding'];seed,fold=expected[item['path']]
        require(value['passed'] is True and value['optimizer_updates']==0 and value['ema_updates']==0 and binding['condition']==MAMBA and binding['seed']==seed and binding['fold']==fold and binding['snapshot']=='best' and binding['evaluator_source']==E12_SOURCE and value['evaluator_execution']==manifest['evaluator_execution'] and binding['evaluator_execution']==manifest['evaluator_execution'],'Completed Mamba cache retains exact actual E12 source')
    comparison=S.read(S.checked(METRICS_REUSE_CHECK))
    require(comparison['passed'] is True and comparison['source']==E1_SOURCE and comparison['manifest']==MAMBA_MANIFEST and comparison['fit_count']==15 and comparison['role_checks']==45 and comparison['differences']==[] and comparison['tolerance'] is False,'Actual unmodified Windows metrics must exactly match all completed Linux roles')
    return {'metrics_reuse_comparison':METRICS_REUSE_CHECK,'original_parent_record':PARENT,'original_parent_status_preserved':'running','original_parent_elapsed_seconds':None,'released_owner':RELEASED,'Mamba_worker':MAMBA_WORKER,'Windows_failed_worker':WINDOWS_WORKER,'Windows_trace':WINDOWS_LOG,'Mamba_manifest':MAMBA_MANIFEST,'Mamba_completed_fits':artifacts,'Windows_failed_header_CPU_loads':1,'Windows_failed_model_constructions':0,'Windows_failed_forward_calls':0,'Windows_failed_optimizer_updates':0,'Windows_failed_attempt_elapsed_seconds':None,'Windows_inclusive_worker_seconds_nested_only':windows['inclusive_evaluator_worker_seconds'],'Mamba_inclusive_worker_seconds_nested_only':linux['inclusive_evaluator_worker_seconds'],'unpersisted_ledger_tail_unknown':True,'old_statuses_rewritten':False,'original_storage_source':STORAGE_SOURCE,'original_evaluator_source':E1_SOURCE,'path_bounds':path_bounds()}

def review_verified(binding):
    value=S.read(S.checked(binding))
    require(value.get('status')=='evaluation_storage_recovery_sources_reviewed' and value.get('passed') is True and value.get('reviewed_sources')==sources() and value.get('CPU_gate')==CPU_GATE,'Exact independently reviewed two-source recovery required')
    S.checked(value['reviewer_source'])
    return value

def verify_recovery():
    path=ROOT/'complete.json';value=S.read(path)
    require(value.get('schema')=='evaluation_storage_recovery_v1' and value.get('passed') is True and value.get('source')==S.desc(SOURCE) and value.get('sources')==sources() and value.get('CPU_gate')==CPU_GATE,'Exact evaluation-only recovery closure required')
    review_verified(value['source_review'])
    require(value['historical_evidence']==historical_metadata() and value['Mamba_models_to_execute']==0 and value['new_worker_modes']==['all_windows'] and value['old_E12_relabelled_complete'] is False,'Exact failed Windows closure and completed Mamba reuse required')
    return value

def create_closure(review_binding):
    RECOVERY.Q.check_pause();review_verified(review_binding)
    require(not(ROOT/'complete.json').exists(),'Immutable recovery closure already exists')
    RECOVERY.verify_gate(CPU_GATE)
    evidence=historical_metadata()
    value={'schema':'evaluation_storage_recovery_v1','passed':True,'source':S.desc(SOURCE),'sources':sources(),'source_review':review_binding,'CPU_gate':CPU_GATE,'historical_evidence':evidence,'Mamba_models_to_execute':0,'new_worker_modes':['all_windows'],'old_E12_relabelled_complete':False,'created_at':S.now()}
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'complete.json').open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return S.desc(ROOT/'complete.json')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--source-review');parser.add_argument('--source-review-sha256')
    args=parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'status':'preparation_only','model_operations':0,'Tensor_reads':0,'process_queries':0,'probability_payload_reads':0,'writes':0}));return 0
    require(args.source_review and args.source_review_sha256,'Actual independent source review required')
    print(json.dumps(create_closure({'path':args.source_review,'sha256':args.source_review_sha256})))
    return 0
if __name__=='__main__':raise SystemExit(main())
