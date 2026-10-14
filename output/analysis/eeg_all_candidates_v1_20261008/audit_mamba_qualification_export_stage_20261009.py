"""Read-only artifact audit of completed EEG2Rep supervision and Mamba stage0.

Uses standard library only. Does not construct models, deserialize tensor
checkpoints, read raw EEG, read classifier probabilities or execute GPU work.
Frozen external representation .npy bytes are checked for shape/finiteness.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ast
import hashlib
import json
import math
import struct
import time

WORK = Path(__file__).resolve().parents[3]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT = WORK / 'output/analysis/eeg_all_candidates_v1_20261008'
CONDITION = 'EEGMamba-frozen-smallhead'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def desc(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and '/evaluation/' not in path.as_posix(), 'Forbidden artifact location')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest}


def checked(record):
    require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, 'Invalid descriptor')
    path = (WORK / record['path']).resolve()
    require(desc(path) == record, 'Artifact hash differs: ' + record['path'])
    return path


def descriptors(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            yield value
        else:
            for child in value.values():
                yield from descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from descriptors(child)


def finite_npy(path):
    require(path.resolve().is_relative_to((OUT / 'frozen_features' / CONDITION / 'windows').resolve()), 'Only external frozen Mamba representations may be read')
    with path.open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'Invalid NPY magic')
        version = tuple(stream.read(2))
        require(version in ((1,0),(2,0),(3,0)), 'Unsupported NPY format')
        header_len = struct.unpack('<H' if version == (1,0) else '<I', stream.read(2 if version == (1,0) else 4))[0]
        header = ast.literal_eval(stream.read(header_len).decode('utf-8' if version == (3,0) else 'latin1'))
        require(header['shape'] == (60,200) and header['descr'] == '<f4' and header['fortran_order'] is False, 'Native Mamba feature contract differs')
        data = stream.read()
        require(len(data) == 60 * 200 * 4, 'Unexpected feature byte count')
        values = struct.unpack('<12000f', data)
        require(all(math.isfinite(value) for value in values), 'Nonfinite native feature')
    return {'shape': [60,200], 'dtype': 'float32_little_endian', 'finite_values': len(values), 'data_bytes': len(data)}


def queue_attempt(job):
    paths = [p for p in (OUT / 'continuation_queue_v3/attempts').glob('*-' + job + '/status.json')
             if read(p).get('status') == 'complete']
    require(len(paths) == 1, 'Expected one completed observed Mamba stage attempt: ' + job)
    path = paths[0]; value = read(path)
    require(value['returncode'] == 0 and value['job']['module'] == 'linux_entry_v2'
            and value['job']['id'] == job and value['registration'] == desc(OUT / 'continuation_queue_v3/registration.json'), 'Q3 completed attempt binding differs')
    checked(value['log'])
    owner_arg = value['command'][value['command'].index('--owner-record') + 1]
    owner_path = (WORK / owner_arg).resolve(); owner = read(owner_path)
    require(owner_path.is_relative_to((OUT / 'continuation_queue/attempts').resolve())
            and owner['action'] == job and owner['pid'] == owner['pgid'] and owner['pid'] > 0, 'Linux actual group-owner binding differs')
    return {'attempt': desc(path), 'log': value['log'], 'linux_owner': desc(owner_path),
            'status': 'complete', 'returncode': 0, 'dispatcher_seconds': value['seconds'],
            'started_at': value['started_at'], 'finished_at': value['finished_at'],
            'cost_scope': 'Outer Windows/WSL launcher wall time; overlaps inner measured work, never added to it.'}


def main():
    registration = read(OUT / 'registration.json'); reg = desc(OUT / 'registration.json')
    q3_path = OUT / 'continuation_queue_v3/registration.json'; q3 = read(q3_path)
    require(len(q3['sources']) == 47, 'Q3 source count differs')
    for source in q3['sources']:
        checked(source)

    supervised_path = REPORT / 'eeg2rep_supervised_cumulative_audit_15fits_949f54a9169e.json'
    sup = read(supervised_path)
    require(sup['passed'] and sup['all15_complete'] and sup['completed_fits_verified'] == sup['planned_fits'] == 15
            and sup['registration'] == reg and sup['condition'] == 'EEG2Rep-supervised', 'Full supervised audit identity/count differs')
    rows = sup['completed_fit_audits']
    require({(r['seed'],r['fold']) for r in rows} == {(s,f) for s in (11,22,33) for f in range(1,6)}, 'Full supervised fold coverage differs')
    require(sum(r['optimizer_updates'] for r in rows) == sup['cumulative_optimizer_updates'] == 5856
            and sum(r['backward_calls'] for r in rows) == sup['cumulative_backward_calls'] == 38920, 'Full supervised aggregate cost differs')
    require(all(sup[key] == 0 for key in ('gpu_calls','model_constructions','optimizer_executions','outer_test_signal_reads','outer_test_probability_reads')), 'Original CPU audit scope differs')
    flags = ('exact_adamw_parameter_groups_states_steps_and_fp32_moments_verified', 'exact_best_selected_scope_and_digest_verified',
             'same_fold_ssl_best_bitwise_initialization_verified', 'selected_and_terminal_input_context_updates_verified',
             'target_predictor_frozen_at_initial_best_terminal', 'full_curves_stop_and_recovery_artifacts_verified',
             'recorded_train_only_prediction_reload_proof_verified')
    for row in rows:
        require(row['passed'] and all(row[key] is True for key in flags)
                and row['independent_prediction_forward_performed'] is False, 'Supervised audit proof flags differ')
        completed = OUT / 'runs/EEG2Rep-supervised' / ('seed_' + str(row['seed'])) / ('fold_' + str(row['fold'])) / 'complete.json'
        require(read(completed) == row['evidence'], 'Supervised audit no longer binds actual completed fit')
    sup_sources = list({r['path']:r for r in descriptors(sup)}.values())
    for source in sup_sources:
        checked(source)

    gpu_path = REPORT / 'eegmamba_official_gpu_check.json'; gpu = read(gpu_path); audit = gpu['load_audit']
    require(gpu['passed'] and gpu['formal_fits'] == 0 and gpu['engineering_optimizer_steps'] == 1
            and gpu['gpu_encode_calls'] == 4 and gpu['official_mamba2_block_count'] == 12, 'Actual Mamba qualification counts differ')
    require(audit['strict'] is True and audit['loaded_keys_before_feature_switch'] == 117
            and audit['missing_keys'] == audit['unexpected_keys'] == [] and audit['feature_dim'] == 200
            and len(audit['official_mamba2_classes']) == 12
            and set(audit['official_mamba2_classes']) == {'mamba_ssm.modules.mamba2.Mamba2'}, 'Official architecture/weight coverage differs')
    require(gpu['feature_shape'] == [2,200] and gpu['single_shape'] == [1,200] and gpu['repeat_max_abs_error'] == 0
            and gpu['frozen_parameters_unchanged_after_step'] is True, 'Mamba output/repeat/frozen qualification differs')
    require(len(audit['finetune_parameter_names']) == 10 and set(audit['finetune_parameter_names']) == set(gpu['finetune_gradient_norms'])
            and all(math.isfinite(v) and v > 0 for v in gpu['finetune_gradient_norms'].values()), 'Not all ten actual Mamba parameters have finite positive gradient norms')
    require(len(gpu['changed_parameter_names']) == 7 and set(gpu['changed_parameter_names']).issubset(audit['finetune_parameter_names']), 'Actual synthetic SGD changed-name scope differs')
    runtime = audit['runtime']
    require({k:runtime[k] for k in ('torch','torch_cuda','mamba_ssm','causal_conv1d','triton')}
            == {'torch':'2.5.1+cu124','torch_cuda':'12.4','mamba_ssm':'2.2.4','causal_conv1d':'1.5.0.post8','triton':'3.1.0'}, 'Qualified Linux runtime differs')
    require(gpu['peak_gpu_allocated_bytes'] == 309073408 and gpu['seconds'] > 0, 'Qualification resource record differs')
    ledger_paths = list((OUT / 'engineering_attempts/EEGMamba').glob('*.json'))
    require(len(ledger_paths) == 1, 'Unexpected additional Mamba qualification attempts need review')
    ledger = read(ledger_paths[0])
    require(ledger['status'] == 'complete' and ledger['passed'] is True
            and ledger['encode_started'] == ledger['encode_returned'] == 4
            and ledger['backward_started'] == ledger['backward_returned'] == 1
            and ledger['sgd_step_started'] == ledger['sgd_step_returned'] == 1
            and ledger['formal_fits'] == ledger['formal_optimizer_updates'] == 0
            and ledger['cohort_data_or_labels_used'] is False, 'Actual started/returned qualification cost differs')
    checked(ledger['source']); checked(ledger['check_source'])
    assets = read(OUT / 'assets/download_manifest.json'); author = assets['sources']['eegmamba']
    require(audit['source']['repository'] == author['repository'] and audit['source']['revision'] == author['revision'], 'Author pinned source identity differs')
    author_sources=[]
    for source in author['files']:
        if source['path'].endswith('.py'):
            value = {k:source[k] for k in ('path','sha256')};checked(value);author_sources.append(value)
    weight = {k:audit['checkpoint'][k] for k in ('path','sha256')}; checked(weight)
    require((WORK / weight['path']).stat().st_size == audit['checkpoint']['bytes'], 'Official checkpoint byte count differs')
    qual_queue = queue_attempt('mamba-check'); export_queue = queue_attempt('mamba-export')

    root = OUT / 'frozen_features' / CONDITION; manifest_path = root / 'manifest.json'; manifest = read(manifest_path)
    export_reg_path = root / 'export_registration.json'; export_reg = read(export_reg_path)
    require(export_reg['condition'] == CONDITION and export_reg['registry'] == reg
            and export_reg['batch_size'] == 4 and export_reg['dtype'] == 'float32'
            and export_reg['encoder_fit_subject_ids'] == [] and export_reg['optimizer_updates'] == 0,
            'Frozen export registration differs')
    checked(export_reg['producer']); checked(export_reg['adapter']); checked(export_reg['raw_manifest'])
    require(export_reg['channel_names'] == registration['configuration']['data']['channels'], 'Channel order differs')
    require(manifest['passed'] is True and manifest['condition'] == CONDITION and manifest['registry'] == reg
            and manifest['subject_ids'] == registration['subject_ids'] and len(manifest['records']) == 37
            and [r['subject_id'] for r in manifest['records']] == registration['subject_ids'], 'Exported37-person coverage/order differs')
    require(manifest['encoder_fit_subject_ids'] == [] and manifest['dimension'] == 200 and manifest['input_seconds'] == 5
            and manifest['native_dimension_retained'] is True and manifest['frozen_state_exact'] is True
            and manifest['frozen_state_before'] == manifest['frozen_state_after'], 'Frozen external native state/input contract differs')
    require(manifest['input_contract'] == audit['input_contract'] and manifest['loading_audit'] == audit,
            'Export used a different qualified Mamba adapter/weight/runtime')
    raw = {r['subject_id']:r['raw'] for r in registration['raw_sources']}
    raw_paths = {r['path'] for r in raw.values()}; verified_nonraw=[]
    for source in manifest['sources']:
        if source['path'] in raw_paths:
            require(source in raw.values(), 'Accepted raw descriptor differs')
        else:
            checked(source); verified_nonraw.append(source)
    feature_proofs=[]
    for row in manifest['records']:
        sid=row['subject_id']; sidecar=read(checked(row['sidecar'])); feature=checked(row['features'])
        require(sidecar['subject_id'] == sid and sidecar['raw'] == raw[sid]
                and sidecar['features'] == row['features'] and sidecar['export_registration'] == desc(export_reg_path)
                and sidecar['shape'] == [60,200] and sidecar['repeat_exact'] is True and sidecar['encoder_fit_subject_ids'] == [], 'Subject sidecar identity/contract differs')
        feature_proofs.append({'subject_id':sid,'feature':row['features'],'sidecar':row['sidecar'],**finite_npy(feature)})
    export_ledgers=[read(checked(source)) for source in manifest['execution_attempts']]
    require(len(export_ledgers) == 1, 'Multiple export attempts require explicit partial-work reconciliation')
    el=export_ledgers[0]; expected_new=37*60; expected_repeat=37*4; expected_batches=37*(60//4+1)
    require(el['status'] == 'complete' and el['optimizer_updates'] == 0
            and el['started_windows'] == el['returned_windows'] == expected_new+expected_repeat
            and el['started_batches'] == el['returned_batches'] == expected_batches, 'Actual export attempt counters differ')
    require(manifest['current_attempt_counts'] == {'new_windows':expected_new,'repeat_windows':expected_repeat,'encoder_forward_batches':expected_batches}
            and manifest['new_encoder_forwards'] == expected_batches, 'Export manifest counts differ from independently recomputed budget')
    require(manifest['seconds_this_attempt'] > 0 and manifest['peak_allocated_bytes'] == 308909056, 'Export real resource record differs')
    lock_path=OUT/'engineering'/(CONDITION+'.json'); lock=read(lock_path); contract=lock['contract']
    canon=hashlib.sha256(json.dumps(contract,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(lock['condition'] == CONDITION and lock['registration'] == reg and lock['configuration_sha256'] == canon, 'Frozen head source lock identity/hash differs')
    require(all(contract[key] is True for key in ('forward_checked','gradient_checked','resume_checked','training_only_engineering_checked'))
            and contract['engineering_optimizer_updates'] == 0 and contract['engineering_gradient_passes'] == 1
            and contract['parameters'] == 26386 and len(contract['trainable_parameter_names']) == 6
            and contract['frozen_parameter_names'] == [] and contract['probe_subject_ids'] == registration['folds']['fold_1']['train'][:8], 'Frozen head qualification scope differs')
    require(contract['weight_sources'] == [desc(manifest_path)] and contract['input_contract']['dimension'] == 200
            and contract['input_contract']['input_seconds'] == 5, 'Frozen head must bind completed native Mamba export')
    for source in contract['implementation_sources'] + contract['weight_sources']:
        checked(source)
    resource_missing=[name for name in ('seconds','forward_backward_seconds','peak_allocated_bytes') if name not in contract]
    result={'kind':'completed_supervision_and_mamba_stage0_independent_artifact_audit', 'passed_for_this_completed_stage':True,
            'full_plan_completed':False,'captured_at_Asia_Shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'source':desc(__file__),'registration':reg,'q3_registration':desc(q3_path),'q3_source_files_verified':47,
            'scope':{'model_constructions':0,'tensor_checkpoint_deserializations':0,'gpu_calls':0,'optimizer_executions':0,
                     'raw_signal_reads':0,'outer_test_probability_or_classifier_metric_reads':0,
                     'frozen_external_feature_arrays_byte_checked':37,'sealed_source_or_old_artifact_writes':0},
            'eeg2rep_supervised_report_review':{'report':desc(supervised_path),'source':sup['source'],'source_and_evidence_descriptors_checked':len(sup_sources),
                'completed_fits':15,'durable_optimizer_updates':5856,'durable_backward_calls':38920,
                'checks':list(flags),'per_fit_updates':[{'seed':r['seed'],'fold':r['fold'],'epochs':r['epochs'],'best_epoch':r['best_epoch'],
                             'updates':r['optimizer_updates'],'backward_calls':r['backward_calls'],'selected_head_changed_count':r['selected_head_changed_count']} for r in rows],
                'scope_limit':'This review rechecks source/artifact hashes and aggregates; it does not repeat the already completed CPU tensor audit or independently rerun prediction forwards.',
                'physical_cost_caveat':'5856/38920 are unique durable trajectory counts. Preserve the separate pause recovery extra4optimizer updates/27backward calls and unknown diagnostic/time cost.'},
            'mamba_qualification':{'report':desc(gpu_path),'attempt':desc(ledger_paths[0]),'queue':qual_queue,'official_weight':weight,
                'author_python_sources_verified':author_sources,'runtime':runtime,'official_mamba2_blocks':12,'strict_loaded_keys':117,
                'encode_calls':4,'encoded_synthetic_windows_from_fixed_source':7,'backward_calls':1,'synthetic_sgd_steps':1,'formal_fits':0,
                'all_ten_selected_parameters_positive_finite_gradient':True,'bitwise_changed_names':gpu['changed_parameter_names'],
                'unchanged_selected_names':sorted(set(audit['finetune_parameter_names'])-set(gpu['changed_parameter_names'])),
                'seconds':gpu['seconds'],'peak_allocated_bytes':gpu['peak_gpu_allocated_bytes'],
                'limitation':'Synthetic SGD changed7of10 selected tensors; this is not proof that every formal fine-tune parameter changed. Actual formal terminal proof remains mandatory.'},
            'mamba_export':{'manifest':desc(manifest_path),'export_registration':desc(export_reg_path),'queue':export_queue,
                'nonraw_sources_checked':verified_nonraw,'accepted_raw_descriptor_bindings_checked_without_reading_raw':37,
                'features':feature_proofs,'execution_attempts':manifest['execution_attempts'],'new_windows':expected_new,'repeat_windows':expected_repeat,
                'total_encoder_windows':expected_new+expected_repeat,'encoder_forward_batches':expected_batches,'optimizer_updates':0,
                'frozen_state':manifest['frozen_state_after'],'seconds':manifest['seconds_this_attempt'],'peak_allocated_bytes':manifest['peak_allocated_bytes'],
                'repeat_scope':'All37 sidecars attest exact repeat under the sealed source equality assertion. This read-only audit does not repeat model inference.',
                'cost_scope':'Actual export timer begins after model load and ledger setup; Windows dispatcher and model-import/load intervals must remain separate/unknown as appropriate, not added twice.'},
            'mamba_frozen_head_engineering':{'source_lock':desc(lock_path),'input_contract':contract['input_contract'],'parameters':26386,
                'trainable_parameter_names':contract['trainable_parameter_names'],'probe_subject_ids':contract['probe_subject_ids'],
                'optimizer_updates':0,'gradient_passes':1,'missing_original_probe_resource_fields':resource_missing,
                'resource_gap_resolution':'12th condition requires new independent zero-update resource measurement after full360/Q3/E2 completion, using frozen supplemental runnerv2. No backfill.'},
            'remaining_stage_claims_not_made':['EEGMamba frozen15formal fits all complete','Any of9fine-tune routes complete','All360complete','Outer-test evaluation complete'],
            'timestamp_interpretation':'Windows dispatcher and WSL timestamps may differ slightly; sequential child wait/dependency source control is reviewed, not inferred from subsecond cross-OS wall-clock ordering.'}
    destination=REPORT/('eeg2rep_mamba_stage_audit_'+str(time.time_ns())+'.json')
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'artifact':desc(destination),'supervised_fits':15,'supervised_durable_updates':5856,'mamba_export_features':37,
                      'mamba_export_forward_batches':expected_batches,'mamba_export_total_windows':expected_new+expected_repeat,
                      'new_auditor_gpu_calls':0,'full_plan_completed':False}))


if __name__=='__main__':
    main()
