"""Review actual recovery transaction evidence; no project imports or live queries."""
from pathlib import Path
import hashlib
import json
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
A = SOURCE.parent
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
Q = OUT / 'continuation_queue_v5'
FOLDER = Q / 'recoveries/process_loss_v3_1791767848030118400'
READS = []


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def desc(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix in ('.json', '.py', '.ps1'), 'text scope')
    raw = path.read_bytes()
    value = {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}
    READS.append(value)
    return value


def read(path_or_binding):
    if isinstance(path_or_binding, dict):
        path = WORK / path_or_binding['path']
        require(desc(path) == path_or_binding, 'descriptor mismatch: ' + str(path))
    else:
        path = Path(path_or_binding)
        desc(path)
    require(path.suffix == '.json', 'JSON only')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    review_path = A / 'independent_q5_recovery_v3_source_review_1791767704319970500.json'
    review = read(review_path)
    require(desc(review_path)['sha256'] == '1262fcf6ee237ff5ab26ca8c79c79d51138186442f4674766ffba0540d4392bc', 'source review')
    manifest_path = Q / 'recovery_registration_v3.json'
    manifest = read(manifest_path)
    require(desc(manifest_path)['sha256'] == '653df2183b1a9cdcd74755f69629d2cf89ca3de8a6b7e9bf26b7df0bfcabbaf4', 'manifest')
    require(manifest['sources'] == review['sources'] and len(manifest['sources']) == 88, 'exact reviewed88')
    require(manifest['source'] == review['reviewed_entry'] and manifest['independent_source_review'] == desc(review_path), 'review binding')
    previous = read(manifest['previous_recovery_registration'])
    require(len(previous['sources']) == 84 and manifest['sources'][:84] == previous['sources'], 'old84 preserved')
    require(manifest['actual_allowed_job_ids'] == ['LaBraM-native4s-lastblock-finetune-fit'], 'target job')
    require(manifest['evaluation_permanently_disabled_in_this_entry'] is True, 'no evaluation')
    require(manifest['engineering_and_optional_qualification_launches_allowed'] is False, 'no probes')
    require(manifest['training_protocol_or_hyperparameters_changed'] is False and manifest['same_formal_F3_and_E3_core'] is True, 'math unchanged')
    for key in ('base_queue_registration', 'registry', 'previous_recovery_closure', 'earlier_recovery_registration'):
        read(manifest[key])

    loss = read(manifest['loss_evidence'])
    cpu = read(manifest['checkpoint_cpu_resume_proof'])
    require(cpu['passed'] is True, 'actual CPU proof')
    closure_path = OUT / 'process_loss_closures/lost_f3_v3_1791767849813215500.json'
    closure = read(closure_path)
    require(desc(closure_path)['sha256'] == '83d4558818cd8727a4ec4b04d18ebfc18653aed6f3a03a611edb5a41f659e791', 'closure')
    require(closure['schema'] == 'lost_f3_process_closure_v3' and closure['passed'] is True, 'closure schema')
    require(closure['recovery_registration'] == desc(manifest_path), 'closure registration')
    require(closure['loss_evidence'] == manifest['loss_evidence'] and closure['checkpoint_cpu_resume_proof'] == manifest['checkpoint_cpu_resume_proof'], 'loss proof binding')
    require((closure['condition'], closure['seed'], closure['fold']) == ('LaBraM-native4s-lastblock-finetune', 33, 1), 'identity')
    for key in ('durable', 'observed_progress', 'known_extra_lower_bounds', 'expected_native_identities'):
        require(closure[key] == loss[key], 'loss details: ' + key)
    require(closure['durable'] == {'epoch': 54, 'optimizer_updates': 216, 'backward_calls': 1458}, 'durable')
    require(closure['observed_progress'] == {'epoch': 55, 'optimizer_updates': 220, 'backward_calls': 1485}, 'observed')
    require(closure['known_extra_lower_bounds'] == {'optimizer_updates': 4, 'backward_calls': 27}, 'extra subset')
    require(closure['known_extra_not_added_again_to_original_returned_counters'] is True, 'nonadditive extra')
    for key in ('actual_exit_code', 'actual_exit_time', 'cause', 'unflushed_tail_counters', 'final_elapsed_seconds', 'queue_attempt_final_elapsed_seconds'):
        require(closure[key] is None, 'unknown must stay null: ' + key)
    require(closure['graceful_terminal_flush'] is False, 'no invented terminal flush')
    require(loss['overnight_pause_not_training'] is True and loss['exact_active_elapsed_seconds'] is None, 'pause scope')
    preserved = {}
    for old, snapshot in (('original_attempt', 'attempt_snapshot'), ('old_queue_attempt_original', 'old_queue_attempt_snapshot')):
        left, right = read(closure[old]), read(closure[snapshot])
        require(closure[old]['sha256'] == closure[snapshot]['sha256'] and left == right, 'old attempt bytes')
        require(left['status'] == 'running', 'old running preserved')
        preserved[old] = closure[old]
        preserved[snapshot] = closure[snapshot]
    read(closure['old_owner_snapshot'])
    require(closure['old_owner_snapshot']['sha256'] == closure['old_owner_original']['sha256'] == loss['old_owner_snapshot']['sha256'], 'old owner snapshot')
    read(loss['old_owner_snapshot'])
    for key in ('history_snapshot', 'progress_snapshot', 'pause_native_proof', 'pause_capture', 'pause_progress_and_observer'):
        read(closure[key])
        require(closure[key] == loss[key], 'immutable pause/history binding')
    # Checkpoint descriptors are compared only. Never read latest.pt or snapshot .pt.
    require(closure['resume_checkpoint'] == loss['resume_checkpoint'] and closure['preserved_checkpoint'] == loss['preserved_checkpoint'], 'checkpoint descriptor equality')
    require(closure['resume_checkpoint']['sha256'] == closure['preserved_checkpoint']['sha256'] == 'd4bd7dcf953610087cea9f488af4a080233d1f5281d4da5e752630bddd4d39ee', 'retained checkpoint digest')

    ready_path = FOLDER / 'recovery_ready.json'
    ready = read(ready_path)
    require(ready['passed'] is True and ready['checkpoint_epoch'] == 54 and ready['natural_handoff_claimed'] is False, 'ready scope')
    require(ready['recovery_registration'] == desc(manifest_path) and ready['recovery_closure'] == desc(closure_path), 'ready linkage')
    require(ready['recovery_execution_source'] == manifest['source'] and ready['recovery_execution_version'] == 3, 'source/version')
    ab_bindings = [manifest['runtime_process_absence_evidence'], ready['second_runtime_process_absence']]
    observations = []
    for binding in ab_bindings:
        obs = read(binding)
        require(obs['schema'] == 'q5_process_absence_observation_v3' and obs['source'] == manifest['source'], 'runtime observation source')
        require(obs['query_succeeded'] is True and obs['all_expected_processes_absent'] is True, 'actual absence')
        require(obs['expected_identities'] == loss['expected_native_identities'], 'exact old births')
        require(set(obs['absent_pids']) == {34592, 10720, 23748, 9840}, 'four absent PIDs')
        require(obs['other_training_workers'] == obs['workspace_training_workers'] == obs['observed_rows'] == obs['exact_getprocess_alive_pids'] == [], 'no competing worker')
        require(obs['workspace_substring_filter_used'] is False and obs['native_birth_api'] == 'GetProcessTimes', 'scope and exact birth')
        allowed = obs['allowed_recovery_processes']
        require([(x['pid'], x['parent_pid'], x['creation_time_100ns']) for x in allowed] == [(30824, 32352, 134362414478907013), (32352, 15212, 134362414478479312)], 'new exact parent/child')
        require(all('recover_q5_after_process_loss_v3 --run' in x['command_line'] for x in allowed), 'entry commands')
        gpu = obs['gpu_and_wsl_admission']
        require(gpu['contexts'] == [{'pid': 14024, 'type': 'C+G', 'process_name': 'C:\\Program Files\\Tencent\\QQNT\\QQ.exe'}], 'only known graphics')
        require(gpu['running_wsl_distributions'] == gpu['wsl_process_inventories'] == [], 'no running WSL')
        require(gpu['model_operations'] == gpu['process_controls'] == gpu['stopped_distro_start_requests'] == 0, 'readonly inventory')
        graphics = gpu['verified_external_graphics']
        require(len(graphics) == 1 and graphics[0]['evidence'] == manifest['external_graphics_evidence'], 'bound graphics proof')
        require(graphics[0]['GetProcessTimes_creation_100ns'] == graphics[0]['actual_process']['creation_time_100ns'] == 134362401342035061, 'graphics exact birth')
        observations.append(obs)
    require(observations[0]['allowed_recovery_processes'] == observations[1]['allowed_recovery_processes'], 'same admission identity twice')
    require(observations[0]['observed_at'] < observations[1]['observed_at'] < ready['prepared_at'], 'two actual observations in order')
    require(closure['runtime_process_absence_evidence'] == ab_bindings[0], 'first fresh admission linkage')
    intent_path = Q / 'recovery_intent_v3.json'
    intent = read(intent_path)
    require(intent['pid'] == 30824 and intent['source'] == manifest['source'] and intent['loss_evidence'] == manifest['loss_evidence'], 'exclusive intent')
    report = {
        'schema': 'independent_actual_q5_recovery_v3_transaction_review', 'passed': True,
        'reviewer': '/root/loss3_independent_review', 'source': desc(SOURCE),
        'source_review': desc(review_path), 'manifest': desc(manifest_path), 'closure': desc(closure_path),
        'ready': desc(ready_path), 'exclusive_intent': desc(intent_path), 'fresh_absence_evidence': ab_bindings,
        'reviewed_sources_count': 88, 'unchanged_previous_source_count': 84,
        'checkpoint_boundary': closure['durable'], 'last_returned_observation': closure['observed_progress'],
        'known_extra_subset_not_additive': closure['known_extra_lower_bounds'],
        'old_attempts_still_running_original_bytes': preserved,
        'unknown_exit_time_code_cause_elapsed_tail_preserved': True,
        'no_natural_handoff_or_complete_claim': True, 'evaluation_permanently_disabled_in_entry': True,
        'checkpoint_tensor_reads': 0, 'checkpoint_binary_hash_reads': 0, 'GPU_queries': 0,
        'process_queries': 0, 'process_actions': 0, 'model_operations': 0, 'test_probability_reads': 0,
        'read_scope': 'Existing immutable JSON evidence and own Python source only, no project imports. GPU/process rows are recorded evidence, not new measurements. Current owner/progress and mutable latest.pt are not re-read.',
        'limits': 'Verifies actual recovery transaction and admission evidence, not current training health, newly completed fits, full plan completion or final acceptance. Current training health remains independently observed by parent.',
        'reads': READS,
    }
    out = A / ('independent_actual_q5_recovery_v3_transaction_review_' + str(time.time_ns()) + '.json')
    with out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write('\n')
    print(json.dumps({'passed': True, 'report': desc(out)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
