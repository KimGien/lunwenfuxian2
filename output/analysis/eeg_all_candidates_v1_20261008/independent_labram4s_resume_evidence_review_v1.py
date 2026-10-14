"""Independent existing-evidence review, with no project imports or tensor reads."""
from pathlib import Path
import ast
import hashlib
import json
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
A = SOURCE.parent
READS = []


def require(ok, message):
    if not ok:
        raise ValueError(message)


def desc(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix in ('.json', '.py', '.ps1'), 'Restricted text/source reader')
    raw = path.read_bytes()
    binding = {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}
    READS.append(binding)
    return binding


def read(binding):
    path = WORK / binding['path']
    require(path.suffix == '.json' and desc(path) == binding, 'JSON bytes differ')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    proof_path = A / 'recovery_cpu_preflight_labram4s33f1_1791767054086345800.json'
    binding = desc(proof_path)
    require(binding['sha256'] == '2b575b700a086b5a14e668b0f8af563dae81173a5f1f7f0215d9ba7657d6f32e', 'Exact actual proof required')
    proof = read(binding)
    require(proof['schema'] == 'actual_cpu_recovery_preflight_labram4s33f1_v1' and proof['passed'] is True, 'Actual corrected schema')
    require((proof['condition'], proof['seed'], proof['fold']) == ('LaBraM-native4s-lastblock-finetune', 33, 1), 'Actual fit identity')
    require(proof['checkpoint_loads'] == 2 and type(proof['checkpoint_loads']) is int, 'Actual two CPU loads')
    for key in ('model_constructions', 'GPU_operations', 'raw_arrays_loaded', 'test_probability_reads', 'process_operations'):
        require(type(proof[key]) is int and proof[key] == 0, 'Unexpected actual operation: ' + key)
    require(desc(WORK / proof['source']['path']) == proof['source'], 'Actual producer changed')
    require(desc(WORK / proof['helper_source']['path']) == proof['helper_source'] and proof['helper_source']['sha256'] == '6c0c6d084b5463ed3cfa2326cc17bb105fa3a9369fd75527d4912d198489071c', 'Sealed validator changed')
    tree = ast.parse((WORK / proof['source']['path']).read_text(encoding='utf-8'))
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    require({'cpu', 'CUDA initialization forbidden', 'Two distinct original CPU checkpoints only'} <= strings or {'cpu', 'GPU initialization forbidden', 'Two distinct original CPU checkpoints only'} <= strings, 'CPU-only producer guards absent')
    pause = read(proof['pause_native_proof'])
    absence = read(proof['process_absence_evidence'])
    require(pause['status'] == 'paused_by_explicit_user' and absence['schema'] == 'q5_process_absence_observation_v3', 'Actual pause/absence schemas')
    require(absence['expected_native_identity_artifact'] == proof['pause_native_proof'] and absence['expected_native_identities'] == pause['identities'], 'Full pause identities differ')
    ids = [v['pid'] for v in pause['identities']]
    require(len(ids) == len(set(ids)) == 4 and absence['queried_pids'] == absence['absent_pids'] == ids, 'Four precise historical objects')
    require(absence['query_succeeded'] is True and absence['all_expected_processes_absent'] is True and absence['workspace_substring_filter_used'] is False and absence['workspace_training_workers'] == absence['observed_rows'] == absence['exact_getprocess_alive_pids'] == [], 'Actual broad absence evidence')
    for key in ('exit_codes', 'actual_exit_time', 'reason', 'actual_exit_code', 'cause'):
        require(absence[key] is None, 'Absence must not invent endpoint')
    for key in ('unflushed_tail_counters', 'final_elapsed_seconds', 'exit_code', 'actual_exit_time', 'reason'):
        require(proof[key] is None, 'Actual CPU proof unknown fields')
    snapshot = read(proof['snapshot_manifest'])
    require(snapshot['source'] == proof['source'] and snapshot['absence'] == proof['process_absence_evidence'] and snapshot['identity'] == [proof['condition'], 33, 1], 'Snapshot source/identity')
    retained = snapshot['retained']
    for name, pair in retained.items():
        require(pair['original']['sha256'] == pair['snapshot']['sha256'], 'Snapshot descriptor byte identity: ' + name)
        if Path(pair['snapshot']['path']).suffix == '.json':
            read(pair['snapshot'])
    trace = read(retained['attempt']['snapshot'])
    parent = read(retained['queue_attempt']['snapshot'])
    owner = read(retained['owner']['snapshot'])
    progress = read(retained['progress.json']['snapshot'])
    history = read(retained['train_history.json']['snapshot'])
    require(trace['status'] == parent['status'] == 'running' and trace['graceful_terminal_flush'] is False, 'Original records remain running')
    for record in (trace, parent):
        require(record.get('seconds') is None and record.get('finished_at') is None, 'No inferred lost elapsed')
    require(owner == pause['owner_snapshot'] and owner['pid'] in ids and owner['child_pid'] in ids, 'Archived owner agrees with real pause')
    durable = proof['durable']
    require(durable == {'epoch': 54, 'optimizer_updates': 216, 'backward_calls': 1458}, 'Actual durable boundary')
    require(progress == proof['observed_progress'] and progress['epoch'] == 55 and progress['last_durable_epoch'] == 54, 'Observed progress differs')
    require(trace['optimizer_step_returned'] == progress['optimizer_updates_completed'] == 220 and trace['backward_returned'] == progress['backward_calls_completed'] == 1485, 'Persisted physical returns')
    require(proof['known_extra_lower_bounds'] == {'optimizer_updates': 220 - 216, 'backward_calls': 1485 - 1458} and proof['known_extra_not_added_again_to_original_returned_counters'] is True, 'Event delta is a subset, not added again')
    rows = history['history'] if isinstance(history, dict) else history
    require(len(rows) == 54 and all(rows[-1][k] == v for k, v in durable.items()), 'Durable history boundary')
    valid = proof['checkpoint_validation']
    require(valid['checkpoint_load_device'] == 'cpu' and valid['cpu_checkpoint_loads'] == 2 and all(valid[k] == v for k, v in durable.items()), 'CPU checkpoint boundary')
    for key in ('resume_full_state_verified', 'optimizer_group_order_and_moment_shapes_verified', 'best_selected_parameter_finiteness_verified', 'cpu_rng_local_restore_and_cuda_saved_byte_shapes_verified', 'raw_role_descriptors_verified_without_loading_raw'):
        require(valid[key] is True, 'Missing full-state guard: ' + key)
    require(retained['latest.pt'] == {'original': proof['resume_checkpoint'], 'snapshot': proof['preserved_checkpoint']} and proof['resume_checkpoint'] in valid['original_checkpoint_bindings'], 'Checkpoint descriptors match actual CPU proof')
    require(proof['selected_terminal_changes'] == {'selected_encoder_changed': 19, 'selected_head_changed': 6, 'latest_encoder_changed': 19, 'latest_head_changed': 6, 'selected_epoch': 52, 'selected_phase': 'lastblock_finetune'}, 'Actual selected/terminal distinction')
    result = {'schema': 'independent_actual_labram4s_loss_cpu_evidence_review_v1', 'passed': True, 'reviewer': '/root/loss3_independent_review', 'source': desc(SOURCE), 'actual_cpu_proof': binding, 'read_bindings': list({x['path']: x for x in READS}.values()), 'durable': durable, 'observed': {'epoch': 55, 'optimizer_updates': 220, 'backward_calls': 1485}, 'extra_lower_bound_subset_not_added': proof['known_extra_lower_bounds'], 'scope': 'Independent source plus existing JSON evidence review; checkpoint tensor/state conclusions rely on source-bound actual two-load producer proof.', 'checkpoint_loads_this_review': 0, 'checkpoint_binary_reads_this_review': 0, 'model_operations': 0, 'GPU_operations': 0, 'process_actions': 0, 'test_probability_reads': 0, 'recovery_launched': False, 'all_plan_completed': False}
    path = A / ('independent_actual_labram4s_loss_cpu_evidence_review_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': desc(path), 'passed': True}, ensure_ascii=True))


if __name__ == '__main__':
    main()
