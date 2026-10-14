"""Assemble the actual Oct11 loss context from already saved CPU/JSON evidence.

Standard-library JSON/source I/O only. Does not invoke project verifiers,
deserialize tensors, query/control processes or create a recovery closure.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
CPU = SOURCE.with_name('recovery_cpu_preflight_luna33f3_1791705459339343500.json')
CPU_SHA = '80b8a90beb18188eb1358ebf8306ed720d17fb6e3c7acc239c263cab3542722f'
SNAPSHOT_SHA = '7e1a6f5d7c0889cf84f3efb75999f7111f64e2754cf214784193ca09e29b1f51'
ABSENCE_SHA = '0752ae58764afcc749b2619eff6fb535fd3026c8e9ecacda1af817ba33c5400a'
NATIVE_SHA = 'c2b499e646c93698c4baee377be514104b0558b382d0153a2e5b19410a38c83c'
IDENTITY = ('LUNA-lastblock-finetune', 33, 3)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def desc(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix in ('.json', '.py'), 'Only workspace JSON/source allowed')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(binding):
    path = (WORK / binding['path']).resolve()
    require(path.suffix == '.json' and desc(path) == binding, 'JSON proof binding changed')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'JSON changed during read')
    return json.loads(raw.decode('utf-8-sig'))


def assemble():
    own_source = desc(SOURCE)
    cpu_binding = desc(CPU); require(cpu_binding['sha256'] == CPU_SHA, 'Actual CPU proof changed')
    cpu = read(cpu_binding)
    require(cpu.get('schema') == 'actual_cpu_recovery_preflight_luna33f3_v1' and cpu.get('passed') is True
            and tuple(cpu.get(k) for k in ('condition', 'seed', 'fold')) == IDENTITY
            and type(cpu['checkpoint_loads']) is int and cpu['checkpoint_loads'] == 2
            and all(type(cpu[k]) is int and cpu[k] == 0 for k in
                    ('model_constructions', 'GPU_operations', 'raw_arrays_loaded', 'test_probability_reads', 'process_operations')),
            'Actual two-load CPU proof required')
    require(desc(WORK / cpu['source']['path']) == cpu['source']
            and desc(WORK / cpu['helper_source']['path']) == cpu['helper_source'], 'CPU producer/helper source changed')
    require(cpu['snapshot_manifest']['sha256'] == SNAPSHOT_SHA and cpu['process_absence_evidence']['sha256'] == ABSENCE_SHA,
            'Actual loss snapshot or absence proof changed')
    snapshot = read(cpu['snapshot_manifest']); absence = read(cpu['process_absence_evidence'])
    native_binding = absence['expected_native_identity_artifact']
    require(native_binding['sha256'] == NATIVE_SHA, 'Historical native identity proof changed')
    native = read(native_binding)
    identities = absence['expected_native_identities']
    require(identities == native['identities'] and len(identities) == 4
            and absence['query_succeeded'] is True and absence['all_expected_processes_absent'] is True
            and absence['queried_pids'] == absence['absent_pids'] == [x['pid'] for x in identities]
            and absence['observed_rows'] == absence['workspace_training_workers'] == []
            and all(k in absence and absence[k] is None for k in ('exit_codes', 'actual_exit_time', 'reason')),
            'Exact four-object absence proof required; endpoint remains unknown')
    retained = snapshot['retained']
    require(snapshot['schema'] == 'q5_loss_byte_snapshot_v2' and snapshot['identity'] == list(IDENTITY)
            and snapshot['absence'] == cpu['process_absence_evidence'] and snapshot['source'] == cpu['source'], 'Snapshot provenance differs')
    for pair in retained.values():
        require(pair['original']['sha256'] == pair['snapshot']['sha256'], 'Retained copy hash differs')
        if Path(pair['original']['path']).suffix == '.json':
            require(read(pair['original']) == read(pair['snapshot']), 'Original JSON/snapshot changed')
    progress = read(retained['progress.json']['snapshot']); trace = read(retained['attempt']['snapshot'])
    durable = cpu['durable']
    observed = {'epoch': progress['epoch'], 'optimizer_updates': progress['optimizer_updates_completed'],
                'backward_calls': progress['backward_calls_completed']}
    require(trace['status'] == 'running' and trace['graceful_terminal_flush'] is False
            and trace['unflushed_tail_unknown'] is True and trace['unflushed_tail_counters'] is None
            and trace.get('seconds') is None and trace.get('finished_at') is None, 'Lost trace endpoint cannot be fabricated')
    require(cpu['buffered_trace_pair'] == retained['attempt'] and cpu['resume_checkpoint'] == retained['latest.pt']['original']
            and cpu['preserved_checkpoint'] == retained['latest.pt']['snapshot']
            and cpu['durable_history_matches_current_and_snapshot'] is True
            and cpu['observed_progress'] == progress, 'CPU-verified snapshots differ')
    extra = {k: observed[k] - durable[k] for k in ('optimizer_updates', 'backward_calls')}
    require(extra == cpu['known_extra_lower_bounds'] and all(type(v) is int and v >= 0 for d in (durable, observed, extra) for v in d.values()),
            'Known replay lower bounds differ')
    previous = desc(OUT / 'continuation_queue_v5/recovery_registration.json')
    base = desc(OUT / 'continuation_queue_v5/registration.json')
    require(previous['sha256'] == 'ec1500ce2671600fca1588e41f14fc559e12300824aaa5c4a2625671f8d2ec9f'
            and base['sha256'] == 'a9db8e215f1d32cd215d550319c0fa1646985c54d2167783a7547b3a6ea5312f', 'Original execution registration changed')
    value = {'schema': 'q5_f3_process_loss_evidence_v2', 'passed': True, 'source': own_source,
        'assembled_at': datetime.now(timezone.utc).isoformat(), 'condition': IDENTITY[0], 'seed': IDENTITY[1], 'fold': IDENTITY[2],
        'previous_recovery_registration': previous, 'base_queue_registration': base,
        'checkpoint_cpu_resume_proof': cpu_binding, 'snapshot_manifest': cpu['snapshot_manifest'],
        'process_absence_evidence': cpu['process_absence_evidence'], 'historical_native_identity_artifact': native_binding,
        'expected_native_identities': identities, 'durable': durable, 'observed_progress': observed,
        'known_extra_lower_bounds': extra, 'actual_exit_code': None, 'actual_exit_time': None, 'cause': None,
        'unflushed_tail_counters': None, 'final_elapsed_seconds': None, 'queue_attempt_final_elapsed_seconds': None,
        'graceful_terminal_flush': False, 'known_extra_not_added_again_to_original_returned_counters': True,
        'prior_user_pause_required': False, 'natural_exit_claimed': False, 'training_complete_promoted': False,
        'full_plan_completed_claimed': False, 'assembler_operations': {'tensor_loads': 0, 'binary_reads': 0,
            'model_constructions': 0, 'GPU_operations': 0, 'process_queries': 0, 'process_controls': 0,
            'project_verifier_calls': 0, 'test_probability_reads': 0}}
    for name, original, copied in (('attempt','original_attempt','attempt_snapshot'), ('owner','old_owner_original','old_owner_snapshot'),
        ('queue_attempt','old_queue_attempt_original','old_queue_attempt_snapshot'), ('latest.pt','resume_checkpoint','preserved_checkpoint'),
        ('train_history.json','train_history','history_snapshot')):
        value[original], value[copied] = retained[name]['original'], retained[name]['snapshot']
    value['progress_snapshot'] = retained['progress.json']['snapshot']
    require(desc(SOURCE) == own_source and desc(CPU) == cpu_binding, 'Assembler/CPU proof changed while composing')
    return value


if __name__ == '__main__':
    value = assemble()
    path = OUT / 'process_loss_recovery_v2' / ('loss_evidence_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'artifact': desc(path), 'durable': value['durable'], 'known_extra_lower_bounds': value['known_extra_lower_bounds'],
                      'full_plan_completed_claimed': False, 'actual_new_tensor_loads': 0}))
