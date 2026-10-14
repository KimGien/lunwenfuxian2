"""JSON/source-only proof of the single Oct10 lost buffered F3 process.

The original running trace stays unchanged. Absence observed later does not
establish an exit code/time, graceful flush, or a zero unobserved tail.
"""
from pathlib import Path
import hashlib
import json

from . import storage as S, registry as R, execution_acceleration_v1 as X

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'process_loss_closures'
ENTRY_SOURCE = SOURCE.with_name('recover_q5_from_pause_v1.py')
Q5_SHA = 'a9db8e215f1d32cd215d550319c0fa1646985c54d2167783a7547b3a6ea5312f'
PAUSE_SHA = '247d0b0dc3329a248ef7ca384b4acb2b2956e403d6b512c6b493306ed697d910'
ATTEMPT_SHA = 'e190ee531b30275322398edeeab6a93c4db4a28dc239510e28a59ea4939fd52f'
CHECKPOINT_SHA = '8844db8f6443ecb7777730fd06aa33655fa6be8b1aa0fee46b4b447983b43782'
CPU_PROOF_SHA = '6e8985f94502b9d36de56fd4a9c3b6d901346e3b5c28f9b62db5d45663632308'
IDENTITY = ('CSBrain-lastblock-finetune', 33, 4)
PIDS = [40840, 18224, 32880, 46716]


def require(value, message):
    if not value:
        raise ValueError(message)


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(S.WORK.resolve()) and path.suffix in ('.json', '.py'), 'Only workspace JSON/source bytes may be read')
    return {'path': path.relative_to(S.WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(binding):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'Expected exact JSON descriptor')
    path = (S.WORK / binding['path']).resolve()
    require(path.suffix == '.json' and descriptor(path) == binding, 'JSON proof changed or is not JSON')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_recovery_registration(binding=None):
    binding = binding or descriptor(S.OUT / 'continuation_queue_v5/recovery_registration.json')
    manifest = read_json(binding)
    base_binding = descriptor(S.OUT / 'continuation_queue_v5/registration.json')
    require(base_binding['sha256'] == Q5_SHA, 'Original Q5 registration changed')
    base = read_json(base_binding)
    require(manifest.get('schema') == 'q5_checkpoint_recovery_execution_v1'
            and type(manifest.get('version')) is int and manifest['version'] == 1
            and manifest.get('source') == descriptor(ENTRY_SOURCE)
            and manifest.get('base_queue_registration') == base_binding
            and manifest.get('registry') == R.registration_descriptor(), 'Recovery manifest lineage differs')
    require(manifest.get('pause_proof', {}).get('sha256') == PAUSE_SHA, 'Expected exact user pause proof')
    read_json(manifest['pause_proof']); read_json(manifest['process_absence_evidence'])
    sources = manifest.get('sources', [])
    require(len(base['sources']) == 76 and len({x['path'] for x in sources}) == len(sources)
            and all(x in sources for x in base['sources']), 'All original76 sources must be preserved')
    required = [ENTRY_SOURCE, SOURCE, S.REPORT / 'audit_completed_finetune_checkpoints_v5.py', S.REPORT / 'cost_aggregation_v6.py']
    require(all(descriptor(p) in sources for p in required), 'Recovery execution/audit/cost source is not bound')
    for source in sources:
        require(descriptor(S.WORK / source['path']) == source, 'Registered recovery source changed')
    return {'passed': True, 'registration': binding, 'value': manifest,
            'base_queue_registration': base_binding, 'source': manifest['source']}


def validate_closure(binding, proof, artifact, record, fit):
    require(proof.get('schema') == 'lost_f3_process_closure_v1'
            and proof.get('status') == 'process_absence_observed' and proof.get('passed') is True,
            'Expected immutable process-absence closure')
    require(tuple(proof.get(k) for k in ('condition', 'seed', 'fold')) == IDENTITY
            and tuple(fit.get(k) for k in ('condition', 'seed', 'fold')) == IDENTITY
            and all(type(proof.get(k)) is int for k in ('seed', 'fold')), 'Closure identity differs')
    lineage = verify_recovery_registration(proof['recovery_registration'])
    manifest = lineage['value']
    require(proof.get('closure_source') == lineage['source'], 'Actual closure producer is not reviewed recovery entry')
    require(proof.get('original_attempt') == artifact and artifact['sha256'] == ATTEMPT_SHA
            and proof['attempt_snapshot']['sha256'] == ATTEMPT_SHA
            and read_json(artifact) == read_json(proof['attempt_snapshot']) == record,
            'Original running F3 and snapshot bytes must remain identical')
    require(record.get('status') == 'running' and record.get('implementation', {}).get('version') == 3
            and record.get('graceful_terminal_flush') is False
            and record.get('seconds') is None and record.get('finished_at') is None,
            'Lost process cannot be rewritten as a graceful or timed terminal attempt')
    observation = X.verify_observer_record(record, artifact, fit)
    require(observation['unflushed_tail_unknown'] is True and observation['unflushed_tail_counters'] is None,
            'Lost buffered tail must remain unknown')
    require(proof.get('pause_proof') == manifest['pause_proof'], 'Pause proof differs from recovery registration')
    pause = read_json(proof['pause_proof'])
    pair = {'original': artifact, 'snapshot': proof['attempt_snapshot'], 'unflushed_tail_unknown_if_runtime_lost': True}
    require(pair in pause.get('attempt_snapshots', []) and pause.get('status') == 'paused_by_user', 'Trace is not the exact paused runtime')
    require(proof.get('process_absence_evidence') == manifest['process_absence_evidence'], 'Absence proof differs')
    absent = read_json(proof['process_absence_evidence'])
    require(absent.get('schema') == 'q4_process_absence_observation_v1' and absent.get('query_succeeded') is True
            and absent.get('all_expected_processes_absent') is True and absent.get('queried_pids') == PIDS
            and absent.get('absent_pids') == PIDS and all(type(n) is int for n in absent['absent_pids'])
            and absent.get('expected_identities') == pause['identities'], 'Actual absent-process observation differs')
    require(all(key in absent and absent[key] is None for key in ('exit_codes', 'actual_exit_time', 'reason')),
            'Later absence cannot establish actual exit codes, time or reason')
    cpu = read_json(proof['checkpoint_cpu_resume_proof'])
    require(proof['checkpoint_cpu_resume_proof']['sha256'] == CPU_PROOF_SHA
            and cpu.get('passed') is True and cpu.get('pause_evidence') == proof['pause_proof']
            and cpu.get('buffered_trace_pair') == pair, 'Actual CPU resume proof differs')
    validation = cpu['checkpoint_validation']
    require(validation.get('resume_full_state_verified') is True and validation.get('checkpoint_load_device') == 'cpu'
            and cpu.get('durable_history_matches_current_and_pause_snapshot') is True
            and type(cpu.get('checkpoint_loads')) is int and cpu['checkpoint_loads'] == 2
            and all(type(cpu.get(k)) is int and cpu[k] == 0 for k in ('model_constructions', 'GPU_operations', 'raw_arrays_loaded', 'test_probability_reads')),
            'Actual CPU-only full-state resume validation required')
    require(proof['resume_checkpoint'] in validation['original_checkpoint_bindings']
            and proof['resume_checkpoint']['sha256'] == CHECKPOINT_SHA
            and proof.get('preserved_checkpoint') == cpu['latest_pause_snapshot']
            and proof['preserved_checkpoint']['sha256'] == CHECKPOINT_SHA,
            'Resume checkpoint descriptor differs; tensors are never reopened by this verifier')
    owner = read_json(proof['old_owner_snapshot']); attempt = read_json(proof['old_queue_attempt_snapshot'])
    require(owner == pause['owner_snapshot'] and owner['pid'] == 40840 and owner['child_pid'] == 18224
            and attempt.get('status') == 'running' and attempt.get('child_pid') == 18224
            and attempt.get('job', {}).get('id') == 'CSBrain-lastblock-finetune-fit'
            and attempt.get('registration') == descriptor(S.OUT / 'continuation_queue_v4/registration.json'),
            'Archived old queue owner/attempt differs')
    durable = {'epoch': 8, 'optimizer_updates': 32, 'backward_calls': 216}
    progress = {'epoch': 9, 'optimizer_updates': 36, 'backward_calls': 243}
    require(proof.get('durable') == durable and proof.get('observed_progress') == progress
            and all(type(n) is int for row in (proof['durable'], proof['observed_progress']) for n in row.values()), 'Recovery boundaries differ')
    require(all(validation[k] == durable[k] for k in durable)
            and pause['active_progress'] == {'epoch': 9, 'optimizer_updates_completed': 36, 'backward_calls_completed': 243, 'last_durable_epoch': 8}
            and record['optimizer_step_returned'] == 36 and record['backward_returned'] == 243,
            'Persisted observer/progress/CPU trajectory boundary differs')
    require(proof.get('known_extra_lower_bounds') == {'optimizer_updates': 4, 'backward_calls': 27}
            and all(type(n) is int for n in proof['known_extra_lower_bounds'].values()), 'Known extra work is not the verified lower-bound delta')
    require(proof.get('graceful_terminal_flush') is False
            and 'unflushed_tail_counters' in proof and proof['unflushed_tail_counters'] is None
            and 'final_elapsed_seconds' in proof and proof['final_elapsed_seconds'] is None,
            'No invented terminal flush, zero tail or elapsed endpoint is permitted')
    return {'passed': True, 'closure': binding, 'recovery_registration': lineage['registration'],
            'closure_source': lineage['source'], 'original_status': 'running', 'process_absence_observed': True,
            'actual_exit_code': None, 'actual_exit_time': None, 'persisted_counters_are_lower_bounds': True,
            'unflushed_tail_unknown': True, 'unflushed_tail_counters': None, 'graceful_terminal_flush': False,
            'known_extra_optimizer_returns_outside_checkpoint': 4, 'known_extra_backward_returns_outside_checkpoint': 27,
            'durable_epoch': 8, 'durable_optimizer_updates': 32, 'durable_backward_calls': 216,
            'final_elapsed_seconds': None, 'extra_work_not_added_again': True,
            'checkpoint_cpu_resume_proof': proof['checkpoint_cpu_resume_proof'],
            'resume_checkpoint_descriptor_not_reopened': proof['resume_checkpoint']}


def verify_lost_f3_attempt(artifact, record, fit):
    if record.get('implementation', {}).get('version') != 3 or record.get('status') != 'running':
        return None
    matches = []
    for path in sorted(ROOT.glob('lost_f3_*.json')):
        binding = descriptor(path); value = read_json(binding)
        if value.get('original_attempt', {}).get('path') == artifact['path']:
            matches.append((binding, value))
    if not matches:
        return None
    require(len(matches) == 1, 'Ambiguous process-loss closures for the same attempt')
    return validate_closure(*matches[0], artifact, record, fit)
