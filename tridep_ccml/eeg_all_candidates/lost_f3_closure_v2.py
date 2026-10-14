"""Verify two immutable F3 losses without rewriting their running records.

The original CSBrain event delegates to sealed L1. The Oct11 LUNA event uses
snapshot/absence/CPU-resume evidence and does not require a prior user pause.
No tensor is deserialized here. X/L1 provenance verification may hash official
weight bytes; the restricted descriptor reader itself reads JSON/source only.
"""
from pathlib import Path
import hashlib
import json

from . import storage as S, registry as R, execution_acceleration_v1 as X
from . import lost_f3_closure_v1 as L1

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'process_loss_closures'
ENTRY_SOURCE = SOURCE.with_name('recover_q5_after_process_loss_v2.py')
L1_SHA = '97dd30a04a0f028890c124001a4f06e4f2b22cf86677a3d6f0731fd53b790a57'
Q5_SHA = L1.Q5_SHA
PREVIOUS_SHA = 'ec1500ce2671600fca1588e41f14fc559e12300824aaa5c4a2625671f8d2ec9f'
IDENTITY = ('LUNA-lastblock-finetune', 33, 3)
LOSS_BINDING = {
    'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/process_loss_recovery_v2/loss_evidence_1791705844098982900.json',
    'sha256': '4421347b22bbae039c60ac219a98bee7722abb7826d9b977c99b06e3d5f0be59'}
UNKNOWN_FIELDS = ('actual_exit_code', 'actual_exit_time', 'cause', 'unflushed_tail_counters',
                  'final_elapsed_seconds', 'queue_attempt_final_elapsed_seconds')
CONTEXT_FIELDS = ('condition', 'seed', 'fold', 'original_attempt', 'attempt_snapshot',
                  'old_owner_original', 'old_owner_snapshot', 'old_queue_attempt_original',
                  'old_queue_attempt_snapshot', 'resume_checkpoint', 'preserved_checkpoint',
                  'train_history', 'history_snapshot', 'progress_snapshot', 'durable',
                  'observed_progress', 'known_extra_lower_bounds', 'checkpoint_cpu_resume_proof',
                  'process_absence_evidence', 'expected_native_identities')


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def binding_shape(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}
            and type(value['path']) is str and type(value['sha256']) is str
            and len(value['sha256']) == 64 and all(c in '0123456789abcdef' for c in value['sha256']),
            'Expected exact source-bound descriptor')
    path = (S.WORK / value['path']).resolve()
    require(path.is_relative_to(S.WORK.resolve()) and not Path(value['path']).is_absolute()
            and path.relative_to(S.WORK).as_posix() == value['path'], 'Descriptor path is not canonical workspace-relative')
    return path


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(S.WORK.resolve()) and path.suffix in ('.json', '.py'),
            'Only workspace JSON/source bytes may be read by this reader')
    return {'path': path.relative_to(S.WORK).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(binding):
    path = binding_shape(binding)
    require(path.suffix == '.json' and descriptor(path) == binding, 'JSON binding changed')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'JSON changed while reading')
    return json.loads(raw.decode('utf-8-sig'))


def verify_delegated_source():
    require(descriptor(Path(L1.__file__))['sha256'] == L1_SHA, 'Sealed L1 verifier changed')


def previous_registration():
    verify_delegated_source()
    binding = descriptor(S.OUT / 'continuation_queue_v5/recovery_registration.json')
    require(binding['sha256'] == PREVIOUS_SHA, 'Original recovery1 registration changed')
    return L1.verify_recovery_registration(binding)


def check_identity(value):
    require(tuple(value.get(k) for k in ('condition', 'seed', 'fold')) == IDENTITY
            and type(value.get('seed')) is int and type(value.get('fold')) is int,
            'This new closure covers only the observed LUNA33/fold3 loss')


def equal_snapshot(original, snapshot, *, reopen_original=False):
    binding_shape(original); binding_shape(snapshot)
    require(original['sha256'] == snapshot['sha256'], 'Snapshot is not original exact bytes')
    value = read_json(snapshot)
    if reopen_original:
        require(read_json(original) == value, 'Original immutable attempt changed')
    return value


def verify_absence(binding, identities):
    value = read_json(binding)
    require(value.get('schema') == 'q5_process_absence_observation_v2'
            and value.get('query_succeeded') is True and value.get('all_expected_processes_absent') is True,
            'Successful actual process-absence observation required')
    require(type(identities) is list and len(identities) == 4
            and all(type(v) is dict and type(v.get('pid')) is int and v['pid'] > 0
                    and type(v.get('parent_pid')) is int and type(v.get('creation_100ns')) is str
                    and v['creation_100ns'].isdigit() and v.get('image') and v.get('command_line') for v in identities),
            'Four exact historical process identities required')
    pids = [v['pid'] for v in identities]
    saved_identities = value.get('expected_identities', value.get('expected_native_identities'))
    require(('expected_identities' not in value or value['expected_identities'] == identities)
            and ('expected_native_identities' not in value or value['expected_native_identities'] == identities),
            'Conflicting absence identity aliases')
    require(len(set(pids)) == 4 and saved_identities == identities
            and value.get('queried_pids') == pids and value.get('absent_pids') == pids
            and value.get('workspace_training_workers') == [] and value.get('observed_rows') == [],
            'Absent process identities or workspace worker inventory differs')
    require(all(k in value and value[k] is None for k in ('exit_codes', 'actual_exit_time', 'reason')),
            'Later absence does not establish exit code, time or reason')
    require(type(value.get('observed_at')) is str and bool(value['observed_at']), 'Absence observation clock missing')
    return value


def verify_recovery_inputs(loss_binding):
    """Validate existing evidence only; no registration/closure/process writes."""
    require(loss_binding == LOSS_BINDING, 'Only the reviewed actual Oct11 loss evidence is registered')
    loss = read_json(loss_binding)
    require(loss.get('schema') == 'q5_f3_process_loss_evidence_v2' and loss.get('passed') is True,
            'Expected immutable actual loss evidence')
    check_identity(loss)
    prior = previous_registration()
    require(loss.get('previous_recovery_registration') == prior['registration']
            and loss.get('base_queue_registration') == prior['base_queue_registration'], 'Loss execution lineage differs')
    for key in CONTEXT_FIELDS:
        require(key in loss, 'Loss context missing: ' + key)
    require(all(k in loss and loss[k] is None for k in UNKNOWN_FIELDS)
            and loss.get('graceful_terminal_flush') is False,
            'Loss evidence must preserve unknown process endpoint and buffered tail')
    absence = verify_absence(loss['process_absence_evidence'], loss['expected_native_identities'])
    record = equal_snapshot(loss['original_attempt'], loss['attempt_snapshot'], reopen_original=True)
    check_identity(record)
    require(record.get('kind') == 'formal' and record.get('status') == 'running'
            and type(record.get('implementation', {}).get('version')) is int
            and record['implementation']['version'] == 3 and record.get('graceful_terminal_flush') is False
            and record.get('seconds') is None and record.get('finished_at') is None,
            'Lost F3 source record must remain running with unknown endpoint')
    observation = X.verify_observer_record(record, loss['original_attempt'], dict(zip(('condition', 'seed', 'fold'), IDENTITY)))
    require(observation['unflushed_tail_unknown'] is True and observation['unflushed_tail_counters'] is None,
            'Lost buffered tail cannot be zero-filled')
    owner = equal_snapshot(loss['old_owner_original'], loss['old_owner_snapshot'])
    queue = equal_snapshot(loss['old_queue_attempt_original'], loss['old_queue_attempt_snapshot'], reopen_original=True)
    require(owner.get('recovery_registration') == queue.get('recovery_registration') == prior['registration']
            and owner.get('source') == queue.get('source') == descriptor(SOURCE.with_name('continuation_queue_v5.py'))
            and owner.get('execution_registration') == queue.get('execution_registration') == prior['base_queue_registration']
            and type(owner.get('execution_version')) is int and owner['execution_version'] == 5
            and type(queue.get('execution_version')) is int and queue['execution_version'] == 5
            and owner.get('recovery_execution_source') == queue.get('recovery_execution_source') == prior['source'],
            'Original queue/owner actual source lineage differs')
    require(owner.get('child_job') == IDENTITY[0] + '-fit' and queue.get('job', {}).get('id') == owner['child_job']
            and queue.get('status') == 'running' and queue.get('returncode') is None
            and queue.get('seconds') is None and queue.get('finished_at') is None
            and type(owner.get('pid')) is int and type(owner.get('child_pid')) is int
            and queue.get('child_pid') == owner['child_pid'], 'Original lost parent job/endpoint differs')
    ids = {v['pid']: v for v in loss['expected_native_identities']}
    require(owner['pid'] in ids and owner['child_pid'] in ids
            and ids[owner['child_pid']]['parent_pid'] == owner['pid']
            and ids[owner['pid']]['parent_pid'] in ids
            and sum(v['parent_pid'] == owner['child_pid'] for v in ids.values()) == 1,
            'Historical four-object ancestry differs from archived owner')
    history = equal_snapshot(loss['train_history'], loss['history_snapshot'])
    progress = read_json(loss['progress_snapshot'])
    cpu = read_json(loss['checkpoint_cpu_resume_proof'])
    require(cpu.get('passed') is True and tuple(cpu.get(k) for k in ('condition', 'seed', 'fold')) == IDENTITY
            and type(cpu.get('checkpoint_loads')) is int and cpu['checkpoint_loads'] == 2
            and all(type(cpu.get(k)) is int and cpu[k] == 0 for k in
                    ('model_constructions', 'GPU_operations', 'raw_arrays_loaded', 'test_probability_reads')),
            'Actual two-load CPU-only full-state resume proof required')
    require(cpu.get('buffered_trace_pair') == {'original': loss['original_attempt'], 'snapshot': loss['attempt_snapshot']}
            and cpu.get('process_absence_evidence') == loss['process_absence_evidence'], 'CPU proof loss/trace binding differs')
    snapshot = read_json(cpu['snapshot_manifest'])
    require(snapshot.get('schema') == 'q5_loss_byte_snapshot_v2' and snapshot.get('identity') == list(IDENTITY)
            and snapshot.get('absence') == loss['process_absence_evidence'] and snapshot.get('source') == cpu.get('source'),
            'Actual CPU snapshot manifest identity/source differs')
    retained = snapshot['retained']
    for name, original, copied in (('attempt', 'original_attempt', 'attempt_snapshot'),
                                   ('owner', 'old_owner_original', 'old_owner_snapshot'),
                                   ('queue_attempt', 'old_queue_attempt_original', 'old_queue_attempt_snapshot'),
                                   ('train_history.json', 'train_history', 'history_snapshot'),
                                   ('latest.pt', 'resume_checkpoint', 'preserved_checkpoint')):
        require(retained.get(name) == {'original': loss[original], 'snapshot': loss[copied]},
                'Loss fields differ from CPU snapshot manifest: ' + name)
    require(retained['progress.json']['snapshot'] == loss['progress_snapshot'], 'Progress snapshot differs')
    validation = cpu['checkpoint_validation']
    require(validation.get('resume_full_state_verified') is True and validation.get('checkpoint_load_device') == 'cpu'
            and validation.get('optimizer_group_order_and_moment_shapes_verified') is True
            and validation.get('best_selected_parameter_finiteness_verified') is True
            and validation.get('cpu_rng_local_restore_and_cuda_saved_byte_shapes_verified') is True
            and validation.get('raw_role_descriptors_verified_without_loading_raw') is True,
            'Full optimizer/RNG/source/selected-state CPU proof required')
    for key in ('resume_checkpoint', 'preserved_checkpoint'):
        binding_shape(loss[key])
    require(loss['resume_checkpoint'] in validation['original_checkpoint_bindings']
            and loss['resume_checkpoint']['sha256'] == loss['preserved_checkpoint']['sha256']
            and cpu.get('preserved_checkpoint') == loss['preserved_checkpoint']
            and cpu.get('resume_checkpoint') == loss['resume_checkpoint']
            and cpu.get('durable_history_matches_current_and_snapshot') is True,
            'Checkpoint/history snapshot is not the CPU-verified resume state')
    durable, observed, extra = loss['durable'], loss['observed_progress'], loss['known_extra_lower_bounds']
    require(set(durable) == set(observed) == {'epoch', 'optimizer_updates', 'backward_calls'}
            and set(extra) == {'optimizer_updates', 'backward_calls'}
            and all(integer(v) for d in (durable, observed, extra) for v in d.values()), 'Boundary counters must be strict integers')
    require(all(validation.get(k) == durable[k] for k in durable), 'Durable counters differ from CPU proof')
    require(observed == {'epoch': progress.get('epoch'), 'optimizer_updates': progress.get('optimizer_updates_completed'),
                         'backward_calls': progress.get('backward_calls_completed')}
            and progress.get('last_durable_epoch') == durable['epoch'], 'Persisted progress boundary differs')
    require(record['optimizer_step_returned'] == observed['optimizer_updates']
            and record['backward_returned'] == observed['backward_calls']
            and observed['epoch'] >= durable['epoch']
            and all(observed[k] >= durable[k] and extra[k] == observed[k] - durable[k] for k in extra),
            'Known extra lower bound differs from persisted returns minus durable checkpoint')
    commit = record.get('latest_durable_checkpoint', {})
    require(all(commit.get(k) == durable[k] for k in durable) and commit.get('path') == loss['resume_checkpoint']['path']
            and commit.get('checkpoint_save_returned') is True, 'F3 committed checkpoint boundary differs')
    rows = history['history'] if type(history) is dict and 'history' in history else history
    require(type(rows) is list and len(rows) == durable['epoch'] and rows
            and all(rows[-1].get(k) == durable[k] for k in durable), 'Retained training history differs from durable boundary')
    # Actual CPU producer and snapshot collector must be source bound; immutable
    # registered execution verifies the reviewed source descriptors again.
    for container in (loss, cpu):
        require(container.get('source') == descriptor(binding_shape(container['source'])), 'Evidence producer source changed')
    return {'passed': True, 'loss_evidence': loss_binding, **{k: loss[k] for k in CONTEXT_FIELDS},
            'previous_recovery_registration': prior['registration'], 'base_queue_registration': prior['base_queue_registration'],
            'observer_validation': observation, 'actual_absence_observation': absence,
            'actual_exit_code': None, 'actual_exit_time': None, 'cause': None,
            'unflushed_tail_counters': None, 'final_elapsed_seconds': None,
            'queue_attempt_final_elapsed_seconds': None, 'graceful_terminal_flush': False}


def verify_recovery_registration(binding=None):
    binding = binding or descriptor(S.OUT / 'continuation_queue_v5/recovery_registration_v2.json')
    manifest = read_json(binding)
    if manifest.get('schema') == 'q5_checkpoint_recovery_execution_v1':
        verify_delegated_source()
        return L1.verify_recovery_registration(binding)
    previous = previous_registration()
    require(manifest.get('schema') == 'q5_checkpoint_recovery_execution_v2'
            and type(manifest.get('version')) is int and manifest['version'] == 2
            and manifest.get('source') == descriptor(ENTRY_SOURCE)
            and manifest.get('base_queue_registration') == previous['base_queue_registration']
            and manifest.get('previous_recovery_registration') == previous['registration']
            and manifest.get('registry') == R.registration_descriptor(), 'Recovery2 manifest lineage differs')
    context = verify_recovery_inputs(manifest['loss_evidence'])
    require(manifest.get('process_absence_evidence') == context['process_absence_evidence']
            and manifest.get('checkpoint_cpu_resume_proof') == context['checkpoint_cpu_resume_proof'],
            'Recovery2 manifest evidence differs')
    required = [ENTRY_SOURCE, SOURCE, S.REPORT / 'audit_completed_finetune_checkpoints_v6.py', S.REPORT / 'cost_aggregation_v7.py']
    sources = manifest.get('sources', [])
    require(len(previous['value']['sources']) == 80 and len(sources) == 84
            and len({x['path'] for x in sources}) == 84
            and all(x in sources for x in previous['value']['sources'])
            and all(descriptor(p) in sources for p in required), 'Recovery2 must preserve all80 and bind exactly four new execution sources')
    for source in sources:
        require(descriptor(binding_shape(source)) == source, 'Registered recovery2 source changed')
    require(manifest.get('cpu_auditor') == descriptor(required[2]) and manifest.get('cost_collector') == descriptor(required[3])
            and manifest.get('same_formal_F3_and_E3_core') is True
            and manifest.get('training_protocol_or_hyperparameters_changed') is False,
            'Recovery2 actual CPU/cost source or unchanged core binding differs')
    return {'passed': True, 'registration': binding, 'value': manifest, 'context': context,
            'base_queue_registration': previous['base_queue_registration'], 'source': manifest['source']}


def validate_closure(binding, proof, artifact, record, fit):
    if proof.get('schema') == 'lost_f3_process_closure_v1':
        verify_delegated_source()
        return L1.validate_closure(binding, proof, artifact, record, fit)
    require(proof.get('schema') == 'lost_f3_process_closure_v2' and proof.get('status') == 'process_absence_observed'
            and proof.get('passed') is True, 'Expected immutable loss2 closure')
    check_identity(proof); check_identity(fit)
    lineage = verify_recovery_registration(proof['recovery_registration'])
    context = lineage['context']
    archives = ('old_owner_snapshot', 'old_queue_attempt_snapshot')
    require(proof.get('closure_source') == lineage['source'] and proof.get('loss_evidence') == context['loss_evidence']
            and all(proof.get(k) == context[k] for k in CONTEXT_FIELDS if k not in archives),
            'Closure2 differs from verified immutable evidence')
    for key in archives:
        require(proof[key]['sha256'] == context[key]['sha256']
                and read_json(proof[key]) == read_json(context[key]), 'Entry archival bytes differ from loss snapshot')
    verify_absence(proof['runtime_process_absence_evidence'], context['expected_native_identities'])
    require(proof['original_attempt'] == artifact and read_json(artifact) == record,
            'Closure2 does not cover the exact supplied original trace')
    require(all(k in proof and proof[k] is None for k in UNKNOWN_FIELDS)
            and proof.get('graceful_terminal_flush') is False
            and proof.get('known_extra_not_added_again_to_original_returned_counters') is True,
            'Closure2 cannot invent endpoint/flush/tail or add repeated work twice')
    return {'passed': True, 'closure': binding, 'recovery_registration': lineage['registration'],
            'closure_source': lineage['source'], 'original_status': 'running', 'process_absence_observed': True,
            'actual_exit_code': None, 'actual_exit_time': None, 'cause': None,
            'persisted_counters_are_lower_bounds': True, 'unflushed_tail_unknown': True,
            'unflushed_tail_counters': None, 'graceful_terminal_flush': False,
            'known_extra_optimizer_returns_outside_checkpoint': context['known_extra_lower_bounds']['optimizer_updates'],
            'known_extra_backward_returns_outside_checkpoint': context['known_extra_lower_bounds']['backward_calls'],
            'durable_epoch': context['durable']['epoch'], 'durable_optimizer_updates': context['durable']['optimizer_updates'],
            'durable_backward_calls': context['durable']['backward_calls'],
            'final_elapsed_seconds': None, 'extra_work_not_added_again': True,
            'original_queue_attempt': context['old_queue_attempt_original'], 'queue_attempt_final_elapsed_seconds': None,
            'checkpoint_cpu_resume_proof': context['checkpoint_cpu_resume_proof'],
            'resume_checkpoint_descriptor_not_reopened': context['resume_checkpoint']}


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
