"""Verify the third F3 loss while delegating both historical closures unchanged.

The Oct12 LaBraM4s loss must match the exact Oct11 native pause proof and the
new absence/CPU full-state evidence. No native control, model construction,
checkpoint tensor loading or registry mutation occurs in this verifier.
Delegated provenance checks may hash official weight bytes.
"""
from pathlib import Path
import hashlib
import json

from . import storage as S, registry as R, execution_acceleration_v1 as X
from . import lost_f3_closure_v2 as L2

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'process_loss_closures'
ENTRY_SOURCE = SOURCE.with_name('recover_q5_after_process_loss_v3.py')
L2_SHA = '89d5a04868c07d377f50712f23dcf271425c060831254862eb2ee27edb5761e6'
Q5_SHA = L2.Q5_SHA
PREVIOUS_SHA = '525f6ed6e838fe267956c54d90d4e6d13d7da3bc710863aca90c35af466782cd'
IDENTITY = ('LaBraM-native4s-lastblock-finetune', 33, 1)
LOSS_BINDING = {'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/process_loss_recovery_v3/loss_evidence_1791767102232649100.json',
                'sha256': '82b713c44721b0966ae0d6846925d7a5ac60c6b2009115b071ec0c6eba316693'}
PAUSE_NATIVE_SHA = '65977f9551a7621f9df0e1701173bf1845794a6c60b7a0d58012237092a98c40'
PAUSE_CAPTURE_SHA = 'c1b3424abe3345190a8b8cee9d89f024a62cd1ef1bc852ebc9d084b9662681ef'
PAUSE_PROGRESS_SHA = '7feba94e10375d50a7fcd565d8fdf4f5ea9f988d2ac0d1d4af76c5b9288e0548'

UNKNOWN_FIELDS = ('actual_exit_code', 'actual_exit_time', 'cause', 'unflushed_tail_counters',
                  'final_elapsed_seconds', 'queue_attempt_final_elapsed_seconds')
CONTEXT_FIELDS = ('condition', 'seed', 'fold', 'original_attempt', 'attempt_snapshot',
                  'old_owner_original', 'old_owner_snapshot', 'old_queue_attempt_original',
                  'old_queue_attempt_snapshot', 'resume_checkpoint', 'preserved_checkpoint',
                  'train_history', 'history_snapshot', 'progress_snapshot', 'durable',
                  'observed_progress', 'known_extra_lower_bounds', 'checkpoint_cpu_resume_proof',
                  'process_absence_evidence', 'expected_native_identities', 'pause_native_proof',
                  'pause_capture', 'pause_progress_and_observer')


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
    require(descriptor(Path(L2.__file__))['sha256'] == L2_SHA, 'Sealed L2 verifier changed')
    L2.verify_delegated_source()


def previous_registration():
    verify_delegated_source()
    binding = descriptor(S.OUT / 'continuation_queue_v5/recovery_registration_v2.json')
    require(binding['sha256'] == PREVIOUS_SHA, 'Original recovery2 registration changed')
    return L2.verify_recovery_registration(binding)


def check_identity(value):
    require(tuple(value.get(k) for k in ('condition', 'seed', 'fold')) == IDENTITY
            and type(value.get('seed')) is int and type(value.get('fold')) is int,
            'This new closure covers only the observed LaBraM4s33/fold1 loss')


def equal_snapshot(original, snapshot, *, reopen_original=False):
    binding_shape(original); binding_shape(snapshot)
    require(original['sha256'] == snapshot['sha256'], 'Snapshot is not original exact bytes')
    value = read_json(snapshot)
    if reopen_original:
        require(read_json(original) == value, 'Original immutable attempt changed')
    return value


def verify_absence(binding, identities):
    value = read_json(binding)
    require(value.get('schema') == 'q5_process_absence_observation_v3'
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
    require(LOSS_BINDING is not None and loss_binding == LOSS_BINDING, 'Only the reviewed actual Oct12 loss evidence is registered')
    loss = read_json(loss_binding)
    require(loss.get('schema') == 'q5_f3_process_loss_evidence_v3' and loss.get('passed') is True,
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
    pause = read_json(loss['pause_native_proof'])
    capture = read_json(loss['pause_capture'])
    paused_progress = read_json(loss['pause_progress_and_observer'])
    require(loss['pause_native_proof']['sha256'] == PAUSE_NATIVE_SHA
            and loss['pause_capture']['sha256'] == PAUSE_CAPTURE_SHA
            and loss['pause_progress_and_observer']['sha256'] == PAUSE_PROGRESS_SHA,
            'Exact Oct11 native pause/capture/progress bytes required')
    require(pause.get('status') == 'paused_by_explicit_user'
            and pause.get('automatic_resume_authorized') is False
            and pause.get('runtime_memory_preserved') is True
            and pause.get('owner_snapshot') == capture.get('owner_snapshot') == owner
            and pause.get('identities') == capture.get('identities') == loss['expected_native_identities'],
            'Lost runtime differs from the exact paused four objects')
    actions, verification = pause.get('actions', []), pause.get('verification', [])
    require(len(actions) == len(verification) == 4
            and {a.get('pid') for a in actions} == {v['pid'] for v in loss['expected_native_identities']}
            and all(type(a.get('native_status')) is int and a['native_status'] == 0 for a in actions),
            'Exactly four successful native pause returns required')
    for ident in loss['expected_native_identities']:
        rows = [v for v in verification if v.get('pid') == ident['pid'] and v.get('role') == ident.get('role')]
        require(len(rows) == 1 and rows[0].get('creation_100ns') == ident['creation_100ns']
                and rows[0].get('threads')
                and all(t.get('state') == 'Wait' and t.get('wait_reason') == 'Suspended' for t in rows[0]['threads']),
                'Original pause suspension verification differs')
    paused_attempts = paused_progress.get('running_observer_records', [])
    exact_paused = [v for v in paused_attempts if v.get('path') == loss['original_attempt']['path']]
    require(len(exact_paused) == 1 and exact_paused[0].get('sha256') == loss['original_attempt']['sha256']
            and exact_paused[0].get('value') == record, 'Original running record differs from the paused trace')
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
    require(cpu.get('pause_native_proof') == loss['pause_native_proof'], 'CPU resume proof must bind the actual native pause')
    snapshot = read_json(cpu['snapshot_manifest'])
    require(snapshot.get('schema') == 'q5_loss_byte_snapshot_v3' and snapshot.get('identity') == list(IDENTITY)
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
    require(paused_progress.get('progress_JSON_reported') == {k: progress.get(k) for k in
            ('epoch', 'optimizer_updates_completed', 'backward_calls_completed', 'last_durable_epoch')},
            'Loss progress differs from the native-pause snapshot')
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
    binding = binding or descriptor(S.OUT / 'continuation_queue_v5/recovery_registration_v3.json')
    manifest = read_json(binding)
    if manifest.get('schema') in ('q5_checkpoint_recovery_execution_v1', 'q5_checkpoint_recovery_execution_v2'):
        verify_delegated_source()
        return L2.verify_recovery_registration(binding)
    previous = previous_registration()
    require(manifest.get('schema') == 'q5_checkpoint_recovery_execution_v3'
            and type(manifest.get('version')) is int and manifest['version'] == 3
            and manifest.get('source') == descriptor(ENTRY_SOURCE)
            and manifest.get('base_queue_registration') == previous['base_queue_registration']
            and manifest.get('previous_recovery_registration') == previous['registration']
            and manifest.get('registry') == R.registration_descriptor(), 'Recovery3 manifest lineage differs')
    context = verify_recovery_inputs(manifest['loss_evidence'])
    require(manifest.get('process_absence_evidence') == context['process_absence_evidence']
            and manifest.get('checkpoint_cpu_resume_proof') == context['checkpoint_cpu_resume_proof'],
            'Recovery3 manifest evidence differs')
    required = [ENTRY_SOURCE, SOURCE, S.REPORT / 'audit_completed_finetune_checkpoints_v8.py', S.REPORT / 'cost_aggregation_v9.py']
    sources = manifest.get('sources', [])
    require(len(previous['value']['sources']) == 84 and len(sources) == 88
            and len({x['path'] for x in sources}) == 88
            and all(x in sources for x in previous['value']['sources'])
            and all(descriptor(p) in sources for p in required), 'Recovery3 must preserve all84 and bind exactly four new execution sources')
    for source in sources:
        require(descriptor(binding_shape(source)) == source, 'Registered recovery3 source changed')
    require(manifest.get('cpu_auditor') == descriptor(required[2]) and manifest.get('cost_collector') == descriptor(required[3])
            and manifest.get('same_formal_F3_and_E3_core') is True
            and manifest.get('training_protocol_or_hyperparameters_changed') is False,
            'Recovery3 actual CPU/cost source or unchanged core binding differs')
    return {'passed': True, 'registration': binding, 'value': manifest, 'context': context,
            'base_queue_registration': previous['base_queue_registration'], 'source': manifest['source']}


def validate_closure(binding, proof, artifact, record, fit):
    if proof.get('schema') in ('lost_f3_process_closure_v1', 'lost_f3_process_closure_v2'):
        verify_delegated_source()
        return L2.validate_closure(binding, proof, artifact, record, fit)
    require(proof.get('schema') == 'lost_f3_process_closure_v3' and proof.get('status') == 'process_absence_observed'
            and proof.get('passed') is True, 'Expected immutable loss3 closure')
    check_identity(proof); check_identity(fit)
    lineage = verify_recovery_registration(proof['recovery_registration'])
    context = lineage['context']
    archives = ('old_owner_snapshot', 'old_queue_attempt_snapshot')
    require(proof.get('closure_source') == lineage['source'] and proof.get('loss_evidence') == context['loss_evidence']
            and all(proof.get(k) == context[k] for k in CONTEXT_FIELDS if k not in archives),
            'Closure3 differs from verified immutable evidence')
    for key in archives:
        require(proof[key]['sha256'] == context[key]['sha256']
                and read_json(proof[key]) == read_json(context[key]), 'Entry archival bytes differ from loss snapshot')
    verify_absence(proof['runtime_process_absence_evidence'], context['expected_native_identities'])
    require(proof['original_attempt'] == artifact and read_json(artifact) == record,
            'Closure3 does not cover the exact supplied original trace')
    require(all(k in proof and proof[k] is None for k in UNKNOWN_FIELDS)
            and proof.get('graceful_terminal_flush') is False
            and proof.get('known_extra_not_added_again_to_original_returned_counters') is True,
            'Closure3 cannot invent endpoint/flush/tail or add repeated work twice')
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
            'resume_checkpoint_descriptor_not_reopened': context['resume_checkpoint'],
            'pause_native_proof': context['pause_native_proof'],
            'pause_active_elapsed_seconds': None, 'overnight_pause_not_training': True}


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
