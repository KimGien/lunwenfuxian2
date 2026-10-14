"""Read-only cost companion for the optional recovered-Q5 natural boundary.

Never imports a guardian/qualifier, queries processes/GPU, loads tensors, calls
evaluation or runs final cost collection. Monotonic control intervals overlap
each other, the Q5 parent timer and probe ledgers; they are not training cost.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve()
REPORT = SOURCE.parent
WORK = SOURCE.parents[3]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
ROOT = OUT / 'execution_acceleration_v1/sync_boundary_v3'
GUARDIAN = REPORT / 'sync_qualification_boundary_guard_v3.py'
GUARDIAN_SHA = '012cc88f26e6693c5e986a43e92ba72a990acd0d3c0c662e4de6c5ee431efc10'
FINALIZER_SHA = '3cb5d159ce95dd5e56e69bd48841dfba0c3ffce9992cf0042fc5ca809d5ced27'
WIDE_SHA = 'a0a0452e852ee005b88558011ca3696250bf1591f24f7e63140f38fe52ce60db'
L_SHA = '97dd30a04a0f028890c124001a4f06e4f2b22cf86677a3d6f0731fd53b790a57'
QUALIFIERS = ('qualify_sync_reduction_v3.py', 'qualify_training_sync_v3.py')
FIXED = {'forward': 96, 'encoder_forward': 96, 'encoder_windows': 768, 'backward': 96,
         'optimizer_step': 0, 'optimizer_step_stub': 12}


def require(value, message):
    if not value: raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def integer(value): return type(value) is int and value >= 0


class Reader:
    """Only source/JSON/Markdown bytes, with a final same-byte recheck."""
    def __init__(self, work=WORK): self.work, self.inputs = Path(work).resolve(), {}
    def descriptor(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(self.work) and path.suffix in ('.json', '.py', '.md'), 'Only workspace source/JSON/Markdown allowed')
        binding = {'path': path.relative_to(self.work).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        previous = self.inputs.get(binding['path'])
        require(previous is None or previous == binding, 'Artifact changed within this review')
        self.inputs[binding['path']] = binding
        return binding
    def check(self, binding):
        require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Exact source descriptor required')
        path = (self.work / binding['path']).resolve()
        require(self.descriptor(path) == binding, 'Source-bound artifact changed')
        return path
    def read(self, binding):
        path = self.check(binding); require(path.suffix == '.json', 'JSON input required')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'JSON changed during same-byte read')
        return json.loads(raw.decode('utf-8-sig'))
    def recheck(self):
        for binding in list(self.inputs.values()): self.check(binding)


def elapsed_span(start, end, seconds):
    """Validate observations without deriving absent durations from timestamps."""
    require(all(value is None or number(value) for value in (start, end, seconds)), 'Invalid monotonic observation')
    if start is not None and end is not None:
        require(end >= start, 'Reversed monotonic interval')
    if seconds is not None:
        require(start is not None and end is not None and math.isclose(end - start, seconds, rel_tol=1e-9, abs_tol=1e-6),
                'Recorded duration disagrees with monotonic endpoints')
    return {'started_monotonic': start, 'ended_monotonic': end, 'seconds': seconds,
            'duration_unknown': seconds is None, 'added_to_formal_training': False}


def control_spans(value):
    hold = elapsed_span(value.get('hold_started_monotonic'), value.get('hold_ended_monotonic'), value.get('hold_seconds'))
    drain = elapsed_span(value.get('hold_started_monotonic'), value.get('children_wait_ended_monotonic'), value.get('children_wait_seconds'))
    probes = []
    for row in value.get('probe_intervals', []):
        require(row.get('qualifier') in QUALIFIERS, 'Unknown qualification launcher')
        span = elapsed_span(row.get('started_monotonic'), row.get('ended_monotonic'), row.get('seconds'))
        if span['started_monotonic'] is not None:
            for lower in (drain['ended_monotonic'], probes[-1]['ended_monotonic'] if probes else None):
                require(lower is None or span['started_monotonic'] >= lower, 'Probe overlaps training drain or prior probe')
        require(hold['ended_monotonic'] is None or span['ended_monotonic'] is None
                or span['ended_monotonic'] <= hold['ended_monotonic'], 'Probe extends outside dispatcher hold')
        probes.append({'qualifier': row['qualifier'], **span, 'overlaps_primary_engineering_ledger': True})
    names = [row['qualifier'] for row in probes]
    require(names == list(QUALIFIERS[:len(names)]), 'Probe launch order/uniqueness differs')
    require(hold['ended_monotonic'] is None or drain['ended_monotonic'] is None
            or drain['ended_monotonic'] <= hold['ended_monotonic'], 'Training drain exceeds held interval')
    return {'hold': hold, 'training_child_drain': drain, 'probes': probes,
            'intervals_are_nested_not_additive': True, 'GPU_busy_seconds': None,
            'pause_or_overnight_added_to_training': False}


def load_normalizer():
    path = REPORT / 'cost_aggregation_v4.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == '46af6b28f80332f99b22f9373bb5902c0f7c42e9676c54a56da9c2f62eddbd40',
            'Sealed cost normalization source changed')
    spec = importlib.util.spec_from_file_location('_sync_companion_normalizer', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sampled_performance(monitor):
    require(type(monitor) is dict and monitor.get('continuous_exclusivity_verified') is False
            and monitor.get('between_sample_exclusivity') == 'unknown', 'Sampled monitoring cannot establish continuous exclusivity')
    samples, errors = monitor.get('samples'), monitor.get('observation_errors')
    require(type(samples) is list and type(errors) is list and number(monitor.get('observer_query_seconds_observed')),
            'Actual GPU observation/overhead records required')
    for sample in samples:
        require(number(sample.get('started_monotonic')) and number(sample.get('returned_monotonic'))
                and sample['returned_monotonic'] >= sample['started_monotonic']
                and type(sample.get('external_contexts')) is list, 'Invalid sampled GPU observation')
        allowed, contexts = sample.get('allowed_current_qualifier_pids'), sample.get('observation', {}).get('contexts')
        require(type(allowed) is list and allowed and all(integer(pid) and pid > 0 for pid in allowed)
                and len(allowed) == len(set(allowed)) and type(contexts) is list, 'Sample ancestry/contexts absent')
        expected = [item for item in contexts if str(item.get('pid')) not in {str(pid) for pid in allowed}]
        require(sample['external_contexts'] == expected, 'External context classification differs from raw observation')
    require(type(monitor.get('post_probe_GPU_observation', {}).get('contexts')) is list,
            'Post-probe GPU observation absent')
    external = any(sample['external_contexts'] for sample in samples) or bool(monitor.get('post_probe_GPU_observation', {}).get('contexts'))
    require(monitor.get('external_context_observed') is bool(external), 'External context disclosure differs from observations')
    clean = bool(samples) and not external and not errors
    require(monitor.get('sampled_exclusivity_passed') is clean
            and monitor.get('performance_contamination_observed_or_unknown') is (not clean), 'Sampled cleanliness result differs')
    return {'sampled_clean': clean, 'continuous_exclusivity_verified': False, 'between_sample_exclusivity': 'unknown',
            'observer_query_seconds_observed': monitor['observer_query_seconds_observed'],
            'observer_overhead_overlaps_probe_interval': True, 'recorded_monitor': monitor}


def primary_summary(value, artifact, *, qualifier, boundary, source, train_count, val_count):
    training = qualifier == QUALIFIERS[1]
    stem = 'training' if training else 'diagnostic'
    require(value.get('schema') == stem + '_sync_qualification_attempt_v3' and value.get('source') == source
            and type(value.get('qualification_execution_version')) is int and value['qualification_execution_version'] == 3,
            'Actual V3 primary source/schema differs')
    require(value.get('boundary_proof' if training else 'safe_stop_proof') == boundary,
            'Primary attempt belongs to another boundary')
    require((value.get('condition'), value.get('seed'), value.get('fold')) == ('CSBrain-lastblock-finetune', 22, 5)
            and type(value.get('seed')) is int and type(value.get('fold')) is int, 'Fixed probe identity differs')
    require(value.get('status') in ('complete', 'failed', 'running'), 'Unknown primary attempt state')
    require(all(type(value.get(key)) is int and value[key] == 0 for key in ('formal_fits', 'optimizer_updates', 'test_forward_count')),
            'Probe must have zero formal fits/updates/test forwards')
    require(value.get('formal_activation_authorized') is False, 'Optional qualification cannot authorize formal activation')
    require('unflushed_hard_crash_tail_exact' in value and value['unflushed_hard_crash_tail_exact'] is None,
            'Unknown unflushed crash tail must not become exact zero')
    for key, count in value.items():
        if key.endswith(('_started', '_returned')):
            require(integer(count), 'Invalid observed operation count: ' + key)
            if key.endswith('_returned') and key[:-9] + '_started' in value:
                require(count <= value[key[:-9] + '_started'], 'Returned count exceeds started')
    if training:
        require(value.get('fixed_engineering_budget') == FIXED and all(type(x) is int for x in value['fixed_engineering_budget'].values()),
                'Training engineering budget differs')
        require(all(type(value.get('optimizer_step_' + suffix)) is int and value['optimizer_step_' + suffix] == 0
                    for suffix in ('started', 'returned')), 'A real optimizer call was observed')
        if value['status'] == 'complete':
            require(all(type(value.get(key + '_' + suffix)) is int and value[key + '_' + suffix] == expected
                        for key, expected in FIXED.items() for suffix in ('started', 'returned')),
                    'Completed training probe must prove actual96 backwards/0 real steps/12 stubs')
    else:
        require(type(value.get('backward_calls')) is int and value['backward_calls'] == 0, 'Diagnostic probe cannot backpropagate')
        if value['status'] == 'complete':
            people = train_count + val_count
            expected = {'forward': 6 * people, 'encoder_forward': 6 * people * 8, 'encoder_windows': 6 * people * 60}
            require(all(type(value.get(k + '_' + suffix)) is int and value[k + '_' + suffix] == count
                        for k, count in expected.items() for suffix in ('started', 'returned')), 'Diagnostic actual six-pass scope differs')
    seconds = value.get('seconds')
    require(seconds is None or number(seconds), 'Attempt duration is invalid')
    peaks = [value[k] for k in ('peak_allocated_bytes', 'peak_allocated_bytes_observed_at_failure') if value.get(k) is not None]
    require(all(integer(x) for x in peaks), 'Allocated peak must be explicit integer bytes')
    norm = load_normalizer().normalized(value)
    require(not norm['counter_issues'], 'Conflicting operation aliases cannot be silently added')
    return {'artifact': artifact, 'source': source, 'qualifier': qualifier, 'status': value['status'],
            'aggregation_role': 'primary_observed_invocation', 'observed_counters': norm['observed_counters'],
            'seconds_observed': seconds, 'peak_allocated_bytes': max(peaks, default=None),
            'optimizer_step_stub_started': value.get('optimizer_step_stub_started'),
            'optimizer_step_stub_returned': value.get('optimizer_step_stub_returned'),
            'model_constructions_started': value.get('model_constructions_started'),
            'model_constructions_returned': value.get('model_constructions_returned'),
            'gpu_stage_entered_as_recorded': value.get('gpu_stage_entered'), 'formal_optimizer_updates': 0,
            'hard_crash_unflushed_tail_exact': None, 'nested_warmup_and_measurement_seconds_added': False,
            'formal_activation_authorized': False, 'qualification_success_inferred_from_cost': False}


def verify_deferred(value, control, source):
    require(value.get('source') == source and value.get('control_registration') == control
            and value.get('status') == 'deferred_external_GPU_contexts', 'Source-bound external-GPU deferral required')
    require(all(type(value.get(k)) is int and value[k] == 0 for k in
                ('model_constructions', 'GPU_forward_calls', 'backward_calls', 'optimizer_updates', 'formal_fits'))
            and value.get('GPU_speedup_claimed') is False, 'Deferred optional probe must retain actual zero model/GPU work')
    require(value.get('observed_GPU_contexts', {}).get('contexts'), 'Deferral needs the actual nonempty GPU observation')
    require(value.get('seconds') is None or number(value['seconds']), 'Deferral timer invalid')
    return {'status': value['status'], 'observed_GPU_contexts': value['observed_GPU_contexts'],
            'model_constructions': 0, 'GPU_forward_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
            'seconds_observed': value.get('seconds'), 'performance_qualification_passed': False,
            'added_to_training_time': False}


def verify_completed_json(reader, artifacts):
    require(len(artifacts) == len({x['path'] for x in artifacts}) == 90, 'Full15 needs exactly90 bound artifacts')
    by_path = {x['path']: x for x in artifacts}; expected = set()
    for seed in (11, 22, 33):
        for fold in range(1, 6):
            folder = OUT / 'runs/CSBrain-lastblock-finetune' / ('seed_' + str(seed)) / ('fold_' + str(fold))
            status_binding = by_path[(folder / 'status.json').relative_to(WORK).as_posix()]
            completion_binding = by_path[(folder / 'complete.json').relative_to(WORK).as_posix()]
            status, complete = reader.read(status_binding), reader.read(completion_binding)
            require(status.get('status') == 'training_complete' and status.get('evidence') == complete,
                    'Actual completed status/manifest mismatch')
            for key in ('checkpoint', 'train_history', 'validation_history', 'metadata'):
                binding = complete[key]; require(by_path.get(binding['path']) == binding, 'Completed artifact differs from guardian binding')
                require((WORK / binding['path']).resolve().parent == folder.resolve(), 'Completed binding belongs to another fit')
                if key != 'checkpoint': reader.read(binding)
                else: require(Path(binding['path']).suffix == '.pt', 'Expected opaque checkpoint descriptor')
                expected.add(binding['path'])
            expected.update((status_binding['path'], completion_binding['path']))
    require(expected == set(by_path), 'Full15 artifact identity set differs')
    return {'verified_fits': 15, 'verified_JSON_bindings': 75, 'checkpoint_descriptors_preserved_without_reading': 15,
            'checkpoint_bytes_or_tensors_read': 0}


def verify_retained_control(owner, attempt, control):
    """Historical snapshots, not current mutable queue owner or attempt bytes."""
    roles = control['roles']
    require(owner.get('pid') == roles['dispatcher']['pid'] and owner.get('child_pid') == roles['launcher']['pid']
            and owner.get('child_job') == 'CSBrain-lastblock-finetune-fit'
            and attempt.get('child_pid') == roles['launcher']['pid'] and attempt.get('status') == 'running'
            and attempt.get('job', {}).get('module') == 'finetune_v3'
            and attempt['job'].get('id') == owner['child_job'], 'Retained queue identity/route differs')
    require(Path(owner['linux_owner']).with_name('status.json').as_posix() == control['queue_attempt']['path'],
            'Retained owner does not reference this original attempt')
    for value in (owner, attempt):
        require(value.get('execution_registration') == control['registration']
                and type(value.get('execution_version')) is int and value['execution_version'] == 5
                and value.get('recovery_registration') == control['recovery_registration']
                and value.get('recovery_closure') == control['recovery_closure'], 'Retained recovery lineage differs')
    require(attempt.get('registration') == control['registration'], 'Retained attempt base queue differs')
    return {'original_attempt': control['queue_attempt'], 'status_in_retained_snapshot': 'running',
            'recorded_snapshot_seconds': attempt.get('seconds'), 'terminal_parent_seconds_not_read_during_companion': True,
            'not_added_to_formal_or_primary_engineering_cost': True}


def collect(result_path=None):
    reader = Reader(); source = reader.descriptor(SOURCE)
    results = [Path(result_path).resolve()] if result_path else sorted(ROOT.glob('same_dispatcher_resume_complete_*.json'))
    pending = {'schema': 'sync_boundary_v3_cost_companion_v1', 'source': source, 'status': 'pending',
               'passed': False, 'final_plan_completion_certificate': False, 'checkpoint_reads': 0,
               'GPU_operations': 0, 'models_constructed': 0, 'test_or_evaluation_calls': 0}
    if len(results) != 1:
        return {**pending, 'reason': 'Exactly one immutable actual same-dispatcher terminal result required', 'results_observed': len(results)}
    result_path = results[0]
    require(result_path.parent == ROOT.resolve() and result_path.name.startswith('same_dispatcher_resume_complete_'), 'Immutable V3 terminal path required')
    result_binding = reader.descriptor(result_path); terminal = reader.read(result_binding)
    guardian = reader.descriptor(GUARDIAN)
    require(guardian['sha256'] == GUARDIAN_SHA and terminal.get('source') == guardian, 'Reviewed actual guardian source required')
    require(terminal.get('phase') == 'qualification_finished_same_Q5_resumed' and terminal.get('passed') is True
            and terminal.get('boundary_workflow_completed') is True and terminal.get('same_dispatcher_resumed') is True
            and terminal.get('dispatcher_suspended') is False and terminal.get('actual_sync_optimization_applied') is False,
            'Guardian did not confirm the same dispatcher resumed without runtime activation')
    require(type(terminal.get('dispatcher_resume_calls_returned')) is int and terminal['dispatcher_resume_calls_returned'] == 1
            and all(type(terminal.get(k)) is int and terminal[k] == 0 for k in
                    ('GPU_training_child_suspend_or_terminate_calls', 'formal_process_terminate_calls', 'formal_queue_spawn_calls')),
            'Same-parent zero-formal-process-control invariants differ')
    control_binding = terminal['control_registration']; control = reader.read(control_binding)
    require(control_binding == reader.descriptor(ROOT / 'registration.json') and control.get('source') == guardian
            and control.get('schema') == 'recovered_q5_sync_control_registration_v3' and type(control.get('version')) is int
            and control['version'] == 3 and control.get('same_dispatcher_resume_only') is True
            and control.get('formal_training_math_changed') is False and control.get('formal_training_process_termination_permitted') is False,
            'Immutable optional control registration differs')
    inspection = reader.read(control['expected'])
    require(inspection.get('source') == guardian and inspection.get('schema') == 'sync_q5_dynamic_boundary_inspection_v3'
            and inspection.get('roles') == control['roles'] == terminal['held_process_identities']
            and terminal['native_handle_identity'] == control['roles']['dispatcher'], 'Bound inspected/held/resumed identities differ')
    require(all(inspection.get(k) == control[k] for k in ('registration', 'recovery_registration', 'recovery_closure', 'queue_owner', 'queue_attempt', 'native_birth_observation')),
            'Inspection/control lineage changed')
    for role in ('dispatcher', 'launcher', 'gpu_child'):
        value = control['roles'][role]
        require(value.get('role') == role and type(value.get('pid')) is int and value['pid'] > 0
                and str(value.get('creation_100ns', '')).isdigit() and value.get('image'), 'Native identity incomplete')
    require(control['roles']['launcher']['parent_pid'] == control['roles']['dispatcher']['pid']
            and control['roles']['gpu_child']['parent_pid'] == control['roles']['launcher']['pid'], 'Native parent chain differs')
    sys.path.insert(0, str(WORK))
    from tridep_ccml.eeg_all_candidates import lost_f3_closure_v1 as L
    require(reader.descriptor(Path(L.__file__))['sha256'] == L_SHA, 'Reviewed recovery verifier changed')
    lineage = L.verify_recovery_registration(control['recovery_registration'])
    require(control['registration'] == lineage['base_queue_registration']
            and control['protected_sources'] == lineage['value']['sources'] and len(control['protected_sources']) == 80,
            'Original recovered80 sources must remain byte-identical')
    closure = reader.read(control['recovery_closure']); original = reader.read(closure['original_attempt'])
    closed = L.verify_lost_f3_attempt(closure['original_attempt'], original, {k: closure[k] for k in ('condition', 'seed', 'fold')})
    require(closed and closed['passed'] is True and closed['closure'] == control['recovery_closure'], 'Actual prior loss closure changed')
    for item in control['protected_sources'] + control['experimental_sources']: reader.check(item)
    require(terminal.get('experimental_sources') == control['experimental_sources'], 'Terminal experimental source set differs')
    sources = {x['path']: x for x in control['experimental_sources']}
    require(len(sources) == len(control['experimental_sources']), 'Duplicate experimental source identities')
    qualifier_sources = {name: reader.descriptor(REPORT / name) for name in QUALIFIERS}
    require(all(sources.get(x['path']) == x for x in [guardian, *qualifier_sources.values()]), 'Guardian/qualifier source absent from control registration')
    reader.read(control['native_birth_observation'])
    reader.read(control['external_graphics_evidence'])
    # Original Q5 owner/attempt paths mutate after resume. Only retained copies
    # can substantiate their historical bytes; never hash the current owner as
    # though it were the pre-resume descriptor.
    snapshots, retained = {}, {}
    for key, original_binding in [('queue_owner_snapshot', control['queue_owner']), ('queue_attempt_snapshot', inspection['queue_attempt'])]:
        binding = control[key]
        require(binding['sha256'] == original_binding['sha256'], 'Retained original control bytes differ')
        retained[key] = reader.read(binding); snapshots[key] = binding
    parent_timer = verify_retained_control(retained['queue_owner_snapshot'], retained['queue_attempt_snapshot'], control)
    completed = verify_completed_json(reader, terminal['completed_CSBrain_artifacts'])
    spans = control_spans(terminal)
    outcomes = terminal.get('outcomes', [])
    require([x['qualifier'] for x in outcomes] == list(QUALIFIERS[:len(outcomes)]), 'Outcome ordering/uniqueness differs')
    require([x['qualifier'] for x in spans['probes']] == [x['qualifier'] for x in outcomes], 'Probe spans lack exact outcomes')
    performance = []
    for outcome, interval in zip(outcomes, terminal.get('probe_intervals', [])):
        code = outcome.get('returncode'); require(type(code) is int, 'Actual qualifier exit code required')
        observed = sampled_performance(interval.get('performance_exclusivity_monitor'))
        require(outcome.get('qualifier_report_passed') is (code == 0)
                and outcome.get('qualification_passed') is (code == 0 and observed['sampled_clean'])
                and outcome.get('performance_sampled_clean') is observed['sampled_clean']
                and outcome.get('continuous_performance_clean_verified') is False
                and outcome.get('performance_between_samples_unknown') is True
                and outcome.get('speed_results_eligible_for_runtime_activation') is False
                and outcome.get('qualification_declined') is (code == 2)
                and outcome.get('execution_failed') is (code not in (0, 2)), 'Exit interpretation differs')
        performance.append({'qualifier': outcome['qualifier'], **observed,
                            'qualifier_report_passed': outcome['qualifier_report_passed'],
                            'qualification_passed_with_sampled_scope': outcome['qualification_passed'],
                            'runtime_activation_authorized': False})
    require(terminal.get('all_qualifications_passed') is (len(outcomes) == 2 and all(x['qualification_passed'] for x in outcomes)),
            'Overall sampled qualification disclosure differs')
    deferred = None
    if terminal.get('deferred_qualification'):
        deferred_value = reader.read(terminal['deferred_qualification'])
        deferred = {'artifact': terminal['deferred_qualification'], **verify_deferred(deferred_value, control_binding, guardian)}
        require(not outcomes and not spans['probes'], 'Initial zero-work deferral cannot also contain probe work')
    boundary = terminal.get('boundary_proof'); primary = []; aliases = []; missing = []; seen = {}
    reports = []
    if boundary:
        proof = reader.read(boundary)
        require(proof.get('schema') == 'sync_qualification_safe_boundary_v3' and proof.get('source') == guardian
                and proof.get('control_registration') == control_binding and proof.get('passed') is True
                and proof.get('completed_CSBrain_artifacts') == terminal['completed_CSBrain_artifacts'], 'Probe boundary identity differs')
        registry = reader.read(proof['original_training_registration']); split = registry['folds']['fold_5']
        for qualifier in QUALIFIERS:
            stem = 'training' if qualifier == QUALIFIERS[1] else 'diagnostic'
            matches = []
            for path in sorted((OUT / 'supervised_engineering/CSBrain-lastblock-finetune').glob(stem + '_sync_attempt_*.json')):
                value = json.loads(path.read_text(encoding='utf-8-sig'))
                if value.get('boundary_proof' if stem == 'training' else 'safe_stop_proof') != boundary: continue
                artifact = reader.descriptor(path); value = reader.read(artifact)
                summary = primary_summary(value, artifact, qualifier=qualifier, boundary=boundary, source=qualifier_sources[qualifier],
                                          train_count=len(split['train']), val_count=len(split['val']))
                identity = (artifact['sha256'], value.get('started_at'))
                if identity in seen:
                    aliases.append({'artifact': artifact, 'alias_of': seen[identity]}); continue
                seen[identity] = artifact; matches.append(summary)
                for key in ('source_bindings',):
                    for item in (value.get(key, {}).values() if isinstance(value.get(key), dict) else value.get(key, [])): reader.check(item)
                token = path.stem.removeprefix(stem + '_sync_attempt_')
                report_path = REPORT / (stem + '_sync_qualification_' + token + '.json')
                if report_path.exists():
                    report_binding = reader.descriptor(report_path); report = reader.read(report_binding)
                    require(report.get('attempt') == artifact and report.get('source') == qualifier_sources[qualifier]
                            and report.get('schema') == stem + '_sync_qualification_v3' and type(report.get('passed')) is bool
                            and report.get('not_formal_training_authorization') is True, 'Qualification report is not bound to this attempt')
                    reports.append({'artifact': report_binding, 'attempt': artifact, 'passed_as_reported': report['passed'],
                                    'selected_variant_as_reported': report.get('selected_variant'), 'not_independent_tensor_revalidation': True})
            require(len(matches) <= 1, 'Multiple distinct attempts for one launched qualifier require explicit inventory expansion')
            launched = qualifier in [x['qualifier'] for x in outcomes]
            require(not matches or launched, 'Primary probe exists without a guardian launch outcome')
            if launched and not matches: missing.append({'qualifier': qualifier, 'unrecorded_work_unknown': True})
            primary.extend(matches)
    require(not outcomes or boundary is not None, 'Launched probes require an actual bound boundary')
    for outcome in outcomes:
        reported = [r for r in reports if any(p['qualifier'] == outcome['qualifier'] and p['artifact'] == r['attempt'] for p in primary)]
        if outcome['returncode'] in (0, 2):
            require(len(reported) == 1 and reported[0]['passed_as_reported'] is (outcome['returncode'] == 0),
                    'Successful/declined qualifier lacks its actual report')
    require(deferred is not None or boundary is not None, 'No actual qualification or deferral evidence')
    reader.recheck()
    return {**pending, 'status': 'verified_terminal_control_cost', 'passed': True,
        'control_registration': control_binding, 'inspection': control['expected'], 'terminal_result': result_binding,
        'recovery_registration': control['recovery_registration'], 'recovery_closure': control['recovery_closure'],
        'original80_sources_verified': True, 'retained_pre_resume_control_snapshots': snapshots,
        'parent_attempt_timer': parent_timer,
        'unretained_historical_owner_or_attempt_bytes_claimed_verified': False,
        'same_dispatcher_identity': control['roles']['dispatcher'], 'completed_CSBrain_JSON_scope': completed,
        'monotonic_intervals': spans, 'deferred_qualification': deferred,
        'performance_exclusivity_observations': performance,
        'remaining_optional_qualification_deferred_after_GPU_race': terminal.get('remaining_optional_qualification_deferred_after_GPU_race', False),
        'primary_engineering_ledgers': primary, 'duplicate_ledger_aliases': aliases, 'missing_primary_ledger_disclosures': missing,
        'qualification_reports': reports, 'all_qualifications_passed_as_recorded': terminal.get('all_qualifications_passed'),
        'actual_sync_optimization_applied': False, 'control_or_parent_seconds_added_to_formal_training': False,
        'primary_seconds_added_again_to_final_cost': False, 'all_plan_exact_physical_work_or_wall_seconds': None,
        'Q5_parent_attempt_seconds_scope': 'Original run_child timer spans process.wait and delayed parent handling; includes hold/probe intervals and is never added to formal F3 or primary probe timers.',
        'old_finalizer_coverage': {'automatically_inventoried': ['inspection_*.json', 'boundary_*.json', 'optional_sync_qualification_deferred_*.json'],
            'bound_only_by_this_companion': ['registration.json', 'same_dispatcher_resume_complete_*.json', 'queue_owner_snapshot.json', 'queue_attempt_snapshot.json'],
            'training_budget_or_old_finalizer_modified': False}, 'inputs': list(reader.inputs.values())}


def render(value):
    lines = ['# Q5 同进程同步资格检查成本附页', '', '此附页是控制与工程成本证据，不是全计划完成证明，也不表示已启用同步优化。', '',
             'Q5 父尝试计时包含等待子进程、挂起及探针区间；这些时间相互嵌套，不能加到正式 F3 训练或探针主账。', '',
             '| 范围 | 已记录秒数 |', '|---|---:|']
    for name, row in [('父进程挂起', value['monotonic_intervals']['hold']), ('等待原训练子进程退出', value['monotonic_intervals']['training_child_drain'])]:
        lines.append(f"| {name} | {row['seconds'] if row['seconds'] is not None else '未知'} |")
    for row in value['monotonic_intervals']['probes']:
        lines.append(f"| {row['qualifier']} 启动器区间（含主账） | {row['seconds'] if row['seconds'] is not None else '未知'} |")
    lines += ['', '训练探针完整预算为 96 次反向、96 次模型前向、768 个编码窗口、0 次真实优化器更新；12 次 step stub 单列。诊断探针没有反向或优化器更新。以下保留实际记录，失败或延期不冒充完成预算。', '',
              '| 主工程尝试 | 状态 | 整次秒数 | 反向返回 | 真实更新返回 | 峰值 bytes |', '|---|---|---:|---:|---:|---:|']
    for row in value['primary_engineering_ledgers']:
        show = lambda v: '未知' if v is None else str(v)
        c = row['observed_counters']
        lines.append(f"| {row['qualifier']} | {row['status']} | {show(row['seconds_observed'])} | {show(c['backward_returned'])} | {show(c['optimizer_returned'])} | {show(row['peak_allocated_bytes'])} |")
    lines += ['', '外部 GPU 占用导致的延期保留零模型/前向/反向/更新；没有记录的延期耗时仍为未知。主账包含预热与重复，嵌套计时不再次求和；显存峰值取最大值。缺失主账、丢失尾部、历史精确总量均不填零。', '',
              '性能观察只有离散采样；采样间隔内是否独占 GPU 仍未知，观察本身也可能扰动计时。资格报告通过与守护器采样结论分别保留，不声称连续独占或自动授权训练启用。', '',
              '最终交付清单将本附页/JSON与原成本报告和实际完整验收绑定；两个探针主账只能在原成本清单中计一次，本附页不再追加小计。', '']
    return '\n'.join(lines)


def verify_final_shape(cost, acceptance, registration):
    """Only validate existing report structure; do not repeat scientific audit."""
    registered = registration.get('fits', [])
    expected = {fit['fit_id'] for fit in registered}
    require(len(registered) == len(expected) == 360, 'Original unique360 registration required')
    require(type(cost.get('schema_version')) is int and cost['schema_version'] == 6
            and type(cost.get('formal_fits_expected')) is int and cost['formal_fits_expected'] == 360
            and cost.get('formal_status_counts') == {'training_complete': 360}
            and cost.get('input_read_errors') == [], 'Final cost has partial coverage or read errors')
    cost_rows, audit_rows = cost.get('formal_fits', []), acceptance.get('training_rows', [])
    require(len(cost_rows) == len(audit_rows) == 360
            and {r['fit_id'] for r in cost_rows} == {r['fit_id'] for r in audit_rows} == expected,
            'Cost/acceptance must cover each original identity exactly once')
    updates = {row['fit_id']: row.get('updates') for row in audit_rows}
    require(all(integer(value) and value > 0 for value in updates.values())
            and all(integer(row.get('formal_durable_updates')) and row['formal_durable_updates'] == updates[row['fit_id']] for row in cost_rows)
            and cost.get('formal_durable_updates_known_subtotal') == sum(updates.values()), 'Final durable update binding differs')
    require(type(acceptance.get('schema_version')) is int and acceptance['schema_version'] == 6
            and all(type(acceptance.get(k)) is int and acceptance[k] == 360 for k in
                    ('formal_identity_count', 'completed_training_json_verified', 'deep_cpu_checkpoint_scope_verified'))
            and acceptance.get('status_counts') == {'training_complete': 360}
            and acceptance.get('verification_errors') == acceptance.get('incomplete_fits') == []
            and len(acceptance.get('cpu_checkpoint_proofs', [])) == 360, 'Actual acceptance is partial or failed')
    requirements = acceptance.get('requirements', [])
    require(len(requirements) == 30 and {row['id'] for row in requirements} == {'R%02d' % i for i in range(1, 31)}
            and all(row.get('status') == 'verified_complete' for row in requirements), 'One or more full-plan requirements remain pending')
    require(all(acceptance.get(k, {}).get('passed') is True for k in ('evaluation', 'semantic_review', 'final_cost', 'q5_execution_lifecycle')),
            'Actual evaluation/semantic/cost/queue proof absent')
    cpu = cost.get('source_bound_cpu_checkpoint_verification', {})
    require(cpu.get('passed') is True and type(cpu.get('verified_fits')) is int and cpu['verified_fits'] == 135
            and type(cpu.get('prior_actual_cpu_checkpoint_loads')) is int and cpu['prior_actual_cpu_checkpoint_loads'] == 405
            and type(cpu.get('new_cpu_checkpoint_loads')) is int and cpu['new_cpu_checkpoint_loads'] == 0,
            'Actual135/405 existing CPU proof summary absent')
    require(cost.get('actual_completed_report_verifier_called') is True and cost.get('existing_outer_prediction_artifacts_read') is True
            and all(type(cost.get(k)) is int and cost[k] == 0 for k in
                    ('gpu_calls_by_finalizer', 'model_constructions_by_finalizer', 'optimizer_updates_by_finalizer', 'new_predictions_computed')),
            'Existing evaluation verification provenance differs')
    require(cost.get('all_plan_physical_optimizer_returns_exact') is None and cost.get('all_plan_wall_seconds_exact') is None,
            'Unknown historical global totals must remain unknown')
    return {'registered_and_verified_fits': 360, 'existing_finetune_CPU_fit_proof': 135,
            'existing_finetune_CPU_load_proof': 405, 'scientific_checks_reexecuted': False}


def final_parent_timer(reader, cost, companion):
    original = companion['parent_attempt_timer']['original_attempt']
    rows = [r for r in cost['cost_records'] if r['artifact']['path'] == original['path']]
    require(len(rows) == 1 and rows[0]['category'] == 'dispatcher_elapsed'
            and rows[0]['aggregation_role'] == 'supplemental_nested_or_unmeasured', 'Original Q5 parent timer must remain supplemental once')
    row = rows[0]; current = reader.read(row['artifact'])
    retained = reader.read(companion['retained_pre_resume_control_snapshots']['queue_attempt_snapshot'])
    require(current.get('status') == 'complete' and type(current.get('returncode')) is int and current['returncode'] == 0
            and all(current.get(k) == retained.get(k) for k in ('job', 'command', 'started_at', 'child_pid', 'registration',
                'execution_registration', 'execution_version', 'source', 'recovery_registration', 'recovery_execution_source', 'recovery_closure')),
            'Terminal parent attempt is not the original completed invocation')
    seconds = current.get('seconds')
    require(seconds is None or number(seconds), 'Invalid final parent seconds')
    require(row['seconds_observed'] == seconds, 'Final parent timer scope differs from producer')
    return {'artifact': row['artifact'], 'seconds_observed_including_hold_and_probe': seconds,
            'seconds_added_to_formal_or_primary_probe': False, 'pure_training_seconds': None,
            'hold_subtracted_to_infer_training_time': False}


def build_bundle(companion_path, final_cost_path, acceptance_path):
    reader = Reader()
    companion_binding, cost_binding, acceptance_binding = [reader.descriptor(p) for p in (companion_path, final_cost_path, acceptance_path)]
    companion, cost, acceptance = [reader.read(x) for x in (companion_binding, cost_binding, acceptance_binding)]
    require(companion.get('source') == reader.descriptor(SOURCE) and companion.get('passed') is True
            and companion.get('status') == 'verified_terminal_control_cost', 'Verified actual companion required')
    for binding in companion['inputs']: reader.check(binding)
    human = companion['human_readable_control_cost_report']; reader.check(human)
    finalizer = reader.descriptor(REPORT / 'finalize_full_plan_cost_v5.py')
    wide = reader.descriptor(REPORT / 'audit_full_plan_acceptance_v6.py')
    require(finalizer['sha256'] == FINALIZER_SHA and wide['sha256'] == WIDE_SHA, 'Sealed final delivery sources differ')
    require(cost.get('finalizer_source') == finalizer and cost.get('status') == 'verified_final_cost'
            and cost.get('final_snapshot') is True and cost.get('draft_not_completion_certificate') is False
            and cost.get('all_available_artifacts_accounted') is True and cost.get('coverage_gaps_disclosed') is True,
            'Actual final cost artifact required; no finalizer is run here')
    require(acceptance.get('source') == wide and acceptance.get('full_plan_completed') is True
            and acceptance.get('objective_satisfied') is True and acceptance.get('final_cost', {}).get('snapshot') == cost_binding,
            'Actual full acceptance must bind this exact final cost')
    registration = reader.read(cost['registration'])
    require(acceptance.get('registration') == cost['registration'], 'Final artifacts use different original training registrations')
    final_shape = verify_final_shape(cost, acceptance, registration)
    require(cost.get('completion_gate', {}).get('recovery_execution_proof', {}).get('recovery_registration') == companion['recovery_registration']
            and cost['completion_gate']['recovery_execution_proof'].get('recovery_closure') == companion['recovery_closure'],
            'Final cost and control companion recovery lineage differ')
    reader.check(cost['human_readable_cost_report'])
    for primary in companion['primary_engineering_ledgers']:
        rows = [r for r in cost['cost_records'] if r['artifact'] == primary['artifact'] and r['aggregation_role'] == 'primary_observed_invocation']
        require(len(rows) == 1 and rows[0]['observed_counters'] == primary['observed_counters']
                and rows[0]['seconds_observed'] == primary['seconds_observed'] and rows[0]['peak_allocated_bytes'] == primary['peak_allocated_bytes'],
                'Actual probe is absent, duplicated or changed in final cost')
    parent_timer = final_parent_timer(reader, cost, companion)
    reader.recheck()
    return {'schema': 'eeg_final_cost_with_sync_boundary_v3_bundle_v1', 'source': reader.descriptor(SOURCE), 'passed': True,
            'final_cost': cost_binding, 'final_cost_human_report': cost['human_readable_cost_report'],
            'full_plan_acceptance': acceptance_binding, 'sync_control_cost_companion': companion_binding,
            'sync_control_human_report': human, 'all_probe_primaries_already_in_final_cost_once': True,
            'existing_final_artifact_shape_verified': final_shape, 'parent_attempt_timer': parent_timer,
            'companion_costs_added_again': False, 'full_plan_verification_rerun': False,
            'GPU_or_evaluation_calls': 0, 'inputs': list(reader.inputs.values())}


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--companion', type=Path)
    parser.add_argument('--final-cost', type=Path)
    parser.add_argument('--final-acceptance', type=Path)
    args = parser.parse_args(); stamp = str(time.time_ns())
    if any((args.companion, args.final_cost, args.final_acceptance)):
        require(all((args.companion, args.final_cost, args.final_acceptance)) and args.result is None, 'Bundle requires all three existing artifacts')
        value = build_bundle(args.companion, args.final_cost, args.final_acceptance)
        path = REPORT / ('final_cost_sync_control_bundle_' + stamp + '.json')
    else:
        value = collect(args.result)
        path = REPORT / ('sync_boundary_v3_cost_companion_' + stamp + '.json')
        if value['passed']:
            human = REPORT / ('同步资格控制成本附页_' + stamp + '.md')
            with human.open('x', encoding='utf-8', newline='\n') as stream: stream.write(render(value))
            value['human_readable_control_cost_report'] = Reader().descriptor(human)
    write_new(path, value)
    print(json.dumps({'artifact': Reader().descriptor(path), 'status': value.get('status'), 'passed': value['passed'],
                      'new_models_GPU_or_evaluation_calls': 0}, ensure_ascii=False))


if __name__ == '__main__': main()
