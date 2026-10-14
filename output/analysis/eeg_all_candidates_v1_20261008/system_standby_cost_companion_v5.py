"""Read-only recorded-timer scope for the observed Oct10 system low-power window.

System residency counters do not measure per-process nontraining time. Preserve
producer elapsed, never subtract wall/SleepDuration/DRIPS, and never infer current
sleep from an unmatched event in a bounded historical window. Standard library
only: source/JSON/XML/Markdown, no project imports or scientific/process work.
"""
from __future__ import annotations
from datetime import datetime
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

SOURCE = Path(__file__).resolve()
REPORT = SOURCE.parent
WORK = SOURCE.parents[3]
REL = 'output/analysis/eeg_all_candidates_v1_20261008/'
OUT = 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/'
EVENTS = {'path': REL+'system_power_event_window_1791635064778.json', 'sha256': '8b55ffd71637b10b583daaadb1aef2f2263f10ec9e94b1c2c99194c3439d8670'}
REVIEW = {'path': REL+'independent_system_sleep_overlap_review_1791635154808605000.json', 'sha256': '29727e34ca640854af6e6594677da32a0495797dc490737193affbccda1c2544'}
NATIVE = {'path': OUT+'monitoring/system_standby/native_observation_1791635509483.json', 'sha256': 'c2b499e646c93698c4baee377be514104b0558b382d0153a2e5b19410a38c83c'}
OWNER_LINK = {'path': OUT+'monitoring/system_standby/owner_snapshot_link_1791635509483.json', 'sha256': 'adcb75e61501f9516fd54c1fb57fb20773ba4d5f5326355286a7716cd006fdcf'}
OWNER_SNAPSHOT = {'path': OUT+'monitoring/system_standby/owner_exact_1791635509483.json', 'sha256': '83dced47231f58bd28ef41c4f00d106cc36331c1b5d86851710865eb52eb0791'}
CPU_AUDIT = {'path': REL+'completed_finetune_checkpoint_audit_v5_1791635187184450200.json', 'sha256': 'afd12763a3de82993fbc206ccacea053671e78142298e156269916116554022c'}
FIT_ID = 'LUNA-lastblock-finetune/seed_33/fold_1'
FIT = OUT+'runs/'+FIT_ID+'/'
F3 = OUT+'finetune_v2_attempts/LUNA-lastblock-finetune/f_1791631397731004200_7e6d59cd/attempt.json'
PARENT = OUT+'continuation_queue_v5/attempts/1791625233145752000-LUNA-lastblock-finetune-fit/status.json'
STARTS = {'F3_formal_attempt': '2026-10-10T19:23:17.731040+08:00', 'Q5_parent_dispatcher': '2026-10-10T17:40:33.146077+08:00'}
MOD = 'tridep_ccml/eeg_all_candidates/'
PINNED = {
    'F3': {'path': MOD+'finetune_v3.py', 'sha256': '13601bd5f78b5c543bbc9d6c4dcb856fd9e72a6d09c55d18a5a38febeb670e59'},
    'F1': {'path': MOD+'finetune.py', 'sha256': 'cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24'},
    'Q5': {'path': MOD+'continuation_queue_v5.py', 'sha256': 'e9a281d44b2e5361215ad178bc6eff96e46327c56cf3b9c2c7050081c9356971'},
    'cost6': {'path': REL+'cost_aggregation_v6.py', 'sha256': 'a16f6562dd9809991639fe77df7fa7cd0f9ca94a264c6a8f31f80f7a1012aa8b'},
    'Q5_registration': {'path': OUT+'continuation_queue_v5/registration.json', 'sha256': 'a9db8e215f1d32cd215d550319c0fa1646985c54d2167783a7547b3a6ea5312f'},
    'recovery_registration': {'path': OUT+'continuation_queue_v5/recovery_registration.json', 'sha256': 'ec1500ce2671600fca1588e41f14fc559e12300824aaa5c4a2625671f8d2ec9f'},
}
PAUSE_SOURCE = {'path': 'output/analysis/eeg_all_candidates_v1_20261008/pause_memory_resume_cost_companion_v5.py', 'sha256': '0f4bd7219f6669db5eac0c562103ede6c4aa695d709a3459c10311165fdcaf3b'}
SCOPES = ('F3_formal_attempt', 'F1_metadata_subscope', 'Q5_parent_dispatcher')
NATIVE_IDS = {38860: ('134360707153787861', 50164), 45980: ('134360707154222972', 38860),
              25396: ('134360988331542432', 45980), 50296: ('134360988331626948', 25396)}


def require(value, message):
    if not value: raise ValueError(message)


def finite(value): return type(value) in (int, float) and math.isfinite(value) and value >= 0


def instant(value):
    require(type(value) is str, 'Timestamp must be text')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Timestamp lacks timezone')
    return result


MAMBA_MANIFEST = {'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/evaluation/EEGMamba-lastblock-finetune/predictions_manifest.json', 'sha256': '9d3a15c32609ae74f7db91f1ad9105f616a0135dfa51de5a0a484645a71d5661'}

def fixed_manifest_path(path):
    return Path(path).resolve() == (WORK / MAMBA_MANIFEST['path']).resolve()

def validate_fixed_manifest_metadata(path, raw):
    if not fixed_manifest_path(path): return None
    require(hashlib.sha256(raw).hexdigest()==MAMBA_MANIFEST['sha256'], 'Fixed E12 manifest bytes changed')
    value=json.loads(raw.decode('utf-8-sig'))
    require(type(value) is dict and set(value)=={'passed','condition','test_gate','registration','supervised_fits_evaluated','primary_oof_predictions','distinct_subjects','evaluation_fit_artifacts','evaluator_source','global_report_generated','evaluator_execution'}, 'Only fixed descriptor/count manifest metadata permitted')
    require(value['passed'] is True and value['condition']=='EEGMamba-lastblock-finetune' and value['global_report_generated'] is False, 'E12 metadata condition/status differs')
    require(all(type(value[k]) is int and value[k]==n for k,n in (('supervised_fits_evaluated',15),('primary_oof_predictions',111),('distinct_subjects',37))), 'Metadata counts differ')
    expected={OUT+'evaluation/EEGMamba-lastblock-finetune/seed_'+str(seed)+'/fold_'+str(fold)+'/best/complete.json' for seed in (11,22,33) for fold in range(1,6)}
    rows=value['evaluation_fit_artifacts']
    require(type(rows) is list and len(rows)==15 and all(type(d) is dict and set(d)=={'path','sha256'} and type(d['sha256']) is str and len(d['sha256'])==64 for d in rows) and {d['path'] for d in rows}==expected, 'Fixed fifteen completion descriptors required')
    require(value['evaluator_source']=={'path':'tridep_ccml/eeg_all_candidates/evaluate_v12.py','sha256':'9a84b1cb5994571f1c486abf59a58a5100f111dbda2bd9ed93dec643bb532473'} and value['evaluator_execution']['version']==12 and value['evaluator_execution']['wrapper_source']==value['evaluator_source'], 'Actual E12 metadata provenance differs')
    return value

class Reader:
    def __init__(self, root=WORK): self.root, self.inputs = Path(root).resolve(), {}
    def path(self, path):
        target = (self.root / path).resolve()
        require(target.is_relative_to(self.root) and target.suffix.lower() in ('.json', '.xml', '.py', '.ps1', '.md'), 'Only workspace source/JSON/XML/Markdown permitted')
        require(fixed_manifest_path(target) or not any(x in target.name.lower() for x in ('prediction', 'probabilit', 'oof_')), 'Prediction evidence is outside scope')
        return target
    def descriptor(self, path, raw):
        return {'path': self.path(path).relative_to(self.root).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}
    def bind(self, path, raw):
        binding = self.descriptor(path, raw)
        require(self.path(path).read_bytes() == raw, 'Evidence changed before immutable binding')
        require(binding['path'] not in self.inputs or self.inputs[binding['path']] == binding, 'Immutable evidence changed')
        self.inputs[binding['path']] = binding
        return binding
    def read(self, binding):
        require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Exact descriptor required')
        path = self.path(binding['path']); raw = path.read_bytes(); validate_fixed_manifest_metadata(path, raw)
        require(self.descriptor(path, raw) == binding, 'Bound evidence hash differs')
        self.bind(path, raw)
        return json.loads(raw.decode('utf-8-sig')) if path.suffix.lower() == '.json' else raw.decode('utf-8-sig')
    def recheck(self):
        for binding in list(self.inputs.values()): self.read(binding)


def parse_events(events, review):
    require(events.get('schema') == 'readonly_windows_power_event_window_v1' and events.get('query_status') == 'success', 'Successful actual event snapshot required')
    require(review.get('schema') == 'independent_system_sleep_overlap_review_v1'
            and review.get('passed_for_observed_event_evidence') is True and review.get('source_event_snapshot') == EVENTS, 'Independent event binding differs')
    rows = events['all_power_provider_events_in_same_window']; structured = []
    ns = {'e': 'http://schemas.microsoft.com/win/2004/08/events/event'}
    require(len(rows) == len({r['record_id'] for r in rows}) == 7, 'Exact unique event window required')
    for row in rows:
        xml = row['xml']; require('<!DOCTYPE' not in xml.upper() and '<!ENTITY' not in xml.upper(), 'External/entity declarations prohibited')
        tree = ET.fromstring(xml); system = tree.find('e:System', ns)
        require(system is not None, 'System event header absent')
        identity = (system.find('e:Provider', ns).get('Name'), int(system.findtext('e:EventID', namespaces=ns)),
                    int(system.findtext('e:EventRecordID', namespaces=ns)), system.find('e:TimeCreated', ns).get('SystemTime'))
        require(identity == (row['provider'], row['event_id'], row['record_id'], row['time_created_utc']), 'XML event identity differs')
        nodes = tree.findall('e:EventData/e:Data', ns); data = {x.get('Name'): x.text for x in nodes}
        require(len(data) == len(nodes) and None not in data, 'Duplicate/unnamed XML event data')
        require(instant(events['query_utc_start']) <= instant(identity[3]) <= instant(events['query_utc_end']), 'Event outside declared window')
        structured.append({k: row[k] for k in ('record_id', 'provider', 'event_id', 'time_created_utc', 'time_created_local')} | {'data': data})
    require(structured == review['structured_provider_event_data'], 'Independent structured XML differs')
    by_id = {r['record_id']: r for r in structured}; entry, end, unpaired = (by_id[i] for i in (149303, 149310, 149313))
    require(entry['event_id'] == 506 and end['event_id'] == 507 and unpaired['event_id'] == 506, 'Entry/exit event IDs differ')
    require(all(r['provider'] == 'Microsoft-Windows-Kernel-Power' and r['data']['BootId'] == '218' for r in (entry, end, unpaired)), 'Provider/boot differs')
    require(entry['data']['ScenarioInstanceIdV2'] == end['data']['ScenarioInstanceIdV2'] == '253'
            and unpaired['data']['ScenarioInstanceIdV2'] == '256', 'Pair must use ScenarioInstanceIdV2')
    require(instant(entry['time_created_utc']) < instant(end['time_created_utc']) < instant(unpaired['time_created_utc']), 'Event sequence reversed')
    data = end['data']; require(data['SleepEntered'] == 'true', 'SleepEntered evidence absent')
    measurements = {k: int(data[k]) for k in ('DurationInUs', 'SleepDurationInUs', 'ActiveResidencyInUs', 'DripsResidencyInUs')}
    require(all(v > 0 for v in measurements.values()), 'Actual nonzero active/residency counters absent')
    require(not any(r['event_id'] == 507 and r['data'].get('BootId') == '218' and r['data'].get('ScenarioInstanceIdV2') == '256' for r in structured), 'New paired exit needs expanded reviewed scope')
    require(review.get('second_entry', {}).get('exit_observed_inside_requested_window') is False
            and review['second_entry'].get('current_state_not_inferred_from_bounded_window') is True, 'Unpaired event scope differs')
    require(all(k in review and review[k] is None for k in ('exact_nontraining_interval_seconds', 'exact_active_training_seconds', 'exact_deduction_from_producer_timers_seconds')), 'System review fabricated exact process timing')
    return {'entry_record_id': 149303, 'exit_record_id': 149310, 'boot_id': 218, 'scenario_instance_v2': 253,
            'entry_utc': entry['time_created_utc'], 'exit_utc': end['time_created_utc'],
            'raw_provider_residency_microseconds': measurements, 'sleep_entered_reported': True,
            'unpaired_entry_record_id': 149313, 'unpaired_entry_utc': unpaired['time_created_utc'],
            'unpaired_exit_observed_in_window': False, 'current_sleep_state_inferred': False,
            'system_interval_is_not_per_process_zero_work': True, 'timer_deduction_seconds': None}


def observe(reader, path):
    """Bounded active JSON read; only stable terminal bytes enter immutable inputs."""
    target = reader.path(path)
    for _ in range(3):
        try:
            raw = target.read_bytes(); record = json.loads(raw.decode('utf-8-sig'))
            require(type(record) is dict, 'Attempt must be an object')
            if record.get('status') not in ('complete', 'failed'): return record, None, 'still_nonterminal'
            if target.read_bytes() != raw: return None, None, 'changed_during_observation'
            return record, reader.bind(path, raw), None
        except (PermissionError, FileNotFoundError, json.JSONDecodeError): time.sleep(.01)
    return None, None, 'transient_read_unresolved'


def verify_native(value, interval):
    require(value.get('schema') == 'system_standby_readonly_native_observation_v1'
            and instant(value['observed_at']) > instant(interval['exit_utc']), 'Actual post-window native observation required')
    rows = value['identities']; require(len(rows) == 4 and {r['pid'] for r in rows} == set(NATIVE_IDS), 'Four original native objects required')
    for row in rows:
        birth, parent = NATIVE_IDS[row['pid']]
        require(type(row['pid']) is type(row['parent_pid']) is int and row['creation_100ns'] == birth and row['parent_pid'] == parent,
                'Native birth/parent identity differs')
        require(row.get('alive') is True and type(row.get('native_exit_code')) is int and row['native_exit_code'] == 259
                and type(row.get('suspended_threads')) is int and row['suspended_threads'] == 0, 'Actual observed alive/unsuspended state differs')
        if row['pid'] in (25396, 50296):
            require('tridep_ccml.eeg_all_candidates.finetune_v3 --conditions LUNA-lastblock-finetune' in row['command_line'], 'LUNA worker command identity differs')
    owner = value['owner_snapshot']
    require(owner.get('pid') == 45980 and owner.get('child_pid') == 25396 and owner.get('child_job') == 'LUNA-lastblock-finetune-fit'
            and owner.get('source') == PINNED['Q5'] and owner.get('execution_registration') == PINNED['Q5_registration']
            and owner.get('recovery_registration') == PINNED['recovery_registration'], 'Retained owner snapshot identity differs')
    require(value.get('no_pause_or_replay_added_by_observer') is True
            and all(type(v) is int and v == 0 for v in value['operations'].values()), 'Observer did work beyond read-only scope')
    require(all(k in value and value[k] is None for k in ('active_training_seconds_exact', 'standby_monotonic_seconds_exact')), 'Native observation does not establish exact active or standby time')
    return {'artifact': NATIVE, 'observed_at': value['observed_at'], 'pids_births': {str(k): b for k, (b, _) in NATIVE_IDS.items()},
            'original_native_objects_observed_alive': True, 'continuous_activity_through_entire_window_proven': False,
            'prior_19_30_and_20_22_entries_are_conversation_excerpts_not_new_native_captures': True,
            'live_owner_or_progress_binding_rechecked': False, 'observation_does_not_prove_current_liveness_at_final_delivery': True}


def verify_owner_snapshot(observation, link, snapshot):
    require(link.get('schema') == 'system_standby_owner_snapshot_link_v1' and link.get('observation') == NATIVE
            and link.get('snapshot') == OWNER_SNAPSHOT, 'Retained owner snapshot link differs')
    require(link.get('matches_observation_owner_binding_sha') is True and link.get('captured_after_observation_while_same_owner_bytes') is True
            and link.get('original_live_owner_not_rechecked_at_future_final') is True, 'Owner snapshot scope differs')
    require(OWNER_SNAPSHOT['sha256'] == observation['owner_binding']['sha256'] and snapshot == observation['owner_snapshot'], 'Retained exact bytes/embedded owner disagree')
    return {'link': OWNER_LINK, 'snapshot': OWNER_SNAPSHOT, 'matches_original_owner_bytes': True, 'current_live_owner_read': False}


def verify_attempt(record, kind, interval):
    require(record['started_at'] == STARTS[kind] and instant(record['started_at']) < instant(interval['entry_utc']), 'Original invocation start differs/does not overlap entry')
    if kind == 'F3_formal_attempt':
        require((record.get('condition'), record.get('seed'), record.get('fold'), record.get('kind')) == ('LUNA-lastblock-finetune', 33, 1, 'formal'), 'Formal identity differs')
        require(type(record['seed']) is type(record['fold']) is int and record['implementation']['wrapper_source'] == PINNED['F3']
                and type(record['implementation']['version']) is int and record['implementation']['version'] == 3, 'Actual F3 source differs')
    else:
        require(record.get('source') == PINNED['Q5'] and record.get('execution_registration') == PINNED['Q5_registration']
                and record.get('recovery_registration') == PINNED['recovery_registration']
                and type(record.get('execution_version')) is int and record['execution_version'] == 5
                and record['job']['id'] == 'LUNA-lastblock-finetune-fit' and type(record.get('child_pid')) is int and record['child_pid'] == 25396, 'Actual original Q5 invocation differs')
    if record.get('status') in ('complete', 'failed'):
        require(instant(record['finished_at']) > instant(interval['exit_utc']), 'Terminal invocation did not span matched event interval')
        require(finite(record.get('seconds')), 'Actual terminal producer seconds absent')
        if kind == 'F3_formal_attempt':
            require(record.get('graceful_terminal_flush') is True and record.get('counter_snapshot_exact_for_process') is True, 'No observed graceful terminal flush')
        else: require(type(record.get('returncode')) is int, 'Parent actual returncode absent')


def timer_row(kind, artifact=None, seconds=None, status='pending'):
    require(kind in SCOPES and status in ('pending', 'actual_terminal'), 'Unknown timer scope/status')
    require((status == 'pending' and seconds is None and artifact is None)
            or (status == 'actual_terminal' and finite(seconds) and type(artifact) is dict), 'Pending cannot claim terminal elapsed')
    return {'scope': kind, 'status': status, 'artifact': artifact,
            'recorded_elapsed_including_observed_low_power_window_seconds': seconds,
            'active_training_elapsed_exact_seconds': None, 'standby_nontraining_exact_seconds': None,
            'active_known_subtotal_contribution': None, 'eligible_for_active_known_subtotal': False,
            'producer_elapsed_preserved': True, 'wall_or_residency_subtracted': False,
            'nested_with_formal_attempt': kind != 'F3_formal_attempt'}


def collect(final_cost, final_acceptance):
    reader = Reader(); source = reader.bind(SOURCE, SOURCE.read_bytes())
    helper_path=reader.path(PAUSE_SOURCE['path']);helper_source=reader.read(PAUSE_SOURCE)
    helper={'__file__':str(helper_path),'__name__':'_text_only_loss_compat'}
    exec(compile(helper_source,str(helper_path),'exec'),helper)
    cr=helper['Reader']();cost,cd=cr.snapshot(final_cost);wide,wd=cr.snapshot(final_acceptance)
    cost=reader.read(cd);wide=reader.read(wd)
    pair=helper['verify_final_pair'](reader.read,cost,wide,cd,wd)
    losses=helper['verify_loss_chain'](reader.read,cost)
    events = reader.read(EVENTS); review = reader.read(REVIEW)
    interval = parse_events(events, review)
    for binding in PINNED.values(): reader.read(binding)
    native_value = reader.read(NATIVE)
    native = verify_native(native_value, interval)
    native['retained_owner'] = verify_owner_snapshot(native_value, reader.read(OWNER_LINK), reader.read(OWNER_SNAPSHOT))
    cpu = reader.read(CPU_AUDIT)
    timers = [timer_row(s) for s in SCOPES]; observed = {}
    for kind, path, index in (('F3_formal_attempt', F3, 0), ('Q5_parent_dispatcher', PARENT, 2)):
        record, binding, reason = observe(reader, path)
        if record is not None:
            verify_attempt(record, kind, interval); observed[kind] = record
        if binding is None and kind=='Q5_parent_dispatcher':
            lost=[r for r in losses['lost_parent_scopes'] if r['artifact']['path']==PARENT]
            require(len(lost)==1, 'Original LUNA parent loss proof is required')
            original=reader.read(lost[0]['artifact']);verify_attempt(original,kind,interval)
            require(record==original and original.get('status')=='running' and original.get('seconds') is None,'Lost parent bytes changed')
            timers[index]={**timer_row(kind),'status':'lost_event_closed_unknown_elapsed','artifact':lost[0]['artifact'],
                'producer_terminal_status':None,'producer_status':'running','producer_terminal':False,
                'lifecycle_closed_by_actual_loss':True,'loss_closure':lost[0]['closure']}
        elif binding is None: timers[index]['pending_reason'] = reason
        else:
            timers[index] = timer_row(kind, binding, record['seconds'], 'actual_terminal')
            timers[index]['producer_terminal_status'] = record['status']
    complete = reader.path(FIT+'complete.json')
    if (timers[0]['status'] == 'actual_terminal'
            and observed['F3_formal_attempt']['status'] == 'complete' and complete.exists()):
        raw = complete.read_bytes(); binding = reader.bind(complete, raw); evidence = json.loads(raw.decode('utf-8-sig'))
        require(evidence.get('passed') is True and evidence['metadata']['path'] == FIT+'metadata.json', 'Completed metadata identity differs')
        metadata = reader.read(evidence['metadata'])
        require((metadata.get('condition'), metadata.get('seed'), metadata.get('fold')) == ('LUNA-lastblock-finetune', 33, 1), 'Metadata fit differs')
        require(metadata.get('execution_acceleration') == observed['F3_formal_attempt'].get('execution_acceleration'), 'Metadata execution differs')
        timers[1] = timer_row('F1_metadata_subscope', evidence['metadata'], metadata['seconds_this_attempt'], 'actual_terminal')
        timers[1]['completion_binding'] = binding
    elif timers[0].get('producer_terminal_status') == 'failed':
        timers[1]['pending_reason'] = 'Original affected F3 failed; later successful retry metadata cannot be assigned to this invocation. Additional reviewed scope required.'
    terminal = all(t['status'] == 'actual_terminal' for t in timers)
    closed=all(t['status'] in ('actual_terminal','lost_event_closed_unknown_elapsed') for t in timers)
    reader.recheck()
    return {'schema': 'system_standby_cost_companion_v5', 'passed': True,
        'status': 'verified_low_power_scope_and_closed_timer_lifecycles' if closed else 'verified_low_power_scope_with_terminal_timers_pending',
        'final_plan_completion_certificate': False, 'source': source,
        'system_event_snapshot': EVENTS, 'independent_system_event_review': REVIEW, 'system_low_power_evidence': interval,
        'native_observation_verification': native,
        'observed_low_power_overlap': True, 'system_interval_is_not_per_process_zero_work': True,
        'no_exact_timer_subtraction': True, 'standby_nontraining_exact_seconds': None, 'active_training_elapsed_exact_seconds': None,
        'affected_active_known_subtotal': None, 'affected_formal_fit_id': FIT_ID,
        'affected_active_unknown_formal_fit_count': 1, 'affected_active_unknown_scopes': 3,
        'affected_original_paths': {'F3_formal_attempt': F3, 'F1_metadata_subscope': FIT+'metadata.json', 'Q5_parent_dispatcher': PARENT},
        'all_affected_producer_timers_terminal': terminal, 'all_affected_timer_lifecycles_closed':closed,
        'final_artifacts':pair,'loss_chain':losses,'affected_timer_observations': timers,
        'same_attempt_identity_preserved': len(observed) == 2,
        'native_process_continuity_independently_verified_by_this_helper': False,
        'new_loss_closure_created': False, 'no_new_process_or_loss_closure_created': True,
        'no_new_process_or_loss_closure_created_scope': 'This read-only companion creates neither; original attempt identity alone does not independently prove continuous native liveness.',
        'extra_optimizer_updates_added': 0, 'extra_backward_calls_added': 0,
        'controller_or_parent_seconds_added_to_formal_cost': False, 'old_cost6_or_noon_gate_already_excludes_this_timer': False,
        'future_low_power_other_timers_require_additional_bound_evidence': True,
        'all_future_sleep_or_timer_scopes_covered': False, 'unpaired_entry_does_not_establish_current_sleep': True,
        'final_scope_closure_ready': False, 'final_scope_closure_required': True,
        'final_scope_closure_reason': 'Bounded historical window ends with unpaired session256. A new source-bound read-only event-range review must identify all additional affected formal identities before final pause-scope closure; this one LUNA fit does not establish that closure.',
        'CPU_audit_observer_elapsed_added_to_formal_cost': False,
        'CPU_audit_standby_adjusted_exact_elapsed': None,
        'CPU_audit_observer_evidence': {'artifact': CPU_AUDIT, 'scope': 'Existing CPU-only observer report; no new checkpoint loads by this helper; its elapsed is not formal training',
                                      'recorded_checkpoint_loads': cpu.get('checkpoint_loads'), 'recorded_elapsed_seconds': cpu.get('seconds')},
        'scope_note': 'Only the bound event window and this original LUNA invocation are reviewed. Active/DRIPS are system observations, not process execution time. CPU audit elapsed is a separate observer scope. Future low-power windows overlapping other invocations need additional source-bound evidence; no new restart/loss or work counters are inferred.',
        'immutable_inputs': list(reader.inputs.values()),
        'fixed_manifest_metadata_exception':MAMBA_MANIFEST, 'operations': {'binary_or_checkpoint_reads': 0, 'tensor_deserializations': 0, 'models_GPU_process_operations': 0,
                       'raw_feature_outertest_reads': 0, 'project_verifier_calls': 0, 'final_cost_collectors_or_delivery_gates_called': 0}}


def human(value):
    lines = ['# 系统低功耗重叠成本附页', '', '本附页不是全计划完成证明。系统事件证实所记录低功耗时段与该 LUNA 尝试重叠；Active 与 DRIPS 均非零，不能声称整段停止训练。', '',
        '原始计时保留，精确有效训练时长、非训练暂停时长及可扣除秒数均未知。不用工具延迟、事件墙钟差或系统驻留计数扣除训练秒数。', '',
        '| 范围 | 状态 | 原始秒数 | 精确有效训练秒数 |', '|---|---|---:|---|']
    for row in value['affected_timer_observations']:
        seconds = row['recorded_elapsed_including_observed_low_power_window_seconds']
        lines.append(f"| {row['scope']} | {row['status']} | {seconds if seconds is not None else '未知'} | 未知 |")
    lines += ['', '只涉及一条正式 fit 及两个重叠计时范围，元数据、父进程和 CPU 审计秒数均不加到正式计时；不增加更新/反向次数，不创建丢失进程记录。', '',
        '窗口内第二条进入事件未见配对退出，不据此推断当前仍睡眠。本附页不覆盖其他未来低功耗事件或训练尝试；如另有受影响范围，须追加绑定证据。旧 cost6 与中午暂停 gate 尚未排除本次 LUNA 计时。', '',
        '最终范围尚未闭合（final_scope_closure_ready=false）：需新的只读事件范围复核，确认所有额外受影响正式身份后，最终交付 gate 才能完成暂停范围校验。当前三项计时即使已结束也不解除此要求。', '']
    lines += ['', '旧 LUNA 父尝试以真实 loss2 证明关闭生命周期，生产者仍 running、elapsed 未知；这不是自然退出或数字终态。', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--final-cost',type=Path);parser.add_argument('--final-acceptance',type=Path);args=parser.parse_args()
    if not args.run: print(json.dumps({'status':'prepared_only','operations':0})); return 0
    require(args.final_cost and args.final_acceptance,'Both actual final artifacts required')
    value = collect(args.final_cost,args.final_acceptance); stamp = str(time.time_ns())
    md = REPORT / ('系统低功耗重叠成本附页_'+stamp+'.md')
    with md.open('x', encoding='utf-8', newline='\n') as stream: stream.write(human(value))
    value['human_readable_report'] = Reader().descriptor(md, md.read_bytes())
    target = REPORT / ('system_standby_cost_companion_v5_'+stamp+'.json')
    with target.open('x', encoding='utf-8', newline='\n') as stream: json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'artifact': Reader().descriptor(target, target.read_bytes()), 'status': value['status'], 'passed': value['passed']}))


if __name__ == '__main__': main()
