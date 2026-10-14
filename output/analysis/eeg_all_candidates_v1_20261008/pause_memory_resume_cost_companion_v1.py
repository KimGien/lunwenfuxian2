"""Read-only cost scope for the exact Oct10 noon native pause/memory resume.

Only stdlib and source/JSON/Markdown bytes. No project imports, binary hashes,
process observation/control, tensors, evaluation or final cost collection.
Producer elapsed values remain untouched. Missing pause QPC prevents exact
active-time correction even when both resume QPC endpoints are available.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

SOURCE = Path(__file__).resolve()
REPORT = SOURCE.parent
WORK = SOURCE.parents[3]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
PAUSE = OUT / 'pause_events/q5_user_pause_1791607365360'
RESUMER = REPORT / 'resume_paused_q5_20261010_v1.ps1'
RESUMER_SHA = '1f227764bc5b82351067319d0815e5d5926bb80baf3a7446b2ff55f761684761'
PAUSE_SHA = 'fcbd717850d8cee8bc57ceb0594cf8dab482d10363f618f8d5cae7406bc37b3b'
NATIVE_SHA = 'acb9b67e4a15548cbcab476fb308ce29bfdc8e2d61a2fd35a39337cce5ac42b2'
F3 = OUT / 'finetune_v2_attempts/REVE-lastblock-finetune/f_1791607293628173200_cafa8752/attempt.json'
FIT = OUT / 'runs/REVE-lastblock-finetune/seed_22/fold_2'
PARENT = OUT / 'continuation_queue_v5/attempts/1791600598599694900-REVE-lastblock-finetune-fit/status.json'
IDENTITY = ('REVE-lastblock-finetune', 22, 2)
ROLES = {'gpu_child': (31128, '134360741986202743'), 'launcher': (16096, '134360741986106595'),
         'dispatcher': (45980, '134360707154222972'), 'dispatcher_launcher': (38860, '134360707153787861')}
TIMER_SOURCES = {
    'finetune_v3.py': '13601bd5f78b5c543bbc9d6c4dcb856fd9e72a6d09c55d18a5a38febeb670e59',
    'finetune.py': 'cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24',
    'continuation_queue_v5.py': 'e9a281d44b2e5361215ad178bc6eff96e46327c56cf3b9c2c7050081c9356971'}
COST_SHA = 'a16f6562dd9809991639fe77df7fa7cd0f9ca94a264c6a8f31f80f7a1012aa8b'


def require(value, message):
    if not value: raise ValueError(message)


def finite(value): return type(value) in (int, float) and math.isfinite(value) and value >= 0
def integer(value): return type(value) is int and value >= 0


class Reader:
    def __init__(self, root=WORK): self.root, self.inputs = Path(root).resolve(), {}
    def path(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root) and path.suffix in ('.json', '.py', '.ps1', '.md'), 'Only JSON/source/Markdown permitted')
        return path
    def descriptor(self, path, raw):
        return {'path': self.path(path).relative_to(self.root).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}
    def bind(self, path):
        path = self.path(path); raw = path.read_bytes(); binding = self.descriptor(path, raw)
        require(binding['path'] not in self.inputs or self.inputs[binding['path']] == binding, 'Immutable evidence changed')
        self.inputs[binding['path']] = binding
        return binding
    def read(self, path, expected=None):
        path = self.path(path); require(path.suffix == '.json', 'JSON required')
        raw = path.read_bytes(); binding = self.descriptor(path, raw)
        require(expected is None or expected == binding, 'Bound JSON changed')
        require(self.bind(path) == binding, 'JSON changed during read')
        return json.loads(raw.decode('utf-8-sig')), binding
    def checked(self, binding):
        require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Exact descriptor required')
        path = self.root / binding['path']
        require(self.bind(path) == binding, 'Descriptor mismatch')
        return path
    def recheck(self):
        for binding in list(self.inputs.values()): self.checked(binding)


def identity_rows(rows):
    require(type(rows) is list and len(rows) == len(ROLES), 'Exactly four native identities required')
    mapped = {row['role']: row for row in rows}
    require(set(mapped) == set(ROLES), 'Native identity role set differs')
    for role, (pid, birth) in ROLES.items():
        row = mapped[role]
        require(type(row.get('pid')) is int and row['pid'] == pid and row.get('creation_100ns') == birth,
                'Native object PID/birth differs')
    return mapped


def clock_window(start, end):
    for row in (start, end):
        require(type(row) is dict and isinstance(row.get('wall'), str), 'Clock point absent')
        require(datetime.fromisoformat(row['wall']).tzinfo is not None, 'Wall clock timezone absent')
        require(type(row.get('qpc')) is str and row['qpc'].isdigit()
                and type(row.get('qpc_frequency')) is str and row['qpc_frequency'].isdigit()
                and int(row['qpc_frequency']) > 0, 'Strict QPC clock required')
    require(start['qpc_frequency'] == end['qpc_frequency'] and int(end['qpc']) >= int(start['qpc']), 'QPC window changed or reversed')
    return {'started': start, 'returned': end,
            'resume_call_observed_window_seconds': (int(end['qpc']) - int(start['qpc'])) / int(start['qpc_frequency']),
            'scope': 'Includes event-write and native-call boundary; not exact native-call compute or complete pause duration',
            'pause_monotonic_exact_seconds': None, 'active_training_elapsed_exact_seconds': None}


def timer_row(kind, binding=None, seconds=None, status='pending'):
    require(seconds is None or finite(seconds), 'Invalid producer elapsed')
    require(status in ('pending', 'actual_terminal'), 'Invalid timer evidence state')
    require(status == 'actual_terminal' or seconds is None, 'Pending is not terminal elapsed')
    return {'scope': kind, 'status': status, 'artifact': binding,
            'recorded_elapsed_including_native_pause_seconds': seconds,
            'active_training_elapsed_exact_seconds': None, 'pause_monotonic_exact_seconds': None,
            'eligible_for_active_known_subtotal': False, 'active_known_subtotal_contribution': None,
            'producer_elapsed_preserved': True, 'wall_pause_subtracted': False,
            'nested_with_formal_attempt': kind in ('F1_metadata_subscope', 'Q5_parent_dispatcher')}


def read_terminal(reader, path):
    """An active atomic-write observation cannot become a stable terminal binding."""
    path = reader.path(path)
    if not path.exists(): return None, None, 'not_created'
    for _ in range(3):
        try:
            raw = path.read_bytes(); value = json.loads(raw.decode('utf-8-sig'))
            if value.get('status') not in ('complete', 'failed'): return None, None, 'still_nonterminal'
            if path.read_bytes() != raw: return None, None, 'changed_during_observation'
            binding = reader.descriptor(path, raw)
            require(reader.bind(path) == binding, 'Terminal changed before binding')
            return value, binding, None
        except (PermissionError, FileNotFoundError, json.JSONDecodeError):
            time.sleep(.01)
    return None, None, 'transient_read_unresolved'


def verify_resume(reader, folder, pause, native, resume_source):
    complete, complete_binding = reader.read(folder / 'native_resume_complete.json')
    pre, pre_binding = reader.read(folder / 'captured_resume_preflight.json')
    release, release_binding = reader.read(folder / 'resume_controls_released.json')
    require(complete.get('status') == 'same_four_objects_resumed_by_explicit_continue'
            and complete.get('source') == pre.get('source') == release.get('source') == resume_source,
            'Actual source-bound complete resume/release required')
    require(pre.get('pause_proof_sha256') == PAUSE_SHA and pre.get('native_pause_sha256') == NATIVE_SHA
            and Path(pre['pause_proof']).resolve() == PAUSE / 'pause_complete.json'
            and Path(pre['native_pause']).resolve() == PAUSE / 'native_pause_complete.json', 'Resume references another pause')
    require(pre['owner_snapshot'] == pause['owner_snapshot'] == native['owner_snapshot'], 'Original owner lineage differs')
    require(pre.get('pause_boundary_qpc_not_measured') is complete.get('pause_boundary_qpc_not_measured') is True
            and pre.get('paused_interval_qpc_exact') is None and complete.get('exact_active_elapsed_seconds') is None
            and complete.get('pause_not_training') is True, 'Unknown pause clock was inferred')
    require(type(complete.get('native_resume_calls')) is int and complete['native_resume_calls'] == 4
            and all(type(complete.get(key)) is int and complete[key] == 0 for key in
                    ('new_queue_processes', 'optimizer_operations_by_controller', 'model_reads_by_controller', 'process_terminate_calls')),
            'Same-memory controller scope differs')
    old_ids, ids, verified = identity_rows(native['identities']), identity_rows(pre['identities']), identity_rows(complete['verification'])
    for role in ROLES:
        require(ids[role]['image'] == old_ids[role]['image'] and ids[role]['command_line'] == old_ids[role]['command_line'], 'Native image/command changed')
    require([row['role'] for row in complete['actions']] == list(ROLES), 'Resume call sequence/multiplicity differs')
    windows = []
    for index, action in enumerate(complete['actions'], 1):
        role = action['role']; pid, birth = ROLES[role]
        require(type(action.get('native_status')) is int and action['native_status'] == 0
                and action['pid'] == pid and action['creation_100ns'] == birth, 'Resume action failed or identity differs')
        start, start_binding = reader.read(folder / ('event_%02d_%s_started.json' % (index, role)))
        returned, return_binding = reader.read(folder / ('event_%02d_%s_returned.json' % (index, role)))
        require(returned == action and start.get('source') == resume_source and start.get('role') == role
                and start.get('pid') == pid and start.get('birth') == birth and start.get('resume_started') is True
                and start.get('native_result') is None, 'Persisted started/returned resume event differs')
        windows.append({'role': role, 'started_event': start_binding, 'returned_event': return_binding,
                        **clock_window(start['clock'], action['returned'])})
    archives = complete['controls_archived']
    require(len(archives) == 4 and archives == pre['controls'] == release['controls']
            and release.get('status') == 'explicit_continue_controls_archived_then_released', 'Four original controls not archived and released')
    expected_controls = {OUT / 'user_pause_q5_owner.json', *(OUT / 'execution_acceleration_v1' / name / 'user_pause.json'
                        for name in ('sync_boundary', 'sync_boundary_v2', 'sync_boundary_v3'))}
    require({Path(row['original']).resolve() for row in archives} == expected_controls, 'Original pause control set differs')
    archived_bindings = []
    for index, row in enumerate(archives, 1):
        path = Path(row['archive']).resolve()
        require(path == folder / ('control_%d.json' % index), 'Control archive path differs')
        value, binding = reader.read(path); require(binding['sha256'] == row['sha256'], 'Control archive bytes differ')
        if index == 1:
            require(Path(value['folder']).resolve() == PAUSE and value['automatic_resume_authorized'] is False, 'Original pause intent differs')
        else: require(value == {'paused': True}, 'Original strict pause marker differs')
        archived_bindings.append(binding)
    return {'status': 'actual_native_resume_and_controls_release_verified', 'native_resume': complete_binding,
            'preflight': pre_binding, 'controls_release': release_binding, 'archived_controls': archived_bindings,
            'same_native_identities': {role: {'pid': pair[0], 'creation_100ns': pair[1]} for role, pair in ROLES.items()},
            'resume_call_windows': windows, 'pause_monotonic_exact_seconds': None,
            'active_training_elapsed_exact_seconds': None, 'new_process_or_loss_closure_required': False}


def collect():
    reader = Reader(); own = reader.bind(SOURCE)
    resume_source = reader.bind(RESUMER); require(resume_source['sha256'] == RESUMER_SHA, 'Reviewed resumer source changed')
    pause, pause_binding = reader.read(PAUSE / 'pause_complete.json')
    native, native_binding = reader.read(PAUSE / 'native_pause_complete.json')
    require(pause_binding['sha256'] == PAUSE_SHA and native_binding['sha256'] == NATIVE_SHA
            and pause.get('native_pause') == native_binding and pause.get('status') == 'paused_by_user'
            and tuple(pause[k] for k in ('condition', 'seed', 'fold')) == IDENTITY, 'Exact noon pause proof differs')
    reader.checked(pause['source']); reader.checked(native['source'])
    identity_rows(native['identities'])
    captured, _ = reader.read(PAUSE / 'captured_identities.json')
    verified_pause, _ = reader.read(PAUSE / 'final_native_verification.json')
    require(captured['source'] == native['source'] and captured['owner_snapshot'] == pause['owner_snapshot']
            and verified_pause.get('all_tasks_verified_paused') is True
            and verified_pause.get('other_task_python_workers') == []
            and verified_pause.get('no_additional_native_control_operations') is True, 'Paused native evidence differs')
    identity_rows(captured['identities'])
    require(len(native['actions']) == 4 and {a['role'] for a in native['actions']} == set(ROLES)
            and all(type(a['native_status']) is int and a['native_status'] == 0 for a in native['actions']), 'Actual pause action set differs')
    for index, role in enumerate(('dispatcher', 'gpu_child', 'launcher', 'dispatcher_launcher'), 1):
        start, _ = reader.read(PAUSE / ('event_%02d_%s_started.json' % (index, role)))
        end, _ = reader.read(PAUSE / ('event_%02d_%s_returned.json' % (index, role)))
        require(start['source'] == native['source'] and start['role'] == end['role'] == role
                and end in native['actions'] and 'clock' not in start and 'qpc' not in start, 'Pause clock/event boundary differs')
    snapshots = {row['original']['path']: row for row in pause['retained']}
    originals, original_records = {}, {}
    for name, path in (('F3', F3), ('Q5_parent', PARENT)):
        pair = snapshots[path.relative_to(WORK).as_posix()]
        value, binding = reader.read(WORK / pair['snapshot']['path'], pair['snapshot'])
        require(binding['sha256'] == pair['original']['sha256'] and value['status'] == 'running', 'Historical snapshot differs')
        originals[name] = pair; original_records[name] = value
    paused_trace = original_records['F3']
    require((paused_trace['condition'], paused_trace['seed'], paused_trace['fold']) == IDENTITY
            and paused_trace['optimizer_step_returned'] == 44 and paused_trace['backward_returned'] == 286
            and paused_trace['latest_durable_checkpoint']['epoch'] == 11, 'Original trajectory snapshot differs')
    for filename, sha in TIMER_SOURCES.items():
        require(reader.bind(WORK / 'tridep_ccml/eeg_all_candidates' / filename)['sha256'] == sha, 'Sealed timer source changed')
    require(reader.bind(REPORT / 'cost_aggregation_v6.py')['sha256'] == COST_SHA, 'Sealed cost6 source changed')
    timers = [timer_row('F3_formal_attempt'), timer_row('F1_metadata_subscope'), timer_row('Q5_parent_dispatcher')]
    folders = sorted(path.parent for path in (OUT / 'pause_events').glob('q5_memory_resume_*/native_resume_complete.json'))
    result = {'schema': 'pause_memory_resume_cost_companion_v1', 'source': own, 'status': 'pending_native_resume_evidence',
        'passed': False, 'final_plan_completion_certificate': False, 'pause_proof': pause_binding, 'native_pause': native_binding,
        'affected_original_paths_and_retained_snapshots': originals, 'resumer_source': resume_source,
        'pause_wall_observed_at': pause['paused_at'], 'pause_boundary_QPC_recorded': False,
        'pause_monotonic_exact_seconds': None, 'active_training_elapsed_exact_seconds': None,
        'affected_timer_observations': timers, 'native_memory_resume_policy_creates_no_new_formal_fit': True,
        'affected_formal_fit_id': 'REVE-lastblock-finetune/seed_22/fold_2',
        'affected_active_unknown_formal_fit_count': 1,
        'pause_snapshot_counters_added_to_terminal': False, 'new_loss_closure_created': False,
        'old_cost6_already_excludes_this_completed_pause': False,
        'affected_active_known_subtotal': None, 'affected_active_unknown_scopes': 3,
        'unknown_scope_count_includes_nested_metadata_and_parent_not_three_formal_fits': True,
        'producer_recorded_seconds_are_not_active_training_seconds': True,
        'controller_or_parent_seconds_added_to_formal_cost': False,
        'reads': {'official_binary_weight_or_checkpoint_bytes': 0, 'tensor_deserializations': 0, 'raw_or_features_or_outertest': 0,
                  'models_GPU_or_process_operations': 0, 'project_verifiers_executed': 0, 'final_cost_collectors_executed': 0}}
    if len(folders) != 1 or not (folders[0] / 'resume_controls_released.json').exists():
        result['reason'] = 'Exactly one completed four-object resume and controls-release evidence pair required'
        result['resume_complete_records_observed'] = len(folders)
    else:
        result['resume_verification'] = verify_resume(reader, folders[0], pause, native, resume_source)
        result.update(status='verified_pause_scope_with_terminal_timers_pending', passed=True)
        for slot, (name, path, kind) in enumerate((('F3', F3, 'F3_formal_attempt'), ('Q5_parent', PARENT, 'Q5_parent_dispatcher'))):
            record, binding, pending_reason = read_terminal(reader, path)
            index = 0 if name == 'F3' else 2
            if record is None:
                timers[index]['pending_reason'] = pending_reason; continue
            old = original_records[name]
            keys = ('condition', 'kind', 'seed', 'fold', 'started_at', 'implementation', 'execution_acceleration') if name == 'F3' else (
                'job', 'command', 'started_at', 'child_pid', 'registration', 'source', 'execution_registration', 'execution_version',
                'recovery_registration', 'recovery_execution_source', 'recovery_closure')
            require(all(record.get(k) == old.get(k) for k in keys), 'Terminal is not the original same-memory invocation')
            if name == 'F3':
                require(record.get('graceful_terminal_flush') is True and record.get('counter_snapshot_exact_for_process') is True,
                        'F3 terminal is not an observed graceful flush')
                require(all(integer(record.get(k)) and record[k] >= value for k, value in old['durable_counter_snapshot'].items()),
                        'Same process counters moved backwards')
            require(finite(record.get('seconds')), 'Actual terminal producer timer missing')
            timers[index] = timer_row(kind, binding, record['seconds'], 'actual_terminal')
            timers[index]['producer_terminal_status'] = record['status']
        complete_path = FIT / 'complete.json'
        if timers[0]['status'] == 'actual_terminal' and complete_path.exists():
            complete, completion_binding = reader.read(complete_path)
            metadata_path = reader.checked(complete['metadata']); metadata, metadata_binding = reader.read(metadata_path, complete['metadata'])
            require(tuple(metadata[k] for k in ('condition', 'seed', 'fold')) == IDENTITY
                    and metadata_path == FIT / 'metadata.json' and finite(metadata.get('seconds_this_attempt')), 'Completed metadata identity/timer differs')
            require(metadata.get('execution_acceleration') == original_records['F3']['execution_acceleration'], 'Metadata execution differs')
            timers[1] = timer_row('F1_metadata_subscope', metadata_binding, metadata['seconds_this_attempt'], 'actual_terminal')
            timers[1]['completion_binding'] = completion_binding
        result['all_affected_producer_timers_terminal'] = all(row['status'] == 'actual_terminal' for row in timers)
        if result['all_affected_producer_timers_terminal']: result['status'] = 'verified_pause_scope_and_recorded_terminal_timers'
    result['scope_note'] = 'F3 monotonic and F1 perf_counter final elapsed include native suspension; Q5 parent overlaps both. Never subtract wall-clock pause from monotonic seconds. Cost6 only nulls still-running/lost F3; its later complete raw elapsed has not excluded this pause.'
    reader.recheck(); result['immutable_inputs'] = list(reader.inputs.values())
    return result


def render(value):
    lines = ['# 原进程暂停恢复成本附页', '', '本附页保留生产者原始计时；不是全计划完成证明。', '',
             'F3 正式尝试与 F1 元数据的计时包含原生暂停，Q5 父尝试计时还与它们重叠。原 cost6 并未自动排除这次已恢复尝试的暂停时间。', '',
             '暂停时没有 QPC，恢复时的 QPC 只能描述恢复调用窗口。精确暂停时长和有效训练时长均为未知，不能用日历时间差扣除，也不能将受影响计时加入有效训练时长的已知小计。', '',
             '| 范围 | 证据状态 | 含暂停的原始秒数 | 有效训练秒数 |', '|---|---|---:|---|']
    for row in value['affected_timer_observations']:
        seconds = row['recorded_elapsed_including_native_pause_seconds']
        lines.append(f"| {row['scope']} | {row['status']} | {'未知' if seconds is None else seconds} | 未知 |")
    lines += ['', '原进程内存继续同一训练轨迹；暂停快照的44次更新、286次反向不再相加，不创建新的丢失进程闭合记录。原始成本与数学来源不改。', '',
              '本工具只读 JSON/源码/Markdown，不导入项目验证器，不读取二进制权重或检查点，不构造模型、不执行 GPU、进程操作、外部测试或最终成本收集。', '']
    return '\n'.join(lines)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    value = collect(); stamp = str(time.time_ns())
    human = REPORT / ('原进程暂停恢复成本附页_' + stamp + '.md')
    with human.open('x', encoding='utf8', newline='\n') as stream: stream.write(render(value))
    value['human_readable_report'] = Reader().bind(human)
    path = REPORT / ('pause_memory_resume_cost_companion_' + stamp + '.json')
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'artifact': Reader().bind(path), 'status': value['status'], 'passed': value['passed'],
                      'active_training_elapsed_exact_seconds': None, 'pause_monotonic_exact_seconds': None}, ensure_ascii=False))


if __name__ == '__main__': main()
